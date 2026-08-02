from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from .arxiv_scraper import ArxivScraper, Paper
    from .citation_extractor import citations_to_dicts, extract_citations
    from .corpus_registry import CorpusRegistry
    from .data_cleaner import clean_sections
    from .identity import arxiv_version, canonical_arxiv_id
    from .metadata_extractor import extract_metadata
    from .paper_discovery import ArxivPaperDiscovery, PaperDiscovery, build_discovery
    from .pdf_downloader import DownloadResult, PDFDownloader
    from .pdf_parser import PDFParser
    from .section_detector import SectionDetector
except ImportError:  # Allows `python ingestion/pipeline.py` from repo root.
    from arxiv_scraper import ArxivScraper, Paper
    from citation_extractor import citations_to_dicts, extract_citations
    from corpus_registry import CorpusRegistry
    from data_cleaner import clean_sections
    from identity import arxiv_version, canonical_arxiv_id
    from metadata_extractor import extract_metadata
    from paper_discovery import ArxivPaperDiscovery, PaperDiscovery, build_discovery
    from pdf_downloader import DownloadResult, PDFDownloader
    from pdf_parser import PDFParser
    from section_detector import SectionDetector


@dataclass(slots=True)
class IngestionResult:
    query: str
    discovery_provider: str
    discovered: int
    downloaded: int
    skipped: int
    failed: int
    processed: int
    metadata_path: str
    processed_dir: str
    errors: list[dict[str, str]]
    processed_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IngestionPipeline:
    """Discover papers, enrich metadata, then ingest the selected PDFs."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        discovery: PaperDiscovery | None = None,
        discovery_provider: str = "auto",
        scraper: ArxivScraper | None = None,
        downloader: PDFDownloader | None = None,
        parser: PDFParser | None = None,
        section_detector: SectionDetector | None = None,
        registry: CorpusRegistry | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / "metadata"
        self.processed_dir = self.data_dir / "processed" / "raw_text"
        self.discovery_provider = discovery_provider
        if discovery is not None:
            self.discovery = discovery
        elif scraper is not None:
            self.discovery = ArxivPaperDiscovery(scraper)
            self.discovery_provider = "arxiv"
        else:
            self.discovery = build_discovery(discovery_provider)
        self.downloader = downloader or PDFDownloader(self.data_dir / "raw" / "arxiv")
        self.parser = parser or PDFParser()
        self.section_detector = section_detector or SectionDetector()
        self.registry = registry or CorpusRegistry(
            registry_path or self.metadata_dir / "corpus_registry.sqlite3"
        )

    def run(
        self,
        query: str,
        max_results: int = 50,
        *,
        resume: bool = False,
        selected_papers: list[Paper] | None = None,
    ) -> IngestionResult:
        run_key = _run_key(self.discovery_provider, query, max_results)
        papers = list(selected_papers or [])
        if selected_papers is None:
            papers = self.registry.load_checkpoint(run_key) if resume else []
            if not papers:
                papers = self.discovery.search(query=query, max_results=max_results)
        papers = _deduplicate_by_identity(papers)
        self.registry.checkpoint(run_key, papers)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        skipped = 0
        failed = 0
        processed = 0
        errors: list[dict[str, str]] = []
        processed_paths: list[str] = []

        for paper in papers:
            registered = self.registry.get(paper.paper_id)
            completed_path = Path(str((registered or {}).get("processed_path") or ""))
            if (
                registered
                and registered["status"] in {"processed", "indexed"}
                and completed_path.is_file()
            ):
                skipped += 1
                continue

            self.registry.mark(paper.paper_id, "downloading", increment_attempts=True)
            download = self.downloader.download(paper)
            if download.downloaded:
                downloaded += 1
            if download.skipped:
                skipped += 1
            if download.error:
                failed += 1
                errors.append(
                    {
                        "paper_id": download.paper_id,
                        "stage": "download",
                        "error": download.error,
                    }
                )
                self.registry.mark(
                    paper.paper_id,
                    "failed",
                    failure_stage="download",
                    error=download.error,
                )
                continue

            self.registry.mark(paper.paper_id, "downloaded", pdf_path=download.pdf_path)
            try:
                self.registry.mark(paper.paper_id, "processing")
                document = self.parser.extract(
                    download.pdf_path, paper_id=paper.paper_id
                )
                sectioned = self.section_detector.detect(document)
                cleaned = clean_sections(sectioned)
                citations = extract_citations(cleaned)
                metadata = extract_metadata(_with_local_path(paper, download))

                metadata_record = metadata.to_dict()
                metadata_record.update(_discovery_metadata(paper))
                metadata_record["canonical_id"] = canonical_arxiv_id(paper.paper_id)
                metadata_record["source_version"] = arxiv_version(paper.paper_id)
                processed_path = self._write_processed_document(
                    paper=paper,
                    metadata=metadata_record,
                    raw_document=document.to_dict(),
                    sectioned_document=cleaned.to_dict(),
                    citations=citations_to_dicts(citations),
                )
                self.registry.mark(
                    paper.paper_id,
                    "processed",
                    pdf_path=download.pdf_path,
                    processed_path=processed_path,
                    metadata=metadata_record,
                )
                processed += 1
                processed_paths.append(str(processed_path))
            except Exception as exc:  # Keep long ingestion runs moving paper by paper.
                failed += 1
                errors.append(
                    {"paper_id": paper.paper_id, "stage": "process", "error": str(exc)}
                )
                self.registry.mark(
                    paper.paper_id,
                    "failed",
                    pdf_path=download.pdf_path,
                    failure_stage="process",
                    error=str(exc),
                )

        metadata_path = self.metadata_dir / "papers.json"
        self.registry.export_json(metadata_path)

        return IngestionResult(
            query=query,
            discovery_provider=self.discovery_provider,
            discovered=len(papers),
            downloaded=downloaded,
            skipped=skipped,
            failed=failed,
            processed=processed,
            metadata_path=str(metadata_path),
            processed_dir=str(self.processed_dir),
            errors=errors,
            processed_paths=processed_paths,
        )

    def _write_processed_document(
        self,
        paper: Paper,
        metadata: dict[str, Any],
        raw_document: dict[str, Any],
        sectioned_document: dict[str, Any],
        citations: list[dict[str, Any]],
    ) -> Path:
        paper_id = canonical_arxiv_id(paper.paper_id)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / f"{paper_id}.json"
        payload = {
            "paper_id": paper.paper_id,
            "metadata": metadata,
            "raw_document": raw_document,
            "sections": sectioned_document["sections"],
            "section_spans": sectioned_document.get("section_spans", {}),
            "section_details": sectioned_document.get("section_details", []),
            "heading_diagnostics": sectioned_document.get("heading_diagnostics", []),
            "citations": citations,
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(output_path)
        return output_path


def run_ingestion(
    query: str,
    max_results: int = 50,
    data_dir: str | Path = "data",
    discovery_provider: str = "auto",
    resume: bool = False,
) -> IngestionResult:
    return IngestionPipeline(
        data_dir=data_dir, discovery_provider=discovery_provider
    ).run(query=query, max_results=max_results, resume=resume)


def _discovery_metadata(paper: Paper) -> dict[str, Any]:
    reserved = set(Paper.__dataclass_fields__)
    return {
        key: value
        for key, value in paper.to_dict().items()
        if key not in reserved and value not in (None, "", [])
    }


def _with_local_path(paper: Paper, download: DownloadResult) -> dict[str, Any]:
    record = paper.to_dict()
    record["local_pdf_path"] = str(download.pdf_path)
    return record


def _safe_path_part(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").strip() or "unknown"


def _run_key(provider: str, query: str, max_results: int) -> str:
    normalized = " ".join(query.casefold().split())
    digest = hashlib.sha256(
        f"{provider.casefold()}\0{normalized}\0{max_results}".encode("utf-8")
    ).hexdigest()
    return digest[:24]


def _deduplicate_by_identity(papers: list[Paper]) -> list[Paper]:
    unique: dict[str, Paper] = {}
    for paper in papers:
        identity = canonical_arxiv_id(paper.paper_id)
        current = unique.get(identity)
        current_version = _version_number(
            arxiv_version(current.paper_id) if current else None
        )
        incoming_version = _version_number(arxiv_version(paper.paper_id))
        if current is None or incoming_version > current_version:
            unique[identity] = paper
    return list(unique.values())


def _version_number(version: str | None) -> int:
    return int(version[1:]) if version and version[1:].isdigit() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper ingestion pipeline.")
    parser.add_argument("--query", required=True, help="Research topic or paper query")
    parser.add_argument(
        "--max-results", type=int, default=50, help="Maximum papers to ingest"
    )
    parser.add_argument("--data-dir", default="data", help="Project data directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the saved discovery checkpoint",
    )
    parser.add_argument(
        "--discovery-provider",
        choices=("auto", "feyman", "arxiv", "alphaxiv-mcp", "research-apis"),
        default="auto",
        help=(
            "Paper discovery provider to use before downloading PDFs. Default auto "
            "uses alphaXiv MCP, then falls back to ArXiv."
        ),
    )
    args = parser.parse_args()

    result = run_ingestion(
        args.query,
        max_results=args.max_results,
        data_dir=args.data_dir,
        discovery_provider=args.discovery_provider,
        resume=args.resume,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
