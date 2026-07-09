from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .arxiv_scraper import ArxivScraper, Paper
    from .citation_extractor import citations_to_dicts, extract_citations
    from .data_cleaner import clean_sections
    from .metadata_extractor import extract_metadata
    from .paper_discovery import ArxivPaperDiscovery, PaperDiscovery, build_discovery
    from .pdf_downloader import DownloadResult, PDFDownloader
    from .pdf_parser import PDFParser
    from .section_detector import SectionDetector
except ImportError:  # Allows `python ingestion/pipeline.py` from repo root.
    from arxiv_scraper import ArxivScraper, Paper
    from citation_extractor import citations_to_dicts, extract_citations
    from data_cleaner import clean_sections
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

    def run(self, query: str, max_results: int = 50) -> IngestionResult:
        papers = self.discovery.search(query=query, max_results=max_results)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        metadata_records: list[dict[str, Any]] = []
        downloaded = 0
        skipped = 0
        failed = 0
        processed = 0
        errors: list[dict[str, str]] = []

        for paper in papers:
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
                continue

            try:
                document = self.parser.extract(
                    download.pdf_path, paper_id=paper.paper_id
                )
                sectioned = self.section_detector.detect(document)
                cleaned = clean_sections(sectioned)
                citations = extract_citations(cleaned)
                metadata = extract_metadata(_with_local_path(paper, download))

                metadata_record = metadata.to_dict()
                metadata_record.update(_discovery_metadata(paper))
                metadata_records.append(metadata_record)
                self._write_processed_document(
                    paper=paper,
                    metadata=metadata_record,
                    raw_document=document.to_dict(),
                    sectioned_document=cleaned.to_dict(),
                    citations=citations_to_dicts(citations),
                )
                processed += 1
            except Exception as exc:  # Keep long ingestion runs moving paper by paper.
                failed += 1
                errors.append(
                    {"paper_id": paper.paper_id, "stage": "process", "error": str(exc)}
                )

        metadata_path = self.metadata_dir / "papers.json"
        metadata_path.write_text(
            json.dumps(metadata_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
        )

    def _write_processed_document(
        self,
        paper: Paper,
        metadata: dict[str, Any],
        raw_document: dict[str, Any],
        sectioned_document: dict[str, Any],
        citations: list[dict[str, Any]],
    ) -> Path:
        category = _safe_path_part(paper.primary_category or "unknown")
        paper_id = _safe_path_part(paper.paper_id)
        output_dir = self.processed_dir / category
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{paper_id}.json"
        payload = {
            "paper_id": paper.paper_id,
            "metadata": metadata,
            "raw_document": raw_document,
            "sections": sectioned_document["sections"],
            "section_spans": sectioned_document.get("section_spans", {}),
            "citations": citations,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output_path


def run_ingestion(
    query: str,
    max_results: int = 50,
    data_dir: str | Path = "data",
    discovery_provider: str = "auto",
) -> IngestionResult:
    return IngestionPipeline(
        data_dir=data_dir, discovery_provider=discovery_provider
    ).run(query=query, max_results=max_results)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper ingestion pipeline.")
    parser.add_argument("--query", required=True, help="Research topic or paper query")
    parser.add_argument(
        "--max-results", type=int, default=50, help="Maximum papers to ingest"
    )
    parser.add_argument("--data-dir", default="data", help="Project data directory")
    parser.add_argument(
        "--discovery-provider",
        choices=("auto", "feyman", "arxiv", "alphaxiv-mcp", "research-apis"),
        default="auto",
        help="Paper discovery provider to use before downloading PDFs. Default auto uses alphaXiv MCP, then falls back to ArXiv.",
    )
    args = parser.parse_args()

    result = run_ingestion(
        args.query,
        max_results=args.max_results,
        data_dir=args.data_dir,
        discovery_provider=args.discovery_provider,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()


