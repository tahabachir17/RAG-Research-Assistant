from pathlib import Path

import ingestion.pdf_downloader as downloader_module
import ingestion.pipeline as pipeline_module
from ingestion.arxiv_scraper import Paper
from ingestion.corpus_registry import CorpusRegistry
from ingestion.identity import arxiv_version, canonical_arxiv_id
from ingestion.pdf_downloader import DownloadResult, PDFDownloader
from ingestion.pipeline import IngestionPipeline


def _paper(paper_id: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        title="Scaling RAG ingestion",
        authors=["Ada Lovelace"],
        summary="A test paper.",
        published="2026-01-01",
        updated="2026-01-02",
        primary_category="cs.AI",
        categories=["cs.AI"],
        pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
        entry_id=f"https://arxiv.org/abs/{paper_id}",
    )


def test_identity_normalizes_versions_urls_and_legacy_ids():
    values = [
        "2005.11401v4",
        "arXiv:2005.11401v4",
        "https://arxiv.org/abs/2005.11401v4",
        "https://arxiv.org/pdf/2005.11401v4.pdf?download=1",
    ]
    assert {canonical_arxiv_id(value) for value in values} == {"2005.11401"}
    assert {arxiv_version(value) for value in values} == {"v4"}
    assert canonical_arxiv_id("arXiv:hep-th/9901001v2") == "hep-th_9901001"


def test_registry_replaces_checkpoints_and_reopens_new_versions(tmp_path):
    registry = CorpusRegistry(tmp_path / "corpus.sqlite3")
    registry.checkpoint("run", [_paper("2005.11401v1"), _paper("2201.00001v1")])
    assert registry.mark(
        "2005.11401v1", "processed", metadata={"paper_id": "2005.11401v1"}
    )

    registry.checkpoint("run", [_paper("2005.11401v2")])

    assert [paper.paper_id for paper in registry.load_checkpoint("run")] == [
        "2005.11401v2"
    ]
    record = registry.get("https://arxiv.org/abs/2005.11401")
    assert record is not None
    assert record["source_version"] == "v2"
    assert record["status"] == "discovered"
    assert record["metadata_json"] is None


class _Discovery:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        self.calls += 1
        return [_paper("2005.11401v1"), _paper("2005.11401v1")]


class _Downloader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0

    def download(self, paper: Paper) -> DownloadResult:
        self.calls += 1
        path = self.root / f"{paper.paper_id}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-test")
        return DownloadResult(paper.paper_id, path, True, False, 200)


class _Document:
    def to_dict(self):
        return {"pages": [], "pages_count": 0}


class _Parser:
    def extract(self, path: Path, paper_id: str):
        return _Document()


class _Sectioned:
    sections = {"abstract": "A sufficiently useful abstract.", "references": ""}

    def to_dict(self):
        return {
            "sections": self.sections,
            "section_spans": {},
            "section_details": [{"section": "abstract"}],
            "heading_diagnostics": [],
        }


class _Detector:
    def detect(self, document: _Document):
        return _Sectioned()


class _Metadata:
    def to_dict(self):
        return {"paper_id": "2005.11401v1", "title": "Scaling RAG ingestion"}


def test_pipeline_uses_registry_checkpoint_and_skips_completed_work(
    tmp_path, monkeypatch
):
    discovery = _Discovery()
    downloader = _Downloader(tmp_path / "downloads")
    monkeypatch.setattr(pipeline_module, "clean_sections", lambda document: document)
    monkeypatch.setattr(pipeline_module, "extract_citations", lambda document: [])
    monkeypatch.setattr(pipeline_module, "extract_metadata", lambda paper: _Metadata())
    pipeline = IngestionPipeline(
        data_dir=tmp_path / "data",
        discovery=discovery,
        downloader=downloader,
        parser=_Parser(),
        section_detector=_Detector(),
    )

    first = pipeline.run("rag", max_results=10)
    second = pipeline.run("rag", max_results=10, resume=True)

    assert first.discovered == 1
    assert first.processed == 1
    assert second.processed == 0
    assert second.skipped == 1
    assert discovery.calls == 1
    assert downloader.calls == 1
    processed = tmp_path / "data" / "processed" / "raw_text"
    assert [path.name for path in processed.glob("*.json")] == ["2005.11401.json"]
    record = pipeline.registry.get("2005.11401v9")
    assert record is not None
    assert record["status"] == "processed"
    assert record["attempts"] == 1


class _Response:
    def __init__(self, status_code: int, body: bytes = b"", headers=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_bytes(self):
        yield self.body


def test_downloader_uses_flat_canonical_path_and_replaces_corrupt_file(
    tmp_path, monkeypatch
):
    responses = iter([_Response(200, b"%PDF-test")])
    monkeypatch.setattr(downloader_module, "fitz", None)
    monkeypatch.setattr(
        downloader_module.httpx, "stream", lambda *args, **kwargs: next(responses)
    )
    raw_dir = tmp_path / "raw"
    corrupt = raw_dir / "2005.11401.pdf"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"partial")

    result = PDFDownloader(raw_dir, backoff_seconds=0).download(_paper("2005.11401v4"))

    assert result.downloaded
    assert result.pdf_path == raw_dir / "2005.11401.pdf"
    assert result.pdf_path.read_bytes() == b"%PDF-test"
    assert not result.pdf_path.with_suffix(".pdf.part").exists()


def test_downloader_retries_rate_limit_and_honors_canonical_dedup(
    tmp_path, monkeypatch
):
    responses = iter(
        [
            _Response(429, headers={"retry-after": "0"}),
            _Response(200, b"%PDF-test"),
        ]
    )
    calls = []
    monkeypatch.setattr(downloader_module, "fitz", None)

    def stream(*args, **kwargs):
        calls.append(args)
        return next(responses)

    monkeypatch.setattr(downloader_module.httpx, "stream", stream)
    downloader = PDFDownloader(tmp_path, max_attempts=2, backoff_seconds=0)

    first = downloader.download(_paper("2005.11401v2"))
    second_paper = _paper("2005.11401v3")
    second_paper.primary_category = "cs.LG"
    second = downloader.download(second_paper)

    assert first.downloaded and len(calls) == 2
    assert second.skipped
    assert second.pdf_path == first.pdf_path
