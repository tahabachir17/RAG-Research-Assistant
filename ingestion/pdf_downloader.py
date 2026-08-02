from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

try:
    from .arxiv_scraper import Paper
    from .identity import canonical_arxiv_id
except ImportError:
    from arxiv_scraper import Paper
    from identity import canonical_arxiv_id

try:
    import fitz
except ImportError:  # Header validation still works without PyMuPDF.
    fitz = None  # type: ignore[assignment]


@dataclass(slots=True)
class DownloadResult:
    paper_id: str
    pdf_path: Path
    downloaded: bool
    skipped: bool
    status_code: int | None = None
    error: str | None = None


class PDFDownloader:
    """Download one validated PDF per canonical ArXiv identity."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/arxiv",
        timeout: float = 60.0,
        overwrite: bool = False,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must not be negative")
        self.raw_dir = Path(raw_dir)
        self.timeout = timeout
        self.overwrite = overwrite
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def download(self, paper: Paper | dict[str, Any]) -> DownloadResult:
        record = paper.to_dict() if isinstance(paper, Paper) else dict(paper)
        paper_id = str(record.get("paper_id") or "").strip()
        if not paper_id:
            raise ValueError("paper record is missing paper_id")

        pdf_path = self.raw_dir / f"{canonical_arxiv_id(paper_id)}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        if pdf_path.exists() and not self.overwrite:
            try:
                _validate_pdf(pdf_path)
                return DownloadResult(paper_id, pdf_path, False, True)
            except ValueError:
                pass

        pdf_url = record.get("pdf_url") or _pdf_url_from_entry(
            record.get("entry_id"), paper_id
        )
        if not pdf_url:
            return DownloadResult(
                paper_id, pdf_path, False, False, error="missing pdf_url"
            )

        part_path = pdf_path.with_suffix(".pdf.part")
        last_error, status_code = "download failed", None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with httpx.stream(
                    "GET", pdf_url, timeout=self.timeout, follow_redirects=True
                ) as response:
                    status_code = response.status_code
                    if response.status_code != 200:
                        last_error = f"download failed with HTTP {response.status_code}"
                        if not _retryable_status(response.status_code):
                            break
                        delay = _retry_delay(response, attempt, self.backoff_seconds)
                    else:
                        with part_path.open("wb") as file:
                            for chunk in response.iter_bytes():
                                file.write(chunk)
                        _validate_pdf(part_path)
                        part_path.replace(pdf_path)
                        return DownloadResult(
                            paper_id, pdf_path, True, False, response.status_code
                        )
            except (httpx.HTTPError, OSError, ValueError) as exc:
                last_error = str(exc)
                delay = self.backoff_seconds * (2 ** (attempt - 1))
            finally:
                if part_path.exists():
                    part_path.unlink()
            if attempt < self.max_attempts:
                time.sleep(delay)

        return DownloadResult(paper_id, pdf_path, False, False, status_code, last_error)


def _pdf_url_from_entry(entry_id: Any, paper_id: str) -> str | None:
    if entry_id:
        entry = str(entry_id).replace("/abs/", "/pdf/")
        return entry if entry.endswith(".pdf") else f"{entry}.pdf"
    if paper_id:
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    return None


def _validate_pdf(path: Path) -> None:
    with path.open("rb") as file:
        header = file.read(5)
    if path.stat().st_size < 5 or header != b"%PDF-":
        raise ValueError("downloaded file is not a valid PDF")
    if fitz is not None:
        try:
            with fitz.open(path) as document:
                if document.page_count < 1:
                    raise ValueError("downloaded PDF has no pages")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"downloaded PDF is corrupt: {exc}") from exc


def _retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code in {408, 425} or status_code >= 500


def _retry_delay(response: httpx.Response, attempt: int, base: float) -> float:
    retry_after = response.headers.get("retry-after")
    try:
        return (
            max(0.0, float(retry_after)) if retry_after else base * (2 ** (attempt - 1))
        )
    except ValueError:
        return base * (2 ** (attempt - 1))
