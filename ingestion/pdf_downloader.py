from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

try:
    from .arxiv_scraper import Paper
except ImportError:
    from arxiv_scraper import Paper


@dataclass(slots=True)
class DownloadResult:
    paper_id: str
    pdf_path: Path
    downloaded: bool
    skipped: bool
    status_code: int | None = None
    error: str | None = None


class PDFDownloader:
    """Download PDFs into data/raw/arxiv/{category}/{paper_id}.pdf."""

    def __init__(self, raw_dir: str | Path = "data/raw/arxiv", timeout: float = 60.0, overwrite: bool = False) -> None:
        self.raw_dir = Path(raw_dir)
        self.timeout = timeout
        self.overwrite = overwrite

    def download(self, paper: Paper | dict[str, Any]) -> DownloadResult:
        record = paper.to_dict() if isinstance(paper, Paper) else dict(paper)
        paper_id = str(record.get("paper_id") or "").strip()
        if not paper_id:
            raise ValueError("paper record is missing paper_id")

        category = _safe_path_part(str(record.get("primary_category") or "unknown"))
        pdf_path = self.raw_dir / category / f"{_safe_path_part(paper_id)}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        if pdf_path.exists() and not self.overwrite:
            return DownloadResult(paper_id=paper_id, pdf_path=pdf_path, downloaded=False, skipped=True)

        pdf_url = record.get("pdf_url") or _pdf_url_from_entry(record.get("entry_id"), paper_id)
        if not pdf_url:
            return DownloadResult(paper_id=paper_id, pdf_path=pdf_path, downloaded=False, skipped=False, error="missing pdf_url")

        try:
            with httpx.stream("GET", pdf_url, timeout=self.timeout, follow_redirects=True) as response:
                if response.status_code != 200:
                    return DownloadResult(
                        paper_id=paper_id,
                        pdf_path=pdf_path,
                        downloaded=False,
                        skipped=False,
                        status_code=response.status_code,
                        error=f"download failed with HTTP {response.status_code}",
                    )
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and not urlparse(str(pdf_url)).path.endswith(".pdf"):
                    return DownloadResult(
                        paper_id=paper_id,
                        pdf_path=pdf_path,
                        downloaded=False,
                        skipped=False,
                        status_code=response.status_code,
                        error=f"unexpected content type: {content_type or 'unknown'}",
                    )
                with pdf_path.open("wb") as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
        except httpx.HTTPError as exc:
            return DownloadResult(paper_id=paper_id, pdf_path=pdf_path, downloaded=False, skipped=False, error=str(exc))

        return DownloadResult(paper_id=paper_id, pdf_path=pdf_path, downloaded=True, skipped=False, status_code=200)


def _pdf_url_from_entry(entry_id: Any, paper_id: str) -> str | None:
    if entry_id:
        entry = str(entry_id).replace("/abs/", "/pdf/")
        return entry if entry.endswith(".pdf") else f"{entry}.pdf"
    if paper_id:
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    return None


def _safe_path_part(value: str) -> str:
    value = value.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"