from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

try:
    from .arxiv_scraper import Paper
except ImportError:
    from arxiv_scraper import Paper


@dataclass(slots=True)
class PaperMeta:
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    primary_category: str | None
    categories: list[str]
    doi: str | None
    url: str | None
    pdf_url: str | None
    published: str | None
    updated: str | None
    abstract: str | None = None
    local_pdf_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_metadata(arxiv_result: Paper | dict[str, Any] | Any) -> PaperMeta:
    record = _as_record(arxiv_result)
    published = _iso_or_none(record.get("published"))
    updated = _iso_or_none(record.get("updated"))
    return PaperMeta(
        paper_id=str(record.get("paper_id") or record.get("id") or ""),
        title=_clean(record.get("title", "")),
        authors=list(record.get("authors") or []),
        year=_year_from_date(published),
        primary_category=record.get("primary_category"),
        categories=list(record.get("categories") or []),
        doi=record.get("doi") or _find_doi(record),
        url=record.get("entry_id") or record.get("url"),
        pdf_url=record.get("pdf_url"),
        published=published,
        updated=updated,
        abstract=_clean(record.get("summary", "")) or None,
        local_pdf_path=record.get("local_pdf_path"),
    )


def _as_record(value: Paper | dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Paper):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    authors = [author.name for author in getattr(value, "authors", [])]
    return {
        "paper_id": value.get_short_id() if hasattr(value, "get_short_id") else getattr(value, "paper_id", ""),
        "title": getattr(value, "title", ""),
        "authors": authors,
        "summary": getattr(value, "summary", ""),
        "published": getattr(value, "published", None),
        "updated": getattr(value, "updated", None),
        "primary_category": getattr(value, "primary_category", None),
        "categories": getattr(value, "categories", []),
        "doi": getattr(value, "doi", None),
        "entry_id": getattr(value, "entry_id", None),
        "pdf_url": getattr(value, "pdf_url", None),
        "journal_ref": getattr(value, "journal_ref", None),
        "comment": getattr(value, "comment", None),
    }


def _find_doi(record: dict[str, Any]) -> str | None:
    joined = " ".join(str(record.get(key, "")) for key in ("doi", "journal_ref", "comment", "summary"))
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", joined, flags=re.I)
    return match.group(0).rstrip(".,;) ") if match else None


def _year_from_date(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"(\d{4})", value)
    return int(match.group(1)) if match else None


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())