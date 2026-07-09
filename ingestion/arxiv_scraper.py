from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

try:
    import arxiv
except ImportError:  # Dependency is required only when running ArXiv discovery.
    arxiv = None  # type: ignore[assignment]


@dataclass(slots=True)
class Paper:
    """Normalized paper record returned by the ArXiv discovery step."""

    paper_id: str
    title: str
    authors: list[str]
    summary: str
    published: str | None
    updated: str | None
    primary_category: str
    categories: list[str]
    pdf_url: str | None
    entry_id: str | None
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "summary": self.summary,
            "published": self.published,
            "updated": self.updated,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "entry_id": self.entry_id,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "comment": self.comment,
            **self.metadata,
        }


class ArxivScraper:
    """Thin ArXiv API client used by the ingestion pipeline."""

    def __init__(
        self,
        page_size: int = 100,
        delay_seconds: float = 3.0,
        num_retries: int = 3,
    ) -> None:
        if arxiv is None:
            self.client = None
            return
        self.client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )

    def search(
        self,
        query: str,
        max_results: int = 50,
        sort_by: Any = None,
        sort_order: Any = None,
    ) -> list[Paper]:
        if arxiv is None:
            raise ImportError("The arxiv package is required for ArxivScraper.search")
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if max_results < 1:
            raise ValueError("max_results must be greater than zero")

        search = arxiv.Search(
            query=query.strip(),
            max_results=max_results,
            sort_by=sort_by or arxiv.SortCriterion.Relevance,
            sort_order=sort_order or arxiv.SortOrder.Descending,
        )
        return [
            paper_from_arxiv_result(result) for result in self.client.results(search)
        ]


def paper_from_arxiv_result(result: Any) -> Paper:
    """Convert an arxiv.Result-like object into the local Paper schema."""

    return Paper(
        paper_id=_short_id(result),
        title=_clean_one_line(getattr(result, "title", "")),
        authors=[author.name for author in getattr(result, "authors", [])],
        summary=_clean_one_line(getattr(result, "summary", "")),
        published=_iso_or_none(getattr(result, "published", None)),
        updated=_iso_or_none(getattr(result, "updated", None)),
        primary_category=getattr(result, "primary_category", "unknown") or "unknown",
        categories=list(getattr(result, "categories", []) or []),
        pdf_url=getattr(result, "pdf_url", None),
        entry_id=getattr(result, "entry_id", None),
        doi=getattr(result, "doi", None),
        journal_ref=getattr(result, "journal_ref", None),
        comment=getattr(result, "comment", None),
    )


def papers_to_dicts(papers: Iterable[Paper]) -> list[dict[str, Any]]:
    return [
        paper.to_dict() if isinstance(paper, Paper) else dict(paper) for paper in papers
    ]


def _short_id(result: Any) -> str:
    if hasattr(result, "get_short_id"):
        return result.get_short_id()
    entry_id = getattr(result, "entry_id", "")
    return entry_id.rstrip("/").split("/")[-1]


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _clean_one_line(value: str) -> str:
    return " ".join(str(value).split())
