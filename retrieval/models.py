"""Shared result models for retrieval backends."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class RetrievalResult:
    """A backend-neutral retrieved chunk with its original payload preserved."""

    chunk_id: str
    text: str
    score: float
    source: str
    paper_id: str | None = None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    section: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunk_id = str(self.chunk_id).strip()
        self.text = str(self.text).strip()
        self.source = str(self.source).strip()
        self.score = float(self.score)
        if not self.chunk_id:
            raise ValueError("Retrieval result is missing chunk_id")
        if not self.text:
            raise ValueError(f"Retrieval result {self.chunk_id!r} is missing text")
        if not self.source:
            raise ValueError("Retrieval result source must not be empty")
        if not math.isfinite(self.score):
            raise ValueError("Retrieval result score must be finite")
        if not isinstance(self.metadata, dict):
            raise TypeError("Retrieval result metadata must be a dictionary")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        score: float,
        source: str,
        fallback_chunk_id: Any = None,
    ) -> "RetrievalResult":
        """Build a result while retaining an unmodified copy of ``payload``."""

        if not isinstance(payload, Mapping):
            raise TypeError("Retrieval payload must be a mapping")
        original = dict(payload)
        nested = payload.get("metadata")
        nested_metadata = nested if isinstance(nested, Mapping) else {}

        def value(name: str, default: Any = None) -> Any:
            candidate = payload.get(name)
            return (
                candidate
                if candidate not in (None, "")
                else nested_metadata.get(name, default)
            )

        authors_value = value("authors", [])
        if isinstance(authors_value, str):
            authors = [authors_value] if authors_value.strip() else []
        elif isinstance(authors_value, (list, tuple, set)):
            authors = [
                str(author).strip() for author in authors_value if str(author).strip()
            ]
        else:
            authors = []

        year_value = value("year")
        try:
            year = int(year_value) if year_value not in (None, "") else None
        except (TypeError, ValueError):
            year = None

        return cls(
            chunk_id=str(value("chunk_id", fallback_chunk_id) or ""),
            text=str(value("text", "") or ""),
            score=score,
            source=source,
            paper_id=_optional_string(value("paper_id")),
            title=_optional_string(value("title")),
            authors=authors,
            year=year,
            section=_optional_string(value("section")),
            url=_optional_string(value("url")),
            metadata=original,
        )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
