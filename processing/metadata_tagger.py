"""Normalize searchable metadata on paper chunks."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable, Mapping

from .chunker import Chunk


class MetadataTagger:
    """Attach a stable, JSON-safe metadata contract to chunks."""

    def tag(
        self, chunk: Chunk, paper_metadata: Mapping[str, Any] | None = None
    ) -> Chunk:
        source = {**dict(paper_metadata or {}), **dict(chunk.metadata or {})}
        metadata = _json_safe(source)
        metadata.update(
            {
                "paper_id": chunk.paper_id,
                "section": _slug(chunk.section) or "other",
                "section_family": _section_family(chunk.section),
            }
        )
        if "year" in metadata:
            metadata["year"] = _year(metadata["year"])
        for key in ("authors", "categories", "keywords"):
            if key in metadata:
                metadata[key] = _string_list(metadata[key])
        return replace(chunk, metadata=metadata)

    def tag_chunks(
        self, chunks: Iterable[Chunk], paper_metadata: Mapping[str, Any] | None = None
    ) -> list[Chunk]:
        return [self.tag(chunk, paper_metadata) for chunk in chunks]


def tag_chunk(chunk: Chunk, paper_metadata: Mapping[str, Any] | None = None) -> Chunk:
    return MetadataTagger().tag(chunk, paper_metadata)


def tag_chunks(
    chunks: Iterable[Chunk], paper_metadata: Mapping[str, Any] | None = None
) -> list[Chunk]:
    return MetadataTagger().tag_chunks(chunks, paper_metadata)


def _section_family(section: str) -> str:
    value = _slug(section)
    groups = {
        "front": {"front_matter", "abstract", "acknowledgements"},
        "context": {"introduction", "related_work", "background"},
        "method": {
            "method",
            "methodology",
            "architecture",
            "approach",
            "proposed_method",
            "proposed_approach",
        },
        "evidence": {"experiments", "results", "analysis", "evaluation", "discussion"},
        "closing": {"limitations", "conclusion", "future_work"},
        "back": {"references", "appendix", "supplementary_material"},
    }
    return next(
        (family for family, members in groups.items() if value in members), "other"
    )


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def _year(value: Any) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return int(match.group()) if match else None


def _string_list(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
