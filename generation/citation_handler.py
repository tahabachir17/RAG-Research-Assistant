"""Validate numeric citations and expose only sources used in an answer."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

try:
    from .context_assembler import CitationSource
except ImportError:
    from context_assembler import CitationSource

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(slots=True)
class CitationValidationResult:
    valid: bool
    cited_numbers: list[int]
    unknown_numbers: list[int]
    unused_numbers: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_citations(
    answer: str, citation_map: dict[int, CitationSource]
) -> CitationValidationResult:
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    _validate_map(citation_map)
    cited = list(dict.fromkeys(int(value) for value in _CITATION.findall(answer)))
    unknown = [number for number in cited if number not in citation_map]
    unused = [number for number in citation_map if number not in cited]
    return CitationValidationResult(not unknown, cited, unknown, unused)


def build_source_list(
    answer: str, citation_map: dict[int, CitationSource]
) -> list[dict[str, Any]]:
    validation = validate_citations(answer, citation_map)
    sources: list[dict[str, Any]] = []
    for number in validation.cited_numbers:
        source = citation_map.get(number)
        if source is None:
            continue
        sources.append(
            {
                "citation_number": number,
                "paper_id": source.paper_id,
                "chunk_id": source.chunk_id,
                "title": source.title,
                "authors": list(source.authors),
                "year": source.year,
                "section": source.section,
                "primary_category": None,
                "categories": [],
                "doi": None,
                "url": source.url,
                "pdf_url": None,
                "published": None,
                "updated": None,
                "abstract": None,
                "local_pdf_path": None,
            }
        )
    return sources


def _validate_map(citation_map: dict[int, CitationSource]) -> None:
    if not isinstance(citation_map, dict):
        raise TypeError("citation_map must be a dictionary")
    for number, source in citation_map.items():
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise ValueError("citation_map keys must be positive integers")
        if not isinstance(source, CitationSource):
            raise TypeError("citation_map values must be CitationSource objects")
