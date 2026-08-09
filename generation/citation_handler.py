"""Validate project citations and expose only sources used in an answer."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

try:
    from .context_assembler import CitationSource
except ImportError:
    from context_assembler import CitationSource

_BRACKETED_TEXT = re.compile(r"\[([^\[\]]+)\]")
_NUMERIC_CITATION_GROUP = re.compile(r"\d+(?:\s*[,;]\s*\d+)*")
_UNSUPPORTED_CITATION = re.compile(
    r"(?:【[^】\r\n]*†[^】\r\n]*】|ã€[^\r\n]*?(?:†|â€ )[^\r\n]*?ã€‘)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class CitationValidationResult:
    valid: bool
    cited_numbers: list[int]
    unknown_numbers: list[int]
    unused_numbers: list[int]
    unsupported_markers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_citations(
    answer: str, citation_map: dict[int, CitationSource]
) -> CitationValidationResult:
    """Validate citation syntax and mapping against the exact prompt context.

    Context-backed answers require at least one mapped citation. An empty answer is
    valid only when no context was supplied. Unsupported provider-native markers
    are hard failures and are retained in the result for auditability.
    """

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    _validate_map(citation_map)
    cited = list(dict.fromkeys(_citation_numbers(answer)))
    unknown = [number for number in cited if number not in citation_map]
    unused = [number for number in citation_map if number not in cited]
    unsupported = unsupported_citation_markers(answer)
    has_required_citation = not citation_map or bool(cited)
    valid = not unknown and not unsupported and has_required_citation
    return CitationValidationResult(valid, cited, unknown, unused, unsupported)


def unsupported_citation_markers(answer: str) -> list[str]:
    """Return provider-native citations that violate the project's ``[n]`` form."""

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    return list(
        dict.fromkeys(match.group(0) for match in _UNSUPPORTED_CITATION.finditer(answer))
    )


def _citation_numbers(answer: str) -> list[int]:
    """Extract singleton and grouped numeric citations in textual order."""

    numbers: list[int] = []
    for match in _BRACKETED_TEXT.finditer(answer):
        content = match.group(1).strip()
        if not _NUMERIC_CITATION_GROUP.fullmatch(content):
            continue
        numbers.extend(int(value) for value in re.findall(r"\d+", content))
    return numbers


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