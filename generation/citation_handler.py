"""Validate project citations and expose only sources used in an answer."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
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
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._+#-][A-Za-z0-9]+)*", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "in", "is", "it", "of", "on", "or", "that", "the", "their",
        "this", "to", "was", "were", "with",
    }
)


@dataclass(slots=True)
class ClaimSupportFlag:
    claim_id: str
    claim: str
    citation_numbers: list[int]
    status: str
    checker: str
    reason: str
    score: float | None = None


@dataclass(slots=True)
class CitationValidationResult:
    valid: bool
    cited_numbers: list[int]
    unknown_numbers: list[int]
    unused_numbers: list[int]
    unsupported_markers: list[str]
    claim_support: list[ClaimSupportFlag] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_citations(
    answer: str,
    citation_map: dict[int, CitationSource],
    *,
    structured_data: Any | None = None,
    min_overlap: float = 0.2,
) -> CitationValidationResult:
    """Validate citation syntax and mapping against the exact prompt context.

    Context-backed answers require at least one mapped citation. An empty answer is
    valid only when no context was supplied. Unsupported provider-native markers
    are hard failures and are retained in the result for auditability.
    """

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    _validate_map(citation_map)
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1")
    cited = list(dict.fromkeys(_citation_numbers(answer)))
    unknown = [number for number in cited if number not in citation_map]
    unused = [number for number in citation_map if number not in cited]
    unsupported = unsupported_citation_markers(answer)
    has_required_citation = not citation_map or bool(cited)
    valid = not unknown and not unsupported and has_required_citation
    support = validate_claim_support(
        answer,
        citation_map,
        structured_data=structured_data,
        min_overlap=min_overlap,
    )
    return CitationValidationResult(valid, cited, unknown, unused, unsupported, support)


def validate_claim_support(
    answer: str,
    citation_map: dict[int, CitationSource],
    *,
    structured_data: Any | None = None,
    min_overlap: float = 0.2,
) -> list[ClaimSupportFlag]:
    """Flag atomic claims whose cited chunks have insufficient lexical overlap."""

    _validate_map(citation_map)
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1")
    flags: list[ClaimSupportFlag] = []
    for claim_id, claim, citations in extract_cited_claims(answer, structured_data):
        unknown = [number for number in citations if number not in citation_map]
        if unknown:
            flags.append(
                ClaimSupportFlag(
                    claim_id,
                    claim,
                    citations,
                    "unknown_citation",
                    "lexical_overlap",
                    f"Unknown citation numbers: {unknown}",
                )
            )
            continue
        if not citations:
            flags.append(
                ClaimSupportFlag(
                    claim_id,
                    claim,
                    [],
                    "missing_citation",
                    "lexical_overlap",
                    "The claim has no citation.",
                )
            )
            continue
        claim_tokens = _content_tokens(claim)
        evidence_tokens = set().union(
            *(_content_tokens(citation_map[number].text) for number in citations)
        )
        shared = claim_tokens & evidence_tokens
        score = len(shared) / len(claim_tokens) if claim_tokens else 0.0
        supported = bool(shared) and score >= min_overlap
        flags.append(
            ClaimSupportFlag(
                claim_id,
                claim,
                citations,
                "supported" if supported else "unsupported",
                "lexical_overlap",
                (
                    "Claim terms overlap cited evidence."
                    if supported
                    else "Insufficient lexical overlap with cited evidence."
                ),
                round(score, 4),
            )
        )
    return flags


def extract_cited_claims(
    answer: str, structured_data: Any | None = None
) -> list[tuple[str, str, list[int]]]:
    """Return stable atomic claim IDs, text, and citations from structured or plain output."""

    if isinstance(structured_data, Mapping):
        raw_claims = structured_data.get("claims")
        if isinstance(raw_claims, Sequence) and not isinstance(raw_claims, (str, bytes)):
            return [
                (
                    f"claim-{index}",
                    str(claim.get("text", "")).strip(),
                    _integer_citations(claim.get("citations")),
                )
                for index, claim in enumerate(raw_claims, 1)
                if isinstance(claim, Mapping) and str(claim.get("text", "")).strip()
            ]
        raw_items = structured_data.get("items")
        if isinstance(raw_items, Sequence) and not isinstance(raw_items, (str, bytes)):
            claims: list[tuple[str, str, list[int]]] = []
            for item_index, item in enumerate(raw_items, 1):
                if not isinstance(item, Mapping):
                    continue
                for field, cell in item.items():
                    if not isinstance(cell, Mapping):
                        continue
                    text = str(cell.get("text", "")).strip()
                    if text and not _is_absent(text):
                        claims.append(
                            (
                                f"claim-{item_index}:{field}",
                                text,
                                _integer_citations(cell.get("citations")),
                            )
                        )
            return claims
    claims = []
    for index, fragment in enumerate(re.split(r"(?<=[.!?])\s+|\n+", answer), 1):
        text = fragment.strip()
        if not text:
            continue
        citations = _citation_numbers(text)
        clean = re.sub(r"\s+([.,;:!?])", r"\1", _BRACKETED_TEXT.sub("", text))
        clean = clean.strip(" |.\t")
        if clean:
            claims.append((f"claim-{index}", clean, citations))
    return claims


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


def _integer_citations(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            number
            for number in value
            if isinstance(number, int) and not isinstance(number, bool)
        )
    )


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in (value.casefold() for value in _TOKEN.findall(str(text)))
        if token not in _STOP_WORDS and len(token) > 1
    }


def content_tokens(text: str) -> set[str]:
    """Return the normalized content-token set used by citation validation."""

    return _content_tokens(text)


def lexical_overlap_score(subject: str, evidence: str) -> float:
    """Score how much of ``subject`` is covered by ``evidence`` lexical content."""

    subject_tokens = _content_tokens(subject)
    if not subject_tokens:
        return 0.0
    return len(subject_tokens & _content_tokens(evidence)) / len(subject_tokens)


def _is_absent(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return normalized.startswith("not reported") or normalized.startswith("not provided")
