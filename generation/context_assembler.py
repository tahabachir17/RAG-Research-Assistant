"""Turn ranked retrieval results into bounded numbered context."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from retrieval.models import RetrievalResult
except ImportError:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from retrieval.models import RetrievalResult


@dataclass(slots=True)
class CitationSource:
    citation_number: int
    paper_id: str
    chunk_id: str
    title: str
    authors: list[str]
    year: int | None
    section: str
    url: str | None
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AssembledContext:
    context_block: str
    citation_map: dict[int, CitationSource]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextAssembler:
    """Format the highest-ranked complete chunks within a token budget."""

    def __init__(
        self,
        max_context_tokens: int = 4000,
        *,
        token_counter: Callable[[str], int] | None = None,
        dedupe_paper_sections: bool = False,
        evidence_packing_mode: str = "gold",
        adjacent_chunk_lookup: Callable[[RetrievalResult], Sequence[RetrievalResult]]
        | None = None,
        section_chunk_lookup: Callable[[RetrievalResult], Sequence[RetrievalResult]]
        | None = None,
    ) -> None:
        if (
            not isinstance(max_context_tokens, int)
            or isinstance(max_context_tokens, bool)
            or max_context_tokens <= 0
        ):
            raise ValueError("max_context_tokens must be a positive integer")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable")
        if evidence_packing_mode not in {"gold", "adjacent", "section"}:
            raise ValueError("evidence_packing_mode must be gold, adjacent, or section")
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter or _whitespace_tokens
        self.dedupe_paper_sections = bool(dedupe_paper_sections)
        self.evidence_packing_mode = evidence_packing_mode
        self.adjacent_chunk_lookup = adjacent_chunk_lookup
        self.section_chunk_lookup = section_chunk_lookup

    def assemble(
        self,
        ranked_chunks: Sequence[RetrievalResult],
        *,
        required_concepts: Sequence[Any] = (),
    ) -> AssembledContext:
        if isinstance(ranked_chunks, (str, bytes)) or not isinstance(
            ranked_chunks, Sequence
        ):
            raise TypeError("ranked_chunks must be a sequence of RetrievalResult")
        if any(not isinstance(item, RetrievalResult) for item in ranked_chunks):
            raise TypeError("ranked_chunks may contain only RetrievalResult objects")
        packed_chunks = self._pack(ranked_chunks, required_concepts)

        blocks: list[str] = []
        citation_map: dict[int, CitationSource] = {}
        seen_sections: set[tuple[str, str]] = set()
        used_tokens = 0
        for result in packed_chunks:
            paper_id = result.paper_id or ""
            section = result.section or "unknown"
            dedupe_key = (paper_id, section.casefold())
            if self.dedupe_paper_sections and dedupe_key in seen_sections:
                continue
            citation_number = len(citation_map) + 1
            title = (
                result.title or _metadata_value(result.metadata, "title") or "Untitled"
            )
            authors = result.authors or _authors(result.metadata)
            year = result.year or _year(result.metadata)
            url = result.url or _metadata_value(result.metadata, "url")
            header = (
                f"[{citation_number}] Title: {title} | Authors: "
                f"{', '.join(authors) if authors else 'Unknown'} | "
                f"Year: {year if year is not None else 'Unknown'}"
            )
            block = f'{header}\nSection: {section}\n"{result.text}"'
            block_tokens = self.token_counter(block)
            if not isinstance(block_tokens, int) or block_tokens < 0:
                raise ValueError("token_counter must return a non-negative integer")
            if used_tokens + block_tokens > self.max_context_tokens:
                # A long high-ranked chunk must not prevent shorter later
                # evidence from using the remaining context budget.
                continue
            used_tokens += block_tokens
            blocks.append(block)
            citation_map[citation_number] = CitationSource(
                citation_number=citation_number,
                paper_id=paper_id,
                chunk_id=result.chunk_id,
                title=title,
                authors=list(authors),
                year=year,
                section=section,
                url=url,
                text=result.text,
            )
            seen_sections.add(dedupe_key)
        return AssembledContext("\n\n".join(blocks), citation_map)

    def _pack(
        self,
        ranked_chunks: Sequence[RetrievalResult],
        required_concepts: Sequence[Any],
    ) -> list[RetrievalResult]:
        if self.evidence_packing_mode == "gold" or len(required_concepts) <= 1:
            return list(ranked_chunks)
        lookup = (
            self.adjacent_chunk_lookup
            if self.evidence_packing_mode == "adjacent"
            else self.section_chunk_lookup
        )
        if lookup is None:
            raise ValueError(
                f"{self.evidence_packing_mode} evidence packing requires a chunk lookup"
            )
        packed: list[RetrievalResult] = []
        seen: set[str] = set()
        for gold in ranked_chunks:
            candidates = [gold, *lookup(gold)]
            for candidate in candidates:
                if not isinstance(candidate, RetrievalResult):
                    raise TypeError("evidence packing lookups must return RetrievalResult")
                if candidate.paper_id != gold.paper_id:
                    continue
                if (
                    self.evidence_packing_mode == "section"
                    and candidate.section != gold.section
                ):
                    continue
                if candidate.chunk_id in seen:
                    continue
                seen.add(candidate.chunk_id)
                packed.append(candidate)
        return packed


def _whitespace_tokens(text: str) -> int:
    return len(text.split())


def _nested_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = metadata.get("metadata")
    return nested if isinstance(nested, Mapping) else {}


def _metadata_value(metadata: Mapping[str, Any], name: str) -> str | None:
    value = metadata.get(name) or _nested_metadata(metadata).get(name)
    normalized = str(value).strip() if value is not None else ""
    return normalized or None


def _authors(metadata: Mapping[str, Any]) -> list[str]:
    value = metadata.get("authors") or _nested_metadata(metadata).get("authors") or []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(author).strip() for author in value if str(author).strip()]
    return []


def _year(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("year") or _nested_metadata(metadata).get("year")
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
