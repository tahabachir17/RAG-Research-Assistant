"""Entity-aware retrieval for questions that name multiple papers."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

from retrieval.models import RetrievalResult

from .entities import extract_named_papers


@dataclass(frozen=True, slots=True)
class EntityRetrievalReport:
    """Evidence coverage for one explicitly named paper."""

    title: str
    query: str
    chunk_ids: tuple[str, ...]

    @property
    def hit(self) -> bool:
        return bool(self.chunk_ids)


def retrieve_per_entity(
    question: str,
    index_path: str,
    *,
    retriever: Any,
    known_titles: Iterable[str] | None = None,
    per_entity_top_k: int = 4,
    candidate_k: int = 30,
    reranker: Any | None = None,
    excluded_sections: Collection[str] | None = None,
    include_general_query: bool = True,
) -> tuple[list[RetrievalResult], list[EntityRetrievalReport]]:
    """Retrieve and round-robin evidence for every named paper title.

    Entity searches are filtered by result title before diversification. This
    prevents a strong result for one paper from consuming another paper's quota.
    """

    from .cli import DEFAULT_EXCLUDED_SECTIONS, retrieve_ranked_results

    titles = extract_named_papers(question, known_titles)
    if len(titles) < 2:
        raise ValueError("entity-aware retrieval requires at least two named titles")
    if excluded_sections is None:
        excluded_sections = DEFAULT_EXCLUDED_SECTIONS

    entity_rankings: list[list[RetrievalResult]] = []
    reports: list[EntityRetrievalReport] = []
    for title in titles:
        entity_query = _entity_query(question, title, titles)
        try:
            ranking = retrieve_ranked_results(
                entity_query,
                index_path,
                top_k=per_entity_top_k,
                candidate_k=max(candidate_k, per_entity_top_k),
                max_chunks_per_paper=per_entity_top_k,
                max_chunks_per_section=2,
                excluded_sections=excluded_sections,
                reranker=reranker,
                retriever=retriever,
                result_filter=lambda result, expected=title: _title_matches(
                    expected, result
                ),
            )
        except ValueError as exc:
            if "No retrieval results" not in str(exc):
                raise
            ranking = []
        entity_rankings.append(ranking)
        reports.append(
            EntityRetrievalReport(
                title=title,
                query=entity_query,
                chunk_ids=tuple(result.chunk_id for result in ranking),
            )
        )

    balanced = _round_robin(entity_rankings)
    if include_general_query:
        try:
            general = retrieve_ranked_results(
                question,
                index_path,
                top_k=per_entity_top_k,
                candidate_k=max(candidate_k, per_entity_top_k),
                excluded_sections=excluded_sections,
                reranker=reranker,
                retriever=retriever,
            )
        except ValueError as exc:
            if "No retrieval results" not in str(exc):
                raise
            general = []
        balanced = _deduplicate([*balanced, *general])
    if not balanced:
        raise ValueError("No retrieval results were found for the supplied question")
    return balanced, reports


def _entity_query(question: str, title: str, named_titles: list[str]) -> str:
    focus = question
    for other_title in named_titles:
        if other_title != title:
            focus = re.sub(re.escape(other_title), " ", focus, flags=re.IGNORECASE)
    focus = re.sub(r"\s+", " ", focus).strip(" ,;:")
    return f'"{title}" {focus}'.strip()


def _title_matches(expected: str, result: RetrievalResult) -> bool:
    actual = result.title
    if not actual and isinstance(result.metadata, dict):
        actual = result.metadata.get("title")
        nested = result.metadata.get("metadata")
        if not actual and isinstance(nested, dict):
            actual = nested.get("title")
    expected_key = _title_key(expected)
    actual_key = _title_key(str(actual or ""))
    return bool(
        expected_key
        and actual_key
        and (expected_key == actual_key or expected_key in actual_key or actual_key in expected_key)
    )


def _title_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _round_robin(rankings: list[list[RetrievalResult]]) -> list[RetrievalResult]:
    ordered: list[RetrievalResult] = []
    for rank in range(max((len(items) for items in rankings), default=0)):
        for items in rankings:
            if rank < len(items):
                ordered.append(items[rank])
    return _deduplicate(ordered)


def _deduplicate(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = set()
    unique: list[RetrievalResult] = []
    for result in results:
        if result.chunk_id in seen:
            continue
        seen.add(result.chunk_id)
        unique.append(result)
    return unique
