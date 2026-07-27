"""Hybrid dense/sparse retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .models import RetrievalResult


class HybridRetriever:
    """Fuse dense and sparse rankings without comparing their raw scores."""

    def __init__(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
        *,
        rrf_k: int = 60,
        default_top_k: int = 20,
        candidate_top_k: int = 50,
    ) -> None:
        for name, retriever in (
            ("dense_retriever", dense_retriever),
            ("sparse_retriever", sparse_retriever),
        ):
            if retriever is None or not callable(getattr(retriever, "search", None)):
                raise TypeError(f"{name} must provide a search method")
        self.rrf_k = _positive_integer(rrf_k, rrf_k, "rrf_k")
        self.default_top_k = _positive_integer(
            default_top_k, default_top_k, "default_top_k"
        )
        self.candidate_top_k = _positive_integer(
            candidate_top_k, candidate_top_k, "candidate_top_k"
        )
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        *,
        candidate_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve candidates from both backends and return their RRF ranking."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        limit = _positive_integer(top_k, self.default_top_k, "top_k")
        candidate_limit = _positive_integer(
            candidate_top_k, max(self.candidate_top_k, limit), "candidate_top_k"
        )
        kwargs = {
            "top_k": candidate_limit,
            "filters": filters,
            "score_threshold": score_threshold,
        }
        return self.fuse(
            self.dense_retriever.search(query, **kwargs),
            self.sparse_retriever.search(query, **kwargs),
            top_k=limit,
        )

    def fuse(
        self, *rankings: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Fuse already-produced rankings; each chunk contributes once per list."""
        limit = _positive_integer(top_k, self.default_top_k, "top_k")
        scores: dict[str, float] = {}
        representatives: dict[str, RetrievalResult] = {}
        best_ranks: dict[str, int] = {}
        for ranking in rankings:
            if not isinstance(ranking, (list, tuple)):
                raise TypeError("each ranking must be a list or tuple")
            seen: set[str] = set()
            for rank, result in enumerate(ranking, start=1):
                if not isinstance(result, RetrievalResult):
                    raise TypeError("rankings may contain only RetrievalResult objects")
                if result.chunk_id in seen:
                    continue
                seen.add(result.chunk_id)
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )
                representatives.setdefault(result.chunk_id, result)
                best_ranks[result.chunk_id] = min(
                    rank, best_ranks.get(result.chunk_id, rank)
                )
        ordered = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], best_ranks[chunk_id], chunk_id),
        )
        return [
            replace(
                representatives[chunk_id],
                score=scores[chunk_id],
                source="hybrid",
                authors=list(representatives[chunk_id].authors),
                metadata=dict(representatives[chunk_id].metadata),
            )
            for chunk_id in ordered[:limit]
        ]

    def health_check(self) -> dict[str, Any]:
        def health(retriever: Any) -> dict[str, Any]:
            check = getattr(retriever, "health_check", None)
            return check() if callable(check) else {"available": True}

        dense, sparse = health(self.dense_retriever), health(self.sparse_retriever)
        flags = ("connected", "collection_exists", "loaded", "mapping_valid")
        healthy = all(
            report.get(flag) is not False
            for report in (dense, sparse)
            for flag in flags
        )
        return {"healthy": healthy, "dense": dense, "sparse": sparse}


def _positive_integer(value: int | None, default: int, name: str) -> int:
    resolved = default if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return resolved
