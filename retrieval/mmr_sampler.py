"""Diversity sampling with Maximal Marginal Relevance (MMR)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .models import RetrievalResult


def maximal_marginal_relevance(
    query_embedding: Sequence[float],
    candidate_embeddings: Sequence[Sequence[float]],
    *,
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[int]:
    """Return selected candidate positions in MMR selection order."""
    _positive(top_k, "top_k")
    if not 0.0 <= float(lambda_mult) <= 1.0:
        raise ValueError("lambda_mult must be between 0 and 1")
    query = np.asarray(query_embedding, dtype=float)
    candidates = np.asarray(candidate_embeddings, dtype=float)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1 or query.size == 0:
        raise ValueError("query_embedding must be a non-empty one-dimensional vector")
    if candidates.size == 0:
        return []
    if candidates.ndim != 2:
        raise ValueError("candidate_embeddings must be a two-dimensional matrix")
    if candidates.shape[1] != query.size:
        raise ValueError("query and candidate embedding dimensions differ")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(candidates)):
        raise ValueError("embeddings contain NaN or infinite values")
    relevance = _cosine(candidates, query.reshape(1, -1)).reshape(-1)
    pairwise = _cosine(candidates, candidates)
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < min(top_k, len(candidates)):

        def score(index: int) -> float:
            redundancy = max(
                (pairwise[index, chosen] for chosen in selected), default=0.0
            )
            return (
                float(lambda_mult) * relevance[index]
                - (1.0 - float(lambda_mult)) * redundancy
            )

        winner = max(remaining, key=lambda index: (score(index), -index))
        selected.append(winner)
        remaining.remove(winner)
    return selected


class MMRSampler:
    """Embed a query and candidate texts, then select a diverse subset."""

    def __init__(
        self, embedder: Any, *, lambda_mult: float = 0.5, default_top_k: int = 5
    ) -> None:
        if embedder is None:
            raise ValueError("embedder is required")
        if not 0.0 <= float(lambda_mult) <= 1.0:
            raise ValueError("lambda_mult must be between 0 and 1")
        _positive(default_top_k, "default_top_k")
        self.embedder, self.lambda_mult, self.default_top_k = (
            embedder,
            float(lambda_mult),
            default_top_k,
        )

    def sample(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
        *,
        lambda_mult: float | None = None,
    ) -> list[RetrievalResult]:
        """Return candidates in MMR order without replacing relevance scores."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError("candidates must be a sequence of RetrievalResult objects")
        if any(not isinstance(item, RetrievalResult) for item in candidates):
            raise TypeError("candidates may contain only RetrievalResult objects")
        limit = self.default_top_k if top_k is None else top_k
        _positive(limit, "top_k")
        diversity = self.lambda_mult if lambda_mult is None else float(lambda_mult)
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("lambda_mult must be between 0 and 1")
        if not candidates:
            return []
        embeddings = _encode(
            self.embedder, [query, *[item.text for item in candidates]]
        )
        if embeddings.shape[0] != len(candidates) + 1:
            raise ValueError(
                "embedder returned a different number of embeddings than texts"
            )
        positions = maximal_marginal_relevance(
            embeddings[0], embeddings[1:], top_k=limit, lambda_mult=diversity
        )
        return [candidates[position] for position in positions]


def _encode(embedder: Any, texts: list[str]) -> np.ndarray:
    if callable(getattr(embedder, "encode_texts", None)):
        encoded = embedder.encode_texts(texts)
    elif callable(getattr(embedder, "encode", None)):
        try:
            encoded = embedder.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
        except TypeError:
            encoded = embedder.encode(texts)
    else:
        raise TypeError("embedder must provide encode_texts or encode")
    array = np.asarray(encoded, dtype=float)
    if array.ndim == 1 and len(texts) == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("embedder output must be a two-dimensional matrix")
    return array


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    products = left @ right.T
    norms = (
        np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    )
    return np.divide(
        products, norms, out=np.zeros_like(products, dtype=float), where=norms != 0
    )


def _positive(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
