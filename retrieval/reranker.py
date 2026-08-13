"""Cross-encoder reranking for a bounded retrieval candidate set."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from .models import RetrievalResult

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Score complete ``(query, passage)`` pairs with Sentence Transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        model: Any | None = None,
        default_top_k: int = 8,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        _positive(default_top_k, "default_top_k")
        if batch_size is not None:
            _positive(batch_size, "batch_size")
        if max_length is not None:
            _positive(max_length, "max_length")
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required to construct the reranker"
                ) from exc
            options = {"max_length": max_length} if max_length is not None else {}
            model = CrossEncoder(model_name, **options)
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must provide a predict method")
        self.model_name, self.model = model_name.strip(), model
        self.default_top_k, self.batch_size, self.max_length = (
            default_top_k,
            batch_size,
            max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Re-score candidates and return stable descending relevance order."""
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
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        kwargs = {"batch_size": self.batch_size} if self.batch_size is not None else {}
        scores = np.asarray(self.model.predict(pairs, **kwargs), dtype=float).reshape(
            -1
        )
        if len(scores) != len(candidates):
            raise ValueError(
                "cross-encoder returned a different number of scores than candidates"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("cross-encoder returned NaN or infinite scores")
        order = np.argsort(-scores, kind="stable")[: min(limit, len(candidates))]
        return [
            replace(
                candidates[index],
                score=float(scores[index]),
                source="reranked",
                authors=list(candidates[index].authors),
                metadata=dict(candidates[index].metadata),
            )
            for index in order
        ]

    def search(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Alias for ``rerank`` for pipeline components with a search interface."""
        return self.rerank(query, candidates, top_k)


def _positive(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


Reranker = CrossEncoderReranker
