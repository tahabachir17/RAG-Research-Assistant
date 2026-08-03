"""Stage-agnostic information-retrieval metrics for ranked identifiers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def precision_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return binary precision at ``k``; missing result slots count as nonrelevant."""

    _validate_k(k)
    labels = set(_unique(relevant))
    return sum(item in labels for item in _unique(ranking)[:k]) / k


def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return the fraction of relevant identifiers retrieved by ``k``."""

    _validate_k(k)
    labels = set(_unique(relevant))
    if not labels:
        return 0.0
    return len(set(_unique(ranking)[:k]) & labels) / len(labels)


def hit_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return 1 when any relevant identifier occurs by ``k``, otherwise 0."""

    _validate_k(k)
    labels = set(_unique(relevant))
    return float(any(item in labels for item in _unique(ranking)[:k]))


def reciprocal_rank(ranking: Sequence[str], relevant: Iterable[str]) -> float:
    """Return reciprocal rank of the first relevant identifier in the ranking."""

    labels = set(_unique(relevant))
    return next(
        (1.0 / rank for rank, item in enumerate(_unique(ranking), 1) if item in labels),
        0.0,
    )


def ndcg_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return normalized discounted cumulative gain with binary relevance."""

    _validate_k(k)
    labels = set(_unique(relevant))
    if not labels:
        return 0.0
    gains = [item in labels for item in _unique(ranking)[:k]]
    dcg = sum(float(gain) / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_count = min(k, len(labels))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg


def ranked_metrics(
    ranking: Sequence[str], relevant: Iterable[str], *, ks: Sequence[int] = (5, 8, 20)
) -> dict[str, float]:
    """Compute all requested metrics for any ranked IDs and relevance labels."""

    labels = _unique(relevant)
    metrics: dict[str, float] = {"mrr": reciprocal_rank(ranking, labels)}
    for k in ks:
        _validate_k(k)
        metrics[f"hit@{k}"] = hit_at_k(ranking, labels, k)
        metrics[f"precision@{k}"] = precision_at_k(ranking, labels, k)
        metrics[f"recall@{k}"] = recall_at_k(ranking, labels, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranking, labels, k)
    return metrics
