"""Dense retrieval from a cosine Qdrant collection."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .models import RetrievalResult

try:
    from qdrant_client.http import models
except ImportError:  # Keep fake-client unit tests usable without qdrant-client.
    models = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Embed queries and retrieve their closest Qdrant chunk payloads.

    Filter values may be scalars (exact match), lists (membership), or numeric
    range dictionaries containing ``gt``, ``gte``, ``lt``, and/or ``lte``.
    Non-core fields are resolved below the indexing payload's ``metadata`` key.
    """

    def __init__(
        self,
        qdrant_client: Any,
        embedder: Any,
        collection_name: str,
        default_top_k: int = 10,
    ) -> None:
        if qdrant_client is None:
            raise ValueError("qdrant_client is required")
        if embedder is None:
            raise ValueError("embedder is required")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not isinstance(default_top_k, int) or default_top_k <= 0:
            raise ValueError("default_top_k must be a positive integer")
        self.client = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name.strip()
        self.default_top_k = default_top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Encode a non-empty query with the indexing embedder and search Qdrant."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        if hasattr(self.embedder, "encode_texts"):
            encoded = self.embedder.encode_texts([query])
        elif hasattr(self.embedder, "encode_query"):
            encoded = self.embedder.encode_query(query)
        elif hasattr(self.embedder, "encode"):
            encoded = self.embedder.encode(query)
        else:
            raise TypeError(
                "embedder must provide encode_texts, encode_query, or encode"
            )
        array = np.asarray(encoded, dtype=float)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        return self.search_by_vector(array, top_k, filters, score_threshold)

    def search_by_vector(
        self,
        query_vector: Sequence[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Search with an already-computed one-dimensional dense vector."""

        limit = self._validate_top_k(top_k)
        vector = np.asarray(query_vector, dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("query_vector must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("query_vector contains NaN or infinite values")
        if score_threshold is not None and not math.isfinite(float(score_threshold)):
            raise ValueError("score_threshold must be finite")

        health = self.health_check()
        if not health["connected"]:
            raise ConnectionError(
                f"Qdrant is not reachable: {health.get('error', 'unknown error')}"
            )
        if not health["collection_exists"]:
            raise LookupError(
                f"Qdrant collection {self.collection_name!r} was not found"
            )
        if health.get("points_count") == 0:
            raise LookupError(f"Qdrant collection {self.collection_name!r} is empty")
        if health.get("distance") and str(health["distance"]).casefold() != "cosine":
            raise ValueError(
                f"Collection distance must be Cosine, got {health['distance']!r}"
            )
        if health.get("dimension_match") is False:
            raise ValueError(
                "Configured embedder dimension does not match the Qdrant collection"
            )
        vector_size = health.get("vector_size")
        if vector_size is not None and vector.size != vector_size:
            raise ValueError(
                f"Query vector dimension {vector.size} does not match collection dimension {vector_size}"
            )

        query_filter = _build_filter(filters)
        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector.tolist(),
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                )
                hits = getattr(response, "points", response)
            elif hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector.tolist(),
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                )
            else:
                raise TypeError("Qdrant client has neither query_points nor search")
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Dense search failed for collection {self.collection_name!r}: {exc}"
            ) from exc

        results: list[RetrievalResult] = []
        for hit in hits or []:
            payload = _attribute(hit, "payload", {}) or {}
            score = _attribute(hit, "score", None)
            point_id = _attribute(hit, "id", None)
            try:
                result = RetrievalResult.from_payload(
                    payload,
                    score=float(score),
                    source="dense",
                    fallback_chunk_id=point_id,
                )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed Qdrant point %r: %s", point_id, exc)
                continue
            results.append(result)
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def health_check(self) -> dict[str, Any]:
        """Report collection connectivity, configuration, and dimension agreement."""

        result: dict[str, Any] = {
            "connected": False,
            "collection_exists": False,
            "collection_name": self.collection_name,
            "points_count": None,
            "vector_size": None,
            "distance": None,
            "collection_status": None,
            "embedder_dimension": _embedder_dimension(self.embedder),
            "dimension_match": None,
        }
        try:
            if hasattr(self.client, "collection_exists"):
                exists = bool(self.client.collection_exists(self.collection_name))
            else:
                try:
                    self.client.get_collection(self.collection_name)
                    exists = True
                except Exception as exc:
                    if "404" in str(exc) or "not found" in str(exc).casefold():
                        exists = False
                    else:
                        raise
            result["connected"] = True
            result["collection_exists"] = exists
            if not exists:
                return result
            info = self.client.get_collection(self.collection_name)
            result["points_count"] = _attribute(info, "points_count", None)
            result["collection_status"] = _enum_value(_attribute(info, "status", None))
            vector_config = _vector_config(info)
            result["vector_size"] = _attribute(vector_config, "size", None)
            result["distance"] = _enum_value(
                _attribute(vector_config, "distance", None)
            )
            expected = result["embedder_dimension"]
            actual = result["vector_size"]
            result["dimension_match"] = (
                expected == actual
                if expected is not None and actual is not None
                else None
            )
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("Qdrant health check failed: %s", exc)
        return result

    def _validate_top_k(self, value: int | None) -> int:
        top_k = self.default_top_k if value is None else value
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k


def _build_filter(filters: dict[str, Any] | None) -> Any | None:
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise TypeError("filters must be a dictionary")
    conditions = []
    range_keys = {"gt", "gte", "lt", "lte"}
    for key, value in filters.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("filter field names must be non-empty strings")
        payload_key = (
            key
            if "." in key or key in {"chunk_id", "paper_id", "section", "text"}
            else f"metadata.{key}"
        )
        if isinstance(value, Mapping):
            if not value or not set(value).issubset(range_keys):
                raise ValueError(f"Invalid range filter for {key!r}")
            if models is None:
                condition = {"key": payload_key, "range": dict(value)}
            else:
                condition = models.FieldCondition(
                    key=payload_key, range=models.Range(**value)
                )
        elif isinstance(value, (list, tuple, set)):
            if not value:
                raise ValueError(f"Membership filter for {key!r} must not be empty")
            if models is None:
                condition = {"key": payload_key, "match": {"any": list(value)}}
            else:
                condition = models.FieldCondition(
                    key=payload_key, match=models.MatchAny(any=list(value))
                )
        else:
            if models is None:
                condition = {"key": payload_key, "match": {"value": value}}
            else:
                condition = models.FieldCondition(
                    key=payload_key, match=models.MatchValue(value=value)
                )
        conditions.append(condition)
    return (
        models.Filter(must=conditions) if models is not None else {"must": conditions}
    )


def _vector_config(info: Any) -> Any:
    config = _attribute(info, "config", {})
    params = _attribute(config, "params", {})
    vectors = _attribute(params, "vectors", {})
    if isinstance(vectors, Mapping):
        if "size" in vectors:
            return vectors
        if len(vectors) == 1:
            return next(iter(vectors.values()))
        if vectors:
            raise ValueError("Named-vector collections require an explicit vector name")
    return vectors


def _embedder_dimension(embedder: Any) -> int | None:
    for owner in (embedder, getattr(embedder, "model", None)):
        if owner is None:
            continue
        for name in ("dimension", "embedding_dimension"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and value > 0:
                return value
        getter = getattr(owner, "get_sentence_embedding_dimension", None)
        if callable(getter):
            value = getter()
            if isinstance(value, int) and value > 0:
                return value
    return None


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    return (
        value.get(name, default)
        if isinstance(value, Mapping)
        else getattr(value, name, default)
    )


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)
