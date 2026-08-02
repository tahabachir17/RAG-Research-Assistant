"""Central construction of dense, sparse, and hybrid retrievers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .fallback_retriever import CorpusEnrichmentRetriever
from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .sparse_retriever import SparseRetriever


def build_retriever(config: Mapping[str, Any], **dependencies: Any) -> Any:
    """Build a retriever from configuration and injectable runtime objects."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    values = dict(config)
    kind = values.pop("type", values.pop("kind", None))
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("config must define retriever type")
    kind = kind.strip().casefold()
    if kind == "dense":
        client = dependencies.get("qdrant_client", values.pop("qdrant_client", None))
        embedder = dependencies.get("embedder", values.pop("embedder", None))
        collection = values.pop("collection_name", None)
        _require(client, "qdrant_client", kind)
        _require(embedder, "embedder", kind)
        _require(collection, "collection_name", kind)
        options = _take(values, "default_top_k")
        _reject_unknown(values, kind)
        return DenseRetriever(client, embedder, collection, **options)
    if kind == "sparse":
        path = dependencies.get("index_path", values.pop("index_path", None))
        _require(path, "index_path", kind)
        options = _take(values, "default_top_k")
        _reject_unknown(values, kind)
        return SparseRetriever(path, **options)
    if kind == "hybrid":
        dense = dependencies.get("dense_retriever", values.pop("dense_retriever", None))
        sparse = dependencies.get(
            "sparse_retriever", values.pop("sparse_retriever", None)
        )
        dense_config = values.pop("dense", values.pop("dense_config", None))
        sparse_config = values.pop("sparse", values.pop("sparse_config", None))
        if dense is None:
            if dense_config is None:
                raise ValueError(
                    "hybrid retriever requires dense config or dense_retriever"
                )
            dense = build_retriever(_with_type(dense_config, "dense"), **dependencies)
        if sparse is None:
            if sparse_config is None:
                raise ValueError(
                    "hybrid retriever requires sparse config or sparse_retriever"
                )
            sparse = build_retriever(
                _with_type(sparse_config, "sparse"), **dependencies
            )
        options = _take(values, "rrf_k", "default_top_k", "candidate_top_k")
        _reject_unknown(values, kind)
        return HybridRetriever(dense, sparse, **options)
    if kind in {"fallback", "corpus_enrichment"}:
        retriever = dependencies.get("retriever", values.pop("retriever", None))
        discovery = dependencies.get("discovery", values.pop("discovery", None))
        ingestion = dependencies.get(
            "ingestion_pipeline", values.pop("ingestion_pipeline", None)
        )
        processing = dependencies.get(
            "processing_pipeline", values.pop("processing_pipeline", None)
        )
        for dependency, name in (
            (retriever, "retriever"),
            (discovery, "discovery"),
            (ingestion, "ingestion_pipeline"),
            (processing, "processing_pipeline"),
        ):
            _require(dependency, name, kind)
        options = _take(
            values,
            "bm25_index_path",
            "max_discovery_results",
            "min_results",
            "min_score",
            "relevance_gate",
        )
        _reject_unknown(values, kind)
        return CorpusEnrichmentRetriever(
            retriever,
            discovery=discovery,
            ingestion_pipeline=ingestion,
            processing_pipeline=processing,
            **options,
        )
    raise ValueError(f"unsupported retriever type: {kind!r}")


class RetrieverFactory:
    """Namespace-style factory for dependency-injection containers."""

    @staticmethod
    def create(config: Mapping[str, Any], **dependencies: Any) -> Any:
        return build_retriever(config, **dependencies)

    build = create


create_retriever = build_retriever


def _take(values: dict[str, Any], *names: str) -> dict[str, Any]:
    return {name: values.pop(name) for name in names if name in values}


def _with_type(config: Any, kind: str) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError(f"{kind} config must be a mapping")
    result = dict(config)
    result.setdefault("type", kind)
    return result


def _require(value: Any, name: str, kind: str) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{kind} retriever requires {name}")


def _reject_unknown(values: Mapping[str, Any], kind: str) -> None:
    if values:
        raise ValueError(
            f"unknown {kind} retriever config keys: {', '.join(sorted(values))}"
        )
