"""Retrieval wrapper that enriches the corpus when static search is insufficient."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from .models import RetrievalResult


class CorpusEnrichmentRetriever:
    """Discover, ingest, index, and retry after an insufficient static search.

    relevance_gate returns True when results are sufficient. When omitted,
    at least min_results are required and min_score is optionally applied
    to the best result. For hybrid RRF scores, callers should normally supply a
    calibrated gate based on reranker or dense evidence.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        discovery: Any,
        ingestion_pipeline: Any,
        processing_pipeline: Any,
        bm25_index_path: str | Path | None = None,
        max_discovery_results: int = 5,
        min_results: int = 1,
        min_score: float | None = None,
        relevance_gate: Callable[[str, list[RetrievalResult]], bool] | None = None,
    ) -> None:
        for name, dependency, method in (
            ("retriever", retriever, "search"),
            ("discovery", discovery, "search"),
            ("ingestion_pipeline", ingestion_pipeline, "run"),
            ("processing_pipeline", processing_pipeline, "process_paths"),
        ):
            if not callable(getattr(dependency, method, None)):
                raise TypeError(f"{name} must provide {method}()")
        if max_discovery_results < 1 or min_results < 1:
            raise ValueError("result limits must be positive")
        self.retriever = retriever
        self.discovery = discovery
        self.ingestion_pipeline = ingestion_pipeline
        self.processing_pipeline = processing_pipeline
        self.bm25_index_path = Path(bm25_index_path) if bm25_index_path else None
        self.max_discovery_results = max_discovery_results
        self.min_results = min_results
        self.min_score = min_score
        self.relevance_gate = relevance_gate
        self.last_enrichment: dict[str, Any] | None = None

    def search(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        initial = self.retriever.search(query, **kwargs)
        if self._is_sufficient(query, initial):
            self.last_enrichment = None
            return initial

        papers = self.discovery.search(
            query=query, max_results=self.max_discovery_results
        )
        if not papers:
            self.last_enrichment = {"discovered": 0, "reason": "no papers discovered"}
            return initial

        ingestion = self.ingestion_pipeline.run(
            query,
            max_results=len(papers),
            selected_papers=list(papers),
        )
        paths = self._processed_paths()
        if not paths:
            self.last_enrichment = {
                "discovered": len(papers),
                "processed": ingestion.processed,
                "reason": "no processed documents available",
            }
            return initial

        indexing = self.processing_pipeline.process_paths(
            paths, index_dense=True, recreate_qdrant=False
        )
        if self.bm25_index_path is not None:
            self.processing_pipeline.bm25_indexer.save(self.bm25_index_path)
            _reload_sparse(self.retriever)

        for paper in papers:
            record = self.ingestion_pipeline.registry.get(paper.paper_id)
            if record and record["status"] == "processed":
                self.ingestion_pipeline.registry.mark(paper.paper_id, "indexed")

        self.last_enrichment = {
            "discovered": len(papers),
            "processed": ingestion.processed,
            "indexed_paths": len(paths),
            "indexing": indexing,
        }
        return self.retriever.search(query, **kwargs)

    def _is_sufficient(self, query: str, results: list[RetrievalResult]) -> bool:
        if self.relevance_gate is not None:
            return bool(self.relevance_gate(query, results))
        if len(results) < self.min_results:
            return False
        return (
            self.min_score is None
            or max(result.score for result in results) >= self.min_score
        )

    def _processed_paths(self) -> list[Path]:
        records = self.ingestion_pipeline.registry.records()
        paths = {
            Path(record["processed_path"])
            for record in records
            if record["status"] in {"processed", "indexed"}
            and record.get("processed_path")
            and Path(record["processed_path"]).is_file()
        }
        return sorted(paths)


def _reload_sparse(retriever: Any) -> None:
    """Reload sparse indexes held directly or by a hybrid retriever."""

    candidates: Iterable[Any] = (
        retriever,
        getattr(retriever, "sparse_retriever", None),
    )
    for candidate in candidates:
        load = getattr(candidate, "load", None)
        if callable(load):
            load()
