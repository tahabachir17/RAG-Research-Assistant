"""Part 9b benchmark retrieval ablation and failure-correlation report."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from processing.bm25_indexer import BM25Indexer
from processing.embedder import Embedder
from processing.qdrant_indexer import QdrantIndexer

try:
    from .retrieval_evaluator import (
        CONFIGS,
        RetrievalEvaluator,
        _local_model_reference,
        build_local_evaluator,
    )
except ImportError:
    from retrieval_evaluator import (
        CONFIGS,
        RetrievalEvaluator,
        _local_model_reference,
        build_local_evaluator,
    )


LOGGER = logging.getLogger(__name__)
BENCHMARK_COLLECTION_PREFIX = "bench_"
DEFAULT_COLLECTION = "bench_external_chunks"
NAMED_FAILURES = (
    "qasa-1409.0575-1",
    "qasa-1907.11692-16",
    "qasper-32a232310babb92991c4b1b75f7aa6b4670ec447",
    "qasper-bf00808353eec22b4801c922cce7b1ec0ff3b777",
    "qasper-bc473c5bd0e1a8be9b2037aa7006fd68217c3f47",
)


def load_reviewed_benchmark_golden(
    external_dir: str | Path,
) -> list[dict[str, Any]]:
    """Use only reviewed QASA/QASPER evidence IDs as chunk relevance labels."""

    directory = Path(external_dir)
    rows: list[dict[str, Any]] = []
    for tier in ("qasa", "qasper"):
        payload = json.loads(
            (directory / f"{tier}_generation_qa.json").read_text(encoding="utf-8")
        )
        for item in payload.get("questions", []):
            relevant = [str(value) for value in item.get("reference_context_ids", [])]
            if not item.get("reviewed") or not relevant:
                continue
            rows.append(
                {
                    "query_id": str(item["id"]),
                    "question": str(item["question"]),
                    "source_tier": tier,
                    "relevant_chunk_ids": relevant,
                    "relevant_paper_ids": [],
                }
            )
    identifiers = [row["query_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("reviewed benchmark question IDs must be globally unique")
    return rows


def ensure_benchmark_dense_collection(
    bm25_path: str | Path,
    qdrant_path: str | Path,
    collection: str,
    *,
    embedding_model: str,
    model_cache: str | Path,
) -> dict[str, Any]:
    """Create an isolated dense collection with production-compatible payloads."""

    from qdrant_client import QdrantClient

    if not collection.startswith(BENCHMARK_COLLECTION_PREFIX):
        raise ValueError(
            "benchmark collection must start with "
            f"{BENCHMARK_COLLECTION_PREFIX!r}; got {collection!r}"
        )
    sparse_index = BM25Indexer.load(bm25_path)
    expected = len(sparse_index.chunks)
    destination = Path(qdrant_path)
    destination.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(destination))
    try:
        if client.collection_exists(collection):
            info = client.get_collection(collection)
            count = int(getattr(info, "points_count", 0) or 0)
            if count == expected:
                return {
                    "status": "reused",
                    "collection": collection,
                    "qdrant_path": str(destination),
                    "points": count,
                    "vector_size": 384,
                }
            raise ValueError(
                f"existing benchmark collection has {count} points; expected {expected}"
            )
        reference = _local_model_reference(embedding_model, Path(model_cache))
        embedder = Embedder(reference)
        texts = [str(chunk.get("text", "")) for chunk in sparse_index.chunks]
        vectors = embedder.encode_texts(texts)
        records = []
        for chunk, vector in zip(sparse_index.chunks, vectors, strict=True):
            record = dict(chunk)
            metadata = dict(record.get("metadata") or {})
            metadata["source"] = "benchmark"
            record["metadata"] = metadata
            record["embedding"] = vector.tolist()
            records.append(record)
        indexed = QdrantIndexer(
            collection_name=collection, client=client
        ).index_embeddings(records)
        return {
            "status": "created",
            "collection": collection,
            "qdrant_path": str(destination),
            "points": indexed,
            "vector_size": int(vectors.shape[1]),
        }
    finally:
        client.close()


def actual_smoke_inventory(
    *,
    production_index: str | Path,
    external_index: str | Path,
    external_top_k: int,
) -> list[dict[str, Any]]:
    """Describe the actual calls made by run_full_ragas_eval, not intended design."""

    return [
        {
            "tier": "manual",
            "retriever_class": "none (frozen IDs via TieredChunkLookup)",
            "indexes": str(production_index),
            "top_k": "frozen 4 IDs/question",
            "rerank": False,
            "mmr": False,
        },
        *[
            {
                "tier": tier,
                "retriever_class": "BM25Indexer.search",
                "indexes": str(external_index),
                "top_k": external_top_k,
                "rerank": False,
                "mmr": False,
            }
            for tier in ("qasa", "qasper")
        ],
    ]


def correlate_failures(
    evaluation: dict[str, Any],
    golden: Sequence[dict[str, Any]],
    index: BM25Indexer,
    *,
    query_ids: Sequence[str] = NAMED_FAILURES,
) -> list[dict[str, Any]]:
    """Show which configurations surfaced reviewed evidence and at what rank."""

    golden_by_id = {str(row["query_id"]): row for row in golden}
    chunk_by_id = {str(row.get("chunk_id")): row for row in index.chunks}
    runs = evaluation["rankings"]
    output: list[dict[str, Any]] = []
    for query_id in query_ids:
        golden_row = golden_by_id.get(query_id)
        if golden_row is None:
            output.append(
                {
                    "query_id": query_id,
                    "status": "unreviewed",
                    "note": "No reviewed evidence IDs; retrieval correlation is unavailable.",
                }
            )
            continue
        ranks: dict[str, list[int]] = {}
        for config in CONFIGS:
            run = next(
                row for row in runs[config] if str(row["query_id"]) == query_id
            )
            ordered = [str(item["chunk_id"]) for item in run["results"]]
            ranks[config] = [
                ordered.index(chunk_id) + 1
                for chunk_id in golden_row["relevant_chunk_ids"]
                if chunk_id in ordered
            ]
        evidence = [
            {
                "chunk_id": chunk_id,
                "section": str(chunk_by_id.get(chunk_id, {}).get("section", "")),
                "text": str(chunk_by_id.get(chunk_id, {}).get("text", "")),
            }
            for chunk_id in golden_row["relevant_chunk_ids"]
        ]
        output.append(
            {
                "query_id": query_id,
                "status": "reviewed",
                "question": golden_row["question"],
                "ranks": ranks,
                "configs_with_top4_hit": [
                    config for config, values in ranks.items() if any(rank <= 4 for rank in values)
                ],
                "evidence": evidence,
            }
        )
    return output


def run_retrieval_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    dense_index = ensure_benchmark_dense_collection(
        args.external_bm25_index,
        args.benchmark_qdrant_path,
        args.collection,
        embedding_model=args.embedding_model,
        model_cache=args.model_cache,
    )
    golden = load_reviewed_benchmark_golden(args.external_dir)
    evaluator: RetrievalEvaluator = build_local_evaluator(args)
    result = evaluator.evaluate(golden)
    external_index = BM25Indexer.load(args.external_bm25_index)
    prior = json.loads(Path(args.part7_report).read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "Select a retrieval configuration for benchmark-tier generation evaluation.",
        "inventory": actual_smoke_inventory(
            production_index=args.production_index,
            external_index=args.external_bm25_index,
            external_top_k=args.smoke_top_k,
        ),
        "dense_index": dense_index,
        "benchmark": {
            "reviewed_questions": len(golden),
            "qasa": sum(row["source_tier"] == "qasa" for row in golden),
            "qasper": sum(row["source_tier"] == "qasper" for row in golden),
            "label_grain": "reviewed chunk evidence",
        },
        "ablation": result,
        "part7": {
            "aggregate": prior["aggregate"],
            "label_coverage": prior["label_coverage"],
        },
        "failure_correlation": correlate_failures(result, golden, external_index),
        "judge_reliability": {
            "status": "pending",
            "note": "Run the networked judge reliability companion and merge its JSON.",
        },
    }


def save_diagnostic(payload: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "retrieval_diagnostic.json"
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=Path("evaluation/data/external_benchmarks"))
    parser.add_argument("--external-bm25-index", type=Path, default=Path("evaluation/data/external_benchmarks/external_bm25_index.pkl"))
    parser.add_argument("--production-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--benchmark-qdrant-path", type=Path, default=Path("evaluation/data/external_benchmarks/qdrant"))
    parser.add_argument("--qdrant-path", type=Path, default=Path("evaluation/data/external_benchmarks/qdrant"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--bm25-index", type=Path, default=Path("evaluation/data/external_benchmarks/external_bm25_index.pkl"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--mmr-lambda", type=float, default=0.5)
    parser.add_argument("--lift-threshold", type=float, default=0.02)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-timeout", type=float, default=30.0)
    parser.add_argument("--smoke-top-k", type=int, default=4)
    parser.add_argument("--part7-report", type=Path, default=Path("evaluation/data/eval_results/retrieval_eval_20260803T181316Z.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/data/eval_results/retrieval_stack_diagnostic_20260812"))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    path = save_diagnostic(run_retrieval_diagnostic(args), args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
