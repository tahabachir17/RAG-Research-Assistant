"""Run the additive Part 9m 30-query retrieval-only re-test."""

from __future__ import annotations

import csv
import json
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Any

from evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from evaluation.run_full_ragas_eval import _parser, build_benchmark_retriever


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/data/part9l_mixed_style_retrieval_qa.json"
OUT = ROOT / "evaluation/data/eval_results/part9m_reduced_retrieval_retest_20260815"
CONFIGS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")


def _rank(ranking: list[str], labels: list[str]) -> int | None:
    wanted = set(labels)
    return next((index for index, chunk_id in enumerate(ranking, 1) if chunk_id in wanted), None)


def _metrics(ranking: list[str], labels: list[str]) -> dict[str, float]:
    return {
        "recall@4": recall_at_k(ranking, labels, 4),
        "recall@8": recall_at_k(ranking, labels, 8),
        "hit@4": float(_rank(ranking[:4], labels) is not None),
        "hit@8": float(_rank(ranking[:8], labels) is not None),
        "mrr": reciprocal_rank(ranking[:4], labels),
        "ndcg@4": ndcg_at_k(ranking, labels, 4),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("recall@4", "recall@8", "hit@4", "hit@8", "mrr", "ndcg@4")
    return {"questions": len(rows), **{name: statistics.fmean(float(row[name]) for row in rows) for name in names}}


def _selection() -> list[dict[str, Any]]:
    triplets = json.loads(SOURCE.read_text(encoding="utf-8"))["triplets"]
    specs: list[dict[str, Any]] = []
    for row in triplets:
        specs.append(
            {
                "query_id": f"{row['base_id']}::vague",
                "base_id": row["base_id"],
                "phrasing_tier": "vague",
                "source_dataset": row["source_dataset"],
                "question": row["vague"],
                "relevant_chunk_ids": row["relevant_chunk_ids"],
            }
        )
    for row in triplets[:10]:
        specs.append(
            {
                "query_id": f"{row['base_id']}::topic_named",
                "base_id": row["base_id"],
                "phrasing_tier": "topic_named",
                "source_dataset": row["source_dataset"],
                "question": row["topic_named"],
                "relevant_chunk_ids": row["relevant_chunk_ids"],
            }
        )
    return specs


def main() -> None:
    args = _parser().parse_args([])
    args.retrieval_config = "hybrid_rerank"
    args.external_top_k = 20
    args.reranker_candidate_k = 20
    wrapped, client = build_benchmark_retriever(args)
    hybrid = wrapped.retriever
    reranker = wrapped.reranker
    specs = _selection()
    traces: dict[str, dict[str, Any]] = {}
    try:
        for position, spec in enumerate(specs, 1):
            question = str(spec["question"])
            started = time.perf_counter()
            dense = hybrid.dense_retriever.search(question, top_k=50)
            sparse = hybrid.sparse_retriever.search(question, top_k=50)
            fused = hybrid.fuse(dense, sparse, top_k=20)
            reranked = reranker.rerank(question, fused, top_k=20)
            traces[str(spec["query_id"])] = {
                "dense": [row.chunk_id for row in dense],
                "bm25": [row.chunk_id for row in sparse],
                "hybrid_rrf": [row.chunk_id for row in fused],
                "hybrid_rerank": [row.chunk_id for row in reranked],
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
            print(f"[{position:02d}/30] {spec['query_id']}", flush=True)
    finally:
        if client is not None:
            client.close()

    scored: list[dict[str, Any]] = []
    for spec in specs:
        trace = traces[str(spec["query_id"])]
        for config in CONFIGS:
            scored.append({**spec, "config": config, **_metrics(trace[config], spec["relevant_chunk_ids"])})
    aggregate = {
        config: {
            tier: _aggregate([row for row in scored if row["config"] == config and row["phrasing_tier"] == tier])
            for tier in ("vague", "topic_named")
        }
        for config in CONFIGS
    }
    payload = {
        "schema_version": 1,
        "created_at": date.today().isoformat(),
        "selection_rule": "all 20 vague phrasings plus topic-named phrasings for the first 10 triplets in unchanged Part 9l file order",
        "retrieval": {
            "backend_candidate_k": 50,
            "fused_candidate_k": 20,
            "reranked_candidate_k": 20,
            "scored_cutoffs": [4, 8],
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
        },
        "coverage": {"queries": len(specs), "vague": 20, "topic_named": 10},
        "aggregate": aggregate,
        "query_manifest": specs,
        "retrieval_traces": traces,
        "per_question_scores": scored,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval_retest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "per_question_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["query_id", "base_id", "phrasing_tier", "source_dataset", "question", "config", "recall@4", "recall@8", "hit@4", "hit@8", "mrr", "ndcg@4"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    print(json.dumps({"output": str(OUT), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
