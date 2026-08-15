from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.full_ragas_evaluation import assemble_evaluation_questions
from evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from evaluation.run_full_ragas_eval import _parser, build_benchmark_retriever
from processing.bm25_indexer import BM25Indexer


ROOT = Path(__file__).resolve().parent
MIXED = ROOT / "evaluation/data/part9l_mixed_style_retrieval_qa.json"
OUT = ROOT / "evaluation/data/eval_results/part9l_mixed_style_retrieval_20260814"
CONFIGS = ("bm25", "hybrid_rrf", "hybrid_rerank")


def rank(ranking: list[str], labels: list[str]) -> int | None:
    wanted = set(labels)
    return next((i for i, item in enumerate(ranking, 1) if item in wanted), None)


def metrics(ranking: list[str], labels: list[str]) -> dict[str, float]:
    return {
        "recall@4": recall_at_k(ranking, labels, 4),
        "recall@8": recall_at_k(ranking, labels, 8),
        "hit@4": float(rank(ranking[:4], labels) is not None),
        "hit@8": float(rank(ranking[:8], labels) is not None),
        "mrr": reciprocal_rank(ranking[:4], labels),
        "ndcg@4": ndcg_at_k(ranking, labels, 4),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    names = ("recall@4", "recall@8", "hit@4", "hit@8", "mrr", "ndcg@4")
    return {
        "questions": len(rows),
        **{name: statistics.fmean(float(row[name]) for row in rows) for name in names},
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def main() -> None:
    args = _parser().parse_args([])
    args.retrieval_config = "hybrid_rerank"
    args.external_top_k = 20
    args.reranker_candidate_k = 20
    wrapped, client = build_benchmark_retriever(args)
    hybrid = wrapped.retriever
    reranker = wrapped.reranker
    external_index = BM25Indexer.load(args.external_index)
    by_chunk = {str(row["chunk_id"]): row for row in external_index.chunks}
    embedder = hybrid.dense_retriever.embedder

    full = assemble_evaluation_questions(
        args.manual,
        args.external_dir,
        manual_limit=0,
        reviewed_external_only=True,
    )
    mixed = json.loads(MIXED.read_text(encoding="utf-8"))["triplets"]
    query_specs: list[dict[str, object]] = []
    for row in full:
        query_specs.append(
            {
                "query_id": row.id,
                "base_id": row.id,
                "scope": "full75",
                "phrasing_tier": "original",
                "source_dataset": row.source_dataset,
                "question": row.question,
                "relevant_chunk_ids": list(row.reference_context_ids),
            }
        )
    full_by_id = {str(row["query_id"]): row for row in query_specs}
    paired_specs: list[dict[str, object]] = []
    for row in mixed:
        base = str(row["base_id"])
        for tier in ("original", "vague", "topic_named"):
            question = str(row[tier])
            spec = {
                "query_id": f"{base}::{tier}",
                "base_id": base,
                "scope": "paired20",
                "phrasing_tier": tier,
                "source_dataset": row["source_dataset"],
                "question": question,
                "relevant_chunk_ids": list(row["relevant_chunk_ids"]),
            }
            paired_specs.append(spec)
            if tier != "original":
                query_specs.append(spec)
            else:
                assert full_by_id[base]["question"] == question

    traces: dict[str, dict[str, object]] = {}
    try:
        for position, spec in enumerate(query_specs, 1):
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
            print(f"[{position:03d}/{len(query_specs)}] {spec['query_id']}", flush=True)
    finally:
        if client is not None:
            client.close()

    scored: list[dict[str, object]] = []
    for spec in query_specs[: len(full)]:
        trace = traces[str(spec["query_id"])]
        for config in CONFIGS:
            scored.append({**spec, "config": config, **metrics(trace[config], spec["relevant_chunk_ids"])})
    for spec in paired_specs:
        trace_key = str(spec["base_id"]) if spec["phrasing_tier"] == "original" else str(spec["query_id"])
        trace = traces[trace_key]
        for config in CONFIGS:
            scored.append({**spec, "config": config, **metrics(trace[config], spec["relevant_chunk_ids"])})

    full_aggregate = {
        config: aggregate([r for r in scored if r["scope"] == "full75" and r["config"] == config])
        for config in CONFIGS
    }
    paired_aggregate = {
        config: {
            tier: aggregate(
                [
                    r
                    for r in scored
                    if r["scope"] == "paired20"
                    and r["config"] == config
                    and r["phrasing_tier"] == tier
                ]
            )
            for tier in ("original", "vague", "topic_named")
        }
        for config in CONFIGS
    }

    diagnostics: list[dict[str, object]] = []
    for spec in [row for row in paired_specs if row["phrasing_tier"] == "vague"]:
        trace = traces[str(spec["query_id"])]
        labels = list(spec["relevant_chunk_ids"])
        if rank(trace["hybrid_rerank"][:4], labels) is not None:
            continue
        fused_rank = rank(trace["hybrid_rrf"], labels)
        sparse_rank = rank(trace["bm25"], labels)
        dense_rank = rank(trace["dense"], labels)
        rerank_rank = rank(trace["hybrid_rerank"], labels)
        stage = "candidate_generation" if fused_rank is None else "reranker_demotion"
        query_tokens = set(external_index.tokenize(str(spec["question"]), external_index.preprocessing_config))
        gold_tokens: set[str] = set()
        gold_texts: list[str] = []
        for chunk_id in labels:
            chunk = by_chunk[chunk_id]
            text = str(chunk["text"])
            gold_texts.append(text)
            gold_tokens.update(external_index.tokenize(text, external_index.preprocessing_config))
        shared = sorted(query_tokens & gold_tokens)
        positive_query = {
            token
            for token in query_tokens
            if float(external_index.index.idf.get(token, 0.0)) > 0.0
        }
        discriminative_shared = sorted(positive_query & gold_tokens)
        vectors = embedder.encode_texts([str(spec["question"]), *gold_texts])
        cosine_scores = [cosine(vectors[0], vector) for vector in vectors[1:]]
        diagnostics.append(
            {
                "query_id": spec["query_id"],
                "base_id": spec["base_id"],
                "source_dataset": spec["source_dataset"],
                "question": spec["question"],
                "miss_stage": stage,
                "gold_rank_bm25_top50": sparse_rank,
                "gold_rank_dense_top50": dense_rank,
                "gold_rank_fused_top20": fused_rank,
                "gold_rank_post_rerank_top20": rerank_rank,
                "query_tokens": sorted(query_tokens),
                "shared_raw_tokens": shared,
                "shared_raw_token_count": len(shared),
                "shared_positive_idf_tokens": discriminative_shared,
                "shared_positive_idf_token_count": len(discriminative_shared),
                "gold_chunk_cosine_scores": dict(zip(labels, cosine_scores, strict=True)),
                "max_gold_cosine": max(cosine_scores),
                "lexical_gap_bm25_miss": sparse_rank is None,
                "semantic_gap_dense_miss": dense_rank is None,
            }
        )

    miss_count = len(diagnostics)
    stage_counts = Counter(str(row["miss_stage"]) for row in diagnostics)
    candidate_rows = [
        row for row in diagnostics if row["miss_stage"] == "candidate_generation"
    ]
    lexical_count = sum(
        bool(row["lexical_gap_bm25_miss"]) for row in candidate_rows
    )
    semantic_count = sum(
        bool(row["semantic_gap_dense_miss"]) for row in candidate_rows
    )
    zero_raw = sum(int(row["shared_raw_token_count"]) == 0 for row in diagnostics if row["miss_stage"] == "candidate_generation")
    zero_discriminative = sum(int(row["shared_positive_idf_token_count"]) == 0 for row in diagnostics if row["miss_stage"] == "candidate_generation")
    candidate_count = stage_counts.get("candidate_generation", 0)
    breakdown = {
        "vague_questions": 20,
        "vague_top4_misses": miss_count,
        "vague_top4_passes": 20 - miss_count,
        "stage_counts": dict(stage_counts),
        "stage_percent_of_misses": {
            key: value / miss_count if miss_count else 0.0 for key, value in stage_counts.items()
        },
        "overlapping_root_attributes": {
            "lexical_gap_bm25_miss_count": lexical_count,
            "lexical_gap_bm25_miss_percent": lexical_count / miss_count if miss_count else 0.0,
            "semantic_gap_dense_miss_count": semantic_count,
            "semantic_gap_dense_miss_percent": semantic_count / miss_count if miss_count else 0.0,
            "reranker_demotion_count": stage_counts.get("reranker_demotion", 0),
            "reranker_demotion_percent": stage_counts.get("reranker_demotion", 0) / miss_count if miss_count else 0.0,
            "note": "Candidate-generation rows only. Lexical and semantic gap attributes can overlap on the same miss."
        },
        "candidate_generation_lexical_overlap": {
            "candidate_generation_misses": candidate_count,
            "zero_raw_token_overlap": zero_raw,
            "zero_positive_idf_overlap": zero_discriminative,
        },
    }
    payload = {
        "schema_version": 1,
        "created_at": "2026-08-14",
        "retrieval": {
            "backend_candidate_k": 50,
            "fused_candidate_k": 20,
            "reranked_candidate_k": 20,
            "scored_cutoffs": [4, 8],
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "bm25_tokenizer": external_index.preprocessing_config,
        },
        "coverage": {"full_original": len(full), "paired_evidence_needs": len(mixed), "paired_queries": len(paired_specs)},
        "full75_aggregate": full_aggregate,
        "paired20_aggregate": paired_aggregate,
        "root_cause_breakdown": breakdown,
        "vague_miss_diagnostics": diagnostics,
        "per_question_scores": scored,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "per_question_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["query_id", "base_id", "scope", "phrasing_tier", "source_dataset", "question", "config", "recall@4", "recall@8", "hit@4", "hit@8", "mrr", "ndcg@4"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    with (OUT / "vague_miss_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["query_id", "base_id", "source_dataset", "question", "miss_stage", "gold_rank_bm25_top50", "gold_rank_dense_top50", "gold_rank_fused_top20", "gold_rank_post_rerank_top20", "shared_raw_token_count", "shared_positive_idf_token_count", "max_gold_cosine", "lexical_gap_bm25_miss", "semantic_gap_dense_miss"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(diagnostics)
    print(json.dumps({"output": str(OUT), "full75": full_aggregate, "paired20": paired_aggregate, "breakdown": breakdown}, indent=2))


if __name__ == "__main__":
    main()
