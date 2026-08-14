"""Score the live Part 9c retrieval path on reviewed external questions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from evaluation.full_ragas_evaluation import assemble_evaluation_questions
from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from evaluation.run_full_ragas_eval import build_benchmark_retriever


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions = assemble_evaluation_questions(
        args.manual,
        args.external_dir,
        manual_limit=0,
        reviewed_external_only=True,
    )
    retriever, client = build_benchmark_retriever(args)
    rows: list[dict[str, Any]] = []
    try:
        for index, question in enumerate(questions, 1):
            started = time.perf_counter()
            results = retriever.search(question.question, top_k=args.external_top_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            ranking = [result.chunk_id for result in results]
            relevant = question.reference_context_ids
            rows.append(
                {
                    "query_id": question.id,
                    "tier": question.source_dataset,
                    "question": question.question,
                    "relevant_count": len(relevant),
                    "retrieved_chunk_ids": ranking,
                    "recall@4": recall_at_k(ranking, relevant, 4),
                    "precision@4": precision_at_k(ranking, relevant, 4),
                    "hit@4": float(bool(set(ranking[:4]).intersection(relevant))),
                    "mrr": reciprocal_rank(ranking, relevant),
                    "ndcg@4": ndcg_at_k(ranking, relevant, 4),
                    "latency_ms": latency_ms,
                }
            )
            print(f"[{index:02d}/{len(questions)}] {question.id} {latency_ms:.1f} ms", flush=True)
    finally:
        if client is not None:
            client.close()
    return {
        "schema_version": 1,
        "questions": len(rows),
        "retrieval": {
            "config": args.retrieval_config,
            "candidate_k": args.reranker_candidate_k,
            "top_k": args.external_top_k,
            "mmr_enabled": args.retrieval_config == "hybrid_rerank_mmr",
            "mmr_min_top_k": args.mmr_min_top_k,
            "mmr_applied_at_top_k": args.external_top_k >= args.mmr_min_top_k,
        },
        "overall": _aggregate(rows),
        "by_tier": {
            tier: _aggregate([row for row in rows if row["tier"] == tier])
            for tier in sorted({row["tier"] for row in rows})
        },
        "per_question": rows,
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    metrics = ("recall@4", "precision@4", "hit@4", "mrr", "ndcg@4")
    return {
        "questions": len(rows),
        **{name: statistics.fmean(float(row[name]) for row in rows) for name in metrics},
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def save(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_scores.json"
    csv_path = output_dir / "retrieval_scores.csv"
    md_path = output_dir / "retrieval_scores.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    columns = [
        "query_id", "tier", "question", "relevant_count", "recall@4",
        "precision@4", "hit@4", "mrr", "ndcg@4", "latency_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in payload["per_question"])
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Part 9c retrieval scores",
        "",
        "Live local evaluation of 75 human-reviewed QASA/QASPER questions. No LLM or external API was used.",
        "",
        "| Tier | N | Recall@4 | Precision@4 | Hit@4 | MRR | nDCG@4 | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in [("Overall", payload["overall"]), *payload["by_tier"].items()]:
        lines.append(
            f"| {label} | {row['questions']} | {row['recall@4']:.4f} | "
            f"{row['precision@4']:.4f} | {row['hit@4']:.4f} | {row['mrr']:.4f} | "
            f"{row['ndcg@4']:.4f} | {row['p50_latency_ms']:.1f} | {row['p95_latency_ms']:.1f} |"
        )
    failures = [row for row in payload["per_question"] if row["hit@4"] == 0.0]
    lines.extend(["", f"Top-4 misses: {len(failures)} of {payload['questions']} questions.", ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    from evaluation.run_full_ragas_eval import _parser as full_parser

    parser = full_parser()
    parser.set_defaults(
        manual=Path("evaluation/data/golden_generation_qa.json"),
        reviewed_external_only=True,
        retrieval_config="hybrid_rerank_mmr",
    )
    parser.add_argument(
        "--score-output-dir",
        type=Path,
        default=Path("evaluation/data/eval_results/part9c_retrieval_20260813"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = save(run(args), args.score_output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
