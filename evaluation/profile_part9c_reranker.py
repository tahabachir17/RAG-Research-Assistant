"""Profile Part 9c reranking candidate sizes from frozen Part 9b candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from retrieval import CrossEncoderReranker, RetrievalResult

try:
    from .retrieval_evaluator import _local_model_reference, ranked_metrics
except ImportError:
    from retrieval_evaluator import _local_model_reference, ranked_metrics


def profile(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    benchmark = _gold(args.external_dir)
    hybrid_rows = diagnostic["ablation"]["rankings"]["hybrid_rrf"]
    model = _local_model_reference(args.model, args.model_cache)
    reranker = CrossEncoderReranker(
        model,
        default_top_k=args.top_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    configurations: dict[str, Any] = {}
    for candidate_k in args.candidate_k:
        latencies: list[float] = []
        recalls: list[float] = []
        per_query: list[dict[str, Any]] = []
        for row in hybrid_rows:
            question = benchmark[row["query_id"]]
            candidates = [
                RetrievalResult.from_payload(item, score=item["score"], source=item["source"])
                for item in row["results"][:candidate_k]
            ]
            started = time.perf_counter()
            reranked = reranker.rerank(
                question["question"], candidates, top_k=args.top_k
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            recall = ranked_metrics(
                [item.chunk_id for item in reranked],
                question["reference_context_ids"],
                ks=(args.top_k,),
            )[f"recall@{args.top_k}"]
            latencies.append(elapsed)
            recalls.append(recall)
            per_query.append(
                {"query_id": row["query_id"], "latency_ms": elapsed, "recall": recall}
            )
        configurations[str(candidate_k)] = {
            "questions": len(per_query),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "mean_latency_ms": statistics.fmean(latencies),
            f"recall@{args.top_k}": statistics.fmean(recalls),
            "per_query": per_query,
        }
    return {
        "model": str(model),
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "configurations": configurations,
    }


def _gold(directory: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for tier in ("qasa", "qasper"):
        payload = json.loads(
            (directory / f"{tier}_generation_qa.json").read_text(encoding="utf-8")
        )
        for row in payload["questions"]:
            if row.get("reviewed") and row.get("reference_context_ids"):
                rows[row["id"]] = row
    return rows


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=Path("evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/retrieval_diagnostic.json"),
    )
    parser.add_argument(
        "--external-dir", type=Path, default=Path("evaluation/data/external_benchmarks")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/data/eval_results/part9c_20260813/reranker_profile.json"),
    )
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "per_query"} for key, value in result["configurations"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
