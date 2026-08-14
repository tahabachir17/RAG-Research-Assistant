"""Run the offline Part 9e retrieval-stage and configuration ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from .full_ragas_evaluation import assemble_evaluation_questions
    from .metrics import ranked_metrics
    from .run_full_ragas_eval import build_benchmark_retriever
    from .run_part9c_retrieval_scores import _parser as part9c_parser
except ImportError:
    from full_ragas_evaluation import assemble_evaluation_questions
    from metrics import ranked_metrics
    from run_full_ragas_eval import build_benchmark_retriever
    from run_part9c_retrieval_scores import _parser as part9c_parser


CONFIGS = ("bm25", "hybrid_rrf", "hybrid_rerank")
KS = (4, 8)


def best_gold_rank(ranking: Sequence[str], gold_ids: Sequence[str]) -> int | None:
    """Return the best one-based rank of any gold ID, or ``None``."""

    labels = {str(item).strip() for item in gold_ids if str(item).strip()}
    return next(
        (rank for rank, item in enumerate(ranking, 1) if str(item).strip() in labels),
        None,
    )


def display_rank(rank: int | None, *, cutoff: int = 20) -> str:
    """Format a rank under the audit's explicit cutoff vocabulary."""

    return str(rank) if rank is not None and rank <= cutoff else f"not in top-{cutoff}"


@dataclass(slots=True)
class StageTrace:
    question_id: str
    tier: str
    gold_chunk_rank_bm25: str
    gold_chunk_rank_dense: str
    gold_chunk_rank_hybrid_rrf: str
    gold_chunk_rank_post_rerank: str
    miss_stage: str


class Part9eAblation:
    """Trace injected Part 9c retrieval components once per reviewed question."""

    def __init__(self, hybrid: Any, reranker: Any, *, backend_k: int = 50) -> None:
        self.hybrid, self.reranker, self.backend_k = hybrid, reranker, backend_k

    def run(self, questions: Sequence[Any]) -> dict[str, Any]:
        score_rows: list[dict[str, Any]] = []
        stage_rows: list[StageTrace] = []
        for position, question in enumerate(questions, 1):
            started = time.perf_counter()
            dense = self.hybrid.dense_retriever.search(
                question.question, top_k=self.backend_k
            )
            dense_ms = _elapsed_ms(started)

            started = time.perf_counter()
            sparse = self.hybrid.sparse_retriever.search(
                question.question, top_k=self.backend_k
            )
            sparse_ms = _elapsed_ms(started)

            started = time.perf_counter()
            fused = self.hybrid.fuse(dense, sparse, top_k=20)
            fusion_ms = _elapsed_ms(started)

            started = time.perf_counter()
            reranked = self.reranker.rerank(question.question, fused, top_k=20)
            rerank_ms = _elapsed_ms(started)

            rankings = {
                "bm25": [row.chunk_id for row in sparse],
                "hybrid_rrf": [row.chunk_id for row in fused],
                "hybrid_rerank": [row.chunk_id for row in reranked],
            }
            latencies = {
                "bm25": sparse_ms,
                "hybrid_rrf": dense_ms + sparse_ms + fusion_ms,
                "hybrid_rerank": dense_ms + sparse_ms + fusion_ms + rerank_ms,
            }
            for config in CONFIGS:
                for k in KS:
                    ranking = rankings[config][:k]
                    score_rows.append(
                        {
                            "question_id": question.id,
                            "tier": question.source_dataset,
                            "config": config,
                            "k": k,
                            "latency_ms": latencies[config],
                            **ranked_metrics(
                                ranking, question.reference_context_ids, ks=(k,)
                            ),
                        }
                    )

            if (
                question.source_dataset == "qasper"
                and best_gold_rank(rankings["hybrid_rerank"][:4], question.reference_context_ids)
                is None
            ):
                hybrid_rank = best_gold_rank(
                    rankings["hybrid_rrf"], question.reference_context_ids
                )
                stage_rows.append(
                    StageTrace(
                        question.id,
                        question.source_dataset,
                        display_rank(
                            best_gold_rank(
                                [row.chunk_id for row in sparse],
                                question.reference_context_ids,
                            )
                        ),
                        display_rank(
                            best_gold_rank(
                                [row.chunk_id for row in dense],
                                question.reference_context_ids,
                            )
                        ),
                        display_rank(hybrid_rank),
                        display_rank(
                            best_gold_rank(
                                rankings["hybrid_rerank"],
                                question.reference_context_ids,
                            )
                        ),
                        "candidate_generation"
                        if hybrid_rank is None
                        else "reranker_demotion",
                    )
                )
            print(f"[{position:02d}/{len(questions)}] {question.id}", flush=True)
        return {
            "schema_version": 1,
            "questions": len(questions),
            "backend_candidate_k": self.backend_k,
            "fused_candidate_k": 20,
            "per_question": score_rows,
            "aggregate": _aggregate(score_rows),
            "miss_stage_isolation": [asdict(row) for row in stage_rows],
        }


def _aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in CONFIGS:
        for k in KS:
            selected = [row for row in rows if row["config"] == config and row["k"] == k]
            for tier in ("overall", "qasa", "qasper"):
                tier_rows = (
                    selected if tier == "overall" else [row for row in selected if row["tier"] == tier]
                )
                latencies = [float(row["latency_ms"]) for row in tier_rows]
                output.append(
                    {
                        "config": config,
                        "k": k,
                        "tier": tier,
                        "questions": len(tier_rows),
                        **{
                            name: statistics.fmean(float(row[f"{name}@{k}"]) for row in tier_rows)
                            for name in ("recall", "precision", "hit", "ndcg")
                        },
                        "mrr": statistics.fmean(float(row["mrr"]) for row in tier_rows),
                        "p50_latency_ms": statistics.median(latencies),
                        "p95_latency_ms": _percentile(latencies, 0.95),
                    }
                )
    return output


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def write_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "config_ablation_scores.json"
    csv_path = output_dir / "config_ablation_scores.csv"
    isolation_path = output_dir / "miss_stage_isolation.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    score_fields = [
        "question_id", "tier", "config", "k", "latency_ms", "recall@4",
        "precision@4", "hit@4", "ndcg@4", "recall@8", "precision@8",
        "hit@8", "ndcg@8", "mrr",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["per_question"])
    isolation_fields = [
        "question_id", "tier", "gold_chunk_rank_bm25", "gold_chunk_rank_dense",
        "gold_chunk_rank_hybrid_rrf", "gold_chunk_rank_post_rerank", "miss_stage",
    ]
    with isolation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=isolation_fields)
        writer.writeheader()
        writer.writerows(payload["miss_stage_isolation"])
    return {"json": json_path, "csv": csv_path, "isolation": isolation_path}


def _parser() -> argparse.ArgumentParser:
    parser = part9c_parser()
    parser.set_defaults(retrieval_config="hybrid_rerank")
    parser.add_argument(
        "--ablation-output-dir",
        type=Path,
        default=Path("evaluation/data/eval_results/part9e_ablation_20260813"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    questions = assemble_evaluation_questions(
        args.manual, args.external_dir, manual_limit=0, reviewed_external_only=True
    )
    wrapped, client = build_benchmark_retriever(args)
    try:
        payload = Part9eAblation(
            wrapped.retriever, wrapped.reranker, backend_k=50
        ).run(questions)
    finally:
        if client is not None:
            client.close()
    print(json.dumps({key: str(value) for key, value in write_outputs(payload, args.ablation_output_dir).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
