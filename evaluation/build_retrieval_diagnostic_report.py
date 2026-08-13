"""Merge Part 9b diagnostics and build a validated-report artifact payload."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .metrics import recall_at_k
from .retrieval_stack_diagnostic import CONFIGS, load_reviewed_benchmark_golden


DISPLAY_NAMES = {
    "dense": "Dense",
    "sparse": "Sparse (BM25)",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_rerank": "Hybrid + rerank",
    "hybrid_rerank_mmr": "Hybrid + rerank + MMR",
}


def build_report(
    diagnostic_path: Path, judge_path: Path, external_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    diagnostic["judge_reliability"] = judge
    golden = load_reviewed_benchmark_golden(external_dir)
    labels = {row["query_id"]: set(row["relevant_chunk_ids"]) for row in golden}
    ablation = _ablation_rows(diagnostic, labels)
    recommendation = _recommend(ablation)
    failure_rows = _failure_rows(diagnostic["failure_correlation"])
    judge_rows = _judge_rows(judge)
    inventory_rows = [
        {
            **row,
            "rerank": "yes" if row["rerank"] else "no",
            "mmr": "yes" if row["mmr"] else "no",
        }
        for row in diagnostic["inventory"]
    ]
    chart_rows = [
        {
            **row,
            "cutoff": f"R@{cutoff}",
            "recall": row[f"recall@{cutoff}"],
        }
        for row in ablation
        for cutoff in (5, 8, 20)
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = [{
        "reviewed_questions": diagnostic["benchmark"]["reviewed_questions"],
        "best_recall8": recommendation["recall@8"],
        "remaining_judge_anomalies": judge["after_anomalies"],
        "recommended_config": recommendation["display_name"],
    }]
    sources = _sources(generated_at)
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Part 9b — Retrieval-Stack Diagnostic",
            "description": "Reviewed benchmark retrieval ablation and judge reliability audit.",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": _cards(),
            "charts": [_recall_chart()],
            "tables": _tables(),
            "blocks": _blocks(diagnostic, judge, recommendation),
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": summary,
                "inventory": inventory_rows,
                "ablation": ablation,
                "recall_chart": chart_rows,
                "failures": failure_rows,
                "judge": judge_rows,
            },
        },
        "sources": sources,
    }
    return diagnostic, artifact


def _ablation_rows(
    diagnostic: dict[str, Any], labels: dict[str, set[str]]
) -> list[dict[str, Any]]:
    aggregate = diagnostic["ablation"]["aggregate"]
    prior = diagnostic["part7"]["aggregate"]
    rows = []
    for config in CONFIGS:
        run_rows = diagnostic["ablation"]["rankings"][config]
        recall4 = statistics.fmean(
            recall_at_k(
                [str(item["chunk_id"]) for item in run["results"]],
                labels[str(run["query_id"])],
                4,
            )
            for run in run_rows
        )
        current = aggregate[config]
        rows.append(
            {
                "config": config,
                "display_name": DISPLAY_NAMES[config],
                "recall@4": recall4,
                "recall@5": current["recall@5"],
                "recall@8": current["recall@8"],
                "recall@20": current["recall@20"],
                "precision@5": current["precision@5"],
                "precision@8": current["precision@8"],
                "precision@20": current["precision@20"],
                "mrr": current["mrr"],
                "ndcg@20": current["ndcg@20"],
                "avg_latency_ms": current["avg_latency_ms"],
                "part7_recall@8": prior[config]["recall@8"],
                "part7_delta_recall@8": current["recall@8"]
                - prior[config]["recall@8"],
                "reviewed_questions": diagnostic["benchmark"]["reviewed_questions"],
                "label_grain": diagnostic["benchmark"]["label_grain"],
            }
        )
    return rows


def _recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["recall@4"],
            row["recall@8"],
            row["mrr"],
            -row["avg_latency_ms"],
        ),
    )


def _failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        ranks = row.get("ranks", {})
        evidence = row.get("evidence", [])
        output.append(
            {
                "query_id": row["query_id"],
                "status": row["status"],
                "dense_ranks": _rank_text(ranks.get("dense", [])),
                "sparse_ranks": _rank_text(ranks.get("sparse", [])),
                "hybrid_ranks": _rank_text(ranks.get("hybrid_rrf", [])),
                "rerank_ranks": _rank_text(ranks.get("hybrid_rerank", [])),
                "mmr_ranks": _rank_text(ranks.get("hybrid_rerank_mmr", [])),
                "top4_configs": ", ".join(
                    DISPLAY_NAMES[value] for value in row.get("configs_with_top4_hit", [])
                ) or "none",
                "evidence_excerpt": " | ".join(
                    str(item.get("text", "")).replace("\n", " ")[:240]
                    for item in evidence[:2]
                ) or row.get("note", ""),
            }
        )
    return output


def _judge_rows(judge: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in judge["cases"]:
        stages = {stage["stage"]: stage for stage in row["stages"]}
        output.append(
            {
                "question_id": row["question_id"],
                "metric": row["metric"],
                "kind": row["kind"],
                "before_score": row.get("before_score"),
                "after_score": row.get("after_score"),
                "fixed": "yes" if row["fixed"] else "no",
                "baseline_status": stages.get("baseline_capture", {}).get("status", "not run"),
                "json_retry_status": stages.get("strict_json_retry", {}).get("status", "not run"),
                "fallback_status": stages.get("fallback_70b", {}).get("status", "not needed"),
                "raw_outputs": sum(len(stage.get("raw_outputs", [])) for stage in row["stages"]),
            }
        )
    return output


def _rank_text(values: list[int]) -> str:
    return ", ".join(str(value) for value in values) or "not in top 20"


def _sources(generated_at: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "retrieval_source",
            "label": "Part 9b retrieval diagnostic",
            "path": "evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/retrieval_diagnostic.json",
            "query": {
                "engine": "Python",
                "language": "python",
                "sql": "SELECT * FROM read_json_auto('evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/retrieval_diagnostic.json')",
                "executed_at": generated_at,
                "description": "RetrievalEvaluator ablation over reviewed QASA/QASPER chunk labels.",
                "tables_used": [
                    "evaluation/data/external_benchmarks/qasa_generation_qa.json",
                    "evaluation/data/external_benchmarks/qasper_generation_qa.json",
                    "evaluation/data/external_benchmarks/external_bm25_index.pkl",
                    "evaluation/data/external_benchmarks/qdrant/bench_external_chunks",
                ],
                "filters": [
                    "reviewed=true",
                    "reference_context_ids is non-empty",
                    "QASA and QASPER tiers only",
                ],
                "metric_definitions": [
                    "Recall@k is the fraction of reviewed relevant chunk IDs present in the first k results, averaged over questions.",
                    "MRR uses the reciprocal rank of the first reviewed relevant chunk.",
                    "nDCG@20 uses binary reviewed chunk relevance with ideal ranking normalization.",
                    "Latency is measured end-to-end per configuration on the local CPU run and is directional, not a production SLA.",
                ],
            },
        },
        {
            "id": "smoke_source",
            "label": "Part 9 smoke evaluation",
            "path": "evaluation/data/eval_results/full_ragas_eval_smoke_5x3_20260812/report.json",
            "query": {
                "engine": "Python",
                "language": "python",
                "sql": "SELECT * FROM read_json_auto('evaluation/data/eval_results/full_ragas_eval_smoke_5x3_20260812/report.json')",
                "executed_at": generated_at,
                "description": "Actual Part 9 smoke-tier retrieval calls and generation anomalies.",
                "tables_used": ["evaluation/run_full_ragas_eval.py"],
            },
        },
        {
            "id": "judge_source",
            "label": "Judge reliability diagnostic",
            "path": "evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/judge_reliability.json",
            "query": {
                "engine": "RAGAS + Groq",
                "language": "python",
                "sql": "SELECT * FROM read_json_auto('evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/judge_reliability.json')",
                "executed_at": generated_at,
                "description": "Six anomalous smoke cases rerun with raw-output capture, JSON-object retry, and conditional 70B fallback.",
                "tables_used": ["evaluation/judge_reliability_diagnostic.py"],
                "filters": ["two invalid faithfulness parses", "four hard-zero answer-relevancy scores"],
            },
        },
        {
            "id": "groq_json_docs",
            "label": "Groq structured-output documentation",
            "href": "https://console.groq.com/docs/structured-outputs",
            "query": {
                "engine": "Documentation",
                "description": "JSON Object Mode guarantees valid JSON syntax, while schema adherence requires schema-constrained structured outputs on a supported model.",
                "executed_at": generated_at,
            },
        },
    ]


def _cards() -> list[dict[str, Any]]:
    return [
        {
            "id": "reviewed_questions",
            "dataset": "summary",
            "sourceId": "retrieval_source",
            "description": "Questions with human-reviewed chunk-level evidence IDs.",
            "metrics": [{"label": "Reviewed questions", "field": "reviewed_questions", "format": "number"}],
        },
        {
            "id": "best_recall8",
            "dataset": "summary",
            "sourceId": "retrieval_source",
            "description": "Highest observed recall at the eight-result cutoff.",
            "metrics": [{"label": "Best Recall@8", "field": "best_recall8", "format": "percent"}],
        },
        {
            "id": "judge_remaining",
            "dataset": "summary",
            "sourceId": "judge_source",
            "description": "Selected judge anomalies still unresolved after the retry policy.",
            "metrics": [{"label": "Judge anomalies remaining", "field": "remaining_judge_anomalies", "format": "number"}],
        },
    ]


def _recall_chart() -> dict[str, Any]:
    return {
        "id": "recall_by_config",
        "title": "Reviewed benchmark recall across configurations and cutoffs",
        "subtitle": "Chunk-level labels make this stricter than the paper-level fallback used in Part 7.",
        "type": "bar",
        "intent": "comparison",
        "question": "Which retrieval configuration surfaces the reviewed evidence most reliably?",
        "rationale": "Grouped bars compare the same recall scale across five configurations and three cutoffs.",
        "dataset": "recall_chart",
        "sourceId": "retrieval_source",
        "encodings": {
            "x": {"field": "display_name", "type": "nominal", "label": "Configuration"},
            "y": {"field": "recall", "type": "quantitative", "format": "percent", "label": "Recall"},
            "color": {"field": "cutoff", "type": "nominal", "label": "Cutoff"},
            "tooltip": [
                {"field": "recall@4", "format": "percent", "label": "Recall@4"},
                {"field": "mrr", "format": "percent", "label": "MRR"},
                {"field": "avg_latency_ms", "format": "number", "label": "Avg latency (ms)"},
            ],
        },
        "valueFormat": "percent",
        "layout": "full",
    }


def _tables() -> list[dict[str, Any]]:
    return [
        {
            "id": "inventory_table",
            "title": "Actual Part 9 smoke retrieval path",
            "dataset": "inventory",
            "sourceId": "smoke_source",
            "defaultSort": {"field": "tier", "direction": "asc"},
            "columns": [
                {"field": "tier", "label": "Tier", "type": "text"},
                {"field": "retriever_class", "label": "Retriever", "type": "text"},
                {"field": "indexes", "label": "Index", "type": "text"},
                {"field": "top_k", "label": "Top-k", "type": "text"},
                {"field": "rerank", "label": "Rerank", "type": "text"},
                {"field": "mmr", "label": "MMR", "type": "text"},
            ],
        },
        {
            "id": "ablation_table",
            "title": "Retrieval ablation on reviewed chunk labels",
            "subtitle": "Recall@4 is included because the smoke generator consumed four contexts.",
            "dataset": "ablation",
            "sourceId": "retrieval_source",
            "defaultSort": {"field": "recall@4", "direction": "desc"},
            "columns": [
                {"field": "display_name", "label": "Configuration", "type": "text"},
                {"field": "recall@4", "label": "R@4", "format": "percent"},
                {"field": "recall@5", "label": "R@5", "format": "percent"},
                {"field": "recall@8", "label": "R@8", "format": "percent"},
                {"field": "recall@20", "label": "R@20", "format": "percent"},
                {"field": "precision@5", "label": "P@5", "format": "percent"},
                {"field": "precision@8", "label": "P@8", "format": "percent"},
                {"field": "precision@20", "label": "P@20", "format": "percent"},
                {"field": "mrr", "label": "MRR", "format": "percent"},
                {"field": "ndcg@20", "label": "nDCG@20", "format": "percent"},
                {"field": "avg_latency_ms", "label": "Avg latency (ms)", "format": "number"},
                {"field": "part7_recall@8", "label": "Part 7 R@8", "format": "percent"},
                {"field": "part7_delta_recall@8", "label": "Δ vs Part 7", "format": "percent", "movement": True},
            ],
        },
        {
            "id": "failure_table",
            "title": "Named generation failures correlated with reviewed evidence ranks",
            "dataset": "failures",
            "sourceId": "retrieval_source",
            "defaultSort": {"field": "query_id", "direction": "asc"},
            "columns": [
                {"field": "query_id", "label": "Question", "type": "text"},
                {"field": "status", "label": "Label status", "type": "text"},
                {"field": "dense_ranks", "label": "Dense ranks", "type": "text"},
                {"field": "sparse_ranks", "label": "Sparse ranks", "type": "text"},
                {"field": "hybrid_ranks", "label": "Hybrid ranks", "type": "text"},
                {"field": "rerank_ranks", "label": "Rerank ranks", "type": "text"},
                {"field": "mmr_ranks", "label": "MMR ranks", "type": "text"},
                {"field": "top4_configs", "label": "Top-4 evidence", "type": "text"},
                {"field": "evidence_excerpt", "label": "Reviewed evidence excerpt", "type": "text"},
            ],
        },
        {
            "id": "judge_table",
            "title": "Judge anomalies before and after reliability retries",
            "dataset": "judge",
            "sourceId": "judge_source",
            "defaultSort": {"field": "question_id", "direction": "asc"},
            "columns": [
                {"field": "question_id", "label": "Question", "type": "text"},
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "kind", "label": "Anomaly", "type": "text"},
                {"field": "before_score", "label": "Before", "format": "number"},
                {"field": "after_score", "label": "After", "format": "number"},
                {"field": "fixed", "label": "Fixed", "type": "text"},
                {"field": "baseline_status", "label": "Baseline", "type": "text"},
                {"field": "json_retry_status", "label": "JSON retry", "type": "text"},
                {"field": "fallback_status", "label": "70B fallback", "type": "text"},
                {"field": "raw_outputs", "label": "Raw outputs", "format": "number"},
            ],
        },
    ]


def _blocks(
    diagnostic: dict[str, Any], judge: dict[str, Any], recommendation: dict[str, Any]
) -> list[dict[str, Any]]:
    lift = diagnostic["ablation"]["reranker_lift"]
    fixed = judge["fixed"]
    remaining = judge["after_anomalies"]
    config = recommendation["display_name"]
    aggregate = diagnostic["ablation"]["aggregate"]
    sparse = aggregate["sparse"]
    recall8_gain = recommendation["recall@8"] - sparse["recall@8"]
    precision8_gain = recommendation["precision@8"] - sparse["precision@8"]
    latency_cost = recommendation["avg_latency_ms"] - sparse["avg_latency_ms"]
    rerank_selected = recommendation["config"] in {
        "hybrid_rerank",
        "hybrid_rerank_mmr",
    }
    generalization = (
        "Part 7’s no-rerank conclusion does **not** generalize to these benchmark tiers under the top-four evidence objective."
        if rerank_selected
        else "Part 7’s conclusion that reranking is not worth its latency **does** generalize to these benchmark tiers under the top-four evidence objective."
    )
    return [
        {"id": "title", "type": "markdown", "body": "# Part 9b — Retrieval-Stack Diagnostic"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "body": (
                "## Executive Summary\n\n"
                f"Use **{config}** for the next benchmark-tier generation evaluation. It ranked first under the decision rule that prioritizes Recall@4—the actual four-context generation budget—then Recall@8, MRR, and latency. "
                "The Part 9 benchmark tiers were BM25-only because the external benchmark builder produced only `external_bm25_index.pkl`, and the smoke runner called `BM25Indexer.search` directly; no dense benchmark collection or retrieval-stack orchestration existed. "
                f"The reliability audit resolved **{fixed} of {fixed + remaining}** selected anomalies; {remaining} remain. No full-eval defaults were changed."
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["reviewed_questions", "best_recall8", "judge_remaining"]},
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "retrieval_source",
            "body": (
                "## Key Findings\n\n"
                f"The reviewed benchmark contains **{diagnostic['benchmark']['reviewed_questions']}** questions ({diagnostic['benchmark']['qasa']} QASA, {diagnostic['benchmark']['qasper']} QASPER), all scored against chunk IDs rather than paper-level fallbacks. "
                f"Hybrid reranking changed Recall@8 by **{lift['recall@8_lift']:+.3f}** relative to hybrid RRF. The table also reports Recall@4 to match the generation context budget."
            ),
        },
        {"id": "recall_chart_block", "type": "chart", "chartId": "recall_by_config"},
        {
            "id": "scope",
            "type": "markdown",
            "body": "## Scope, Data, and Metric Definitions\n\nThe ablation includes only QASA/QASPER rows marked reviewed with non-empty `reference_context_ids`: 75 questions total. Each configuration used candidate_k=50 and final_k=20; hybrid used reciprocal-rank fusion with k=60, and MMR used λ=0.5. Recall, MRR, and nDCG use binary reviewed chunk relevance. Latency is local CPU wall time and should be treated as directional.",
        },
        {"id": "inventory_block", "type": "table", "tableId": "inventory_table"},
        {"id": "ablation_block", "type": "table", "tableId": "ablation_table"},
        {
            "id": "methodology",
            "type": "markdown",
            "body": "## Methodology\n\nA benchmark-scoped Qdrant collection was built additively from all 2,777 external benchmark chunks using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional cosine vectors) and the production Qdrant payload pattern. The production `ai_papers` collection was not modified. The same `RetrievalEvaluator` and metric functions used in Part 7 evaluated dense, BM25 sparse, hybrid RRF, hybrid plus cross-encoder reranking, and hybrid plus reranking plus MMR.",
        },
        {
            "id": "failure_correlation",
            "type": "markdown",
            "body": "## Generation-Failure Correlation\n\nFor the named QASA/QASPER failures, the table reports every reviewed evidence rank within the top 20 and identifies configurations that placed at least one reviewed chunk in the generator’s top four. A row marked unreviewed cannot support a retrieval-causality claim.",
        },
        {"id": "failure_block", "type": "table", "tableId": "failure_table"},
        {
            "id": "judge_reliability",
            "type": "markdown",
            "sourceId": "judge_source",
            "body": (
                "## Judge Reliability\n\n"
                f"The audit reran two invalid faithfulness parses and four hard-zero answer-relevancy cases while capturing raw judge text before RAGAS parsing. The default 8B judge call was free-form. A retry combined RAGAS’s schema-oriented prompt with Groq JSON Object Mode; this guaranteed JSON syntax but did not guarantee the expected RAGAS top-level schema. The 8B retry returned a schema-mismatched `analysis` wrapper for the faithfulness cases, and the long `gen-005` outputs ended before closing their JSON under the 1,024-token cap. The 70B judge ran only when the retry did not repair the anomaly. **{fixed} of {fixed + remaining}** selected anomalies were repaired, leaving **{remaining}**; three hard-zero answer-relevancy scores remained zero under 70B. Raw outputs and per-stage errors are preserved in the source JSON."
            ),
        },
        {"id": "judge_block", "type": "table", "tableId": "judge_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## Limitations and Robustness\n\nPart 7 flagged 40 of 50 questions as unreviewed title-derived bootstrap and evaluated all 50 with paper-level fallback labels, so its metrics are not directly comparable to this stricter chunk-level review. The ablation’s local CPU latency includes model warm-up and contention and is not a deployment benchmark. The benchmark dense collection is isolated from production and therefore covers only external benchmark chunks. Judge retries are a targeted six-case diagnostic, not a fresh full RAGAS evaluation.",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommendation and Next Steps\n\n"
                f"Adopt **{config}** as the single benchmark retrieval configuration for the next generation-evaluation run, preserving top_k=4 at the generator boundary. Against the current BM25-only path, it changes Recall@8 by **{recall8_gain:+.3f}**, Precision@8 by **{precision8_gain:+.3f}**, and measured local latency by **{latency_cost:+.1f} ms/query**. {generalization} Keep the new benchmark dense collection additive and leave `run_full_ragas_eval.py` defaults unchanged until this recommendation is deliberately applied. Do not change the default judge yet: JSON Object Mode alone repaired none of the six selected cases, and the full tested sequence repaired only two. A separate reviewed judge change should add explicit schema validation/repair and enough output budget for long faithfulness records before using the conditional 70B fallback."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": "## Further Questions\n\nBefore the next full 110-question run, decide whether the local CPU latency tradeoff is acceptable in the target environment and whether the unreviewed named failure should receive chunk-level evidence labels. A small repeated timing run without concurrent model work would provide cleaner serving-cost estimates.",
        },
    ]


def save_outputs(
    diagnostic: dict[str, Any], artifact: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output_dir / "retrieval_diagnostic.json"
    diagnostic_path.write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    artifact_path = output_dir / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = artifact["snapshot"]["datasets"]["ablation"]
    with (output_dir / "ablation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "failure_correlation.json").write_text(
        json.dumps(
            artifact["snapshot"]["datasets"]["failures"],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=Path("evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/retrieval_diagnostic.json"))
    parser.add_argument("--judge", type=Path, default=Path("evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/judge_reliability.json"))
    parser.add_argument("--external-dir", type=Path, default=Path("evaluation/data/external_benchmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/data/eval_results/retrieval_stack_diagnostic_20260812"))
    args = parser.parse_args(argv)
    diagnostic, artifact = build_report(args.diagnostic, args.judge, args.external_dir)
    save_outputs(diagnostic, artifact, args.output_dir)
    print(args.output_dir / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
