"""Keep retrieval, controlled generation, and end-to-end evaluation distinct."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EvaluationLayer:
    key: str
    title: str
    purpose: str
    metrics: dict[str, float | None]


def build_evaluation_layers(payload: dict[str, Any]) -> list[EvaluationLayer]:
    """Derive three non-overlapping reporting views from evaluator output."""

    questions = payload.get("questions", [])
    metric_rows = {
        str(row.get("id")): row
        for row in payload.get("ragas", {}).get("questions", [])
    }
    gold_ids = {
        str(row.get("id"))
        for row in questions
        if row.get("reference_context_ids")
        and row.get("retrieved_chunk_ids") == row.get("reference_context_ids")
    }
    return [
        EvaluationLayer(
            "retrieval",
            "Retrieval",
            "Did the retriever find the answer evidence?",
            {
                "context_precision": _mean_metric(metric_rows.values(), "context_precision"),
                "context_recall": _mean_metric(metric_rows.values(), "context_recall"),
            },
        ),
        EvaluationLayer(
            "controlled_generation",
            "Controlled generation",
            "Can the generator answer when given perfect gold evidence?",
            {
                "answer_correctness": _mean_metric(
                    (metric_rows[item] for item in gold_ids if item in metric_rows),
                    "answer_correctness",
                ),
                "faithfulness": _mean_metric(
                    (metric_rows[item] for item in gold_ids if item in metric_rows),
                    "faithfulness",
                ),
                "concept_recall": _mean_values(
                    row.get("concept_recall")
                    for row in questions
                    if str(row.get("id")) in gold_ids
                ),
            },
        ),
        EvaluationLayer(
            "end_to_end_rag",
            "End-to-end RAG",
            "Do retrieval and generation succeed together?",
            {
                "answer_correctness": _mean_metric(metric_rows.values(), "answer_correctness"),
                "faithfulness": _mean_metric(metric_rows.values(), "faithfulness"),
            },
        ),
    ]


def render_layered_sections(layers: list[EvaluationLayer]) -> str:
    lines: list[str] = []
    for layer in layers:
        lines.extend([f"## {layer.title}", "", layer.purpose, ""])
        for metric, value in layer.metrics.items():
            rendered = "unavailable" if value is None else f"{value:.3f}"
            lines.append(f"- {metric}: {rendered}")
        lines.append("")
    return "\n".join(lines)


def _mean_metric(rows: Any, metric: str) -> float | None:
    return _mean_values(row.get(metric) for row in rows)


def _mean_values(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.fmean(numeric) if numeric else None
