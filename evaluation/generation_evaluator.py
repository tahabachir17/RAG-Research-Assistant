"""Offline generation-quality evaluator independent of retrieval scoring."""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from generation.cli import run_generation
    from retrieval.models import RetrievalResult
    from .generation_golden import GenerationGoldenQuestion
    from .generation_metrics import (
        claim_level_citation_coverage,
        qualifying_item_precision,
        qualifying_item_recall,
        required_field_completeness,
        truncation_rate,
        unsupported_claim_rate,
    )
    from .llm_judge import LLMJudge
except ImportError:
    from generation.cli import run_generation
    from retrieval.models import RetrievalResult
    from generation_golden import GenerationGoldenQuestion
    from generation_metrics import claim_level_citation_coverage, qualifying_item_precision, qualifying_item_recall, required_field_completeness, truncation_rate, unsupported_claim_rate
    from llm_judge import LLMJudge


class GenerationEvaluator:
    """Exercise the real generation/validation/repair path over frozen contexts."""

    def __init__(
        self,
        *,
        llm: Any,
        chunk_lookup: Callable[[list[str]], list[RetrievalResult]],
        provider: str,
        model: str,
        judge: LLMJudge | None = None,
        max_retries: int = 1,
        max_context_tokens: int = 2500,
        cost_estimator: Callable[[Any], float | None] | None = None,
    ) -> None:
        self.llm, self.chunk_lookup = llm, chunk_lookup
        self.provider, self.model, self.judge = provider, model, judge
        self.max_retries, self.cost_estimator = max_retries, cost_estimator
        self.max_context_tokens = max_context_tokens

    def evaluate(self, questions: list[GenerationGoldenQuestion]) -> dict[str, Any]:
        runs = [self._evaluate_one(question) for question in questions]
        return build_generation_result(
            runs,
            provider=self.provider,
            model=self.model,
            judge=self.judge,
        )

    def evaluate_one(self, question: GenerationGoldenQuestion) -> dict[str, Any]:
        """Evaluate one question so callers can checkpoint immediately."""

        return self._evaluate_one(question)

    def _evaluate_one(self, question: GenerationGoldenQuestion) -> dict[str, Any]:
        chunks = self.chunk_lookup(question.retrieved_chunk_ids)
        if [chunk.chunk_id for chunk in chunks] != question.retrieved_chunk_ids:
            raise ValueError(f"{question.id}: chunk lookup must preserve every frozen chunk id in order")
        generated = run_generation(
            question.question,
            chunks,
            llm=self.llm,
            required_fields=question.required_fields,
            max_items=question.max_items,
            max_retries=self.max_retries,
            max_context_tokens=self.max_context_tokens,
        )
        claims = _claims(generated.answer, generated.structured_data)
        missing = {failure.split(":", 1)[1] for failure in generated.validation_failures or [] if failure.startswith("missing_required_field:")}
        present_fields = [field for field in question.required_fields if field not in missing]
        candidates = list(dict.fromkeys(question.expected_qualifying_items + list(question.excluded_items)))
        predicted = [item for item in candidates if re.search(rf"\b{re.escape(item)}\b", generated.answer, re.IGNORECASE)]
        evidence = [{"citation_number": index, "chunk_id": chunk.chunk_id, "text": chunk.text} for index, chunk in enumerate(chunks, 1)]
        subjects = [
            {
                "subject_id": claim.get("subject_id", f"claim-{index}"),
                "check": "claim_support",
                "text": claim["text"],
                "citations": claim["citations"],
                "field": claim.get("field"),
            }
            for index, claim in enumerate(claims, 1)
        ]
        subjects.extend({"subject_id": f"item-{index}", "check": "item_qualification", "text": item} for index, item in enumerate(predicted, 1))
        judged = self.judge.judge(question_id=question.id, question=question.question, answer=generated.answer, evidence=evidence, subjects=subjects, exclusion_criteria=question.excluded_items) if self.judge else None
        item_by_subject = {f"item-{index}": item for index, item in enumerate(predicted, 1)}
        judged_items = [item_by_subject[verdict.subject_id] for verdict in judged.verdicts if verdict.check == "item_qualification" and verdict.verdict in {"supported", "partially_supported"} and verdict.subject_id in item_by_subject] if judged and judged.judge_status == "judged" else None
        calibration = None
        if judged and judged.judge_status == "judged" and question.calibration_verdicts:
            actual = {verdict.subject_id: verdict.verdict for verdict in judged.verdicts}
            pairs = [(str(item.get("verdict", "")), actual.get(str(item.get("subject_id", "")))) for item in question.calibration_verdicts]
            comparable = [(human, judge_label) for human, judge_label in pairs if judge_label is not None]
            calibration = sum(human == judge_label for human, judge_label in comparable) / len(comparable) if comparable else None
        failures = generated.validation_failures or []
        return {
            "id": question.id,
            "paper_id": question.paper_id,
            "title": question.title,
            "question": question.question,
            "reviewed": question.reviewed,
            "answer": generated.answer,
            "structured_data": generated.structured_data,
            "sources": generated.sources,
            "context_chunk_ids": generated.context_chunk_ids,
            "finish_reason": generated.finish_reason,
            "final_attempt": generated.final_attempt,
            "retry_count": int(generated.final_attempt == "repaired"),
            "validation_failures": failures,
            "citation_valid": generated.citations_valid and not any(failure in {"missing_citation", "citation_out_of_range", "unsupported_citation_format"} for failure in failures),
            "claim_citation_coverage": claim_level_citation_coverage(claims),
            "required_field_completeness": required_field_completeness(present_fields, question.required_fields),
            "max_item_compliant": "too_many_items" not in failures,
            "qualifying_item_precision": qualifying_item_precision(judged_items, question.expected_qualifying_items) if judged_items is not None else None,
            "qualifying_item_recall": qualifying_item_recall(judged_items, question.expected_qualifying_items) if judged_items is not None else None,
            "judge_status": judged.judge_status if judged else "disabled",
            "judge_error": judged.error if judged else None,
            "judge_verdicts": [asdict(verdict) for verdict in judged.verdicts] if judged else [],
            "calibration_exact_agreement": calibration,
            "latency_ms": generated.latency_ms,
            "estimated_cost": self.cost_estimator(generated) if self.cost_estimator else None,
        }


def build_generation_result(
    runs: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    judge: LLMJudge | None = None,
) -> dict[str, Any]:
    """Aggregate already-completed question rows without model calls."""

    judged_labels = [
        verdict["verdict"]
        for run in runs
        for verdict in run["judge_verdicts"]
        if verdict["check"] == "claim_support"
    ]
    calibration = [
        run["calibration_exact_agreement"]
        for run in runs
        if run["calibration_exact_agreement"] is not None
    ]
    item_labels = [
        verdict["verdict"]
        for run in runs
        for verdict in run["judge_verdicts"]
        if verdict["check"] == "item_qualification"
    ]
    aggregate = {
        "questions": len(runs),
        "citation_validity_rate": _mean(run["citation_valid"] for run in runs),
        "claim_level_citation_coverage": _mean(
            run["claim_citation_coverage"] for run in runs
        ),
        "required_field_completeness": _mean(
            run["required_field_completeness"] for run in runs
        ),
        "truncation_rate": truncation_rate(run["finish_reason"] for run in runs),
        "retry_rate": _mean(run["retry_count"] > 0 for run in runs),
        "max_item_compliance_rate": _mean(run["max_item_compliant"] for run in runs),
        "qualifying_item_precision": _nullable_mean(
            run["qualifying_item_precision"] for run in runs
        ),
        "qualifying_item_recall": _nullable_mean(
            run["qualifying_item_recall"] for run in runs
        ),
        "unsupported_claim_rate": unsupported_claim_rate(judged_labels),
        "incorrect_classification_rate": unsupported_claim_rate(item_labels),
        "unsupported_qualifying_items": sum(
            label == "unsupported" for label in item_labels
        ),
        "grounded_claim_rate": (
            _mean(label == "supported" for label in judged_labels)
            if judged_labels
            else None
        ),
        "judge_coverage": _mean(run["judge_status"] == "judged" for run in runs),
        "calibration_exact_agreement": _nullable_mean(calibration),
        "avg_latency_ms": _mean(run["latency_ms"] for run in runs),
        "avg_cost": _nullable_mean(run["estimated_cost"] for run in runs),
    }
    return {
        "provider": provider,
        "model": model,
        "judge": {
            "enabled": judge is not None,
            "provider": judge.judge_provider if judge else None,
            "model": judge.judge_model if judge else None,
        },
        "aggregate": aggregate,
        "questions": runs,
    }


def save_generation_outputs(
    evaluation: dict[str, Any],
    output_dir: str | Path,
    *,
    stamp: str | None = None,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = directory / f"generation_eval_{stamp}"
    json_path, csv_path, markdown_path = base.with_suffix(".json"), base.with_suffix(".csv"), base.with_suffix(".md")
    _atomic_text(json_path, json.dumps(evaluation, indent=2, ensure_ascii=False))
    ragas_metrics = list(evaluation.get("ragas", {}).get("metrics", []))
    fields = ["id", "reviewed", "citation_valid", "claim_citation_coverage", "required_field_completeness", "finish_reason", "retry_count", "max_item_compliant", "qualifying_item_precision", "qualifying_item_recall", "judge_status", "latency_ms", "estimated_cost", *ragas_metrics]
    ragas_by_id = {
        str(row.get("id")): row
        for row in evaluation.get("ragas", {}).get("questions", [])
    }
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".part")
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                field: (
                    row.get(field)
                    if field not in ragas_metrics
                    else ragas_by_id.get(str(row.get("id")), {}).get(field)
                )
                for field in fields
            }
            for row in evaluation["questions"]
        )
    os.replace(csv_temporary, csv_path)
    _atomic_text(markdown_path, _markdown(evaluation))
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _claims(answer: str, structured_data: Any | None = None) -> list[dict[str, Any]]:
    structured_claims = _structured_claims(structured_data)
    if structured_claims is not None:
        return structured_claims
    claims: list[dict[str, Any]] = []
    for text in re.split(r"(?<=[.!?])\s+|\n+", answer):
        normalized = text.strip()
        if not normalized:
            continue
        citations = [int(value) for value in re.findall(r"\[(\d+)\]", normalized)]
        # Legacy table outputs have no structured payload. Ignore headers and
        # separators, but retain cited data rows as coarse fallback claims.
        if normalized.startswith("|") and not citations:
            continue
        claims.append({"text": normalized, "citations": citations, "cited": bool(citations)})
    return claims


def _structured_claims(structured_data: Any | None) -> list[dict[str, Any]] | None:
    if not isinstance(structured_data, Mapping):
        return None
    raw_claims = structured_data.get("claims")
    if isinstance(raw_claims, Sequence) and not isinstance(raw_claims, (str, bytes)):
        return [
            {
                "subject_id": f"claim-{index}",
                "text": str(claim.get("text", "")).strip(),
                "citations": list(claim.get("citations", [])),
                "cited": bool(claim.get("citations")),
            }
            for index, claim in enumerate(raw_claims, 1)
            if isinstance(claim, Mapping) and str(claim.get("text", "")).strip()
        ]
    raw_items = structured_data.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return None
    claims: list[dict[str, Any]] = []
    for item_index, item in enumerate(raw_items, 1):
        if not isinstance(item, Mapping):
            continue
        for field, cell in item.items():
            if not isinstance(cell, Mapping):
                continue
            text = str(cell.get("text", "")).strip()
            if not text or _is_absent_value(text):
                continue
            citations = list(cell.get("citations", []))
            claims.append(
                {
                    "subject_id": f"claim-{item_index}:{field}",
                    "field": str(field),
                    "text": text,
                    "citations": citations,
                    "cited": bool(citations),
                }
            )
    return claims


def _is_absent_value(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return normalized.startswith("not reported") or normalized.startswith("not provided")


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else 0.0


def _nullable_mean(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    return statistics.fmean(items) if items else None


def _markdown(evaluation: dict[str, Any]) -> str:
    aggregate = evaluation["aggregate"]
    cost = "unavailable" if aggregate["avg_cost"] is None else f"{aggregate['avg_cost']:.6f}"
    lines = [
        "# Generation evaluation", "",
        f"Provider/model: `{evaluation['provider']}` / `{evaluation['model']}`",
        f"Judge: `{evaluation.get('judge', {}).get('provider')}` / `{evaluation.get('judge', {}).get('model')}`",
        "",
        "| citation valid | claim coverage | field completeness | truncated | retries | grounded | judge coverage | avg ms | avg cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {aggregate['citation_validity_rate']:.3f} | {aggregate['claim_level_citation_coverage']:.3f} | {aggregate['required_field_completeness']:.3f} | {aggregate['truncation_rate']:.3f} | {aggregate['retry_rate']:.3f} | {_format_nullable(aggregate['grounded_claim_rate'])} | {aggregate['judge_coverage']:.3f} | {aggregate['avg_latency_ms']:.1f} | {cost} |",
        "", "## Per-question failures", "",
    ]
    coverage = evaluation.get("coverage")
    if coverage:
        lines[4:4] = [
            f"Dataset coverage: {coverage.get('evaluated', 0)}/{coverage.get('total', 0)} evaluated; "
            f"{coverage.get('reviewed', 0)} reviewed; {coverage.get('unreviewed', 0)} unreviewed.",
            "",
        ]
    for row in evaluation["questions"]:
        failures = ", ".join(row["validation_failures"]) or "none"
        lines.append(f"- `{row['id']}`: validation={failures}; judge={row['judge_status']}; latency={row['latency_ms']} ms.")
    ragas = evaluation.get("ragas")
    if ragas:
        lines.extend(["", "## RAGAS (reference-free)", ""])
        if ragas.get("status") in {"completed", "partial"}:
            scores = ragas.get("aggregate", {})
            metric_names = list(ragas.get("metrics", [])) or [
                "faithfulness",
                "answer_relevancy",
                "context_utilization",
            ]
            lines.extend(
                [
                    "| " + " | ".join(name.replace("_", " ") for name in metric_names) + " |",
                    "|" + "---:|" * len(metric_names),
                    "| "
                    + " | ".join(
                        _format_nullable(scores.get(name))
                        for name in metric_names
                    )
                    + " |",
                ]
            )
            if ragas.get("reason"):
                lines.append(f"\nNote: {ragas['reason']}.")
        else:
            lines.append(f"RAGAS status: {ragas.get('status', 'unknown')} ({ragas.get('reason', 'no reason')}).")
    return "\n".join(lines) + "\n"


def _format_nullable(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.3f}"


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
