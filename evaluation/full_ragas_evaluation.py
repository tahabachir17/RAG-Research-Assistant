"""Assembly, resumable metric caching, and reporting for the Part 9 RAGAS run."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import statistics
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from processing.bm25_indexer import BM25Indexer
    from retrieval.models import RetrievalResult
    from .generation_golden import GenerationGoldenQuestion, load_generation_golden
except ImportError:
    from processing.bm25_indexer import BM25Indexer
    from retrieval.models import RetrievalResult
    from generation_golden import GenerationGoldenQuestion, load_generation_golden


LOGGER = logging.getLogger(__name__)
FULL_RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)
TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
    "answer_correctness": 0.75,
}
DIAGNOSTIC_ONLY_METRICS = frozenset({"answer_relevancy"})


def metric_release_result(metric: str, mean: float | None) -> str:
    """Return release status while keeping uncalibrated metrics diagnostic-only."""

    if metric in DIAGNOSTIC_ONLY_METRICS:
        return "diagnostic only"
    if mean is None:
        return "unavailable"
    return "pass" if mean > TARGETS[metric] else "below"


def assemble_evaluation_questions(
    manual_path: str | Path,
    external_dir: str | Path,
    *,
    manual_limit: int | None = None,
    qasa_limit: int | None = None,
    qasper_limit: int | None = None,
) -> list[GenerationGoldenQuestion]:
    """Merge manual, QASA, and QASPER inputs and explicitly skip empty SciDQA."""

    manual = [
        replace(
            row,
            source_dataset="manual",
            alignment_status="aligned",
        )
        for row in load_generation_golden(manual_path)[:manual_limit]
    ]
    directory = Path(external_dir)
    tiers: list[GenerationGoldenQuestion] = []
    for name, limit in (("qasa", qasa_limit), ("qasper", qasper_limit)):
        rows = load_generation_golden(directory / f"{name}_generation_qa.json")
        tiers.extend(
            replace(
                row,
                source_dataset=name,
                alignment_status="aligned" if row.reviewed else "unreviewed",
            )
            for row in rows[:limit]
        )
    scidqa = load_generation_golden(directory / "scidqa_generation_qa.json")
    if not scidqa:
        LOGGER.info("SciDQA contains 0 records; skipping tier")
    else:
        LOGGER.warning("SciDQA contains %d records but is out of scope; skipping", len(scidqa))
    combined = [*manual, *tiers]
    identifiers = [row.id for row in combined]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise ValueError(f"question ids collide across tiers: {duplicates}")
    return combined


def retrieve_external_contexts(
    questions: Sequence[GenerationGoldenQuestion],
    retriever: Any,
    *,
    top_k: int = 4,
) -> list[GenerationGoldenQuestion]:
    """Retrieve benchmark contexts for external questions only."""

    configured: list[GenerationGoldenQuestion] = []
    for question in questions:
        if question.source_dataset == "manual":
            configured.append(question)
            continue
        matches = retriever.search(question.question, top_k=top_k)
        chunk_ids = [
            str(row.chunk_id if hasattr(row, "chunk_id") else row.get("chunk_id"))
            for row in matches
            if (row.chunk_id if hasattr(row, "chunk_id") else row.get("chunk_id"))
        ]
        if not chunk_ids:
            raise ValueError(f"{question.id}: configured benchmark retriever returned no chunks")
        configured.append(replace(question, retrieved_chunk_ids=chunk_ids))
    return configured


class TieredChunkLookup:
    """Resolve IDs from production and external indexes without mixing searches."""

    def __init__(self, *indexes: BM25Indexer) -> None:
        self._chunks: dict[str, dict[str, Any]] = {}
        for index in indexes:
            for chunk in index.chunks:
                identifier = str(chunk.get("chunk_id", ""))
                if identifier in self._chunks and self._chunks[identifier] != chunk:
                    raise ValueError(f"chunk id collision across indexes: {identifier}")
                self._chunks[identifier] = dict(chunk)

    def __call__(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        missing = [item for item in chunk_ids if item not in self._chunks]
        if missing:
            raise KeyError(f"chunks are missing from the routed indexes: {missing}")
        return [
            RetrievalResult.from_payload(self._chunks[item], score=0.0, source="frozen")
            for item in chunk_ids
        ]


@dataclass(slots=True)
class MetricJsonlCache:
    """Append-only cache keyed by question and metric; latest entry wins."""

    path: Path
    _lock: threading.Lock

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def entries(self) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        if not self.path.exists():
            return result
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (str(row["question_id"]), str(row["metric"]))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid metric cache line {number}: {exc}") from exc
            result[key] = row
        return result

    def completed(self) -> set[tuple[str, str]]:
        return {
            key
            for key, row in self.entries().items()
            if row.get("status") in {"completed", "unavailable"}
        }

    def append(
        self,
        question_id: str,
        metric: str,
        *,
        status: str,
        value: float | None = None,
        reason: str | None = None,
    ) -> None:
        if status not in {"completed", "unavailable", "failed"}:
            raise ValueError(f"invalid cache status: {status}")
        row = {
            "question_id": question_id,
            "metric": metric,
            "status": status,
            "value": value,
            "reason": reason,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def aggregate_scores(
    questions: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate every metric by source tier and independently by alignment status."""

    scores = {str(row["id"]): row for row in metric_rows}
    return {
        "by_source_tier": _aggregate_slice(questions, scores, "source_tier"),
        "by_alignment_status": _aggregate_slice(questions, scores, "alignment_status"),
    }


def _aggregate_slice(
    questions: Sequence[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted({str(row.get(field, "")) for row in questions})
    for group in groups:
        rows = [row for row in questions if str(row.get(field, "")) == group]
        for metric in FULL_RAGAS_METRICS:
            values = [
                scores.get(str(row["id"]), {}).get(metric)
                for row in rows
            ]
            finite = [float(value) for value in values if _finite(value)]
            output.append(
                {
                    field: group,
                    "metric": metric,
                    "mean": statistics.fmean(finite) if finite else None,
                    "scored": len(finite),
                    "unavailable": len(rows) - len(finite),
                    "total": len(rows),
                }
            )
    return output


def write_full_ragas_outputs(
    run_dir: str | Path,
    *,
    payload: dict[str, Any],
) -> dict[str, Path]:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    csv_path = directory / "per_question.csv"
    markdown_path = directory / "summary.md"
    _atomic_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
    scores = {str(row["id"]): row for row in payload["ragas"]["questions"]}
    fields = [
        "question_id", "question", "source_tier", "alignment_status",
        "generated_answer", *FULL_RAGAS_METRICS,
    ]
    temporary = csv_path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["questions"]:
            metric_row = scores.get(str(row["id"]), {})
            writer.writerow(
                {
                    "question_id": row["id"],
                    "question": row["question"],
                    "source_tier": row["source_tier"],
                    "alignment_status": row["alignment_status"],
                    "generated_answer": row["answer"],
                    **{
                        metric: (
                            metric_row.get(metric)
                            if metric_row.get(metric) is not None
                            else "unavailable"
                        )
                        for metric in FULL_RAGAS_METRICS
                    },
                }
            )
    os.replace(temporary, csv_path)
    _atomic_text(markdown_path, _markdown(payload))
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Full generation RAGAS evaluation",
        "",
        f"Generator: `{payload['generator']['provider']}` / `{payload['generator']['model']}`  ",
        f"RAGAS judge: `{payload['judge']['provider']}` / `{payload['judge']['model']}`",
        "",
    ]
    for title, key, group_field in (
        ("By source tier", "by_source_tier", "source_tier"),
        ("By alignment status", "by_alignment_status", "alignment_status"),
    ):
        lines.extend([
            f"## {title}", "",
            "| group | metric | mean | target | result | scored | unavailable |",
            "|---|---|---:|---:|---|---:|---:|",
        ])
        for row in payload["aggregates"][key]:
            mean = row["mean"]
            target = TARGETS[row["metric"]]
            result = metric_release_result(row["metric"], mean)
            lines.append(
                f"| {row[group_field]} | {row['metric']} | {_score(mean)} | > {target:.2f} | {result} | {row['scored']} | {row['unavailable']} |"
            )
        lines.append("")
    failures = _lowest_examples(payload)
    lines.extend(["## Lowest-scoring examples", ""])
    if not failures:
        lines.append("No scored examples are available yet.")
    else:
        for row in failures:
            lines.append(
                f"- `{row['id']}` ({row['source_tier']}): {row['metric']}={row['score']:.3f} — {row['question']}"
            )
    return "\n".join(lines) + "\n"


def _lowest_examples(payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in payload["questions"]}
    metric_values = {
        metric: [
            float(row[metric])
            for row in payload["ragas"]["questions"]
            if _finite(row.get(metric))
        ]
        for metric in FULL_RAGAS_METRICS
    }
    means = {
        metric: statistics.fmean(values)
        for metric, values in metric_values.items()
        if values
    }
    if not means:
        return []
    lowest_metric = min(means, key=lambda metric: (means[metric], metric))
    candidates: list[dict[str, Any]] = []
    for score_row in payload["ragas"]["questions"]:
        if _finite(score_row.get(lowest_metric)):
            question = by_id[str(score_row["id"])]
            candidates.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "source_tier": question["source_tier"],
                    "metric": lowest_metric,
                    "score": float(score_row[lowest_metric]),
                }
            )
    return sorted(candidates, key=lambda row: (row["score"], row["id"]))[:limit]


def metric_rows_from_cache(
    question_ids: Iterable[str], cache: MetricJsonlCache
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    entries = cache.entries()
    rows: list[dict[str, Any]] = []
    progress: dict[str, dict[str, str]] = {}
    for identifier in question_ids:
        row: dict[str, Any] = {"id": identifier}
        for metric in FULL_RAGAS_METRICS:
            cached = entries.get((identifier, metric))
            if cached is None:
                continue
            progress.setdefault(identifier, {})[metric] = str(cached.get("status"))
            row[metric] = cached.get("value")
            if cached.get("reason"):
                row.setdefault("reasons", {})[metric] = cached["reason"]
        rows.append(row)
    return rows, progress


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _score(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.3f}"


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
