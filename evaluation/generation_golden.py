"""Schema and validation for the frozen generation-quality question set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GenerationGoldenQuestion:
    id: str
    question: str
    retrieved_chunk_ids: list[str]
    expected_qualifying_items: list[str]
    excluded_items: dict[str, str]
    required_fields: list[str]
    max_items: int | None
    reviewed: bool
    calibration_verdicts: list[dict[str, str]]
    paper_id: str = ""
    title: str = ""
    reference_answer: str | None = None


def load_generation_golden(path: str | Path, *, require_reviewed: bool = False) -> list[GenerationGoldenQuestion]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("generation golden data must have schema_version 1")
    records = payload.get("questions")
    if not isinstance(records, list):
        raise TypeError("questions must be a JSON array")
    questions: list[GenerationGoldenQuestion] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each generation question must be an object")
        identifier = str(record.get("id", "")).strip()
        question = str(record.get("question", "")).strip()
        if not identifier or not question or identifier in seen:
            raise ValueError("question ids must be unique and question text non-empty")
        seen.add(identifier)
        item = GenerationGoldenQuestion(
            identifier,
            question,
            _strings(record.get("retrieved_chunk_ids", [])),
            _strings(record.get("expected_qualifying_items", [])),
            {str(key): str(value) for key, value in dict(record.get("excluded_items", {})).items()},
            _strings(record.get("required_fields", [])),
            record.get("max_items"),
            bool(record.get("reviewed", False)),
            list(record.get("calibration_verdicts", [])),
            str(record.get("paper_id", "")).strip(),
            str(record.get("title", "")).strip(),
            _optional_text(
                record.get("reference_answer", record.get("ground_truth"))
            ),
        )
        if not item.retrieved_chunk_ids:
            raise ValueError(f"{identifier}: retrieved_chunk_ids must not be empty")
        if item.max_items is not None and (not isinstance(item.max_items, int) or item.max_items <= 0):
            raise ValueError(f"{identifier}: max_items must be positive or null")
        if require_reviewed and not item.reviewed:
            raise ValueError(f"{identifier}: unreviewed question cannot be used for release gating")
        questions.append(item)
    return questions


def calibration_exact_agreement(human: list[str], judge: list[str]) -> float:
    if len(human) != len(judge):
        raise ValueError("human and judge calibration labels must align")
    return sum(left == right for left, right in zip(human, judge)) / len(human) if human else 0.0


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("golden list fields must be arrays")
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
