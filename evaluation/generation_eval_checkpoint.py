"""Atomic, resumable state for the generation evaluation command."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GenerationEvalCheckpoint:
    path: Path
    payload: dict[str, Any]
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def create(cls, path: str | Path, payload: dict[str, Any]) -> "GenerationEvalCheckpoint":
        checkpoint = cls(Path(path), payload)
        checkpoint.save()
        return checkpoint

    @classmethod
    def load(cls, path: str | Path) -> "GenerationEvalCheckpoint":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("generation evaluation checkpoint must be a JSON object")
        return cls(source, payload)

    def completed_question_ids(self) -> set[str]:
        return {str(row["id"]) for row in self.payload.get("questions", [])}

    def metric_completed(self, question_id: str, metric: str) -> bool:
        return self.payload.get("metric_progress", {}).get(question_id, {}).get(metric) in {
            "completed",
            "unavailable",
        }

    def record_question(self, row: dict[str, Any]) -> None:
        with self._lock:
            rows = self.payload.setdefault("questions", [])
            by_id = {str(item["id"]): index for index, item in enumerate(rows)}
            identifier = str(row["id"])
            if identifier in by_id:
                rows[by_id[identifier]] = row
            else:
                rows.append(row)
            self._save_unlocked()

    def record_metric(
        self,
        question_id: str,
        metric: str,
        *,
        status: str,
        value: float | None = None,
        reason: str | None = None,
    ) -> None:
        if status not in {"completed", "unavailable", "failed"}:
            raise ValueError(f"invalid metric checkpoint status: {status}")
        with self._lock:
            ragas = self.payload.setdefault("ragas", {})
            rows = ragas.setdefault("questions", [])
            row = next((item for item in rows if str(item.get("id")) == question_id), None)
            if row is None:
                row = {"id": question_id}
                rows.append(row)
            row[metric] = value
            if reason:
                row.setdefault("reasons", {})[metric] = reason
            elif isinstance(row.get("reasons"), dict):
                row["reasons"].pop(metric, None)
            self.payload.setdefault("metric_progress", {}).setdefault(question_id, {})[
                metric
            ] = status
            self._save_unlocked()

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".part")
        temporary.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.path)


def latest_compatible_checkpoint(
    output_dir: str | Path, signature: dict[str, Any]
) -> Path | None:
    """Find the newest checkpoint produced with the same run-defining inputs."""

    candidates = sorted(
        Path(output_dir).glob("generation_eval_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("run_signature") == signature:
            return candidate
    return None
