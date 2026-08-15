"""Run the additive Part 9m 40-question live CLI benchmark.

The benchmark reuses the Part 9l paired evidence needs without modifying that
dataset.  Each subprocess is the production CLI command required by the
benchmark; EvaluationRateLimitClient wraps the subprocess adapter so transient
and 429 failures use the repository's bounded evaluation retry policy.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from evaluation.rate_limit_client import EvaluationRateLimitClient
from generation.llm_client import LLMClientError


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/data/part9l_mixed_style_retrieval_qa.json"
OUT = ROOT / "evaluation/data/eval_results/part9m_reduced_live_benchmark_20260815"
CHECKPOINT = OUT / "live_checkpoint.json"
FINAL = OUT / "reduced_live_results.json"
MANIFEST = OUT / "question_manifest.json"


def _parse_final_json(stdout: str) -> tuple[str, dict[str, Any]]:
    decoder = json.JSONDecoder()
    for start in reversed([i for i, char in enumerate(stdout) if char == "{"]):
        try:
            payload, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if not stdout[start + end :].strip() and isinstance(payload, dict):
            return stdout[:start].rstrip(), payload
    raise ValueError("CLI stdout did not end with a JSON object")


def _retry_after(stderr: str) -> float | None:
    match = re.search(r"Please try again in (?:(\d+)m)?([\d.]+)s", stderr)
    if not match:
        return None
    return int(match.group(1) or 0) * 60.0 + float(match.group(2))


class _CLIProcessAdapter:
    provider = "groq"

    def __init__(self) -> None:
        self.command: list[str] = []
        self.completed: subprocess.CompletedProcess[str] | None = None

    def set_command(self, command: list[str]) -> None:
        self.command = command

    def complete(self, _system: str, _user: str):
        self.completed = subprocess.run(
            self.command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if self.completed.returncode == 0:
            return self.completed
        stderr = self.completed.stderr.strip()
        status = 429 if "429" in stderr or "rate limit" in stderr.casefold() else None
        raise LLMClientError(
            "groq",
            stderr or f"CLI exited {self.completed.returncode}",
            status_code=status,
            retry_after=_retry_after(stderr),
        )


def _question_manifest() -> list[dict[str, Any]]:
    triplets = json.loads(SOURCE.read_text(encoding="utf-8"))["triplets"]
    questions: list[dict[str, Any]] = []
    for tier, prefix in (("vague", "part9m-vague"), ("topic_named", "part9m-topic")):
        for index, row in enumerate(triplets, 1):
            questions.append(
                {
                    "id": f"{prefix}-{index:02d}",
                    "base_id": row["base_id"],
                    "source_dataset": row["source_dataset"],
                    "style": "vague_casual" if tier == "vague" else "topic_named",
                    "question": row[tier],
                    "original_question": row["original"],
                    "reviewed_relevant_chunk_ids": row["relevant_chunk_ids"],
                }
            )
    return questions


def _save_checkpoint(rows: list[dict[str, Any]], *, stop_reason: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_questions": len(rows),
                "stop_reason": stop_reason,
                "questions": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    questions = _question_manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": date.today().isoformat(),
                "selection_rule": "all 20 vague followed by all 20 topic-named phrasings from the unchanged Part 9l paired-20 triplets",
                "questions": questions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    if CHECKPOINT.exists():
        rows = json.loads(CHECKPOINT.read_text(encoding="utf-8"))["questions"]
    completed_ids = {str(row["id"]) for row in rows}
    stop_reason: str | None = None
    adapter = _CLIProcessAdapter()
    bounded = EvaluationRateLimitClient(
        adapter,
        max_retries=2,
        default_wait_seconds=10.0,
        max_wait_seconds=60.0,
        requests_per_second=0.08,
    )

    for position, question in enumerate(questions, 1):
        if question["id"] in completed_ids:
            print(f"[{position:02d}/40] resume {question['id']}", flush=True)
            continue
        tier_position = position if position <= 20 else position - 20
        show_prompt = tier_position <= 5
        command = [
            sys.executable,
            "-m",
            "generation.cli",
            str(question["question"]),
            "--retrieve",
            "--live",
            "--provider",
            "groq",
        ]
        if show_prompt:
            command.append("--show-prompt")
        adapter.set_command(command)
        try:
            completed = bounded.complete("", str(question["question"]))
        except Exception as exc:
            stop_reason = f"{question['id']}: {type(exc).__name__}: {exc}"
            _save_checkpoint(rows, stop_reason=stop_reason)
            print(f"STOP: {stop_reason}", file=sys.stderr, flush=True)
            break
        prompt_capture, payload = _parse_final_json(completed.stdout)
        rows.append(
            {
                **question,
                "command": "python -m generation.cli <question> --retrieve --live --provider groq"
                + (" --show-prompt" if show_prompt else ""),
                "prompt_capture": prompt_capture if show_prompt else None,
                "cli_stderr": completed.stderr.strip(),
                "cli_payload": payload,
            }
        )
        _save_checkpoint(rows)
        print(f"[{position:02d}/40] generated {question['id']}", flush=True)

    payload = {
        "schema_version": 1,
        "created_at": date.today().isoformat(),
        "command_contract": "python -m generation.cli <question> --retrieve --live --provider groq",
        "rate_limit_wrapper": "evaluation.rate_limit_client.EvaluationRateLimitClient(max_retries=2)",
        "completion_status": "complete" if len(rows) == len(questions) else "partial",
        "completed_questions": len(rows),
        "skipped_questions": len(questions) - len(rows),
        "stop_reason": stop_reason,
        "show_prompt_question_ids": [row["id"] for row in rows if row["prompt_capture"]],
        "questions": rows,
    }
    FINAL.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(FINAL), "completed": len(rows), "skipped": 40 - len(rows), "stop_reason": stop_reason}, indent=2))
    return 0 if len(rows) == len(questions) else 2


if __name__ == "__main__":
    raise SystemExit(main())
