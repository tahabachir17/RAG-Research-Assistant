from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "evaluation/data/human_style_generation_qa.json"
OUT = ROOT / "evaluation/data/eval_results/part9l_human_query_benchmark_20260814"
CHECKPOINT = OUT / "live_checkpoint.json"
FINAL = OUT / "human_style_live_results.json"


def parse_final_json(stdout: str) -> tuple[str, dict[str, object]]:
    decoder = json.JSONDecoder()
    starts = [i for i, char in enumerate(stdout) if char == "{"]
    for start in reversed(starts):
        try:
            payload, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if not stdout[start + end :].strip() and isinstance(payload, dict):
            return stdout[:start].rstrip(), payload
    raise ValueError("CLI stdout did not end with a JSON object")


def save(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(
        json.dumps({"schema_version": 1, "questions": rows}, indent=2),
        encoding="utf-8",
    )


def run_cli(command: list[str], question_id: str) -> subprocess.CompletedProcess[str]:
    while True:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if completed.returncode == 0:
            return completed
        rate_limit = re.search(
            r"Please try again in (?:(\d+)m)?([\d.]+)s", completed.stderr
        )
        if not rate_limit:
            return completed
        minutes = int(rate_limit.group(1) or 0)
        remaining = minutes * 60.0 + float(rate_limit.group(2)) + 3.0
        while remaining > 0:
            interval = min(50.0, remaining)
            print(
                f"[quota] {question_id}: retrying in {remaining:.0f}s",
                flush=True,
            )
            time.sleep(interval)
            remaining -= interval


def main() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    rows: list[dict[str, object]] = []
    if CHECKPOINT.exists():
        rows = json.loads(CHECKPOINT.read_text(encoding="utf-8"))["questions"]
    completed = {str(row["id"]) for row in rows if row.get("cli_payload")}
    finalize_partial = os.getenv("PART9L_FINALIZE_PARTIAL") == "1"
    for position, question in enumerate(questions, 1):
        if question["id"] in completed:
            print(f"[{position:02d}/{len(questions)}] resume {question['id']}", flush=True)
            continue
        if finalize_partial:
            print(f"[{position:02d}/{len(questions)}] skipped {question['id']}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "generation.cli",
            question["question"],
            "--retrieve",
            "--live",
            "--provider",
            "groq",
        ]
        if position <= 5:
            command.append("--show-prompt")
        completed_process = run_cli(command, str(question["id"]))
        if completed_process.returncode != 0:
            print(completed_process.stderr, file=sys.stderr, flush=True)
            raise RuntimeError(
                f"{question['id']}: CLI exited {completed_process.returncode}"
            )
        prompt_capture, payload = parse_final_json(completed_process.stdout)
        rows.append(
            {
                **question,
                "command": "python -m generation.cli <question> --retrieve --live --provider groq"
                + (" --show-prompt" if position <= 5 else ""),
                "prompt_capture": prompt_capture if position <= 5 else None,
                "cli_stderr": completed_process.stderr.strip(),
                "cli_payload": payload,
            }
        )
        save(rows)
        print(f"[{position:02d}/{len(questions)}] generated {question['id']}", flush=True)

    from generation.cli import build_application_retriever, retrieve_ranked_results

    retriever = build_application_retriever(
        ROOT / "data/processed/bm25_index.pkl", default_top_k=30
    )
    for row in rows:
        ranked = retrieve_ranked_results(
            str(row["question"]),
            ROOT / "data/processed/bm25_index.pkl",
            top_k=5,
            candidate_k=30,
            max_chunks_per_paper=2,
            max_chunks_per_section=1,
            retriever=retriever,
        )
        reconstructed_ids = [item.chunk_id for item in ranked]
        actual_ids = list(row["cli_payload"]["context_chunk_ids"])
        row["retrieval_reconstruction_matches_cli_context"] = reconstructed_ids == actual_ids
        row["retrieved_chunks"] = [
            {
                "rank": rank,
                "chunk_id": item.chunk_id,
                "paper_id": item.paper_id,
                "title": item.title,
                "section": item.section,
                "score": item.score,
                "source": item.source,
                "text": item.text,
            }
            for rank, item in enumerate(ranked, 1)
        ]
    FINAL.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2026-08-14",
                "command_contract": "generation.cli --retrieve --live --provider groq",
                "completion_status": "partial" if len(rows) < len(questions) else "complete",
                "completed_questions": len(rows),
                "skipped_questions": len(questions) - len(rows),
                "show_prompt_question_ids": [row["id"] for row in rows[:5]],
                "questions": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(FINAL), "questions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
