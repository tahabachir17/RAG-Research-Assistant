from __future__ import annotations

import json

from evaluation.generation_eval_checkpoint import (
    GenerationEvalCheckpoint,
    compatible_run_signature,
)


def test_partial_checkpoint_skips_completed_pairs_and_retries_failed(tmp_path):
    path = tmp_path / "generation_eval_run.json"
    checkpoint = GenerationEvalCheckpoint.create(
        path,
        {
            "questions": [{"id": "q1"}],
            "ragas": {"questions": []},
            "metric_progress": {},
        },
    )
    checkpoint.record_metric("q1", "faithfulness", status="completed", value=0.9)
    checkpoint.record_metric("q1", "answer_relevancy", status="failed", reason="timeout")

    resumed = GenerationEvalCheckpoint.load(path)
    assert resumed.completed_question_ids() == {"q1"}
    assert resumed.metric_completed("q1", "faithfulness") is True
    assert resumed.metric_completed("q1", "answer_relevancy") is False


def test_checkpoint_write_is_promoted_without_part_file(tmp_path):
    path = tmp_path / "generation_eval_run.json"
    checkpoint = GenerationEvalCheckpoint.create(path, {"questions": []})
    checkpoint.record_question({"id": "q1"})

    assert json.loads(path.read_text(encoding="utf-8"))["questions"] == [{"id": "q1"}]
    assert not path.with_suffix(".json.part").exists()


def test_replacing_answer_invalidates_all_derived_metrics(tmp_path):
    path = tmp_path / "generation_eval_run.json"
    checkpoint = GenerationEvalCheckpoint.create(
        path,
        {
            "questions": [{"id": "q1", "answer": "old"}],
            "ragas": {"questions": []},
            "metric_progress": {},
        },
    )
    checkpoint.record_metric("q1", "faithfulness", status="completed", value=0.5)
    checkpoint.record_metric(
        "q1", "context_precision", status="unavailable", reason="no reference"
    )

    checkpoint.record_question({"id": "q1", "answer": "new"})

    resumed = GenerationEvalCheckpoint.load(path)
    assert resumed.payload["metric_progress"].get("q1") is None
    assert resumed.payload["ragas"]["questions"] == []


def test_resume_signature_allows_only_structured_parser_hash_change():
    stored = {
        "dataset_sha256": "dataset",
        "model": "model",
        "provenance": {
            "prompt_sha256": "prompt",
            "structured_contract_sha256": "old",
        },
    }
    current = {
        "dataset_sha256": "dataset",
        "model": "model",
        "provenance": {
            "prompt_sha256": "prompt",
            "structured_contract_sha256": "new",
        },
    }
    assert compatible_run_signature(stored, current)
    current["model"] = "different"
    assert not compatible_run_signature(stored, current)
