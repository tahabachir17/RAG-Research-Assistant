from __future__ import annotations

import json

from evaluation.generation_eval_checkpoint import GenerationEvalCheckpoint


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
