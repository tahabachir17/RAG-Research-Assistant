from __future__ import annotations

import json

from evaluation.generation_golden import load_generation_golden
from evaluation.generation_metrics import concept_recall


def test_concept_recall_full_coverage_ignores_case_and_punctuation():
    answer = "Parallel convolutional computation; recurrent pooling."

    assert concept_recall(
        answer, ["parallel convolutional computation", "recurrent pooling"]
    ) == 1.0


def test_concept_recall_partial_coverage():
    assert concept_recall(
        "QRNNs use recurrent pooling.",
        ["parallel convolutional computation", "recurrent pooling"],
    ) == 0.5


def test_concept_recall_zero_coverage_and_no_requirements():
    assert concept_recall("A vague answer.", ["recurrent pooling"]) == 0.0
    assert concept_recall("Anything.", []) is None


def test_required_concepts_load_from_golden_schema(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "questions": [
                    {
                        "id": "qrnn-02",
                        "question": "How do QRNNs work?",
                        "retrieved_chunk_ids": ["chunk-1"],
                        "expected_qualifying_items": [],
                        "excluded_items": {},
                        "required_fields": [],
                        "max_items": None,
                        "reviewed": True,
                        "calibration_verdicts": [],
                        "required_concepts": [
                            "parallel convolutional computation",
                            "recurrent pooling",
                            "long-distance context",
                            "sequence order",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    question = load_generation_golden(path)[0]

    assert question.required_concepts == [
        "parallel convolutional computation",
        "recurrent pooling",
        "long-distance context",
        "sequence order",
    ]
