from __future__ import annotations

import json

import pytest

from evaluation.generation_golden import load_generation_golden
from evaluation.generation_metrics import concept_recall, concept_recall_details


def test_concept_recall_full_coverage_ignores_case_and_punctuation():
    answer = "Parallel convolutional computation; recurrent pooling for long distance context."

    assert concept_recall(
        answer,
        [
            "parallel convolutional computation",
            "recurrent pooling",
            "long-distance context",
        ],
    ) == 1.0


def test_concept_recall_partial_coverage():
    assert concept_recall(
        "QRNNs use recurrent pooling.",
        ["parallel convolutional computation", "recurrent pooling"],
    ) == 0.5


def test_concept_recall_zero_coverage_and_no_requirements():
    assert concept_recall("A vague answer.", ["recurrent pooling"]) == 0.0
    assert concept_recall("Anything.", []) is None


def test_concept_recall_credits_bounded_alias_with_auditable_details():
    required = [
        {
            "concept": "parallel convolutional computation",
            "aliases": [
                "convolutional aspects computed in parallel",
                "computed in parallel (e.g., convolutionally)",
                "convolutional computation",
            ],
        },
        "recurrent pooling",
        "long-distance context",
    ]
    answer = (
        "QRNNs use convolutional aspects computed in parallel, recurrent pooling, "
        "and long-distance context."
    )

    assert concept_recall(answer, required) == 1.0
    assert concept_recall_details(answer, required) == [
        {
            "concept": "parallel convolutional computation",
            "matched": True,
            "matched_by": "alias",
            "matched_phrase": "convolutional aspects computed in parallel",
        },
        {
            "concept": "recurrent pooling",
            "matched": True,
            "matched_by": "primary",
            "matched_phrase": "recurrent pooling",
        },
        {
            "concept": "long-distance context",
            "matched": True,
            "matched_by": "primary",
            "matched_phrase": "long-distance context",
        },
    ]


def test_aliases_do_not_credit_generic_summary():
    required = [
        {
            "concept": "parallel convolutional computation",
            "aliases": ["convolutional aspects computed in parallel"],
        },
        "recurrent pooling",
        "long-distance context",
    ]

    assert concept_recall("QRNNs exploit parallelism and context.", required) == 0.0


def test_full_set_aliases_do_not_credit_missing_concepts_in_incomplete_answers():
    questions = {
        question.id: question
        for question in load_generation_golden(
            "evaluation/data/controlled_generation_qa.json", require_reviewed=True
        )
    }
    cases = {
        "controlled-qrnn-02": (
            "QRNNs exploit parallelism and context.",
            0.0,
            {
                "parallel convolutional computation",
                "recurrent pooling",
                "long-distance context",
            },
        ),
        "controlled-enquirer-02": (
            "The model handles multi-step compositional queries without predefined "
            "logical operations by stacking several executors, which learn the "
            "logic of operations via end-to-end training using Query-Answer pairs.",
            0.5,
            {
                "operations conditioned on the query",
                "intermediate table annotations in layered memory",
            },
        ),
        "controlled-video-02": (
            "Averaging ignores the temporal structure and loses the order of "
            "appearances of objects.",
            2 / 3,
            {"fuses temporally distinct events and objects"},
        ),
    }

    for question_id, (answer, expected_score, absent_concepts) in cases.items():
        required = questions[question_id].required_concepts
        details = concept_recall_details(answer, required)
        assert concept_recall(answer, required) == expected_score
        assert {
            row["concept"] for row in details if not row["matched"]
        }.issuperset(absent_concepts)


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


def test_structured_required_concept_loads_aliases(tmp_path):
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
                            {
                                "concept": "parallel convolutional computation",
                                "aliases": [
                                    "convolutional aspects computed in parallel"
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_generation_golden(path)[0].required_concepts == [
        {
            "concept": "parallel convolutional computation",
            "aliases": ["convolutional aspects computed in parallel"],
        }
    ]


def test_required_concept_aliases_are_bounded(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "questions": [
                    {
                        "id": "q1",
                        "question": "How does it work?",
                        "retrieved_chunk_ids": ["chunk-1"],
                        "required_concepts": [
                            {
                                "concept": "mechanism",
                                "aliases": [f"alias {index}" for index in range(6)],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at most 5 aliases"):
        load_generation_golden(path)
