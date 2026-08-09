from __future__ import annotations

from evaluation.ragas_evaluator import _build_metrics, build_ragas_records
from retrieval.models import RetrievalResult


def test_build_ragas_records_preserves_question_answer_and_context_order():
    generation = {
        "questions": [
            {"id": "q1", "question": "Question?", "answer": "Answer [1]."}
        ]
    }
    chunks = {
        "c1": RetrievalResult("c1", "First", 1.0, "frozen"),
        "c2": RetrievalResult("c2", "Second", 0.5, "frozen"),
    }

    records = build_ragas_records(
        generation,
        context_lookup=lambda ids: [chunks[item] for item in ids],
        chunk_ids_by_question={"q1": ["c2", "c1"]},
    )

    assert records == [
        {
            "id": "q1",
            "question": "Question?",
            "answer": "Answer [1].",
            "contexts": ["Second", "First"],
        }
    ]


def test_build_ragas_records_rejects_missing_context_mapping():
    generation = {"questions": [{"id": "q1", "question": "Q", "answer": "A"}]}

    try:
        build_ragas_records(
            generation,
            context_lookup=lambda ids: [],
            chunk_ids_by_question={},
        )
    except KeyError as exc:
        assert "q1" in str(exc)
    else:
        raise AssertionError("missing context mapping should fail")


def test_answer_relevancy_uses_provider_compatible_single_generation():
    metrics = _build_metrics()
    assert [metric.name for metric in metrics] == [
        "faithfulness",
        "answer_relevancy",
        "context_utilization",
    ]
    assert metrics[1].strictness == 1


def test_ragas_prefers_the_exact_context_seen_by_generation():
    generation = {
        "questions": [
            {
                "id": "q1",
                "question": "Question?",
                "answer": "Answer [1].",
                "context_chunk_ids": ["c1"],
            }
        ]
    }
    chunks = {
        "c1": RetrievalResult("c1", "Seen", 1.0, "frozen"),
        "c2": RetrievalResult("c2", "Frozen but trimmed", 0.5, "frozen"),
    }

    records = build_ragas_records(
        generation,
        context_lookup=lambda ids: [chunks[item] for item in ids],
        chunk_ids_by_question={"q1": ["c1", "c2"]},
    )

    assert records[0]["contexts"] == ["Seen"]
