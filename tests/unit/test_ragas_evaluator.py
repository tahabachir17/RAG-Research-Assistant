from __future__ import annotations

from config import Settings
from evaluation.ragas_evaluator import (
    _build_metrics,
    _ragas_llm_options,
    build_ragas_records,
)
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


def test_ragas_qwen_uses_local_lmstudio_endpoint():
    settings = Settings(
        _env_file=None,
        JUDGE_PROVIDER="qwen",
        JUDGE_MODEL="qwen-local",
        LLM_REQUEST_TIMEOUT_SECONDS=90,
    )
    options = _ragas_llm_options(settings)
    assert options["model"] == "qwen-local"
    assert options["base_url"] == settings.LMSTUDIO_BASE_URL
    assert options["timeout"] == 90


def test_ragas_gemini_preserves_existing_openai_key_fallback():
    settings = Settings(
        _env_file=None,
        JUDGE_PROVIDER="gemini",
        JUDGE_MODEL="gemini-3.5-flash-lite",
        GEMINI_API_KEY=None,
        OPENAI_API_KEY="existing-gemini-key",
        LLM_REQUEST_TIMEOUT_SECONDS=20,
    )
    options = _ragas_llm_options(settings)
    assert options["api_key"] == "existing-gemini-key"
    assert options["base_url"] == settings.GEMINI_BASE_URL
    assert "temperature" not in options
