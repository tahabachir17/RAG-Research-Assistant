from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.pydantic_v1 import BaseModel

from config import Settings
from evaluation.ragas_evaluator import (
    _batches,
    _build_metrics,
    _install_ragas_schema_guard,
    _ragas_llm_options,
    _validate_ragas_payload,
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


def test_ragas_projects_structured_table_to_factual_narrative():
    generation = {
        "questions": [
            {
                "id": "q1",
                "question": "Question?",
                "answer": "| method | limitations |\n| --- | --- |",
                "structured_data": {
                    "items": [
                        {
                            "method": {"text": "Scales adapter outputs", "citations": [1]},
                            "limitations": {
                                "text": "Not reported in the supplied passages.",
                                "citations": [],
                            },
                        }
                    ]
                },
            }
        ]
    }
    records = build_ragas_records(
        generation,
        context_lookup=lambda ids: [RetrievalResult("c1", "Seen", 1.0, "frozen")],
        chunk_ids_by_question={"q1": ["c1"]},
    )

    assert records[0]["answer"] == "Contribution 1. Method: Scales adapter outputs."


def test_ragas_projects_legacy_markdown_table_to_factual_narrative():
    generation = {
        "questions": [
            {
                "id": "q1",
                "question": "Question?",
                "answer": (
                    "| problem | method | limitations |\n"
                    "| --- | --- | --- |\n"
                    "| Gradient instability [1] | Scaling [1] | "
                    "Not reported in the supplied passages. |"
                ),
                "structured_data": None,
            }
        ]
    }
    records = build_ragas_records(
        generation,
        context_lookup=lambda ids: [RetrievalResult("c1", "Seen", 1.0, "frozen")],
        chunk_ids_by_question={"q1": ["c1"]},
    )

    assert records[0]["answer"] == (
        "Contribution 1. Problem: Gradient instability [1]. Method: Scaling [1]."
    )


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


class _AnswerSchema(BaseModel):
    question: str
    noncommittal: int


def test_ragas_schema_guard_rejects_valid_json_with_wrong_top_level_keys():
    with pytest.raises(ValueError, match="top-level keys"):
        _validate_ragas_payload(
            {"analysis": [{"question": "Q", "noncommittal": 0}]},
            _AnswerSchema,
        )


def test_ragas_schema_guard_repairs_schema_invalid_json_once():
    from ragas.llms.output_parser import RagasoutputParser

    _install_ragas_schema_guard()
    parser = RagasoutputParser(pydantic_object=_AnswerSchema)

    class LLM:
        calls = 0

        async def generate(self, prompt):
            self.calls += 1
            return SimpleNamespace(
                generations=[
                    [SimpleNamespace(text='{"question":"Repaired?","noncommittal":0}')]
                ]
            )

    llm = LLM()
    prompt = SimpleNamespace(to_string=lambda: "original prompt")
    parsed = asyncio.run(
        parser.aparse(
            '{"analysis":[{"question":"Wrong","noncommittal":0}]}',
            prompt,
            llm,
            max_retries=1,
        )
    )
    assert parsed.question == "Repaired?"
    assert llm.calls == 1


def test_ragas_schema_guard_raises_after_repair_exhaustion():
    from ragas.llms.output_parser import RagasoutputParser

    _install_ragas_schema_guard()
    parser = RagasoutputParser(pydantic_object=_AnswerSchema)
    prompt = SimpleNamespace(to_string=lambda: "original prompt")

    with pytest.raises(ValueError, match="schema validation"):
        asyncio.run(
            parser.aparse(
                '{"analysis":[]}',
                prompt,
                SimpleNamespace(),
                max_retries=0,
            )
        )


def test_faithfulness_batches_bound_per_call_output_size():
    assert _batches(list(range(11)), 4) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10],
    ]
