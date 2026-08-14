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
    _parse_ragas_payload,
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
    assert options["max_retries"] == 0
    assert options["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_ragas_groq_requests_json_object_mode():
    settings = Settings(
        _env_file=None,
        JUDGE_PROVIDER="groq",
        JUDGE_MODEL="llama-3.1-8b-instant",
        GROQ_API_KEY="test-key",
    )
    options = _ragas_llm_options(settings)
    assert options["model_kwargs"] == {
        "response_format": {"type": "json_object"}
    }


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


def test_ragas_parser_extracts_json_from_prose_and_trailing_text():
    payload = _parse_ragas_payload(
        'Result:\n{"question":"Relevant?","noncommittal":0}\nDone.',
        _AnswerSchema,
    )

    assert payload == {"question": "Relevant?", "noncommittal": 0}


def test_ragas_parser_unwraps_schema_valid_array():
    from ragas.metrics._faithfulness import _statements_output_parser

    payload = _parse_ragas_payload(
        '{"analysis":[{"sentence_index":0,"simpler_statements":["A claim."]}]}',
        _statements_output_parser.pydantic_object,
    )

    assert payload == [{"sentence_index": 0, "simpler_statements": ["A claim."]}]


def test_ragas_parser_unwraps_deeply_nested_schema_valid_array():
    from ragas.metrics._faithfulness import _faithfulness_output_parser

    payload = _parse_ragas_payload(
        '{"analysis":{"statements":[{"statement":"A.","reason":"Supported.","verdict":1}]}}',
        _faithfulness_output_parser.pydantic_object,
    )

    assert payload[0]["verdict"] == 1


def test_ragas_parser_wraps_one_valid_array_item():
    from ragas.metrics._faithfulness import _faithfulness_output_parser

    payload = _parse_ragas_payload(
        '{"statement":"A.","reason":"Supported.","verdict":1}',
        _faithfulness_output_parser.pydantic_object,
    )

    assert payload == [
        {"statement": "A.", "reason": "Supported.", "verdict": 1}
    ]


def test_ragas_parser_wraps_one_valid_array_item_inside_wrapper():
    from ragas.metrics._faithfulness import _faithfulness_output_parser

    payload = _parse_ragas_payload(
        '{"analysis":{"statement":"A.","reason":"Supported.","verdict":1}}',
        _faithfulness_output_parser.pydantic_object,
    )

    assert payload[0]["statement"] == "A."


def test_ragas_parser_skips_schema_invalid_json_before_valid_payload():
    payload = _parse_ragas_payload(
        '{"analysis":[]}\n{"question":"Recovered?","noncommittal":0}',
        _AnswerSchema,
    )

    assert payload == {"question": "Recovered?", "noncommittal": 0}


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


def test_chunked_faithfulness_uses_normalized_sentences_as_claims(monkeypatch):
    from evaluation.ragas_evaluator import _chunked_faithfulness_ascore

    class Segmenter:
        def segment(self, answer):
            return ["First claim.", "Second claim."]

    class Parsed:
        def dicts(self):
            return [
                {"statement": "First claim.", "reason": "supported", "verdict": 1},
                {"statement": "Second claim.", "reason": "supported", "verdict": 1},
            ]

    class LLM:
        calls = 0

        async def generate(self, prompt, **kwargs):
            self.calls += 1
            return SimpleNamespace(generations=[[SimpleNamespace(text="[]")]])

    metric = SimpleNamespace(
        llm=LLM(),
        sentence_segmenter=Segmenter(),
        _reproducibility=1,
        max_retries=0,
        _create_nli_prompt=lambda row, statements: SimpleNamespace(
            to_string=lambda: str(statements)
        ),
    )
    from ragas.metrics import _faithfulness

    async def parse(*args, **kwargs):
        return Parsed()

    monkeypatch.setattr(
        type(_faithfulness._faithfulness_output_parser), "aparse", parse
    )
    score = asyncio.run(
        _chunked_faithfulness_ascore(
            metric,
            {"question": "Q?", "answer": "First claim. Second claim.", "contexts": ["C"]},
            None,
        )
    )

    assert score == 1.0
    assert metric.llm.calls == 2


def test_evaluate_with_ragas_surfaces_worker_failures(monkeypatch):
    import ragas

    from evaluation.ragas_evaluator import evaluate_with_ragas

    captured = {}

    def fail_evaluate(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("judge output was invalid")

    monkeypatch.setattr(ragas, "evaluate", fail_evaluate)
    with pytest.raises(RuntimeError, match="judge output was invalid"):
        evaluate_with_ragas(
            [{"id": "q1", "question": "Q?", "answer": "A.", "contexts": ["C."]}],
            llm=SimpleNamespace(),
            embeddings=SimpleNamespace(),
            metric_names=["faithfulness"],
        )

    assert captured["raise_exceptions"] is True
