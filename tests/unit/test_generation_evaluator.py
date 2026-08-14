from __future__ import annotations

from evaluation.generation_evaluator import (
    GenerationEvaluator,
    _claims,
    save_generation_outputs,
)
from evaluation.generation_golden import GenerationGoldenQuestion
from generation.llm_client import LLMCompletion
from retrieval.models import RetrievalResult


class FakeLLM:
    def complete(self, system, user, stream=False):
        return LLMCompletion(
            '{"items":[{'
            '"pipeline_stage":{"text":"retrieval","citations":[1]},'
            '"selection_mechanism":{"text":"filter","citations":[1]},'
            '"dataset":{"text":"test set","citations":[1]},'
            '"benefit":{"text":"gain","citations":[1]},'
            '"limitations":{"text":"cost","citations":[1]}}]}',
            "stop",
        )


def test_evaluator_uses_runtime_path_and_writes_all_artifacts(tmp_path):
    question = GenerationGoldenQuestion("q1", "Compare.", ["c1"], ["filter"], {}, ["pipeline_stage", "selection_mechanism", "dataset", "benefit", "limitations"], 3, True, [])
    chunk = RetrievalResult("c1", "Filtering improved retrieval but added cost.", 1.0, "frozen", paper_id="p1")
    result = GenerationEvaluator(llm=FakeLLM(), chunk_lookup=lambda ids: [chunk], provider="fake", model="answerer", max_retries=1).evaluate([question])
    assert result["aggregate"]["citation_validity_rate"] == 1.0
    assert result["aggregate"]["required_field_completeness"] == 1.0
    assert result["questions"][0]["context_chunk_ids"] == ["c1"]
    paths = save_generation_outputs(result, tmp_path)
    assert set(paths) == {"json", "csv", "markdown"}
    assert all(path.is_file() for path in paths.values())


def test_save_outputs_reuses_explicit_checkpoint_stamp(tmp_path):
    question = GenerationGoldenQuestion("q1", "Compare.", ["c1"], [], {}, [], None, True, [])
    chunk = RetrievalResult("c1", "Evidence.", 1.0, "frozen")

    class PlainLLM:
        def complete_json(self, system, user):
            return LLMCompletion('{"answer_status":"answered","summary":"","claims":[{"text":"Evidence.","citations":[1]}]}', "stop")

    result = GenerationEvaluator(
        llm=PlainLLM(),
        chunk_lookup=lambda ids: [chunk],
        provider="fake",
        model="answerer",
    ).evaluate([question])
    first = save_generation_outputs(result, tmp_path, stamp="checkpoint")
    result["ragas"] = {"status": "completed", "aggregate": {}, "questions": []}
    second = save_generation_outputs(result, tmp_path, stamp="checkpoint")
    assert first == second
    assert '"ragas"' in first["json"].read_text(encoding="utf-8")


def test_structured_table_cells_become_atomic_claims_without_header_or_abstentions():
    structured = {
        "answer_status": "answered",
        "summary": "",
        "items": [
            {
                "method": {"text": "ScaLearn scales adapter outputs.", "citations": [1]},
                "evaluation": {
                    "text": "Evaluated on GLUE and SuperGLUE.",
                    "citations": [1, 2],
                },
                "limitations": {
                    "text": "Not reported in the supplied passages.",
                    "citations": [],
                },
            }
        ],
    }
    claims = _claims("| method | evaluation | limitations |", structured)
    assert claims == [
        {
            "subject_id": "claim-1:method",
            "field": "method",
            "text": "ScaLearn scales adapter outputs.",
            "citations": [1],
            "cited": True,
        },
        {
            "subject_id": "claim-1:evaluation",
            "field": "evaluation",
            "text": "Evaluated on GLUE and SuperGLUE.",
            "citations": [1, 2],
            "cited": True,
        },
    ]


def test_mechanism_question_packs_adjacent_evidence_for_multiple_concepts():
    gold = RetrievalResult(
        "gold",
        "Features are computed convolutionally in parallel.",
        1.0,
        "frozen",
        paper_id="p1",
        section="conclusion",
    )
    neighbor = RetrievalResult(
        "neighbor",
        "The recurrent pooling layer preserves context.",
        1.0,
        "frozen",
        paper_id="p1",
        section="methodology",
    )
    question = GenerationGoldenQuestion(
        id="qrnn",
        question="How do QRNNs combine convolutional and recurrent properties?",
        retrieved_chunk_ids=["gold"],
        expected_qualifying_items=[],
        excluded_items={},
        required_fields=[],
        max_items=None,
        reviewed=True,
        calibration_verdicts=[],
        required_concepts=["parallel convolution", "recurrent pooling"],
    )

    class MechanismLLM:
        def complete_json(self, system, user):
            assert "Features are computed convolutionally in parallel." in user
            assert "The recurrent pooling layer preserves context." in user
            return LLMCompletion(
                '{"answer_status":"answered","summary":"","claims":['
                '{"text":"QRNNs use parallel convolution and recurrent pooling.",'
                '"citations":[1,2]}]}',
                "stop",
            )

    result = GenerationEvaluator(
        llm=MechanismLLM(),
        chunk_lookup=lambda ids: [gold],
        provider="fake",
        model="answerer",
        evidence_packing_mode="adjacent",
        adjacent_chunk_lookup=lambda chunk: [neighbor],
    ).evaluate([question])

    assert result["questions"][0]["context_chunk_ids"] == ["gold", "neighbor"]
    assert [
        row["matched_by"]
        for row in result["questions"][0]["concept_recall_details"]
    ] == ["primary", "primary"]
