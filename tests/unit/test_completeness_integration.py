from __future__ import annotations

from evaluation.generation_metrics import concept_recall
from evaluation.layered_reporting import (
    build_evaluation_layers,
    render_layered_sections,
)
from generation.cli import run_generation
from generation.llm_client import LLMCompletion
from retrieval.models import RetrievalResult


class CompleteQrnnLLM:
    def __init__(self):
        self.system = ""
        self.user = ""

    def complete_json(self, system, user):
        self.system, self.user = system, user
        return LLMCompletion(
            '{"answer_status":"answered","summary":"","claims":['
            '{"text":"QRNNs use parallel convolutional computation.","citations":[1]},'
            '{"text":"They use recurrent pooling for long-distance context and sequence order.","citations":[1]}]}',
            "stop",
        )


def test_prompt_to_generation_to_concept_score_to_layered_report():
    llm = CompleteQrnnLLM()
    chunk = RetrievalResult(
        "qrnn-gold",
        (
            "QRNNs use parallel convolutional computation and recurrent pooling "
            "for long-distance context and sequence order."
        ),
        1.0,
        "frozen",
        paper_id="1611.01576v2",
        section="method",
    )
    required = [
        "parallel convolutional computation",
        "recurrent pooling",
        "long-distance context",
        "sequence order",
    ]

    generated = run_generation(
        "How do QRNNs combine convolutional and recurrent sequence models?",
        [chunk],
        llm=llm,
        max_retries=0,
        enable_faithfulness_verifier=False,
    )
    recall = concept_recall(generated.answer, required)
    payload = {
        "questions": [
            {
                "id": "qrnn-02",
                "reference_context_ids": ["qrnn-gold"],
                "retrieved_chunk_ids": ["qrnn-gold"],
                "concept_recall": recall,
            }
        ],
        "ragas": {
            "questions": [
                {
                    "id": "qrnn-02",
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                    "answer_correctness": 0.90,
                    "faithfulness": 1.0,
                }
            ]
        },
    }
    report = render_layered_sections(build_evaluation_layers(payload))

    assert "COMPLETENESS CONTRACT:" in llm.system
    assert "Question type: mechanism" in llm.user
    assert "parallel convolutional computation" in generated.answer
    assert "recurrent pooling" in generated.answer
    assert recall == 1.0
    assert "## Retrieval" in report
    assert "## Controlled generation" in report
    assert "## End-to-end RAG" in report
