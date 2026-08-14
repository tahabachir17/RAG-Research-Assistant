from __future__ import annotations

from evaluation.generation_golden import load_generation_golden


def test_candidate_golden_has_20_hard_questions_but_cannot_gate_before_human_review():
    questions = load_generation_golden("evaluation/data/golden_generation_qa.json")
    assert len(questions) == 20
    assert all(not question.reviewed for question in questions)
    assert all(question.paper_id for question in questions)
    assert all(question.retrieved_chunk_ids for question in questions)
    import pytest
    with pytest.raises(ValueError, match="unreviewed"):
        load_generation_golden("evaluation/data/golden_generation_qa.json", require_reviewed=True)


def test_controlled_generation_golden_has_reviewed_exact_evidence_and_references():
    questions = load_generation_golden(
        "evaluation/data/controlled_generation_qa.json", require_reviewed=True
    )

    assert len(questions) == 12
    assert all(question.reference_answer for question in questions)
    assert all(1 <= len(question.required_concepts) <= 5 for question in questions)
    assert all(
        question.retrieved_chunk_ids == question.reference_context_ids
        for question in questions
    )
