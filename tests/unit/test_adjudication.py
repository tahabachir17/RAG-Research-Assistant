from __future__ import annotations

import pytest

from evaluation.adjudication import (
    AdjudicationConfig,
    adjudicate_disputed_case,
    adjudication_reasons,
)


def _config():
    return AdjudicationConfig(enabled=True)


def test_adjudication_triggers_for_low_correctness():
    assert "correctness_below_threshold" in adjudication_reasons(
        primary_correctness=0.70,
        concept_recall=0.20,
        answer_relevancy=0.70,
        config=_config(),
    )


def test_adjudication_triggers_for_high_coverage_low_correctness():
    assert "high_concept_recall_low_correctness" in adjudication_reasons(
        primary_correctness=0.60,
        concept_recall=1.0,
        answer_relevancy=0.60,
        config=_config(),
    )


def test_adjudication_triggers_for_correctness_relevancy_disagreement():
    assert adjudication_reasons(
        primary_correctness=0.90,
        concept_recall=0.50,
        answer_relevancy=0.30,
        config=_config(),
    ) == ["correctness_relevancy_disagreement"]


def test_adjudication_is_disabled_or_skipped_when_no_condition_applies():
    assert adjudication_reasons(
        primary_correctness=0.90,
        concept_recall=0.50,
        answer_relevancy=0.80,
        config=_config(),
    ) == []
    assert adjudication_reasons(
        primary_correctness=0.10,
        concept_recall=1.0,
        answer_relevancy=1.0,
        config=AdjudicationConfig(enabled=False),
    ) == []


def test_secondary_score_never_overwrites_primary_and_flags_disagreement():
    result = adjudicate_disputed_case(
        primary_correctness=0.50,
        concept_recall=1.0,
        answer_relevancy=0.90,
        config=_config(),
        secondary_judge=lambda: 0.85,
        generator_model="llama-3.3-70b-versatile",
        primary_judge_model="llama-3.1-8b-instant",
        secondary_judge_model="gemini-3.5-flash-lite",
    )

    assert result["primary_correctness"] == 0.50
    assert result["secondary_correctness"] == 0.85
    assert result["disagreement"] is True


def test_secondary_judge_cannot_be_the_generator_or_primary_judge():
    common = dict(
        primary_correctness=0.50,
        concept_recall=1.0,
        answer_relevancy=0.90,
        config=_config(),
        secondary_judge=lambda: 0.85,
        generator_model="generator-70b",
        primary_judge_model="primary-8b",
    )
    with pytest.raises(ValueError, match="separate from generator"):
        adjudicate_disputed_case(**common, secondary_judge_model="generator-70b")
    with pytest.raises(ValueError, match="differ from primary"):
        adjudicate_disputed_case(**common, secondary_judge_model="primary-8b")
