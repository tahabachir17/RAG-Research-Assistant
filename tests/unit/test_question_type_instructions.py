from __future__ import annotations

import pytest

from generation.prompt_manager import (
    QUESTION_TYPE_INSTRUCTIONS,
    PromptManager,
    classify_question_type,
    question_type_instruction,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is a quasi-recurrent neural network?", "direct_fact"),
        ("How does a QRNN work?", "mechanism"),
        ("Why is a QRNN better than an LSTM?", "causes_evidence"),
        ("What are the limitations and future work?", "limitations_future_work"),
        ("Compare QRNNs and LSTMs.", "comparison"),
    ],
)
def test_question_type_classifier(question, expected):
    assert classify_question_type(question) == expected


def test_each_question_type_has_a_distinct_fragment():
    assert len(QUESTION_TYPE_INSTRUCTIONS) == 5
    assert len(set(QUESTION_TYPE_INSTRUCTIONS.values())) == 5


def test_question_type_fragment_is_injected_through_jinja_template():
    fragment = question_type_instruction("How does a QRNN work?")

    _, user = PromptManager().render(
        "qa_prompt",
        context="[1] Evidence.",
        question="How does a QRNN work?",
        question_type_instruction=fragment,
    )

    assert fragment in user
    assert "Enumerate each supported mechanism" in user
