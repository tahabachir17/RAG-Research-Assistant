from __future__ import annotations

import json
from pathlib import Path

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
        (
            "how do QRNNs combine the useful properties of convolutional and recurrent sequence models?",
            "mechanism",
        ),
        (
            "In 'Quasi-Recurrent Neural Networks', how do QRNNs combine the useful properties of convolutional and recurrent sequence models?",
            "mechanism",
        ),
        ("Why is a QRNN better than an LSTM?", "causes_evidence"),
        ("What are the limitations and future work?", "limitations_future_work"),
        ("Compare QRNNs and LSTMs.", "comparison"),
        (
            "Compare 'Paper A' and 'Paper B', including limitations and future work.",
            "comparison",
        ),
    ],
)
def test_question_type_classifier(question, expected):
    assert classify_question_type(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("In 'A Paper', how do X and Y combine?", "mechanism"),
        ("In 'A Paper', what does X execute?", "direct_fact"),
        ("In 'A Paper', why does X fail?", "causes_evidence"),
        (
            "In 'A Paper', what are the limitations and future improvements?",
            "limitations_future_work",
        ),
        ("In 'A Paper', how does X compare to Y?", "comparison"),
    ],
)
def test_title_prefixed_question_type_classifier(question, expected):
    assert classify_question_type(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("According to 'A Paper', why does X fail?", "causes_evidence"),
        (
            "According to the paper 'A Paper', how much faster is X?",
            "direct_fact",
        ),
        ("As described in 'A Paper', how do X and Y combine?", "mechanism"),
        ("Which mechanisms model local and global structure?", "mechanism"),
    ],
)
def test_general_attribution_and_plural_classifier(question, expected):
    assert classify_question_type(question) == expected


def test_all_controlled_questions_have_expected_answer_structure():
    payload = json.loads(
        Path("evaluation/data/controlled_generation_qa.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "controlled-submodular-01": "direct_fact",
        "controlled-submodular-02": "causes_evidence",
        "controlled-enquirer-01": "direct_fact",
        "controlled-enquirer-02": "mechanism",
        "controlled-qrnn-01": "direct_fact",
        "controlled-qrnn-02": "mechanism",
        "controlled-nsm-01": "direct_fact",
        "controlled-nsm-02": "mechanism",
        "controlled-video-01": "mechanism",
        "controlled-video-02": "causes_evidence",
        "controlled-reddit-01": "direct_fact",
        "controlled-reddit-02": "limitations_future_work",
    }

    assert {
        row["id"]: classify_question_type(row["question"])
        for row in payload["questions"]
    } == expected


def test_each_question_type_has_a_distinct_fragment():
    assert len(QUESTION_TYPE_INSTRUCTIONS) == 5
    assert len(set(QUESTION_TYPE_INSTRUCTIONS.values())) == 5


def test_technical_term_requirement_is_scoped_to_mechanism_questions():
    mechanism = QUESTION_TYPE_INSTRUCTIONS["mechanism"]
    assert "specific mechanism with a technical term" in mechanism
    assert "name each one individually" in mechanism
    assert all(
        "specific mechanism with a technical term" not in instruction
        for question_type, instruction in QUESTION_TYPE_INSTRUCTIONS.items()
        if question_type != "mechanism"
    )


def test_supported_qualifier_requirement_is_scoped_to_direct_fact_questions():
    direct_fact = QUESTION_TYPE_INSTRUCTIONS["direct_fact"]
    assert "qualifying detail" in direct_fact
    assert "condition, scope, or comparison" in direct_fact
    assert all(
        "qualifying detail" not in instruction
        for question_type, instruction in QUESTION_TYPE_INSTRUCTIONS.items()
        if question_type != "direct_fact"
    )


def test_process_completeness_requirement_is_scoped_to_mechanism_questions():
    mechanism = QUESTION_TYPE_INSTRUCTIONS["mechanism"]
    assert "intermediate results" in mechanism
    assert "what objective is optimized" in mechanism
    assert all(
        "intermediate results" not in instruction
        for question_type, instruction in QUESTION_TYPE_INSTRUCTIONS.items()
        if question_type != "mechanism"
    )


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
