from __future__ import annotations

import pytest

from generation.prompt_manager import (
    PromptManager,
    compound_question_instruction,
    detect_question_parts,
)


def test_prompt_manager_loads_caches_and_renders_strict_variables(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_text(
        "system: 'Use {{ context }}'\nuser: 'Question: {{ question }}'\n",
        encoding="utf-8",
    )
    manager = PromptManager(tmp_path)

    loaded = manager.load("sample.yaml")
    system, user = manager.render("sample", context="evidence", question="why?")

    assert manager.load("sample") is loaded
    assert system == "Use evidence"
    assert user == "Question: why?"
    assert loaded.to_dict()["name"] == "sample"


def test_prompt_manager_rejects_missing_files_variables_and_malformed_templates(
    tmp_path,
):
    manager = PromptManager(tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        manager.load("missing")

    (tmp_path / "bad.yaml").write_text("system: ok\n", encoding="utf-8")
    with pytest.raises(ValueError, match="user"):
        manager.load("bad")

    (tmp_path / "required.yaml").write_text(
        "system: '{{ context }}'\nuser: '{{ question }}'\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="question"):
        manager.render("required", context="x")
    (tmp_path / "syntax.yaml").write_text(
        "system: '{{ broken'\nuser: ok\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Jinja syntax"):
        manager.render("syntax")

    with pytest.raises(ValueError, match="template_name"):
        manager.load("../escape")


def test_repository_prompt_templates_render():
    manager = PromptManager()
    system, user = manager.render("qa_prompt", context="[1] evidence", question="Q")
    assert "ONLY" in system
    assert "Preserve the scope" in system
    assert "paper title alone" in system
    assert "[1] evidence" in user
    assert "Question: Q" in user
    assert "direct answer in the first sentence" in system


def test_detect_question_parts_handles_multiple_marks_and_question_stems():
    assert detect_question_parts("What method is used? Why does it help?") == [
        "What method is used",
        "Why does it help",
    ]
    assert detect_question_parts(
        "What method is used and how is it evaluated?"
    ) == ["What method is used", "how is it evaluated"]


def test_detect_question_parts_does_not_split_simple_coordinated_nouns():
    assert detect_question_parts("Compare BM25 and dense retrieval.") == [
        "Compare BM25 and dense retrieval."
    ]
    assert compound_question_instruction("What is BM25?") == ""


def test_detect_question_parts_does_not_split_quoted_paper_list():
    question = "Compare 'Paper A' and 'Paper B' and explain their future work."

    assert detect_question_parts(question) == [question.strip("?")]


def test_compound_question_instruction_lists_each_detected_part():
    instruction = compound_question_instruction(
        "Which model is used? How was it evaluated?"
    )
    assert "Part 1: Which model is used" in instruction
    assert "Part 2: How was it evaluated" in instruction
