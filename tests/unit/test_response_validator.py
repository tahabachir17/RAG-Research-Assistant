from __future__ import annotations

from generation.context_assembler import CitationSource
from generation.llm_client import LLMCompletion
from generation.response_validator import ResponseValidator, generate_with_validation


def _map():
    return {1: CitationSource(1, "p1", "c1", "Paper", [], 2024, "results", None)}


def test_validator_reports_granular_runtime_failures():
    result = ResponseValidator(_map()).validate("Claim 【2†L1-L8】 [9].", finish_reason="length")
    assert result.valid is False
    assert result.failures == ["truncated", "citation_out_of_range", "unsupported_citation_format"]


def test_validator_enforces_fields_complete_tables_and_max_items():
    answer = "| pipeline stage | benefit |\n|---|---|\n| retrieval | |\n| generation | gain |"
    result = ResponseValidator(_map(), required_fields=["pipeline_stage", "dataset"], max_items=1).validate(answer + "\n[1]")
    assert "missing_required_field:dataset" in result.failures
    assert "incomplete_table" in result.failures
    assert "too_many_items" in result.failures


def test_empty_answer_without_context_is_allowed():
    assert ResponseValidator({}).validate("").valid is True


def test_generation_repairs_once_with_specific_failures():
    class Fake:
        def __init__(self):
            self.calls = []
            self.responses = iter([LLMCompletion("uncited", "length"), LLMCompletion("fixed [1]", "stop")])

        def complete(self, system, user, stream=False):
            self.calls.append(user)
            return next(self.responses)

    fake = Fake()
    result = generate_with_validation(fake, "system", "user", ResponseValidator(_map()), max_retries=1)
    assert result.final_attempt == "repaired"
    assert result.validation.valid is True
    assert len(result.attempts) == 2
    assert '"truncated"' in fake.calls[1]
    assert '"missing_citation"' in fake.calls[1]