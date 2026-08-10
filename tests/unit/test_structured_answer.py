from __future__ import annotations

import pytest

from generation.structured_answer import (
    StructuredAnswerError,
    parse_and_render_structured_answer,
    parse_and_render_structured_narrative,
    structured_answer_instruction,
)


def test_structured_answer_renders_citations_per_cell():
    raw = '{"items":[{"method":{"text":"Dense retrieval","citations":[1]},"limitations":{"text":"Not reported in the supplied passages.","citations":[]}}]}'
    rendered, rows = parse_and_render_structured_answer(
        raw,
        required_fields=["method", "limitations"],
        valid_citations={1},
        max_items=2,
    )
    assert "Dense retrieval [1]" in rendered
    assert rows["items"][0]["limitations"]["citations"] == []


def test_structured_answer_instruction_bounds_local_model_output():
    instruction = structured_answer_instruction(["method", "limitations"], 2)
    assert "summary must be an empty string" in instruction
    assert "at most 18 words" in instruction
    assert "one central contribution" in instruction
    assert "Do not combine evidence about different methods" in instruction
    assert "Do not split one method or contribution" in instruction


def test_structured_answer_rejects_uncited_factual_cell():
    raw = '{"items":[{"method":{"text":"Dense retrieval","citations":[]}}]}'
    with pytest.raises(StructuredAnswerError, match="structured_field_uncited"):
        parse_and_render_structured_answer(
            raw,
            required_fields=["method"],
            valid_citations={1},
        )


def test_structured_answer_rejects_nonempty_answered_summary_and_verbose_cell():
    verbose = " ".join(f"word{index}" for index in range(19))
    raw = (
        '{"answer_status":"answered","summary":"unrequested summary","items":['
        f'{{"method":{{"text":"{verbose}","citations":[1]}}}}]}}'
    )
    with pytest.raises(StructuredAnswerError) as caught:
        parse_and_render_structured_answer(
            raw,
            required_fields=["method"],
            valid_citations={1},
        )
    assert "structured_answer_summary_not_empty" in caught.value.failures
    assert "structured_field_too_long:1:method" in caught.value.failures


def test_structured_answer_allows_explicit_uncited_abstention():
    raw = '{"answer_status":"insufficient_evidence","summary":"The supplied passages do not support a qualifying method.","items":[]}'
    rendered, structured = parse_and_render_structured_answer(
        raw,
        required_fields=["method"],
        valid_citations={1},
    )
    assert rendered.startswith("The supplied passages")
    assert structured["answer_status"] == "insufficient_evidence"


def test_structured_answer_rejects_row_with_only_absent_values():
    raw = '{"items":[{"method":{"text":"Not reported in the supplied passages.","citations":[]},"limitations":{"text":"Not reported in the supplied passages.","citations":[]}}]}'
    with pytest.raises(StructuredAnswerError, match="structured_item_empty:1"):
        parse_and_render_structured_answer(
            raw,
            required_fields=["method", "limitations"],
            valid_citations={1},
        )


def test_structured_answer_rejects_near_duplicate_rows():
    raw = '{"items":[{"method":{"text":"Scales adapter outputs with learned parameters","citations":[1]}},{"method":{"text":"Scales adapter outputs using learned parameters","citations":[1]}}]}'
    with pytest.raises(StructuredAnswerError, match="structured_items_duplicate:1:2"):
        parse_and_render_structured_answer(
            raw,
            required_fields=["method"],
            valid_citations={1},
        )


def test_structured_narrative_renders_atomic_claim_citations():
    raw = '{"answer_status":"answered","summary":"","claims":[{"text":"Scaling prevents softmax saturation.","citations":[1]},{"text":"Values are combined using attention weights.","citations":[2]}]}'
    rendered, structured = parse_and_render_structured_narrative(
        raw,
        valid_citations={1, 2},
    )
    assert rendered == (
        "Scaling prevents softmax saturation. [1]\n\n"
        "Values are combined using attention weights. [2]"
    )
    assert len(structured["claims"]) == 2


def test_structured_narrative_rejects_overcitation_outside_context():
    raw = '{"answer_status":"answered","summary":"","claims":[{"text":"Claim.","citations":[9]}]}'
    with pytest.raises(StructuredAnswerError, match="structured_citation_out_of_range"):
        parse_and_render_structured_narrative(raw, valid_citations={1})
