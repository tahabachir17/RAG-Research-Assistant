from __future__ import annotations

import pytest

from generation.context_assembler import ContextAssembler
from retrieval import RetrievalResult


def _result(chunk_id, text, *, paper_id="paper", section="method", score=1.0):
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source="test",
        paper_id=paper_id,
        title=f"Title {paper_id}",
        authors=["Ada", "Grace"],
        year=2024,
        section=section,
        url=f"https://example.test/{paper_id}",
    )


def test_context_assembler_numbers_sources_and_preserves_provenance():
    assembled = ContextAssembler(max_context_tokens=100).assemble(
        [_result("c1", "first evidence"), _result("c2", "second evidence")]
    )

    assert "[1] Title: Title paper" in assembled.context_block
    assert 'Section: method\n"first evidence"' in assembled.context_block
    assert list(assembled.citation_map) == [1, 2]
    assert assembled.citation_map[2].chunk_id == "c2"
    assert assembled.to_dict()["citation_map"][1]["paper_id"] == "paper"


def test_context_assembler_enforces_budget_without_cutting_chunks():
    def counter(text):
        return 4 if "first" in text else 7

    assembled = ContextAssembler(max_context_tokens=5, token_counter=counter).assemble(
        [_result("c1", "first"), _result("c2", "second")]
    )
    assert list(assembled.citation_map) == [1]
    assert "second" not in assembled.context_block


def test_context_assembler_skips_oversized_chunk_and_keeps_later_evidence():
    def counter(text):
        return 20 if "oversized" in text else 4

    assembled = ContextAssembler(max_context_tokens=5, token_counter=counter).assemble(
        [_result("c1", "oversized"), _result("c2", "short evidence")]
    )
    assert [source.chunk_id for source in assembled.citation_map.values()] == ["c2"]
    assert "short evidence" in assembled.context_block


def test_context_assembler_empty_input_and_optional_section_deduplication():
    assembler = ContextAssembler(dedupe_paper_sections=True)
    assert assembler.assemble([]).context_block == ""
    assembled = assembler.assemble(
        [
            _result("c1", "one"),
            _result("c2", "two"),
            _result("c3", "three", section="results"),
        ]
    )
    assert [source.chunk_id for source in assembled.citation_map.values()] == [
        "c1",
        "c3",
    ]
    with pytest.raises(TypeError, match="RetrievalResult"):
        assembler.assemble([object()])
