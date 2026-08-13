from __future__ import annotations

from evaluation.evidence_packing import run_packing_comparison
from generation.context_assembler import ContextAssembler
from retrieval.models import RetrievalResult


def _chunk(chunk_id, text, *, section="method"):
    return RetrievalResult(
        chunk_id,
        text,
        1.0,
        "frozen",
        paper_id="paper-1",
        section=section,
    )


def test_three_packing_modes_preserve_per_chunk_citations():
    gold = _chunk("gold", "gold evidence")
    previous = _chunk("previous", "preceding evidence")
    following = _chunk("following", "following evidence")
    other_section = _chunk("other", "other section evidence", section="results")
    concepts = ["mechanism one", "mechanism two"]

    adjacent = ContextAssembler(
        evidence_packing_mode="adjacent",
        adjacent_chunk_lookup=lambda chunk: [previous, following, other_section],
    ).assemble([gold], required_concepts=concepts)
    section = ContextAssembler(
        evidence_packing_mode="section",
        section_chunk_lookup=lambda chunk: [previous, gold, following],
    ).assemble([gold], required_concepts=concepts)

    assert [source.chunk_id for source in adjacent.citation_map.values()] == [
        "gold",
        "previous",
        "following",
    ]
    assert [source.chunk_id for source in section.citation_map.values()] == [
        "gold",
        "previous",
        "following",
    ]
    assert '[2]' in adjacent.context_block and '"preceding evidence"' in adjacent.context_block


def test_single_concept_keeps_gold_only_even_when_expansion_is_configured():
    gold = _chunk("gold", "gold evidence")
    neighbor = _chunk("neighbor", "neighbor evidence")
    assembled = ContextAssembler(
        evidence_packing_mode="adjacent",
        adjacent_chunk_lookup=lambda chunk: [neighbor],
    ).assemble([gold], required_concepts=["one concept"])

    assert [source.chunk_id for source in assembled.citation_map.values()] == ["gold"]


def test_comparison_reports_correctness_and_faithfulness_side_by_side():
    gold = _chunk("gold", "gold evidence")
    neighbor = _chunk("neighbor", "neighbor evidence")

    report = run_packing_comparison(
        [gold],
        ["one", "two"],
        adjacent_chunk_lookup=lambda chunk: [neighbor],
        section_chunk_lookup=lambda chunk: [neighbor],
        score_context=lambda mode, context: {
            "answer_correctness": {"gold": 0.6, "adjacent": 0.8, "section": 0.75}[mode],
            "faithfulness": 1.0,
        },
    )

    assert [row["mode"] for row in report["packing_modes"]] == [
        "gold",
        "adjacent",
        "section",
    ]
    assert all("answer_correctness" in row and "faithfulness" in row for row in report["packing_modes"])
