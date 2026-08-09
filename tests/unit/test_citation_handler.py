from __future__ import annotations

from generation.citation_handler import build_source_list, validate_citations
from generation.context_assembler import CitationSource


def _source(number, paper_id):
    return CitationSource(
        citation_number=number,
        paper_id=paper_id,
        chunk_id=f"chunk-{number}",
        title=f"Paper {number}",
        authors=["Author"],
        year=2020 + number,
        section="results",
        url=f"https://example.test/{paper_id}",
    )


def test_validate_citations_tracks_order_unknown_and_unused_numbers():
    citation_map = {1: _source(1, "p1"), 2: _source(2, "p2")}
    result = validate_citations("Claim [2], repeated [2], invented [9].", citation_map)
    assert result.valid is False
    assert result.cited_numbers == [2, 9]
    assert result.unknown_numbers == [9]
    assert result.unused_numbers == [1]


def test_build_source_list_uses_first_citation_order_and_ignores_unknowns():
    citation_map = {1: _source(1, "p1"), 2: _source(2, "p2")}
    sources = build_source_list("Compare [2][1] with an invalid [7].", citation_map)
    assert [source["paper_id"] for source in sources] == ["p2", "p1"]
    assert sources[0]["chunk_id"] == "chunk-2"
    assert sources[0]["primary_category"] is None


def test_grouped_citations_are_validated_and_exposed_as_sources():
    citation_map = {
        1: _source(1, "p1"),
        2: _source(2, "p2"),
        3: _source(3, "p3"),
    }
    result = validate_citations("Supported [2, 3]; unknown [8,9].", citation_map)
    assert result.valid is False
    assert result.cited_numbers == [2, 3, 8, 9]
    assert result.unknown_numbers == [8, 9]
    assert [
        item["paper_id"] for item in build_source_list("Claim [2, 3].", citation_map)
    ] == ["p2", "p3"]


def test_empty_answer_with_context_is_invalid_and_sources_are_unused():
    result = validate_citations("", {1: _source(1, "p1")})
    assert result.valid is False
    assert result.cited_numbers == []
    assert result.unused_numbers == [1]


def test_empty_answer_without_context_is_valid():
    result = validate_citations("", {})
    assert result.valid is True


def test_nonempty_uncited_answer_with_context_is_invalid():
    assert validate_citations("An unsupported answer.", {1: _source(1, "p1")}).valid is False


def test_provider_native_citation_markup_is_hard_rejected():
    result = validate_citations("Claim 【2†L1-L8】 and valid [1].", {1: _source(1, "p1")})
    assert result.valid is False
    assert result.unsupported_markers == ["【2†L1-L8】"]
