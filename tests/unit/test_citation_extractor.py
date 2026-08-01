from ingestion.citation_extractor import extract_citations
from ingestion.pdf_parser import RawDocument, RawPage


def test_numbered_multiline_doi_arxiv_url_and_fields():
    text = """[1] Smith, A. A Useful Paper. Journal of Tests. (2020).
https://doi.org/10.1145/1234.5678
[2] Doe, B. Attention Work. NeurIPS. 2017. arXiv:1706.03762
https://arxiv.org/abs/1706.03762"""
    citations = extract_citations({"sections": {"references": text}})
    assert len(citations) == 2
    assert citations[0].reference_number == 1
    assert citations[0].doi == "10.1145/1234.5678"
    assert citations[0].year == 2020
    assert citations[1].arxiv_id == "1706.03762"
    assert citations[1].url == "https://arxiv.org/abs/1706.03762"
    assert citations[0].raw_text.endswith("10.1145/1234.5678")


def test_dot_numbered_across_pages_and_heading_detection():
    raw = RawDocument(
        "p",
        "p.pdf",
        2,
        [
            RawPage(1, "Body.\nBibliography\n1. Smith, A. A paper title.", 40),
            RawPage(
                2, "Journal Name, 2021.\n2. Doe, B. Another title. Venue, 2022.", 60
            ),
        ],
    )
    citations = extract_citations(raw)
    assert [item.reference_number for item in citations] == [1, 2]
    assert "Journal Name" in citations[0].raw_text


def test_unnumbered_entries_and_deduplication():
    text = """Smith, A. First Useful Title. Journal. (2020).
Doe, B. Second Useful Title. Conference. (2021).
Smith, A. First Useful Title. Journal. (2020)."""
    citations = extract_citations({"sections": {"works_cited": text}})
    assert len(citations) == 2
    assert citations[0].authors


def test_malformed_empty_and_no_reference_heading():
    assert extract_citations({"sections": {"references": "nonsense words only"}})
    assert extract_citations({"sections": {"references": ""}}) == []
    assert extract_citations("Introduction\nA paper with [1] inline citation.") == []
