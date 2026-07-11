from ingestion.arxiv_scraper import Paper
from ingestion.citation_extractor import extract_citations
from ingestion.data_cleaner import clean_text
from ingestion.paper_discovery import (
    AlphaXivMCPPaperDiscovery,
    AlphaXivThenArxivDiscovery,
    FeymanPaperDiscovery,
    ResearchAPIPaperDiscovery,
    _alphaxiv_argument_sets,
    build_discovery,
    dedupe_papers,
    score_papers,
)
from ingestion.pdf_parser import RawDocument, RawPage
from ingestion.section_detector import SectionDetector, detect_section


class FakeAlphaXivMCPDiscovery(AlphaXivMCPPaperDiscovery):
    def _call_discover_papers(self, query: str, max_results: int):
        return {
            "results": [
                {
                    "arxiv_id": "2005.11401",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "authors": ["Patrick Lewis", "Ethan Perez"],
                    "abstract_preview": "RAG combines parametric memory with retrieved passages.",
                    "publication_date": "2020-05-22",
                    "organizations": ["Meta AI"],
                }
            ]
        }


class FakeFeymanDiscovery(FeymanPaperDiscovery):
    def _search_command(self, query: str, max_results: int):
        return {
            "results": [
                {
                    "arxiv_id": "2005.11401",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "authors": [{"name": "Patrick Lewis"}, {"name": "Ethan Perez"}],
                    "abstract": "RAG combines parametric and non-parametric memory.",
                    "primary_category": "cs.CL",
                    "pdf_url": "https://arxiv.org/pdf/2005.11401.pdf",
                    "url": "https://arxiv.org/abs/2005.11401",
                }
            ]
        }


def _paper(paper_id, title, summary, doi=None):
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=["Ada Lovelace"],
        summary=summary,
        published=None,
        updated=None,
        primary_category="cs.CL",
        categories=["cs.CL"],
        pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
        entry_id=f"https://arxiv.org/abs/{paper_id}",
        doi=doi,
    )


def test_detect_section_headings():
    assert detect_section("Abstract") == "abstract"
    assert detect_section("2 Methodology") == "method"
    assert detect_section("This method improves recall") is None


def test_feyman_discovery_normalizes_papers():
    papers = FakeFeymanDiscovery(command="feyman").search("rag", max_results=5)
    assert len(papers) == 1
    assert papers[0].paper_id == "2005.11401"
    assert papers[0].primary_category == "cs.CL"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2005.11401.pdf"
    assert papers[0].metadata["discovery_source"] == "feyman"


def test_section_detector_extracts_expected_sections():
    raw = RawDocument(
        paper_id="1234.5678v1",
        pdf_path="paper.pdf",
        pages_count=1,
        pages=[RawPage(page_number=1, text="Abstract\nWe study retrieval augmented generation.\n1 Introduction\nRAG combines retrieval and generation.\nReferences\n[1] Lewis et al. Retrieval-Augmented Generation. 2020. arXiv:2005.11401", char_count=190)],
    )
    sectioned = SectionDetector().detect(raw)
    assert "retrieval augmented generation" in sectioned.sections["abstract"]
    assert "RAG combines" in sectioned.sections["introduction"]
    assert "Lewis" in sectioned.sections["references"]


def test_extract_citations_from_references_section():
    class Doc:
        sections = {"references": "[1] Lewis et al. Retrieval-Augmented Generation. 2020. arXiv:2005.11401 [2] Vaswani et al. Attention Is All You Need. 2017. doi:10.5555/3295222.3295349"}
    citations = extract_citations(Doc())
    assert len(citations) == 2
    assert citations[0].arxiv_id == "2005.11401"
    assert citations[1].doi == "10.5555/3295222.3295349"


def test_clean_text_normalizes_pdf_artifacts():
    cleaned = clean_text("Header repeated\nA hyphen-\nated word.\n\n5\n\\textbf{Result}  has   spaces.")
    assert "hyphenated" in cleaned
    assert "5" not in cleaned.splitlines()
    assert "Result" in cleaned
    assert "  " not in cleaned


def test_research_api_provider_can_be_built():
    assert isinstance(build_discovery("research-apis"), ResearchAPIPaperDiscovery)


def test_dedupe_papers_uses_arxiv_id_doi_and_title():
    papers = [_paper("2005.11401v1", "Retrieval Augmented Generation", "RAG", doi="10.1/a"), _paper("2005.11401v2", "Retrieval Augmented Generation", "RAG"), _paper("9999.00001", "Different title", "Different", doi="10.1/a"), _paper("9999.00002", "Retrieval   augmented generation!", "Same title")]
    deduped = dedupe_papers(papers)
    assert len(deduped) == 1
    assert deduped[0].paper_id == "2005.11401v1"


def test_score_papers_prefers_relevant_title_and_abstract_without_embeddings():
    scored = score_papers("retrieval augmented generation", [_paper("1", "Graph neural networks", "Molecular property prediction"), _paper("2", "Retrieval augmented generation", "RAG retrieves passages for question answering")], use_embeddings=False)
    assert scored[0].paper_id == "2"
    assert "selection_score" in scored[0].metadata


def test_alphaxiv_mcp_discovery_normalizes_papers():
    papers = FakeAlphaXivMCPDiscovery(command="fake-mcp").search("rag", max_results=5)
    assert len(papers) == 1
    assert papers[0].paper_id == "2005.11401"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2005.11401.pdf"
    assert papers[0].entry_id == "https://arxiv.org/abs/2005.11401"
    assert papers[0].metadata["discovery_source"] == "alphaxiv_mcp"
    assert papers[0].metadata["organizations"] == ["Meta AI"]


def test_alphaxiv_provider_can_be_built():
    assert isinstance(build_discovery("alphaxiv-mcp"), AlphaXivMCPPaperDiscovery)


def test_alphaxiv_argument_sets_preserve_domain_phrases():
    argument_sets = _alphaxiv_argument_sets("earthquake analysis using deep learning", difficulty=5, groq_api_key=None)
    assert any("deep learning" in args["keywords"] for args in argument_sets)
    assert any("seismic" in args["keywords"] for args in argument_sets)
    assert any(args["difficulty"] >= 7 for args in argument_sets)


class BrokenDiscovery:
    def search(self, query: str, max_results: int = 50):
        raise RuntimeError("alphaXiv unavailable")


class FakeArxivFallback:
    def search(self, query: str, max_results: int = 50):
        return [_paper("2005.11401v4", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "RAG combines parametric and retrieved knowledge.")]


def test_auto_discovery_falls_back_to_arxiv_when_alphaxiv_fails():
    discovery = AlphaXivThenArxivDiscovery(primary=BrokenDiscovery(), fallback=FakeArxivFallback())
    papers = discovery.search("retrieval augmented generation", max_results=5)
    assert len(papers) == 1
    assert discovery.last_provider_used == "arxiv"
    assert papers[0].metadata["discovery_provider_used"] == "arxiv"
    assert "alphaXiv unavailable" in papers[0].metadata["fallback_reason"]


def test_default_discovery_uses_auto_fallback_provider():
    assert isinstance(build_discovery(), AlphaXivThenArxivDiscovery)
