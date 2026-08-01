from ingestion.pdf_parser import RawDocument, RawPage
from ingestion.section_detector import SectionDetector, detect_section


def doc(*pages):
    raw_pages = [RawPage(i, text, len(text)) for i, text in enumerate(pages, 1)]
    return SectionDetector().detect(
        RawDocument("paper", "paper.pdf", len(raw_pages), raw_pages)
    )


def test_standard_roman_aliases_and_abstract():
    result = doc(
        "Title\nAuthors\n\nAbstract\nSummary text.\n\nI. Introduction\nIntro text.\n\nII. Methods\nMethod text."
    )
    assert result.sections["front_matter"].startswith("Title")
    assert result.sections["abstract"] == "Summary text."
    assert result.sections["introduction"] == "Intro text."
    assert result.sections["methodology"] == "Method text."
    assert detect_section("Proposed Approach") == "methodology"


def test_subsection_hierarchy_and_final_line_preservation():
    result = doc(
        "3 Methodology\nOverview.\n\n3.2 Training Objective\nLoss details.\nFinal line"
    )
    assert "Final line" in result.sections["methodology"]
    detail = next(
        item
        for item in result.section_details
        if item.get("subsection_number") == "3.2"
    )
    assert detail["section_number"] == "3"
    assert detail["canonical_section"] == "methodology"


def test_references_conclusion_future_work_and_page_transition():
    result = doc(
        "Conclusions and Future Work\nClosing text.",
        "REFERENCES\n[1] A. Author. Title. 2020.",
    )
    assert result.sections["conclusion"] == "Closing text."
    assert "Author" in result.sections["references"]
    assert result.section_spans["references"] == {"start_page": 2, "end_page": 2}


def test_false_headings_and_no_headings_preserve_content():
    text = "Introduction to this paper is provided here.\nFigure 1 Model architecture\n- Methods are useful\nLast sentence."
    result = doc(text)
    assert all(
        part in result.sections["front_matter"]
        for part in (
            "Introduction to this paper",
            "Figure 1",
            "Methods are useful",
            "Last sentence",
        )
    )
    assert not result.heading_diagnostics


def test_duplicate_section_names_append_instead_of_overwrite():
    result = doc("1 Results\nFirst result.\n\nResults\nSecond result.")
    assert "First result" in result.sections["results"]
    assert "Second result" in result.sections["results"]
    assert (
        len(
            [
                d
                for d in result.heading_diagnostics
                if d["canonical_section"] == "results"
            ]
        )
        == 2
    )


def test_multiline_heading_and_content_preservation():
    result = doc(
        "2 Experimental\nSetup\nConfiguration text.\n\n3 Related Work\nNeural Retrieval\nThis line remains content."
    )
    assert result.sections["experiments"] == "Configuration text."
    assert "Neural Retrieval" in result.sections["related_work"]
    assert "This line remains" in result.sections["related_work"]
    assert any(
        d["matched_rule"].startswith("multiline_") for d in result.heading_diagnostics
    )
