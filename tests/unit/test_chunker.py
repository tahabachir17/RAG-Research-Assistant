from processing.chunker import SectionAwareChunker


class WordTokenizer:
    def spans(self, text):
        import re

        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def test_short_sections_keep_labels_and_exact_offsets():
    document = {
        "paper_id": "1234.5678",
        "metadata": {"title": "A paper"},
        "sections": {"abstract": "  A compact abstract.  ", "method": "Method details."},
    }
    chunks = SectionAwareChunker(tokenizer=WordTokenizer()).chunk(document)

    assert [chunk.section for chunk in chunks] == ["abstract", "method"]
    for chunk in chunks:
        source = document["sections"][chunk.section]
        assert source[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.metadata == document["metadata"]


def test_long_section_is_overlapping_and_deterministic():
    text = " ".join(f"word{i}" for i in range(24))
    document = {"paper_id": "paper", "sections": {"method": text}}
    chunker = SectionAwareChunker(max_tokens=10, overlap_tokens=2, tokenizer=WordTokenizer())

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert len(first) == 3
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].text.split()[-2:] == first[1].text.split()[:2]
    assert all(len(chunk.text.split()) <= 10 for chunk in first)


def test_falls_back_to_raw_pages_when_sections_are_empty():
    document = {
        "paper_id": "paper",
        "sections": {"abstract": ""},
        "raw_document": {"pages": [{"text": "First page."}, {"text": "Second page."}]},
    }

    chunks = SectionAwareChunker(tokenizer=WordTokenizer()).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].section == "unknown"
    assert chunks[0].text == "First page.\n\nSecond page."


def test_configuration_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        SectionAwareChunker(max_tokens=10, overlap_tokens=10)
