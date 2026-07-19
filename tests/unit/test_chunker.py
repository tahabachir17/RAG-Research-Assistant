from processing.chunker import Chunk, SectionAwareChunker


class WordTokenizer:
    def spans(self, text):
        import re

        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def test_short_sections_keep_labels_and_exact_offsets():
    document = {
        "paper_id": "1234.5678",
        "metadata": {"title": "A paper"},
        "sections": {
            "abstract": "  A compact abstract.  ",
            "method": "Method details.",
        },
    }
    chunks = SectionAwareChunker(tokenizer=WordTokenizer()).chunk(document)

    assert [chunk.section for chunk in chunks] == ["abstract", "method"]
    for chunk in chunks:
        source = document["sections"][chunk.section]
        assert source[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.metadata == document["metadata"]


def test_normal_sectioned_document_returns_unique_chunks():
    document = {
        "paper_id": "1234.5678",
        "sections": {
            "abstract": "A concise abstract.",
            "introduction": "An introduction to retrieval augmented generation.",
            "method": "The method combines dense and sparse retrieval.",
        },
    }

    chunks = SectionAwareChunker(tokenizer=WordTokenizer()).chunk(document)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.section for chunk in chunks] == [
        "abstract",
        "introduction",
        "method",
    ]
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_long_section_is_overlapping_and_deterministic():
    text = " ".join(f"word{i}" for i in range(24))
    document = {"paper_id": "paper", "sections": {"method": text}}
    chunker = SectionAwareChunker(
        max_tokens=10, overlap_tokens=2, tokenizer=WordTokenizer()
    )

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


def test_empty_sections_are_ignored_when_other_sections_have_text():
    document = {
        "paper_id": "paper",
        "sections": {"abstract": "", "introduction": "Useful text.", "method": None},
    }

    chunks = SectionAwareChunker(tokenizer=WordTokenizer()).chunk(document)

    assert [chunk.section for chunk in chunks] == ["introduction"]


def test_missing_or_malformed_metadata_is_safe():
    without_metadata = {"paper_id": "paper", "sections": {"method": "Text."}}
    malformed_metadata = {
        "paper_id": "paper",
        "metadata": "not-a-mapping",
        "sections": {"method": "Text."},
    }

    for document in (without_metadata, malformed_metadata):
        chunks = SectionAwareChunker(tokenizer=WordTokenizer()).chunk(document)
        assert chunks[0].metadata == {}


def test_malformed_sections_fall_back_to_document_text():
    document = {
        "paper_id": "paper",
        "sections": ["not", "a", "mapping"],
        "text": "Fallback text.",
    }

    chunks = SectionAwareChunker(tokenizer=WordTokenizer()).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].section == "unknown"
    assert chunks[0].text == "Fallback text."


def test_configuration_rejects_invalid_values():
    import pytest

    with pytest.raises(ValueError):
        SectionAwareChunker(max_tokens=10, overlap_tokens=10)

    with pytest.raises(ValueError):
        SectionAwareChunker(min_tokens=0)
