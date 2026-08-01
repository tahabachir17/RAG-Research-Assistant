from processing.bm25_indexer import BM25Indexer
from processing.chunker import Chunk


def make_chunk(chunk_id, text, section="unknown"):
    return Chunk(
        chunk_id=chunk_id,
        paper_id=f"paper-{chunk_id}",
        section=section,
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={"source": "test"},
    )


def sample_chunks():
    return [
        make_chunk("dense", "neural semantic dense vector embeddings", "method"),
        make_chunk("sparse", "lexical keyword sparse bm25 retrieval", "method"),
        make_chunk("vision", "image recognition convolutional network", "experiments"),
    ]


def test_build_creates_index_and_stores_serializable_chunks():
    indexer = BM25Indexer()

    indexer.build(sample_chunks())

    assert indexer.index is not None
    assert len(indexer.chunks) == 3
    assert indexer.chunks[0]["chunk_id"] == "dense"


def test_search_returns_relevant_chunk_and_preserves_fields():
    indexer = BM25Indexer()
    indexer.build(sample_chunks())

    results = indexer.search("sparse bm25")

    assert results[0]["chunk_id"] == "sparse"
    assert results[0]["paper_id"] == "paper-sparse"
    assert results[0]["section"] == "method"
    assert results[0]["text"] == "lexical keyword sparse bm25 retrieval"
    assert results[0]["metadata"] == {"source": "test"}
    assert isinstance(results[0]["score"], float)


def test_top_k_is_respected():
    indexer = BM25Indexer()
    indexer.build(sample_chunks())

    assert len(indexer.search("retrieval", top_k=2)) == 2
    assert indexer.search("retrieval", top_k=0) == []


def test_save_creates_file_and_load_restores_search(tmp_path):
    path = tmp_path / "nested" / "bm25.pkl"
    indexer = BM25Indexer()
    indexer.build(sample_chunks())
    expected = indexer.search("semantic embeddings", top_k=1)

    indexer.save(path)
    restored = BM25Indexer.load(path)

    assert path.is_file()
    assert restored.search("semantic embeddings", top_k=1) == expected


def test_empty_chunk_list_is_safe(tmp_path):
    indexer = BM25Indexer()

    indexer.build([])

    assert indexer.index is None
    assert indexer.chunks == []
    assert indexer.search("anything") == []

    path = tmp_path / "empty.pkl"
    indexer.save(path)
    restored = BM25Indexer.load(path)
    assert restored.search("anything") == []


def test_empty_query_is_safe():
    indexer = BM25Indexer()
    indexer.build(sample_chunks())

    assert indexer.search("") == []
    assert indexer.search("   !!!   ") == []


def test_empty_chunk_text_is_not_indexed():
    indexer = BM25Indexer()

    indexer.build([make_chunk("empty", " "), *sample_chunks()])

    assert len(indexer.chunks) == 3
    assert all(chunk["chunk_id"] != "empty" for chunk in indexer.chunks)
