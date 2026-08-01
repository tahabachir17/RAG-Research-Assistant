from types import SimpleNamespace

import numpy as np
import pytest

from processing.bm25_indexer import BM25Indexer
from processing.chunker import Chunk
from retrieval import DenseRetriever, QueryProcessor, RetrievalResult, SparseRetriever


def _chunk(chunk_id, text, *, year=2024, section="abstract"):
    return Chunk(
        chunk_id=chunk_id,
        paper_id=f"paper-{chunk_id}",
        section=section,
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={
            "title": f"Title {chunk_id}",
            "authors": ["Ada"],
            "year": year,
            "category": "cs.CL",
            "url": f"https://example.test/{chunk_id}",
        },
    )


def _saved_index(tmp_path):
    path = tmp_path / "bm25.pkl"
    indexer = BM25Indexer()
    indexer.build(
        [
            _chunk("rag", "retrieval augmented generation grounds answers", year=2024),
            _chunk("bm25", "bm25 lexical sparse keyword retrieval", year=2020),
            _chunk(
                "vision",
                "vision transformer image classification",
                year=2022,
                section="method",
            ),
        ]
    )
    indexer.save(path)
    return path


def test_retrieval_result_preserves_complete_payload():
    payload = {
        "chunk_id": "c1",
        "paper_id": "p1",
        "section": "abstract",
        "text": "A useful chunk",
        "metadata": {"title": "Paper", "authors": ["Ada"], "year": "2024", "custom": 7},
    }
    result = RetrievalResult.from_payload(payload, score=0.8, source="dense")
    assert result.title == "Paper"
    assert result.year == 2024
    assert result.authors == ["Ada"]
    assert result.metadata == payload


@pytest.mark.parametrize("bad_query", ["", " \t\n "])
def test_query_processor_rejects_empty_queries(bad_query):
    with pytest.raises(ValueError, match="empty"):
        QueryProcessor().process(bad_query)


def test_query_processor_preserves_technical_terms_and_uses_bm25_tokenizer():
    query = "  GPT-4\tC++  F1-score\nLLaMA-3 Q-learning  "
    processed = QueryProcessor().process(query)
    assert processed.cleaned_query == "GPT-4 C++ F1-score LLaMA-3 Q-learning"
    assert processed.dense_query == processed.cleaned_query
    assert processed.sparse_tokens == [
        "gpt-4",
        "c++",
        "f1-score",
        "llama-3",
        "q-learning",
    ]
    assert processed.expanded_query is None


def test_query_expansion_is_optional_bounded_and_retains_original():
    plain = QueryProcessor().process("RAG evaluation")
    expanded = QueryProcessor(enable_expansion=True).process("RAG evaluation")
    assert plain.expanded_query is None
    assert expanded.expanded_query.startswith("RAG evaluation")
    assert "retrieval augmented generation" in expanded.expanded_query
    assert expanded.dense_query == expanded.expanded_query


def test_invalid_expansion_dictionary_is_rejected():
    with pytest.raises(ValueError):
        QueryProcessor(expansion_dictionary={"RAG": []})
    with pytest.raises(TypeError):
        QueryProcessor().process(42)
    with pytest.raises(ValueError, match="tokenizer"):
        QueryProcessor(preprocessing_config={"tokenizer": "unknown"})


def test_sparse_retriever_search_filter_threshold_and_health(tmp_path):
    retriever = SparseRetriever(_saved_index(tmp_path), default_top_k=2)
    results = retriever.search(["bm25", "retrieval"])
    assert results[0].chunk_id == "bm25"
    assert results[0].source == "sparse"
    assert results[0].metadata["metadata"]["category"] == "cs.CL"
    assert len(results) == 2

    filtered = retriever.search("retrieval", filters={"year": {"gte": 2024}})
    assert [result.chunk_id for result in filtered] == ["rag"]
    assert retriever.search("retrieval", score_threshold=999) == []
    health = retriever.health_check()
    assert health["loaded"] is True
    assert health["document_count"] == health["chunk_mapping_count"] == 3
    assert health["mapping_valid"] is True
    assert health["version"] == "2.0"


def test_sparse_retriever_errors_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        SparseRetriever(tmp_path / "missing.pkl")
    retriever = SparseRetriever(_saved_index(tmp_path))
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("rag", top_k=0)
    with pytest.raises(TypeError, match="only strings"):
        retriever.search(["rag", 1])
    with pytest.raises(ValueError, match="range"):
        retriever.search("rag", filters={"year": {"between": [2020, 2024]}})


class FakeEmbedder:
    dimension = 3

    def encode_texts(self, texts):
        assert texts == ["dense query"]
        return np.asarray([[0.1, 0.2, 0.3]])


class FakeQdrantClient:
    def __init__(self, *, exists=True, points_count=2, distance="Cosine"):
        self.exists = exists
        self.points_count = points_count
        self.distance = distance
        self.last_query = None

    def collection_exists(self, name):
        return self.exists

    def get_collection(self, name):
        return SimpleNamespace(
            points_count=self.points_count,
            status=SimpleNamespace(value="green"),
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(
                        size=3, distance=SimpleNamespace(value=self.distance)
                    )
                )
            ),
        )

    def query_points(self, **kwargs):
        self.last_query = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="fallback-id",
                    score=0.5,
                    payload={
                        "chunk_id": "low",
                        "text": "lower",
                        "metadata": {"year": 2023},
                    },
                ),
                SimpleNamespace(
                    id="high-id",
                    score=0.9,
                    payload={
                        "chunk_id": "high",
                        "paper_id": "paper-high",
                        "section": "abstract",
                        "text": "higher",
                        "metadata": {"title": "High", "year": 2024},
                    },
                ),
                SimpleNamespace(id="broken", score=1.0, payload={"chunk_id": "broken"}),
            ]
        )


def test_dense_retriever_search_maps_sorts_and_builds_filters():
    client = FakeQdrantClient()
    retriever = DenseRetriever(client, FakeEmbedder(), "papers", default_top_k=2)
    results = retriever.search(
        "dense query",
        filters={"year": {"gte": 2020}, "section": ["abstract", "method"]},
        score_threshold=0.2,
    )
    assert [result.chunk_id for result in results] == ["high", "low"]
    assert results[0].source == "dense"
    assert results[0].title == "High"
    assert client.last_query["with_payload"] is True
    assert client.last_query["limit"] == 2
    assert client.last_query["score_threshold"] == 0.2
    assert client.last_query["query_filter"] is not None


def test_dense_health_and_dimension_validation():
    retriever = DenseRetriever(FakeQdrantClient(), FakeEmbedder(), "papers")
    health = retriever.health_check()
    assert health["connected"] is True
    assert health["collection_exists"] is True
    assert health["vector_size"] == health["embedder_dimension"] == 3
    assert health["dimension_match"] is True
    with pytest.raises(ValueError, match="dimension"):
        retriever.search_by_vector([0.1, 0.2])


@pytest.mark.parametrize(
    ("client", "error"),
    [
        (FakeQdrantClient(exists=False), LookupError),
        (FakeQdrantClient(points_count=0), LookupError),
        (FakeQdrantClient(distance="Dot"), ValueError),
    ],
)
def test_dense_collection_failures_are_clear(client, error):
    retriever = DenseRetriever(client, FakeEmbedder(), "papers")
    with pytest.raises(error):
        retriever.search_by_vector([0.1, 0.2, 0.3])


def test_dense_validates_inputs():
    with pytest.raises(ValueError, match="collection_name"):
        DenseRetriever(FakeQdrantClient(), FakeEmbedder(), "")
    retriever = DenseRetriever(FakeQdrantClient(), FakeEmbedder(), "papers")
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("dense query", top_k=0)
    with pytest.raises(ValueError, match="range"):
        retriever.search_by_vector([0.1, 0.2, 0.3], filters={"year": {"between": 2020}})
