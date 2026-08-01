import numpy as np
import pytest

from retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    MMRSampler,
    RetrievalResult,
    build_retriever,
    maximal_marginal_relevance,
)


def _result(chunk_id, score, source="test"):
    return RetrievalResult(chunk_id, f"text {chunk_id}", score, source)


class _Retriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.results


def test_hybrid_rrf_fuses_duplicates_and_ignores_raw_score_scale():
    dense = _Retriever([_result("dense", 1000), _result("shared", 0.1)])
    sparse = _Retriever([_result("shared", -5), _result("sparse", 9999)])
    retriever = HybridRetriever(dense, sparse, rrf_k=60, default_top_k=3)

    results = retriever.search("query", candidate_top_k=7)

    assert [result.chunk_id for result in results] == ["shared", "dense", "sparse"]
    assert all(result.source == "hybrid" for result in results)
    assert dense.calls[0][1]["top_k"] == 7


class _CrossEncoder:
    def predict(self, pairs):
        assert pairs == [("query", "text a"), ("query", "text b")]
        return np.asarray([0.2, 0.9])


def test_cross_encoder_reranker_scores_pairs_and_preserves_payload():
    candidates = [_result("a", 10), _result("b", 1)]
    results = CrossEncoderReranker(model=_CrossEncoder(), default_top_k=2).rerank(
        "query", candidates
    )
    assert [result.chunk_id for result in results] == ["b", "a"]
    assert [result.score for result in results] == [0.9, 0.2]
    assert candidates[0].source == "test"


class _Embedder:
    def encode_texts(self, texts):
        assert len(texts) == 4
        return np.asarray([[1, 0], [1, 0], [0.99, 0.01], [0.6, 0.8]])


def test_mmr_prefers_diverse_candidate_and_function_validates_dimensions():
    candidates = [_result("best", 3), _result("duplicate", 2), _result("diverse", 1)]
    selected = MMRSampler(_Embedder(), lambda_mult=0.4, default_top_k=2).sample(
        "query", candidates
    )
    assert [result.chunk_id for result in selected] == ["best", "diverse"]
    with pytest.raises(ValueError, match="dimensions"):
        maximal_marginal_relevance([1, 0], [[1, 0, 0]], top_k=1)


def test_factory_builds_all_supported_types(tmp_path):
    sparse = _Retriever([])
    dense = _Retriever([])
    hybrid = build_retriever(
        {"type": "hybrid", "rrf_k": 30},
        dense_retriever=dense,
        sparse_retriever=sparse,
    )
    assert isinstance(hybrid, HybridRetriever)
    assert hybrid.rrf_k == 30
    with pytest.raises(ValueError, match="unsupported"):
        build_retriever({"type": "unknown"})
