from __future__ import annotations

import math

import numpy as np

from evaluation.metrics import ndcg_at_k, precision_at_k, ranked_metrics, recall_at_k
from evaluation.retrieval_evaluator import RetrievalEvaluator
from retrieval import HybridRetriever, RetrievalResult


def result(chunk_id: str, score: float, paper_id: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id, f"text for {chunk_id}", score, "fake", paper_id=paper_id
    )


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=None):
        return self.results[:top_k]


class PromotingReranker:
    def rerank(self, query, candidates, top_k=None):
        return sorted(candidates, key=lambda item: item.chunk_id != "relevant")[:top_k]


class IdentityMMR:
    def sample(self, query, candidates, top_k=None):
        return list(candidates[:top_k])


def test_stage_agnostic_metrics_use_ranked_ids_and_binary_labels():
    ranking = ["x", "a", "b"]
    relevant = ["a", "b"]
    assert recall_at_k(ranking, relevant, 2) == 0.5
    assert precision_at_k(ranking, relevant, 2) == 0.5
    assert ranked_metrics(ranking, relevant, ks=(2,))["mrr"] == 0.5
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert np.isclose(ndcg_at_k(ranking, relevant, 2), expected)


def test_evaluator_flags_or_passes_measured_reranker_lift():
    irrelevant = [result(f"n{i:02d}", 100 - i) for i in range(19)]
    relevant = result("relevant", 0.01)
    dense = FakeRetriever([*irrelevant, relevant])
    sparse = FakeRetriever([*irrelevant, relevant])
    hybrid = HybridRetriever(dense, sparse, default_top_k=20, candidate_top_k=20)
    evaluator = RetrievalEvaluator(
        dense=dense,
        sparse=sparse,
        hybrid=hybrid,
        reranker=PromotingReranker(),
        mmr=IdentityMMR(),
        candidate_k=20,
        final_k=20,
        reranker_lift_threshold=0.02,
    )
    output = evaluator.evaluate(
        [
            {
                "query_id": "q1",
                "question": "q",
                "relevant_chunk_ids": ["relevant"],
                "relevant_paper_ids": [],
            }
        ]
    )
    lift = output["reranker_lift"]
    assert lift["pre_rerank_recall@20"] == 1.0
    assert lift["pre_rerank_recall@8"] == 0.0
    assert lift["post_rerank_recall@8"] == 1.0
    assert lift["passed"] is True
