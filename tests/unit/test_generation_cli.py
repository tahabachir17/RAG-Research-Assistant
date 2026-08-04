from __future__ import annotations

import json

import pytest

from generation.cli import (
    add_retrieved_evidence,
    diversify_ranked_results,
    load_ranked_results,
    main,
    retrieve_ranked_results,
    run_generation,
)
from retrieval.models import RetrievalResult


def test_offline_generation_harness_returns_resolved_source():
    results = load_ranked_results()
    response = run_generation("What attention is used?", results)
    assert response.citations_valid is True
    assert response.sources[0]["paper_id"] == "1706.03762"
    assert response.sources[0]["chunk_id"] == "demo-transformer-method"


def test_harness_loads_evaluator_rankings_and_selects_query(tmp_path):
    payload = {
        "rankings": {
            "hybrid_rerank": [
                {
                    "query_id": "q1",
                    "results": [
                        {
                            "chunk_id": "c1",
                            "text": "evidence",
                            "score": 1.0,
                            "source": "reranked",
                            "paper_id": "p1",
                        }
                    ],
                }
            ]
        }
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    results = load_ranked_results(path, config="hybrid_rerank", query_id="q1")
    assert results[0].chunk_id == "c1"
    with pytest.raises(ValueError, match="missing"):
        load_ranked_results(path, config="hybrid_rerank", query_id="missing")


def test_harness_retrieves_new_question_from_local_index(monkeypatch, tmp_path):
    expected = RetrievalResult(
        chunk_id="c1", text="retrieved evidence", score=2.0, source="sparse"
    )

    class FakeSparseRetriever:
        def __init__(self, index_path, default_top_k):
            assert index_path == tmp_path / "bm25.pkl"
            assert default_top_k == 3

        def search(self, question):
            assert question == "How does RAG work?"
            return [expected]

    monkeypatch.setattr("generation.cli.SparseRetriever", FakeSparseRetriever)
    results = retrieve_ranked_results(
        "How does RAG work?", tmp_path / "bm25.pkl", top_k=3
    )
    assert results == [expected]


def test_retrieval_reranks_candidates_before_diversification(monkeypatch, tmp_path):
    first = RetrievalResult("c1", "first", 2.0, "sparse", paper_id="p1")
    second = RetrievalResult("c2", "second", 1.0, "sparse", paper_id="p2")

    class FakeSparseRetriever:
        def __init__(self, index_path, default_top_k):
            assert default_top_k == 2

        def search(self, question):
            return [first, second]

    class FakeReranker:
        def rerank(self, question, candidates, top_k):
            assert question == "scope-sensitive question"
            assert candidates == [first, second]
            assert top_k == 2
            return [second, first]

    monkeypatch.setattr("generation.cli.SparseRetriever", FakeSparseRetriever)
    results = retrieve_ranked_results(
        "scope-sensitive question",
        tmp_path / "bm25.pkl",
        top_k=2,
        candidate_k=2,
        reranker=FakeReranker(),
    )
    assert results == [second, first]


def test_diversification_limits_papers_sections_versions_and_exact_chunks():
    candidates = [
        RetrievalResult(
            "ref", "citation list", 12.0, "sparse", paper_id="paper-r", section="references"
        ),
        RetrievalResult(
            "front",
            "title and authors",
            11.0,
            "sparse",
            paper_id="paper-f",
            section="front matter",
        ),
        RetrievalResult(
            "c1", "best", 10.0, "sparse", paper_id="1234.5678v1", section="method"
        ),
        RetrievalResult(
            "c1",
            "duplicate",
            9.5,
            "sparse",
            paper_id="1234.5678v1",
            section="method",
        ),
        RetrievalResult(
            "c2",
            "same section",
            9.0,
            "sparse",
            paper_id="1234.5678v2",
            section="method",
        ),
        RetrievalResult(
            "c3",
            "other section",
            8.0,
            "sparse",
            paper_id="1234.5678v2",
            section="results",
        ),
        RetrievalResult(
            "c4", "other paper", 7.0, "sparse", paper_id="paper-2", section="method"
        ),
        RetrievalResult(
            "c5", "third paper", 6.0, "sparse", paper_id="paper-3", section="method"
        ),
    ]

    selected = diversify_ranked_results(
        candidates,
        top_k=4,
        max_chunks_per_paper=2,
        max_chunks_per_section=1,
    )

    assert [result.chunk_id for result in selected] == ["c1", "c3", "c4", "c5"]

    diagnostic = diversify_ranked_results(
        candidates[:1],
        top_k=1,
        excluded_sections=(),
    )
    assert [result.chunk_id for result in diagnostic] == ["ref"]


def test_retrieved_evidence_includes_full_cited_chunk():
    result = RetrievalResult(
        chunk_id="c1",
        text="The complete paragraph used as evidence.",
        score=4.25,
        source="sparse",
    )
    payload = {"answer": "Grounded answer [1].", "sources": [{"chunk_id": "c1"}]}
    enriched = add_retrieved_evidence(payload, [result])
    assert enriched["sources"][0] == {
        "chunk_id": "c1",
        "retrieval_rank": 1,
        "retrieval_score": 4.25,
        "retrieval_source": "sparse",
        "retrieved_text": "The complete paragraph used as evidence.",
    }
    assert payload["sources"][0] == {"chunk_id": "c1"}


def test_main_runs_retrieval_generation_and_evidence(monkeypatch, capsys):
    result = RetrievalResult(
        chunk_id="c1",
        text="Full retrieved paragraph.",
        score=5.0,
        source="sparse",
        paper_id="p1",
    )

    class FakeLiveClient:
        def complete(self, system, user, *, stream=False):
            assert "Full retrieved paragraph." in user
            return "Generated from retrieved evidence [1]."

    monkeypatch.setattr(
        "generation.cli.retrieve_ranked_results",
        lambda question, index_path, **kwargs: [result],
    )
    monkeypatch.setattr("generation.cli._live_client", lambda args: FakeLiveClient())

    assert main(["A difficult question", "--retrieve", "--live"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == "Generated from retrieved evidence [1]."
    assert payload["citations_valid"] is True
    assert payload["sources"][0]["retrieved_text"] == "Full retrieved paragraph."
