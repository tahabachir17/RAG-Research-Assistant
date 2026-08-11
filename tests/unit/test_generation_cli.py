from __future__ import annotations

import json

import pytest

from generation.citation_handler import ClaimSupportFlag
from generation.cli import (
    add_retrieved_evidence,
    build_evidence_queries,
    diversify_ranked_results,
    fuse_query_results,
    load_ranked_results,
    main,
    retrieve_ranked_results,
    prioritize_explicit_rag_evidence,
    run_generation,
)
from retrieval.models import RetrievalResult


def test_offline_generation_harness_returns_resolved_source():
    results = load_ranked_results()
    response = run_generation("What attention is used?", results)
    assert response.citations_valid is True
    assert response.sources[0]["paper_id"] == "1706.03762"
    assert response.sources[0]["chunk_id"] == "demo-transformer-method"
    assert response.answer == "The supplied context describes scaled dot-product attention. [1]"


def test_generation_appends_optional_verifier_flags_to_response():
    class Verifier:
        def verify(self, context, answer, *, structured_data=None):
            return [
                ClaimSupportFlag(
                    "claim-1",
                    "The supplied context describes scaled dot-product attention.",
                    [1],
                    "supported",
                    "llm_self_check",
                    "Directly supported.",
                )
            ]

    response = run_generation(
        "What attention is used?",
        load_ranked_results(),
        faithfulness_verifier=Verifier(),
    )

    assert [flag["checker"] for flag in response.claim_support] == [
        "lexical_overlap",
        "llm_self_check",
    ]


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
        "How does RAG work?",
        tmp_path / "bm25.pkl",
        top_k=3,
        expand_evidence_queries=False,
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
        expand_evidence_queries=False,
    )
    assert results == [second, first]


def test_expanded_retrieval_bounds_candidates_before_reranking(monkeypatch, tmp_path):
    calls = 0

    class FakeSparseRetriever:
        def __init__(self, index_path, default_top_k):
            assert default_top_k == 2

        def search(self, question):
            nonlocal calls
            calls += 1
            return [
                RetrievalResult(
                    f"c{calls}-a",
                    f"first {calls}",
                    2.0,
                    "sparse",
                    paper_id=f"p{calls}-a",
                ),
                RetrievalResult(
                    f"c{calls}-b",
                    f"second {calls}",
                    1.0,
                    "sparse",
                    paper_id=f"p{calls}-b",
                ),
            ]

    class FakeReranker:
        def rerank(self, question, candidates, top_k):
            assert len(candidates) == top_k == 2
            return candidates

    monkeypatch.setattr("generation.cli.SparseRetriever", FakeSparseRetriever)
    results = retrieve_ranked_results(
        "Compare retrieval methods in RAG",
        tmp_path / "bm25.pkl",
        top_k=2,
        candidate_k=2,
        reranker=FakeReranker(),
        expand_evidence_queries=True,
    )
    assert calls == 4
    assert len(results) == 2


def test_diversification_limits_papers_sections_versions_and_exact_chunks():
    candidates = [
        RetrievalResult(
            "ref",
            "citation list",
            12.0,
            "sparse",
            paper_id="paper-r",
            section="references",
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

    assert [result.chunk_id for result in selected] == ["c1", "c4", "c5", "c3"]

    diagnostic = diversify_ranked_results(
        candidates[:1],
        top_k=1,
        excluded_sections=(),
    )
    assert [result.chunk_id for result in diagnostic] == ["ref"]


def test_evidence_queries_cover_evaluation_and_limitations():
    queries = build_evidence_queries(
        "Compare RAG retrieval methods. Identify three methods and report details."
    )
    assert len(queries) == 4
    assert "dataset benchmark metric" in queries[2]
    assert "limitations drawbacks" in queries[3]


def test_multi_query_fusion_rewards_chunks_found_across_facets():
    shared = RetrievalResult("shared", "shared", 9.0, "sparse", paper_id="p1")
    first = RetrievalResult("first", "first", 10.0, "sparse", paper_id="p2")
    second = RetrievalResult("second", "second", 10.0, "sparse", paper_id="p3")
    fused = fuse_query_results([[first, shared], [second, shared]])
    assert fused[0].chunk_id == "shared"
    assert fused[0].source == "multi_query_sparse"


def test_explicit_rag_scope_uses_passage_text_not_paper_title():
    unrelated = RetrievalResult(
        "other",
        "Video-text retrieval results.",
        2.0,
        "test",
        title="A RAG Survey",
    )
    rag = RetrievalResult(
        "rag",
        "The method was evaluated in retrieval-augmented generation (RAG).",
        1.0,
        "test",
    )
    strict_question = "Only classify a method if the passage explicitly evaluates it in a RAG setting."
    assert [
        item.chunk_id
        for item in prioritize_explicit_rag_evidence(strict_question, [unrelated, rag])
    ] == ["rag", "other"]
    assert prioritize_explicit_rag_evidence(
        "Compare ordinary retrieval models.", [unrelated, rag]
    ) == [unrelated, rag]


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
