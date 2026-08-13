from __future__ import annotations

import json

import pytest

from config import Settings
from evaluation.full_ragas_evaluation import (
    MetricJsonlCache,
    aggregate_scores,
    assemble_evaluation_questions,
)
from evaluation.run_full_ragas_eval import (
    RETRIEVAL_CONFIGS,
    _parser,
    _resolve_ragas_target,
    build_benchmark_retriever,
)
from retrieval import RetrievalResult, SparseRetriever


def _write(path, questions):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "questions": questions}), encoding="utf-8"
    )


def _manual(identifier="manual-1"):
    return {
        "id": identifier,
        "question": "Manual?",
        "retrieved_chunk_ids": ["production-1"],
        "expected_qualifying_items": [],
        "excluded_items": {},
        "required_fields": [],
        "max_items": None,
        "reviewed": False,
        "calibration_verdicts": [],
    }


def _external(identifier, source, reviewed):
    return {
        "id": identifier,
        "question": "External?",
        "reference_answer": "Reference.",
        "reference_context_ids": ["gold-1"] if reviewed else [],
        "retrieved_chunk_ids": [],
        "reviewed": reviewed,
        "source_dataset": source,
    }


def test_assembly_tags_tiers_and_skips_empty_scidqa(tmp_path):
    manual = tmp_path / "manual.json"
    external = tmp_path / "external"
    _write(manual, [_manual()])
    _write(external / "qasa_generation_qa.json", [_external("qasa-1", "qasa", True)])
    _write(external / "qasper_generation_qa.json", [_external("qasper-1", "qasper", False)])
    _write(external / "scidqa_generation_qa.json", [])

    rows = assemble_evaluation_questions(manual, external)

    assert [(row.source_dataset, row.alignment_status) for row in rows] == [
        ("manual", "aligned"),
        ("qasa", "aligned"),
        ("qasper", "unreviewed"),
    ]


def test_assembly_rejects_cross_tier_id_collision(tmp_path):
    manual = tmp_path / "manual.json"
    external = tmp_path / "external"
    _write(manual, [_manual("same")])
    _write(external / "qasa_generation_qa.json", [_external("same", "qasa", True)])
    _write(external / "qasper_generation_qa.json", [])
    _write(external / "scidqa_generation_qa.json", [])

    try:
        assemble_evaluation_questions(manual, external)
    except ValueError as exc:
        assert "collide" in str(exc)
    else:
        raise AssertionError("cross-tier duplicate should fail")


def test_jsonl_cache_resumes_completed_and_unavailable_pairs(tmp_path):
    cache = MetricJsonlCache(tmp_path / ".cache" / "run.jsonl")
    cache.append("q1", "faithfulness", status="completed", value=0.9)
    cache.append("q1", "context_recall", status="unavailable", reason="unaligned")
    cache.append("q1", "answer_relevancy", status="failed", reason="429")

    resumed = MetricJsonlCache(cache.path)
    assert resumed.completed() == {
        ("q1", "faithfulness"),
        ("q1", "context_recall"),
    }


def test_aggregate_separates_tier_and_alignment_with_counts():
    questions = [
        {"id": "q1", "source_tier": "qasa", "alignment_status": "aligned"},
        {"id": "q2", "source_tier": "qasa", "alignment_status": "unreviewed"},
    ]
    metrics = [
        {"id": "q1", "faithfulness": 0.8, "context_recall": 0.6},
        {"id": "q2", "faithfulness": 1.0, "context_recall": None},
    ]

    result = aggregate_scores(questions, metrics)
    faithfulness = next(
        row
        for row in result["by_source_tier"]
        if row["source_tier"] == "qasa" and row["metric"] == "faithfulness"
    )
    recall = next(
        row
        for row in result["by_alignment_status"]
        if row["alignment_status"] == "unreviewed" and row["metric"] == "context_recall"
    )
    assert faithfulness["mean"] == 0.9
    assert faithfulness["scored"] == 2
    assert recall["mean"] is None
    assert recall["unavailable"] == 1


def test_part9_judge_defaults_to_instant_but_honors_environment_fields():
    defaults = Settings(_env_file=None)
    assert _resolve_ragas_target(defaults, None, None) == (
        "groq",
        "llama-3.1-8b-instant",
    )
    configured = Settings(
        _env_file=None,
        RAGAS_JUDGE_PROVIDER="groq",
        RAGAS_JUDGE_MODEL="replacement-judge",
    )
    assert _resolve_ragas_target(configured, None, None) == (
        "groq",
        "replacement-judge",
    )


def test_part9_defaults_to_factory_hybrid_rerank_and_bench_collection():
    args = _parser().parse_args([])
    assert args.retrieval_config == "hybrid_rerank"
    assert args.benchmark_collection == "bench_external_chunks"
    assert args.reranker_candidate_k == 20
    assert args.reranker_max_length == 128
    assert args.judge_max_tokens == 2048
    assert "hybrid_rerank_mmr" not in RETRIEVAL_CONFIGS
    assert args.retrieval_config in RETRIEVAL_CONFIGS


def test_benchmark_retriever_enforces_collection_prefix_before_model_loading():
    args = _parser().parse_args(["--benchmark-collection", "ai_papers"])
    with pytest.raises(ValueError, match="must start with 'bench_'"):
        build_benchmark_retriever(args)


def test_sparse_is_allowed_only_when_explicitly_selected(tmp_path):
    from processing.bm25_indexer import BM25Indexer
    from processing.chunker import Chunk

    path = tmp_path / "external.pkl"
    index = BM25Indexer()
    index.build(
        [
            Chunk(
                chunk_id="c1",
                paper_id="p1",
                section="body",
                text="benchmark text",
                start_char=0,
                end_char=14,
                metadata={},
            )
        ]
    )
    index.save(path)
    args = _parser().parse_args(
        [
            "--retrieval-config",
            "sparse",
            "--external-index",
            str(path),
        ]
    )
    retriever, client = build_benchmark_retriever(args)
    assert isinstance(retriever, SparseRetriever)
    assert client is None


def test_reranking_wrapper_retrieves_fifty_candidates_then_returns_top_k():
    from evaluation.run_full_ragas_eval import _RerankingRetriever

    class Retriever:
        def __init__(self):
            self.kwargs = None

        def search(self, query, **kwargs):
            self.kwargs = kwargs
            return [
                RetrievalResult(str(i), str(i), 1.0, "hybrid")
                for i in range(kwargs["top_k"])
            ]

    class Reranker:
        def rerank(self, query, candidates, top_k):
            assert len(candidates) == 20
            return list(reversed(candidates))[:top_k]

    base = Retriever()
    wrapped = _RerankingRetriever(base, Reranker())
    results = wrapped.search("query", top_k=4)
    assert base.kwargs == {"top_k": 20, "candidate_top_k": 50}
    assert [row.chunk_id for row in results] == ["19", "18", "17", "16"]
