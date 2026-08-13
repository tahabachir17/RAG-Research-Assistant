from __future__ import annotations

import json

import pytest

from processing.bm25_indexer import BM25Indexer
from evaluation.retrieval_stack_diagnostic import (
    actual_smoke_inventory,
    correlate_failures,
    ensure_benchmark_dense_collection,
    load_reviewed_benchmark_golden,
)


def test_benchmark_collection_requires_prefix(tmp_path):
    with pytest.raises(ValueError, match="must start with 'bench_'"):
        ensure_benchmark_dense_collection(
            tmp_path / "bm25.pkl",
            tmp_path / "qdrant",
            "external_chunks",
            embedding_model="ignored",
            model_cache=tmp_path,
        )


def _write(path, questions):
    path.write_text(
        json.dumps({"schema_version": 1, "questions": questions}), encoding="utf-8"
    )


def test_reviewed_golden_excludes_unreviewed_and_preserves_evidence(tmp_path):
    _write(
        tmp_path / "qasa_generation_qa.json",
        [
            {
                "id": "qasa-1",
                "question": "Why?",
                "reviewed": True,
                "reference_context_ids": ["gold"],
            },
            {
                "id": "qasa-2",
                "question": "Why not?",
                "reviewed": False,
                "reference_context_ids": [],
            },
        ],
    )
    _write(tmp_path / "qasper_generation_qa.json", [])

    rows = load_reviewed_benchmark_golden(tmp_path)

    assert rows == [
        {
            "query_id": "qasa-1",
            "question": "Why?",
            "source_tier": "qasa",
            "relevant_chunk_ids": ["gold"],
            "relevant_paper_ids": [],
        }
    ]


def test_inventory_reports_actual_sparse_external_path():
    rows = actual_smoke_inventory(
        production_index="production.pkl",
        external_index="external.pkl",
        external_top_k=4,
    )
    assert rows[0]["retriever_class"].startswith("none")
    assert rows[1]["retriever_class"] == "BM25Indexer.search"
    assert rows[1]["top_k"] == 4
    assert rows[1]["rerank"] is False
    assert rows[1]["mmr"] is False


def test_failure_correlation_reports_ranks_and_unreviewed_rows():
    index = BM25Indexer()
    index.chunks = [
        {"chunk_id": "gold", "text": "Needed fact.", "section": "results"}
    ]
    golden = [
        {
            "query_id": "reviewed",
            "question": "Question?",
            "relevant_chunk_ids": ["gold"],
        }
    ]
    run = {
        "query_id": "reviewed",
        "results": [{"chunk_id": "gold", "text": "Needed fact."}],
    }
    evaluation = {
        "rankings": {
            config: [run]
            for config in (
                "dense",
                "sparse",
                "hybrid_rrf",
                "hybrid_rerank",
                "hybrid_rerank_mmr",
            )
        }
    }

    rows = correlate_failures(
        evaluation, golden, index, query_ids=("reviewed", "unreviewed")
    )

    assert rows[0]["ranks"]["dense"] == [1]
    assert rows[0]["evidence"][0]["text"] == "Needed fact."
    assert rows[1]["status"] == "unreviewed"
