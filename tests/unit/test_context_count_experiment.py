from __future__ import annotations

from evaluation.generation_golden import GenerationGoldenQuestion
from evaluation.run_context_count_experiment import build_context_pools, save_experiment
from processing.bm25_indexer import BM25Indexer
from processing.chunker import Chunk


def _question():
    return GenerationGoldenQuestion(
        "q1",
        "What does the method contribute?",
        ["frozen-1", "frozen-2"],
        [],
        {},
        [],
        3,
        False,
        [],
        "1234.5678v2",
        "Paper",
    )


def test_context_pool_is_nested_and_keeps_frozen_chunks_first():
    index = BM25Indexer()
    index.build(
        [
            Chunk("frozen-1", "1234.5678v2", "abstract", "method contribution", 0, 1, {}),
            Chunk("frozen-2", "1234.5678v2", "results", "strong results", 0, 1, {}),
            Chunk("extra-1", "1234.5678v2", "methodology", "method details", 0, 1, {}),
            Chunk("extra-2", "1234.5678v2", "limitations", "method limitations", 0, 1, {}),
            Chunk("other", "9999.0000v1", "abstract", "method contribution", 0, 1, {}),
        ]
    )

    pools = build_context_pools([_question()], index, max_count=4)

    assert pools["q1"][:2] == ["frozen-1", "frozen-2"]
    assert len(pools["q1"]) == 4
    assert "other" not in pools["q1"]


def test_context_experiment_writes_raw_metric_artifacts(tmp_path):
    result = {
        "runs": [
            {
                "chunk_count": 4,
                "ragas": {
                    "aggregate": {
                        "faithfulness": 0.5,
                        "answer_relevancy": 0.6,
                        "context_utilization": 0.7,
                    }
                },
            }
        ]
    }

    paths = save_experiment(result, tmp_path)

    assert all(path.is_file() for path in paths.values())
    assert "| 4 | 0.5000 | 0.6000 | 0.7000 |" in paths["markdown"].read_text(
        encoding="utf-8"
    )
