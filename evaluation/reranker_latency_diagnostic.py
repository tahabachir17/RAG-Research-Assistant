"""Measure cross-encoder load, cold prediction, and warm prediction latency."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Sequence

from processing.bm25_indexer import BM25Indexer

from .retrieval_evaluator import _local_model_reference


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "evaluation/data/external_benchmarks/external_bm25_index.pkl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/data/eval_results/"
            "retrieval_stack_diagnostic_20260812/reranker_latency.json"
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--thread-counts",
        default="1,2,4,8",
        help="Comma-separated PyTorch CPU thread counts to test on 20 candidates.",
    )
    args = parser.parse_args(argv)

    from sentence_transformers import CrossEncoder

    reference = _local_model_reference(args.model, args.model_cache)
    texts = [
        str(row["text"]) for row in BM25Indexer.load(args.index).chunks[:50]
    ]
    query = "What findings were reported?"
    started = time.perf_counter()
    model = CrossEncoder(reference)
    load_seconds = time.perf_counter() - started
    result = {
        "model": reference,
        "load_seconds": load_seconds,
        "batch_size": 32,
        "predict": {},
        "thread_sweep": {},
    }
    for candidate_count in (20, 50):
        timings = []
        pairs = [(query, text) for text in texts[:candidate_count]]
        for _ in range(args.repeats + 1):
            started = time.perf_counter()
            model.predict(pairs, batch_size=32)
            timings.append(time.perf_counter() - started)
        result["predict"][str(candidate_count)] = {
            "cold_seconds": timings[0],
            "warm_seconds": timings[1:],
            "warm_median_seconds": statistics.median(timings[1:]),
        }
    import torch

    pairs = [(query, text) for text in texts[:20]]
    for thread_count in (int(value) for value in args.thread_counts.split(",")):
        torch.set_num_threads(thread_count)
        started = time.perf_counter()
        model.predict(pairs, batch_size=32)
        result["thread_sweep"][str(thread_count)] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
