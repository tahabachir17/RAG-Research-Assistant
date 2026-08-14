"""Evaluate the authored controlled benchmark against the production BM25 corpus."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from evaluation.metrics import ranked_metrics
from processing.bm25_indexer import BM25Indexer


def evaluate(benchmark: dict[str, Any], index: BM25Indexer) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for question in benchmark["questions"]:
        started = time.perf_counter()
        results = index.search(question["question"], top_k=20)
        latency_ms = (time.perf_counter() - started) * 1000.0
        ranking = [str(row["chunk_id"]) for row in results]
        metrics = ranked_metrics(ranking, question["relevant_chunk_ids"], ks=(4, 8, 20))
        rows.append(
            {
                "question_id": question["id"],
                "paper_id": question["paper_id"],
                "title": question["title"],
                "difficulty": question["difficulty"],
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "best_gold_rank": _best_rank(ranking, question["relevant_chunk_ids"]),
                "latency_ms": latency_ms,
                **metrics,
            }
        )
    return {
        "schema_version": 1,
        "retriever": "production_bm25",
        "corpus_chunks": len(index.chunks),
        "questions": len(rows),
        "aggregate": {
            group: _aggregate(
                rows if group == "overall" else [row for row in rows if row["difficulty"] == group]
            )
            for group in ("overall", "easy", "moderate")
        },
        "per_question": rows,
    }


def _best_rank(ranking: Sequence[str], relevant: Sequence[str]) -> int | None:
    labels = set(relevant)
    return next((rank for rank, item in enumerate(ranking, 1) if item in labels), None)


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "questions": len(rows),
        **{
            metric: statistics.fmean(float(row[metric]) for row in rows)
            for metric in (
                "recall@4", "hit@4", "mrr", "ndcg@4",
                "recall@8", "hit@8", "ndcg@8",
                "recall@20", "hit@20", "ndcg@20",
            )
        },
        "p50_latency_ms": statistics.median(float(row["latency_ms"]) for row in rows),
    }


def write_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "controlled_retrieval_scores.json"
    csv_path = output_dir / "controlled_retrieval_scores.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["per_question"][0]))
        writer.writeheader()
        writer.writerows(payload["per_question"])
    return {"json": json_path, "csv": csv_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("evaluation/data/controlled_retrieval_qa.json")
    )
    parser.add_argument(
        "--index", type=Path, default=Path("data/processed/bm25_index.pkl")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/data/eval_results/controlled_retrieval_20260813"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    payload = evaluate(benchmark, BM25Indexer.load(args.index))
    paths = write_outputs(payload, args.output_dir)
    print(json.dumps({"aggregate": payload["aggregate"], "paths": {key: str(value) for key, value in paths.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
