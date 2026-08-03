"""Evaluate retrieval stages independently from answer generation.

The evaluator accepts injected pipeline components for tests and provides a CLI
for the project's local Qdrant and BM25 artifacts. Relevance is chunk-level
when reviewed chunk labels exist; otherwise it falls back explicitly to the
paper labels in the golden record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    MMRSampler,
    RetrievalResult,
    SparseRetriever,
)

from .metrics import ranked_metrics, recall_at_k

KS = (5, 8, 20)
CONFIGS = ("dense", "sparse", "hybrid_rrf", "hybrid_rerank", "hybrid_rerank_mmr")


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a retrieval golden set."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = (
        payload.get("questions", payload) if isinstance(payload, dict) else payload
    )
    if not isinstance(records, list) or not records:
        raise ValueError("golden set must contain a non-empty question list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for position, item in enumerate(records):
        if not isinstance(item, dict):
            raise TypeError(f"golden record {position} must be an object")
        question = str(item.get("question", item.get("query", ""))).strip()
        query_id = str(item.get("query_id", f"q{position + 1:03d}")).strip()
        chunk_ids = _string_list(item.get("relevant_chunk_ids", []))
        paper_ids = _string_list(item.get("relevant_paper_ids", []))
        if not question or not query_id:
            raise ValueError(f"golden record {position} is missing question/query_id")
        if query_id in seen:
            raise ValueError(f"duplicate golden query_id: {query_id}")
        if not chunk_ids and not paper_ids:
            raise ValueError(f"golden record {query_id} has no relevance labels")
        seen.add(query_id)
        normalized.append(
            {
                **item,
                "query_id": query_id,
                "question": question,
                "relevant_chunk_ids": chunk_ids,
                "relevant_paper_ids": paper_ids,
            }
        )
    return normalized


class RetrievalEvaluator:
    """Run and score the dense-to-MMR retrieval ablation."""

    def __init__(
        self,
        *,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        hybrid: HybridRetriever,
        reranker: CrossEncoderReranker,
        mmr: MMRSampler,
        candidate_k: int = 20,
        final_k: int = 20,
        reranker_lift_threshold: float = 0.02,
    ) -> None:
        if candidate_k < max(KS) or final_k < max(KS):
            raise ValueError("candidate_k and final_k must be at least 20")
        self.dense, self.sparse, self.hybrid = dense, sparse, hybrid
        self.reranker, self.mmr = reranker, mmr
        self.candidate_k, self.final_k = candidate_k, final_k
        self.reranker_lift_threshold = float(reranker_lift_threshold)

    def evaluate(self, golden: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Execute all configurations and return aggregate plus query diagnostics."""

        runs: dict[str, list[dict[str, Any]]] = {name: [] for name in CONFIGS}
        per_query: list[dict[str, Any]] = []
        lift_rows: list[dict[str, Any]] = []
        all_failed: list[dict[str, str]] = []

        for position, item in enumerate(golden, 1):
            print(
                f"[{position}/{len(golden)}] {item['query_id']}: retrieval",
                file=sys.stderr,
                flush=True,
            )
            query_id, question = item["query_id"], item["question"]

            started = time.perf_counter()
            dense_results = self.dense.search(question, top_k=self.candidate_k)
            dense_ms = _elapsed_ms(started)

            started = time.perf_counter()
            sparse_results = self.sparse.search(question, top_k=self.candidate_k)
            sparse_ms = _elapsed_ms(started)

            started = time.perf_counter()
            hybrid_results = self.hybrid.fuse(
                dense_results, sparse_results, top_k=self.candidate_k
            )
            hybrid_ms = dense_ms + sparse_ms + _elapsed_ms(started)

            started = time.perf_counter()
            reranked_results = self.reranker.rerank(
                question, hybrid_results, top_k=self.final_k
            )
            rerank_ms = hybrid_ms + _elapsed_ms(started)

            started = time.perf_counter()
            mmr_results = self.mmr.sample(
                question, reranked_results, top_k=self.final_k
            )
            mmr_ms = rerank_ms + _elapsed_ms(started)

            result_sets = {
                "dense": (dense_results, dense_ms),
                "sparse": (sparse_results, sparse_ms),
                "hybrid_rrf": (hybrid_results, hybrid_ms),
                "hybrid_rerank": (reranked_results, rerank_ms),
                "hybrid_rerank_mmr": (mmr_results, mmr_ms),
            }
            query_hits: list[bool] = []
            for config, (results, latency_ms) in result_sets.items():
                ranking, relevant, grain = _ranking_and_labels(results, item)
                metrics = ranked_metrics(ranking, relevant, ks=KS)
                query_hits.append(bool(metrics["hit@20"]))
                runs[config].append(
                    {
                        "query_id": query_id,
                        "latency_ms": latency_ms,
                        "metrics": metrics,
                        "results": [_serialize_result(result) for result in results],
                    }
                )
                per_query.append(
                    {
                        "query_id": query_id,
                        "question": question,
                        "config": config,
                        "evaluation_grain": grain,
                        "latency_ms": latency_ms,
                        **metrics,
                    }
                )

            before_ranking, relevant, grain = _ranking_and_labels(hybrid_results, item)
            after_ranking, _, _ = _ranking_and_labels(reranked_results, item)
            pre8 = recall_at_k(before_ranking, relevant, 8)
            pre20 = recall_at_k(before_ranking, relevant, 20)
            post8 = recall_at_k(after_ranking, relevant, 8)
            lift_rows.append(
                {
                    "query_id": query_id,
                    "evaluation_grain": grain,
                    "pre_rerank_recall@8": pre8,
                    "pre_rerank_recall@20": pre20,
                    "post_rerank_recall@8": post8,
                    "recall@8_lift": post8 - pre8,
                }
            )
            if not any(query_hits):
                all_failed.append({"query_id": query_id, "question": question})

        aggregate = {config: _aggregate(runs[config]) for config in CONFIGS}
        pre8 = statistics.fmean(row["pre_rerank_recall@8"] for row in lift_rows)
        pre20 = statistics.fmean(row["pre_rerank_recall@20"] for row in lift_rows)
        post8 = statistics.fmean(row["post_rerank_recall@8"] for row in lift_rows)
        lift = post8 - pre8
        targeted = {
            "pre_rerank_recall@20": pre20,
            "pre_rerank_recall@8": pre8,
            "post_rerank_recall@8": post8,
            "recall@8_lift": lift,
            "promotion_efficiency": (post8 - pre8) / (pre20 - pre8)
            if pre20 > pre8
            else 0.0,
            "meaningful_lift_threshold": self.reranker_lift_threshold,
            "passed": lift >= self.reranker_lift_threshold,
            "flag": None
            if lift >= self.reranker_lift_threshold
            else "Reranked top-8 recall did not meaningfully improve over hybrid top-8 recall.",
            "per_query": lift_rows,
        }
        grain_counts = {
            "chunk": sum(bool(item["relevant_chunk_ids"]) for item in golden),
            "paper_fallback": sum(not item["relevant_chunk_ids"] for item in golden),
        }
        return {
            "aggregate": aggregate,
            "reranker_lift": targeted,
            "label_coverage": grain_counts,
            "all_configs_failed": all_failed,
            "per_query": per_query,
            "rankings": runs,
        }


def save_outputs(
    evaluation: dict[str, Any],
    output_dir: str | Path,
    *,
    golden_path: str | Path,
    runtime_config: dict[str, Any],
) -> dict[str, Path]:
    """Save timestamped JSON, CSV, Markdown, and a stable ranking cache."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output / f"retrieval_eval_{stamp}"
    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "golden_path": str(Path(golden_path)),
        "golden_sha256": hashlib.sha256(Path(golden_path).read_bytes()).hexdigest(),
        "runtime_config": runtime_config,
        **evaluation,
    }
    json_path, csv_path, md_path = (
        base.with_suffix(".json"),
        base.with_suffix(".csv"),
        base.with_suffix(".md"),
    )
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    columns = [
        "config",
        "recall@5",
        "recall@8",
        "recall@20",
        "precision@5",
        "precision@8",
        "precision@20",
        "mrr",
        "ndcg@5",
        "ndcg@8",
        "ndcg@20",
        "avg_latency_ms",
    ]
    rows = [{"config": name, **evaluation["aggregate"][name]} for name in CONFIGS]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    md_path.write_text(_markdown(rows, evaluation), encoding="utf-8")
    cache_dir = output.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "retrieval_candidates.json"
    cache_path.write_text(
        json.dumps(
            {
                "created_at": document["created_at"],
                "golden_sha256": document["golden_sha256"],
                "rankings": evaluation["rankings"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "cache": cache_path,
    }


def build_local_evaluator(args: argparse.Namespace) -> RetrievalEvaluator:
    """Construct production components from local artifacts and environment."""

    from dotenv import load_dotenv
    from processing.embedder import Embedder
    from qdrant_client import QdrantClient

    load_dotenv()
    cache_folder = str(Path(args.model_cache).resolve())
    os.environ.setdefault("HF_HOME", cache_folder)
    embedding_reference = _local_model_reference(
        args.embedding_model, Path(cache_folder)
    )
    embedder = Embedder(embedding_reference)
    if args.qdrant_path:
        client = QdrantClient(path=args.qdrant_path)
    else:
        client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)
    dense = DenseRetriever(
        client, embedder, args.collection, default_top_k=args.candidate_k
    )
    cached_health = dense.health_check()
    if not cached_health.get("connected") or not cached_health.get("collection_exists"):
        raise RuntimeError(f"dense retrieval is unhealthy: {cached_health}")
    dense.health_check = lambda: cached_health

    sparse = SparseRetriever(args.bm25_index, default_top_k=args.candidate_k)
    hybrid = HybridRetriever(
        dense,
        sparse,
        rrf_k=args.rrf_k,
        default_top_k=args.candidate_k,
        candidate_top_k=args.candidate_k,
    )
    reranker_reference = _local_model_reference(args.reranker_model, Path(cache_folder))
    reranker = CrossEncoderReranker(reranker_reference, default_top_k=args.final_k)
    mmr = MMRSampler(embedder, lambda_mult=args.mmr_lambda, default_top_k=args.final_k)
    return RetrievalEvaluator(
        dense=dense,
        sparse=sparse,
        hybrid=hybrid,
        reranker=reranker,
        mmr=mmr,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        reranker_lift_threshold=args.lift_threshold,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    golden = load_golden(args.golden)[: args.limit]
    evaluator = build_local_evaluator(args)
    result = evaluator.evaluate(golden)
    paths = save_outputs(
        result,
        args.output_dir,
        golden_path=args.golden,
        runtime_config={
            key: value for key, value in vars(args).items() if key not in {"qdrant_url"}
        },
    )
    print(Path(paths["markdown"]).read_text(encoding="utf-8"))
    print("\nSaved:", *(str(path) for path in paths.values()), sep="\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="evaluation/data/golden_retrieval.json")
    parser.add_argument("--output-dir", default="evaluation/data/eval_results")
    parser.add_argument("--bm25-index", default="data/processed/bm25_index.pkl")
    parser.add_argument("--qdrant-path", default="data/qdrant")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-timeout", type=float, default=30.0)
    parser.add_argument(
        "--collection", default=os.getenv("QDRANT_COLLECTION", "ai_papers")
    )
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--model-cache", default="data/model_cache")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--mmr-lambda", type=float, default=0.5)
    parser.add_argument("--lift-threshold", type=float, default=0.02)
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N questions"
    )
    return parser


def _ranking_and_labels(
    results: Sequence[RetrievalResult], golden: dict[str, Any]
) -> tuple[list[str], list[str], str]:
    chunk_labels = golden["relevant_chunk_ids"]
    if chunk_labels:
        return [result.chunk_id for result in results], chunk_labels, "chunk"
    return (
        [result.paper_id or "" for result in results],
        golden["relevant_paper_ids"],
        "paper_fallback",
    )


def _aggregate(runs: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "recall@5",
        "recall@8",
        "recall@20",
        "precision@5",
        "precision@8",
        "precision@20",
        "mrr",
        "ndcg@5",
        "ndcg@8",
        "ndcg@20",
    ]
    result = {
        key: statistics.fmean(run["metrics"][key] for run in runs) for key in keys
    }
    result["avg_latency_ms"] = statistics.fmean(run["latency_ms"] for run in runs)
    return result


def _serialize_result(result: RetrievalResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["text"] = result.text
    return payload


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("relevance labels must be JSON arrays")
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _local_model_reference(model_name: str, cache_folder: Path) -> str:
    """Use a complete local HF snapshot without an avoidable remote request."""

    model_directory = cache_folder / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots = model_directory / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
        weight_names = ("model.safetensors", "pytorch_model.bin")
        complete = [
            path
            for path in candidates
            if (path / "config.json").is_file()
            and any((path / name).is_file() for name in weight_names)
        ]
        if complete:
            return str(complete[-1])
    return model_name


def _markdown(rows: Sequence[dict[str, Any]], evaluation: dict[str, Any]) -> str:
    columns = [
        "config",
        "R@5",
        "R@8",
        "R@20",
        "P@5",
        "P@8",
        "P@20",
        "MRR",
        "nDCG@5",
        "nDCG@8",
        "nDCG@20",
        "avg ms",
    ]
    keys = [
        "config",
        "recall@5",
        "recall@8",
        "recall@20",
        "precision@5",
        "precision@8",
        "precision@20",
        "mrr",
        "ndcg@5",
        "ndcg@8",
        "ndcg@20",
        "avg_latency_ms",
    ]
    lines = [
        "# Retrieval evaluation",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in rows:
        values = [
            str(row[key]) if key == "config" else f"{row[key]:.4f}" for key in keys
        ]
        lines.append("| " + " | ".join(values) + " |")
    lift = evaluation["reranker_lift"]
    status = "PASS" if lift["passed"] else "FLAG"
    lines.extend(
        [
            "",
            "## Targeted reranker lift",
            "",
            f"**{status}:** hybrid R@8={lift['pre_rerank_recall@8']:.4f}, "
            f"hybrid R@20={lift['pre_rerank_recall@20']:.4f}, "
            f"reranked R@8={lift['post_rerank_recall@8']:.4f}, "
            f"R@8 lift={lift['recall@8_lift']:.4f}.",
            "",
        ]
    )
    if lift["flag"]:
        lines.append(f"> {lift['flag']}\n")
    coverage = evaluation["label_coverage"]
    lines.append(
        f"Label grain: {coverage['chunk']} chunk-reviewed questions; "
        f"{coverage['paper_fallback']} paper-label fallback questions."
    )
    lines.append(
        f"Questions missed by every configuration at @20: {len(evaluation['all_configs_failed'])}."
    )
    lines.extend(["", "## Interpretation", ""])
    metric_names = [
        "recall@5",
        "recall@8",
        "recall@20",
        "precision@5",
        "precision@8",
        "precision@20",
        "mrr",
        "ndcg@5",
        "ndcg@8",
        "ndcg@20",
    ]
    for metric in metric_names:
        best = max(row[metric] for row in rows)
        leaders = [row["config"] for row in rows if abs(row[metric] - best) < 1e-12]
        lines.append(f"- Best {metric}: {', '.join(leaders)} ({best:.4f}).")
    by_config = {row["config"]: row for row in rows}
    hybrid = by_config["hybrid_rrf"]
    dense = by_config["dense"]
    sparse = by_config["sparse"]
    hybrid_beats = [
        metric
        for metric in metric_names
        if hybrid[metric] > dense[metric] and hybrid[metric] > sparse[metric]
    ]
    lines.append(
        f"- Hybrid strictly beats both single retrievers on: "
        f"{', '.join(hybrid_beats) if hybrid_beats else 'no reported metric'}."
    )
    rerank_cost = (
        by_config["hybrid_rerank"]["avg_latency_ms"] - hybrid["avg_latency_ms"]
    )
    mmr_cost = (
        by_config["hybrid_rerank_mmr"]["avg_latency_ms"]
        - by_config["hybrid_rerank"]["avg_latency_ms"]
    )
    lines.append(
        f"- Reranking adds {rerank_cost:.2f} ms/query and MMR adds another "
        f"{mmr_cost:.2f} ms/query relative to the preceding stage."
    )
    failures = evaluation["all_configs_failed"]
    if failures:
        lines.append(
            "- All-config @20 failures: "
            + ", ".join(item["query_id"] for item in failures)
            + "."
        )
    if coverage["paper_fallback"]:
        lines.append(
            "- Caveat: these are paper-level fallback metrics until chunk labels "
            "are manually reviewed."
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
