"""Run the combined manual + external Part 9 generation RAGAS evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from config import Settings
from generation import build_llm_client
from processing.bm25_indexer import BM25Indexer
from retrieval import (
    CrossEncoderReranker,
    SparseRetriever,
    build_retriever,
)

RETRIEVAL_CONFIGS = {
    "dense",
    "sparse",
    "hybrid_rrf",
    "hybrid_rerank",
}
BENCHMARK_COLLECTION_PREFIX = "bench_"

try:
    from .full_ragas_evaluation import (
        FULL_RAGAS_METRICS,
        MetricJsonlCache,
        TieredChunkLookup,
        aggregate_scores,
        assemble_evaluation_questions,
        metric_rows_from_cache,
        retrieve_external_contexts,
        write_full_ragas_outputs,
    )
    from .generation_eval_checkpoint import GenerationEvalCheckpoint
    from .generation_evaluator import GenerationEvaluator, build_generation_result
    from .ragas_evaluator import build_ragas_clients, build_ragas_records, evaluate_with_ragas
    from .rate_limit_client import EvaluationRateLimitClient, _is_transient
except ImportError:
    from full_ragas_evaluation import (
        FULL_RAGAS_METRICS,
        MetricJsonlCache,
        TieredChunkLookup,
        aggregate_scores,
        assemble_evaluation_questions,
        metric_rows_from_cache,
        retrieve_external_contexts,
        write_full_ragas_outputs,
    )
    from generation_eval_checkpoint import GenerationEvalCheckpoint
    from generation_evaluator import GenerationEvaluator, build_generation_result
    from ragas_evaluator import build_ragas_clients, build_ragas_records, evaluate_with_ragas
    from rate_limit_client import EvaluationRateLimitClient, _is_transient


LOGGER = logging.getLogger(__name__)


class _RerankingRetriever:
    """Apply one cached reranker behind the generation search interface."""

    def __init__(self, retriever: Any, reranker: Any, candidate_k: int = 20) -> None:
        self.retriever, self.reranker = retriever, reranker
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 4, **_: Any) -> list[Any]:
        candidates = self.retriever.search(
            query, top_k=self.candidate_k, candidate_top_k=50
        )
        return self.reranker.rerank(query, candidates, top_k=top_k)


def build_benchmark_retriever(args: argparse.Namespace) -> tuple[Any, Any | None]:
    """Build the benchmark retriever through the production retriever factory."""

    assert args.retrieval_config in RETRIEVAL_CONFIGS
    if not args.benchmark_collection.startswith(BENCHMARK_COLLECTION_PREFIX):
        raise ValueError(
            "benchmark collection must start with "
            f"{BENCHMARK_COLLECTION_PREFIX!r}; got {args.benchmark_collection!r}"
        )
    client = None
    embedder = None
    if args.retrieval_config != "sparse":
        from processing.embedder import Embedder
        from qdrant_client import QdrantClient

        cache = Path(args.model_cache)
        os.environ.setdefault("HF_HOME", str(cache.resolve()))
        embedder = Embedder(_local_retrieval_model(args.embedding_model, cache))
        client = QdrantClient(path=str(args.benchmark_qdrant_path))
    common = {"default_top_k": args.external_top_k}
    if args.retrieval_config == "sparse":
        retriever = build_retriever(
            {"type": "sparse", "index_path": args.external_index, **common}
        )
    elif args.retrieval_config == "dense":
        retriever = build_retriever(
            {
                "type": "dense",
                "collection_name": args.benchmark_collection,
                **common,
            },
            qdrant_client=client,
            embedder=embedder,
        )
    else:
        hybrid = build_retriever(
            {
                "type": "hybrid",
                "dense": {
                    "collection_name": args.benchmark_collection,
                    "default_top_k": 50,
                },
                "sparse": {
                    "index_path": args.external_index,
                    "default_top_k": 50,
                },
                "rrf_k": args.rrf_k,
                "default_top_k": args.external_top_k,
                "candidate_top_k": 50,
            },
            qdrant_client=client,
            embedder=embedder,
        )
        retriever = hybrid
        if args.retrieval_config == "hybrid_rerank":
            import torch

            torch.set_num_threads(args.reranker_cpu_threads)
            started = time.perf_counter()
            reranker = CrossEncoderReranker(
                _local_retrieval_model(
                    args.reranker_model, Path(args.model_cache)
                ),
                default_top_k=args.external_top_k,
                batch_size=args.reranker_batch_size,
                max_length=args.reranker_max_length,
            )
            LOGGER.info(
                "Loaded benchmark reranker once in %.3fs (candidate_k=%d, "
                "max_length=%d, cpu_threads=%d)",
                time.perf_counter() - started,
                args.reranker_candidate_k,
                args.reranker_max_length,
                args.reranker_cpu_threads,
            )
            retriever = _RerankingRetriever(
                hybrid, reranker, candidate_k=args.reranker_candidate_k
            )
    if args.retrieval_config != "sparse":
        assert not isinstance(retriever, (BM25Indexer, SparseRetriever))
    LOGGER.info(
        "Benchmark retrieval config=%s instantiated=%s",
        args.retrieval_config,
        type(retriever).__name__,
    )
    return retriever, client


def _local_retrieval_model(model_name: str, cache: Path) -> str:
    try:
        from .retrieval_evaluator import _local_model_reference
    except ImportError:
        from retrieval_evaluator import _local_model_reference

    return _local_model_reference(model_name, cache)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    base_settings = Settings()
    judge_provider, judge_model = _resolve_ragas_target(
        base_settings, args.judge_provider, args.judge_model
    )
    settings = base_settings.model_copy(
        update={
            "LLM_PROVIDER": args.generator_provider,
            "LLM_MODEL": args.generator_model,
            "RAGAS_JUDGE_PROVIDER": judge_provider,
            "RAGAS_JUDGE_MODEL": judge_model,
            "RAGAS_REQUESTS_PER_SECOND": args.requests_per_second,
            "JUDGE_MAX_TOKENS": args.judge_max_tokens,
        }
    )
    _validate_credentials(settings)
    questions = assemble_evaluation_questions(
        args.manual,
        args.external_dir,
        manual_limit=args.manual_limit,
        qasa_limit=args.qasa_limit,
        qasper_limit=args.qasper_limit,
    )
    production_index = BM25Indexer.load(args.production_index)
    external_retriever, benchmark_client = build_benchmark_retriever(args)
    try:
        questions = retrieve_external_contexts(
            questions, external_retriever, top_k=args.external_top_k
        )
    finally:
        if benchmark_client is not None:
            benchmark_client.close()
    external_index = BM25Indexer.load(args.external_index)
    lookup = TieredChunkLookup(production_index, external_index)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"full_ragas_eval_{run_id}"
    checkpoint_path = run_dir / "generation_checkpoint.json"
    checkpoint = (
        GenerationEvalCheckpoint.load(checkpoint_path)
        if checkpoint_path.exists()
        else GenerationEvalCheckpoint.create(
            checkpoint_path,
            {
                "questions": [],
                "question_order": [question.id for question in questions],
                "errors": [],
            },
        )
    )
    if checkpoint.payload.get("question_order") != [row.id for row in questions]:
        raise ValueError("run-id already exists with a different question selection")

    generator = EvaluationRateLimitClient(
        build_llm_client(settings),
        max_retries=args.retries,
        default_wait_seconds=args.backoff_seconds,
        backoff_seconds=args.backoff_seconds,
        requests_per_second=args.generation_requests_per_second,
    )
    evaluator = GenerationEvaluator(
        llm=generator,
        chunk_lookup=lookup,
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        judge=None,
        max_retries=settings.GENERATION_MAX_RETRIES,
        max_context_tokens=args.max_context_tokens,
    )
    completed = checkpoint.completed_question_ids()
    for question in questions:
        if question.id in completed:
            continue
        try:
            checkpoint.record_question(evaluator.evaluate_one(question))
            LOGGER.info("Checkpointed generation for %s", question.id)
        except Exception as exc:
            checkpoint.payload.setdefault("errors", []).append(
                {"question_id": question.id, "stage": "generation", "error": f"{type(exc).__name__}: {exc}"}
            )
            checkpoint.save()
            LOGGER.error("Generation failed for %s: %s", question.id, exc)

    order = {item: index for index, item in enumerate(checkpoint.payload["question_order"])}
    generated = sorted(
        checkpoint.payload.get("questions", []),
        key=lambda row: order.get(str(row["id"]), len(order)),
    )
    base = build_generation_result(
        generated, provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, judge=None
    )
    question_by_id = {row.id: row for row in questions}
    cache = MetricJsonlCache(run_dir / ".cache" / f"{run_id}.jsonl")
    completed_metrics = cache.completed()
    ragas_llm, embeddings = build_ragas_clients(settings)
    for generation_row in generated:
        question = question_by_id[str(generation_row["id"])]
        records = build_ragas_records(
            {"questions": [generation_row]},
            context_lookup=lookup,
            chunk_ids_by_question={question.id: question.retrieved_chunk_ids},
            reference_by_question={question.id: question.reference_answer},
        )
        for metric in FULL_RAGAS_METRICS:
            if (question.id, metric) in completed_metrics:
                continue
            unavailable_reason = _unavailable_reason(question, metric)
            if unavailable_reason:
                cache.append(question.id, metric, status="unavailable", reason=unavailable_reason)
                continue
            try:
                result = _retry(
                    lambda metric=metric: evaluate_with_ragas(
                        records,
                        llm=ragas_llm,
                        embeddings=embeddings,
                        timeout=args.ragas_timeout,
                        max_workers=1,
                        max_retries=args.ragas_max_retries,
                        metric_names=[metric],
                    ),
                    retries=args.retries,
                    backoff_seconds=args.backoff_seconds,
                )
                value = result.get("questions", [{}])[0].get(metric)
                cache.append(
                    question.id,
                    metric,
                    status="completed" if value is not None else "failed",
                    value=value,
                    reason=result.get("reason") if value is None else None,
                )
            except Exception as exc:
                cache.append(
                    question.id,
                    metric,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                LOGGER.error("RAGAS failed for %s/%s: %s", question.id, metric, exc)

    metric_rows, progress = metric_rows_from_cache(
        (str(row["id"]) for row in generated), cache
    )
    for row in base["questions"]:
        row["source_tier"] = row.get("source_tier") or row.get("source_dataset") or "manual"
        row["alignment_status"] = row.get("alignment_status") or "aligned"
    aggregates = aggregate_scores(base["questions"], metric_rows)
    terminal = sum(
        progress.get(str(row["id"]), {}).get(metric) in {"completed", "unavailable"}
        for row in generated
        for metric in FULL_RAGAS_METRICS
    )
    expected = len(generated) * len(FULL_RAGAS_METRICS)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "generator": {"provider": settings.LLM_PROVIDER, "model": settings.LLM_MODEL},
        "judge": {"provider": settings.RAGAS_JUDGE_PROVIDER, "model": settings.RAGAS_JUDGE_MODEL},
        "retrieval": {
            "manual_index": str(args.production_index),
            "external_index": str(args.external_index),
            "external_top_k": args.external_top_k,
            "external_config": args.retrieval_config,
            "benchmark_qdrant_path": str(args.benchmark_qdrant_path),
            "benchmark_collection": args.benchmark_collection,
            "instantiated_class": type(external_retriever).__name__,
            "reranker_candidate_k": (
                args.reranker_candidate_k
                if args.retrieval_config == "hybrid_rerank"
                else None
            ),
            "reranker_batch_size": (
                args.reranker_batch_size
                if args.retrieval_config == "hybrid_rerank"
                else None
            ),
            "reranker_max_length": (
                args.reranker_max_length
                if args.retrieval_config == "hybrid_rerank"
                else None
            ),
            "reranker_cpu_threads": (
                args.reranker_cpu_threads
                if args.retrieval_config == "hybrid_rerank"
                else None
            ),
            "mmr": False,
        },
        "coverage": {
            "requested": len(questions),
            "generated": len(generated),
            "scidqa": 0,
        },
        "questions": base["questions"],
        "ragas": {
            "status": "completed" if terminal == expected else "partial",
            "metrics": list(FULL_RAGAS_METRICS),
            "questions": metric_rows,
            "metric_progress": progress,
        },
        "aggregates": aggregates,
        "errors": checkpoint.payload.get("errors", []),
    }
    paths = write_full_ragas_outputs(run_dir, payload=payload)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0 if len(generated) == len(questions) and terminal == expected else 2


def _unavailable_reason(question: Any, metric: str) -> str | None:
    if metric in {"context_precision", "context_recall"} and (
        question.alignment_status != "aligned" or not question.reference_context_ids
    ):
        return "gold evidence is unavailable or not confidently aligned"
    if metric in {"context_precision", "context_recall", "answer_correctness"} and not question.reference_answer:
        return "no reference answer is available"
    return None


def _retry(
    operation: Callable[[], Any], *, retries: int, backoff_seconds: float
) -> Any:
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == retries or not _is_transient(exc):
                raise
            wait = backoff_seconds * (2**attempt)
            LOGGER.warning("Transient judge failure; retrying in %.1fs: %s", wait, exc)
            time.sleep(wait)
    raise AssertionError("bounded retry loop did not return")


def _validate_credentials(settings: Settings) -> None:
    if settings.LLM_PROVIDER == "groq" and not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required for generation")
    if settings.RAGAS_JUDGE_PROVIDER == "groq" and not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required for the RAGAS judge")


def _resolve_ragas_target(
    settings: Settings,
    requested_provider: str | None,
    requested_model: str | None,
) -> tuple[str, str]:
    """Prefer CLI/env configuration, with the Part 9 Groq defaults otherwise."""

    configured_fields = settings.model_fields_set
    provider = requested_provider or (
        settings.RAGAS_JUDGE_PROVIDER
        if "RAGAS_JUDGE_PROVIDER" in configured_fields
        else "groq"
    )
    model = requested_model or (
        settings.RAGAS_JUDGE_MODEL
        if "RAGAS_JUDGE_MODEL" in configured_fields
        else "llama-3.1-8b-instant"
    )
    return provider, model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, default=Path("evaluation/data/golden_generation_qa.json"))
    parser.add_argument("--external-dir", type=Path, default=Path("evaluation/data/external_benchmarks"))
    parser.add_argument("--production-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--external-index", type=Path, default=Path("evaluation/data/external_benchmarks/external_bm25_index.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/data/eval_results"))
    parser.add_argument("--run-id")
    parser.add_argument("--generator-provider", choices=("groq",), default="groq")
    parser.add_argument("--generator-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--judge-provider", choices=("groq",))
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--external-top-k", type=int, default=4)
    parser.add_argument(
        "--retrieval-config",
        choices=sorted(RETRIEVAL_CONFIGS),
        default="hybrid_rerank",
    )
    parser.add_argument(
        "--benchmark-qdrant-path",
        type=Path,
        default=Path("evaluation/data/external_benchmarks/qdrant"),
    )
    parser.add_argument("--benchmark-collection", default="bench_external_chunks")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--reranker-candidate-k", type=int, default=20)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--reranker-max-length", type=int, default=128)
    parser.add_argument("--reranker-cpu-threads", type=int, default=8)
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--manual-limit", type=int)
    parser.add_argument("--qasa-limit", type=int)
    parser.add_argument("--qasper-limit", type=int)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument("--requests-per-second", type=float, default=0.05)
    parser.add_argument("--generation-requests-per-second", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-max-retries", type=int, default=1)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
