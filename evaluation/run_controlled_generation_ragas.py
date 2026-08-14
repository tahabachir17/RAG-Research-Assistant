"""Evaluate generation on reviewed gold contexts with RAGAS and judge fallback.

The benchmark deliberately bypasses retrieval: every answer is generated from the
reviewed reference chunk IDs. Gemini Flash Lite is the primary RAGAS judge and
Groq-hosted Llama 3.1 8B is used only when a primary metric call raises or returns
no finite score.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

from config import Settings
from generation import build_llm_client
from processing.bm25_indexer import BM25Indexer

from .full_ragas_evaluation import (
    FULL_RAGAS_METRICS,
    MetricJsonlCache,
    TieredChunkLookup,
    aggregate_scores,
    metric_rows_from_cache,
    write_full_ragas_outputs,
)
from .generation_eval_checkpoint import GenerationEvalCheckpoint
from .generation_evaluator import GenerationEvaluator, build_generation_result
from .generation_golden import load_generation_golden
from .ragas_evaluator import (
    _ragas_llm_options,
    build_ragas_clients,
    build_ragas_records,
    evaluate_with_ragas,
)
from .rate_limit_client import EvaluationRateLimitClient, _is_transient


LOGGER = logging.getLogger(__name__)


def validate_controlled_golden(golden_path: Path, index: BM25Indexer) -> list[Any]:
    """Require reviewed references and exact, locally available frozen evidence."""

    questions = load_generation_golden(golden_path, require_reviewed=True)
    available = {str(chunk.get("chunk_id", "")) for chunk in index.chunks}
    errors: list[str] = []
    for question in questions:
        if not question.reference_answer:
            errors.append(f"{question.id}: reference_answer is missing")
        if question.retrieved_chunk_ids != question.reference_context_ids:
            errors.append(f"{question.id}: frozen and reference context IDs differ")
        missing = [item for item in question.retrieved_chunk_ids if item not in available]
        if missing:
            errors.append(f"{question.id}: index is missing {missing}")
    if errors:
        raise ValueError("invalid controlled generation golden set: " + "; ".join(errors))
    if not questions:
        raise ValueError("controlled generation golden set is empty")
    return questions


def _settings(args: argparse.Namespace, *, judge_provider: str, judge_model: str) -> Settings:
    return Settings(
        LLM_PROVIDER=args.generator_provider,
        LLM_MODEL=args.generator_model,
        ENABLE_FAITHFULNESS_VERIFIER=False,
        RAGAS_JUDGE_PROVIDER=judge_provider,
        RAGAS_JUDGE_MODEL=judge_model,
        RAGAS_REQUESTS_PER_SECOND=args.judge_requests_per_second,
        JUDGE_MAX_TOKENS=args.judge_max_tokens,
        LLM_REQUEST_TIMEOUT_SECONDS=args.request_timeout,
    )


def _retry(operation: Callable[[], Any], args: argparse.Namespace) -> Any:
    for attempt in range(args.retries + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == args.retries or not _is_transient(exc):
                raise
            wait = args.backoff_seconds * (2**attempt)
            LOGGER.warning("Transient evaluation failure; retrying in %.1fs: %s", wait, exc)
            time.sleep(wait)
    raise AssertionError("bounded retry loop did not return")


def _evaluate_metric(
    record: list[dict[str, Any]],
    metric: str,
    llm: Any,
    embeddings: Any,
    args: argparse.Namespace,
) -> float:
    result = _retry(
        lambda: evaluate_with_ragas(
            record,
            llm=llm,
            embeddings=embeddings,
            timeout=args.ragas_timeout,
            max_workers=1,
            max_retries=args.ragas_max_retries,
            metric_names=[metric],
        ),
        args,
    )
    value = result.get("questions", [{}])[0].get(metric)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(result.get("reason") or "metric returned no finite score")
    return float(value)


def _difficulty_by_id(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["id"]): str(row.get("difficulty", "unspecified"))
        for row in payload["questions"]
    }


def _aggregate_by_difficulty(
    questions: list[dict[str, Any]], metric_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in metric_rows}
    output: list[dict[str, Any]] = []
    for difficulty in sorted({str(row["difficulty"]) for row in questions}):
        selected = [row for row in questions if row["difficulty"] == difficulty]
        for metric in FULL_RAGAS_METRICS:
            values = [by_id.get(str(row["id"]), {}).get(metric) for row in selected]
            finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
            output.append(
                {
                    "difficulty": difficulty,
                    "metric": metric,
                    "mean": sum(finite) / len(finite) if finite else None,
                    "scored": len(finite),
                    "unavailable": len(selected) - len(finite),
                    "total": len(selected),
                }
            )
    return output


def run(args: argparse.Namespace) -> int:
    index = BM25Indexer.load(args.production_index)
    questions = validate_controlled_golden(args.golden, index)
    if args.limit is not None:
        questions = questions[: args.limit]
    lookup = TieredChunkLookup(index)
    difficulty = _difficulty_by_id(args.golden)
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    primary_settings = _settings(
        args, judge_provider=args.judge_provider, judge_model=args.judge_model
    )
    fallback_settings = _settings(
        args,
        judge_provider=args.fallback_judge_provider,
        judge_model=args.fallback_judge_model,
    )
    if not (primary_settings.GEMINI_API_KEY or primary_settings.OPENAI_API_KEY):
        raise ValueError(
            "GEMINI_API_KEY or OPENAI_API_KEY is required for the primary Gemini judge"
        )
    if not fallback_settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required for generation and Llama fallback")

    signature = {
        "golden": str(args.golden.resolve()),
        "production_index": str(args.production_index.resolve()),
        "question_ids": [row.id for row in questions],
        "generator": [args.generator_provider, args.generator_model],
        "primary_judge": [args.judge_provider, args.judge_model],
        "fallback_judge": [args.fallback_judge_provider, args.fallback_judge_model],
    }
    checkpoint_path = run_dir / "generation_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = GenerationEvalCheckpoint.load(checkpoint_path)
        if checkpoint.payload.get("run_signature") != signature:
            raise ValueError("run-id already exists with a different benchmark signature")
    else:
        checkpoint = GenerationEvalCheckpoint.create(
            checkpoint_path,
            {"schema_version": 1, "run_signature": signature, "questions": [], "errors": []},
        )

    generator = EvaluationRateLimitClient(
        build_llm_client(primary_settings),
        max_retries=args.retries,
        default_wait_seconds=args.backoff_seconds,
        backoff_seconds=args.backoff_seconds,
        requests_per_second=args.generation_requests_per_second,
    )
    evaluator = GenerationEvaluator(
        llm=generator,
        chunk_lookup=lookup,
        provider=args.generator_provider,
        model=args.generator_model,
        judge=None,
        max_retries=primary_settings.GENERATION_MAX_RETRIES,
        max_context_tokens=args.max_context_tokens,
        enable_faithfulness_verifier=False,
        evidence_packing_mode="adjacent",
        adjacent_chunk_lookup=lookup.adjacent_chunks,
        section_chunk_lookup=lookup.section_chunks,
    )
    completed = checkpoint.completed_question_ids()
    for question in questions:
        if question.id in completed:
            continue
        try:
            checkpoint.record_question(evaluator.evaluate_one(question))
            LOGGER.info("Generated and checkpointed %s", question.id)
        except Exception as exc:
            checkpoint.payload.setdefault("errors", []).append(
                {"question_id": question.id, "stage": "generation", "error": f"{type(exc).__name__}: {exc}"}
            )
            checkpoint.save()
            LOGGER.error("Generation failed for %s: %s", question.id, exc)

    order = {question.id: position for position, question in enumerate(questions)}
    generated = sorted(
        checkpoint.payload.get("questions", []),
        key=lambda row: order.get(str(row["id"]), len(order)),
    )
    base = build_generation_result(
        generated, provider=args.generator_provider, model=args.generator_model, judge=None
    )
    primary_llm, embeddings = build_ragas_clients(primary_settings)
    from langchain_openai import ChatOpenAI

    fallback_llm = ChatOpenAI(**_ragas_llm_options(fallback_settings))
    cache = MetricJsonlCache(run_dir / "metric_cache.jsonl")
    terminal = cache.completed()
    question_by_id = {question.id: question for question in questions}
    for generation_row in generated:
        question = question_by_id[str(generation_row["id"])]
        record = build_ragas_records(
            {"questions": [generation_row]},
            context_lookup=lookup,
            chunk_ids_by_question={question.id: question.retrieved_chunk_ids},
            reference_by_question={question.id: question.reference_answer},
        )
        for metric in FULL_RAGAS_METRICS:
            if (question.id, metric) in terminal:
                continue
            provider, model = args.judge_provider, args.judge_model
            try:
                try:
                    value = _evaluate_metric(record, metric, primary_llm, embeddings, args)
                except Exception as primary_exc:
                    provider, model = args.fallback_judge_provider, args.fallback_judge_model
                    LOGGER.warning(
                        "Primary judge failed for %s/%s; using %s/%s: %s",
                        question.id, metric, provider, model, primary_exc,
                    )
                    value = _evaluate_metric(record, metric, fallback_llm, embeddings, args)
                cache.append(
                    question.id,
                    metric,
                    status="completed",
                    value=value,
                    judge_provider=provider,
                    judge_model=model,
                )
            except Exception as exc:
                cache.append(
                    question.id,
                    metric,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    judge_provider=provider,
                    judge_model=model,
                )
                LOGGER.error("Both judges failed for %s/%s: %s", question.id, metric, exc)

    metric_rows, progress = metric_rows_from_cache(
        (str(row["id"]) for row in generated), cache
    )
    for row in base["questions"]:
        row["source_tier"] = "controlled"
        row["alignment_status"] = "aligned"
        row["difficulty"] = difficulty[str(row["id"])]
    aggregates = aggregate_scores(base["questions"], metric_rows)
    aggregates["by_difficulty"] = _aggregate_by_difficulty(base["questions"], metric_rows)
    entries = cache.entries()
    fallback_count = sum(
        row.get("status") == "completed"
        and row.get("judge_model") == args.fallback_judge_model
        for row in entries.values()
    )
    expected = len(questions) * len(FULL_RAGAS_METRICS)
    scored = sum(
        status == "completed" for values in progress.values() for status in values.values()
    )
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "experiment": "controlled_generation_with_frozen_gold_contexts",
        "generator": {"provider": args.generator_provider, "model": args.generator_model},
        "judge": {
            "provider": args.judge_provider,
            "model": args.judge_model,
            "fallback_provider": args.fallback_judge_provider,
            "fallback_model": args.fallback_judge_model,
            "fallback_policy": "errors_or_missing_scores_only",
            "fallback_metric_calls": fallback_count,
        },
        "retrieval": {
            "mode": "bypassed_frozen_gold_contexts",
            "index": str(args.production_index),
            "golden": str(args.golden),
        },
        "coverage": {"requested": len(questions), "generated": len(generated), "metric_values": scored, "expected_metric_values": expected},
        "questions": base["questions"],
        "generation_aggregate": base["aggregate"],
        "ragas": {
            "status": (
                "completed"
                if len(generated) == len(questions) and scored == expected
                else "partial"
            ),
            "metrics": list(FULL_RAGAS_METRICS),
            "questions": metric_rows,
            "metric_progress": progress,
        },
        "aggregates": aggregates,
        "errors": checkpoint.payload.get("errors", []),
    }
    paths = write_full_ragas_outputs(run_dir, payload=payload)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0 if len(generated) == len(questions) and scored == expected else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--golden", type=Path, default=Path("evaluation/data/controlled_generation_qa.json"))
    result.add_argument("--production-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    result.add_argument("--output-dir", type=Path, default=Path("evaluation/data/eval_results"))
    result.add_argument("--run-id", default="controlled_generation_ragas")
    result.add_argument("--limit", type=int)
    result.add_argument("--generator-provider", default="groq")
    result.add_argument("--generator-model", default="llama-3.3-70b-versatile")
    result.add_argument("--judge-provider", default="gemini")
    result.add_argument("--judge-model", default="gemini-3.5-flash-lite")
    result.add_argument("--fallback-judge-provider", default="groq")
    result.add_argument("--fallback-judge-model", default="llama-3.1-8b-instant")
    result.add_argument("--generation-requests-per-second", type=float, default=1.0)
    result.add_argument("--judge-requests-per-second", type=float, default=0.2)
    result.add_argument("--max-context-tokens", type=int, default=2500)
    result.add_argument("--judge-max-tokens", type=int, default=2048)
    result.add_argument("--request-timeout", type=float, default=120.0)
    result.add_argument("--ragas-timeout", type=int, default=180)
    result.add_argument("--ragas-max-retries", type=int, default=1)
    result.add_argument("--retries", type=int, default=2)
    result.add_argument("--backoff-seconds", type=float, default=5.0)
    result.add_argument("--log-level", default="INFO")
    return result


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
