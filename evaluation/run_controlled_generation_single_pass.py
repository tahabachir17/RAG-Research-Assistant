"""Run one controlled, generation-only pass with frozen evidence.

This command deliberately omits RAGAS and LLM-judge calls.  It exists for the
manual concept-recall calibration pass that precedes repeated live runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import Settings
from generation import build_llm_client
from processing.bm25_indexer import BM25Indexer

from .full_ragas_evaluation import TieredChunkLookup
from .generation_eval_checkpoint import GenerationEvalCheckpoint
from .generation_evaluator import GenerationEvaluator, build_generation_result
from .rate_limit_client import EvaluationRateLimitClient
from .run_controlled_generation_ragas import validate_controlled_golden


LOGGER = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> int:
    index = BM25Indexer.load(args.production_index)
    questions = validate_controlled_golden(args.golden, index)
    if args.question_id:
        requested = set(args.question_id)
        available = {question.id for question in questions}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown controlled question ids: {missing}")
        questions = [question for question in questions if question.id in requested]
    lookup = TieredChunkLookup(index)
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        LLM_PROVIDER=args.generator_provider,
        LLM_MODEL=args.generator_model,
        ENABLE_FAITHFULNESS_VERIFIER=False,
        LLM_REQUEST_TIMEOUT_SECONDS=args.request_timeout,
    )
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required for controlled generation")

    signature = {
        "mode": "single_generation_only",
        "golden": str(args.golden.resolve()),
        "production_index": str(args.production_index.resolve()),
        "question_ids": [question.id for question in questions],
        "generator": [args.generator_provider, args.generator_model],
        "evidence_packing": "adjacent_for_mechanism_else_gold",
    }
    checkpoint_path = run_dir / "generation_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = GenerationEvalCheckpoint.load(checkpoint_path)
        if checkpoint.payload.get("run_signature") != signature:
            raise ValueError("run-id already exists with a different benchmark signature")
    else:
        checkpoint = GenerationEvalCheckpoint.create(
            checkpoint_path,
            {
                "schema_version": 1,
                "run_signature": signature,
                "questions": [],
                "errors": [],
            },
        )

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
        provider=args.generator_provider,
        model=args.generator_model,
        judge=None,
        max_retries=settings.GENERATION_MAX_RETRIES,
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
                {
                    "question_id": question.id,
                    "stage": "generation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            checkpoint.save()
            LOGGER.error("Generation failed for %s: %s", question.id, exc)

    order = {question.id: position for position, question in enumerate(questions)}
    generated = sorted(
        checkpoint.payload.get("questions", []),
        key=lambda row: order.get(str(row["id"]), len(order)),
    )
    result = build_generation_result(
        generated,
        provider=args.generator_provider,
        model=args.generator_model,
        judge=None,
    )
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "experiment": "controlled_generation_single_pass",
        "generator": result["provider"],
        "model": result["model"],
        "retrieval": {
            "mode": "bypassed_frozen_gold_contexts",
            "evidence_packing": "adjacent_for_mechanism_else_gold",
            "index": str(args.production_index),
            "golden": str(args.golden),
        },
        "coverage": {
            "requested": len(questions),
            "generated": len(generated),
        },
        "questions": result["questions"],
        "generation_aggregate": result["aggregate"],
        "errors": checkpoint.payload.get("errors", []),
    }
    report_path = run_dir / "report.json"
    temporary = report_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(report_path)
    print(json.dumps({"json": str(report_path)}, indent=2))
    return 0 if len(generated) == len(questions) else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--golden",
        type=Path,
        default=Path("evaluation/data/controlled_generation_qa.json"),
    )
    result.add_argument(
        "--production-index",
        type=Path,
        default=Path("data/processed/bm25_index.pkl"),
    )
    result.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/data/eval_results")
    )
    result.add_argument("--run-id", default="controlled_generation_single_pass")
    result.add_argument(
        "--question-id",
        action="append",
        help="Generate only this controlled question ID; may be repeated.",
    )
    result.add_argument("--generator-provider", default="groq")
    result.add_argument("--generator-model", default="llama-3.3-70b-versatile")
    result.add_argument("--generation-requests-per-second", type=float, default=1.0)
    result.add_argument("--max-context-tokens", type=int, default=2500)
    result.add_argument("--request-timeout", type=float, default=120.0)
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
