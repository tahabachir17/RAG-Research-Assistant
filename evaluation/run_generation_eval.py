"""Run the frozen generation evaluation against a configured live model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import Settings
from generation import build_llm_client
from processing.bm25_indexer import BM25Indexer
from retrieval.models import RetrievalResult

from .generation_evaluator import GenerationEvaluator, save_generation_outputs
from .generation_golden import load_generation_golden
from .llm_judge import LLMJudge
from .rate_limit_client import EvaluationRateLimitClient
from .ragas_evaluator import (
    build_ragas_clients,
    build_ragas_records,
    evaluate_with_ragas,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    base_settings = Settings()
    generation_provider, generation_model = _resolve_generation_target(
        base_settings, args.provider, args.model
    )
    judge_provider, judge_model = _resolve_evaluation_target(
        base_settings, args.judge_provider, args.judge_model
    )
    overrides: dict[str, Any] = {
        "LLM_PROVIDER": generation_provider,
        "LLM_MODEL": generation_model,
        "JUDGE_PROVIDER": judge_provider,
        "JUDGE_MODEL": judge_model,
    }
    if args.max_tokens:
        overrides["LLM_MAX_TOKENS"] = args.max_tokens
    if args.request_timeout:
        overrides["LLM_REQUEST_TIMEOUT_SECONDS"] = args.request_timeout
    if args.max_retries is not None:
        overrides["GENERATION_MAX_RETRIES"] = args.max_retries
    if args.judge_max_tokens:
        overrides["JUDGE_MAX_TOKENS"] = args.judge_max_tokens
    if args.ragas_requests_per_second:
        overrides["RAGAS_REQUESTS_PER_SECOND"] = args.ragas_requests_per_second
    settings = base_settings.model_copy(update=overrides)
    provider = settings.LLM_PROVIDER.strip().casefold()
    if provider in {"lmstudio", "lm-studio"}:
        try:
            _check_lmstudio(settings, settings.LLM_MODEL)
        except RuntimeError as exc:
            parser.error(str(exc))
    if settings.JUDGE_PROVIDER.strip().casefold() == "qwen":
        try:
            _check_lmstudio(settings, settings.JUDGE_MODEL)
        except RuntimeError as exc:
            parser.error(str(exc))
    try:
        _validate_credentials(
            settings,
            judge_enabled=not args.no_llm_judge,
            ragas_enabled=not args.no_ragas,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (
        not args.no_llm_judge
        and provider == settings.JUDGE_PROVIDER.strip().casefold()
        and settings.LLM_MODEL.strip() == settings.JUDGE_MODEL.strip()
    ):
        parser.error("the semantic judge must differ from the generation model")

    print(
        json.dumps(
            {
                "generation": {"provider": provider, "model": settings.LLM_MODEL},
                "llm_judge": {
                    "provider": settings.JUDGE_PROVIDER,
                    "model": settings.JUDGE_MODEL,
                    "enabled": not args.no_llm_judge,
                },
                "ragas": {
                    "provider": settings.JUDGE_PROVIDER,
                    "model": settings.JUDGE_MODEL,
                    "enabled": not args.no_ragas,
                },
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    if args.dry_run:
        return 0

    questions = load_generation_golden(
        args.golden, require_reviewed=args.require_reviewed
    )
    if args.limit is not None:
        questions = questions[: args.limit]
    lookup = FrozenChunkLookup(args.bm25_index)
    judge = None
    if not args.no_llm_judge:
        judge_client_provider = (
            "lmstudio"
            if settings.JUDGE_PROVIDER.strip().casefold() == "qwen"
            else settings.JUDGE_PROVIDER
        )
        judge_settings = settings.model_copy(
            update={
                "LLM_PROVIDER": judge_client_provider,
                "LLM_MODEL": settings.JUDGE_MODEL,
                "LLM_MAX_TOKENS": settings.JUDGE_MAX_TOKENS,
                "LLM_TEMPERATURE": settings.JUDGE_TEMPERATURE,
            }
        )
        judge = LLMJudge(
            EvaluationRateLimitClient(
                build_llm_client(judge_settings),
                max_retries=args.rate_limit_retries,
                default_wait_seconds=args.rate_limit_default_wait,
            ),
            judge_model=settings.JUDGE_MODEL,
            model_under_test=settings.LLM_MODEL,
            judge_provider=settings.JUDGE_PROVIDER,
            cache_path=args.judge_cache,
            temperature=settings.JUDGE_TEMPERATURE,
        )
    evaluator = GenerationEvaluator(
        llm=EvaluationRateLimitClient(
            build_llm_client(settings),
            max_retries=args.rate_limit_retries,
            default_wait_seconds=args.rate_limit_default_wait,
        ),
        chunk_lookup=lookup,
        provider=provider,
        model=settings.LLM_MODEL,
        judge=judge,
        max_retries=settings.GENERATION_MAX_RETRIES,
        max_context_tokens=args.max_context_tokens,
    )
    result = evaluator.evaluate(questions)
    result["provenance"] = {
        "evaluation_schema_version": 2,
        "golden_path": str(args.golden),
        "golden_sha256": _sha256(args.golden),
        "bm25_index_path": str(args.bm25_index),
        "prompt_path": "config/prompts/qa_prompt.yaml",
        "prompt_sha256": _sha256(Path("config/prompts/qa_prompt.yaml")),
        "max_context_tokens": args.max_context_tokens,
        "max_output_tokens": settings.LLM_MAX_TOKENS,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Checkpoint deterministic generation and semantic judge results before the
    # slower RAGAS stage. The same paths are overwritten with final metrics.
    paths = save_generation_outputs(result, args.output_dir, stamp=stamp)
    if not args.no_ragas:
        try:
            records = build_ragas_records(
                result,
                context_lookup=lookup,
                chunk_ids_by_question={question.id: question.retrieved_chunk_ids for question in questions},
            )
            ragas_llm, ragas_embeddings = build_ragas_clients(settings)
            result["ragas"] = evaluate_with_ragas(
                records,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                timeout=args.ragas_timeout,
                max_workers=args.ragas_workers,
                max_retries=args.ragas_max_retries,
            )
        except Exception as exc:
            result["ragas"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "aggregate": {},
                "questions": [],
            }
        result["ragas"]["judge"] = {
            "provider": settings.JUDGE_PROVIDER,
            "model": settings.JUDGE_MODEL,
        }
    paths = save_generation_outputs(result, args.output_dir, stamp=stamp)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return _evaluation_exit_code(
        result,
        judge_required=not args.no_llm_judge,
        ragas_required=not args.no_ragas,
    )


class FrozenChunkLookup:
    """Resolve frozen evaluation IDs from the trusted local BM25 artifact."""

    def __init__(self, index_path: str | Path) -> None:
        index = BM25Indexer.load(index_path)
        self._chunks = {
            str(chunk.get("chunk_id", "")): dict(chunk) for chunk in index.chunks
        }

    def __call__(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in self._chunks]
        if missing:
            raise KeyError(f"Frozen chunks are missing from the BM25 artifact: {missing}")
        return [
            RetrievalResult.from_payload(
                self._chunks[chunk_id], score=0.0, source="frozen"
            )
            for chunk_id in chunk_ids
        ]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_lmstudio(settings: Settings, model: str) -> None:
    url = f"{settings.LMSTUDIO_BASE_URL.rstrip('/')}/models"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {settings.LMSTUDIO_API_KEY}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "LM Studio is not reachable. In LM Studio, open Developer, start the "
            f"local server, and confirm it listens at {settings.LMSTUDIO_BASE_URL}."
        ) from exc
    available = {
        str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)
    }
    if available and model not in available:
        raise RuntimeError(
            f"LM Studio does not report model {model!r}. Available models: "
            + ", ".join(sorted(available))
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden", type=Path, default=Path("evaluation/data/golden_generation_qa.json")
    )
    parser.add_argument(
        "--bm25-index", type=Path, default=Path("data/processed/bm25_index.pkl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation/data/eval_results")
    )
    parser.add_argument(
        "--provider",
        choices=("groq", "gemini"),
        help="Answer-generation provider (Groq or Gemini).",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Maximum output tokens per generation or repair call (default: 1024)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        help="Per-request timeout in seconds; use a larger value for a local Qwen judge.",
    )
    parser.add_argument("--max-retries", type=int, choices=(0, 1))
    parser.add_argument("--limit", type=int, choices=range(1, 21))
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument(
        "--judge-provider",
        choices=("groq", "gemini", "qwen"),
        help="Provider used by both the semantic LLM judge and RAGAS; qwen uses LM Studio.",
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument(
        "--judge-cache",
        type=Path,
        default=Path("evaluation/data/eval_results/llm_judge_cache.json"),
    )
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip reference-free RAGAS metrics.",
    )
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-workers", type=int, default=1)
    parser.add_argument(
        "--ragas-max-retries",
        type=int,
        default=1,
        choices=range(0, 4),
        help="Retries for each RAGAS model operation (default: 1)",
    )
    parser.add_argument("--ragas-requests-per-second", type=float)
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=2,
        choices=range(0, 6),
        help="Retries per generation/judge call after a provider 429 (default: 2)",
    )
    parser.add_argument(
        "--rate-limit-default-wait",
        type=float,
        default=10.0,
        help="Fallback wait when a 429 response has no retry-after value",
    )
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved provider/model matrix without making model calls.",
    )
    return parser


def _resolve_generation_target(
    settings: Settings,
    requested_provider: str | None,
    requested_model: str | None,
) -> tuple[str, str]:
    provider = (requested_provider or settings.LLM_PROVIDER).strip().casefold()
    if provider not in {"groq", "gemini"}:
        raise ValueError("generation provider must be groq or gemini")
    if requested_model:
        model = requested_model.strip()
    elif requested_provider:
        configured_provider = settings.LLM_PROVIDER.strip().casefold()
        model = (
            settings.LLM_MODEL
            if provider == configured_provider
            else settings.GROQ_MODEL if provider == "groq" else settings.GEMINI_MODEL
        )
    else:
        model = settings.LLM_MODEL.strip()
    if not model:
        raise ValueError("generation model must not be empty")
    return provider, model


def _resolve_evaluation_target(
    settings: Settings,
    requested_provider: str | None,
    requested_model: str | None,
) -> tuple[str, str]:
    provider = (requested_provider or settings.JUDGE_PROVIDER).strip().casefold()
    if provider not in {"groq", "gemini", "qwen"}:
        raise ValueError("judge provider must be groq, gemini, or qwen")
    if requested_model:
        model = requested_model.strip()
    elif requested_provider:
        configured_provider = settings.JUDGE_PROVIDER.strip().casefold()
        model = (
            settings.JUDGE_MODEL
            if provider == configured_provider
            else {
                "groq": settings.GROQ_MODEL,
                "gemini": settings.GEMINI_MODEL,
                "qwen": settings.LMSTUDIO_MODEL,
            }[provider]
        )
    else:
        model = settings.JUDGE_MODEL.strip()
    if not model:
        raise ValueError("judge model must not be empty")
    return provider, model


def _validate_credentials(
    settings: Settings,
    *,
    judge_enabled: bool,
    ragas_enabled: bool,
) -> None:
    required = {settings.LLM_PROVIDER.strip().casefold()}
    if judge_enabled or ragas_enabled:
        required.add(settings.JUDGE_PROVIDER.strip().casefold())
    if "groq" in required and not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required by the resolved provider matrix")
    if "gemini" in required and not (
        settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
    ):
        raise ValueError(
            "GEMINI_API_KEY or the existing Gemini-compatible OPENAI_API_KEY is required"
        )


def _evaluation_exit_code(
    result: dict[str, Any],
    *,
    judge_required: bool,
    ragas_required: bool,
) -> int:
    """Fail the command when a requested evaluation layer did not complete."""

    if judge_required and result.get("aggregate", {}).get("judge_coverage") != 1.0:
        print(
            "Evaluation artifacts were saved, but the semantic judge did not cover every question.",
            file=sys.stderr,
        )
        return 2
    if ragas_required and result.get("ragas", {}).get("status") != "completed":
        reason = result.get("ragas", {}).get("reason", "unknown RAGAS failure")
        print(
            f"Evaluation artifacts were saved, but RAGAS did not complete: {reason}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
