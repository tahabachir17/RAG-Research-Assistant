"""Run and incrementally checkpoint the full generation evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from config import Settings
from generation import build_llm_client
from processing.bm25_indexer import BM25Indexer
from retrieval.models import RetrievalResult

from .generation_eval_checkpoint import (
    GenerationEvalCheckpoint,
    latest_compatible_checkpoint,
)
from .generation_evaluator import (
    GenerationEvaluator,
    build_generation_result,
    save_generation_outputs,
)
from .generation_golden import GenerationGoldenQuestion, load_generation_golden
from .llm_judge import LLMJudge
from .ragas_evaluator import (
    DEFAULT_METRIC_NAMES,
    REFERENCE_METRIC_NAMES,
    build_ragas_clients,
    build_ragas_records,
    evaluate_with_ragas,
)
from .rate_limit_client import EvaluationRateLimitClient, _is_transient


LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    base_settings = Settings()
    generation_provider, generation_model = _resolve_generation_target(
        base_settings, args.generator_provider, args.model
    )
    judge_provider, judge_model = _resolve_evaluation_target(
        base_settings, args.judge_provider, args.judge_model
    )
    overrides: dict[str, Any] = {
        "LLM_PROVIDER": generation_provider,
        "LLM_MODEL": generation_model,
        "JUDGE_PROVIDER": judge_provider,
        "JUDGE_MODEL": judge_model,
        "RAGAS_JUDGE_PROVIDER": judge_provider,
        "RAGAS_JUDGE_MODEL": judge_model,
        "LLM_MAX_TOKENS": args.max_tokens,
        "RAGAS_REQUESTS_PER_SECOND": args.requests_per_second,
    }
    if args.request_timeout:
        overrides["LLM_REQUEST_TIMEOUT_SECONDS"] = args.request_timeout
    if args.max_retries is not None:
        overrides["GENERATION_MAX_RETRIES"] = args.max_retries
    if args.judge_max_tokens:
        overrides["JUDGE_MAX_TOKENS"] = args.judge_max_tokens
    settings = base_settings.model_copy(update=overrides)
    provider = settings.LLM_PROVIDER.strip().casefold()

    if provider in {"lmstudio", "lm-studio"}:
        _check_lmstudio(settings, settings.LLM_MODEL)
    if settings.JUDGE_PROVIDER.strip().casefold() == "qwen":
        _check_lmstudio(settings, settings.JUDGE_MODEL)
    try:
        _validate_credentials(
            settings,
            judge_enabled=not args.no_llm_judge,
            ragas_enabled=not args.no_ragas,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (
        (not args.no_llm_judge or not args.no_ragas)
        and provider == settings.JUDGE_PROVIDER.strip().casefold()
        and settings.LLM_MODEL.strip() == settings.JUDGE_MODEL.strip()
    ):
        parser.error("the judge must differ from the generation model")

    matrix = {
        "generation": {"provider": provider, "model": settings.LLM_MODEL},
        "judge": {
            "provider": settings.JUDGE_PROVIDER,
            "model": settings.JUDGE_MODEL,
            "semantic_enabled": not args.no_llm_judge,
            "ragas_enabled": not args.no_ragas,
        },
    }
    print(json.dumps(matrix, indent=2), file=sys.stderr)
    if args.dry_run:
        return 0

    all_questions = load_generation_golden(args.dataset)
    coverage = _dataset_coverage(all_questions)
    print(json.dumps({"dataset_coverage": coverage}, indent=2), file=sys.stderr)
    if args.require_reviewed and coverage["unreviewed"]:
        parser.error(
            f"--require-reviewed was set, but {coverage['unreviewed']} questions are unreviewed"
        )
    questions = all_questions[: args.limit] if args.limit is not None else all_questions
    index = BM25Indexer.load(args.bm25_index)
    if args.chunk_count is not None:
        questions = _with_chunk_count(questions, index, args.chunk_count)
    lookup = FrozenChunkLookup(index=index)

    signature = _run_signature(args, settings, questions)
    checkpoint = _open_checkpoint(
        args,
        signature,
        matrix=matrix,
        coverage=coverage,
    )
    checkpoint_path = checkpoint.path
    stamp = checkpoint_path.stem.removeprefix("generation_eval_")

    judge = _build_judge(settings, args) if not args.no_llm_judge else None
    generator = EvaluationRateLimitClient(
        build_llm_client(settings),
        max_retries=args.retries,
        default_wait_seconds=args.backoff_seconds,
        backoff_seconds=args.backoff_seconds,
        requests_per_second=args.requests_per_second,
    )
    evaluator = GenerationEvaluator(
        llm=generator,
        chunk_lookup=lookup,
        provider=provider,
        model=settings.LLM_MODEL,
        judge=judge,
        max_retries=settings.GENERATION_MAX_RETRIES,
        max_context_tokens=args.max_context_tokens,
    )

    _run_generation_stage(
        questions,
        evaluator=evaluator,
        checkpoint=checkpoint,
        workers=args.workers,
    )
    _refresh_generation_summary(checkpoint, provider, settings.LLM_MODEL, judge)

    if not args.no_ragas:
        ragas_llm, ragas_embeddings = build_ragas_clients(settings)
        _run_ragas_stage(
            questions,
            checkpoint=checkpoint,
            lookup=lookup,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            args=args,
        )
    else:
        checkpoint.payload["ragas"] = {
            "status": "disabled",
            "metrics": list(DEFAULT_METRIC_NAMES),
            "aggregate": {},
            "questions": [],
        }
        checkpoint.save()

    _finalize_ragas(checkpoint, len(questions), enabled=not args.no_ragas)
    checkpoint.payload["coverage"]["evaluated"] = len(
        checkpoint.completed_question_ids().intersection(q.id for q in questions)
    )
    checkpoint.payload["coverage"]["requested"] = len(questions)
    checkpoint.payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint.save()
    paths = save_generation_outputs(checkpoint.payload, args.output_dir, stamp=stamp)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return _evaluation_exit_code(
        checkpoint.payload,
        judge_required=not args.no_llm_judge,
        ragas_required=not args.no_ragas,
    )


class FrozenChunkLookup:
    """Resolve frozen evaluation IDs from the trusted local BM25 artifact."""

    def __init__(
        self, index_path: str | Path | None = None, *, index: BM25Indexer | None = None
    ) -> None:
        loaded = index or BM25Indexer.load(index_path)
        self._chunks = {
            str(chunk.get("chunk_id", "")): dict(chunk) for chunk in loaded.chunks
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


def _run_generation_stage(
    questions: list[GenerationGoldenQuestion],
    *,
    evaluator: GenerationEvaluator,
    checkpoint: GenerationEvalCheckpoint,
    workers: int,
) -> None:
    completed = {
        str(row["id"])
        for row in checkpoint.payload.get("questions", [])
        if evaluator.judge is None or row.get("judge_status") == "judged"
    }
    pending = [question for question in questions if question.id not in completed]
    LOGGER.info(
        "Generation: %d complete, %d pending", len(questions) - len(pending), len(pending)
    )
    failures = checkpoint.payload.setdefault("errors", [])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluator.evaluate_one, question): question for question in pending}
        for future in as_completed(futures):
            question = futures[future]
            try:
                checkpoint.record_question(future.result())
                LOGGER.info("Checkpointed generation for %s", question.id)
            except Exception as exc:
                LOGGER.error("Generation failed for %s: %s", question.id, exc)
                failures.append(
                    {
                        "question_id": question.id,
                        "stage": "generation",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                checkpoint.save()


def _run_ragas_stage(
    questions: list[GenerationGoldenQuestion],
    *,
    checkpoint: GenerationEvalCheckpoint,
    lookup: FrozenChunkLookup,
    llm: Any,
    embeddings: Any,
    args: argparse.Namespace,
) -> None:
    generation_by_id = {
        str(row["id"]): row for row in checkpoint.payload.get("questions", [])
    }
    for question in questions:
        generation_row = generation_by_id.get(question.id)
        if generation_row is None:
            continue
        records = build_ragas_records(
            {"questions": [generation_row]},
            context_lookup=lookup,
            chunk_ids_by_question={question.id: question.retrieved_chunk_ids},
            reference_by_question={question.id: question.reference_answer},
        )
        for metric in DEFAULT_METRIC_NAMES:
            if checkpoint.metric_completed(question.id, metric):
                continue
            if metric in REFERENCE_METRIC_NAMES and not question.reference_answer:
                checkpoint.record_metric(
                    question.id,
                    metric,
                    status="unavailable",
                    reason="golden question has no reviewed reference_answer",
                )
                continue
            try:
                result = _retry_call(
                    lambda metric=metric: evaluate_with_ragas(
                        records,
                        llm=llm,
                        embeddings=embeddings,
                        timeout=args.ragas_timeout,
                        max_workers=1,
                        max_retries=args.ragas_max_retries,
                        metric_names=[metric],
                    ),
                    retries=args.retries,
                    backoff_seconds=args.backoff_seconds,
                    label=f"RAGAS {question.id}/{metric}",
                )
                value = result.get("questions", [{}])[0].get(metric)
                status = "completed" if value is not None else "failed"
                checkpoint.record_metric(
                    question.id,
                    metric,
                    status=status,
                    value=value,
                    reason=result.get("reason") if status == "failed" else None,
                )
            except Exception as exc:
                checkpoint.record_metric(
                    question.id,
                    metric,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                LOGGER.error("RAGAS failed for %s/%s: %s", question.id, metric, exc)


def _retry_call(
    operation: Callable[[], Any],
    *,
    retries: int,
    backoff_seconds: float,
    label: str,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == retries or not _is_transient(exc):
                raise
            wait = backoff_seconds * (2**attempt)
            LOGGER.warning(
                "Transient failure for %s; retry %d/%d in %.2fs: %s",
                label,
                attempt + 1,
                retries,
                wait,
                exc,
            )
            sleeper(wait)
    raise AssertionError("bounded retry loop did not return")


def _refresh_generation_summary(
    checkpoint: GenerationEvalCheckpoint,
    provider: str,
    model: str,
    judge: LLMJudge | None,
) -> None:
    order = {
        identifier: index
        for index, identifier in enumerate(checkpoint.payload.get("question_order", []))
    }
    rows = sorted(
        checkpoint.payload.get("questions", []),
        key=lambda row: order.get(str(row["id"]), len(order)),
    )
    result = build_generation_result(rows, provider=provider, model=model, judge=judge)
    checkpoint.payload.update(result)
    checkpoint.save()


def _finalize_ragas(
    checkpoint: GenerationEvalCheckpoint, question_count: int, *, enabled: bool
) -> None:
    if not enabled:
        return
    ragas = checkpoint.payload.setdefault("ragas", {})
    rows = ragas.setdefault("questions", [])
    progress = checkpoint.payload.get("metric_progress", {})
    aggregate: dict[str, float | None] = {}
    for metric in DEFAULT_METRIC_NAMES:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        aggregate[metric] = sum(values) / len(values) if values else None
    terminal = sum(
        progress.get(str(row.get("id")), {}).get(metric)
        in {"completed", "unavailable"}
        for row in rows
        for metric in DEFAULT_METRIC_NAMES
    )
    expected = question_count * len(DEFAULT_METRIC_NAMES)
    unavailable = sum(
        progress.get(str(row.get("id")), {}).get(metric) == "unavailable"
        for row in rows
        for metric in DEFAULT_METRIC_NAMES
    )
    ragas.update(
        {
            "status": "completed" if terminal == expected else "partial",
            "reason": (
                f"{unavailable} reference-dependent values unavailable from the dataset"
                if terminal == expected and unavailable
                else None if terminal == expected else f"{expected - terminal} metric values incomplete"
            ),
            "metrics": list(DEFAULT_METRIC_NAMES),
            "aggregate": aggregate,
            "completed_values": terminal - unavailable,
            "unavailable_values": unavailable,
            "expected_values": expected,
        }
    )
    checkpoint.save()


def _open_checkpoint(
    args: argparse.Namespace,
    signature: dict[str, Any],
    *,
    matrix: dict[str, Any],
    coverage: dict[str, int],
) -> GenerationEvalCheckpoint:
    resume_path = args.resume_from
    if args.resume and resume_path is None:
        resume_path = latest_compatible_checkpoint(args.output_dir, signature)
        if resume_path is None:
            raise ValueError("--resume found no compatible generation evaluation checkpoint")
    if resume_path is not None:
        checkpoint = GenerationEvalCheckpoint.load(resume_path)
        if checkpoint.payload.get("run_signature") != signature:
            raise ValueError("resume checkpoint does not match dataset/provider/run settings")
        LOGGER.info("Resuming %s", checkpoint.path)
        return checkpoint
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_dir / f"generation_eval_{stamp}.json"
    return GenerationEvalCheckpoint.create(
        path,
        {
            "checkpoint_schema_version": 1,
            "run_signature": signature,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "provider": matrix["generation"]["provider"],
            "model": matrix["generation"]["model"],
            "judge": matrix["judge"],
            "coverage": coverage,
            "question_order": signature["question_ids"],
            "aggregate": {},
            "questions": [],
            "ragas": {"questions": []},
            "metric_progress": {},
            "errors": [],
            "provenance": signature["provenance"],
        },
    )


def _build_judge(settings: Settings, args: argparse.Namespace) -> LLMJudge:
    client_provider = (
        "lmstudio" if settings.JUDGE_PROVIDER.strip().casefold() == "qwen" else settings.JUDGE_PROVIDER
    )
    judge_settings = settings.model_copy(
        update={
            "LLM_PROVIDER": client_provider,
            "LLM_MODEL": settings.JUDGE_MODEL,
            "LLM_MAX_TOKENS": settings.JUDGE_MAX_TOKENS,
            "LLM_TEMPERATURE": settings.JUDGE_TEMPERATURE,
        }
    )
    client = EvaluationRateLimitClient(
        build_llm_client(judge_settings),
        max_retries=args.retries,
        default_wait_seconds=args.backoff_seconds,
        backoff_seconds=args.backoff_seconds,
        requests_per_second=args.requests_per_second,
    )
    return LLMJudge(
        client,
        judge_model=settings.JUDGE_MODEL,
        model_under_test=settings.LLM_MODEL,
        judge_provider=settings.JUDGE_PROVIDER,
        cache_path=args.judge_cache,
        temperature=settings.JUDGE_TEMPERATURE,
    )


def _with_chunk_count(
    questions: list[GenerationGoldenQuestion], index: BM25Indexer, count: int
) -> list[GenerationGoldenQuestion]:
    configured: list[GenerationGoldenQuestion] = []
    for question in questions:
        selected = list(dict.fromkeys(question.retrieved_chunk_ids))
        if len(selected) < count:
            query_tokens = set(index.tokenize(question.question))
            candidates = sorted(
                (
                    chunk
                    for chunk in index.chunks
                    if _same_paper(str(chunk.get("paper_id", "")), question.paper_id)
                ),
                key=lambda chunk: _candidate_chunk_key(
                    chunk, query_tokens, index.preprocessing_config
                ),
            )
            for chunk in candidates:
                chunk_id = str(chunk.get("chunk_id", ""))
                if chunk_id and chunk_id not in selected:
                    selected.append(chunk_id)
                if len(selected) >= count:
                    break
        if len(selected) < count:
            raise ValueError(f"{question.id}: only {len(selected)} same-paper chunks available")
        configured.append(replace(question, retrieved_chunk_ids=selected[:count]))
    return configured


def _same_paper(candidate: str, expected: str) -> bool:
    def normalize(value: str) -> str:
        return re.sub(r"v\d+$", "", value.strip().casefold())

    return bool(expected) and normalize(candidate) == normalize(expected)


def _candidate_chunk_key(
    chunk: dict[str, Any],
    query_tokens: set[str],
    preprocessing_config: dict[str, Any],
) -> tuple[int, int, str]:
    priority = {
        "abstract": 0,
        "conclusion": 1,
        "results": 2,
        "limitations": 3,
        "methodology": 4,
    }
    section = str(chunk.get("section", "")).casefold()
    overlap = len(
        query_tokens
        & set(BM25Indexer.tokenize(str(chunk.get("text", "")), preprocessing_config))
    )
    return priority.get(section, 9), -overlap, str(chunk.get("chunk_id", ""))


def _dataset_coverage(questions: list[GenerationGoldenQuestion]) -> dict[str, int]:
    reviewed = sum(question.reviewed for question in questions)
    references = sum(bool(question.reference_answer) for question in questions)
    return {
        "total": len(questions),
        "reviewed": reviewed,
        "unreviewed": len(questions) - reviewed,
        "with_reference_answer": references,
        "without_reference_answer": len(questions) - references,
        "evaluated": 0,
    }


def _run_signature(
    args: argparse.Namespace,
    settings: Settings,
    questions: list[GenerationGoldenQuestion],
) -> dict[str, Any]:
    return {
        "dataset_sha256": _sha256(args.dataset),
        "generator_provider": settings.LLM_PROVIDER,
        "generator_model": settings.LLM_MODEL,
        "judge_provider": settings.JUDGE_PROVIDER,
        "judge_model": settings.JUDGE_MODEL,
        "chunk_count": args.chunk_count,
        "max_context_tokens": args.max_context_tokens,
        "question_ids": [question.id for question in questions],
        "semantic_judge_enabled": not args.no_llm_judge,
        "ragas_enabled": not args.no_ragas,
        "provenance": {
            "evaluation_schema_version": 4,
            "dataset_path": str(args.dataset),
            "dataset_sha256": _sha256(args.dataset),
            "bm25_index_path": str(args.bm25_index),
            "prompt_path": "config/prompts/qa_prompt.yaml",
            "prompt_sha256": _sha256(Path("config/prompts/qa_prompt.yaml")),
            "structured_contract_path": "generation/structured_answer.py",
            "structured_contract_sha256": _sha256(Path("generation/structured_answer.py")),
            "max_context_tokens": args.max_context_tokens,
            "max_output_tokens": settings.LLM_MAX_TOKENS,
        },
    }


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
            "LM Studio is not reachable. Start its local server at "
            f"{settings.LMSTUDIO_BASE_URL}."
        ) from exc
    available = {
        str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)
    }
    if available and model not in available:
        raise RuntimeError(
            f"LM Studio does not report model {model!r}. Available: "
            + ", ".join(sorted(available))
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        "--golden",
        dest="dataset",
        type=Path,
        default=Path("evaluation/data/golden_generation_qa.json"),
    )
    parser.add_argument("--bm25-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/data/eval_results"))
    parser.add_argument(
        "--generator-provider",
        "--provider",
        dest="generator_provider",
        choices=("groq", "gemini"),
        default="groq",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--judge-provider", choices=("groq", "gemini", "qwen"), default="gemini"
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--chunk-count", "--top-k", dest="chunk_count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 33))
    parser.add_argument("--requests-per-second", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=3, choices=range(0, 11))
    parser.add_argument("--backoff-seconds", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--max-retries", type=int, choices=(0, 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument(
        "--judge-cache",
        type=Path,
        default=Path("evaluation/data/eval_results/llm_judge_cache.json"),
    )
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-max-retries", type=int, default=1, choices=range(0, 6))
    # Backward-compatible aliases from the one-off runner.
    parser.add_argument("--ragas-workers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--ragas-requests-per-second", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--rate-limit-retries", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rate-limit-default-wait", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
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
        configured = settings.LLM_PROVIDER.strip().casefold()
        model = settings.LLM_MODEL if provider == configured else (
            settings.GROQ_MODEL if provider == "groq" else settings.GEMINI_MODEL
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
        configured = settings.JUDGE_PROVIDER.strip().casefold()
        model = settings.JUDGE_MODEL if provider == configured else {
            "groq": settings.GROQ_MODEL,
            "gemini": settings.GEMINI_MODEL,
            "qwen": settings.LMSTUDIO_MODEL,
        }[provider]
    else:
        model = settings.JUDGE_MODEL.strip()
    if not model:
        raise ValueError("judge model must not be empty")
    return provider, model


def _validate_credentials(
    settings: Settings, *, judge_enabled: bool, ragas_enabled: bool
) -> None:
    required = {settings.LLM_PROVIDER.strip().casefold()}
    if judge_enabled or ragas_enabled:
        required.add(settings.JUDGE_PROVIDER.strip().casefold())
    if "groq" in required and not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required by the resolved provider matrix")
    if "gemini" in required and not (settings.GEMINI_API_KEY or settings.OPENAI_API_KEY):
        raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY is required")


def _evaluation_exit_code(
    result: dict[str, Any], *, judge_required: bool, ragas_required: bool
) -> int:
    requested = int(result.get("coverage", {}).get("requested", result.get("aggregate", {}).get("questions", 0)))
    if result.get("aggregate", {}).get("questions", 0) != requested:
        print("Evaluation artifacts were saved, but generation is incomplete.", file=sys.stderr)
        return 2
    if judge_required and result.get("aggregate", {}).get("judge_coverage") != 1.0:
        print("Evaluation artifacts were saved, but the semantic judge is incomplete.", file=sys.stderr)
        return 2
    if ragas_required and result.get("ragas", {}).get("status") != "completed":
        reason = result.get("ragas", {}).get("reason", "unknown RAGAS failure")
        print(f"Evaluation artifacts were saved, but RAGAS is incomplete: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
