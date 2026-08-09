"""Run the frozen generation evaluation against a configured live model."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from config import Settings
from generation import build_llm_client
from processing.bm25_indexer import BM25Indexer
from retrieval.models import RetrievalResult

from .generation_evaluator import GenerationEvaluator, save_generation_outputs
from .generation_golden import load_generation_golden
from .llm_judge import LLMJudge
from .ragas_evaluator import (
    build_ragas_clients,
    build_ragas_records,
    evaluate_with_ragas,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["LLM_PROVIDER"] = args.provider
    if args.model:
        overrides["LLM_MODEL"] = args.model
    if args.max_tokens:
        overrides["LLM_MAX_TOKENS"] = args.max_tokens
    if args.max_retries is not None:
        overrides["GENERATION_MAX_RETRIES"] = args.max_retries
    if args.judge_provider:
        overrides["JUDGE_PROVIDER"] = args.judge_provider
    if args.judge_model:
        overrides["JUDGE_MODEL"] = args.judge_model
    if args.judge_max_tokens:
        overrides["JUDGE_MAX_TOKENS"] = args.judge_max_tokens
    if args.ragas_requests_per_second:
        overrides["RAGAS_REQUESTS_PER_SECOND"] = args.ragas_requests_per_second
    settings = Settings(**overrides)
    provider = settings.LLM_PROVIDER.strip().casefold()
    if provider in {"lmstudio", "lm-studio"}:
        try:
            _check_lmstudio(settings, settings.LLM_MODEL)
        except RuntimeError as exc:
            parser.error(str(exc))

    questions = load_generation_golden(
        args.golden, require_reviewed=args.require_reviewed
    )
    if args.limit is not None:
        questions = questions[: args.limit]
    lookup = FrozenChunkLookup(args.bm25_index)
    judge = None
    if not args.no_llm_judge:
        judge_settings = Settings(
            LLM_PROVIDER=settings.JUDGE_PROVIDER,
            LLM_MODEL=settings.JUDGE_MODEL,
            LLM_MAX_TOKENS=settings.JUDGE_MAX_TOKENS,
            LLM_TEMPERATURE=settings.JUDGE_TEMPERATURE,
        )
        judge = LLMJudge(
            build_llm_client(judge_settings),
            judge_model=settings.JUDGE_MODEL,
            model_under_test=settings.LLM_MODEL,
            cache_path=args.judge_cache,
            temperature=settings.JUDGE_TEMPERATURE,
        )
    evaluator = GenerationEvaluator(
        llm=build_llm_client(settings),
        chunk_lookup=lookup,
        provider=provider,
        model=settings.LLM_MODEL,
        judge=judge,
        max_retries=settings.GENERATION_MAX_RETRIES,
        max_context_tokens=args.max_context_tokens,
    )
    result = evaluator.evaluate(questions)
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
    paths = save_generation_outputs(result, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


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
        "--provider", choices=("groq", "claude", "openai", "lmstudio", "ollama")
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Maximum output tokens per generation or repair call (default: 1024)",
    )
    parser.add_argument("--max-retries", type=int, choices=(0, 1))
    parser.add_argument("--limit", type=int, choices=range(1, 21))
    parser.add_argument("--max-context-tokens", type=int, default=2500)
    parser.add_argument(
        "--judge-provider",
        choices=("groq", "claude", "openai", "lmstudio", "ollama"),
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
    parser.add_argument("--ragas-requests-per-second", type=float)
    parser.add_argument("--require-reviewed", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
