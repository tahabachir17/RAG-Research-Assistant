"""Reference-free RAGAS evaluation for generated answers and frozen contexts."""

from __future__ import annotations

import math
import statistics
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

from datasets import Dataset


DEFAULT_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_utilization")


def build_ragas_records(
    generation_result: dict[str, Any],
    *,
    context_lookup: Callable[[list[str]], Sequence[Any]],
    chunk_ids_by_question: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Convert generation output to the question/answer/contexts RAGAS schema."""

    records: list[dict[str, Any]] = []
    for row in generation_result.get("questions", []):
        question_id = str(row["id"])
        chunk_ids = row.get("context_chunk_ids") or chunk_ids_by_question.get(question_id)
        if chunk_ids is None:
            raise KeyError(f"No frozen context mapping for question {question_id!r}")
        contexts = [str(chunk.text) for chunk in context_lookup(chunk_ids)]
        if not contexts:
            raise ValueError(f"{question_id}: RAGAS requires at least one context")
        records.append(
            {
                "id": question_id,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "contexts": contexts,
            }
        )
    return records


def evaluate_with_ragas(
    records: list[dict[str, Any]],
    *,
    llm: Any,
    embeddings: Any,
    timeout: int = 180,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Run reference-free RAGAS metrics and return JSON-serializable results."""

    if not records:
        return {
            "status": "skipped",
            "reason": "no generation records",
            "metrics": list(DEFAULT_METRIC_NAMES),
            "aggregate": {},
            "questions": [],
        }

    from ragas import evaluate
    from ragas.run_config import RunConfig

    dataset = Dataset.from_list(
        [{key: row[key] for key in ("question", "answer", "contexts")} for row in records]
    )
    result = evaluate(
        dataset,
        metrics=_build_metrics(),
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=timeout,
            max_retries=5,
            max_wait=60,
            max_workers=max_workers,
        ),
        raise_exceptions=False,
    )
    scored = result.to_pandas().to_dict(orient="records")
    question_rows: list[dict[str, Any]] = []
    for source, scores in zip(records, scored):
        question_rows.append(
            {
                "id": source["id"],
                **{name: _finite_or_none(scores.get(name)) for name in DEFAULT_METRIC_NAMES},
            }
        )
    aggregate = {
        name: _nullable_mean(row[name] for row in question_rows)
        for name in DEFAULT_METRIC_NAMES
    }
    available = sum(
        row[name] is not None
        for row in question_rows
        for name in DEFAULT_METRIC_NAMES
    )
    expected = len(question_rows) * len(DEFAULT_METRIC_NAMES)
    status = "completed" if available == expected else "partial" if available else "failed"
    return {
        "status": status,
        "reason": None if status == "completed" else f"{expected - available} of {expected} metric values were unavailable",
        "metrics": list(DEFAULT_METRIC_NAMES),
        "aggregate": aggregate,
        "questions": question_rows,
    }


def build_ragas_clients(settings: Any) -> tuple[Any, Any]:
    """Build a judge LLM and local embeddings without exposing credentials."""

    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    provider = settings.JUDGE_PROVIDER.strip().casefold()
    options: dict[str, Any] = {
        "model": settings.JUDGE_MODEL,
        "temperature": 0.0,
        "max_tokens": settings.JUDGE_MAX_TOKENS,
    }
    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for the RAGAS judge")
        from langchain_core._api import LangChainBetaWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainBetaWarning)
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=settings.RAGAS_REQUESTS_PER_SECOND,
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            )
        options.update(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            max_retries=5,
            rate_limiter=rate_limiter,
        )
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for the RAGAS judge")
        options["api_key"] = settings.OPENAI_API_KEY
        if settings.OPENAI_BASE_URL:
            options["base_url"] = settings.OPENAI_BASE_URL
    elif provider in {"lmstudio", "lm-studio"}:
        options.update(
            api_key=settings.LMSTUDIO_API_KEY,
            base_url=settings.LMSTUDIO_BASE_URL,
        )
    else:
        raise ValueError(
            "RAGAS currently supports groq, openai, or lmstudio judge providers"
        )

    llm = ChatOpenAI(**options)
    model_name = _local_embedding_snapshot(settings.EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder="data/model_cache",
        model_kwargs={"local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return llm, embeddings


def _build_metrics() -> list[Any]:
    """Use provider-compatible, independently instantiated RAGAS metrics."""

    from ragas.metrics import AnswerRelevancy, ContextUtilization, Faithfulness

    return [
        Faithfulness(),
        # RAGAS maps strictness to the OpenAI-compatible `n` parameter, while
        # Groq accepts only n=1.
        AnswerRelevancy(strictness=1),
        ContextUtilization(),
    ]


def _local_embedding_snapshot(model_name: str) -> str:
    cache_name = model_name.replace("/", "--")
    snapshots = Path("data/model_cache/hub") / f"models--sentence-transformers--{cache_name}" / "snapshots"
    candidates = sorted(path for path in snapshots.glob("*") if path.is_dir())
    return str(candidates[-1].resolve()) if candidates else model_name


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nullable_mean(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    return statistics.fmean(items) if items else None
