"""Reference-free RAGAS evaluation for generated answers and frozen contexts."""

from __future__ import annotations

import math
import statistics
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

from datasets import Dataset


DEFAULT_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_utilization",
    "answer_correctness",
)
LEGACY_REFERENCE_FREE_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_utilization",
)
REFERENCE_METRIC_NAMES = frozenset(
    {"context_precision", "context_recall", "answer_correctness"}
)


def build_ragas_records(
    generation_result: dict[str, Any],
    *,
    context_lookup: Callable[[list[str]], Sequence[Any]],
    chunk_ids_by_question: dict[str, list[str]],
    reference_by_question: dict[str, str | None] | None = None,
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
        record = {
                "id": question_id,
                "question": str(row["question"]),
                "answer": _ragas_answer(row),
                "contexts": contexts,
            }
        reference = (reference_by_question or {}).get(question_id)
        if reference:
            record["ground_truth"] = reference
        records.append(record)
    return records


def _ragas_answer(row: dict[str, Any]) -> str:
    """Project structured output into prose that RAGAS can segment reliably."""

    structured = row.get("structured_data")
    if not isinstance(structured, dict):
        return str(row["answer"])
    claims = structured.get("claims")
    if isinstance(claims, list):
        values = [
            str(claim.get("text", "")).strip()
            for claim in claims
            if isinstance(claim, dict) and str(claim.get("text", "")).strip()
        ]
        if values:
            return " ".join(_as_sentence(value) for value in values)
    items = structured.get("items")
    if not isinstance(items, list):
        return str(row["answer"])
    paragraphs: list[str] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        facts: list[str] = []
        for field, cell in item.items():
            if not isinstance(cell, dict):
                continue
            value = str(cell.get("text", "")).strip()
            if not value or _is_absent_value(value):
                continue
            label = str(field).replace("_", " ").strip().capitalize()
            facts.append(f"{label}: {value.rstrip('.')}.")
        if facts:
            paragraphs.append(f"Contribution {index}. " + " ".join(facts))
    return "\n\n".join(paragraphs) or str(row["answer"])


def _as_sentence(value: str) -> str:
    return value if value.endswith((".", "!", "?")) else f"{value}."


def _is_absent_value(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    return normalized.startswith("not reported") or normalized.startswith("not provided")


def evaluate_with_ragas(
    records: list[dict[str, Any]],
    *,
    llm: Any,
    embeddings: Any,
    timeout: int = 180,
    max_workers: int = 1,
    max_retries: int = 1,
    metric_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run reference-free RAGAS metrics and return JSON-serializable results."""

    selected_metrics = tuple(metric_names or LEGACY_REFERENCE_FREE_METRIC_NAMES)
    if not records:
        return {
            "status": "skipped",
            "reason": "no generation records",
            "metrics": list(selected_metrics),
            "aggregate": {},
            "questions": [],
        }

    from ragas import evaluate
    from ragas.run_config import RunConfig

    unknown = set(selected_metrics) - set(DEFAULT_METRIC_NAMES)
    if unknown:
        raise ValueError(f"unknown RAGAS metrics: {sorted(unknown)}")
    if REFERENCE_METRIC_NAMES.intersection(selected_metrics) and any(
        not row.get("ground_truth") for row in records
    ):
        raise ValueError("reference-dependent RAGAS metrics require ground_truth")
    fields = ["question", "answer", "contexts"]
    if any(row.get("ground_truth") for row in records):
        fields.append("ground_truth")
    dataset = Dataset.from_list(
        [{key: row[key] for key in fields if key in row} for row in records]
    )
    result = evaluate(
        dataset,
        metrics=_build_metrics(selected_metrics),
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=timeout,
            max_retries=max_retries,
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
                **{name: _finite_or_none(scores.get(name)) for name in selected_metrics},
            }
        )
    aggregate = {
        name: _nullable_mean(row[name] for row in question_rows)
        for name in selected_metrics
    }
    available = sum(
        row[name] is not None
        for row in question_rows
        for name in selected_metrics
    )
    expected = len(question_rows) * len(selected_metrics)
    status = "completed" if available == expected else "partial" if available else "failed"
    return {
        "status": status,
        "reason": None if status == "completed" else f"{expected - available} of {expected} metric values were unavailable",
        "metrics": list(selected_metrics),
        "aggregate": aggregate,
        "questions": question_rows,
    }


def build_ragas_clients(settings: Any) -> tuple[Any, Any]:
    """Build a judge LLM and local embeddings without exposing credentials."""

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    options = _ragas_llm_options(settings)
    llm = ChatOpenAI(**options)
    model_name = _local_embedding_snapshot(settings.EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder="data/model_cache",
        model_kwargs={"local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return llm, embeddings


def _ragas_llm_options(settings: Any) -> dict[str, Any]:
    """Resolve the OpenAI-compatible RAGAS judge without constructing it."""

    explicit_fields = getattr(settings, "model_fields_set", set())
    use_ragas_fields = bool(
        {"RAGAS_JUDGE_PROVIDER", "RAGAS_JUDGE_MODEL"}.intersection(explicit_fields)
    )
    provider = (
        getattr(settings, "RAGAS_JUDGE_PROVIDER", settings.JUDGE_PROVIDER)
        if use_ragas_fields
        else settings.JUDGE_PROVIDER
    ).strip().casefold()
    model = (
        getattr(settings, "RAGAS_JUDGE_MODEL", settings.JUDGE_MODEL)
        if use_ragas_fields
        else settings.JUDGE_MODEL
    )
    options: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": settings.JUDGE_MAX_TOKENS,
        "timeout": settings.LLM_REQUEST_TIMEOUT_SECONDS,
    }
    from langchain_core._api import LangChainBetaWarning
    from langchain_core.rate_limiters import InMemoryRateLimiter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangChainBetaWarning)
        options["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=settings.RAGAS_REQUESTS_PER_SECOND,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for the RAGAS judge")
        options.update(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            max_retries=5,
        )
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for the RAGAS judge")
        options["api_key"] = settings.OPENAI_API_KEY
        if settings.OPENAI_BASE_URL:
            options["base_url"] = settings.OPENAI_BASE_URL
    elif provider == "gemini":
        api_key = settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or OPENAI_API_KEY is required for the RAGAS judge"
            )
        options.update(
            api_key=api_key,
            base_url=settings.GEMINI_BASE_URL,
            max_retries=2,
        )
        if model.casefold().startswith("gemini-3.5"):
            options.pop("temperature", None)
    elif provider in {"qwen", "lmstudio", "lm-studio"}:
        options.update(
            api_key=settings.LMSTUDIO_API_KEY,
            base_url=settings.LMSTUDIO_BASE_URL,
        )
    else:
        raise ValueError(
            "RAGAS currently supports groq, gemini, openai, or qwen/LM Studio judge providers"
        )

    return options


def _build_metrics(
    metric_names: Sequence[str] = LEGACY_REFERENCE_FREE_METRIC_NAMES,
) -> list[Any]:
    """Use provider-compatible, independently instantiated RAGAS metrics."""

    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        ContextUtilization,
        Faithfulness,
    )

    factories = {
        "faithfulness": Faithfulness,
        # RAGAS maps strictness to the OpenAI-compatible `n` parameter, while
        # Groq accepts only n=1.
        "answer_relevancy": lambda: AnswerRelevancy(strictness=1),
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
        "context_utilization": ContextUtilization,
        "answer_correctness": AnswerCorrectness,
    }
    return [factories[name]() for name in metric_names]


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
