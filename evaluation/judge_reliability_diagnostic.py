"""Capture and retry anomalous Part 9 RAGAS judge outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import Settings
from langchain_core.callbacks import BaseCallbackHandler
from processing.bm25_indexer import BM25Indexer

try:
    from .full_ragas_evaluation import TieredChunkLookup
    from .ragas_evaluator import (
        _local_embedding_snapshot,
        _ragas_llm_options,
        build_ragas_records,
        evaluate_with_ragas,
    )
except ImportError:
    from full_ragas_evaluation import TieredChunkLookup
    from ragas_evaluator import (
        _local_embedding_snapshot,
        _ragas_llm_options,
        build_ragas_records,
        evaluate_with_ragas,
    )


LOGGER = logging.getLogger(__name__)


class RawJudgeCapture(BaseCallbackHandler):
    """LangChain callback that preserves model text before RAGAS parsing."""

    def __init__(self) -> None:
        self.outputs: list[str] = []

    def on_llm_end(self, response: Any, **_: Any) -> None:
        for group in getattr(response, "generations", []) or []:
            for generation in group:
                message = getattr(generation, "message", None)
                value = getattr(message, "content", None) if message is not None else None
                if value is None:
                    value = getattr(generation, "text", None)
                if value is not None:
                    self.outputs.append(str(value))


def smoke_anomalies(report: dict[str, Any]) -> list[dict[str, str]]:
    """Select invalid-JSON metric failures and suspicious hard-zero relevancy."""

    result: list[dict[str, str]] = []
    for row in report.get("ragas", {}).get("questions", []):
        identifier = str(row.get("id", ""))
        reasons = row.get("reasons") if isinstance(row.get("reasons"), dict) else {}
        for metric, reason in reasons.items():
            if "unavailable" in str(reason) and metric in {
                "faithfulness",
                "answer_relevancy",
            }:
                result.append(
                    {
                        "question_id": identifier,
                        "metric": str(metric),
                        "kind": "invalid_json",
                    }
                )
        if row.get("answer_relevancy") == 0:
            result.append(
                {
                    "question_id": identifier,
                    "metric": "answer_relevancy",
                    "kind": "hard_zero",
                }
            )
    return list(
        {
            (row["question_id"], row["metric"]): row for row in result
        }.values()
    )


def anomaly_fixed(kind: str, score: Any) -> bool:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    return value > 0.0 if kind == "hard_zero" else True


def run_judge_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    from langchain_huggingface import HuggingFaceEmbeddings

    report = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
    anomalies = smoke_anomalies(report)
    production = BM25Indexer.load(args.production_index)
    external = BM25Indexer.load(args.external_index)
    lookup = TieredChunkLookup(production, external)
    generated = {str(row["id"]): row for row in report["questions"]}
    references = _reference_answers(args.external_dir)
    embeddings = HuggingFaceEmbeddings(
        model_name=_local_embedding_snapshot(args.embedding_model),
        cache_folder="data/model_cache",
        model_kwargs={"local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    rows: list[dict[str, Any]] = []
    for position, anomaly in enumerate(anomalies, 1):
        identifier, metric = anomaly["question_id"], anomaly["metric"]
        LOGGER.info("Judge anomaly %d/%d: %s/%s", position, len(anomalies), identifier, metric)
        generation_row = generated[identifier]
        records = build_ragas_records(
            {"questions": [generation_row]},
            context_lookup=lookup,
            chunk_ids_by_question={
                identifier: list(generation_row.get("context_chunk_ids", []))
            },
            reference_by_question={identifier: references.get(identifier)},
        )
        stages = []
        baseline = _run_stage(
            records,
            metric,
            embeddings,
            provider="groq",
            model=args.primary_model,
            strict_json=False,
            timeout=args.timeout,
            requests_per_second=args.requests_per_second,
            max_tokens=args.judge_max_tokens,
        )
        stages.append({"stage": "baseline_capture", **baseline})
        strict = _run_stage(
            records,
            metric,
            embeddings,
            provider="groq",
            model=args.primary_model,
            strict_json=True,
            timeout=args.timeout,
            requests_per_second=args.requests_per_second,
            max_tokens=args.judge_max_tokens,
        )
        stages.append({"stage": "strict_json_retry", **strict})
        final = strict
        if not anomaly_fixed(anomaly["kind"], strict.get("score")):
            fallback = _run_stage(
                records,
                metric,
                embeddings,
                provider="groq",
                model=args.fallback_model,
                strict_json=True,
                timeout=args.timeout,
                requests_per_second=args.requests_per_second,
                max_tokens=args.judge_max_tokens,
            )
            stages.append({"stage": "fallback_70b", **fallback})
            final = fallback
        rows.append(
            {
                **anomaly,
                "question": generation_row["question"],
                "answer": generation_row["answer"],
                "before_score": _original_score(report, identifier, metric),
                "after_score": final.get("score"),
                "fixed": anomaly_fixed(anomaly["kind"], final.get("score")),
                "stages": stages,
            }
        )
        _atomic_json(Path(args.output), _summary(rows, complete=False))
    payload = _summary(rows, complete=True)
    _atomic_json(Path(args.output), payload)
    return payload


def _run_stage(
    records: list[dict[str, Any]],
    metric: str,
    embeddings: Any,
    *,
    provider: str,
    model: str,
    strict_json: bool,
    timeout: int,
    requests_per_second: float,
    max_tokens: int,
) -> dict[str, Any]:
    from langchain_openai import ChatOpenAI

    capture = RawJudgeCapture()
    settings = Settings().model_copy(
        update={
            "RAGAS_JUDGE_PROVIDER": provider,
            "RAGAS_JUDGE_MODEL": model,
            "JUDGE_MAX_TOKENS": max_tokens,
            "RAGAS_REQUESTS_PER_SECOND": requests_per_second,
        }
    )
    options = _ragas_llm_options(settings)
    options["callbacks"] = [capture]
    if strict_json:
        options["model_kwargs"] = {"response_format": {"type": "json_object"}}
    try:
        result = evaluate_with_ragas(
            records,
            llm=ChatOpenAI(**options),
            embeddings=embeddings,
            timeout=timeout,
            max_workers=1,
            max_retries=1,
            metric_names=[metric],
        )
        score = result.get("questions", [{}])[0].get(metric)
        return {
            "provider": provider,
            "model": model,
            "json_mode": strict_json,
            "score": score,
            "status": "completed" if score is not None else "invalid",
            "raw_outputs": capture.outputs,
        }
    except Exception as exc:
        return {
            "provider": provider,
            "model": model,
            "json_mode": strict_json,
            "score": None,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "raw_outputs": capture.outputs,
        }


def _reference_answers(external_dir: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    directory = Path(external_dir)
    for tier in ("qasa", "qasper"):
        payload = json.loads(
            (directory / f"{tier}_generation_qa.json").read_text(encoding="utf-8")
        )
        result.update(
            {
                str(row["id"]): str(row["reference_answer"])
                for row in payload.get("questions", [])
            }
        )
    return result


def _original_score(report: dict[str, Any], identifier: str, metric: str) -> Any:
    row = next(
        item for item in report["ragas"]["questions"] if str(item["id"]) == identifier
    )
    return row.get(metric)


def _summary(rows: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if complete else "in_progress",
        "before_anomalies": len(rows),
        "after_anomalies": sum(not row["fixed"] for row in rows),
        "fixed": sum(row["fixed"] for row in rows),
        "cases": rows,
        "default_used_json_mode": False,
        "diagnostic_retry_used_json_mode": True,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, default=Path("evaluation/data/eval_results/full_ragas_eval_smoke_5x3_20260812/report.json"))
    parser.add_argument("--external-dir", type=Path, default=Path("evaluation/data/external_benchmarks"))
    parser.add_argument("--production-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--external-index", type=Path, default=Path("evaluation/data/external_benchmarks/external_bm25_index.pkl"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--primary-model", default="llama-3.1-8b-instant")
    parser.add_argument("--fallback-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=0.3,
        help="Single-worker diagnostic throttle; isolated from full-eval defaults.",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/judge_reliability.json"))
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    payload = run_judge_diagnostic(args)
    print(json.dumps({key: payload[key] for key in ("before_anomalies", "fixed", "after_anomalies")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
