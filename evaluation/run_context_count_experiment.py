"""Measure generation quality with nested 4/5/8-chunk context pools."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import Settings
from generation import build_llm_client
from processing.bm25_indexer import BM25Indexer

from .generation_evaluator import GenerationEvaluator
from .generation_golden import GenerationGoldenQuestion, load_generation_golden
from .ragas_evaluator import build_ragas_clients, build_ragas_records, evaluate_with_ragas
from .rate_limit_client import EvaluationRateLimitClient
from .run_generation_eval import (
    FrozenChunkLookup,
    _resolve_evaluation_target,
    _resolve_generation_target,
    _validate_credentials,
)


def build_context_pools(
    questions: Sequence[GenerationGoldenQuestion],
    index: BM25Indexer,
    *,
    max_count: int,
) -> dict[str, list[str]]:
    """Keep frozen evidence first and add ranked same-paper chunks once."""

    pools: dict[str, list[str]] = {}
    for question in questions:
        selected = list(dict.fromkeys(question.retrieved_chunk_ids))
        if len(selected) < max_count:
            query_tokens = set(index.tokenize(question.question))
            candidates = sorted(
                [
                chunk
                for chunk in index.chunks
                if _same_paper(str(chunk.get("paper_id", "")), question.paper_id)
                ],
                key=lambda chunk: _candidate_chunk_key(
                    chunk, query_tokens, index.preprocessing_config
                ),
            )
            for chunk in candidates:
                chunk_id = str(chunk.get("chunk_id", ""))
                if chunk_id and chunk_id not in selected:
                    selected.append(chunk_id)
                if len(selected) >= max_count:
                    break
        if len(selected) < max_count:
            raise ValueError(
                f"{question.id}: only {len(selected)} context chunks available; "
                f"cannot run count {max_count}"
            )
        pools[question.id] = selected[:max_count]
    return pools


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    base = Settings()
    generation_provider, generation_model = _resolve_generation_target(
        base, args.provider, args.model
    )
    judge_provider, judge_model = _resolve_evaluation_target(
        base, args.judge_provider, args.judge_model
    )
    if generation_provider == judge_provider and generation_model == judge_model:
        raise ValueError("RAGAS judge must differ from the generation model")
    settings = base.model_copy(
        update={
            "LLM_PROVIDER": generation_provider,
            "LLM_MODEL": generation_model,
            "LLM_MAX_TOKENS": args.max_tokens,
            "JUDGE_PROVIDER": judge_provider,
            "JUDGE_MODEL": judge_model,
            "RAGAS_REQUESTS_PER_SECOND": args.requests_per_second,
        }
    )
    _validate_credentials(settings, judge_enabled=False, ragas_enabled=True)
    questions = load_generation_golden(args.golden)
    if args.limit is not None:
        questions = questions[: args.limit]
    index = BM25Indexer.load(args.bm25_index)
    counts = sorted(set(args.counts))
    pools = build_context_pools(questions, index, max_count=max(counts))
    lookup = FrozenChunkLookup(args.bm25_index)
    generator = EvaluationRateLimitClient(
        build_llm_client(settings), max_retries=args.rate_limit_retries
    )
    ragas_llm, embeddings = build_ragas_clients(settings)
    runs: list[dict[str, Any]] = []
    for count in counts:
        configured_questions = [
            replace(question, retrieved_chunk_ids=pools[question.id][:count])
            for question in questions
        ]
        generation = GenerationEvaluator(
            llm=generator,
            chunk_lookup=lookup,
            provider=generation_provider,
            model=generation_model,
            max_retries=settings.GENERATION_MAX_RETRIES,
            max_context_tokens=args.max_context_tokens,
        ).evaluate(configured_questions)
        records = build_ragas_records(
            generation,
            context_lookup=lookup,
            chunk_ids_by_question={
                question.id: question.retrieved_chunk_ids
                for question in configured_questions
            },
        )
        ragas = evaluate_with_ragas(
            records,
            llm=ragas_llm,
            embeddings=embeddings,
            timeout=args.ragas_timeout,
            max_workers=1,
            max_retries=args.ragas_max_retries,
        )
        runs.append(
            {
                "chunk_count": count,
                "context_chunk_ids": {
                    question.id: question.retrieved_chunk_ids
                    for question in configured_questions
                },
                "generation": generation,
                "ragas": ragas,
            }
        )
    return {
        "experiment": "generation_context_count_ablation",
        "counts": counts,
        "generation": {"provider": generation_provider, "model": generation_model},
        "ragas_judge": {"provider": judge_provider, "model": judge_model},
        "questions": len(questions),
        "runs": runs,
    }


def save_experiment(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_dir / f"context_count_experiment_{stamp}"
    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")
    markdown_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [_score_row(run) for run in result["runs"]]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Generation context-count experiment",
        "",
        "| chunks | faithfulness | answer relevancy | context utilization |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['chunk_count']} | {_metric(row['faithfulness'])} | "
            f"{_metric(row['answer_relevancy'])} | "
            f"{_metric(row['context_utilization'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _score_row(run: dict[str, Any]) -> dict[str, Any]:
    aggregate = run.get("ragas", {}).get("aggregate", {})
    return {
        "chunk_count": run["chunk_count"],
        "faithfulness": aggregate.get("faithfulness"),
        "answer_relevancy": aggregate.get("answer_relevancy"),
        "context_utilization": aggregate.get("context_utilization"),
    }


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


def _metric(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.4f}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("evaluation/data/golden_generation_qa.json"))
    parser.add_argument("--bm25-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/data/eval_results"))
    parser.add_argument("--counts", type=int, nargs="+", default=[4, 5, 8])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--provider", choices=("groq", "gemini"), default="groq")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--judge-provider", choices=("groq", "gemini", "qwen"), default="groq")
    parser.add_argument("--judge-model", default="llama-3.1-8b-instant")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=4000)
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-max-retries", type=int, default=1)
    parser.add_argument("--rate-limit-retries", type=int, default=2)
    parser.add_argument("--requests-per-second", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if any(count < 1 for count in args.counts):
        raise ValueError("all chunk counts must be positive")
    result = run_experiment(args)
    paths = save_experiment(result, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
