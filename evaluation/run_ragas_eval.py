"""Add reference-free RAGAS scores to an existing generation-evaluation result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from config import Settings

from .generation_evaluator import save_generation_outputs
from .generation_golden import load_generation_golden
from .ragas_evaluator import build_ragas_clients, build_ragas_records, evaluate_with_ragas
from .run_generation_eval import FrozenChunkLookup, _resolve_evaluation_target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--golden", type=Path, default=Path("evaluation/data/golden_generation_qa.json"))
    parser.add_argument("--bm25-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/data/eval_results"))
    parser.add_argument(
        "--judge-provider", choices=("groq", "gemini", "qwen")
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--requests-per-second", type=float)
    args = parser.parse_args(argv)

    base_settings = Settings()
    judge_provider, judge_model = _resolve_evaluation_target(
        base_settings, args.judge_provider, args.judge_model
    )
    overrides = {
        "JUDGE_PROVIDER": judge_provider,
        "JUDGE_MODEL": judge_model,
    }
    if args.judge_max_tokens:
        overrides["JUDGE_MAX_TOKENS"] = args.judge_max_tokens
    if args.requests_per_second:
        overrides["RAGAS_REQUESTS_PER_SECOND"] = args.requests_per_second
    settings = base_settings.model_copy(update=overrides)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    questions = load_generation_golden(args.golden)
    lookup = FrozenChunkLookup(args.bm25_index)
    records = build_ragas_records(
        result,
        context_lookup=lookup,
        chunk_ids_by_question={question.id: question.retrieved_chunk_ids for question in questions},
    )
    llm, embeddings = build_ragas_clients(settings)
    result["ragas"] = evaluate_with_ragas(
        records,
        llm=llm,
        embeddings=embeddings,
        timeout=args.timeout,
        max_workers=args.workers,
    )
    result["ragas"]["judge"] = {
        "provider": settings.JUDGE_PROVIDER,
        "model": settings.JUDGE_MODEL,
    }
    paths = save_generation_outputs(result, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
