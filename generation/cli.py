"""Minimal command-line harness for exercising the generation layer.

Offline mode is deterministic and requires no credentials. Use ``--live`` with
``--provider`` to call a configured Groq, Claude, OpenAI, LM Studio, or Ollama model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections.abc import AsyncIterator, Collection, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

DEFAULT_EXCLUDED_SECTIONS = frozenset(
    {
        "references",
        "bibliography",
        "front_matter",
        "acknowledgements",
        "acknowledgments",
    }
)
EVIDENCE_QUERY_SUFFIXES = (
    "method approach retrieval selection filtering reranking",
    "experiment evaluation dataset benchmark metric results improvement",
    "limitations drawbacks failure cases computational cost future work",
)
_EXPLICIT_RAG_EVIDENCE = re.compile(
    r"\bretrieval[\s-]*augmented[\s-]*generation\b|\bRAG\b", re.IGNORECASE
)


try:
    from config.settings import Settings
    from retrieval.models import RetrievalResult
    from retrieval.reranker import CrossEncoderReranker
    from retrieval.sparse_retriever import SparseRetriever
except ImportError:
    import sys

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.settings import Settings
    from retrieval.models import RetrievalResult
    from retrieval.reranker import CrossEncoderReranker
    from retrieval.sparse_retriever import SparseRetriever

try:
    from .citation_handler import CitationValidationResult, validate_citations
    from .context_assembler import ContextAssembler
    from .faithfulness_verifier import (
        FaithfulnessVerifier,
        build_faithfulness_verifier,
    )
    from .llm_client import LLMClient, build_llm_client
    from .prompt_manager import PromptManager, compound_question_instruction
    from .response_formatter import GeneratedAnswer, format_response
    from .response_validator import ResponseValidator, generate_with_validation
    from .streaming_handler import stream_answer_events
    from .structured_answer import (
        parse_and_render_structured_answer,
        parse_and_render_structured_narrative,
        structured_answer_instruction,
        structured_narrative_instruction,
    )
except ImportError:
    from citation_handler import CitationValidationResult, validate_citations
    from context_assembler import ContextAssembler
    from faithfulness_verifier import FaithfulnessVerifier, build_faithfulness_verifier
    from llm_client import LLMClient, build_llm_client
    from prompt_manager import PromptManager, compound_question_instruction
    from response_formatter import GeneratedAnswer, format_response
    from response_validator import ResponseValidator, generate_with_validation
    from streaming_handler import stream_answer_events
    from structured_answer import (
        parse_and_render_structured_answer,
        parse_and_render_structured_narrative,
        structured_answer_instruction,
        structured_narrative_instruction,
    )


class OfflineDemoClient:
    """Deterministic client proving orchestration without testing model quality."""

    answer = (
        '{"answer_status":"answered","summary":"","claims":['
        '{"text":"The supplied context describes scaled dot-product attention.",'
        '"citations":[1]}]}'
    )

    def complete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | Iterator[str]:
        if stream:
            return iter(
                ("The supplied context describes ", "scaled dot-product attention [1].")
            )
        return self.answer

    async def acomplete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | AsyncIterator[str]:
        if not stream:
            return self.answer

        async def tokens() -> AsyncIterator[str]:
            yield "The supplied context describes "
            yield "scaled dot-product attention [1]."

        return tokens()


def run_generation(
    question: str,
    ranked_results: Sequence[RetrievalResult],
    *,
    llm: LLMClient | None = None,
    template_name: str = "qa_prompt",
    max_context_tokens: int = 1000,
    show_prompt: bool = False,
    required_fields: Sequence[str] = (),
    max_items: int | None = None,
    max_retries: int | None = None,
    faithfulness_verifier: FaithfulnessVerifier | None = None,
) -> GeneratedAnswer:
    """Run generation through deterministic validation and one repair attempt."""

    started_at = time.monotonic()
    context = ContextAssembler(max_context_tokens=max_context_tokens).assemble(ranked_results)
    if not context.citation_map:
        raise ValueError("No complete retrieval chunk fits in the context budget")
    system, user = PromptManager().render(template_name, context=context.context_block, question=question)
    compound_instruction = compound_question_instruction(question)
    if compound_instruction:
        user = user + "\n\n" + compound_instruction
    response_parser = None
    if required_fields:
        user = user + "\n\n" + structured_answer_instruction(required_fields, max_items)

        def response_parser(text: str):
            return parse_and_render_structured_answer(
                text,
                required_fields=required_fields,
                valid_citations=set(context.citation_map),
                max_items=max_items,
            )
    else:
        user = user + "\n\n" + structured_narrative_instruction()

        def response_parser(text: str):
            return parse_and_render_structured_narrative(
                text,
                valid_citations=set(context.citation_map),
            )
    if show_prompt:
        print("--- SYSTEM ---")
        print(system)
        print("--- USER ---")
        print(user)
    settings = Settings()
    retries = settings.GENERATION_MAX_RETRIES if max_retries is None else max_retries
    generator = llm or OfflineDemoClient()
    outcome = generate_with_validation(
        generator,
        system,
        user,
        ResponseValidator(context.citation_map, required_fields=required_fields, max_items=max_items),
        max_retries=retries,
        response_parser=response_parser,
    )
    citation_validation = validate_citations(
        outcome.answer,
        context.citation_map,
        structured_data=outcome.structured_data,
    )
    verifier = faithfulness_verifier
    if verifier is None and settings.ENABLE_FAITHFULNESS_VERIFIER:
        verifier = build_faithfulness_verifier(settings)
    if verifier is not None:
        verifier_flags = verifier.verify(
            context,
            outcome.answer,
            structured_data=outcome.structured_data,
        )
        citation_validation.claim_support = [
            *(citation_validation.claim_support or []),
            *verifier_flags,
        ]
    if (
        isinstance(outcome.structured_data, dict)
        and outcome.structured_data.get("answer_status") == "insufficient_evidence"
        and not citation_validation.cited_numbers
    ):
        citation_validation = CitationValidationResult(
            True,
            [],
            [],
            sorted(context.citation_map),
            [],
        )
    return format_response(
        outcome.answer,
        context,
        citation_validation,
        started_at,
        finish_reason=outcome.finish_reason,
        final_attempt=outcome.final_attempt,
        validation_failures=outcome.validation.failures,
        provider=getattr(generator, "last_provider", None)
        or getattr(generator, "provider", "offline"),
        structured_data=outcome.structured_data,
    )


async def run_streaming_generation(
    question: str,
    ranked_results: Sequence[RetrievalResult],
    *,
    llm: LLMClient,
    template_name: str = "qa_prompt",
    max_context_tokens: int = 1000,
    events: bool = False,
    show_prompt: bool = False,
) -> None:
    """Print raw deltas or newline-delimited SSE-ready event payloads."""

    context = ContextAssembler(max_context_tokens=max_context_tokens).assemble(
        ranked_results
    )
    if not context.citation_map:
        raise ValueError("No complete retrieval chunk fits in the context budget")
    system, user = PromptManager().render(
        template_name, context=context.context_block, question=question
    )
    if show_prompt:
        print("--- SYSTEM ---")
        print(system)
        print("--- USER ---")
        print(user)
    async for event in stream_answer_events(llm, system, user, context.citation_map):
        if events:
            print(json.dumps(event, ensure_ascii=False), flush=True)
        elif event["type"] == "token":
            print(event["text"], end="", flush=True)
        else:
            print()
            print(json.dumps(event, indent=2, ensure_ascii=False))


def load_ranked_results(
    path: str | Path | None = None,
    *,
    config: str = "hybrid_rerank_mmr",
    query_id: str | None = None,
) -> list[RetrievalResult]:
    """Load direct result lists or a retrieval-evaluator ranking artifact."""

    if path is None:
        return _demo_results()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records: Any
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        records = payload["results"]
    elif isinstance(payload, dict) and isinstance(payload.get("rankings"), dict):
        runs = payload["rankings"].get(config)
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"No ranking runs found for configuration {config!r}")
        run = next(
            (item for item in runs if item.get("query_id") == query_id),
            runs[0] if query_id is None else None,
        )
        if run is None:
            raise ValueError(f"Query ID {query_id!r} was not found in {config!r}")
        records = run.get("results")
    else:
        raise ValueError(
            "Result JSON must be a list, contain 'results', or contain evaluator 'rankings'"
        )
    if not isinstance(records, list):
        raise ValueError("Selected result payload is not a list")
    return [_retrieval_result(record) for record in records]


def diversify_ranked_results(
    candidates: Sequence[RetrievalResult],
    *,
    top_k: int = 8,
    max_chunks_per_paper: int = 2,
    max_chunks_per_section: int = 1,
    excluded_sections: Collection[str] = DEFAULT_EXCLUDED_SECTIONS,
) -> list[RetrievalResult]:
    """Filter low-information sections while prioritizing distinct papers."""

    _positive_integer(top_k, "top_k")
    _positive_integer(max_chunks_per_paper, "max_chunks_per_paper")
    _positive_integer(max_chunks_per_section, "max_chunks_per_section")
    if isinstance(excluded_sections, (str, bytes)):
        raise TypeError("excluded_sections must be a collection of section names")
    excluded = {_normalized_section_name(value) for value in excluded_sections}
    paper_counts: dict[str, int] = {}
    section_counts: dict[tuple[str, str], int] = {}
    seen_chunks: set[str] = set()
    selected: list[RetrievalResult] = []

    def consider(candidate: RetrievalResult, *, paper_limit: int) -> None:
        if len(selected) >= top_k:
            return
        if not isinstance(candidate, RetrievalResult):
            raise TypeError("candidates may contain only RetrievalResult objects")
        if candidate.chunk_id in seen_chunks:
            return
        section = _normalized_section_name(candidate.section or "unknown")
        if section in excluded:
            return
        paper_key = _normalized_paper_key(candidate)
        section_key = (paper_key, section)
        if paper_counts.get(paper_key, 0) >= paper_limit:
            return
        if section_counts.get(section_key, 0) >= max_chunks_per_section:
            return
        selected.append(candidate)
        seen_chunks.add(candidate.chunk_id)
        paper_counts[paper_key] = paper_counts.get(paper_key, 0) + 1
        section_counts[section_key] = section_counts.get(section_key, 0) + 1

    # Cover distinct papers/methods before adding complementary sections from
    # papers already represented in the context.
    for candidate in candidates:
        consider(candidate, paper_limit=1)
    if len(selected) < top_k and max_chunks_per_paper > 1:
        for candidate in candidates:
            consider(candidate, paper_limit=max_chunks_per_paper)
    return selected


def build_evidence_queries(question: str) -> list[str]:
    """Build focused retrieval queries for methods, evaluation, and limits."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("question must not be empty")
    first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    base = first_sentence if len(normalized.split()) >= 16 else normalized
    return [normalized, *(f"{base} {suffix}" for suffix in EVIDENCE_QUERY_SUFFIXES)]


def fuse_query_results(
    rankings: Sequence[Sequence[RetrievalResult]], *, rrf_k: int = 60
) -> list[RetrievalResult]:
    """Fuse multi-query rankings by chunk ID using reciprocal-rank fusion."""

    _positive_integer(rrf_k, "rrf_k")
    scores: dict[str, float] = {}
    representatives: dict[str, RetrievalResult] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            if not isinstance(result, RetrievalResult):
                raise TypeError("rankings may contain only RetrievalResult objects")
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
            if result.chunk_id not in representatives:
                representatives[result.chunk_id] = result
                first_seen[result.chunk_id] = order
                order += 1
    ranked_ids = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return [
        replace(representatives[key], score=scores[key], source="multi_query_sparse")
        for key in ranked_ids
    ]


def prioritize_explicit_rag_evidence(
    question: str, candidates: Sequence[RetrievalResult]
) -> list[RetrievalResult]:
    """Prioritize passage-level RAG evidence when the question requires it."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence of RetrievalResult objects")
    if any(not isinstance(item, RetrievalResult) for item in candidates):
        raise TypeError("candidates may contain only RetrievalResult objects")
    normalized = question.casefold()
    requires_explicit_rag = (
        "explicit" in normalized or "only classify" in normalized
    ) and (
        "retrieval-augmented generation" in normalized
        or re.search(r"\brag\b", normalized) is not None
    )
    if not requires_explicit_rag:
        return list(candidates)
    explicit = [
        candidate
        for candidate in candidates
        if _EXPLICIT_RAG_EVIDENCE.search(candidate.text)
    ]
    explicit_ids = {candidate.chunk_id for candidate in explicit}
    return explicit + [
        candidate for candidate in candidates if candidate.chunk_id not in explicit_ids
    ]


def retrieve_ranked_results(
    question: str,
    index_path: str | Path,
    *,
    top_k: int = 5,
    candidate_k: int | None = None,
    max_chunks_per_paper: int = 2,
    max_chunks_per_section: int = 1,
    excluded_sections: Collection[str] = DEFAULT_EXCLUDED_SECTIONS,
    reranker: Any | None = None,
    expand_evidence_queries: bool = False,
) -> list[RetrievalResult]:
    """Search broadly, then select diverse evidence from the local BM25 corpus."""

    _positive_integer(top_k, "top_k")
    candidate_limit = top_k if candidate_k is None else candidate_k
    _positive_integer(candidate_limit, "candidate_k")
    candidate_limit = max(top_k, candidate_limit)
    retriever = SparseRetriever(index_path, default_top_k=candidate_limit)
    queries = (
        build_evidence_queries(question) if expand_evidence_queries else [question]
    )
    rankings = [retriever.search(query) for query in queries]
    candidates = fuse_query_results(rankings) if len(rankings) > 1 else rankings[0]
    candidates = prioritize_explicit_rag_evidence(question, candidates)
    # Keep cross-encoder work bounded even though several sparse runs were fused.
    candidates = candidates[:candidate_limit]
    if reranker is not None:
        rerank = getattr(reranker, "rerank", None)
        if not callable(rerank):
            raise TypeError("reranker must provide a rerank method")
        candidates = rerank(question, candidates, top_k=candidate_limit)
    results = diversify_ranked_results(
        candidates,
        top_k=top_k,
        max_chunks_per_paper=max_chunks_per_paper,
        max_chunks_per_section=max_chunks_per_section,
        excluded_sections=excluded_sections,
    )
    if not results:
        raise ValueError("No retrieval results were found for the supplied question")
    return results


def build_local_reranker(model_path: str | Path | None = None) -> CrossEncoderReranker:
    """Build the cross-encoder from an explicit path or the project model cache."""

    resolved = (
        Path(model_path) if model_path is not None else _cached_reranker_snapshot()
    )
    if resolved is None or not resolved.is_dir():
        raise FileNotFoundError(
            "No local cross-encoder snapshot was found; provide --reranker-model"
        )
    return CrossEncoderReranker(model_name=str(resolved))


def _cached_reranker_snapshot() -> Path | None:
    snapshots = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "model_cache"
        / "hub"
        / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"
        / "snapshots"
    )
    if not snapshots.is_dir():
        return None
    available = sorted(path for path in snapshots.iterdir() if path.is_dir())
    return available[-1] if available else None


def _normalized_section_name(value: object) -> str:
    return re.sub(r"[\s-]+", "_", str(value).strip().casefold())


def _normalized_paper_key(result: RetrievalResult) -> str:
    identity = (result.paper_id or result.title or "").strip().casefold()
    if identity:
        return re.sub(r"v\d+$", "", " ".join(identity.split()))
    return f"chunk:{result.chunk_id}"


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def add_retrieved_evidence(
    payload: dict[str, Any], ranked_results: Sequence[RetrievalResult]
) -> dict[str, Any]:
    """Attach the complete retrieved chunk to every cited source for auditing."""

    by_chunk_id = {
        result.chunk_id: (rank, result)
        for rank, result in enumerate(ranked_results, start=1)
    }
    enriched = dict(payload)
    sources: list[dict[str, Any]] = []
    for source in payload.get("sources", []):
        item = dict(source)
        match = by_chunk_id.get(str(item.get("chunk_id", "")))
        if match is not None:
            rank, result = match
            item.update(
                {
                    "retrieval_rank": rank,
                    "retrieval_score": result.score,
                    "retrieval_source": result.source,
                    "retrieved_text": result.text,
                }
            )
        sources.append(item)
    enriched["sources"] = sources
    return enriched


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.retrieve and not args.live:
        parser.error(
            "--retrieve requires --live to avoid the fixed offline demo answer"
        )
    if args.rerank and not args.retrieve:
        parser.error("--rerank requires --retrieve")
    if args.retrieve:
        results = retrieve_ranked_results(
            args.question,
            args.index_path,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            max_chunks_per_paper=args.max_chunks_per_paper,
            max_chunks_per_section=args.max_chunks_per_section,
            excluded_sections=(
                ()
                if args.include_low_information_sections
                else DEFAULT_EXCLUDED_SECTIONS
            ),
            reranker=(
                build_local_reranker(args.reranker_model) if args.rerank else None
            ),
            expand_evidence_queries=args.evidence_query_expansion,
        )
    else:
        results = load_ranked_results(
            args.results_json, config=args.config, query_id=args.query_id
        )
    llm: LLMClient = _live_client(args) if args.live else OfflineDemoClient()
    if args.stream or args.events:
        asyncio.run(
            run_streaming_generation(
                args.question,
                results,
                llm=llm,
                template_name=args.template,
                max_context_tokens=args.max_context_tokens,
                events=args.events,
                show_prompt=args.show_prompt,
            )
        )
    else:
        response = run_generation(
            args.question,
            results,
            llm=llm,
            template_name=args.template,
            max_context_tokens=args.max_context_tokens,
            show_prompt=args.show_prompt,
        )
        payload = response.to_dict()
        if args.retrieve:
            payload = add_retrieved_evidence(payload, results)
        print(_console_json(payload))
    return 0


def _console_json(payload: Any) -> str:
    """Serialize CLI output safely for Windows console code pages."""

    return json.dumps(payload, indent=2, ensure_ascii=True)


def _live_client(args: argparse.Namespace) -> LLMClient:
    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["LLM_PROVIDER"] = args.provider
    if args.model:
        overrides["LLM_MODEL"] = args.model
    if args.max_tokens is not None:
        overrides["LLM_MAX_TOKENS"] = args.max_tokens
    return build_llm_client(Settings(**overrides))


def _retrieval_result(record: Any) -> RetrievalResult:
    if not isinstance(record, dict):
        raise TypeError("Each retrieval result must be a JSON object")
    return RetrievalResult(
        chunk_id=record.get("chunk_id", ""),
        text=record.get("text", ""),
        score=record.get("score", 0.0),
        source=record.get("source", "json"),
        paper_id=record.get("paper_id"),
        title=record.get("title"),
        authors=list(record.get("authors") or []),
        year=record.get("year"),
        section=record.get("section"),
        url=record.get("url"),
        metadata=dict(record.get("metadata") or {}),
    )


def _demo_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="demo-transformer-method",
            text=(
                "The Transformer computes attention with scaled dot products "
                "between queries and keys, followed by a softmax over values."
            ),
            score=0.99,
            source="demo",
            paper_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
            section="method",
            url="https://arxiv.org/abs/1706.03762",
        )
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="What attention mechanism does the Transformer use?",
    )
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--results-json", type=Path)
    evidence.add_argument(
        "--retrieve",
        action="store_true",
        help="Search the local BM25 corpus for evidence before generation",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "bm25_index.pkl",
        help="Trusted BM25 artifact used by --retrieve",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of diverse chunks passed to generation",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
        help="BM25 candidate pool size before diversification",
    )
    parser.add_argument(
        "--max-chunks-per-paper",
        type=int,
        default=2,
        help="Maximum selected chunks from one normalized paper ID",
    )
    parser.add_argument(
        "--max-chunks-per-section",
        type=int,
        default=1,
        help="Maximum selected chunks from one section of one paper",
    )
    parser.add_argument(
        "--include-low-information-sections",
        action="store_true",
        help="Allow references, bibliography, front matter, and acknowledgements",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use the local cross-encoder before diversity selection (slower)",
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        help="Local cross-encoder model directory; defaults to the project cache",
    )
    parser.add_argument(
        "--evidence-query-expansion",
        action="store_true",
        help="Opt into four BM25 facet searches; slower and intended for difficult offline analysis",
    )
    parser.add_argument("--config", default="hybrid_rerank_mmr")
    parser.add_argument("--query-id")
    parser.add_argument("--template", default="qa_prompt")
    parser.add_argument("--max-context-tokens", type=int, default=4000)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--provider", choices=("router", "groq", "gemini", "claude", "openai", "lmstudio", "ollama")
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum completion tokens for a live model request",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--stream", action="store_true", help="Print raw text deltas")
    output.add_argument(
        "--events", action="store_true", help="Print JSON token/done events"
    )
    parser.add_argument("--show-prompt", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
