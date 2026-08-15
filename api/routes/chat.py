"""Single-turn retrieval-augmented chat endpoint."""

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import (
    get_faithfulness_verifier,
    get_llm,
    get_reranker,
    get_retriever,
    get_settings,
)
from api.schemas import (
    ChatRequest,
    ChatResponse as BaseChatResponse,
    ErrorResponse,
    SourceChunk,
)
from config.settings import Settings
from generation.cli import retrieve_ranked_results, run_generation
from generation.entities import extract_named_papers
from generation.faithfulness_verifier import FaithfulnessVerifier
from generation.llm_client import LLMClient, LLMClientError
from generation.query_decomposition import retrieve_per_entity

router = APIRouter(tags=["chat"])


class ChatResponse(BaseChatResponse):
    """Live response plus the provider/model that produced the final answer."""

    answered_by: str


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={503: {"model": ErrorResponse}},
)
def chat(
    request: ChatRequest,
    retriever=Depends(get_retriever),
    llm: Annotated[LLMClient, Depends(get_llm)] = None,
    verifier: Annotated[
        FaithfulnessVerifier | None, Depends(get_faithfulness_verifier)
    ] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,
) -> ChatResponse:
    started_at = time.monotonic()
    named_papers = extract_named_papers(request.question)
    is_comparison = len(named_papers) >= 2
    missing_evidence: list[str] = []
    try:
        reranker = get_reranker() if request.use_rerank else None
        if is_comparison:
            ranked, entity_reports = retrieve_per_entity(
                request.question,
                str(settings.BM25_INDEX_PATH),
                retriever=retriever,
                per_entity_top_k=max(4, request.top_k),
                candidate_k=max(30, settings.HYBRID_TOP_K),
                reranker=reranker,
            )
            missing_evidence = [
                report.title for report in entity_reports if not report.hit
            ]
            generation_options = {
                "template_name": "compare_prompt",
                "required_fields": [
                    "paper", "problem", "method", "evaluation", "limitations"
                ],
                "max_items": len(named_papers),
                "exact_items": True,
                "max_context_tokens": 2500 * len(named_papers),
            }
        else:
            ranked = retrieve_ranked_results(
                request.question,
                settings.BM25_INDEX_PATH,
                top_k=request.top_k,
                candidate_k=max(request.top_k, settings.HYBRID_TOP_K),
                reranker=reranker,
                retriever=retriever,
            )
            generation_options = {"max_context_tokens": 2500}
        generated = run_generation(
            request.question,
            ranked,
            llm=llm,
            faithfulness_verifier=verifier,
            enable_faithfulness_verifier=verifier is not None,
            **generation_options,
        )
        answered_by = _answered_by(llm)
    except LLMClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation provider unavailable; try again shortly.",
        ) from exc
    except (TimeoutError, ConnectionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation provider unavailable; try again shortly.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        # Provider SDKs can raise their own exception types before normalization.
        if _looks_like_provider_failure(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Generation provider unavailable; try again shortly.",
            ) from exc
        raise

    by_id = {result.chunk_id: result for result in ranked}
    sent_results = [
        by_id[chunk_id]
        for chunk_id in generated.context_chunk_ids
        if chunk_id in by_id
    ]
    sources = [
        SourceChunk(
            chunk_id=str(source.get("chunk_id") or ""),
            paper_id=str(source.get("paper_id") or ""),
            title=str(source.get("title") or "Untitled"),
            section=str(source.get("section") or "unknown"),
            citation_number=int(source["citation_number"]),
            url=source.get("url"),
        )
        for source in generated.sources
    ]
    return ChatResponse(
        answer=generated.answer,
        sources=sources,
        retrieved_chunks=[result.text for result in sent_results],
        citations_valid=generated.citations_valid,
        latency_ms=max(0, round((time.monotonic() - started_at) * 1000)),
        used_rerank=request.use_rerank,
        answered_by=answered_by,
        named_papers=named_papers,
        papers_without_evidence=missing_evidence,
    )


def _looks_like_provider_failure(exc: Exception) -> bool:
    module = type(exc).__module__.casefold()
    name = type(exc).__name__.casefold()
    return any(
        marker in module or marker in name
        for marker in ("groq", "openai", "anthropic", "ollama", "timeout", "connection")
    )


def _answered_by(llm: LLMClient) -> str:
    """Return the successful router leg without exposing credentials."""

    clients = getattr(llm, "clients", None)
    attempts = getattr(llm, "last_attempts", None)
    if isinstance(clients, list) and isinstance(attempts, list):
        successful_index = len(attempts)
        if successful_index < len(clients):
            client = clients[successful_index]
            provider = str(getattr(client, "provider", "unknown"))
            model = getattr(getattr(client, "settings", None), "LLM_MODEL", None)
            return f"{provider}/{model}" if model else provider
    provider = str(getattr(llm, "provider", "unknown"))
    model = getattr(getattr(llm, "settings", None), "LLM_MODEL", None)
    return f"{provider}/{model}" if model else provider
