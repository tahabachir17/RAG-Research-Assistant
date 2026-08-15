"""Readiness endpoint that degrades cleanly when generation is unavailable."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from api.dependencies import get_llm, get_retriever
from api.schemas import HealthResponse
from generation.llm_client import LLMClient

router = APIRouter(tags=["health"])
_PING_TIMEOUT_SECONDS = 1.5


@router.get("/health", response_model=HealthResponse)
def health(
    retriever=Depends(get_retriever),
    llm: Annotated[LLMClient, Depends(get_llm)] = None,
) -> HealthResponse:
    retriever_ready = _retriever_has_chunks(retriever)
    llm_ready = _llm_ping(llm)
    return HealthResponse(
        status="ready" if retriever_ready and llm_ready else "degraded",
        retriever_ready=retriever_ready,
        llm_ready=llm_ready,
    )


def _retriever_has_chunks(retriever: Any) -> bool:
    sparse = getattr(retriever, "sparse_retriever", retriever)
    chunks = getattr(sparse, "chunks", None)
    return isinstance(chunks, list) and len(chunks) > 0


def _llm_ping(llm: LLMClient) -> bool:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-health")
    future = executor.submit(_provider_ping, llm)
    try:
        future.result(timeout=_PING_TIMEOUT_SECONDS)
        return True
    except (Exception, FutureTimeout):
        return False
    finally:
        # Do not let a slow provider make the readiness endpoint exceed 2 seconds.
        executor.shutdown(wait=False, cancel_futures=True)


def _provider_ping(llm: LLMClient) -> None:
    # OpenAI-compatible SDKs expose a cheaper model-list request that consumes
    # no generation tokens. Fall back to a tiny completion for other clients.
    models = getattr(getattr(llm, "client", None), "models", None)
    list_models = getattr(models, "list", None)
    if callable(list_models):
        list_models()
        return
    llm.complete("You are a health probe. Reply with OK only.", "OK?")
