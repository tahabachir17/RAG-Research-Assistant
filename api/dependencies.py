"""Long-lived application dependencies."""

from functools import lru_cache

from config.settings import Settings
from generation.cli import build_application_retriever
from generation.faithfulness_verifier import (
    FaithfulnessVerifier,
    build_faithfulness_verifier,
)
from generation.llm_client import LLMClient, build_llm_client
from generation.provider_router import ProviderRouter
from retrieval.reranker import CrossEncoderReranker


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_retriever():
    settings = get_settings()
    return build_application_retriever(
        settings.BM25_INDEX_PATH,
        default_top_k=max(settings.DENSE_TOP_K, settings.SPARSE_TOP_K),
        retrieval_config="hybrid",
        qdrant_path=settings.QDRANT_PATH,
        collection_name=settings.QDRANT_COLLECTION,
        embedding_model=settings.EMBEDDING_MODEL,
    )


@lru_cache(maxsize=1)
def get_llm() -> ProviderRouter:
    settings = get_settings()
    chain = (
        ("groq", settings.GROQ_MODEL),
        ("gemini", settings.GEMINI_MODEL),
        ("groq", settings.GROQ_FALLBACK_MODEL),
    )
    clients: list[LLMClient] = []
    for provider, model in chain:
        client = build_llm_client(
            settings.model_copy(
                update={"LLM_PROVIDER": provider, "LLM_MODEL": model}
            )
        )
        clients.append(client)
    return ProviderRouter(clients)


@lru_cache(maxsize=1)
def get_faithfulness_verifier() -> FaithfulnessVerifier | None:
    settings = get_settings()
    if not settings.ENABLE_FAITHFULNESS_VERIFIER:
        return None
    return build_faithfulness_verifier(settings)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    # Intentionally lazy: the cross-encoder is loaded only after explicit opt-in.
    return CrossEncoderReranker(default_top_k=get_settings().RERANK_TOP_K)
