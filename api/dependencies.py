"""Long-lived application dependencies."""

from functools import lru_cache

from config.settings import Settings
from generation.cli import build_application_retriever
from generation.faithfulness_verifier import (
    FaithfulnessVerifier,
    build_faithfulness_verifier,
)
from generation.llm_client import LLMClient, build_llm_client
from generation.live_retry_client import LiveRetryClient
from generation.provider_router import ProviderRouter
from ingestion.paper_discovery import build_discovery
from ingestion.pipeline import IngestionPipeline
from processing.pipeline import ProcessingPipeline
from processing.qdrant_indexer import QdrantIndexer
from retrieval.fallback_retriever import CorpusEnrichmentRetriever
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
def get_known_paper_titles() -> frozenset[str]:
    """Return the canonical paper titles loaded by the sparse corpus index."""

    retriever = get_retriever()
    for _ in range(4):
        sparse = getattr(retriever, "sparse_retriever", None)
        if sparse is not None:
            retriever = sparse
            break
        nested = getattr(retriever, "retriever", None)
        if nested is None:
            break
        retriever = nested

    titles: set[str] = set()
    for chunk in getattr(retriever, "chunks", None) or []:
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata")
        nested_metadata = metadata if isinstance(metadata, dict) else {}
        title = str(chunk.get("title") or nested_metadata.get("title") or "").strip()
        if title:
            titles.add(title)
    return frozenset(titles)


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
        clients.append(
            LiveRetryClient(
                client,
                max_wait_seconds=settings.LLM_RATE_LIMIT_MAX_WAIT_SECONDS,
                default_wait_seconds=settings.LLM_RATE_LIMIT_DEFAULT_WAIT_SECONDS,
            )
        )
    return ProviderRouter(clients)


@lru_cache(maxsize=1)
def get_discovery():
    return build_discovery(get_settings().DISCOVERY_PROVIDER)


@lru_cache(maxsize=1)
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(data_dir="data", discovery=get_discovery())


@lru_cache(maxsize=1)
def get_processing_pipeline() -> ProcessingPipeline:
    """Build an indexer sharing the live retriever's Qdrant client and embedder."""

    settings = get_settings()
    retriever = get_retriever()
    dense = getattr(retriever, "dense_retriever", None)
    if dense is None:
        raise RuntimeError("Corpus enrichment requires the hybrid dense retriever")
    return ProcessingPipeline(
        embedder=dense.embedder,
        qdrant_indexer=QdrantIndexer(
            collection_name=settings.QDRANT_COLLECTION,
            client=dense.client,
        ),
    )


@lru_cache(maxsize=1)
def get_corpus_enrichment_retriever() -> CorpusEnrichmentRetriever | None:
    settings = get_settings()
    if not settings.ENABLE_CORPUS_ENRICHMENT:
        return None
    return CorpusEnrichmentRetriever(
        get_retriever(),
        discovery=get_discovery(),
        ingestion_pipeline=get_ingestion_pipeline(),
        processing_pipeline=get_processing_pipeline(),
        bm25_index_path=settings.BM25_INDEX_PATH,
        max_discovery_results=settings.CORPUS_ENRICHMENT_MAX_RESULTS,
    )


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
