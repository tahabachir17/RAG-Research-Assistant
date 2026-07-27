"""Retrieval, fusion, reranking, and diversity primitives."""

from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .mmr_sampler import MMRSampler, maximal_marginal_relevance
from .models import RetrievalResult
from .query_processor import ProcessedQuery, QueryProcessor
from .reranker import CrossEncoderReranker, Reranker
from .retriever_factory import RetrieverFactory, build_retriever, create_retriever
from .sparse_retriever import SparseRetriever

__all__ = [
    "CrossEncoderReranker",
    "DenseRetriever",
    "HybridRetriever",
    "MMRSampler",
    "ProcessedQuery",
    "QueryProcessor",
    "Reranker",
    "RetrievalResult",
    "RetrieverFactory",
    "SparseRetriever",
    "build_retriever",
    "create_retriever",
    "maximal_marginal_relevance",
]
