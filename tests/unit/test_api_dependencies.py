from __future__ import annotations

from types import SimpleNamespace

from api import dependencies
from generation.live_retry_client import LiveRetryClient


def test_known_paper_titles_are_read_from_hybrid_sparse_chunks(monkeypatch):
    sparse = SimpleNamespace(
        chunks=[
            {"metadata": {"title": "Paper A"}},
            {"title": "Paper B", "metadata": {}},
            {"metadata": {"title": "Paper A"}},
            {"metadata": {}},
        ]
    )
    hybrid = SimpleNamespace(sparse_retriever=sparse)
    monkeypatch.setattr(dependencies, "get_retriever", lambda: hybrid)
    dependencies.get_known_paper_titles.cache_clear()

    try:
        assert dependencies.get_known_paper_titles() == frozenset(
            {"Paper A", "Paper B"}
        )
    finally:
        dependencies.get_known_paper_titles.cache_clear()


def test_get_llm_wraps_every_router_leg_with_live_retry(monkeypatch):
    settings = SimpleNamespace(
        GROQ_MODEL="groq-primary",
        GEMINI_MODEL="gemini-primary",
        GROQ_FALLBACK_MODEL="groq-fallback",
        LLM_RATE_LIMIT_MAX_WAIT_SECONDS=12.0,
        LLM_RATE_LIMIT_DEFAULT_WAIT_SECONDS=3.0,
        model_copy=lambda update: SimpleNamespace(**update),
    )
    built = []

    def build(config):
        client = SimpleNamespace(
            provider=config.LLM_PROVIDER,
            settings=SimpleNamespace(LLM_MODEL=config.LLM_MODEL),
        )
        built.append(client)
        return client

    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "build_llm_client", build)
    dependencies.get_llm.cache_clear()
    try:
        router = dependencies.get_llm()
        assert len(router.clients) == 3
        assert all(isinstance(client, LiveRetryClient) for client in router.clients)
        assert [client.settings.LLM_MODEL for client in router.clients] == [
            "groq-primary",
            "gemini-primary",
            "groq-fallback",
        ]
        assert all(client.max_wait_seconds == 12.0 for client in router.clients)
    finally:
        dependencies.get_llm.cache_clear()
