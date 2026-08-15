from __future__ import annotations

from types import SimpleNamespace

from api.routes import chat as chat_module
from api.schemas import ChatRequest
from config.settings import Settings
from retrieval.models import RetrievalResult


def _ranked(chunk_id: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=chunk_id,
            text=f"text {chunk_id}",
            score=1.0,
            source="test",
            paper_id="paper",
            title="Paper",
            section="method",
        )
    ]


def _generated(status: str, chunk_id: str, *, sourced: bool):
    sources = []
    context_ids = [chunk_id]
    answer = "There is not enough evidence."
    if sourced:
        answer = "Grounded answer [1]."
        sources = [
            {
                "chunk_id": chunk_id,
                "paper_id": "paper",
                "title": "Paper",
                "section": "method",
                "citation_number": 1,
                "url": None,
            }
        ]
    return SimpleNamespace(
        answer=answer,
        sources=sources,
        context_chunk_ids=context_ids,
        citations_valid=True,
        structured_data={"answer_status": status},
    )


class _Enrichment:
    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error

    def enrich(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return _ranked("new")


def _call_chat(monkeypatch, enrichment):
    retrievals = [_ranked("old"), _ranked("new")]
    generations = [
        _generated("insufficient_evidence", "old", sourced=False),
        _generated("answered", "new", sourced=True),
    ]
    monkeypatch.setattr(
        chat_module,
        "retrieve_ranked_results",
        lambda *args, **kwargs: retrievals.pop(0),
    )
    monkeypatch.setattr(
        chat_module, "run_generation", lambda *args, **kwargs: generations.pop(0)
    )
    monkeypatch.setattr(chat_module.get_known_paper_titles, "cache_clear", lambda: None)
    llm = SimpleNamespace(provider="test", settings=SimpleNamespace(LLM_MODEL="model"))
    return chat_module.chat(
        ChatRequest(question="new research", top_k=5),
        retriever=SimpleNamespace(),
        llm=llm,
        verifier=None,
        settings=Settings(ENABLE_FAITHFULNESS_VERIFIER=False),
        known_titles=frozenset(),
        enrichment=enrichment,
    )


def test_chat_enriches_only_after_abstention_and_returns_retry(monkeypatch):
    enrichment = _Enrichment()
    response = _call_chat(monkeypatch, enrichment)

    assert response.answer == "Grounded answer [1]."
    assert response.sources[0].chunk_id == "new"
    assert response.retrieved_chunks == ["text new"]
    assert len(enrichment.calls) == 1


def test_chat_preserves_initial_abstention_when_enrichment_fails(monkeypatch):
    enrichment = _Enrichment(RuntimeError("discovery unavailable"))
    retrievals = [_ranked("old")]
    initial = _generated("insufficient_evidence", "old", sourced=False)
    monkeypatch.setattr(
        chat_module,
        "retrieve_ranked_results",
        lambda *args, **kwargs: retrievals.pop(0),
    )
    monkeypatch.setattr(chat_module, "run_generation", lambda *args, **kwargs: initial)
    llm = SimpleNamespace(provider="test", settings=SimpleNamespace(LLM_MODEL="model"))

    response = chat_module.chat(
        ChatRequest(question="new research", top_k=5),
        retriever=SimpleNamespace(),
        llm=llm,
        verifier=None,
        settings=Settings(ENABLE_FAITHFULNESS_VERIFIER=False),
        known_titles=frozenset(),
        enrichment=enrichment,
    )

    assert response.answer == "There is not enough evidence."
    assert response.retrieved_chunks == ["text old"]
    assert len(enrichment.calls) == 1
