from __future__ import annotations

import pytest

from config import Settings
from generation.llm_client import LLMClientError, LLMCompletion
from generation.provider_router import ProviderRouter, build_zero_cost_router


class FakeClient:
    def __init__(self, provider, result):
        self.provider, self.result, self.calls = provider, result, 0

    def complete(self, system, user, stream=False):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeJsonClient(FakeClient):
    def complete_json(self, system, user):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_router_falls_back_after_rate_limit():
    groq = FakeClient("groq", LLMClientError("groq", "quota", status_code=429))
    gemini = FakeClient("gemini", LLMCompletion("answer", "stop"))
    local = FakeClient("lmstudio", LLMCompletion("local", "stop"))
    router = ProviderRouter([groq, gemini, local])
    assert router.complete("system", "user").text == "answer"
    assert router.last_provider == "gemini"
    assert [attempt.provider for attempt in router.last_attempts] == ["groq"]
    assert local.calls == 0


def test_router_reports_all_failures():
    clients = [
        FakeClient("groq", LLMClientError("groq", "quota", status_code=429)),
        FakeClient("gemini", LLMClientError("gemini", "down", status_code=503)),
    ]
    with pytest.raises(LLMClientError, match="all configured providers failed"):
        ProviderRouter(clients).complete("system", "user")


def test_router_uses_native_json_method_when_available():
    client = FakeJsonClient("groq", LLMCompletion('{"ok":true}', "stop"))
    router = ProviderRouter([client])
    assert router.complete_json("system", "user").text == '{"ok":true}'
    assert router.last_provider == "groq"


def test_zero_cost_router_uses_provider_specific_models(monkeypatch):
    observed = []

    def fake_builder(settings):
        observed.append((settings.LLM_PROVIDER, settings.LLM_MODEL))
        return FakeClient(settings.LLM_PROVIDER, LLMCompletion("ok", "stop"))

    monkeypatch.setattr("generation.provider_router.build_llm_client", fake_builder)
    settings = Settings(
        _env_file=None,
        ROUTER_PROVIDERS="groq,gemini,lmstudio",
        GROQ_MODEL="groq-model",
        GEMINI_MODEL="gemini-model",
        LMSTUDIO_MODEL="local-model",
    )
    router = build_zero_cost_router(settings)
    assert len(router.clients) == 3
    assert observed == [
        ("groq", "groq-model"),
        ("gemini", "gemini-model"),
        ("lmstudio", "local-model"),
    ]
