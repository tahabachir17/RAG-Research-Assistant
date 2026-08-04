from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import Settings
from generation import llm_client as llm_module
from generation.llm_client import (
    ClaudeClient,
    LLMClientError,
    OllamaClient,
    OpenAIClient,
    build_llm_client,
)


class _AsyncEvents:
    def __init__(self, values):
        self.values = iter(values)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.values)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _SyncCreate:
    def __init__(self, response, events):
        self.response, self.events, self.calls = response, events, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.events) if kwargs.get("stream") else self.response


class _AsyncCreate(_SyncCreate):
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _AsyncEvents(self.events) if kwargs.get("stream") else self.response


def _settings(provider):
    return Settings(
        _env_file=None,
        LLM_PROVIDER=provider,
        LLM_MODEL="test-model",
        ANTHROPIC_API_KEY="test",
        OPENAI_API_KEY="test",
    )


@pytest.mark.asyncio
async def test_claude_client_normalizes_sync_async_and_streaming_text():
    response = SimpleNamespace(content=[SimpleNamespace(text="answer")])
    events = [
        SimpleNamespace(delta=SimpleNamespace(text="a")),
        SimpleNamespace(delta=SimpleNamespace(text="b")),
    ]
    sync = _SyncCreate(response, events)
    async_create = _AsyncCreate(response, events)
    client = ClaudeClient(
        _settings("claude"),
        client=SimpleNamespace(messages=sync),
        async_client=SimpleNamespace(messages=async_create),
    )

    assert client.complete("system", "user") == "answer"
    assert list(client.complete("system", "user", stream=True)) == ["a", "b"]
    assert await client.acomplete("system", "user") == "answer"
    stream = await client.acomplete("system", "user", stream=True)
    assert [text async for text in stream] == ["a", "b"]
    assert sync.calls[0]["messages"][0]["content"] == "user"


@pytest.mark.asyncio
async def test_openai_client_normalizes_sync_async_and_streaming_text():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    events = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="x"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="y"))]),
    ]
    sync = _SyncCreate(response, events)
    async_create = _AsyncCreate(response, events)
    client = OpenAIClient(
        _settings("openai"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        async_client=SimpleNamespace(chat=SimpleNamespace(completions=async_create)),
    )

    assert client.complete("system", "user") == "answer"
    assert list(client.complete("system", "user", stream=True)) == ["x", "y"]
    assert await client.acomplete("system", "user") == "answer"
    stream = await client.acomplete("system", "user", stream=True)
    assert [text async for text in stream] == ["x", "y"]


class _OllamaSync:
    def chat(self, **kwargs):
        events = [{"message": {"content": "one"}}, {"message": {"content": "two"}}]
        return (
            iter(events) if kwargs.get("stream") else {"message": {"content": "answer"}}
        )


class _OllamaAsync:
    async def chat(self, **kwargs):
        events = [{"message": {"content": "one"}}, {"message": {"content": "two"}}]
        return (
            _AsyncEvents(events)
            if kwargs.get("stream")
            else {"message": {"content": "answer"}}
        )


@pytest.mark.asyncio
async def test_ollama_client_normalizes_sync_async_and_streaming_text():
    client = OllamaClient(
        _settings("ollama"), client=_OllamaSync(), async_client=_OllamaAsync()
    )
    assert client.complete("system", "user") == "answer"
    assert list(client.complete("system", "user", stream=True)) == ["one", "two"]
    assert await client.acomplete("system", "user") == "answer"
    stream = await client.acomplete("system", "user", stream=True)
    assert [text async for text in stream] == ["one", "two"]


def test_provider_errors_are_wrapped_with_status_and_factory_rejects_unknown():
    class ProviderFailure(RuntimeError):
        status_code = 429

    class Broken:
        def create(self, **kwargs):
            raise ProviderFailure("rate limited")

    client = ClaudeClient(
        _settings("claude"),
        client=SimpleNamespace(messages=Broken()),
        async_client=SimpleNamespace(messages=Broken()),
    )
    with pytest.raises(LLMClientError, match="status 429") as caught:
        client.complete("system", "user")
    assert caught.value.provider == "claude"
    assert caught.value.status_code == 429

    with pytest.raises(ValueError, match="Unsupported"):
        build_llm_client(_settings("other"))


@pytest.mark.parametrize(
    ("provider", "attribute"),
    [
        ("claude", "ClaudeClient"),
        ("openai", "OpenAIClient"),
        ("ollama", "OllamaClient"),
    ],
)
def test_factory_selects_each_provider_without_live_clients(
    monkeypatch, provider, attribute
):
    sentinel = object()
    monkeypatch.setattr(llm_module, attribute, lambda settings: sentinel)
    assert build_llm_client(_settings(provider)) is sentinel
