from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import Settings
from generation import llm_client as llm_module
from generation.llm_client import (
    ClaudeClient,
    GeminiClient,
    LLMClientError,
    LMStudioClient,
    OllamaClient,
    OpenAIClient,
    build_llm_client,
    coerce_completion,
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
        LLM_REQUEST_TIMEOUT_SECONDS=20.0,
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


def test_openai_json_completion_requests_json_object_mode():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
    )
    sync = _SyncCreate(response, [])
    client = OpenAIClient(
        _settings("openai"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        async_client=SimpleNamespace(chat=SimpleNamespace(completions=_AsyncCreate(response, []))),
    )
    assert client.complete_json("system", "return json").text == '{"ok":true}'
    assert sync.calls[0]["response_format"] == {"type": "json_object"}


def test_json_mode_preserves_provider_rejected_draft_for_repair():
    class JsonFailure(RuntimeError):
        status_code = 400
        body = {
            "error": {
                "code": "json_validate_failed",
                "failed_generation": '{"citations":[1][2]}',
            }
        }

    class Broken:
        def create(self, **kwargs):
            raise JsonFailure("invalid json")

    client = OpenAIClient(
        _settings("openai"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=Broken())),
        async_client=SimpleNamespace(chat=SimpleNamespace(completions=Broken())),
    )
    completion = client.complete_json("system", "return json")
    assert completion.text == '{"citations":[1][2]}'
    assert completion.finish_reason == "provider_json_validation_failed"


def test_lmstudio_client_uses_local_openai_compatibility_settings(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    fake_module = SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeOpenAI)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    client = LMStudioClient(_settings("lmstudio"))
    assert client.provider == "lmstudio"
    assert created == [
        {"api_key": "lm-studio", "base_url": "http://127.0.0.1:1234/v1", "timeout": 20.0, "max_retries": 0},
        {"api_key": "lm-studio", "base_url": "http://127.0.0.1:1234/v1", "timeout": 20.0, "max_retries": 0},
    ]


def test_lmstudio_json_completion_uses_supported_text_mode():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
    )
    sync = _SyncCreate(response, [])
    client = LMStudioClient(
        _settings("lmstudio"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        async_client=SimpleNamespace(
            chat=SimpleNamespace(completions=_AsyncCreate(response, []))
        ),
    )

    assert client.complete_json("system", "return json").text == '{"ok":true}'
    assert sync.calls[0]["response_format"] == {"type": "text"}


def test_gemini_client_uses_compatibility_endpoint_and_own_provider(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAI, AsyncOpenAI=FakeOpenAI),
    )
    settings = _settings("gemini")
    settings.GEMINI_API_KEY = "gemini-test"
    client = GeminiClient(settings)
    assert client.provider == "gemini"
    assert created == [
        {"api_key": "gemini-test", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "timeout": 20.0, "max_retries": 0},
        {"api_key": "gemini-test", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "timeout": 20.0, "max_retries": 0},
    ]


def test_gemini_35_request_omits_deprecated_temperature():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
    )
    sync = _SyncCreate(response, [])
    settings = _settings("gemini")
    settings.GEMINI_API_KEY = "gemini-test"
    settings.LLM_MODEL = "gemini-3.5-flash-lite"
    client = GeminiClient(
        settings,
        client=SimpleNamespace(chat=SimpleNamespace(completions=sync)),
        async_client=SimpleNamespace(
            chat=SimpleNamespace(completions=_AsyncCreate(response, []))
        ),
    )

    client.complete_json("system", "return json")
    assert "temperature" not in sync.calls[0]

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


def test_provider_error_captures_retry_after_header():
    class Response:
        status_code = 429
        headers = {"retry-after": "2.5"}

    class ProviderFailure(RuntimeError):
        response = Response()

    class Broken:
        def create(self, **kwargs):
            raise ProviderFailure("quota")

    client = ClaudeClient(
        _settings("claude"),
        client=SimpleNamespace(messages=Broken()),
        async_client=SimpleNamespace(messages=Broken()),
    )
    with pytest.raises(LLMClientError) as caught:
        client.complete("system", "user")
    assert caught.value.retry_after == 2.5


def test_provider_error_parses_retry_delay_from_message():
    class ProviderFailure(RuntimeError):
        status_code = 429

    class Broken:
        def create(self, **kwargs):
            raise ProviderFailure("Please try again in 7.075s")

    client = ClaudeClient(
        _settings("claude"),
        client=SimpleNamespace(messages=Broken()),
        async_client=SimpleNamespace(messages=Broken()),
    )
    with pytest.raises(LLMClientError) as caught:
        client.complete("system", "user")
    assert caught.value.retry_after == 7.075


@pytest.mark.parametrize(
    ("provider", "attribute"),
    [
        ("claude", "ClaudeClient"),
        ("openai", "OpenAIClient"),
        ("gemini", "GeminiClient"),
        ("lmstudio", "LMStudioClient"),
        ("ollama", "OllamaClient"),
    ],
)
def test_factory_selects_each_provider_without_live_clients(
    monkeypatch, provider, attribute
):
    sentinel = object()
    monkeypatch.setattr(llm_module, attribute, lambda settings: sentinel)
    assert build_llm_client(_settings(provider)) is sentinel

@pytest.mark.parametrize(
    ("provider", "response", "expected"),
    [
        ("claude", SimpleNamespace(content=[SimpleNamespace(text="cut")], stop_reason="max_tokens", usage=SimpleNamespace(input_tokens=2, output_tokens=3)), "max_tokens"),
        ("openai", SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="cut"), finish_reason="length")], usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3)), "length"),
    ],
)
def test_nonstreaming_completion_captures_finish_reason(provider, response, expected):
    sync = _SyncCreate(response, [])
    if provider == "claude":
        client = ClaudeClient(_settings(provider), client=SimpleNamespace(messages=sync), async_client=SimpleNamespace(messages=_AsyncCreate(response, [])))
    else:
        client = OpenAIClient(_settings(provider), client=SimpleNamespace(chat=SimpleNamespace(completions=sync)), async_client=SimpleNamespace(chat=SimpleNamespace(completions=_AsyncCreate(response, []))))
    completion = client.complete("system", "user")
    assert completion.finish_reason == expected
    assert completion.output_tokens == 3


def test_ollama_completion_captures_done_reason():
    class Sync:
        def chat(self, **kwargs):
            return {"message": {"content": "cut"}, "done_reason": "length", "eval_count": 4}
    client = OllamaClient(_settings("ollama"), client=Sync(), async_client=_OllamaAsync())
    completion = client.complete("system", "user")
    assert completion.finish_reason == "length"
    assert completion.output_tokens == 4


def test_completion_repairs_common_windows_1252_mojibake():
    assert coerce_completion("San Franciscoâ€™s freeway").text == "San Francisco’s freeway"
