"""Provider-neutral synchronous and asynchronous LLM clients."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

try:
    from config.settings import Settings
except ImportError:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.settings import Settings


class LLMClientError(RuntimeError):
    """Normalized provider failure exposed to generation callers."""

    def __init__(
        self, provider: str, message: str, *, status_code: int | None = None
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        status = f" (status {status_code})" if status_code is not None else ""
        super().__init__(f"{provider} LLM request failed{status}: {message}")


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | Iterator[str]: ...

    async def acomplete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | AsyncIterator[str]: ...


class ClaudeClient:
    provider = "claude"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        self.settings = settings or Settings()
        if client is None or async_client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("anthropic is required for ClaudeClient") from exc
            if not self.settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is required for ClaudeClient")
            client = client or anthropic.Anthropic(
                api_key=self.settings.ANTHROPIC_API_KEY
            )
            async_client = async_client or anthropic.AsyncAnthropic(
                api_key=self.settings.ANTHROPIC_API_KEY
            )
        self.client, self.async_client = client, async_client

    def complete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | Iterator[str]:
        _validate_prompt(system, user)
        if stream:
            return self._stream(system, user)
        try:
            response = self.client.messages.create(**self._request(system, user))
            return _anthropic_text(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def acomplete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | AsyncIterator[str]:
        _validate_prompt(system, user)
        if stream:
            return self._astream(system, user)
        try:
            response = await self.async_client.messages.create(
                **self._request(system, user)
            )
            return _anthropic_text(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.settings.LLM_MODEL,
            "max_tokens": self.settings.LLM_MAX_TOKENS,
            "temperature": self.settings.LLM_TEMPERATURE,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

    def _stream(self, system: str, user: str) -> Iterator[str]:
        try:
            events = self.client.messages.create(
                **self._request(system, user), stream=True
            )
            for event in events:
                text = _anthropic_event_text(event)
                if text:
                    yield text
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def _astream(self, system: str, user: str) -> AsyncIterator[str]:
        try:
            events = await self.async_client.messages.create(
                **self._request(system, user), stream=True
            )
            async for event in events:
                text = _anthropic_event_text(event)
                if text:
                    yield text
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc


class OpenAIClient:
    provider = "openai"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        self.settings = settings or Settings()
        if client is None or async_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("openai is required for OpenAIClient") from exc
            if not self.settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAIClient")
            options = {"api_key": self.settings.OPENAI_API_KEY}
            if self.settings.OPENAI_BASE_URL:
                options["base_url"] = self.settings.OPENAI_BASE_URL
            client = client or openai.OpenAI(**options)
            async_client = async_client or openai.AsyncOpenAI(**options)
        self.client, self.async_client = client, async_client

    def complete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | Iterator[str]:
        _validate_prompt(system, user)
        if stream:
            return self._stream(system, user)
        try:
            response = self.client.chat.completions.create(
                **self._request(system, user)
            )
            return _openai_text(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def acomplete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | AsyncIterator[str]:
        _validate_prompt(system, user)
        if stream:
            return self._astream(system, user)
        try:
            response = await self.async_client.chat.completions.create(
                **self._request(system, user)
            )
            return _openai_text(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.LLM_TEMPERATURE,
            "max_tokens": self.settings.LLM_MAX_TOKENS,
        }

    def _stream(self, system: str, user: str) -> Iterator[str]:
        try:
            events = self.client.chat.completions.create(
                **self._request(system, user), stream=True
            )
            for event in events:
                text = _openai_delta(event)
                if text:
                    yield text
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def _astream(self, system: str, user: str) -> AsyncIterator[str]:
        try:
            events = await self.async_client.chat.completions.create(
                **self._request(system, user), stream=True
            )
            async for event in events:
                text = _openai_delta(event)
                if text:
                    yield text
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc


class GroqClient(OpenAIClient):
    """Groq Cloud client using Groq's official Python SDK."""

    provider = "groq"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        resolved = settings or Settings()
        if client is None or async_client is None:
            try:
                import groq
            except ImportError as exc:
                raise ImportError(
                    "groq is required for GroqClient; install it with 'pip install groq'"
                ) from exc
            if not resolved.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is required for GroqClient")
            client = client or groq.Groq(api_key=resolved.GROQ_API_KEY)
            async_client = async_client or groq.AsyncGroq(
                api_key=resolved.GROQ_API_KEY
            )
        self.settings = resolved
        self.client, self.async_client = client, async_client


class OllamaClient:
    provider = "ollama"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        self.settings = settings or Settings()
        if client is None or async_client is None:
            try:
                import ollama
            except ImportError as exc:
                raise ImportError("ollama is required for OllamaClient") from exc
            options = (
                {"host": self.settings.OLLAMA_HOST} if self.settings.OLLAMA_HOST else {}
            )
            client = client or ollama.Client(**options)
            async_client = async_client or ollama.AsyncClient(**options)
        self.client, self.async_client = client, async_client

    def complete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | Iterator[str]:
        _validate_prompt(system, user)
        if stream:
            return self._stream(system, user)
        try:
            response = self.client.chat(**self._request(system, user))
            return _ollama_text(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def acomplete(
        self, system: str, user: str, *, stream: bool = False
    ) -> str | AsyncIterator[str]:
        _validate_prompt(system, user)
        if stream:
            return self._astream(system, user)
        try:
            response = await self.async_client.chat(**self._request(system, user))
            return _ollama_text(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": self.settings.LLM_TEMPERATURE},
        }

    def _stream(self, system: str, user: str) -> Iterator[str]:
        try:
            events = self.client.chat(**self._request(system, user), stream=True)
            for event in events:
                text = _ollama_text(event)
                if text:
                    yield text
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def _astream(self, system: str, user: str) -> AsyncIterator[str]:
        try:
            events = await self.async_client.chat(
                **self._request(system, user), stream=True
            )
            async for event in events:
                text = _ollama_text(event)
                if text:
                    yield text
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    resolved = settings or Settings()
    provider = resolved.LLM_PROVIDER.strip().casefold()
    if provider in {"claude", "anthropic"}:
        return ClaudeClient(resolved)
    if provider == "openai":
        return OpenAIClient(resolved)
    if provider == "groq":
        return GroqClient(resolved)
    if provider == "ollama":
        return OllamaClient(resolved)
    raise ValueError(f"Unsupported LLM provider: {resolved.LLM_PROVIDER!r}")


def _validate_prompt(system: str, user: str) -> None:
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system prompt must be a non-empty string")
    if not isinstance(user, str) or not user.strip():
        raise ValueError("user prompt must be a non-empty string")


def _client_error(provider: str, exc: Exception) -> LLMClientError:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    return LLMClientError(provider, str(exc), status_code=status_code)


def _value(value: Any, name: str, default: Any = None) -> Any:
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _anthropic_text(response: Any) -> str:
    content = _value(response, "content", []) or []
    return "".join(str(_value(block, "text", "") or "") for block in content)


def _anthropic_event_text(event: Any) -> str:
    delta = _value(event, "delta", None)
    return str(_value(delta, "text", "") or "") if delta is not None else ""


def _openai_text(response: Any) -> str:
    choices = _value(response, "choices", []) or []
    if not choices:
        return ""
    message = _value(choices[0], "message", None)
    return str(_value(message, "content", "") or "")


def _openai_delta(event: Any) -> str:
    choices = _value(event, "choices", []) or []
    if not choices:
        return ""
    delta = _value(choices[0], "delta", None)
    return str(_value(delta, "content", "") or "")


def _ollama_text(response: Any) -> str:
    message = _value(response, "message", None)
    return str(_value(message, "content", "") or "")
