"""Provider-neutral synchronous and asynchronous LLM clients."""

from __future__ import annotations

import sys
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, overload, runtime_checkable

try:
    from config.settings import Settings
except ImportError:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config.settings import Settings


class LLMClientError(RuntimeError):
    """Normalized provider failure exposed to generation callers."""

    def __init__(self, provider: str, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        status = f" (status {status_code})" if status_code is not None else ""
        super().__init__(f"{provider} LLM request failed{status}: {message}")


@dataclass(slots=True)
class LLMCompletion:
    text: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if not isinstance(other, LLMCompletion):
            return NotImplemented
        return (self.text, self.finish_reason, self.input_tokens, self.output_tokens) == (other.text, other.finish_reason, other.input_tokens, other.output_tokens)


@dataclass(slots=True)
class LLMStreamChunk:
    text: str = ""
    finish_reason: str | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if not isinstance(other, LLMStreamChunk):
            return NotImplemented
        return (self.text, self.finish_reason) == (other.text, other.finish_reason)


@runtime_checkable
class LLMClient(Protocol):
    @overload
    def complete(
        self, system: str, user: str, *, stream: Literal[False] = False
    ) -> LLMCompletion: ...

    @overload
    def complete(
        self, system: str, user: str, *, stream: Literal[True]
    ) -> Iterator[LLMStreamChunk]: ...

    @overload
    def complete(
        self, system: str, user: str, *, stream: bool
    ) -> LLMCompletion | Iterator[LLMStreamChunk]: ...

    @overload
    async def acomplete(
        self, system: str, user: str, *, stream: Literal[False] = False
    ) -> LLMCompletion: ...

    @overload
    async def acomplete(
        self, system: str, user: str, *, stream: Literal[True]
    ) -> AsyncIterator[LLMStreamChunk]: ...

    @overload
    async def acomplete(
        self, system: str, user: str, *, stream: bool
    ) -> LLMCompletion | AsyncIterator[LLMStreamChunk]: ...


class ClaudeClient:
    provider = "claude"

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None, async_client: Any | None = None) -> None:
        self.settings = settings or Settings()
        if client is None or async_client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("anthropic is required for ClaudeClient") from exc
            if not self.settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is required for ClaudeClient")
            options = {
                "api_key": self.settings.ANTHROPIC_API_KEY,
                "timeout": self.settings.LLM_REQUEST_TIMEOUT_SECONDS,
                "max_retries": self.settings.LLM_TRANSPORT_MAX_RETRIES,
            }
            client = client or anthropic.Anthropic(**options)
            async_client = async_client or anthropic.AsyncAnthropic(**options)
        self.client, self.async_client = client, async_client

    def complete(self, system: str, user: str, *, stream: bool = False) -> LLMCompletion | Iterator[LLMStreamChunk]:
        _validate_prompt(system, user)
        if stream:
            return self._stream(system, user)
        try:
            return _anthropic_completion(self.client.messages.create(**self._request(system, user)))
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def acomplete(self, system: str, user: str, *, stream: bool = False) -> LLMCompletion | AsyncIterator[LLMStreamChunk]:
        _validate_prompt(system, user)
        if stream:
            return self._astream(system, user)
        try:
            response = await self.async_client.messages.create(**self._request(system, user))
            return _anthropic_completion(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {"model": self.settings.LLM_MODEL, "max_tokens": self.settings.LLM_MAX_TOKENS, "temperature": self.settings.LLM_TEMPERATURE, "system": system, "messages": [{"role": "user", "content": user}]}

    def _stream(self, system: str, user: str) -> Iterator[LLMStreamChunk]:
        try:
            for event in self.client.messages.create(**self._request(system, user), stream=True):
                chunk = _anthropic_stream_chunk(event)
                if chunk.text or chunk.finish_reason:
                    yield chunk
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def _astream(self, system: str, user: str) -> AsyncIterator[LLMStreamChunk]:
        try:
            events = await self.async_client.messages.create(**self._request(system, user), stream=True)
            async for event in events:
                chunk = _anthropic_stream_chunk(event)
                if chunk.text or chunk.finish_reason:
                    yield chunk
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc


class OpenAIClient:
    provider = "openai"

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None, async_client: Any | None = None) -> None:
        self.settings = settings or Settings()
        if client is None or async_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("openai is required for OpenAIClient") from exc
            if not self.settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAIClient")
            options: dict[str, Any] = {
                "api_key": self.settings.OPENAI_API_KEY,
                "timeout": self.settings.LLM_REQUEST_TIMEOUT_SECONDS,
                "max_retries": self.settings.LLM_TRANSPORT_MAX_RETRIES,
            }
            if self.settings.OPENAI_BASE_URL:
                options["base_url"] = self.settings.OPENAI_BASE_URL
            client = client or openai.OpenAI(**options)
            async_client = async_client or openai.AsyncOpenAI(**options)
        self.client, self.async_client = client, async_client

    def complete(self, system: str, user: str, *, stream: bool = False) -> LLMCompletion | Iterator[LLMStreamChunk]:
        _validate_prompt(system, user)
        if stream:
            return self._stream(system, user)
        try:
            return _openai_completion(self.client.chat.completions.create(**self._request(system, user)))
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    def complete_json(self, system: str, user: str) -> LLMCompletion:
        """Request provider-enforced JSON for deterministic downstream parsing."""

        _validate_prompt(system, user)
        try:
            request = self._request(system, user)
            request["response_format"] = self._json_response_format()
            return _openai_completion(self.client.chat.completions.create(**request))
        except Exception as exc:
            failed_generation = _failed_json_generation(exc)
            if failed_generation is not None:
                # Some compatible providers return the rejected draft inside a
                # 400 response. Preserve it so deterministic parsing can report
                # the exact failure and the bounded repair prompt can correct it.
                return LLMCompletion(
                    failed_generation,
                    "provider_json_validation_failed",
                )
            raise _client_error(self.provider, exc) from exc

    def _json_response_format(self) -> dict[str, Any]:
        return {"type": "json_object"}

    async def acomplete(self, system: str, user: str, *, stream: bool = False) -> LLMCompletion | AsyncIterator[LLMStreamChunk]:
        _validate_prompt(system, user)
        if stream:
            return self._astream(system, user)
        try:
            response = await self.async_client.chat.completions.create(**self._request(system, user))
            return _openai_completion(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {"model": self.settings.LLM_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": self.settings.LLM_TEMPERATURE, "max_tokens": self.settings.LLM_MAX_TOKENS}

    def _stream(self, system: str, user: str) -> Iterator[LLMStreamChunk]:
        try:
            for event in self.client.chat.completions.create(**self._request(system, user), stream=True):
                chunk = _openai_stream_chunk(event)
                if chunk.text or chunk.finish_reason:
                    yield chunk
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def _astream(self, system: str, user: str) -> AsyncIterator[LLMStreamChunk]:
        try:
            events = await self.async_client.chat.completions.create(**self._request(system, user), stream=True)
            async for event in events:
                chunk = _openai_stream_chunk(event)
                if chunk.text or chunk.finish_reason:
                    yield chunk
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc


class GroqClient(OpenAIClient):
    provider = "groq"

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None, async_client: Any | None = None) -> None:
        resolved = settings or Settings()
        if client is None or async_client is None:
            try:
                import groq
            except ImportError as exc:
                raise ImportError("groq is required for GroqClient; install it with 'pip install groq'") from exc
            if not resolved.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is required for GroqClient")
            options = {
                "api_key": resolved.GROQ_API_KEY,
                "timeout": resolved.LLM_REQUEST_TIMEOUT_SECONDS,
                "max_retries": resolved.LLM_TRANSPORT_MAX_RETRIES,
            }
            client = client or groq.Groq(**options)
            async_client = async_client or groq.AsyncGroq(**options)
        self.settings, self.client, self.async_client = resolved, client, async_client


class GeminiClient(OpenAIClient):
    """Gemini's OpenAI-compatible endpoint with truthful provider metadata."""

    provider = "gemini"

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None, async_client: Any | None = None) -> None:
        resolved = settings or Settings()
        if client is None or async_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("openai is required for GeminiClient") from exc
            api_key = resolved.GEMINI_API_KEY or resolved.OPENAI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY is required for GeminiClient")
            options = {
                "api_key": api_key,
                "base_url": resolved.GEMINI_BASE_URL.rstrip("/") + "/",
                "timeout": resolved.LLM_REQUEST_TIMEOUT_SECONDS,
                "max_retries": resolved.LLM_TRANSPORT_MAX_RETRIES,
            }
            client = client or openai.OpenAI(**options)
            async_client = async_client or openai.AsyncOpenAI(**options)
        self.settings, self.client, self.async_client = resolved, client, async_client

    def _request(self, system: str, user: str) -> dict[str, Any]:
        request = super()._request(system, user)
        if self.settings.LLM_MODEL.casefold().startswith("gemini-3.5"):
            request.pop("temperature", None)
        return request


class LMStudioClient(OpenAIClient):
    """OpenAI-compatible client configured for LM Studio's local API server."""

    provider = "lmstudio"

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None, async_client: Any | None = None) -> None:
        resolved = settings or Settings()
        if client is None or async_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("openai is required for LMStudioClient") from exc
            options = {
                "api_key": resolved.LMSTUDIO_API_KEY,
                "base_url": resolved.LMSTUDIO_BASE_URL.rstrip("/"),
                "timeout": resolved.LLM_REQUEST_TIMEOUT_SECONDS,
                "max_retries": resolved.LLM_TRANSPORT_MAX_RETRIES,
            }
            client = client or openai.OpenAI(**options)
            async_client = async_client or openai.AsyncOpenAI(**options)
        self.settings, self.client, self.async_client = resolved, client, async_client

    def _json_response_format(self) -> dict[str, Any]:
        """Use LM Studio text mode; the strict parser validates the JSON draft.

        LM Studio rejects OpenAI's ``json_object`` mode, while a generic
        ``json_schema`` can make small local models generate until timeout.
        The prompt still requests JSON exclusively and the response validator
        rejects or repairs malformed output before it reaches callers.
        """

        return {"type": "text"}

class OllamaClient:
    provider = "ollama"

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None, async_client: Any | None = None) -> None:
        self.settings = settings or Settings()
        if client is None or async_client is None:
            try:
                import ollama
            except ImportError as exc:
                raise ImportError("ollama is required for OllamaClient") from exc
            options = {"host": self.settings.OLLAMA_HOST} if self.settings.OLLAMA_HOST else {}
            client = client or ollama.Client(**options)
            async_client = async_client or ollama.AsyncClient(**options)
        self.client, self.async_client = client, async_client

    def complete(self, system: str, user: str, *, stream: bool = False) -> LLMCompletion | Iterator[LLMStreamChunk]:
        _validate_prompt(system, user)
        if stream:
            return self._stream(system, user)
        try:
            return _ollama_completion(self.client.chat(**self._request(system, user)))
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def acomplete(self, system: str, user: str, *, stream: bool = False) -> LLMCompletion | AsyncIterator[LLMStreamChunk]:
        _validate_prompt(system, user)
        if stream:
            return self._astream(system, user)
        try:
            response = await self.async_client.chat(**self._request(system, user))
            return _ollama_completion(response)
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    def _request(self, system: str, user: str) -> dict[str, Any]:
        return {"model": self.settings.LLM_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "options": {"temperature": self.settings.LLM_TEMPERATURE}}

    def _stream(self, system: str, user: str) -> Iterator[LLMStreamChunk]:
        try:
            for event in self.client.chat(**self._request(system, user), stream=True):
                chunk = _ollama_stream_chunk(event)
                if chunk.text or chunk.finish_reason:
                    yield chunk
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc

    async def _astream(self, system: str, user: str) -> AsyncIterator[LLMStreamChunk]:
        try:
            events = await self.async_client.chat(**self._request(system, user), stream=True)
            async for event in events:
                chunk = _ollama_stream_chunk(event)
                if chunk.text or chunk.finish_reason:
                    yield chunk
        except Exception as exc:
            raise _client_error(self.provider, exc) from exc


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    resolved = settings or Settings()
    provider = resolved.LLM_PROVIDER.strip().casefold()
    if provider in {"claude", "anthropic"}:
        return ClaudeClient(resolved)
    if provider == "openai":
        return OpenAIClient(resolved)
    if provider == "gemini":
        return GeminiClient(resolved)
    if provider in {"lmstudio", "lm-studio"}:
        return LMStudioClient(resolved)
    if provider == "groq":
        return GroqClient(resolved)
    if provider == "ollama":
        return OllamaClient(resolved)
    if provider in {"router", "auto"}:
        try:
            from .provider_router import build_zero_cost_router
        except ImportError:
            from provider_router import build_zero_cost_router

        return build_zero_cost_router(resolved)
    raise ValueError(f"Unsupported LLM provider: {resolved.LLM_PROVIDER!r}")


def coerce_completion(value: LLMCompletion | str) -> LLMCompletion:
    """Normalize legacy injected fakes while production clients return metadata."""

    if isinstance(value, LLMCompletion):
        return LLMCompletion(
            _repair_common_mojibake(value.text),
            value.finish_reason,
            value.input_tokens,
            value.output_tokens,
        )
    if isinstance(value, str):
        return LLMCompletion(_repair_common_mojibake(value))
    raise TypeError("non-streaming LLM completion must be LLMCompletion or string")


def _repair_common_mojibake(text: str) -> str:
    """Repair UTF-8 text mistakenly decoded as Windows-1252 by local servers."""

    if not any(marker in text for marker in ("Ã", "Â", "â", "ðŸ")):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _validate_prompt(system: str, user: str) -> None:
    if not isinstance(system, str) or not system.strip():
        raise ValueError("system prompt must be a non-empty string")
    if not isinstance(user, str) or not user.strip():
        raise ValueError("user prompt must be a non-empty string")


def _client_error(provider: str, exc: Exception) -> LLMClientError:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    retry_value = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        retry_after = float(retry_value) if retry_value is not None else None
    except (TypeError, ValueError):
        retry_after = None
    if retry_after is None:
        match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", str(exc), re.IGNORECASE)
        if match:
            retry_after = float(match.group(1))
    return LLMClientError(
        provider,
        str(exc),
        status_code=status_code,
        retry_after=retry_after,
    )


def _failed_json_generation(exc: Exception) -> str | None:
    """Extract a provider-rejected JSON draft without rewriting its content."""

    bodies: list[Any] = [getattr(exc, "body", None)]
    response = getattr(exc, "response", None)
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            bodies.append(json_method())
        except Exception:
            pass
    for body in bodies:
        if not isinstance(body, dict):
            continue
        error = body.get("error", body)
        if not isinstance(error, dict):
            continue
        value = error.get("failed_generation")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _value(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _anthropic_completion(response: Any) -> LLMCompletion:
    content = _value(response, "content", []) or []
    usage = _value(response, "usage", None)
    return LLMCompletion("".join(str(_value(block, "text", "") or "") for block in content), _value(response, "stop_reason", None), _integer_or_none(_value(usage, "input_tokens", None)), _integer_or_none(_value(usage, "output_tokens", None)))


def _anthropic_stream_chunk(event: Any) -> LLMStreamChunk:
    delta = _value(event, "delta", None)
    return LLMStreamChunk(str(_value(delta, "text", "") or "") if delta is not None else "", _value(delta, "stop_reason", None))


def _openai_completion(response: Any) -> LLMCompletion:
    choices = _value(response, "choices", []) or []
    if not choices:
        return LLMCompletion("")
    message, usage = _value(choices[0], "message", None), _value(response, "usage", None)
    return LLMCompletion(str(_value(message, "content", "") or ""), _value(choices[0], "finish_reason", None), _integer_or_none(_value(usage, "prompt_tokens", None)), _integer_or_none(_value(usage, "completion_tokens", None)))


def _openai_stream_chunk(event: Any) -> LLMStreamChunk:
    choices = _value(event, "choices", []) or []
    if not choices:
        return LLMStreamChunk()
    delta = _value(choices[0], "delta", None)
    return LLMStreamChunk(str(_value(delta, "content", "") or ""), _value(choices[0], "finish_reason", None))


def _ollama_completion(response: Any) -> LLMCompletion:
    message = _value(response, "message", None)
    return LLMCompletion(str(_value(message, "content", "") or ""), _value(response, "done_reason", None), _integer_or_none(_value(response, "prompt_eval_count", None)), _integer_or_none(_value(response, "eval_count", None)))


def _ollama_stream_chunk(response: Any) -> LLMStreamChunk:
    message = _value(response, "message", None)
    finish = _value(response, "done_reason", None) if _value(response, "done", False) else None
    return LLMStreamChunk(str(_value(message, "content", "") or ""), finish)
