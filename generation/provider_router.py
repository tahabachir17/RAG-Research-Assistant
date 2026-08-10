"""Quota-aware fallback routing across zero-cost LLM providers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, overload

from config.settings import Settings

try:
    from .llm_client import LLMClient, LLMClientError, LLMCompletion, LLMStreamChunk, build_llm_client
except ImportError:
    from llm_client import LLMClient, LLMClientError, LLMCompletion, LLMStreamChunk, build_llm_client


@dataclass(slots=True)
class ProviderAttempt:
    provider: str
    status_code: int | None
    retry_after: float | None
    error: str


class ProviderRouter:
    """Try configured providers in order without hiding the final failure.

    Routing is deliberately immediate: callers do not wait through a cloud
    provider's quota window when another free provider or the local model is
    available. Transport-level retry policy remains owned by provider SDKs.
    """

    provider = "router"

    def __init__(self, clients: Sequence[LLMClient]) -> None:
        if not clients:
            raise ValueError("ProviderRouter requires at least one client")
        self.clients = list(clients)
        self.last_attempts: list[ProviderAttempt] = []
        self.last_provider: str | None = None

    @overload
    def complete(self, system: str, user: str, *, stream: Literal[False] = False) -> LLMCompletion: ...

    @overload
    def complete(self, system: str, user: str, *, stream: Literal[True]) -> Iterator[LLMStreamChunk]: ...

    def complete(self, system: str, user: str, *, stream: bool = False):
        if stream:
            # A streaming request can fail after bytes have reached the caller;
            # switching providers then would create a corrupt mixed response.
            self.last_provider = getattr(self.clients[0], "provider", "unknown")
            return self.clients[0].complete(system, user, stream=True)
        self.last_attempts = []
        for client in self.clients:
            provider = str(getattr(client, "provider", "unknown"))
            try:
                completion = client.complete(system, user)
                self.last_provider = provider
                return completion
            except LLMClientError as exc:
                self.last_attempts.append(ProviderAttempt(provider, exc.status_code, exc.retry_after, str(exc)))
        details = "; ".join(attempt.error for attempt in self.last_attempts)
        raise LLMClientError(self.provider, f"all configured providers failed: {details}")

    def complete_json(self, system: str, user: str) -> LLMCompletion:
        """Route a provider-native JSON request with the same fallback policy."""

        self.last_attempts = []
        for client in self.clients:
            provider = str(getattr(client, "provider", "unknown"))
            try:
                method = getattr(client, "complete_json", None)
                completion = method(system, user) if callable(method) else client.complete(system, user)
                self.last_provider = provider
                return completion
            except LLMClientError as exc:
                self.last_attempts.append(
                    ProviderAttempt(provider, exc.status_code, exc.retry_after, str(exc))
                )
        details = "; ".join(attempt.error for attempt in self.last_attempts)
        raise LLMClientError(self.provider, f"all configured providers failed: {details}")

    @overload
    async def acomplete(self, system: str, user: str, *, stream: Literal[False] = False) -> LLMCompletion: ...

    @overload
    async def acomplete(self, system: str, user: str, *, stream: Literal[True]) -> AsyncIterator[LLMStreamChunk]: ...

    async def acomplete(self, system: str, user: str, *, stream: bool = False):
        if stream:
            self.last_provider = getattr(self.clients[0], "provider", "unknown")
            return await self.clients[0].acomplete(system, user, stream=True)
        self.last_attempts = []
        for client in self.clients:
            provider = str(getattr(client, "provider", "unknown"))
            try:
                completion = await client.acomplete(system, user)
                self.last_provider = provider
                return completion
            except LLMClientError as exc:
                self.last_attempts.append(ProviderAttempt(provider, exc.status_code, exc.retry_after, str(exc)))
        details = "; ".join(attempt.error for attempt in self.last_attempts)
        raise LLMClientError(self.provider, f"all configured providers failed: {details}")


def build_zero_cost_router(settings: Settings | None = None) -> ProviderRouter:
    """Build Groq -> Gemini -> LM Studio routing from environment settings."""

    resolved = settings or Settings()
    providers = [value.strip().casefold() for value in resolved.ROUTER_PROVIDERS.split(",") if value.strip()]
    if not providers:
        raise ValueError("ROUTER_PROVIDERS must contain at least one provider")
    model_by_provider = {
        "groq": resolved.GROQ_MODEL,
        "gemini": resolved.GEMINI_MODEL,
        "lmstudio": resolved.LMSTUDIO_MODEL,
        "lm-studio": resolved.LMSTUDIO_MODEL,
    }
    clients = [
        build_llm_client(
            resolved.model_copy(
                update={"LLM_PROVIDER": provider, "LLM_MODEL": model_by_provider.get(provider, resolved.LLM_MODEL)}
            )
        )
        for provider in providers
    ]
    return ProviderRouter(clients)
