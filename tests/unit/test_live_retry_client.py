from __future__ import annotations

from types import SimpleNamespace

import pytest

from generation.live_retry_client import LiveRetryClient
from generation.llm_client import LLMClientError, LLMCompletion
from generation.provider_router import ProviderRouter


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _SequenceClient:
    provider = "groq"
    settings = SimpleNamespace(LLM_MODEL="test-model")

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, system, user, *, stream=False):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def acomplete(self, system, user, *, stream=False):
        return self.complete(system, user, stream=stream)


def _rate_limit(retry_after: float | None = None) -> LLMClientError:
    return LLMClientError(
        "groq", "rate limit; try again in 2s", status_code=429, retry_after=retry_after
    )


def test_waits_for_retry_after_then_returns_same_provider_answer():
    clock = _Clock()
    inner = _SequenceClient([_rate_limit(2.0), LLMCompletion("ok")])
    client = LiveRetryClient(inner, max_wait_seconds=10, sleeper=clock.sleep, clock=clock)

    assert client.complete("system", "user").text == "ok"
    assert clock.sleeps == [2.0]
    assert inner.calls == 2


def test_wall_clock_budget_exhaustion_reaches_router_fallback():
    clock = _Clock()
    first = _SequenceClient([_rate_limit(9.0), _rate_limit(9.0)])
    wrapped = LiveRetryClient(first, max_wait_seconds=3, sleeper=clock.sleep, clock=clock)
    second = _SequenceClient([LLMCompletion("fallback")])
    second.provider = "gemini"
    router = ProviderRouter([wrapped, second])

    assert router.complete("system", "user").text == "fallback"
    assert clock.sleeps == [3.0]
    assert len(router.last_attempts) == 1


def test_non_rate_limit_error_is_not_retried():
    clock = _Clock()
    inner = _SequenceClient(
        [LLMClientError("groq", "bad request", status_code=400)]
    )
    client = LiveRetryClient(inner, sleeper=clock.sleep, clock=clock)

    with pytest.raises(LLMClientError, match="bad request"):
        client.complete("system", "user")
    assert clock.sleeps == []
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_async_completion_uses_the_same_wall_clock_budget():
    clock = _Clock()

    async def async_sleep(seconds: float) -> None:
        clock.sleep(seconds)

    inner = _SequenceClient([_rate_limit(1.5), LLMCompletion("async ok")])
    client = LiveRetryClient(
        inner,
        max_wait_seconds=4,
        sleeper=clock.sleep,
        async_sleeper=async_sleep,
        clock=clock,
    )

    assert (await client.acomplete("system", "user")).text == "async ok"
    assert clock.sleeps == [1.5]
