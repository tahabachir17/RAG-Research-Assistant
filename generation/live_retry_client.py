"""Bounded wait-and-retry behavior for interactive rate limits."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from generation.llm_client import LLMClientError


class LiveRetryClient:
    """Stay on one provider while a short rate-limit window expires.

    The budget is measured with a monotonic clock, so provider-call duration and
    sleep duration both count toward the configured wall-clock limit.
    """

    def __init__(
        self,
        client: Any,
        *,
        max_wait_seconds: float = 115.0,
        default_wait_seconds: float = 5.0,
        sleeper: Callable[[float], None] = time.sleep,
        async_sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must not be negative")
        if default_wait_seconds <= 0:
            raise ValueError("default_wait_seconds must be positive")
        self.client = client
        self.provider = getattr(client, "provider", "unknown")
        self.settings = getattr(client, "settings", None)
        self.max_wait_seconds = max_wait_seconds
        self.default_wait_seconds = default_wait_seconds
        self.sleeper = sleeper
        self.async_sleeper = async_sleeper
        self.clock = clock

    def complete(self, system: str, user: str, *, stream: bool = False):
        # Streaming failures can occur after output has reached the caller, when
        # retrying would splice two responses together.
        if stream:
            return self.client.complete(system, user, stream=True)
        return self._call("complete", system, user)

    def complete_json(self, system: str, user: str):
        method = (
            "complete_json"
            if callable(getattr(self.client, "complete_json", None))
            else "complete"
        )
        return self._call(method, system, user)

    async def acomplete(self, system: str, user: str, *, stream: bool = False):
        if stream:
            return await self.client.acomplete(system, user, stream=True)
        deadline = self.clock() + self.max_wait_seconds
        attempt = 0
        while True:
            try:
                return await self.client.acomplete(system, user)
            except LLMClientError as exc:
                remaining = deadline - self.clock()
                if not _is_rate_limited(exc) or remaining <= 0:
                    raise
                proposed = (
                    exc.retry_after
                    if exc.retry_after is not None and exc.retry_after > 0
                    else self.default_wait_seconds * (2**attempt)
                )
                await self.async_sleeper(min(remaining, proposed))
                attempt += 1

    def _call(self, method_name: str, system: str, user: str):
        method = getattr(self.client, method_name)
        deadline = self.clock() + self.max_wait_seconds
        attempt = 0
        while True:
            try:
                return method(system, user)
            except LLMClientError as exc:
                remaining = deadline - self.clock()
                if not _is_rate_limited(exc) or remaining <= 0:
                    raise
                proposed = (
                    exc.retry_after
                    if exc.retry_after is not None and exc.retry_after > 0
                    else self.default_wait_seconds * (2**attempt)
                )
                self.sleeper(min(remaining, proposed))
                attempt += 1


def _is_rate_limited(exc: LLMClientError) -> bool:
    message = str(exc).casefold()
    return exc.status_code == 429 or "try again in" in message or "rate limit" in message
