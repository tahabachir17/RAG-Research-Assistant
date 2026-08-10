"""Bounded rate-limit retries for offline evaluation runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from generation.llm_client import LLMClientError


class EvaluationRateLimitClient:
    """Wait on 429 responses during batch evaluation instead of aborting.

    Interactive routing intentionally does not use this wrapper because users
    should receive an immediate provider fallback. Evaluation prefers preserving
    the selected model comparison and can safely wait for a short quota reset.
    """

    def __init__(
        self,
        client: Any,
        *,
        max_retries: int = 2,
        default_wait_seconds: float = 10.0,
        max_wait_seconds: float = 60.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if default_wait_seconds <= 0 or max_wait_seconds <= 0:
            raise ValueError("wait values must be positive")
        self.client = client
        self.provider = getattr(client, "provider", "unknown")
        self.max_retries = max_retries
        self.default_wait_seconds = default_wait_seconds
        self.max_wait_seconds = max_wait_seconds
        self.sleeper = sleeper

    def complete(self, system: str, user: str, *, stream: bool = False):
        if stream:
            return self.client.complete(system, user, stream=True)
        return self._call("complete", system, user)

    def complete_json(self, system: str, user: str):
        method = "complete_json" if callable(getattr(self.client, "complete_json", None)) else "complete"
        return self._call(method, system, user)

    def _call(self, method_name: str, system: str, user: str):
        method = getattr(self.client, method_name)
        for attempt in range(self.max_retries + 1):
            try:
                return method(system, user)
            except LLMClientError as exc:
                if exc.status_code != 429 or attempt == self.max_retries:
                    raise
                wait = exc.retry_after or self.default_wait_seconds
                self.sleeper(min(self.max_wait_seconds, max(0.1, wait + 0.25)))
        raise AssertionError("bounded rate-limit retry loop did not return")
