"""Bounded transient-failure retries and throttling for offline evaluations."""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from generation.llm_client import LLMClientError


class EvaluationRateLimitClient:
    """Retry transient provider failures during long-running evaluations.

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
        backoff_seconds: float | None = None,
        requests_per_second: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if default_wait_seconds <= 0 or max_wait_seconds <= 0:
            raise ValueError("wait values must be positive")
        if backoff_seconds is not None and backoff_seconds <= 0:
            raise ValueError("backoff_seconds must be positive")
        if requests_per_second is not None and requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.client = client
        self.provider = getattr(client, "provider", "unknown")
        self.max_retries = max_retries
        self.default_wait_seconds = default_wait_seconds
        self.max_wait_seconds = max_wait_seconds
        self.backoff_seconds = backoff_seconds or default_wait_seconds
        self.requests_per_second = requests_per_second
        self.sleeper = sleeper
        self.logger = logger or logging.getLogger(__name__)
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0

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
                self._throttle()
                return method(system, user)
            except Exception as exc:
                if not _is_transient(exc) or attempt == self.max_retries:
                    raise
                retry_after = getattr(exc, "retry_after", None)
                wait = retry_after or self.backoff_seconds * (2**attempt)
                wait = min(self.max_wait_seconds, max(0.1, wait + 0.25))
                self.logger.warning(
                    "Transient %s failure from %s; retry %d/%d in %.2fs: %s",
                    method_name,
                    self.provider,
                    attempt + 1,
                    self.max_retries,
                    wait,
                    exc,
                )
                self.sleeper(wait)
        raise AssertionError("bounded rate-limit retry loop did not return")

    def _throttle(self) -> None:
        if self.requests_per_second is None:
            return
        interval = 1.0 / self.requests_per_second
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_request_at - now)
            if wait:
                self.sleeper(wait)
                now = time.monotonic()
            self._next_request_at = now + interval


def _is_transient(exc: BaseException) -> bool:
    """Return whether a provider failure is safe to retry."""

    if isinstance(exc, (TimeoutError, ConnectionError, socket.gaierror)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    if isinstance(exc, LLMClientError):
        if exc.status_code == 429 or (exc.status_code is not None and exc.status_code >= 500):
            return True
        message = str(exc).casefold()
    else:
        message = str(exc).casefold()
    transient_markers = (
        "temporary failure in name resolution",
        "name resolution",
        "getaddrinfo",
        "dns",
        "timed out",
        "timeout",
        "connection reset",
        "connection error",
        "connection refused",
        "connection aborted",
        "service unavailable",
    )
    if any(marker in message for marker in transient_markers):
        return True
    cause = exc.__cause__ or exc.__context__
    return cause is not None and cause is not exc and _is_transient(cause)
