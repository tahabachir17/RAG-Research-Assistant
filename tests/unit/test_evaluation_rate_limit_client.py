from __future__ import annotations

import pytest

from evaluation.rate_limit_client import EvaluationRateLimitClient
from generation.llm_client import LLMClientError, LLMCompletion


class FakeClient:
    provider = "groq"

    def __init__(self, failures=1):
        self.failures = failures
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMClientError("groq", "quota", status_code=429, retry_after=1.5)
        return LLMCompletion('{"ok":true}', "stop")


def test_evaluation_client_waits_and_retries_rate_limit():
    waits = []
    base = FakeClient()
    client = EvaluationRateLimitClient(base, max_retries=2, sleeper=waits.append)
    assert client.complete_json("system", "user").text == '{"ok":true}'
    assert base.calls == 2
    assert waits == [1.75]


def test_evaluation_client_does_not_retry_non_rate_error():
    class Broken(FakeClient):
        def complete_json(self, system, user):
            raise LLMClientError("groq", "bad request", status_code=400)

    with pytest.raises(LLMClientError):
        EvaluationRateLimitClient(Broken(), sleeper=lambda value: None).complete_json("s", "u")


def test_evaluation_client_retries_dns_failure_then_succeeds():
    class DnsFlap(FakeClient):
        def complete_json(self, system, user):
            self.calls += 1
            if self.calls == 1:
                raise LLMClientError("groq", "Temporary failure in name resolution")
            return LLMCompletion('{"ok":true}', "stop")

    waits = []
    base = DnsFlap(failures=0)
    result = EvaluationRateLimitClient(
        base,
        max_retries=2,
        backoff_seconds=2,
        sleeper=waits.append,
    ).complete_json("s", "u")
    assert result.text == '{"ok":true}'
    assert base.calls == 2
    assert waits == [2.25]


def test_evaluation_client_exhausts_transient_retries():
    base = FakeClient(failures=5)
    with pytest.raises(LLMClientError):
        EvaluationRateLimitClient(
            base, max_retries=2, sleeper=lambda value: None
        ).complete_json("s", "u")
    assert base.calls == 3
