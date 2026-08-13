from __future__ import annotations

from generation.context_assembler import AssembledContext, CitationSource
from generation.faithfulness_verifier import FaithfulnessVerifier
from generation.llm_client import LLMCompletion


def _context():
    return AssembledContext(
        '[1] "Dense retrieval improves grounding."',
        {
            1: CitationSource(
                1,
                "paper",
                "chunk",
                "Title",
                ["Author"],
                2024,
                "results",
                None,
                "Dense retrieval improves grounding.",
            )
        },
    )


class _VerifierClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, system, user):
        self.calls.append((system, user))
        if isinstance(self.response, Exception):
            raise self.response
        return LLMCompletion(self.response, "stop")


def test_faithfulness_verifier_returns_shared_claim_support_flags():
    client = _VerifierClient(
        '{"verdicts":[{"claim_id":"claim-1","status":"supported","reason":"Directly stated."}]}'
    )
    flags = FaithfulnessVerifier(client).verify(
        _context(),
        "Dense retrieval improves grounding [1].",
        structured_data={
            "claims": [
                {
                    "text": "Dense retrieval improves grounding",
                    "citations": [1],
                }
            ]
        },
    )

    assert flags[0].status == "supported"
    assert flags[0].checker == "llm_self_check"
    assert "Dense retrieval improves grounding" in client.calls[0][1]


def test_faithfulness_verifier_surfaces_provider_failure_as_unverified():
    flags = FaithfulnessVerifier(_VerifierClient(RuntimeError("offline"))).verify(
        _context(), "Unsupported assertion [1]."
    )

    assert flags[0].status == "unverified"
    assert flags[0].reason == "Verifier failed: RuntimeError"


def test_faithfulness_verifier_rejects_incomplete_verdict_coverage():
    flags = FaithfulnessVerifier(_VerifierClient('{"verdicts":[]}')).verify(
        _context(), "Claim [1]."
    )

    assert flags[0].status == "unverified"
    assert flags[0].reason == "Verifier failed: ValueError"


def test_faithfulness_verifier_skips_empty_context_and_answer():
    context = AssembledContext("", {})
    client = _VerifierClient(RuntimeError("must not be called"))

    assert FaithfulnessVerifier(client).verify(context, "") == []
    assert client.calls == []
