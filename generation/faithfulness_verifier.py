"""Optional LLM audit of generated claims against their cited evidence."""

from __future__ import annotations

import json
from typing import Any

try:
    from config.settings import Settings
except ImportError:
    from config import Settings

try:
    from .citation_handler import ClaimSupportFlag, extract_cited_claims
    from .context_assembler import AssembledContext
    from .llm_client import LLMClient, build_llm_client, coerce_completion
except ImportError:
    from citation_handler import ClaimSupportFlag, extract_cited_claims
    from context_assembler import AssembledContext
    from llm_client import LLMClient, build_llm_client, coerce_completion


class FaithfulnessVerifier:
    """Use one bounded, evidence-only model call to audit atomic claims."""

    def __init__(self, client: LLMClient) -> None:
        if not callable(getattr(client, "complete", None)) and not callable(
            getattr(client, "complete_json", None)
        ):
            raise TypeError("client must provide complete() or complete_json()")
        self.client = client

    def verify(
        self,
        assembled_context: AssembledContext,
        answer: str,
        *,
        structured_data: Any | None = None,
    ) -> list[ClaimSupportFlag]:
        if not isinstance(assembled_context, AssembledContext):
            raise TypeError("assembled_context must be AssembledContext")
        claims = extract_cited_claims(answer, structured_data)
        if not claims:
            return []
        system = (
            "Audit each claim using only its cited evidence. Return ONLY JSON with a "
            "verdicts array. Every verdict must contain claim_id, status (supported or "
            "unsupported), and a reason of at most 12 words. Return exactly one verdict "
            "for every claim_id. Do not use uncited evidence."
        )
        user = json.dumps(
            {
                "claims": [
                    {"claim_id": claim_id, "claim": claim, "citations": citations}
                    for claim_id, claim, citations in claims
                ],
                "evidence": [
                    {"citation_number": number, "text": source.text}
                    for number, source in assembled_context.citation_map.items()
                ],
            },
            ensure_ascii=False,
        )
        try:
            json_method = getattr(self.client, "complete_json", None)
            raw = (
                json_method(system, user)
                if callable(json_method)
                else self.client.complete(system, user)
            )
            payload = json.loads(_strip_fence(coerce_completion(raw).text))
            verdicts = _parse_verdicts(payload, {claim_id for claim_id, _, _ in claims})
        except Exception as exc:
            return [
                ClaimSupportFlag(
                    claim_id,
                    claim,
                    citations,
                    "unverified",
                    "llm_self_check",
                    f"Verifier failed: {type(exc).__name__}",
                )
                for claim_id, claim, citations in claims
            ]
        by_id = {claim_id: (claim, citations) for claim_id, claim, citations in claims}
        return [
            ClaimSupportFlag(
                claim_id,
                by_id[claim_id][0],
                by_id[claim_id][1],
                status,
                "llm_self_check",
                reason,
            )
            for claim_id, status, reason in verdicts
        ]


def build_faithfulness_verifier(
    settings: Settings | None = None,
) -> FaithfulnessVerifier:
    resolved = settings or Settings()
    verifier_settings = resolved.model_copy(
        update={
            "LLM_PROVIDER": resolved.FAITHFULNESS_VERIFIER_PROVIDER,
            "LLM_MODEL": resolved.FAITHFULNESS_VERIFIER_MODEL,
            "LLM_MAX_TOKENS": resolved.FAITHFULNESS_VERIFIER_MAX_TOKENS,
            "LLM_TEMPERATURE": 0.0,
        }
    )
    return FaithfulnessVerifier(build_llm_client(verifier_settings))


def _parse_verdicts(
    payload: Any, expected_ids: set[str]
) -> list[tuple[str, str, str]]:
    records = payload.get("verdicts") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("verifier JSON must contain a verdicts array")
    verdicts: list[tuple[str, str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each verifier verdict must be an object")
        claim_id = str(record.get("claim_id", "")).strip()
        status = str(record.get("status", "")).strip().casefold()
        reason = str(record.get("reason", "")).strip()
        if not claim_id or status not in {"supported", "unsupported"}:
            raise ValueError("verifier returned an invalid claim_id or status")
        verdicts.append((claim_id, status, reason))
    returned = [claim_id for claim_id, _, _ in verdicts]
    if len(returned) != len(set(returned)) or set(returned) != expected_ids:
        raise ValueError("verifier verdict coverage mismatch")
    return verdicts


def _strip_fence(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1])
    return stripped
