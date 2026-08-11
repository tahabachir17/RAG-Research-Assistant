"""Evidence-only semantic judge for generation evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from generation.llm_client import LLMClient, coerce_completion
except ImportError:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from generation.llm_client import LLMClient, coerce_completion

_ALLOWED = frozenset({"supported", "partially_supported", "unsupported"})


@dataclass(slots=True)
class JudgeVerdict:
    subject_id: str
    verdict: str
    rationale: str = ""
    check: str = "claim_support"


@dataclass(slots=True)
class JudgeResult:
    judge_status: str
    verdicts: list[JudgeVerdict]
    error: str | None = None
    cached: bool = False


class LLMJudge:
    """Judge only semantic checks using supplied evidence and no outside knowledge."""

    def __init__(self, client: LLMClient, *, judge_model: str, model_under_test: str, judge_provider: str | None = None, cache_path: str | Path | None = None, temperature: float = 0.0, rubric_version: str = "atomic-evidence-v2") -> None:
        if not judge_model.strip() or judge_model.strip() == model_under_test.strip():
            raise ValueError("judge model must be non-empty and differ from model under test")
        if temperature != 0.0:
            raise ValueError("judge temperature must be 0.0 for repeatable evaluation")
        self.client = client
        self.judge_provider = judge_provider or getattr(client, "provider", None)
        self.judge_model = judge_model
        self.model_under_test = model_under_test
        self.rubric_version = rubric_version
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = self._load_cache()
        self._cache_lock = threading.Lock()

    def judge(
        self,
        *,
        question_id: str,
        question: str,
        answer: str,
        evidence: list[dict[str, Any]],
        subjects: list[dict[str, Any]],
        inclusion_criteria: str = "",
        exclusion_criteria: dict[str, str] | None = None,
    ) -> JudgeResult:
        key = self._key(question_id, question, answer, evidence, subjects, inclusion_criteria, exclusion_criteria or {})
        with self._cache_lock:
            if key in self._cache:
                cached = _result_from_payload(self._cache[key])
                cached.cached = True
                return cached
        system = (
            "You are an evidence-only evaluator. Use no outside knowledge. Judge only: "
            "claim support by cited evidence, item qualification, limitation attribution, "
            "or whether named items are distinct. Return strict JSON with a 'verdicts' array. "
            "Each verdict must contain subject_id, check, verdict (supported, "
            "partially_supported, or unsupported), and a rationale of at most 12 words. "
            "For claim_support, use only top-level evidence entries whose citation_number "
            "appears in that subject's citations; do not use an uncited passage or transfer "
            "support between different methods, rows, or fields. "
            "Return exactly one verdict for every supplied subject_id and no others."
        )
        user = json.dumps(
            {
                "question": question,
                "answer": answer,
                "evidence": evidence,
                "subjects": subjects,
                "inclusion_criteria": inclusion_criteria,
                "exclusion_criteria": exclusion_criteria or {},
            },
            ensure_ascii=False,
        )
        try:
            verdicts = self._request_verdicts(
                system,
                user,
                {str(subject["subject_id"]) for subject in subjects},
            )
            result = JudgeResult("judged", verdicts)
        except Exception as exc:
            result = JudgeResult("unjudged", [], f"{type(exc).__name__}: {exc}")
        # Transient/exhausted failures must remain retryable on a resumed run.
        if result.judge_status == "judged":
            with self._cache_lock:
                self._cache[key] = asdict(result)
                self._save_cache()
        return result

    def _request_verdicts(
        self,
        system: str,
        user: str,
        expected_subject_ids: set[str],
    ) -> list[JudgeVerdict]:
        """Use native JSON mode and allow one format-only correction."""

        prompt = user
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                json_method = getattr(self.client, "complete_json", None)
                raw = (
                    json_method(system, prompt)
                    if callable(json_method)
                    else self.client.complete(system, prompt)
                )
                completion = coerce_completion(raw)
                payload = json.loads(_strip_fence(completion.text))
                return _parse_verdicts(payload, expected_subject_ids)
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    prompt = (
                        f"{user}\n\nYour previous response was not valid judge JSON. "
                        "Return only one JSON object containing a verdicts array; "
                        "do not include Markdown or commentary."
                    )
        assert last_error is not None
        raise last_error

    def _key(
        self,
        question_id: str,
        question: str,
        answer: str,
        evidence: list[dict[str, Any]],
        subjects: list[dict[str, Any]],
        inclusion_criteria: str,
        exclusion_criteria: dict[str, str],
    ) -> str:
        payload = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "subjects": subjects,
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "model_under_test": self.model_under_test,
            "rubric_version": self.rubric_version,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path is None or not self.cache_path.is_file():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".part")
        temporary.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, self.cache_path)


def _parse_verdicts(
    payload: Any,
    expected_subject_ids: set[str] | None = None,
) -> list[JudgeVerdict]:
    records = payload.get("verdicts") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("judge JSON must contain a verdicts array")
    verdicts: list[JudgeVerdict] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each judge verdict must be an object")
        label = str(record.get("verdict", "")).strip().casefold()
        subject_id = str(record.get("subject_id", "")).strip()
        if label not in _ALLOWED or not subject_id:
            raise ValueError("judge returned an invalid label or blank subject_id")
        verdicts.append(JudgeVerdict(subject_id, label, str(record.get("rationale", "")), str(record.get("check", "claim_support"))))
    returned_ids = [verdict.subject_id for verdict in verdicts]
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("judge returned duplicate subject_id values")
    if expected_subject_ids is not None and set(returned_ids) != expected_subject_ids:
        missing = sorted(expected_subject_ids - set(returned_ids))
        unexpected = sorted(set(returned_ids) - expected_subject_ids)
        raise ValueError(
            f"judge verdict coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    return verdicts


def _result_from_payload(payload: dict[str, Any]) -> JudgeResult:
    return JudgeResult(str(payload.get("judge_status", "unjudged")), [JudgeVerdict(**item) for item in payload.get("verdicts", [])], payload.get("error"), bool(payload.get("cached", False)))


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
    return stripped
