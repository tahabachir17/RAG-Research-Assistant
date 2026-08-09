"""Deterministic response validation and one bounded repair path."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

try:
    from .citation_handler import validate_citations
    from .context_assembler import CitationSource
    from .llm_client import LLMClient, LLMCompletion, coerce_completion
except ImportError:
    from citation_handler import validate_citations
    from context_assembler import CitationSource
    from llm_client import LLMClient, LLMCompletion, coerce_completion

_TRUNCATION_REASONS = frozenset({"length", "max_tokens", "max_output_tokens", "model_length"})


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationAttempt:
    attempt: str
    completion: LLMCompletion
    validation: ValidationResult
    latency_ms: float


@dataclass(slots=True)
class ValidatedGeneration:
    answer: str
    finish_reason: str | None
    final_attempt: str
    validation: ValidationResult
    attempts: list[GenerationAttempt]

    @property
    def retry_count(self) -> int:
        return max(0, len(self.attempts) - 1)


class ResponseValidator:
    """Validate a response against the exact context and per-task contract."""

    def __init__(
        self,
        citation_map: dict[int, CitationSource],
        *,
        required_fields: Sequence[str] = (),
        max_items: int | None = None,
        item_counter: Callable[[str, Any | None], int | None] | None = None,
    ) -> None:
        if not isinstance(citation_map, dict):
            raise TypeError("citation_map must be a dictionary")
        if isinstance(required_fields, (str, bytes)):
            raise TypeError("required_fields must be a sequence of names")
        fields = [str(field).strip() for field in required_fields]
        if any(not field for field in fields):
            raise ValueError("required_fields may not contain blanks")
        if max_items is not None and (not isinstance(max_items, int) or isinstance(max_items, bool) or max_items <= 0):
            raise ValueError("max_items must be a positive integer or None")
        if item_counter is not None and not callable(item_counter):
            raise TypeError("item_counter must be callable")
        self.citation_map = dict(citation_map)
        self.required_fields = list(dict.fromkeys(fields))
        self.max_items = max_items
        self.item_counter = item_counter or _infer_item_count

    def validate(
        self,
        answer: str,
        *,
        finish_reason: str | None = None,
        structured_data: Any | None = None,
        item_count: int | None = None,
    ) -> ValidationResult:
        failures: list[str] = []
        if _is_truncated(finish_reason):
            failures.append("truncated")
        citations = validate_citations(answer, self.citation_map)
        if self.citation_map and not citations.cited_numbers:
            failures.append("missing_citation")
        if citations.unknown_numbers:
            failures.append("citation_out_of_range")
        if citations.unsupported_markers:
            failures.append("unsupported_citation_format")
        fields, incomplete_table = _present_fields(answer, structured_data)
        for field in self.required_fields:
            if _normalize_field(field) not in fields:
                failures.append(f"missing_required_field:{field}")
        if incomplete_table:
            failures.append("incomplete_table")
        count = item_count if item_count is not None else self.item_counter(answer, structured_data)
        if count is not None and self.max_items is not None and count > self.max_items:
            failures.append("too_many_items")
        return ValidationResult(not failures, list(dict.fromkeys(failures)))


def generate_with_validation(
    client: LLMClient,
    system: str,
    user: str,
    validator: ResponseValidator,
    *,
    max_retries: int = 1,
    logger: Any | None = None,
) -> ValidatedGeneration:
    """Call the production LLM path, repairing invalid output at most once."""

    if max_retries not in {0, 1}:
        raise ValueError("max_retries currently supports only 0 or 1")
    attempts: list[GenerationAttempt] = []
    prompt = user
    for index in range(max_retries + 1):
        started = time.perf_counter()
        completion = coerce_completion(client.complete(system, prompt))
        validation = validator.validate(completion.text, finish_reason=completion.finish_reason)
        attempt = GenerationAttempt(
            "original" if index == 0 else "repaired",
            completion,
            validation,
            (time.perf_counter() - started) * 1000.0,
        )
        attempts.append(attempt)
        if logger is not None:
            logger.info("generation_validation", extra={"attempt": attempt.attempt, "failures": validation.failures})
        if validation.valid or index == max_retries:
            return ValidatedGeneration(completion.text, completion.finish_reason, attempt.attempt, validation, attempts)
        prompt = _repair_prompt(user, completion.text, validation.failures)
    raise AssertionError("bounded generation loop did not return")


def _repair_prompt(original_user: str, answer: str, failures: Sequence[str]) -> str:
    return (
        f"{original_user}\n\nYour previous answer failed deterministic validation. "
        "Return a complete corrected answer only; do not discuss the repair.\n"
        f"Failures: {json.dumps(list(failures), ensure_ascii=False)}\n"
        f"Previous answer:\n{answer}"
    )


def _is_truncated(reason: str | None) -> bool:
    return bool(reason and str(reason).strip().casefold() in _TRUNCATION_REASONS)


def _normalize_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _present_fields(answer: str, structured_data: Any | None) -> tuple[set[str], bool]:
    fields: set[str] = set()
    incomplete = False
    if isinstance(structured_data, Mapping):
        fields.update(_normalize_field(str(key)) for key in structured_data)
    elif isinstance(structured_data, Sequence) and not isinstance(structured_data, (str, bytes)):
        rows = [row for row in structured_data if isinstance(row, Mapping)]
        if rows:
            key_sets = [{_normalize_field(str(key)) for key in row} for row in rows]
            fields.update(set.intersection(*key_sets))
            incomplete = any(keys != key_sets[0] for keys in key_sets[1:])
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    table_rows = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if table_rows:
        cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in table_rows]
        if cells:
            fields.update(_normalize_field(cell) for cell in cells[0])
            expected = len(cells[0])
            incomplete = incomplete or any(len(row) != expected or any(not cell for cell in row) for row in cells[2:])
    for match in re.finditer(r"(?im)^\s*(?:[-*]\s*)?([A-Za-z][\w -]{1,50})\s*:\s*\S", answer):
        fields.add(_normalize_field(match.group(1)))
    return fields, incomplete


def _infer_item_count(answer: str, structured_data: Any | None) -> int | None:
    if isinstance(structured_data, Sequence) and not isinstance(structured_data, (str, bytes)):
        return len(structured_data)
    table_rows = [line for line in answer.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(table_rows) >= 2:
        return max(0, len(table_rows) - 2)
    enumerated = re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+", answer)
    return len(enumerated) if enumerated else None