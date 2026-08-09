"""Pure generation-quality metrics with no provider dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def qualifying_item_precision(predicted: Iterable[str], expected: Iterable[str]) -> float | None:
    predicted_set, expected_set = _normalized_set(predicted), _normalized_set(expected)
    if not expected_set:
        return None
    return len(predicted_set & expected_set) / len(predicted_set) if predicted_set else 0.0


def qualifying_item_recall(predicted: Iterable[str], expected: Iterable[str]) -> float | None:
    predicted_set, expected_set = _normalized_set(predicted), _normalized_set(expected)
    return len(predicted_set & expected_set) / len(expected_set) if expected_set else None


def citation_validity_rate(validity: Iterable[bool]) -> float:
    return _boolean_rate(validity)


def claim_level_citation_coverage(claims: Sequence[Mapping[str, Any] | bool]) -> float:
    values = [bool(claim if isinstance(claim, bool) else claim.get("cited", False)) for claim in claims]
    return _boolean_rate(values)


def required_field_completeness(present: Iterable[str], required: Iterable[str]) -> float:
    present_set, required_set = _normalized_set(present), _normalized_set(required)
    return len(present_set & required_set) / len(required_set) if required_set else 1.0


def truncation_rate(finish_reasons: Iterable[str | None]) -> float:
    truncated = {"length", "max_tokens", "max_output_tokens", "model_length"}
    reasons = list(finish_reasons)
    if not reasons:
        return 0.0
    return sum(bool(reason and str(reason).casefold() in truncated) for reason in reasons) / len(reasons)


def unsupported_claim_rate(verdicts: Iterable[str]) -> float | None:
    labels = [str(verdict).strip().casefold() for verdict in verdicts]
    judged = [label for label in labels if label in {"supported", "partially_supported", "unsupported"}]
    return sum(label == "unsupported" for label in judged) / len(judged) if judged else None


def incorrect_classification_rate(verdicts: Iterable[bool]) -> float:
    values = list(verdicts)
    return sum(not bool(value) for value in values) / len(values) if values else 0.0


def retry_rate(retry_counts: Iterable[int]) -> float:
    counts = list(retry_counts)
    return sum(count > 0 for count in counts) / len(counts) if counts else 0.0


def _boolean_rate(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(bool(value) for value in items) / len(items) if items else 0.0


def _normalized_set(values: Iterable[str]) -> set[str]:
    return {" ".join(str(value).strip().casefold().split()) for value in values if str(value).strip()}
