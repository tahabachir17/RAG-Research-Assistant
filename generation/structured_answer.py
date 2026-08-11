"""Strict structured cited answers with deterministic Markdown rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

MAX_STRUCTURED_CELL_WORDS = 18


@dataclass(slots=True)
class StructuredAnswerError(ValueError):
    failures: list[str]

    def __str__(self) -> str:
        return ", ".join(self.failures)


def structured_narrative_instruction(max_claims: int = 4) -> str:
    """Return the provider contract used for ordinary question answering."""

    return (
        "Return ONLY valid JSON, without a Markdown fence or explanatory text. "
        "Use this schema: {\"answer_status\": \"answered\", \"claims\": "
        "[{\"text\": \"one concise factual claim\", \"citations\": [1]}], "
        "\"summary\": \"\"}. answer_status must be answered or "
        "insufficient_evidence. For answered responses, return between 1 and "
        f"{max_claims} non-overlapping claims and leave summary empty. Every claim "
        "must be ordered so the first claim directly answers the question. "
        "must contain only one main factual assertion and at least one numbered "
        "context citation that explicitly supports it. Cite only the strongest "
        "necessary passages; do not cite a passage merely because it mentions the "
        "topic. Do not announce a count of claims, sources, or items unless the user "
        "explicitly asks for one. For insufficient_evidence, return an empty claims "
        "array and put a concise explanation in summary."
    )


def parse_and_render_structured_narrative(
    text: str,
    *,
    valid_citations: set[int],
    max_claims: int = 4,
) -> tuple[str, dict[str, Any]]:
    """Validate cited atomic claims and render them as readable Markdown."""

    try:
        payload = json.loads(_strip_fence(text))
    except (TypeError, json.JSONDecodeError):
        raise StructuredAnswerError(["structured_output_invalid_json"])
    if not isinstance(payload, Mapping):
        raise StructuredAnswerError(["structured_output_invalid_object"])
    status = str(payload.get("answer_status", "")).strip().casefold()
    summary = str(payload.get("summary", "")).strip()
    raw_claims = payload.get("claims")
    failures: list[str] = []
    if status not in {"answered", "insufficient_evidence"}:
        failures.append("structured_answer_status_invalid")
    if not isinstance(raw_claims, list):
        failures.append("structured_claims_missing")
        raw_claims = []
    if status == "answered" and not raw_claims:
        failures.append("structured_answer_empty")
    if status == "answered" and summary:
        failures.append("structured_answer_summary_not_empty")
    if len(raw_claims) > max_claims:
        failures.append("too_many_claims")
    if status == "insufficient_evidence" and raw_claims:
        failures.append("structured_abstention_has_claims")
    if status == "insufficient_evidence" and not summary:
        failures.append("structured_abstention_missing_summary")
    claims: list[dict[str, Any]] = []
    for index, raw_claim in enumerate(raw_claims, start=1):
        if not isinstance(raw_claim, Mapping):
            failures.append(f"structured_claim_invalid:{index}")
            continue
        claim_text = str(raw_claim.get("text", "")).strip()
        raw_citations = raw_claim.get("citations")
        if not claim_text:
            failures.append(f"structured_claim_empty:{index}")
        if not isinstance(raw_citations, list) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_citations
        ):
            failures.append(f"structured_citations_invalid:{index}")
            citations: list[int] = []
        else:
            citations = list(dict.fromkeys(raw_citations))
        if claim_text and not citations:
            failures.append(f"structured_claim_uncited:{index}")
        if any(value not in valid_citations for value in citations):
            failures.append(f"structured_citation_out_of_range:{index}")
        claims.append({"text": claim_text, "citations": citations})
    if failures:
        raise StructuredAnswerError(list(dict.fromkeys(failures)))
    structured = {
        "answer_status": status,
        "summary": summary,
        "claims": claims,
    }
    if status == "insufficient_evidence":
        return summary, structured
    rendered = "\n\n".join(
        f'{claim["text"]} ' + "".join(f"[{number}]" for number in claim["citations"])
        for claim in claims
    )
    return rendered, structured


def structured_answer_instruction(required_fields: Sequence[str], max_items: int | None) -> str:
    fields = [str(field).strip() for field in required_fields]
    limit = f" at most {max_items}" if max_items is not None else ""
    example_fields = ", ".join(
        f'{json.dumps(field)}: {{"text": "concise evidence-grounded value", "citations": [1]}}'
        for field in fields
    )
    return (
        "Return ONLY valid JSON, without a Markdown fence or explanatory text. "
        "Return an object with answer_status, summary, and items. answer_status must "
        "be answered or insufficient_evidence. Use insufficient_evidence only when no "
        "requested item is supported, then return an empty items array and a concise summary. "
        "For answered responses, summary must be an empty string. Keep every factual "
        "text value to at most 18 words and make it a self-contained claim, not a vague label. "
        "Each item must describe one central contribution of the paper named in the question. "
        "Put the strongest and most direct answer in the first item. "
        "The item limit is a ceiling, not a target; return fewer items when the passages do "
        "not support more distinct contributions. "
        "Do not split one method or contribution into multiple rows. Every row must have a "
        "different problem-method pair and a non-overlapping principal finding. Omit a row "
        "when every requested field would be Not reported. "
        "Do not present a baseline, related method, dataset, or generic research topic as a "
        "separate contribution. Before writing a factual value, verify that its cited passages "
        "explicitly support that value for the same method or contribution. Do not combine "
        "evidence about different methods. If support is absent or ambiguous, use exactly "
        '"Not reported in the supplied passages." with an empty citations array. '
        f'The "items" array may contain{limit} items. '
        "Every item must contain exactly the requested fields. Every factual value "
        "must be an object with text and citations. citations must contain only the "
        "numbered context passages that explicitly support that value. When a detail "
        "is absent, use text \"Not reported in the supplied passages.\" and an empty "
        f"citations array. Schema example: {{\"answer_status\": \"answered\", "
        f"\"summary\": \"\", \"items\": [{{{example_fields}}}]}}"
    )


def parse_and_render_structured_answer(
    text: str,
    *,
    required_fields: Sequence[str],
    valid_citations: set[int],
    max_items: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    failures: list[str] = []
    try:
        payload = json.loads(_strip_fence(text))
    except (TypeError, json.JSONDecodeError):
        raise StructuredAnswerError(["structured_output_invalid_json"])
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        raise StructuredAnswerError(["structured_output_missing_items"])
    status = str(payload.get("answer_status", "answered")).strip().casefold()
    summary = str(payload.get("summary", "")).strip()
    if status not in {"answered", "insufficient_evidence"}:
        failures.append("structured_answer_status_invalid")
    if status == "answered" and not items:
        failures.append("structured_answer_empty")
    if status == "answered" and summary:
        failures.append("structured_answer_summary_not_empty")
    if status == "insufficient_evidence" and items:
        failures.append("structured_abstention_has_items")
    if status == "insufficient_evidence" and not summary:
        failures.append("structured_abstention_missing_summary")
    if max_items is not None and len(items) > max_items:
        failures.append("too_many_items")
    fields = [str(field).strip() for field in required_fields]
    normalized_rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(items, start=1):
        if not isinstance(raw_row, Mapping):
            failures.append(f"structured_item_invalid:{row_index}")
            continue
        row: dict[str, Any] = {}
        factual_values = 0
        for field in fields:
            cell = raw_row.get(field)
            if not isinstance(cell, Mapping):
                failures.append(f"structured_field_missing:{row_index}:{field}")
                continue
            cell_text = str(cell.get("text", "")).strip()
            raw_citations = cell.get("citations", [])
            if not cell_text:
                failures.append(f"structured_field_empty:{row_index}:{field}")
            if len(re.findall(r"\b[\w'-]+\b", cell_text)) > MAX_STRUCTURED_CELL_WORDS:
                failures.append(f"structured_field_too_long:{row_index}:{field}")
            if not isinstance(raw_citations, list) or any(
                not isinstance(value, int) or isinstance(value, bool) for value in raw_citations
            ):
                failures.append(f"structured_citations_invalid:{row_index}:{field}")
                citations: list[int] = []
            else:
                citations = list(dict.fromkeys(raw_citations))
            if any(value not in valid_citations for value in citations):
                failures.append(f"structured_citation_out_of_range:{row_index}:{field}")
            if cell_text and not _is_absent(cell_text) and not citations:
                failures.append(f"structured_field_uncited:{row_index}:{field}")
            if cell_text and not _is_absent(cell_text):
                factual_values += 1
            row[field] = {"text": cell_text, "citations": citations}
        if factual_values == 0:
            failures.append(f"structured_item_empty:{row_index}")
        normalized_rows.append(row)
    for left_index, left in enumerate(normalized_rows):
        for right_index in range(left_index + 1, len(normalized_rows)):
            if _rows_are_near_duplicates(left, normalized_rows[right_index], fields):
                failures.append(
                    f"structured_items_duplicate:{left_index + 1}:{right_index + 1}"
                )
    if failures:
        raise StructuredAnswerError(list(dict.fromkeys(failures)))
    structured = {
        "answer_status": status,
        "summary": summary,
        "items": normalized_rows,
    }
    if status == "insufficient_evidence":
        return summary, structured
    return render_structured_answer(normalized_rows, fields), structured


def render_structured_answer(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    headers = [str(field).replace("_", " ").strip() for field in fields]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for field in fields:
            cell = row[field]
            value = str(cell["text"]).replace("|", "\\|").replace("\n", " ").strip()
            markers = "".join(f"[{number}]" for number in cell["citations"])
            cells.append(f"{value} {markers}".strip())
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _strip_fence(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def _is_absent(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return normalized.startswith("not reported") or normalized.startswith("not provided")


def _rows_are_near_duplicates(
    left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]
) -> bool:
    """Catch repeated rows without pretending to perform semantic similarity."""

    def signature(row: Mapping[str, Any]) -> str:
        values: list[str] = []
        for field in fields:
            cell = row.get(field)
            if not isinstance(cell, Mapping):
                continue
            value = str(cell.get("text", "")).strip()
            if value and not _is_absent(value):
                values.append(" ".join(re.findall(r"\w+", value.casefold())))
        return " ".join(values)

    left_text = signature(left)
    right_text = signature(right)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    return SequenceMatcher(None, left_text, right_text).ratio() >= 0.88
