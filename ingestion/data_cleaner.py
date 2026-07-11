from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

_PAGE_NUMBER_RE = re.compile(r"(?m)^\s*\d+\s*$")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_SINGLE_LINEBREAK_RE = re.compile(r"(?<![.!?:;])\n(?!\n)")
_LATEX_COMMAND_RE = re.compile(r"\\(?:[a-zA-Z]+\*?|.)\s*(?:\{([^{}]*)\})?")
_SPACES_RE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Normalize extracted PDF text before chunking and retrieval."""

    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _PAGE_NUMBER_RE.sub("", text)
    text = _LATEX_COMMAND_RE.sub(lambda match: match.group(1) or " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = _remove_repeated_lines(text)
    text = _SINGLE_LINEBREAK_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _SPACES_RE.sub(" ", text)
    return text.strip()


def clean_sections(sectioned_doc: Any) -> Any:
    sections = getattr(sectioned_doc, "sections", None)
    if not isinstance(sections, dict):
        raise TypeError("sectioned_doc must expose a sections dictionary")
    cleaned = {name: clean_text(text) for name, text in sections.items()}
    if hasattr(sectioned_doc, "__dataclass_fields__"):
        return replace(sectioned_doc, sections=cleaned)
    sectioned_doc.sections = cleaned
    return sectioned_doc


def _remove_repeated_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    counts: dict[str, int] = {}
    for line in lines:
        normalized = re.sub(r"\s+", " ", line).strip().lower()
        if 5 <= len(normalized) <= 120:
            counts[normalized] = counts.get(normalized, 0) + 1

    total_lines = max(len(lines), 1)
    repeated = {line for line, count in counts.items() if count >= 3 and count / total_lines < 0.25}
    kept: list[str] = []
    for original in lines:
        normalized = re.sub(r"\s+", " ", original).strip().lower()
        if normalized in repeated:
            continue
        kept.append(original)
    return "\n".join(kept)