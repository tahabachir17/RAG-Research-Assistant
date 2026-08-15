"""Detect paper titles explicitly named in user questions."""

from __future__ import annotations

import re
from collections.abc import Iterable


_QUOTED_TITLE = re.compile(
    r"['\"\u2018\u201c]([^'\"\u2019\u201d\n]{2,200})['\"\u2019\u201d]"
)


def _normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def extract_named_papers(
    question: str, known_titles: Iterable[str] | None = None
) -> list[str]:
    """Return named paper titles in mention order.

    Corpus titles are matched as case-insensitive, whitespace-normalized
    substrings, allowing normal unquoted comparison questions. Quoted spans
    remain a fallback for titles outside the supplied corpus.
    """

    if not isinstance(question, str):
        raise TypeError("question must be a string")

    normalized_question = _normalized(question)
    candidates: list[tuple[int, int, int, str]] = []
    if known_titles is not None:
        unique_titles = {
            _normalized(title): str(title).strip()
            for title in known_titles
            if str(title).strip()
        }
        for normalized_title, title in unique_titles.items():
            boundary_pattern = re.compile(
                rf"(?<![a-z0-9]){re.escape(normalized_title)}(?![a-z0-9])"
            )
            match = boundary_pattern.search(normalized_question)
            if match is not None:
                candidates.append((match.start(), -len(normalized_title), 0, title))

    for match in _QUOTED_TITLE.finditer(question):
        title = " ".join(match.group(1).split()).strip(" ,;:.")
        normalized_title = _normalized(title)
        start = normalized_question.find(normalized_title)
        if title and start >= 0:
            candidates.append((start, -len(normalized_title), 1, title))

    titles: list[str] = []
    seen: set[str] = set()
    occupied: list[tuple[int, int]] = []
    for start, negative_length, _source_priority, title in sorted(candidates):
        end = start - negative_length
        if any(
            start < used_end and end > used_start
            for used_start, used_end in occupied
        ):
            continue
        key = _normalized(title)
        if key not in seen:
            titles.append(title)
            seen.add(key)
            occupied.append((start, end))

    return titles


def is_multi_paper_question(
    question: str,
    known_titles: Iterable[str] | None = None,
    *,
    minimum: int = 2,
) -> bool:
    """Return whether at least ``minimum`` paper titles are named."""

    if minimum < 1:
        raise ValueError("minimum must be positive")
    return len(extract_named_papers(question, known_titles)) >= minimum
