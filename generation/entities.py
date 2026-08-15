"""Extract paper entities explicitly named in user questions."""

from __future__ import annotations

import re


_QUOTED_TEXT = re.compile(
    r"'([^'\n]+)'|\"([^\"\n]+)\"|‘([^’\n]+)’|“([^”\n]+)”"
)


def extract_named_papers(question: str) -> list[str]:
    """Return unique, non-empty quoted paper titles in mention order."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    titles: list[str] = []
    seen: set[str] = set()
    for match in _QUOTED_TEXT.finditer(question):
        title = next(group for group in match.groups() if group is not None)
        title = " ".join(title.split()).strip(" ,;:.")
        key = title.casefold()
        if title and key not in seen:
            seen.add(key)
            titles.append(title)
    return titles


def is_multi_paper_question(question: str) -> bool:
    """Return whether the question explicitly names at least two papers."""

    return len(extract_named_papers(question)) >= 2

