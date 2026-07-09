from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Citation:
    citation_id: str
    raw_text: str
    title: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REFERENCE_SPLIT_RE = re.compile(
    r"(?m)(?=^\s*(?:\[\d+\]|\d+\.|\d+\)|[A-Z][A-Za-z-]+,\s+[A-Z].*?\(\d{4}\)))"
)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
ARXIV_RE = re.compile(r"(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def extract_citations(sectioned_doc_or_text: Any) -> list[Citation]:
    text = _reference_text(sectioned_doc_or_text)
    if not text:
        return []
    entries = _split_references(text)
    return [
        _parse_reference(index, entry) for index, entry in enumerate(entries, start=1)
    ]


def citations_to_dicts(citations: list[Citation]) -> list[dict[str, Any]]:
    return [citation.to_dict() for citation in citations]


def _reference_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    sections = getattr(value, "sections", None)
    if isinstance(sections, dict):
        return str(sections.get("references", ""))
    if isinstance(value, dict):
        sections = value.get("sections")
        if isinstance(sections, dict):
            return str(sections.get("references", ""))
        return str(value.get("references", ""))
    return ""


def _split_references(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if not text:
        return []

    marker_pattern = re.compile(r"(?:^|\s)(?:\[\d+\]|\d{1,3}[.)])\s+")
    matches = list(marker_pattern.finditer(text))
    if len(matches) > 1:
        entries: list[str] = []
        for current, following in zip(matches, matches[1:] + [None]):
            start = current.start()
            end = following.start() if following else len(text)
            entries.append(text[start:end].strip())
        return [entry for entry in entries if len(entry.split()) >= 4]

    chunks = [
        chunk.strip() for chunk in REFERENCE_SPLIT_RE.split(text) if chunk.strip()
    ]
    if len(chunks) > 1:
        return chunks
    return [text]


def _parse_reference(index: int, raw_text: str) -> Citation:
    raw_text = raw_text.strip()
    doi_match = DOI_RE.search(raw_text)
    arxiv_match = ARXIV_RE.search(raw_text)
    year_match = YEAR_RE.search(raw_text)
    return Citation(
        citation_id=str(index),
        raw_text=raw_text,
        title=_guess_title(raw_text),
        year=int(year_match.group(1)) if year_match else None,
        doi=doi_match.group(0).rstrip(".,;) ") if doi_match else None,
        arxiv_id=arxiv_match.group(1) if arxiv_match else None,
    )


def _guess_title(raw_text: str) -> str | None:
    without_marker = re.sub(r"^\s*(?:\[\d+\]|\d+\.|\d+\))\s*", "", raw_text)
    quoted = re.search(r"[\"“](.+?)[\"”]", without_marker)
    if quoted:
        return quoted.group(1).strip()
    parts = [
        part.strip() for part in re.split(r"\.\s+", without_marker) if part.strip()
    ]
    if len(parts) >= 2:
        candidate = parts[1]
        if 3 <= len(candidate.split()) <= 30:
            return candidate.rstrip(".")
    return None
