from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Protocol


class PageLike(Protocol):
    text: str


@dataclass(slots=True)
class Section:
    name: str
    text: str
    start_page: int | None = None
    end_page: int | None = None


@dataclass(slots=True)
class SectionedDoc:
    paper_id: str
    sections: dict[str, str]
    section_spans: dict[str, dict[str, int | None]]

    def to_dict(self) -> dict:
        return asdict(self)


SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "abstract": re.compile(r"^(?:abstract|summary)$", re.I),
    "introduction": re.compile(r"^(?:\d+\.?\s*)?introduction$", re.I),
    "related_work": re.compile(
        r"^(?:\d+\.?\s*)?(?:related work|background|prior work)$", re.I
    ),
    "method": re.compile(
        r"^(?:\d+\.?\s*)?(?:methodology|methods?|approach|proposed method|model|architecture)$",
        re.I,
    ),
    "experiments": re.compile(
        r"^(?:\d+\.?\s*)?(?:experiments?|experimental setup|evaluation|results?|experiments and results)$",
        re.I,
    ),
    "conclusion": re.compile(
        r"^(?:\d+\.?\s*)?(?:conclusions?|discussion and conclusion)$", re.I
    ),
    "references": re.compile(r"^(?:\d+\.?\s*)?(?:references|bibliography)$", re.I),
}

DEFAULT_SECTIONS = (
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "conclusion",
    "references",
)


class SectionDetector:
    """Detect high-level research-paper sections using heading heuristics."""

    def detect(self, raw_doc: object) -> SectionedDoc:
        paper_id = str(getattr(raw_doc, "paper_id", "unknown"))
        pages = getattr(raw_doc, "pages", [])
        return extract_sections(pages, paper_id)


def detect_section(line: str) -> str | None:
    candidate = _normalize_heading(line)
    if not candidate:
        return None
    for section, pattern in SECTION_PATTERNS.items():
        if pattern.match(candidate):
            return section
    return None


def extract_sections(
    pages: Iterable[PageLike | Mapping[str, object]], paper_id: str
) -> SectionedDoc:
    sections = {section: "" for section in DEFAULT_SECTIONS}
    spans: dict[str, dict[str, int | None]] = {
        section: {"start_page": None, "end_page": None} for section in DEFAULT_SECTIONS
    }
    current_section: str | None = None

    for fallback_page_number, page in enumerate(pages, start=1):
        text = _page_text(page)
        page_number = _page_number(page, fallback_page_number)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            detected = detect_section(line)
            if detected:
                current_section = detected
                if spans[detected]["start_page"] is None:
                    spans[detected]["start_page"] = page_number
                spans[detected]["end_page"] = page_number
                continue

            if current_section and line:
                sections[current_section] += f"{line}\n"
                spans[current_section]["end_page"] = page_number

    cleaned_sections = {
        key: _join_section_lines(value) for key, value in sections.items()
    }
    return SectionedDoc(
        paper_id=paper_id, sections=cleaned_sections, section_spans=spans
    )


def _normalize_heading(line: str) -> str | None:
    value = re.sub(r"\s+", " ", line.strip())
    value = value.strip(" .:;-\t")
    if not value:
        return None
    if len(value) > 80 or len(value.split()) > 8:
        return None
    if value.endswith("."):
        return None
    lower = value.lower()
    if re.match(r"^(we|our|this|that|the|a|an|in|on|for|with|by|from|to|of)\b", lower):
        return None
    return lower


def _join_section_lines(text: str) -> str:
    text = re.sub(r"(?<![.?!:;])\n(?!\n)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _page_text(page: PageLike | Mapping[str, object]) -> str:
    if isinstance(page, Mapping):
        return str(page.get("text", ""))
    return str(getattr(page, "text", ""))


def _page_number(page: PageLike | Mapping[str, object], fallback: int) -> int:
    if isinstance(page, Mapping):
        value = page.get("page_number", fallback)
    else:
        value = getattr(page, "page_number", fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
