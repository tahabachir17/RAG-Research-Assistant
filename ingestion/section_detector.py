from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
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
    section_details: list[dict[str, object]] = field(default_factory=list)
    heading_diagnostics: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


SECTION_ALIASES: dict[str, str] = {
    "abstract": "abstract",
    "summary": "abstract",
    "introduction": "introduction",
    "background": "related_work",
    "preliminaries": "related_work",
    "related work": "related_work",
    "prior work": "related_work",
    "literature review": "related_work",
    "methodology": "methodology",
    "method": "methodology",
    "methods": "methodology",
    "proposed method": "methodology",
    "proposed approach": "methodology",
    "approach": "methodology",
    "model": "methodology",
    "architecture": "methodology",
    "experimental setup": "experiments",
    "experiment": "experiments",
    "experiments": "experiments",
    "implementation details": "experiments",
    "dataset": "experiments",
    "datasets": "experiments",
    "evaluation": "experiments",
    "experiments and results": "results",
    "experimental results": "results",
    "result": "results",
    "results": "results",
    "analysis": "analysis",
    "discussion": "discussion",
    "limitations": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "future work": "conclusion",
    "conclusion and future work": "conclusion",
    "conclusions and future work": "conclusion",
    "discussion and conclusion": "conclusion",
    "acknowledgements": "acknowledgements",
    "acknowledgments": "acknowledgements",
    "references": "references",
    "bibliography": "references",
    "works cited": "references",
    "literature cited": "references",
    "appendix": "appendix",
    "appendices": "appendix",
    "supplementary material": "supplementary_material",
    "supplemental material": "supplementary_material",
}

DEFAULT_SECTIONS = (
    "front_matter",
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "experiments",
    "results",
    "analysis",
    "discussion",
    "limitations",
    "conclusion",
    "acknowledgements",
    "references",
    "appendix",
    "supplementary_material",
    "other",
)

_NUMBERED_RE = re.compile(
    r"^(?P<number>(?:\d+(?:\.\d+)*|[IVXLCDM]+))[.)]?\s+(?P<title>.+)$", re.I
)
_CAPTION_RE = re.compile(r"^(?:fig(?:ure)?|table|algorithm|equation|eq\.)\s*\d+", re.I)
_BULLET_RE = re.compile(r"^(?:[-*•]|\(?[a-z]\))\s+", re.I)


class SectionDetector:
    """Detect scientific-paper sections without discarding unclassified text."""

    def __init__(self) -> None:
        self.last_diagnostics: list[dict[str, object]] = []

    def detect(self, raw_doc: object) -> SectionedDoc:
        result = extract_sections(
            getattr(raw_doc, "pages", []), str(getattr(raw_doc, "paper_id", "unknown"))
        )
        self.last_diagnostics = result.heading_diagnostics
        return result

    def diagnostics(self) -> list[dict[str, object]]:
        return list(self.last_diagnostics)


def detect_section(line: str) -> str | None:
    detected = _detect_heading(line, True, True)
    return str(detected["canonical_section"]) if detected else None


def extract_sections(
    pages: Iterable[PageLike | Mapping[str, object]], paper_id: str
) -> SectionedDoc:
    sections = {name: "" for name in DEFAULT_SECTIONS}
    spans = {name: {"start_page": None, "end_page": None} for name in DEFAULT_SECTIONS}
    details: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    current_section = "front_matter"
    current_top: dict[str, object] | None = None

    for fallback_page, page in enumerate(pages, start=1):
        page_number = _page_number(page, fallback_page)
        lines = _page_text(page).splitlines()
        skip_next = False
        for index, raw_line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            line = raw_line.strip()
            detected = _detect_heading(
                line,
                index == 0 or not lines[index - 1].strip(),
                index == len(lines) - 1 or not lines[index + 1].strip(),
            )
            if (
                not detected
                and index + 1 < len(lines)
                and line
                and lines[index + 1].strip()
            ):
                combined = f"{line} {lines[index + 1].strip()}"
                detected = _detect_heading(
                    combined, index == 0 or not lines[index - 1].strip(), True
                )
                if detected:
                    detected["matched_rule"] = f"multiline_{detected['matched_rule']}"
                    skip_next = True
            if detected:
                number = detected.get("number")
                if (
                    isinstance(number, str)
                    and "." in number
                    and current_top is not None
                ):
                    detected["canonical_section"] = current_top["canonical_section"]
                    detected.update(
                        subsection_title=detected["raw_heading"],
                        subsection_number=number,
                        section_title=current_top["raw_heading"],
                        section_number=current_top.get("number"),
                    )
                else:
                    current_top = dict(detected)
                    detected.update(
                        section_title=detected["raw_heading"], section_number=number
                    )
                current_section = str(detected["canonical_section"])
                sections.setdefault(current_section, "")
                spans.setdefault(
                    current_section, {"start_page": None, "end_page": None}
                )
                if spans[current_section]["start_page"] is None:
                    spans[current_section]["start_page"] = page_number
                spans[current_section]["end_page"] = page_number
                diagnostics.append(
                    {
                        key: value
                        for key, value in {
                            "raw_heading": detected["raw_heading"],
                            "canonical_section": current_section,
                            "confidence": detected["confidence"],
                            "matched_rule": detected["matched_rule"],
                            "page_number": page_number,
                        }.items()
                    }
                )
                details.append({**detected, "page_number": page_number})
                continue
            sections[current_section] += raw_line.rstrip() + "\n"
            if line:
                if spans[current_section]["start_page"] is None:
                    spans[current_section]["start_page"] = page_number
                spans[current_section]["end_page"] = page_number

    return SectionedDoc(
        paper_id,
        {k: _join_section_lines(v) for k, v in sections.items()},
        spans,
        details,
        diagnostics,
    )


def _detect_heading(
    line: str, previous_blank: bool, next_blank: bool
) -> dict[str, object] | None:
    raw = re.sub(r"\s+", " ", line.strip())
    if not raw or len(raw) > 100 or len(raw.split()) > 12:
        return None
    if (
        _CAPTION_RE.match(raw)
        or _BULLET_RE.match(raw)
        or raw.startswith(("http://", "https://"))
    ):
        return None
    if raw[-1:] in ".,;!?":
        return None
    number = None
    title = raw.strip(" .:;–—-")
    numbered = _NUMBERED_RE.match(title)
    if numbered:
        number, title = (
            numbered.group("number"),
            numbered.group("title").strip(" .:;–—-"),
        )
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    canonical = SECTION_ALIASES.get(normalized)
    if canonical:
        signals = (
            1 + bool(number) + bool(raw.isupper()) + bool(previous_blank or next_blank)
        )
        rule = (
            "numbered_known_alias"
            if number
            else "uppercase_known_alias"
            if raw.isupper()
            else "known_alias"
        )
        return {
            "raw_heading": raw,
            "canonical_section": canonical,
            "number": number,
            "confidence": min(0.99, 0.68 + 0.09 * signals),
            "matched_rule": rule,
        }
    if (
        number
        and "." in number
        and (title.isupper() or title.istitle())
        and len(title.split()) <= 8
    ):
        return {
            "raw_heading": raw,
            "canonical_section": "other",
            "number": number,
            "confidence": 0.72 if previous_blank or next_blank else 0.64,
            "matched_rule": "numbered_subsection",
        }
    return None


def _join_section_lines(text: str) -> str:
    text = re.sub(r"(?<![.?!:;])\n(?!\s*\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _page_text(page: PageLike | Mapping[str, object]) -> str:
    return (
        str(page.get("text", ""))
        if isinstance(page, Mapping)
        else str(getattr(page, "text", ""))
    )


def _page_number(page: PageLike | Mapping[str, object], fallback: int) -> int:
    value = (
        page.get("page_number", fallback)
        if isinstance(page, Mapping)
        else getattr(page, "page_number", fallback)
    )
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
