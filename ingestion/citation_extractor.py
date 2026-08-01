from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class Citation:
    citation_id: str
    raw_text: str
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    venue: str | None = None
    reference_number: int | None = None
    parse_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.authors is None:
            self.authors = []

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DOI_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I
)
ARXIV_RE = re.compile(
    r"(?:arXiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.I
)
URL_RE = re.compile(r"https?://[^\s<>]+", re.I)
YEAR_RE = re.compile(r"(?<!\d)(?:\(|\b)(19\d{2}|20\d{2})(?:\)|\b)")
MARKER_RE = re.compile(r"^\s*(?:\[(?P<bracket>\d+)\]|(?P<plain>\d{1,3})[.)])\s+")
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:references|bibliography|works cited|literature cited)\s*$", re.I
)


def extract_citations(sectioned_doc_or_text: Any) -> list[Citation]:
    text = _reference_text(sectioned_doc_or_text)
    if not text.strip():
        return []
    citations = [_parse_reference(entry) for entry in _split_references(text)]
    return _deduplicate(citations)


def citations_to_dicts(citations: Iterable[Citation]) -> list[dict[str, Any]]:
    return [citation.to_dict() for citation in citations]


def _reference_text(value: Any) -> str:
    if isinstance(value, str):
        return _slice_reference_section(value)
    sections = getattr(value, "sections", None)
    if isinstance(sections, Mapping):
        return _reference_from_sections(sections)
    if isinstance(value, Mapping):
        nested = value.get("sections")
        if isinstance(nested, Mapping):
            return _reference_from_sections(nested)
        if "references" in value:
            return str(value.get("references") or "")
        pages = value.get("pages") or (value.get("raw_document") or {}).get("pages", [])
        if isinstance(pages, list):
            return _slice_reference_section(
                "\n".join(_page_text(page) for page in pages)
            )
    pages = getattr(value, "pages", None)
    if pages is not None:
        return _slice_reference_section("\n".join(_page_text(page) for page in pages))
    return ""


def _reference_from_sections(sections: Mapping[str, Any]) -> str:
    for key, text in sections.items():
        if re.sub(r"[_-]+", " ", str(key).lower()).strip() in {
            "references",
            "bibliography",
            "works cited",
            "literature cited",
        }:
            return str(text or "")
    return ""


def _slice_reference_section(text: str) -> str:
    lines = text.splitlines()
    start = next(
        (i + 1 for i, line in enumerate(lines) if REFERENCE_HEADING_RE.match(line)),
        None,
    )
    if start is None:
        return ""  # Raw papers are not scanned indiscriminately without a heading.
    return "\n".join(lines[start:])


def _split_references(text: str) -> list[str]:
    lines = _clean_reference_lines(text)
    if not lines:
        return []
    combined = " ".join(lines)
    inline_marker = re.compile(r"(?<!\S)(?:\[\d+\]|\d{1,3}[.)])\s+")
    markers = list(inline_marker.finditer(combined))
    if markers:
        entries = []
        for marker, following in zip(markers, markers[1:] + [None]):
            entries.append(
                combined[
                    marker.start() : following.start() if following else len(combined)
                ]
            )
        return [entry.strip() for entry in entries if _substantive(entry)]

    # Unnumbered bibliographies usually begin with an author surname; continuation
    # lines are joined unless a new author-like line follows a completed entry.
    entries: list[str] = []
    current: list[str] = []
    author_start = re.compile(r"^[A-Z][A-Za-z'’\-]+,\s+(?:[A-Z](?:\.|\b)|[A-Z][a-z]+)")
    for line in lines:
        if current and author_start.match(line) and _looks_complete(" ".join(current)):
            entries.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(" ".join(current))
    return [entry for entry in entries if _substantive(entry)]


def _clean_reference_lines(text: str) -> list[str]:
    lines: list[str] = []
    seen_short: set[str] = set()
    for raw in text.replace("\x00", " ").splitlines():
        line = " ".join(raw.split())
        if not line or REFERENCE_HEADING_RE.match(line) or re.fullmatch(r"\d+", line):
            continue
        normalized = line.casefold()
        # Repeated short page headers/footers are ignored after their first occurrence.
        if len(line) < 80 and normalized in seen_short:
            continue
        if len(line) < 80:
            seen_short.add(normalized)
        lines.append(line)
    return lines


def _parse_reference(raw_text: str) -> Citation:
    raw = " ".join(raw_text.split()).strip()
    marker = MARKER_RE.match(raw)
    reference_number = (
        int(marker.group("bracket") or marker.group("plain")) if marker else None
    )
    body = MARKER_RE.sub("", raw, count=1)
    doi_match, arxiv_match, year_match, url_match = (
        DOI_RE.search(body),
        ARXIV_RE.search(body),
        YEAR_RE.search(body),
        URL_RE.search(body),
    )
    doi = doi_match.group(1).rstrip(".,;)") if doi_match else None
    arxiv_id = arxiv_match.group(1) if arxiv_match else None
    url = url_match.group(0).rstrip(".,;)") if url_match else None
    title = _guess_title(body)
    authors = _guess_authors(body, title)
    venue = _guess_venue(body, title, year_match.group(1) if year_match else None)
    signals = sum(
        value is not None and value != []
        for value in (title, authors, year_match, doi, arxiv_id, url)
    )
    confidence = min(0.95, 0.25 + 0.1 * signals + (0.1 if marker else 0.0))
    identity = doi or arxiv_id or _normalize(title or raw)
    citation_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return Citation(
        citation_id,
        raw,
        title,
        authors,
        int(year_match.group(1)) if year_match else None,
        doi,
        arxiv_id,
        url,
        venue,
        reference_number,
        confidence,
    )


def _guess_title(body: str) -> str | None:
    quoted = re.search(r"[\"“](.+?)[\"”]", body)
    if quoted and 2 <= len(quoted.group(1).split()) <= 40:
        return quoted.group(1).strip()
    parts = [part.strip() for part in re.split(r"\.\s+", body) if part.strip()]
    for candidate in parts[1:3]:
        candidate = YEAR_RE.sub("", candidate).strip(" ,()")
        if 3 <= len(candidate.split()) <= 40 and not candidate.lower().startswith(
            ("doi", "http", "arxiv")
        ):
            return candidate
    return None


def _guess_authors(body: str, title: str | None) -> list[str]:
    prefix = (
        body.split(title, 1)[0]
        if title and title in body
        else re.split(r"\(?(?:19|20)\d{2}\)?", body, maxsplit=1)[0]
    )
    prefix = prefix.strip(" .,;:")
    if not prefix or len(prefix.split()) > 35:
        return []
    values = re.split(r"\s+(?:and|&)\s+|\s*;\s*", prefix)
    return [value.strip(" ,.") for value in values if 1 <= len(value.split()) <= 10][
        :20
    ]


def _guess_venue(body: str, title: str | None, year: str | None) -> str | None:
    if not title or title not in body:
        return None
    tail = body.split(title, 1)[1]
    if year:
        tail = tail.replace(year, "", 1)
    candidate = re.split(r"(?:doi\s*:|https?://|arXiv\s*:)", tail, flags=re.I)[0].strip(
        " .,:;()"
    )
    return candidate if 1 <= len(candidate.split()) <= 20 else None


def _deduplicate(citations: list[Citation]) -> list[Citation]:
    result: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        if citation.doi:
            key = ("doi", citation.doi.casefold())
        elif citation.arxiv_id:
            key = ("arxiv", re.sub(r"v\d+$", "", citation.arxiv_id.casefold()))
        elif citation.title and len(_normalize(citation.title)) >= 12:
            key = ("title", _normalize(citation.title))
        else:
            key = ("raw", _normalize(citation.raw_text))
        if key not in seen:
            seen.add(key)
            result.append(citation)
    return result


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _looks_complete(value: str) -> bool:
    return bool(
        YEAR_RE.search(value)
        and (
            value.rstrip().endswith(".") or DOI_RE.search(value) or URL_RE.search(value)
        )
    )


def _substantive(value: str) -> bool:
    return len(MARKER_RE.sub("", value).split()) >= 3


def _page_text(page: Any) -> str:
    return (
        str(page.get("text", ""))
        if isinstance(page, Mapping)
        else str(getattr(page, "text", ""))
    )
