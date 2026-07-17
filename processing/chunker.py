"""Section-aware chunking for parsed research papers.

Offsets on :class:`Chunk` are relative to the original section text.  This makes
the stored text auditable with ``section_text[start_char:end_char]``.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence


class Tokenizer(Protocol):
    def spans(self, text: str) -> list[tuple[int, int]]: ...


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _RegexTokenizer:
    """Dependency-free token offsets, close to common subword token counts."""

    _token = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in self._token.finditer(text)]


class _TiktokenTokenizer:
    def __init__(self, encoding_name: str) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)

    def spans(self, text: str) -> list[tuple[int, int]]:
        token_ids = self._encoding.encode(text, disallowed_special=())
        if not token_ids:
            return []
        _, starts = self._encoding.decode_with_offsets(token_ids)
        return [
            (start, starts[index + 1] if index + 1 < len(starts) else len(text))
            for index, start in enumerate(starts)
        ]


class SectionAwareChunker:
    """Split each paper section into bounded, overlapping retrieval chunks.

    Windows prefer paragraph and sentence endings near the token limit.  Empty
    sections are ignored.  Short sections remain independently labelled because
    merging an abstract or conclusion into a neighbour weakens retrieval filters.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 80,
        min_tokens: int = 40,
        *,
        encoding_name: str = "cl100k_base",
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("overlap_tokens must be between 0 and max_tokens - 1")
        if min_tokens < 1:
            raise ValueError("min_tokens must be positive")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = _TiktokenTokenizer(encoding_name)
            except (ImportError, KeyError):
                self.tokenizer = _RegexTokenizer()

    def chunk(self, document: object) -> list[Chunk]:
        paper_id = str(_field(document, "paper_id", "unknown"))
        metadata = _metadata(document)
        sections = _sections(document)
        if not any(str(text).strip() for text in sections.values()):
            fallback = _fallback_text(document)
            sections = {"unknown": fallback} if fallback.strip() else {}

        chunks: list[Chunk] = []
        seen_chunk_ids: set[str] = set()
        for section, value in sections.items():
            text = str(value or "")
            section_name = str(section or "unknown").strip() or "unknown"
            for start, end in self._windows(text):
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"rag-paper:{paper_id}:{section_name}:{start}:{end}:{text[start:end]}",
                    )
                )
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        section=section_name,
                        text=text[start:end],
                        start_char=start,
                        end_char=end,
                        metadata=dict(metadata),
                    )
                )
        return chunks

    def _windows(self, text: str) -> list[tuple[int, int]]:
        spans = self.tokenizer.spans(text)
        if not spans:
            return []
        boundaries = _natural_boundaries(text)
        result: list[tuple[int, int]] = []
        start_token = 0
        while start_token < len(spans):
            limit_token = min(start_token + self.max_tokens, len(spans))
            end_char = spans[limit_token - 1][1]
            if limit_token < len(spans):
                minimum_tokens = min(self.min_tokens, limit_token - start_token)
                minimum = spans[start_token + minimum_tokens - 1][1]
                candidates = [
                    point for point in boundaries if minimum <= point <= end_char
                ]
                if candidates:
                    end_char = candidates[-1]
                    while (
                        limit_token > start_token
                        and spans[limit_token - 1][0] >= end_char
                    ):
                        limit_token -= 1

            start_char, end_char = _trim_span(text, spans[start_token][0], end_char)
            if start_char < end_char:
                result.append((start_char, end_char))
            if limit_token >= len(spans):
                break
            next_token = max(start_token + 1, limit_token - self.overlap_tokens)
            start_token = next_token
        return result


def chunk_document(
    document: object,
    max_tokens: int = 512,
    overlap_tokens: int = 80,
    min_tokens: int = 40,
) -> list[Chunk]:
    """Convenience wrapper for :class:`SectionAwareChunker`."""

    return SectionAwareChunker(max_tokens, overlap_tokens, min_tokens).chunk(document)


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(document: object) -> Mapping[str, Any]:
    value = _field(document, "metadata", {})
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def _sections(document: object) -> dict[str, Any]:
    value = _field(document, "sections", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _fallback_text(document: object) -> str:
    direct = _field(document, "text", "")
    if direct:
        return str(direct)
    raw = _field(document, "raw_document", None) or document
    pages: Sequence[Any] = _field(raw, "pages", []) or []
    return "\n\n".join(str(_field(page, "text", "")) for page in pages).strip()


def _natural_boundaries(text: str) -> list[int]:
    # Paragraphs have priority because they occur later at an equal search range.
    return sorted(
        {
            match.end()
            for match in re.finditer(r"(?:[.!?](?:[\"')\]]*)\s+|\n\s*\n)", text)
        }
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
