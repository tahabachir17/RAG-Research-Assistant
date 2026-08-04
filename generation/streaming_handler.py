"""Provider-neutral token and SSE-ready generation streams."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

try:
    from .citation_handler import build_source_list, validate_citations
    from .context_assembler import CitationSource
    from .llm_client import LLMClient
except ImportError:
    from citation_handler import build_source_list, validate_citations
    from context_assembler import CitationSource
    from llm_client import LLMClient


async def stream_answer(
    llm: LLMClient,
    system: str,
    user: str,
    citation_map: dict[int, CitationSource],
) -> AsyncGenerator[str, None]:
    """Yield text immediately while retaining enough state to validate at EOF."""

    parts: list[str] = []
    async for text in _text_stream(llm, system, user):
        parts.append(text)
        yield text
    validate_citations("".join(parts), citation_map)


async def stream_answer_events(
    llm: LLMClient,
    system: str,
    user: str,
    citation_map: dict[int, CitationSource],
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield token events and one final citation-aware completion event."""

    started_at = time.monotonic()
    parts: list[str] = []
    async for text in _text_stream(llm, system, user):
        parts.append(text)
        yield {"type": "token", "text": text}
    answer = "".join(parts)
    validation = validate_citations(answer, citation_map)
    yield {
        "type": "done",
        "sources": build_source_list(answer, citation_map),
        "latency_ms": max(0, round((time.monotonic() - started_at) * 1000)),
        "citations_valid": validation.valid,
        "unknown_citations": validation.unknown_numbers,
    }


async def _text_stream(llm: LLMClient, system: str, user: str) -> AsyncIterator[str]:
    result = await llm.acomplete(system, user, stream=True)
    if isinstance(result, str):
        if result:
            yield result
        return
    async for delta in result:
        if not isinstance(delta, str):
            raise TypeError("LLM stream yielded a non-string delta")
        if delta:
            yield delta
