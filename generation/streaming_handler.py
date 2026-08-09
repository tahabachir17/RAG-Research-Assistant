"""Provider-neutral token and SSE-ready generation streams."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

try:
    from .citation_handler import build_source_list
    from .context_assembler import CitationSource
    from .llm_client import LLMClient, LLMStreamChunk
    from .response_validator import ResponseValidator
except ImportError:
    from citation_handler import build_source_list
    from context_assembler import CitationSource
    from llm_client import LLMClient, LLMStreamChunk
    from response_validator import ResponseValidator


async def stream_answer(
    llm: LLMClient,
    system: str,
    user: str,
    citation_map: dict[int, CitationSource],
) -> AsyncGenerator[str, None]:
    """Yield text immediately and validate text plus finish reason at EOF."""

    parts: list[str] = []
    finish_reason: str | None = None
    async for chunk in _completion_stream(llm, system, user):
        if chunk.text:
            parts.append(chunk.text)
            yield chunk.text
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
    ResponseValidator(citation_map).validate("".join(parts), finish_reason=finish_reason)


async def stream_answer_events(
    llm: LLMClient,
    system: str,
    user: str,
    citation_map: dict[int, CitationSource],
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield token events and a validation-aware final completion event."""

    started_at = time.monotonic()
    parts: list[str] = []
    finish_reason: str | None = None
    async for chunk in _completion_stream(llm, system, user):
        if chunk.text:
            parts.append(chunk.text)
            yield {"type": "token", "text": chunk.text}
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
    answer = "".join(parts)
    validation = ResponseValidator(citation_map).validate(answer, finish_reason=finish_reason)
    yield {
        "type": "done",
        "sources": build_source_list(answer, citation_map),
        "latency_ms": max(0, round((time.monotonic() - started_at) * 1000)),
        "citations_valid": not any(failure in {"missing_citation", "citation_out_of_range", "unsupported_citation_format"} for failure in validation.failures),
        "unknown_citations": [int(value) for value in __import__("re").findall(r"\[(\d+)\]", answer) if int(value) not in citation_map],
        "finish_reason": finish_reason,
        "validation_failures": validation.failures,
    }


async def _completion_stream(llm: LLMClient, system: str, user: str) -> AsyncIterator[LLMStreamChunk]:
    result = await llm.acomplete(system, user, stream=True)
    if isinstance(result, str):
        if result:
            yield LLMStreamChunk(result)
        return
    async for value in result:
        if isinstance(value, str):
            yield LLMStreamChunk(value)
        elif isinstance(value, LLMStreamChunk):
            yield value
        else:
            raise TypeError("LLM stream yielded an unsupported chunk")