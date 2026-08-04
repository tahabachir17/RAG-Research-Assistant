from __future__ import annotations

import pytest

from generation.context_assembler import CitationSource
from generation.llm_client import LLMClientError
from generation.streaming_handler import stream_answer, stream_answer_events


def _source():
    return CitationSource(
        1, "paper", "chunk", "Title", ["Author"], 2024, "results", None
    )


class _StreamingLLM:
    def __init__(self, parts, error=None):
        self.parts, self.error = parts, error

    async def acomplete(self, system, user, *, stream=False):
        assert stream is True

        async def generate():
            for part in self.parts:
                yield part
            if self.error:
                raise self.error

        return generate()


@pytest.mark.asyncio
async def test_stream_answer_delivers_incremental_text():
    stream = stream_answer(
        _StreamingLLM(["claim ", "[1]"]), "system", "user", {1: _source()}
    )
    assert [part async for part in stream] == ["claim ", "[1]"]


@pytest.mark.asyncio
async def test_stream_answer_events_finishes_with_sources_and_validation():
    events = [
        event
        async for event in stream_answer_events(
            _StreamingLLM(["claim ", "[1]"]), "system", "user", {1: _source()}
        )
    ]
    assert events[:2] == [
        {"type": "token", "text": "claim "},
        {"type": "token", "text": "[1]"},
    ]
    assert events[-1]["type"] == "done"
    assert events[-1]["citations_valid"] is True
    assert events[-1]["sources"][0]["paper_id"] == "paper"


@pytest.mark.asyncio
async def test_stream_interruption_propagates_without_false_done_event():
    error = LLMClientError("fake", "connection lost")
    stream = stream_answer_events(
        _StreamingLLM(["partial"], error), "system", "user", {1: _source()}
    )
    first = await anext(stream)
    assert first == {"type": "token", "text": "partial"}
    with pytest.raises(LLMClientError, match="connection lost"):
        await anext(stream)
