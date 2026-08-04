from __future__ import annotations

import time

from generation.citation_handler import validate_citations
from generation.context_assembler import AssembledContext, CitationSource
from generation.response_formatter import format_response


def test_format_response_matches_chat_contract_and_filters_sources():
    context = AssembledContext(
        "context",
        {
            1: CitationSource(1, "p1", "c1", "One", ["A"], 2020, "method", None),
            2: CitationSource(2, "p2", "c2", "Two", ["B"], 2021, "results", None),
        },
    )
    answer = "Supported [2], unknown [9]."
    validation = validate_citations(answer, context.citation_map)
    generated = format_response(answer, context, validation, time.monotonic() - 0.01)

    payload = generated.to_dict()
    assert payload["answer"] == answer
    assert [source["paper_id"] for source in payload["sources"]] == ["p2"]
    assert payload["citations_valid"] is False
    assert payload["unknown_citations"] == [9]
    assert payload["latency_ms"] >= 0
