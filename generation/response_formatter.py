"""Format generated text into the API-facing response contract."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

try:
    from .citation_handler import (
        CitationValidationResult,
        build_source_list,
    )
    from .context_assembler import AssembledContext
except ImportError:
    from citation_handler import CitationValidationResult, build_source_list
    from context_assembler import AssembledContext


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: int
    citations_valid: bool
    unknown_citations: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_response(
    answer: str,
    assembled_context: AssembledContext,
    validation: CitationValidationResult,
    started_at: float,
) -> GeneratedAnswer:
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    if not isinstance(assembled_context, AssembledContext):
        raise TypeError("assembled_context must be AssembledContext")
    if not isinstance(validation, CitationValidationResult):
        raise TypeError("validation must be CitationValidationResult")
    if not isinstance(started_at, (int, float)) or isinstance(started_at, bool):
        raise TypeError("started_at must be a monotonic timestamp")
    latency_ms = max(0, round((time.monotonic() - float(started_at)) * 1000))
    return GeneratedAnswer(
        answer=answer,
        sources=build_source_list(answer, assembled_context.citation_map),
        latency_ms=latency_ms,
        citations_valid=validation.valid,
        unknown_citations=list(validation.unknown_numbers),
    )
