"""Controlled A/B/C evaluation for multi-concept evidence packing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

try:
    from generation.context_assembler import AssembledContext, ContextAssembler
    from retrieval.models import RetrievalResult
except ImportError:
    from generation.context_assembler import AssembledContext, ContextAssembler
    from retrieval.models import RetrievalResult


PACKING_MODES = ("gold", "adjacent", "section")


@dataclass(slots=True)
class PackingComparisonRow:
    mode: str
    answer_correctness: float
    faithfulness: float
    chunk_ids: list[str]


def run_packing_comparison(
    gold_chunks: Sequence[RetrievalResult],
    required_concepts: Sequence[Any],
    *,
    score_context: Callable[[str, AssembledContext], dict[str, float]],
    adjacent_chunk_lookup: Callable[[RetrievalResult], Sequence[RetrievalResult]],
    section_chunk_lookup: Callable[[RetrievalResult], Sequence[RetrievalResult]],
    max_context_tokens: int = 4000,
) -> dict[str, Any]:
    """Run gold, adjacent, and section modes with the same scoring callback."""

    rows: list[PackingComparisonRow] = []
    for mode in PACKING_MODES:
        assembled = ContextAssembler(
            max_context_tokens=max_context_tokens,
            evidence_packing_mode=mode,
            adjacent_chunk_lookup=adjacent_chunk_lookup,
            section_chunk_lookup=section_chunk_lookup,
        ).assemble(gold_chunks, required_concepts=required_concepts)
        scores = score_context(mode, assembled)
        rows.append(
            PackingComparisonRow(
                mode=mode,
                answer_correctness=float(scores["answer_correctness"]),
                faithfulness=float(scores["faithfulness"]),
                chunk_ids=[
                    source.chunk_id for source in assembled.citation_map.values()
                ],
            )
        )
    return {"packing_modes": [asdict(row) for row in rows]}
