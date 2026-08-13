"""Build an evidence-aligned, reviewable generation evaluation set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from processing.bm25_indexer import BM25Indexer


DEFAULT_PAPER_IDS = (
    "2310.01217v3",
    "2403.09635v2",
    "2405.15793v3",
    "2402.05435v2",
    "2310.02124v3",
    "2310.03084v2",
    "2405.10938v3",
    "2402.14845v1",
    "2303.15265v1",
    "2308.08774v1",
)

_SECTION_PRIORITY = (
    "abstract",
    "methodology",
    "experiments",
    "results",
    "limitations",
    "conclusion",
    "discussion",
    "introduction",
    "analysis",
)
_EXCLUDED_SECTIONS = frozenset(
    {"references", "bibliography", "front_matter", "acknowledgements", "acknowledgments"}
)


def build_payload(
    chunks: Iterable[dict[str, Any]],
    *,
    paper_ids: Iterable[str] = DEFAULT_PAPER_IDS,
    contexts_per_question: int = 4,
) -> dict[str, Any]:
    """Create two grounded questions per paper with paper-consistent contexts."""

    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_paper[str(chunk.get("paper_id", ""))].append(chunk)

    records: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        paper_chunks = by_paper.get(paper_id, [])
        if not paper_chunks:
            raise ValueError(f"Paper {paper_id!r} is absent from the BM25 index")
        metadata = paper_chunks[0].get("metadata") or {}
        title = str(metadata.get("title", "")).strip()
        if not title:
            raise ValueError(f"Paper {paper_id!r} has no title metadata")
        context_ids = [
            str(chunk["chunk_id"])
            for chunk in select_context_chunks(paper_chunks, limit=contexts_per_question)
        ]
        if len(context_ids) < contexts_per_question:
            raise ValueError(f"Paper {paper_id!r} has insufficient usable sections")

        records.extend(
            [
                _record(
                    len(records) + 1,
                    paper_id,
                    title,
                    (
                        f"Using the frozen passages from '{title}', summarize up to three "
                        "central contributions. Use an evidence table with columns problem, "
                        "method, evaluation, findings, and limitations. Explicitly report "
                        "when a requested detail is absent from the passages. Keep each cell "
                        "concise and include at least one numbered citation in every data row."
                    ),
                    context_ids,
                    ["problem", "method", "evaluation", "findings", "limitations"],
                ),
                _record(
                    len(records) + 2,
                    paper_id,
                    title,
                    (
                        f"Using only the frozen passages from '{title}', identify up to three "
                        "empirically evaluated claims. Use an evidence table with columns "
                        "claim, dataset, baseline, result, and limitations. Distinguish "
                        "reported experimental evidence from author assertions. Keep each cell "
                        "concise and include at least one numbered citation in every data row."
                    ),
                    context_ids,
                    ["claim", "dataset", "baseline", "result", "limitations"],
                ),
            ]
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Evidence-aligned candidate set bootstrapped from papers present in the BM25 index.",
            "Every frozen chunk is constrained to the record's declared paper_id.",
            "Records remain reviewed=false until a human verifies questions, passages, and labels.",
            "Expected qualifying items and calibration verdicts intentionally remain empty until review.",
        ],
        "questions": records,
    }


def select_context_chunks(
    chunks: Iterable[dict[str, Any]], *, limit: int = 4
) -> list[dict[str, Any]]:
    """Select diverse evidence sections while excluding front/reference material."""

    usable = [
        chunk
        for chunk in chunks
        if str(chunk.get("section", "")).casefold() not in _EXCLUDED_SECTIONS
        and str(chunk.get("text", "")).strip()
    ]
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in usable:
        by_section[str(chunk.get("section", "unknown")).casefold()].append(chunk)

    selected: list[dict[str, Any]] = []
    for section in _SECTION_PRIORITY:
        candidates = by_section.get(section, [])
        if not candidates:
            continue
        if section == "abstract":
            chosen = min(candidates, key=lambda item: int(item.get("start_char", 0)))
        else:
            chosen = max(candidates, key=lambda item: len(str(item.get("text", ""))))
        selected.append(chosen)
        if len(selected) == limit:
            return selected

    used = {str(chunk.get("chunk_id")) for chunk in selected}
    for chunk in sorted(usable, key=lambda item: int(item.get("start_char", 0))):
        if str(chunk.get("chunk_id")) in used:
            continue
        selected.append(chunk)
        if len(selected) == limit:
            break
    return selected


def audit_payload(payload: dict[str, Any], chunks: Iterable[dict[str, Any]]) -> list[str]:
    """Return release-blocking structural/alignment failures."""

    lookup = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    failures: list[str] = []
    records = payload.get("questions", [])
    if len(records) != 20:
        failures.append(f"expected 20 questions, found {len(records)}")
    for record in records:
        identifier = str(record.get("id", "<missing>"))
        paper_id = str(record.get("paper_id", ""))
        chunk_ids = record.get("retrieved_chunk_ids", [])
        if not paper_id:
            failures.append(f"{identifier}: missing paper_id provenance")
        if not chunk_ids:
            failures.append(f"{identifier}: no frozen chunks")
        for chunk_id in chunk_ids:
            chunk = lookup.get(str(chunk_id))
            if chunk is None:
                failures.append(f"{identifier}: missing chunk {chunk_id}")
            elif str(chunk.get("paper_id")) != paper_id:
                failures.append(
                    f"{identifier}: chunk {chunk_id} belongs to {chunk.get('paper_id')}, not {paper_id}"
                )
        if record.get("reviewed") is not False:
            failures.append(f"{identifier}: bootstrap records must remain unreviewed")
        if not record.get("required_fields"):
            failures.append(f"{identifier}: no required output fields")
    return failures


def _record(
    number: int,
    paper_id: str,
    title: str,
    question: str,
    context_ids: list[str],
    required_fields: list[str],
) -> dict[str, Any]:
    return {
        "id": f"gen-{number:03d}",
        "paper_id": paper_id,
        "title": title,
        "question_source": "evidence_aligned_bootstrap",
        "question": question,
        "retrieved_chunk_ids": list(context_ids),
        "expected_qualifying_items": [],
        "required_concepts": [],
        "excluded_items": {},
        "required_fields": required_fields,
        "max_items": 3,
        "reviewed": False,
        "calibration_verdicts": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/data/golden_generation_qa.json")
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    chunks = BM25Indexer.load(args.index).chunks
    if args.audit_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = build_payload(chunks)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    failures = audit_payload(payload, chunks)
    if failures:
        print("\n".join(f"ERROR: {failure}" for failure in failures))
        return 1
    print(f"Validated {len(payload['questions'])} evidence-aligned questions in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
