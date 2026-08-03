"""Review top-20 retrieved chunks and update golden chunk labels interactively.

First run the retrieval evaluator; its stable cache contains the hybrid top-20
candidate pool. Then run this module and mark each candidate relevant (y), not
relevant (n), skip (s), or quit (q). Progress is saved after every question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_candidate_dump(cache_path: Path, output_path: Path) -> list[dict[str, Any]]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    hybrid = cache["rankings"]["hybrid_rrf"]
    dump = []
    for run in hybrid:
        dump.append(
            {
                "query_id": run["query_id"],
                "candidates": [
                    {
                        "rank": rank,
                        "chunk_id": result["chunk_id"],
                        "paper_id": result.get("paper_id"),
                        "section": result.get("section"),
                        "score": result["score"],
                        "text": result["text"],
                    }
                    for rank, result in enumerate(run["results"][:20], 1)
                ],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"questions": dump}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dump


def review(golden_path: Path, candidates_path: Path) -> None:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    questions = golden["questions"] if isinstance(golden, dict) else golden
    by_id = {item["query_id"]: item for item in questions}
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))["questions"]
    for group in candidates:
        item = by_id[group["query_id"]]
        print(f"\n[{item['query_id']}] {item['question']}")
        accepted = set(item.get("relevant_chunk_ids", []))
        completed = True
        for candidate in group["candidates"]:
            print(
                f"\n#{candidate['rank']} {candidate['chunk_id']} | paper={candidate['paper_id']} | {candidate['section']}"
            )
            print(" ".join(candidate["text"].split())[:700])
            answer = input("Relevant? [y]es/[n]o/[s]kip/[q]uit: ").strip().casefold()
            if answer == "q":
                _save(golden_path, golden)
                return
            if answer == "y":
                accepted.add(candidate["chunk_id"])
            elif answer == "s":
                completed = False
        item["relevant_chunk_ids"] = sorted(accepted)
        item["annotation_status"] = (
            "chunk_reviewed" if completed else "chunk_review_partial"
        )
        _save(golden_path, golden)


def _save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden", type=Path, default=Path("evaluation/data/golden_retrieval.json")
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("evaluation/data/cache/retrieval_candidates.json"),
    )
    parser.add_argument(
        "--candidates", type=Path, default=Path("evaluation/data/label_candidates.json")
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    build_candidate_dump(args.cache, args.candidates)
    print(f"Candidate dump: {args.candidates}")
    if not args.prepare_only:
        review(args.golden, args.candidates)


if __name__ == "__main__":
    main()
