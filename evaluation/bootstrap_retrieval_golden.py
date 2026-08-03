"""Bootstrap a reviewable retrieval golden set when golden_qa.json is absent.

The first ten questions come from the existing retrieval-evaluation notebook.
Additional records are deliberately marked as title-derived and needing human
review; they provide paper labels but never pretend to be chunk judgments.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

CURATED = [
    (
        "traffic_1",
        "How can traffic video data be used for freeway ramp metering?",
        "2012.12104v1",
    ),
    (
        "traffic_2",
        "What advantages do traffic cameras provide over point detectors for ramp control?",
        "2012.12104v1",
    ),
    (
        "grid_1",
        "How should deep reinforcement learning algorithms be designed for power grid voltage control?",
        "2012.13026v1",
    ),
    (
        "grid_2",
        "Which imitation learning approach maps power grid operating points directly to control actions?",
        "2012.13026v1",
    ),
    (
        "biometric_1",
        "Why do fuzzy commitments provide insufficient protection for deep facial biometric templates?",
        "2012.13293v1",
    ),
    (
        "biometric_2",
        "How can a reconstruction attack recover a face from a protected biometric template?",
        "2012.13293v1",
    ),
    (
        "portfolio_1",
        "How does portfolio-based algorithm selection generalize to unseen problem instances?",
        "2012.13315v1",
    ),
    (
        "portfolio_2",
        "How are algorithm portfolios and selectors trained from typical problem instances?",
        "2012.13315v1",
    ),
    (
        "dialogue_1",
        "What is the DECODE task for contradiction detection in dialogue?",
        "2012.13391v2",
    ),
    (
        "dialogue_2",
        "Does structured utterance modeling improve contradiction detection in conversations?",
        "2012.13391v2",
    ),
]


def bootstrap(registry: Path, output: Path, count: int = 50) -> None:
    records = [
        {
            "query_id": query_id,
            "question": question,
            "relevant_chunk_ids": [],
            "relevant_paper_ids": [paper_id],
            "annotation_status": "needs_chunk_review",
            "question_source": "curated_retrieval_notebook",
        }
        for query_id, question, paper_id in CURATED
    ]
    used = {paper_id for _, _, paper_id in CURATED}
    with sqlite3.connect(registry) as connection:
        rows = connection.execute(
            "SELECT paper_id, metadata_json FROM papers WHERE status = 'indexed' ORDER BY canonical_id"
        ).fetchall()
    for paper_id, metadata_json in rows:
        if len(records) >= count:
            break
        if paper_id in used:
            continue
        metadata = json.loads(metadata_json or "{}")
        title = str(metadata.get("title", "")).strip()
        if not title:
            continue
        records.append(
            {
                "query_id": f"bootstrap_{len(records) + 1:03d}",
                "question": f"What are the main findings of the paper titled '{title}'?",
                "relevant_chunk_ids": [],
                "relevant_paper_ids": [paper_id],
                "annotation_status": "needs_question_and_chunk_review",
                "question_source": "registry_title_bootstrap",
            }
        )
        used.add(paper_id)
    if len(records) < count:
        raise RuntimeError(
            f"only {len(records)} indexed questions could be bootstrapped"
        )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Paper-only labels are scored as a fallback until chunk review is complete.",
            "registry_title_bootstrap questions leak titles and are diagnostics, not an unbiased benchmark.",
            "Run evaluation.label_retrieval after an evaluation to confirm top-20 chunks.",
        ],
        "questions": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=Path("data/metadata/corpus_registry.sqlite3")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/data/golden_retrieval.json")
    )
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    bootstrap(args.registry, args.output, args.count)
    print(f"Wrote {args.count} reviewable questions to {args.output}")


if __name__ == "__main__":
    main()
