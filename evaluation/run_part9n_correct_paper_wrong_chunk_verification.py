"""Verify whether correct-paper/wrong-chunk top-four results answer the query.

This is an additive, read-only diagnostic over saved Part 9m rankings and the
unchanged external benchmark index.  Manual labels are fixed in this file so
the generated JSON and Markdown report remain reproducible.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from generation.citation_handler import lexical_overlap_score
from processing.bm25_indexer import BM25Indexer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/data/eval_results/part9m_reduced_retrieval_retest_20260815/retrieval_retest.json"
INDEX = ROOT / "evaluation/data/external_benchmarks/external_bm25_index.pkl"
OUT = ROOT / "evaluation/data/eval_results/part9n_verification_20260815"
ARTIFACT = OUT / "correct_paper_wrong_chunk_verification.json"
REPORT = ROOT / "reports/part9n_correct_paper_wrong_chunk_verification.md"
THRESHOLD = 0.55


JUDGMENTS: dict[str, tuple[str, str]] = {
    "qasa-1911.03814-7::vague": (
        "answers it",
        "A retrieved methodology passage states that no test mention-entity pairs were observed in training and identifies evaluation on held-out test domains; that is enough to explain the zero-shot designation.",
    ),
    "qasa-1703.10593-13::vague": (
        "answers it",
        "The retrieved introduction and abstract directly contrast aligned pairs with separate unpaired domain collections and explain adversarial mapping plus cycle consistency.",
    ),
    "qasa-1703.06870-15::vague": (
        "same topic, wrong specific evidence",
        "The abstract says Mask R-CNN generalizes to pose/keypoint detection, but it omits the requested adaptation mechanism: one mask per keypoint type with one-hot targets and a spatial softmax loss.",
    ),
    "qasa-1503.04069-9::vague": (
        "answers it",
        "The retrieved front matter independently states that every LSTM variant was optimized separately for each task using random search, which answers how the comparison was kept fair.",
    ),
    "qasa-1610.06475-8::vague": (
        "same topic, wrong specific evidence",
        "The introduction says place recognition detects returns and closes loops, but it does not give the actual loop-detection and geometric-validation procedure requested.",
    ),
    "qasa-1511.07247-12::vague": (
        "answers it",
        "Retrieved passages describe NetVLAD as a differentiable generalized VLAD pooling layer whose parameters are learned by backpropagation inside an end-to-end CNN.",
    ),
    "qasper-bc473c5bd0e1a8be9b2037aa7006fd68217c3f47::vague": (
        "answers it",
        "Retrieved passages independently support human evaluation with expert raters, document-level context, ranking/significance analysis, and attention to references—enough to answer how a parity claim should be tested.",
    ),
    "qasper-fb5ce11bfd74e9d7c322444b006a27f2ff32a0cf::vague": (
        "same topic, wrong specific evidence",
        "The retrieved results passage describes the color/shape task and success criterion but omits the requested success rates (97.6%, 96.0%, and 79.0% for shape alone).",
    ),
    "qasper-b0799e26152197aeb3aa3b11687a6cc9f6c31011::vague": (
        "same topic, wrong specific evidence",
        "The passages establish multimodal hate detection and its challenges but do not describe the requested FCM, SCM, and TKM ways of combining visual and textual representations.",
    ),
    "qasper-73738e42d488b32c9db89ac8adefc75403fa2653::vague": (
        "same topic, wrong specific evidence",
        "The abstract and conclusion say adaptation improves over a baseline but contain none of the EM/F1 values needed to answer how much it improved.",
    ),
    "qasper-234ccc1afcae4890e618ff2a7b06fc1e513ea640::vague": (
        "same topic, wrong specific evidence",
        "The introduction and conclusion say augmentation improves robustness, but omit the requested clean/adversarial accuracy values and improvement magnitude.",
    ),
    "qasa-1511.04587-13::topic_named": (
        "same topic, wrong specific evidence",
        "The retrieved passages discuss residual learning, high learning rates, and gradient clipping but never state the Euclidean/mean-squared residual reconstruction loss asked for.",
    ),
    "qasa-1503.04069-9::topic_named": (
        "answers it",
        "A retrieved hyperparameter-search passage states that 27 separate random searches covered every nine-variant/three-dataset combination, with 200 trials each and independently sampled settings.",
    ),
    "qasa-1706.02413-8::topic_named": (
        "same topic, wrong specific evidence",
        "The retrieved passages discuss MSG/MRG accuracy and general computation, but not the reason MRG is cheaper: it avoids large-neighborhood feature extraction at the lowest levels.",
    ),
}


def _percentage(count: int, total: int) -> float:
    return count / total if total else 0.0


def _bucket_counts(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = (
        "answers it",
        "same topic, wrong specific evidence",
        "unrelated section",
    )
    output: dict[str, dict[str, Any]] = {}
    for tier in ("vague", "topic_named", "overall"):
        rows = cases if tier == "overall" else [row for row in cases if row["tier"] == tier]
        counts = Counter(str(row["manual_label"]) for row in rows)
        output[tier] = {
            "cases": len(rows),
            "buckets": {
                bucket: {
                    "count": counts.get(bucket, 0),
                    "fraction": _percentage(counts.get(bucket, 0), len(rows)),
                }
                for bucket in buckets
            },
        }
    return output


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    index = BM25Indexer.load(INDEX)
    chunk_by_id = {str(row["chunk_id"]): row for row in index.chunks}
    cases: list[dict[str, Any]] = []

    for spec in payload["query_manifest"]:
        query_id = str(spec["query_id"])
        top_four = payload["retrieval_traces"][query_id]["hybrid_rerank"][:4]
        gold_ids = [str(value) for value in spec["relevant_chunk_ids"]]
        gold_paper_ids = {str(chunk_by_id[value].get("paper_id")) for value in gold_ids}
        retrieved_same_paper = [
            chunk_id
            for chunk_id in top_four
            if str(chunk_by_id[chunk_id].get("paper_id")) in gold_paper_ids
        ]
        if not retrieved_same_paper or any(chunk_id in set(gold_ids) for chunk_id in top_four):
            continue

        combined_retrieved_text = " ".join(
            str(chunk_by_id[chunk_id].get("text", ""))
            for chunk_id in retrieved_same_paper
        )
        per_gold_overlap = {
            gold_id: lexical_overlap_score(
                str(chunk_by_id[gold_id].get("text", "")),
                combined_retrieved_text,
            )
            for gold_id in gold_ids
        }
        score = max(per_gold_overlap.values(), default=0.0)
        manual_label, rationale = JUDGMENTS[query_id]
        cases.append(
            {
                "query_id": query_id,
                "base_id": spec["base_id"],
                "tier": spec["phrasing_tier"],
                "question": spec["question"],
                "gold_chunk_ids": gold_ids,
                "gold_chunks": [
                    {
                        "chunk_id": gold_id,
                        "paper_id": chunk_by_id[gold_id].get("paper_id"),
                        "section": chunk_by_id[gold_id].get("section"),
                    }
                    for gold_id in gold_ids
                ],
                "retrieved_same_paper_top4": [
                    {
                        "rank": top_four.index(chunk_id) + 1,
                        "chunk_id": chunk_id,
                        "paper_id": chunk_by_id[chunk_id].get("paper_id"),
                        "section": chunk_by_id[chunk_id].get("section"),
                    }
                    for chunk_id in retrieved_same_paper
                ],
                "lexical_overlap_score": score,
                "lexical_overlap_threshold": THRESHOLD,
                "lexical_overlap_pass": score >= THRESHOLD,
                "per_gold_chunk_overlap": per_gold_overlap,
                "manual_label": manual_label,
                "manual_rationale": rationale,
                "usable_answer": manual_label == "answers it",
            }
        )

    found = Counter(str(row["tier"]) for row in cases)
    expected = {"vague": 11, "topic_named": 3, "total": 14}
    observed = {
        "vague": found.get("vague", 0),
        "topic_named": found.get("topic_named", 0),
        "total": len(cases),
    }
    count_matches = observed == expected
    if set(JUDGMENTS) != {str(row["query_id"]) for row in cases}:
        raise AssertionError("manual judgment IDs do not match reproduced cases")

    bucket_counts = _bucket_counts(cases)
    usable = sum(bool(row["usable_answer"]) for row in cases)
    overlap_passes = sum(bool(row["lexical_overlap_pass"]) for row in cases)
    agreement = sum(
        bool(row["usable_answer"]) == bool(row["lexical_overlap_pass"])
        for row in cases
    )
    result = {
        "schema_version": 1,
        "created_at": date.today().isoformat(),
        "source_artifact": str(SOURCE.relative_to(ROOT)),
        "source_index": str(INDEX.relative_to(ROOT)),
        "selection_rule": "correct paper appears in hybrid+rerank top 4 and no reviewed gold chunk appears in top 4",
        "expected_case_counts": expected,
        "observed_case_counts": observed,
        "count_matches_expectation": count_matches,
        "overlap_method": "For each reviewed gold chunk, generation.citation_handler.lexical_overlap_score(gold_text, concatenated_same_paper_retrieved_top4_text); case score is the maximum over gold chunks.",
        "overlap_threshold": THRESHOLD,
        "manual_unit": "A case is answers it when at least one retrieved same-paper top-four passage independently contains enough evidence to answer the literal question.",
        "presentation_notes": {
            "audience": "technical",
            "delivery_surface": "user-requested Markdown report",
            "chart_omitted_reason": "The sample has only 14 auditable cases and the decision depends on exact per-case labels, ranks, sections, and overlap values; a chart would hide rather than clarify the evidence.",
        },
        "summary": {
            "bucket_counts": bucket_counts,
            "usable_answers": usable,
            "usable_fraction": _percentage(usable, len(cases)),
            "overlap_threshold_passes": overlap_passes,
            "overlap_human_binary_agreements": agreement,
            "overlap_human_binary_agreement_fraction": _percentage(agreement, len(cases)),
            "overlap_false_negatives": sum(
                bool(row["usable_answer"]) and not bool(row["lexical_overlap_pass"])
                for row in cases
            ),
            "overlap_false_positives": sum(
                not bool(row["usable_answer"]) and bool(row["lexical_overlap_pass"])
                for row in cases
            ),
        },
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    lines = [
        "# Part 9n — Correct-Paper, Wrong-Chunk Verification",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Technical summary",
        "",
        f"The diagnostic reproduced **{observed['total']} cases: {observed['vague']} vague and {observed['topic_named']} topic-named**. "
        + ("This exactly matches the expected 11 + 3 split." if count_matches else "This differs from the expected 11 + 3 split and is flagged as a count mismatch."),
        "",
        f"Only **{usable}/{len(cases)} ({pct(_percentage(usable, len(cases)))})** of the correct-paper/wrong-reviewed-chunk cases contain a retrieved same-paper passage that would independently let a generator answer the literal question. The remaining **{len(cases) - usable}/{len(cases)} ({pct(_percentage(len(cases) - usable, len(cases)))})** stay on the same topic but omit the requested mechanism, comparison, or number. No case was wholly unrelated to the question.",
        "",
        f"The 0.55 lexical-overlap check passes **{overlap_passes}/{len(cases)}** cases. It therefore misses all {usable} human-usable alternate passages and agrees with the binary human judgment only on the {len(cases) - usable} non-answering cases ({pct(_percentage(agreement, len(cases)))} overall agreement).",
        "",
        "## Most same-paper alternates do not answer the question",
        "",
        "| Tier | Cases | Answers it | Same topic, wrong evidence | Unrelated section |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for tier, label in (("vague", "Vague/casual"), ("topic_named", "Topic-named"), ("overall", "Overall")):
        summary = bucket_counts[tier]
        values = summary["buckets"]
        lines.append(
            f"| {label} | {summary['cases']} | {values['answers it']['count']} ({pct(values['answers it']['fraction'])}) | {values['same topic, wrong specific evidence']['count']} ({pct(values['same topic, wrong specific evidence']['fraction'])}) | {values['unrelated section']['count']} ({pct(values['unrelated section']['fraction'])}) |"
        )
    lines.extend(
        [
            "",
            "The correct-paper framing was therefore optimistic for a majority of these misses. It identifies useful paper selection, but not sufficient passage selection.",
            "",
            "## Per-case verification",
            "",
            "The overlap score is directional: reviewed-gold content tokens covered by the concatenated same-paper passages retrieved in the top four. `Pass` means score ≥ 0.55.",
            "",
            "| Tier | Question ID | Same-paper ranks | Sections | Overlap | Pass | Manual label |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in cases:
        ranks = ", ".join(str(item["rank"]) for item in row["retrieved_same_paper_top4"])
        sections = ", ".join(str(item["section"]) for item in row["retrieved_same_paper_top4"])
        lines.append(
            f"| {row['tier']} | `{row['query_id']}` | {ranks} | {sections} | {row['lexical_overlap_score']:.3f} | {'Yes' if row['lexical_overlap_pass'] else 'No'} | **{row['manual_label']}** |"
        )
    lines.extend(["", "## Manual evidence notes", ""])
    for row in cases:
        lines.extend(
            [
                f"### `{row['query_id']}` — {row['manual_label']}",
                "",
                f"**Question:** {row['question']}",
                "",
                row["manual_rationale"],
                "",
            ]
        )
    lines.extend(
        [
            "## Scope and methodology",
            "",
            "The source is the saved Part 9m retrieval trace; retrieval was not rerun. For every query, the diagnostic selected hybrid+rerank top-four results from the reviewed source paper only when none of the exact reviewed chunk IDs appeared in the top four. It loaded both retrieved and reviewed-gold text from the unchanged external benchmark BM25 index.",
            "",
            "The automated check imports `generation.citation_handler.lexical_overlap_score`, the same content-token overlap function used by `evaluation/audit_part9d_qasper_coverage.py`. For multiple gold chunks, it mirrors that audit's max-over-gold approach against combined candidate evidence.",
            "",
            "Manual labels were assigned by reading the literal question, every same-paper passage actually retrieved in the top four, and the reviewed gold passages. `Answers it` requires at least one retrieved passage to contain sufficient evidence on its own; merely naming the method, task, or improvement direction does not qualify when the question asks for a mechanism or number.",
            "",
            "## Limitations and robustness",
            "",
            "The manual classification is a single-reviewer judgment, not a new golden label or disposition. It is used only for this diagnostic. The lexical check measures token coverage between passages, not semantic entailment, and its complete lack of threshold passes shows that 0.55 is unsuitable as a standalone detector of alternate answer-bearing chunks in this sample.",
            "",
            "## Part D implication",
            "",
            f"The exact-chunk metric does understate quality in **{usable}/{len(cases)}** same-paper misses, but a majority (**{len(cases) - usable}/{len(cases)}**) would still not produce a usable answer. Part D should therefore keep the reranker/final-boundary work **and** explicitly measure within-paper section/passage ranking. Correct-paper Hit@4 should not replace exact-chunk or answerability review as the success criterion.",
            "",
            "A useful next diagnostic is to score section-aware reranking on these eight non-answering cases while separately retaining the six answer-bearing alternates as label-completeness cases. No existing golden, alias, prompt, matcher, retriever configuration, or disposition was changed.",
            "",
            "## Further question",
            "",
            "Should the six verified alternate answer passages be proposed for a separate human-review queue as possible additional acceptable evidence labels? This report does not add them to any golden set.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(ARTIFACT), "report": str(REPORT), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
