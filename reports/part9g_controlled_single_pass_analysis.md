# Part 9g controlled-generation single-pass analysis

## Scope and run status

Exactly one generation-only pass was run over all 12 reviewed questions in
`controlled_generation_qa.json` using Groq
`llama-3.3-70b-versatile`. Retrieval remained bypassed in favor of the frozen
gold chunks; mechanism questions used the current adjacent-evidence route. The
pass made no RAGAS or other LLM-judge calls and did not start the three-run
protocol.

All 12 answers completed. The client encountered transient Groq token-rate
limits, waited, and completed each request within its bounded retry policy; the
checkpoint has no generation errors.

The immutable generation artifact is
`evaluation/data/eval_results/controlled_generation_part9f_single_pass_20260814/report.json`.
Its stored `concept_recall` values are the pre-calibration values captured at
generation time. The `new concept_recall` values below are deterministic
rescoring results after the hand-authored aliases from this audit were added.
For an apples-to-apples comparison, `old concept_recall` is the previous live
answer rescored with that same final alias set. `old answer_correctness` is
copied unchanged from
`controlled_generation_ragas_completeness_20260814`.

## Per-question results

| Question | Old concept_recall | New concept_recall | Old answer_correctness | Material answer change? | Qualitative read vs score |
|---|---:|---:|---:|---|---|
| controlled-submodular-01 | 1.000 | 1.000 | 0.581 | No | Agree: both concepts present |
| controlled-submodular-02 | 1.000 | 1.000 | 0.234 | No | Agree: all four concepts present |
| controlled-enquirer-01 | 0.500 | 0.500 | 0.551 | No | Agree: table values as answers remains absent |
| controlled-enquirer-02 | 0.250 | 0.750 | 0.556 | Yes | Agree: layered intermediate annotations remain absent |
| controlled-qrnn-01 | 0.500 | 0.500 | 0.569 | No | Agree: train/test scope remains absent |
| controlled-qrnn-02 | 0.000 | 1.000 | 0.405 | Yes | Agree: all three mechanisms are now explicit |
| controlled-nsm-01 | 1.000 | 1.000 | 0.862 | No | Agree: all three concepts present |
| controlled-nsm-02 | 0.750 | 0.750 | 0.748 | No | Agree: optimizing task reward remains absent |
| controlled-video-01 | 0.750 | 1.000 | 0.950 | Yes | Agree: relevant-segment selection was added |
| controlled-video-02 | 0.667 | 0.667 | 0.550 | No | Agree: incoherent fusion of distinct events/objects remains absent |
| controlled-reddit-01 | 1.000 | 1.000 | 0.580 | No | Agree: all three concepts present |
| controlled-reddit-02 | 1.000 | 1.000 | 0.930 | No | Agree: all four concepts present |

The final alias-aware mean concept recall is `0.8472` for the new pass versus
`0.7014` for the previous live answers. No question has a disagreement between
the qualitative manual read and its final deterministic score.

## Manual alias audit

Eighteen aliases were added across nine questions. Each is a normalized
substring of the new answer and represents a concrete paraphrase of one
required concept. Examples include:

- `extractive multi-document summarization` for `extractive summarization`;
- `improves on manually tuned summarization models`, `handle large numbers of
  features`, and `control of overfitting` for the corresponding Submodular-02
  concepts;
- the query representation carrying semantic information and being sent to
  executors for `operations conditioned on the query`, and `end-to-end training
  using query-answer pairs` for the corresponding Enquirer-02 concepts;
- the answer's exact sequence-to-sequence/key-variable-memory and symbolic
  computer/Lisp-interpreter phrasings for NSM-01;
- `iterative ML training process` and `bootstrap training` for NSM-02;
- `3-D CNN representation` and `local temporal structure` for Video-01;
- `ignores the temporal structure` and `order of appearances of objects` for
  Video-02;
- hyphenation, singular/plural, and result phrasing for Reddit-01; and
- the answer's exact oracle, timing/reaction, and combinatorial-complexity
  phrasings for Reddit-02.

No alias was added for any genuinely absent concept. Every concept still has at
most five aliases, and matching remains case-folded, punctuation-normalized
substring matching only: no fuzzy distance, embeddings, semantic model, or LLM
matching was introduced.

## Over-crediting spot-check

The Part 9f known-incomplete answers were rescored with the full final alias
set, with concept-level traces inspected:

- the old vague QRNN-02 answer remains `0.0`; none of its three generic claims
  match a technical mechanism;
- the old partial Enquirer-02 answer scores `0.25`; only its genuinely present
  end-to-end/query-answer-pair concept is credited, while conditioned operations
  and layered intermediate annotations remain uncredited;
- the old partial Video-02 answer scores `0.667`; its explicit temporal-structure
  and appearance-order claims are credited, while incoherent fusion of distinct
  events and objects remains uncredited.

Across the new pass, all five manually identified absent concepts also remain
unmatched. The aliases widened phrasing coverage but did not credit any absent
idea.

## Evidence check for the remaining omissions

All five remaining concepts are extractable from the context actually supplied
for their questions:

- Enquirer-01 says the answer is “one of the values in the tables.”
- Enquirer-02 says intermediate annotations are stored “in the memory of each
  layer.”
- QRNN-01 says “up to 16 times faster at train and test time.”
- NSM-02 says REINFORCE is applied “to directly optimize the task reward.”
- Video-02 says temporally distinct events and objects may be “fused
  incoherently.”

These are generation omissions, not routing, evidence-packing, or matching
failures.

## Go/no-go

**NO-GO for the three-run live protocol.** The scorer is now calibrated and its
manual cross-check is clean, but five required concepts that are present in the
delivered evidence were omitted by generation across several question types.
That is a broader prompt-adherence issue requiring a separately scoped targeted
fix before spending the three runs. This pass made no prompt, evidence-packing,
routing, retrieval, ingestion, judge, API, frontend, or CI changes.
