# Part 9i full 12-question single-pass re-verification

## Scope and run integrity

Part A accepts NSM-02 and Video-02 as documented generation limitations rather
than blocking defects. NSM-02's supplied context is Explicit, as established in
`part9h_nsm02_context_check.md`, but two targeted regenerations omitted the
task-reward objective. `part9h_video02_candidate_search.md` found no second
confirming causal example among 20 candidates across QASA, QASPER, and the
controlled set. No third NSM mechanism change or Video causal-fragment change
was made.

Part B ran exactly one checkpointed generation-only pass over all 12 reviewed
questions. The immutable generation artifact is
`evaluation/data/eval_results/controlled_generation_part9i_single_pass_20260814/report.json`.
It records:

- Groq `llama-3.3-70b-versatile` as the generator;
- 12 requested and 12 generated answers, with no generation errors;
- frozen reviewed contexts, with adjacent packing for mechanism questions;
- direct context precision and recall of `1.0`;
- `judge_status: disabled` for every question.

No RAGAS or other LLM-judge call was made, no judge cache was written, and the
three-run protocol was not started. The live client encountered only Groq 429
token-rate limits; all were handled within the existing two-retry bound.

## Final alias-aware comparison

The baseline below is the Part 9g answer set rescored with the current final
Part 9h concept definitions and aliases. This is the apples-to-apples Part
9g/9h baseline: `0.8889`. The originally published Part 9g mean was `0.8472`;
the difference is the later Part 9h Enquirer-01 golden rescope and alias
calibration, not a different generation sample.

A change greater than `0.25` is flagged as material. The threshold is used as a
review flag, not as proof from one stochastic sample.

| Question | Prior concept_recall | New concept_recall | Material change? | Notes |
|---|---:|---:|---|---|
| controlled-submodular-01 | 1.000 | 1.000 | No | Stable complete answer. |
| controlled-submodular-02 | 1.000 | 1.000 | No | Stable complete answer. |
| controlled-enquirer-01 | 1.000 | 1.000 | No | Resolved Part 9h golden rescope remains complete. |
| controlled-enquirer-02 | 0.750 | 0.750 | No | Known variability remains: end-to-end training is present, while layered intermediate annotations are absent in this sample. |
| controlled-qrnn-01 | 0.500 | 1.000 | **Yes (+0.500)** | The answer now includes both the speedup and its train/test scope, consistent with the retained direct-fact qualifier fix. |
| controlled-qrnn-02 | 1.000 | 1.000 | No | Stable complete mechanism answer. |
| controlled-nsm-01 | 1.000 | 1.000 | No | Stable complete answer. |
| controlled-nsm-02 | 0.750 | 0.750 | No | REINFORCE is named, but its task-reward objective is again omitted; accepted limitation status is unchanged. |
| controlled-video-01 | 1.000 | 1.000 | No | Stable complete answer. |
| controlled-video-02 | 0.667 | 0.667 | No | Temporal structure and appearance order are present; incoherent fusion remains omitted, consistent with its accepted limitation. |
| controlled-reddit-01 | 1.000 | 1.000 | No | Stable complete answer. |
| controlled-reddit-02 | 1.000 | 1.000 | No | Stable complete answer. |

Mean final alias-aware concept recall increased from `0.8889` to `0.9306`
(`+0.0417`). Eleven questions stayed within the band; QRNN-01 was the sole
material mover. Its improvement is directionally consistent with the scoped
Part 9h fix, but remains one generated sample.

## Alias addition and manual evidence check

One genuine new paraphrase was confirmed and added during deterministic
rescoring:

| Question / concept | Added alias | Before | After | Evidence justification |
|---|---|---:|---:|---|
| controlled-enquirer-02 / `end-to-end training from query-answer pairs` | `trained using query-answer pairs in an end-to-end fashion` | 0.500 | 0.750 | The generated answer uses this wording, and supplied adjacent chunk `4125ea01-f459-54de-b9b4-d4d68ea70dcb` explicitly states that the model can be trained using Query-Answer pairs with the execution logic optimized in an end-to-end fashion. |

This is the concept's third alias, below the five-alias cap. It does not match
or credit the absent layered-memory concept. No other alias was added. The raw
generation artifact's stored aggregate (`0.9097`) is the pre-addition score;
the final report value (`0.9306`) is the deterministic post-addition rescore,
following the same calibration convention as Part 9g.

Manual inspection of all final unmatched concepts found exactly three genuine
omissions, with no remaining matcher disagreement:

- Enquirer-02: intermediate table annotations stored in layered memory;
- NSM-02: REINFORCE directly optimizing task reward;
- Video-02: incoherent fusion of temporally distinct events and objects.

## Previously identified omissions

- **Enquirer-01:** resolved; the correctly scoped concept scores `1.0`.
- **QRNN-01:** resolved; the speedup's train/test qualifier is present and the
  score is `1.0`.
- **Enquirer-02:** resolved with known variability; the prior targeted run
  demonstrated the layered-state mechanism, while this sample again omits it
  and scores `0.75`.
- **NSM-02:** accepted limitation; this sample is consistent with the prior two
  targeted omissions and does not alter its disposition.
- **Video-02:** accepted limitation; this sample is consistent with the prior
  omission and does not alter its disposition.

A one-off favorable NSM-02 or Video-02 sample would not have overturned either
accepted-limitation decision; neither happened here.

## Recommendation

**GO for the separate three-run controlled-generation protocol.** This pass
reconfirms the resolved items, preserves the known Enquirer-02 variability, and
contains no new unexplained omission outside the two limitations explicitly
accepted in Part A. The recommendation is based on this completed pass and its
manual concept trace, not on an expectation that NSM-02 or Video-02 will improve.

This recommendation does not execute or otherwise begin the three-run protocol.
