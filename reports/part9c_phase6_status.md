# Part 9c — Phase 6 Evaluation Status

Date: 2026-08-13

## Outcome

The reviewed benchmark selection was validated at exactly 75 questions: 55 QASA
and 20 QASPER. The populated `bench_external_chunks` collection passed preflight
with 2,777 points. The full runner instantiated hybrid dense+sparse RRF,
cross-encoder reranking, and the MMR wrapper; at the generator's top-k 4 boundary
the evidence-backed MMR guard correctly bypassed diversity reordering.

The external evaluation could not produce defensible RAGAS scores in this
environment. All 75 retrieval calls completed, but the first Groq generation
requests failed with connection errors after bounded retries. An unsandboxed
one-question probe was then denied because it would transmit repository-derived
questions and passages to the third-party Groq service without explicit user
authorization. The run was stopped rather than record 75 identical failures or
present diagnostic/partial scores as final results.

## Preserved artifacts

| File | State |
| --- | --- |
| `evaluation/data/eval_results/full_ragas_eval_part9c_75_reviewed_20260813/generation_checkpoint.json` | 75-question order, 0 generated, 2 connection-error records; resumable with the same run ID. |
| `evaluation/data/eval_results/part9c_20260813/reranker_profile.json` | Complete 75-question candidate-k 10/20 latency and Recall@4 profile. |
| `reports/part9c_latency_report.md` | Full latency breakdown and flagged mitigation recommendation. |

## RAGAS target table

| Tier | Faithfulness > .85 | Answer relevancy > .80 | Context precision > .75 | Context recall > .70 | Answer correctness > .75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| QASA (55) | blocked | blocked | blocked | blocked | blocked |
| QASPER (20) | blocked | blocked | blocked | blocked | blocked |
| Overall (75) | blocked | blocked | blocked | blocked | blocked |

No metric is classified as below target because no fresh judge score completed.
The blocker is external generation/judge connectivity and data-transfer approval,
not retrieval, generation quality, or judge schema behavior.

## Authorized resume

After explicit approval to transmit the reviewed benchmark questions and
retrieved passages to Groq, rerun the existing command with run ID
`part9c_75_reviewed_20260813`. Generation and per-question RAGAS metric caches
will resume. Do not change the question selection or retrieval configuration for
that run ID.
