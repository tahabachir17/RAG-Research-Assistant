# Part 9m — Reduced Retrieval-Only Re-test

Date: 2026-08-15

## Technical summary

Part B completed **30/30 retrieval-only queries** with no LLM calls. The
selection uses all 20 vague/casual phrasings because they are the most
failure-sensitive tier, plus the topic-named phrasing for the first 10 triplets
in the unchanged Part 9l file order. This keeps the full vague signal and adds
a deterministic named-language comparison within the 30-query cap.

All 90 overlapping per-question/config score rows for BM25, hybrid RRF, and
hybrid+rerank match the saved Part 9l results exactly. The full 20-query vague
aggregates also reproduce Part 9l exactly, so the re-test passes its overlap
sanity check.

## Results by phrasing tier

| Config | Phrasing tier | N | Recall@4 | Recall@8 | Hit@4 | Hit@8 | MRR | nDCG@4 | Hit@4 pass/fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | Vague/casual | 20 | 0.2000 | 0.2250 | 0.3500 | 0.4000 | 0.2625 | 0.1984 | 7 / 13 |
| BM25 | Topic-named | 10 | 0.2000 | 0.3833 | 0.4000 | 0.6000 | 0.3500 | 0.2235 | 4 / 6 |
| Dense | Vague/casual | 20 | 0.1250 | 0.1667 | 0.1500 | 0.2500 | 0.0500 | 0.0663 | 3 / 17 |
| Dense | Topic-named | 10 | 0.4167 | 0.4833 | 0.6000 | 0.6000 | 0.4083 | 0.3478 | 6 / 4 |
| Hybrid RRF | Vague/casual | 20 | 0.1833 | 0.3083 | 0.3000 | 0.5000 | 0.1792 | 0.1605 | 6 / 14 |
| Hybrid RRF | Topic-named | 10 | 0.2667 | 0.5833 | 0.4000 | 0.8000 | 0.3333 | 0.2581 | 4 / 6 |
| Hybrid + rerank | Vague/casual | 20 | 0.1417 | 0.2500 | 0.2000 | 0.4000 | 0.1083 | 0.1061 | **4 / 16** |
| Hybrid + rerank | Topic-named | 10 | 0.4500 | 0.5000 | 0.7000 | 0.7000 | 0.4750 | 0.3903 | **7 / 3** |

The dense-only results expose the sharpest wording effect: Hit@4 rises from
0.15 on vague phrasing to 0.60 on the selected topic-named slice. Hybrid RRF
raises vague Hit@4 to 0.30, but reranking reduces it to 0.20. Conversely, the
reranker improves the selected topic-named slice from 0.40 to 0.70 Hit@4.
These are descriptive results for unequal tier sizes (20 versus 10), not a
paired significance test over all 20 evidence needs.

## Scope, components, and metric definitions

The run uses the same retrieval components and unchanged labels as Part 9l:

- BM25 with `technical_terms_v2` preprocessing;
- dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`;
- reciprocal-rank fusion to 20 candidates;
- `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking over those 20 candidates.

Each backend retrieves 50 candidates before fusion. Recall@4/8 measures the
share of reviewed relevant chunks retrieved by the cutoff. Hit@4/8 is binary
per question. MRR is evaluated at the production top-four boundary, and
nDCG@4 uses the unchanged binary chunk relevance labels.

## Robustness and Part D decision signal

The exact Part 9l reproduction is the main robustness check: 90 overlapping
rows were compared at all six metrics with zero mismatches. For the 20 vague
queries under hybrid+rerank, 16 miss at top four. Their mutually exclusive
stage split also reproduces Part 9l:

| Miss stage | Misses | Share of 16 misses |
| --- | ---: | ---: |
| Candidate generation: no reviewed gold in fused top 20 | 7 | 43.75% |
| Reranker demotion: reviewed gold fused, then placed outside top 4 | 9 | **56.25%** |

This reduced re-test therefore preserves the prior decision signal: reranker
demotion remains the slightly larger first target for Part D, while candidate
generation is a substantial secondary failure mode. The dense tier adds useful
detail—vague wording itself creates a severe semantic-retrieval gap—but it does
not overturn the observed final-boundary diagnosis on this controlled set.

## Limitations and next step

The topic-named tier contains only the first 10 Part 9l triplets, all from QASA,
whereas the vague tier contains 10 QASA and 10 QASPER items. Its absolute
metrics should therefore be read as a deterministic diagnostic slice, not as a
balanced overall tier estimate. No prompt, expansion, reranker cutoff, or
retriever configuration was changed.

Part D should begin with a measurement-preserving reranker/final-boundary
experiment and retain candidate-generation diagnostics as a guardrail. The
blocked Part A live benchmark must be resumed separately under explicit user
direction after UTF-8 prompt capture is enabled.

## Audit artifacts

Machine-readable artifacts are under
`evaluation/data/eval_results/part9m_reduced_retrieval_retest_20260815/`:

- `retrieval_retest.json` — selection rule, model/config metadata, all rankings, per-question metrics, and aggregates;
- `per_question_scores.csv` — 120 query/config score rows.

The additive runner is
`evaluation/run_part9m_reduced_retrieval_retest.py`. Existing Part 9l reports,
golden files, aliases, prompts, matchers, and dispositions were not modified.
