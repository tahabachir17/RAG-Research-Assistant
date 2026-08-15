# Part 9l — Mixed-Style Retrieval Re-test and Vague-Query Root Cause

Date: 2026-08-15

## Scope and integrity checks

This additive run re-executed retrieval for all 75 unchanged, human-reviewed
QASA/QASPER questions and added 20 paired evidence needs. Each paired need has
three phrasings: the unchanged original, a new vague/casual phrasing, and a new
method-or-topic-named phrasing that does not name the paper title.

The 20 selected base IDs, original question strings, and reviewed chunk-ID
lists were checked against the assembled 75-question benchmark before scoring:
all 20 matched exactly. The subset contains 10 QASA and 10 QASPER questions.
No existing golden file, label, alias, prompt, or disposition was changed.

Retrieval used the existing Part 9e components: BM25 with
`technical_terms_v2`, dense retrieval with
`sentence-transformers/all-MiniLM-L6-v2`, RRF fusion to 20 candidates, and
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranking. Metrics use the unchanged
reviewed chunk labels. MRR is computed at the production top-four boundary.

## Full 75-question retrieval re-test

| Config | N | Recall@4 | Recall@8 | Hit@4 | Hit@8 | MRR | nDCG@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 75 | 0.3952 | 0.5063 | 0.4933 | 0.6133 | 0.3911 | 0.3583 |
| Hybrid RRF | 75 | 0.3756 | 0.4952 | 0.4667 | 0.6133 | 0.3111 | 0.2986 |
| Hybrid + rerank | 75 | 0.2486 | 0.3419 | 0.3600 | 0.4800 | 0.2400 | 0.2136 |

The recomputed values reproduce the Part 9e results. At the production
hybrid+rerank top-four boundary, **27/75 pass and 48/75 fail**. BM25 remains the
best of these three tested configurations on every reported full-set metric.

## Paired 20-question phrasing results

| Config | Phrasing tier | N | Recall@4 | Recall@8 | Hit@4 | Hit@8 | MRR | nDCG@4 | Hit@4 pass/fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | Original | 20 | 0.3583 | 0.5000 | 0.5500 | 0.7000 | 0.4500 | 0.3456 | 11 / 9 |
| BM25 | Vague/casual | 20 | 0.2000 | 0.2250 | 0.3500 | 0.4000 | 0.2625 | 0.1984 | 7 / 13 |
| BM25 | Topic-named | 20 | 0.1500 | 0.2833 | 0.3000 | 0.4500 | 0.2167 | 0.1464 | 6 / 14 |
| Hybrid RRF | Original | 20 | 0.2750 | 0.4667 | 0.4000 | 0.6500 | 0.2542 | 0.2204 | 8 / 12 |
| Hybrid RRF | Vague/casual | 20 | 0.1833 | 0.3083 | 0.3000 | 0.5000 | 0.1792 | 0.1605 | 6 / 14 |
| Hybrid RRF | Topic-named | 20 | 0.2000 | 0.4333 | 0.3500 | 0.6000 | 0.2583 | 0.1872 | 7 / 13 |
| Hybrid + rerank | Original | 20 | 0.3167 | 0.4333 | 0.5000 | 0.6000 | 0.3292 | 0.2710 | 10 / 10 |
| Hybrid + rerank | Vague/casual | 20 | 0.1417 | 0.2500 | 0.2000 | 0.4000 | 0.1083 | 0.1061 | **4 / 16** |
| Hybrid + rerank | Topic-named | 20 | 0.2750 | 0.3917 | 0.4500 | 0.5500 | 0.3000 | 0.2390 | 9 / 11 |

The production configuration exposes the wording sensitivity directly. Naming
the method or topic raises Hit@4 from 0.20 on vague questions to 0.45, but it
still does not recover the subset's 0.50 original-tier result. The named tier is
not universally easier: BM25 alone performs worse on these particular named
paraphrases than on the vague versions. That counterexample is why the report
keeps each configuration and phrasing tier separate.

## Vague-tier miss-stage isolation

Under hybrid+rerank, the vague tier has **4/20 top-four passes and 16/20
misses**. Applying the Part 9e rule gives:

| Miss stage | Misses | Share of 16 misses |
| --- | ---: | ---: |
| Candidate generation: no gold in fused top 20 | 7 | 43.75% |
| Reranker demotion: gold in fused top 20 but outside final top 4 | 9 | **56.25%** |

Reranker demotion is the largest mutually exclusive stage failure. Four of its
nine misses land at final ranks 5–8 and are recovered at k=8; the other five
remain below rank 8.

## Candidate-generation diagnosis

The following table covers all seven vague candidate-generation misses. A
missing backend rank means the reviewed gold chunk is outside that backend's
top 50. Cosine is the maximum actual query-to-reviewed-gold cosine across the
question's labels, not a rank proxy.

| Evidence need | BM25 rank | Dense rank | Positive-IDF shared tokens | Max gold cosine |
| --- | ---: | ---: | --- | ---: |
| VDSR reconstruction loss | >50 | 34 | `and, between, error, image, one, the, they` | 0.3396 |
| Wikia zero-shot split | >50 | >50 | `as, on, zero-shot` | 0.1857 |
| ORB-SLAM2 loop closure | >50 | 28 | `a, and, it, system, that, to` | 0.3824 |
| Human/MT parity recommendations | >50 | 15 | `as, be, evaluation, machine, the, translation` | 0.6568 |
| Multimodal hate detection models | >50 | 16 | `and, be, can, text, to` | 0.5729 |
| Semi-supervised NLG/NLU attention | >50 | >50 | `the` | 0.2065 |
| Syntax-and-lexicon LM datasets | 48 | >50 | `language, model, test, the, to, used, were, which` | 0.3498 |

### Does BM25 have any lexical overlap?

Yes. **All 7/7 candidate-generation misses share at least one raw token and at
least one positive-IDF token with a reviewed gold chunk.** Therefore the narrow
hypothesis that vague queries fail because `technical_terms_v2` produces zero
lexical overlap is denied.

The broader lexical-gap hypothesis is still supported: BM25 places no gold in
its top 50 for 6/7 candidate misses, and the remaining gold is only rank 48.
Most observed overlap is generic (`the`, `and`, `model`, `system`) because the
current BM25 configuration does not remove stop words. The missing issue is
discriminative lexical alignment and rank, not literal token intersection.

### Dense semantic gap

Dense retrieval also places no gold in its top 50 for 3/7 candidate misses.
Their maximum gold cosine scores are 0.1857, 0.2065, and 0.3498. The other four
candidate misses have a dense gold rank of 15–34 (cosine 0.3396–0.6568) but lose
the evidence during top-20 fusion, so a high pairwise cosine by itself does not
guarantee candidate survival in the full corpus.

## Exclusive failure-mode breakdown

To avoid double-counting a miss that fails both sparse and dense retrieval, the
16 vague top-four misses are assigned hierarchically: reranker demotion first;
among candidate misses, `semantic-gap` means dense also misses top 50, otherwise
`lexical-gap` means BM25 misses top 50 while dense finds the gold.

| Exclusive primary failure mode | Misses | Share of 16 misses |
| --- | ---: | ---: |
| Lexical gap, dense still retrieves | 4 | 25.00% |
| Semantic gap, dense also misses | 3 | 18.75% |
| Reranker demotion | 9 | **56.25%** |

Two of the three semantic-gap cases also miss BM25 top 50; those overlapping
attributes are deliberately counted once in the exclusive table. Viewed as
non-exclusive diagnostics, 6/16 misses have a candidate-stage BM25 top-50 miss,
3/16 have a candidate-stage dense top-50 miss, and 9/16 are reranker demotions.

## Root-cause conclusion

For this paired vague tier, the first Part D experiment should target the
reranker/final-boundary behavior: it is the largest exclusive cause (56.25%)
and four misses are immediately below the top-four cutoff. Candidate generation
is still a major secondary problem (43.75%); within it, query expansion or
lexical normalization should be evaluated before replacing the dense model,
because dense retrieval finds four of the seven candidate misses while BM25
finds none at a useful rank. The three true dense misses remain a smaller,
clearly identified semantic-retrieval slice.

Machine-readable per-question scores and diagnostics are under
`evaluation/data/eval_results/part9l_mixed_style_retrieval_20260814/`.
