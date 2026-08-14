# Part 9c — Live Retrieval Score Analysis

Date: 2026-08-13

The production-style retrieval path was evaluated locally on all 75 reviewed
external questions (55 QASA and 20 QASPER). The run used dense retrieval plus
BM25 reciprocal-rank fusion, 20 candidates entering the MiniLM cross-encoder,
and four final contexts. MMR was configured but correctly bypassed at k=4. No
LLM or external API was used.

| Tier | N | Recall@4 | Precision@4 | Hit@4 | MRR | nDCG@4 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 75 | 0.2486 | 0.0933 | 0.3600 | 0.2400 | 0.2136 | 868.9 ms | 1,117.3 ms |
| QASA | 55 | 0.2965 | 0.1091 | 0.4182 | 0.2985 | 0.2655 | 810.8 ms | 1,180.5 ms |
| QASPER | 20 | 0.1167 | 0.0500 | 0.2000 | 0.0792 | 0.0710 | 931.1 ms | 1,043.9 ms |

## Interpretation

Retrieval is operational and latency is stable, but evidence quality at the
four-context generator boundary is weak. Only 27 of 75 questions retrieve any
reviewed evidence in the top four; 48 miss completely. QASA is materially better
than QASPER, while QASPER's 20% hit rate and 0.079 MRR identify it as the primary
retrieval bottleneck.

Precision@4 is expected to be numerically low when a question has only one or a
few relevant chunks, but the 0.36 Hit@4 and 0.249 Recall@4 show that this is not
just a denominator effect: relevant evidence is absent from most generated
context sets. Generation or RAGAS tuning cannot repair those misses.

The live p50 of 869 ms and p95 of 1.12 s are usable for offline evaluation and
far better than the historical 50-candidate profile. The cross-encoder remains
the dominant cost, but relevance—not latency—is currently the first-order issue.

## Recommended next experiments

1. Compare sparse BM25, hybrid RRF, and hybrid rerank at the actual k=4 boundary
   on this same live 75-question set. Historical evidence suggests reranking can
   demote some QASPER evidence even while improving aggregate larger-k recall.
2. Increase the final context budget from k=4 to k=8 as an analysis experiment.
   This directly tests whether evidence is present but ranked just below the
   generator boundary.
3. Stratify the 48 misses into candidate-generation misses versus reranker
   demotions by checking reviewed-evidence ranks before and after reranking.
4. Tune QASPER separately only if the stratification confirms a systematic tier
   difference; likely levers are BM25 weighting, fusion weights, and reranker
   candidate depth rather than MMR.

The machine-readable result and per-question rows are under
`evaluation/data/eval_results/part9c_retrieval_20260813/`.
