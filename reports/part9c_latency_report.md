# Part 9c — Retrieval Latency Report

Date: 2026-08-13

## Full-stack breakdown

The Part 9b 75-question run measured dense at 64.25 ms/query, sparse at
54.33 ms/query, and hybrid RRF at 119.92 ms/query. The implied fusion overhead
was 1.35 ms/query. Its original 50-candidate reranker configuration raised the
mean to 28,439.68 ms/query, and post-rerank MMR raised it to 32,427.84 ms/query.

Part 9c re-profiled the full runner's real `candidate_k=20`, `top_k=4`,
`max_length=128`, `batch_size=32` configuration over all 75 reviewed QASA/QASPER
questions using the frozen hybrid candidates:

| Candidate k | Recall@4 | Mean rerank | p50 rerank | p95 rerank |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.2841 | 570.2 ms | 530.9 ms | 857.2 ms |
| 20 (current default) | 0.2486 | 991.3 ms | 973.0 ms | 1,244.7 ms |

At candidate-k 20, the estimated production-style total is about 1,111 ms per
query (64.25 dense + 54.33 sparse + 1.35 fusion + 991.30 rerank), making the
reranker approximately 89.2% of retrieval latency. MMR contributes zero latency
at the generator's k=4 boundary because Part 9c bypasses it below k=20.

The profile artifact, including all per-query timings, is
`evaluation/data/eval_results/part9c_20260813/reranker_profile.json`.

## Recommendation

Candidate-k 10 cuts median reranking latency by 45.4% and did not lose Recall@4
on these frozen candidates; it improved it by 0.0356. This is the preferred
mitigation to validate in a fresh end-to-end retrieval run. Per the Part 9c
constraint, the default remains candidate-k 20 until that fresh comparison is
reviewed. Batching is already enabled. A smaller cross-encoder is not recommended
before candidate-size validation because the current MiniLM-L6 model is already
small and candidate reduction offers a lower-risk lever.

## Caveats

The profile reuses frozen hybrid candidates, which isolates reranker cost and
ranking behavior but is not a new dense/sparse retrieval run. CPU contention and
warm-cache state make absolute latency environment-specific. Recall values here
use the generator's actual top four, while the original Part 9b report primarily
reported reranked lists produced from 50 candidates.
