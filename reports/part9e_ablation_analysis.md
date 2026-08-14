# Part 9e — QASPER Retrieval-Stage Ablation

Date: 2026-08-13

## Headline finding

Of the 16 QASPER questions missed by the current hybrid+rerank top four,
**11 are candidate-generation misses** and **5 are reranker-demotion misses**.
The dominant failure occurs before the cross-encoder: no reviewed gold chunk
reaches the fused top-20 candidate list for 68.8% of these misses.

## Miss-stage isolation

| Question ID | BM25 rank | Dense rank | Hybrid RRF rank | Post-rerank rank | Miss stage |
| --- | ---: | ---: | ---: | ---: | --- |
| `qasper-bc473c5bd0e1a8be9b2037aa7006fd68217c3f47` | 4 | not in top-20 | 8 | 9 | reranker_demotion |
| `qasper-32a232310babb92991c4b1b75f7aa6b4670ec447` | 2 | 3 | 1 | 12 | reranker_demotion |
| `qasper-fb5ce11bfd74e9d7c322444b006a27f2ff32a0cf` | not in top-20 | 11 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-559c1307610a15427caeb8aff4d2c01ae5c9de20` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-83f567489da49966af3dc5df2d9d20232bb8cb1e` | 6 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-efc65e5032588da4a134d121fe50d49fe8fe5e8c` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-36a9230fadf997d3b0c5fc8af8d89bd48bf04f12` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-9651fbd887439bf12590244c75e714f15f50f73d` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-71fca845edd33f6e227eccde10db73b99a7e157b` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-994ac7aa662d16ea64b86510fcf9efa13d17b478` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-f2c5da398e601e53f9f545947f61de5f40ede1ee` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-0fa81adf00662694e1dc74475ae2b9283c50748c` | 7 | 16 | 5 | 6 | reranker_demotion |
| `qasper-b3fcab006a9e51a0178a1f64d1d084a895bd8d5c` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-9508e9ec675b6512854e830fa89fa6a747b520c5` | not in top-20 | not in top-20 | not in top-20 | not in top-20 | candidate_generation |
| `qasper-234ccc1afcae4890e618ff2a7b06fc1e513ea640` | not in top-20 | 5 | 11 | 20 | reranker_demotion |
| `qasper-2ff3898fbb5954aa82dd2f60b37dd303449c81ba` | 8 | not in top-20 | 19 | 16 | reranker_demotion |

Two candidate-generation failures are specifically fusion losses: `qasper-fb5...`
is dense rank 11 but absent from fused top 20, and `qasper-83f...` is BM25 rank 6
but absent from fused top 20. The remaining nine candidate-generation misses
have no gold chunk in either backend's top 20.

## Configuration and cutoff ablation

| Config | k | Tier | N | Recall@k | Precision@k | Hit@k | MRR | nDCG@k | p50 ms | p95 ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 4 | Overall | 75 | 0.3952 | 0.1433 | 0.4933 | 0.3911 | 0.3583 | 36.1 | 70.2 |
| BM25 | 4 | QASA | 55 | 0.4784 | 0.1727 | 0.5818 | 0.4788 | 0.4448 | 41.8 | 74.1 |
| BM25 | 4 | QASPER | 20 | 0.1667 | 0.0625 | 0.2500 | 0.1500 | 0.1204 | 30.9 | 40.9 |
| BM25 | 8 | Overall | 75 | 0.5063 | 0.0967 | 0.6133 | 0.4091 | 0.4012 | 36.1 | 70.2 |
| BM25 | 8 | QASA | 55 | 0.5935 | 0.1136 | 0.6909 | 0.4954 | 0.4898 | 41.8 | 74.1 |
| BM25 | 8 | QASPER | 20 | 0.2667 | 0.0500 | 0.4000 | 0.1717 | 0.1577 | 30.9 | 40.9 |
| Hybrid RRF | 4 | Overall | 75 | 0.3756 | 0.1333 | 0.4667 | 0.3111 | 0.2986 | 105.9 | 160.1 |
| Hybrid RRF | 4 | QASA | 55 | 0.4758 | 0.1682 | 0.5818 | 0.3788 | 0.3708 | 108.6 | 167.7 |
| Hybrid RRF | 4 | QASPER | 20 | 0.1000 | 0.0375 | 0.1500 | 0.1250 | 0.1000 | 96.1 | 123.4 |
| Hybrid RRF | 8 | Overall | 75 | 0.4952 | 0.0917 | 0.6133 | 0.3350 | 0.3467 | 105.9 | 160.1 |
| Hybrid RRF | 8 | QASA | 55 | 0.5784 | 0.1091 | 0.7091 | 0.4002 | 0.4148 | 108.6 | 167.7 |
| Hybrid RRF | 8 | QASPER | 20 | 0.2667 | 0.0438 | 0.3500 | 0.1558 | 0.1592 | 96.1 | 123.4 |
| Hybrid + rerank | 4 | Overall | 75 | 0.2486 | 0.0933 | 0.3600 | 0.2400 | 0.2136 | 1,282.4 | 1,976.9 |
| Hybrid + rerank | 4 | QASA | 55 | 0.2965 | 0.1091 | 0.4182 | 0.2985 | 0.2655 | 1,300.6 | 1,844.7 |
| Hybrid + rerank | 4 | QASPER | 20 | 0.1167 | 0.0500 | 0.2000 | 0.0792 | 0.0710 | 1,248.6 | 1,989.0 |
| Hybrid + rerank | 8 | Overall | 75 | 0.3419 | 0.0683 | 0.4800 | 0.2609 | 0.2509 | 1,282.4 | 1,976.9 |
| Hybrid + rerank | 8 | QASA | 55 | 0.4056 | 0.0818 | 0.5636 | 0.3240 | 0.3098 | 1,300.6 | 1,844.7 |
| Hybrid + rerank | 8 | QASPER | 20 | 0.1667 | 0.0312 | 0.2500 | 0.0875 | 0.0888 | 1,248.6 | 1,989.0 |

## Direct answers

- **Does QASPER improve at k=8?** Yes, but the magnitude depends on the
  configuration. BM25 Recall rises from 0.1667 to 0.2667 and Hit from 0.25 to
  0.40. Hybrid RRF Recall rises from 0.10 to 0.2667 and Hit from 0.15 to 0.35.
  Hybrid+rerank improves less: Recall 0.1167 to 0.1667 and Hit 0.20 to 0.25.
- **Does hybrid RRF beat BM25 on QASPER?** No at k=4: BM25 leads by 0.0667
  Recall and 0.10 Hit. At k=8 their Recall ties at 0.2667, but BM25 still has
  higher Hit (0.40 vs 0.35), MRR, and much lower latency.
- **Does reranking help QASPER?** Not materially. At k=4 it improves hybrid RRF
  Recall slightly (0.10 to 0.1167) and Hit (0.15 to 0.20), but remains below
  BM25 and sharply reduces MRR/nDCG. At k=8 it hurts hybrid RRF: Recall falls
  from 0.2667 to 0.1667 and Hit from 0.35 to 0.25. This is consistent with the
  five observed reranker-demotion misses.
- **Is QASPER weak everywhere?** Yes. Its best tested result is BM25 at k=8
  (Recall 0.2667, Hit 0.40), still well below QASA. The current hybrid+rerank
  configuration is not the best of the tested choices for QASPER.

This report is diagnostic only. No retrieval, processing, ingestion, generation,
or evaluation-layer fix is proposed or applied.
