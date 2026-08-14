# Controlled Retrieval Benchmark

Date: 2026-08-13

## Design

This pilot benchmark contains 12 manually authored, self-contained questions
grounded in six papers already present in the production corpus. Each record
includes the paper title and ID, a paraphrased expected answer, a difficulty
label, and exact answer-bearing chunk IDs. All gold chunks were verified in the
99,141-chunk production BM25 index, and all answers were checked against their
designated evidence.

The evaluation used production BM25 over the entire 1,195-paper corpus. It did
not restrict retrieval by paper ID; paper titles were included in the questions
to make open-corpus retrieval well posed.

## Results

| Group | N | Recall@4 | Hit@4 | MRR | nDCG@4 | Recall@8 | Hit@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 12 | 0.7917 | 0.8333 | 0.4125 | 0.4906 | 0.9167 | 0.9167 | 0.5426 |
| Easy | 6 | 0.7500 | 0.8333 | 0.4306 | 0.5043 | 0.8333 | 0.8333 | 0.5439 |
| Moderate | 6 | 0.8333 | 0.8333 | 0.3944 | 0.4769 | 1.0000 | 1.0000 | 0.5414 |

Ten of 12 questions retrieve answer-bearing evidence in the top four; 11 of 12
do so by top eight. All moderate questions succeed by k=8. This is substantially
better than the previous reviewed benchmarks:

| Benchmark | Hit@4 | Hit@8 |
| --- | ---: | ---: |
| Controlled authored questions | 0.8333 | 0.9167 |
| Part 9e reviewed QASA+QASPER, BM25 | 0.4933 | 0.6133 |
| Part 9e QASPER only, BM25 | 0.2500 | 0.4000 |

## Interpretation

The retriever is not generally nonfunctional. It performs reasonably well when
queries are self-contained, identify the source document, and have complete
answer-chunk labels. The earlier QASPER result is therefore strongly affected by
questions that assume the source paper is already known and use expressions such
as “the authors,” “they,” or “the proposed method” without enough open-corpus
identification context.

The controlled benchmark is intentionally small and author-created, so it is a
diagnostic pilot rather than a release threshold. Its questions are not merely
trivial: the moderate subset asks how/why questions and achieves 100% Hit@8.
This suggests question completeness matters more than nominal difficulty.

One controlled question about Neural Symbolic Machine components still misses at
top 20. Its terminology collides with later neural-symbolic papers containing
similar phrases, showing that full-corpus lexical retrieval can fail even with a
title-aware query. The retrieval layer therefore still needs improvement, but
the earlier 20% QASPER Hit@4 substantially understates its behavior on properly
specified questions.

## Recommended benchmark structure

Maintain three separate tracks rather than replacing difficult questions:

1. Controlled open-corpus questions like this set, expanded and independently
   reviewed, for basic retrieval health.
2. Paper-filtered QASPER questions for within-document evidence localization.
3. Original underspecified open-corpus questions as an explicit stress test.

The benchmark and machine-readable results are stored at
`evaluation/data/controlled_retrieval_qa.json` and
`evaluation/data/eval_results/controlled_retrieval_20260813/`.
