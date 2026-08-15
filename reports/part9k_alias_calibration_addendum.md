# Part 9k — Enquirer-02 alias-calibration addendum

## Decision

The Enquirer-02 matcher false negatives are closed with exactly two bounded,
verbatim aliases. No scoring logic, thresholds, required concepts, or aliases for
other questions were changed.

## Added aliases

| Required concept | Added verbatim alias | Source answer |
|---|---|---|
| operations conditioned on the query | `each executor models a specific type of operation conditioned on the query` | Part 9j run 3 |
| end-to-end training from query-answer pairs | `trained using query-answer pairs, where the distributed representations of queries and the table are optimized together with the query execution logic in an end-to-end fashion` | Part 9j run 2 |

The source answers are preserved in
`evaluation/data/eval_results/part9j_three_run_20260815_run2/report.json` and
`evaluation/data/eval_results/part9j_three_run_20260815_run3/report.json`.

## Re-score

Using the deterministic concept matcher on the unchanged Part 9j answers:

| Part 9j answer | Before | After |
|---|---:|---:|
| run 1 | 0.75 | 0.75 |
| run 2 | 0.50 | 0.75 |
| run 3 | 0.25 | 0.75 |

The post-calibration result is therefore `0.75 / 0.75 / 0.75`. Alias counts for
the four Enquirer-02 concepts are `2 / 2 / 1 / 4`; every concept remains below
the ceiling of five aliases.

## Anti-overcredit checks

The calibration does not make known partial answers look complete:

- The old vague QRNN-02 answer in
  `controlled_generation_ragas_completeness_20260814/report.json` remains
  `0.0`; none of its three mechanism concepts match.
- The old partial Video-02 answer in
  `controlled_generation_part9f_single_pass_20260814/report.json` remains
  `0.6667`; temporal structure and appearance order match, while fusion does
  not.

These checks support a narrow false-negative correction rather than a general
loosening of the matcher.
