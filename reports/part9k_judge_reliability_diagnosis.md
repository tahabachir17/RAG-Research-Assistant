# Part 9k — judge reliability diagnosis

## Decision

The AnswerCorrectness variance is not caused by an unpinned project judge
temperature. It is inherent observed variance in the selected judge/model path,
amplified by AnswerCorrectness's three sequential LLM classification/extraction
calls. No judge configuration change is justified.

RAGAS AnswerCorrectness must remain a supporting diagnostic, not a single-run
release gate. Deterministic concept recall and source-based manual review remain
the primary evidence for controlled-generation completeness.

## Effective temperature audit

- `Settings.JUDGE_TEMPERATURE` is `0.0`.
- `_ragas_llm_options()` deliberately omits `temperature` for model names that
  begin with `gemini-3.5`.
- The installed `ChatOpenAI` object consequently exposes its client default of
  `0.7`, but that is not the effective RAGAS request value: RAGAS 0.1.14
  `BaseRagasLLM.generate()` supplies `1e-8` per call when no temperature is
  provided, and `LangchainLLMWrapper.agenerate_text()` forwards it.
- More importantly, Google's current Gemini 3.5 Flash-Lite documentation says
  `temperature`, `top_p`, and `top_k` are deprecated and ignored for this model.
  Removing or changing a local temperature value therefore cannot make these
  judge calls deterministic. See [Using the latest Gemini
  models](https://ai.google.dev/gemini-api/docs/latest-model) and the
  [Gemini 3.5 Flash-Lite model
  page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite?authuser=0).

The omission in `_ragas_llm_options()` is therefore compatible with the model's
current API contract and is not the variance root cause.

## AnswerCorrectness call path

For one scored row, the installed RAGAS implementation makes three judge calls:

1. extract statements from the generated answer;
2. extract statements from the reference answer;
3. classify generated statements into true-positive, false-positive, and
   false-negative groups.

It then computes semantic similarity locally and combines factuality and
similarity with weights `0.75 / 0.25`. The judge uses one completion per call,
not self-consistency sampling. An invalid output can trigger an additional
parser-repair call, but every targeted trial below produced exactly three raw
subcalls, so parser repair does not explain these results.

Relevant local implementations are `evaluation/ragas_evaluator.py`,
`venv/Lib/site-packages/ragas/metrics/_answer_correctness.py`, and
`venv/Lib/site-packages/ragas/llms/base.py`.

## Targeted identical-input experiment

QRNN-01 and NSM-01 had byte-identical answers across the three Part 9j runs.
Each answer was re-evaluated three times with the configured Gemini primary,
AnswerCorrectness only, and fallback disabled.

| Question | trial 1 | trial 2 | trial 3 | mean | population SD | min–max |
|---|---:|---:|---:|---:|---:|---:|
| QRNN-01 | 0.9627 | 0.9627 | 0.9627 | 0.9627 | 0.0000 | 0.0000 |
| NSM-01 | 0.1744 | 0.1744 | 0.7369 | 0.3619 | 0.2652 | 0.5625 |

Historical Part 9j identical-answer values independently showed ranges of
`0.25` for QRNN-01 (`0.9627 / 0.7127 / 0.9627`) and `0.60` for NSM-01
(`0.7744 / 0.1744 / 0.4744`). The targeted NSM-01 reproduction, with exactly
three subcalls in every trial and no fallback, rules out generation differences,
fallback routing, and parser repair as necessary causes.

## Standing interpretation policy

1. Never use one AnswerCorrectness observation as a pass/fail release gate.
2. When the metric is used on an unchanged answer and configuration, run at
   least three independent judge evaluations and report the mean, population
   standard deviation, and min–max range.
3. Treat changes within the currently observed per-question envelope of `0.60`
   as non-diagnostic unless deterministic or manual source evidence corroborates
   them. This is an empirical calibration from two controls, not a guaranteed
   upper bound.
4. Preserve the raw per-trial values and call/error metadata; do not report only
   their average.
5. Prefer deterministic concept recall for explicitly enumerated coverage and
   manual source review for semantic correctness. Use AnswerCorrectness as
   supporting evidence and investigate only effects that clearly exceed the
   calibrated noise or reproduce under a more deterministic judge.

## Conclusion

The effective project setting was already as close to deterministic as this
stack permits, while Gemini 3.5 Flash-Lite ignores the relevant sampling
controls. The reproduced spread on byte-identical NSM-01 input establishes
judge/metric variability. Pinning a nominal temperature would be a misleading
fix, so the appropriate corrective action is the standing multi-run
interpretation policy above.
