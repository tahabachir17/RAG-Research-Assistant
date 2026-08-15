# Part 9m — Reduced Live Human-Style Benchmark

Date: 2026-08-15

## Technical summary

Part A is **partial: 0/40 completed and 40/40 skipped**. The run stopped on the
first question before sending its generation request to Groq. The required
`--show-prompt` output contained the Unicode `ﬁ` ligature from a retrieved
passage; printing it through the Windows CP1252 console raised
`UnicodeEncodeError`. In accordance with the stop-on-any-error rule, the
harness checkpointed the failure and did not continue or silently skip ahead.

There are therefore no retrieval-relevance, faithfulness, completeness, or
strict end-to-end rates to report. The 40 uncompleted questions are not counted
as failures, and no aggregate “the app works X%” result is calculated.

## Quota status before the run

A one-output-token Groq preflight was executed through
`EvaluationRateLimitClient` before the benchmark. For the configured
`llama-3.3-70b-versatile` model, the response reported:

| Limit | Available at preflight |
| --- | ---: |
| Requests/day | 999 of 1,000 remaining |
| Tokens/minute | 11,962 of 12,000 remaining |
| Documented tokens/day | 100,000 |
| Preflight usage | 38 total tokens |

The current Groq documentation identifies the response request headers as
daily-request limits and the token headers as minute-token limits. It lists a
100,000-token daily cap for this model. With the repository's 1,000-token
context budget, short structured answers, persistent pacing below the
12,000-token/minute cap, and bounded retries, the quota check was sufficient to
start the 40-call plan. Source: [Groq rate limits](https://console.groq.com/docs/rate-limits).

Only the 38-token preflight reached Groq. The failed benchmark CLI process
stopped while printing its prompt, before live generation.

## Benchmark design

The additive question manifest reuses all 20 evidence needs from the unchanged
Part 9l paired set:

- 20 vague/casual phrasings, followed by
- the corresponding 20 topic-named phrasings.

This creates a controlled pair for every evidence need and introduces no new
gold labels. The planned command for every question was:

```text
python -m generation.cli <question> --retrieve --live --provider groq
```

`--show-prompt` was planned for the first five questions in each tier. Each
exact CLI subprocess was placed behind the existing
`EvaluationRateLimitClient(max_retries=2)` wrapper, with a persistent pacing
interval. No prompt, alias, matcher, disposition, retriever setting, or golden
file was changed.

## Explicit completion status

| Tier | Planned | Completed | Skipped | Prompt captures |
| --- | ---: | ---: | ---: | ---: |
| Vague/casual | 20 | 0 | 20 | 0 of 5 |
| Topic-named | 20 | 0 | 20 | 0 of 5 |
| **Total** | **40** | **0** | **40** | **0 of 10** |

Status: **partial**. Stop reason:

```text
part9m-vague-01: UnicodeEncodeError: the Windows CP1252 stdout encoder could
not encode U+FB01 (`ﬁ`) while generation.cli printed --show-prompt output.
```

The prompt-capture requirement is explicitly incomplete. No question was
retried after this non-transient harness failure, and none of the remaining 39
questions was attempted.

## Results and limitations

No question reached a completed end-to-end response, so manual answer scoring
would be undefined. The correct interpretation is “benchmark not executed,”
not a 0% application score. This run provides no evidence for comparing vague
and topic-named generation quality.

The blocker is isolated to prompt capture on the Windows console. A future
authorized resume should force UTF-8 for the CLI subprocess while leaving the
captured prompt bytes, production prompt text, retrieval stack, and generation
settings unchanged.

## Audit artifacts

Machine-readable records are under
`evaluation/data/eval_results/part9m_reduced_live_benchmark_20260815/`:

- `quota_preflight.json` — quota headers, documented daily cap, and go/no-go decision;
- `question_manifest.json` — all 40 reused questions and reviewed labels;
- `live_checkpoint.json` — immediate 0-question checkpoint and stop reason;
- `reduced_live_results.json` — explicit partial status, 0 completed, 40 skipped.

The new runner is `evaluation/run_part9m_reduced_live_benchmark.py`. Existing
Part 9l and golden artifacts were not modified.
