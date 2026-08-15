# Part 9k — QRNN-02 stability investigation

## Decision

The configured Groq generation temperature was already `0.0`, and the outgoing
request also contained `temperature: 0.0`. The Part 9j swing is therefore not an
unnoticed nonzero-temperature configuration error.

A single mechanism-only prompt strengthening was tested. It did not stabilize
QRNN-02 and is **discarded**: the first two controlled trials already diverged
from `1.0000` to `0.3333`, so the intervention cannot satisfy the stated
three-run stability objective regardless of the third value.

## Temperature verification

Runtime construction with `Settings()` and `GroqClient._request()` produced:

```text
provider: groq
model: llama-3.3-70b-versatile
configured_temperature: 0.0
request_temperature: 0.0
max_tokens: 1024
```

This confirms both the setting and the API request layer. Temperature zero does
not guarantee byte-identical hosted-model output, so the observed Part 9j
generation variation remains plausible.

## Narrow intervention

Only the `mechanism` entry in `QUESTION_TYPE_INSTRUCTIONS` was temporarily
extended with:

> Present every distinct mechanism in its own sentence or list item; do not
> combine two mechanisms into one summary clause.

The shared completeness preamble, every other question type, retrieval, the
benchmark question set, NSM-02, and Video-02 were unchanged. A scoped unit-test
assertion ensured that this text existed only in the mechanism instruction.

## QRNN-02-only trials

Each trial used frozen gold-context evidence packing, the same configured Groq
model, and the deterministic three-concept matcher.

| Trial | Concept recall | Parallel convolution | Recurrent pooling | Long-distance context |
|---|---:|---|---|---|
| 1 | 1.0000 | matched | matched | matched |
| 2 | 0.3333 | missing | matched | missing |
| 3 | 0.3333 | missing | matched | missing |

Trial 1 explicitly said that convolutional aspects are computed in parallel,
named recurrent pooling, and named long-distance context. Trials 2 and 3
reverted to the generic phrase “parallelism and context”; only recurrent
pooling matched. This is a real completeness loss, not a matcher false
negative. The final distribution is `1.0000 / 0.3333 / 0.3333` (mean `0.5556`,
population SD `0.3143`).

Artifacts are under
`evaluation/data/eval_results/part9k_qrnn02_stability_20260815_run{1,2,3}`.

## Recommendation

Discard the tested sentence: it adds prompt complexity without reliable
adherence. The experiment does not justify changing the shared preamble,
loosening the matcher, or reopening accepted NSM-02/Video-02 limitations.
QRNN-02 remains a documented generation-completeness instability at
temperature zero. Any future intervention should be separately proposed and
tested rather than inferred from this failed instruction.
