# Part 9j three-run controlled-generation protocol analysis

## Protocol integrity and outcome

Three independent, checkpointed runs completed under the frozen Part 9i
configuration:

- `part9j_three_run_20260815_run1`
- `part9j_three_run_20260815_run2`
- `part9j_three_run_20260815_run3`

Each run generated all 12 controlled answers with Groq
`llama-3.3-70b-versatile` from frozen gold/adjacent contexts and completed all
60 RAGAS values with the configured Gemini-compatible
`gemini-3.5-flash-lite` primary judge. Each artifact reports 12/12 generated,
60/60 metric values, `ragas.status: completed`, and zero errors. The judge
fallback count was zero in every run and zero overall.

No prompt, routing, evidence-packing, golden-set, alias, scorer, or source-code
change was made during or after the protocol. Population standard deviation is
used throughout because these are the complete three protocol runs, not a
sample from a larger executed set.

## Per-question RAGAS results

### Faithfulness

| Question | Run 1 | Run 2 | Run 3 | Mean | Pop. SD |
|---|---:|---:|---:|---:|---:|
| controlled-submodular-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-submodular-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-enquirer-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-enquirer-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-qrnn-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-qrnn-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-nsm-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-nsm-02 | 0.5000 | 0.0000 | 0.0000 | 0.1667 | 0.2357 |
| controlled-video-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-video-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-reddit-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-reddit-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Answer relevancy

| Question | Run 1 | Run 2 | Run 3 | Mean | Pop. SD |
|---|---:|---:|---:|---:|---:|
| controlled-submodular-01 | 0.7270 | 0.7303 | 0.4642 | 0.6405 | 0.1247 |
| controlled-submodular-02 | 0.9273 | 0.9273 | 0.9273 | 0.9273 | 0.0000 |
| controlled-enquirer-01 | 0.7054 | 0.7054 | 0.7054 | 0.7054 | 0.0000 |
| controlled-enquirer-02 | 0.8352 | 0.8970 | 0.8970 | 0.8764 | 0.0291 |
| controlled-qrnn-01 | 0.8416 | 0.8416 | 0.8416 | 0.8416 | 0.0000 |
| controlled-qrnn-02 | 0.7377 | 0.8936 | 0.8818 | 0.8377 | 0.0709 |
| controlled-nsm-01 | 0.5586 | 0.5586 | 0.5586 | 0.5586 | 0.0000 |
| controlled-nsm-02 | 0.6096 | 0.6401 | 0.6096 | 0.6197 | 0.0144 |
| controlled-video-01 | 0.6554 | 0.6554 | 0.6554 | 0.6554 | 0.0000 |
| controlled-video-02 | 0.9377 | 0.9377 | 0.9377 | 0.9377 | 0.0000 |
| controlled-reddit-01 | 0.2918 | 0.2918 | 0.2918 | 0.2918 | 0.0000 |
| controlled-reddit-02 | 0.3582 | 0.3508 | 0.3508 | 0.3533 | 0.0035 |

### Context precision

| Question | Run 1 | Run 2 | Run 3 | Mean | Pop. SD |
|---|---:|---:|---:|---:|---:|
| controlled-submodular-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-submodular-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-enquirer-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-enquirer-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-qrnn-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-qrnn-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-nsm-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-nsm-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-video-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-video-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-reddit-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-reddit-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Context recall

| Question | Run 1 | Run 2 | Run 3 | Mean | Pop. SD |
|---|---:|---:|---:|---:|---:|
| controlled-submodular-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-submodular-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-enquirer-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-enquirer-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-qrnn-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-qrnn-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-nsm-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-nsm-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-video-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-video-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-reddit-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-reddit-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Answer correctness

| Question | Run 1 | Run 2 | Run 3 | Mean | Pop. SD |
|---|---:|---:|---:|---:|---:|
| controlled-submodular-01 | 0.4958 | 0.7320 | 0.5809 | 0.6029 | 0.0977 |
| controlled-submodular-02 | 0.4162 | 0.2340 | 0.2715 | 0.3073 | 0.0786 |
| controlled-enquirer-01 | 0.5511 | 0.5511 | 0.5511 | 0.5511 | 0.0000 |
| controlled-enquirer-02 | 0.4430 | 0.5105 | 0.5192 | 0.4909 | 0.0341 |
| controlled-qrnn-01 | 0.9627 | 0.7127 | 0.9627 | 0.8794 | 0.1179 |
| controlled-qrnn-02 | 0.6564 | 0.5154 | 0.5918 | 0.5879 | 0.0576 |
| controlled-nsm-01 | 0.7744 | 0.1744 | 0.4744 | 0.4744 | 0.2449 |
| controlled-nsm-02 | 0.5248 | 0.5248 | 0.5248 | 0.5248 | 0.0000 |
| controlled-video-01 | 0.6910 | 0.5643 | 0.6910 | 0.6488 | 0.0597 |
| controlled-video-02 | 0.1202 | 0.1202 | 0.3454 | 0.1953 | 0.1062 |
| controlled-reddit-01 | 0.9554 | 0.9554 | 0.9554 | 0.9554 | 0.0000 |
| controlled-reddit-02 | 0.9305 | 0.8725 | 0.8725 | 0.8918 | 0.0274 |

## Per-question deterministic concept recall

These values are fresh deterministic rescoring results from the three saved
answers using the unchanged Part 9i golden definitions and alias set.

| Question | Run 1 | Run 2 | Run 3 | Mean | Pop. SD |
|---|---:|---:|---:|---:|---:|
| controlled-submodular-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-submodular-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-enquirer-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-enquirer-02 | 0.7500 | 0.5000 | 0.2500 | 0.5000 | 0.2041 |
| controlled-qrnn-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-qrnn-02 | 1.0000 | 0.6667 | 0.3333 | 0.6667 | 0.2722 |
| controlled-nsm-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-nsm-02 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.0000 |
| controlled-video-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-video-02 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.0000 |
| controlled-reddit-01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| controlled-reddit-02 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

The strict fixed matcher shows the largest generation-side spread on QRNN-02
(`SD 0.2722`, scores `1.0000/0.6667/0.3333`) and Enquirer-02 (`SD 0.2041`,
scores `0.7500/0.5000/0.2500`). NSM-02 and Video-02 remain stable at their
accepted-limitation scores; their status is not re-opened.

Manual review found that the fixed matcher under-credits Enquirer-02 in runs 2
and 3. Run 2 explicitly says the model is trained from Query-Answer pairs with
the execution logic optimized end to end; run 3 says the same and explicitly
states that each executor models an operation conditioned on the query. Those
phrases do not match the frozen substring aliases because of intervening wording
and `table`/`KB` variation. On semantic manual review, Enquirer-02 is `0.75` in
all three runs, with different one-concept omissions. This is documented as a
post-protocol matcher finding only; no alias or score artifact was changed.

QRNN-02 still exhibits genuine generation variability under strict and manual
review. Run 2 omits an explicit parallel-convolution computation link, while
run 3 additionally omits recurrent pooling. That finding remains even after the
Enquirer matcher false negatives are separated.

## Answer-correctness versus concept-recall cross-check

Direction uses a `0.001` deadband so insignificant floating-point differences
are flat. A row is flagged if either adjacent transition has different
directions, including one metric moving while the other is flat. This directly
implements the requested judge-noise check.

| Question | Correctness R1/R2/R3 | CR R1/R2/R3 | AC delta R3-R1 | CR delta R3-R1 | Transition comparison | Flag? | Exact answers |
|---|---|---|---:|---:|---|---|---|
| controlled-submodular-01 | 0.496/0.732/0.581 | 1.000/1.000/1.000 | +0.085 | +0.000 | 1→2: AC up, CR flat; 2→3: AC down, CR flat | **FLAG** | Vary |
| controlled-submodular-02 | 0.416/0.234/0.272 | 1.000/1.000/1.000 | -0.145 | +0.000 | 1→2: AC down, CR flat; 2→3: AC up, CR flat | **FLAG** | Vary |
| controlled-enquirer-01 | 0.551/0.551/0.551 | 1.000/1.000/1.000 | +0.000 | +0.000 | Both flat | No | All same |
| controlled-enquirer-02 | 0.443/0.511/0.519 | 0.750/0.500/0.250 | +0.076 | -0.500 | Both transitions: AC up, CR down | **FLAG** | Vary |
| controlled-qrnn-01 | 0.963/0.713/0.963 | 1.000/1.000/1.000 | +0.000 | +0.000 | 1→2: AC down, CR flat; 2→3: AC up, CR flat | **FLAG** | All same |
| controlled-qrnn-02 | 0.656/0.515/0.592 | 1.000/0.667/0.333 | -0.065 | -0.667 | 1→2: both down; 2→3: AC up, CR down | **FLAG** | Vary |
| controlled-nsm-01 | 0.774/0.174/0.474 | 1.000/1.000/1.000 | -0.300 | +0.000 | 1→2: AC down, CR flat; 2→3: AC up, CR flat | **FLAG** | All same |
| controlled-nsm-02 | 0.525/0.525/0.525 | 0.750/0.750/0.750 | +0.000 | +0.000 | Both flat | No | Vary only by punctuation |
| controlled-video-01 | 0.691/0.564/0.691 | 1.000/1.000/1.000 | +0.000 | +0.000 | 1→2: AC down, CR flat; 2→3: AC up, CR flat | **FLAG** | Vary |
| controlled-video-02 | 0.120/0.120/0.345 | 0.667/0.667/0.667 | +0.225 | +0.000 | 1→2: both flat; 2→3: AC up, CR flat | **FLAG** | Vary |
| controlled-reddit-01 | 0.955/0.955/0.955 | 1.000/1.000/1.000 | +0.000 | +0.000 | Both flat | No | Vary |
| controlled-reddit-02 | 0.930/0.872/0.872 | 1.000/1.000/1.000 | -0.058 | +0.000 | 1→2: AC down, CR flat; 2→3: both flat | **FLAG** | Vary |

Nine of 12 questions are flagged. The clearest judge-noise controls are exact
unchanged answers:

- QRNN-01 has identical answer text and concept recall `1.0` in every run, but
  answer correctness is `0.9627/0.7127/0.9627`, a `0.2500` range.
- NSM-01 has identical answer text and concept recall `1.0` in every run, but
  answer correctness is `0.7744/0.1744/0.4744`, a `0.6000` range.

The opposite-direction cases are also substantive. Enquirer-02 correctness
rises in both transitions while fixed-matcher recall falls in both; QRNN-02
correctness rises from run 2 to run 3 while deterministic recall falls from
`0.6667` to `0.3333`. These results reproduce the Part 9c warning: one RAGAS
correctness delta cannot be interpreted as an answer-quality delta.

## Aggregate stability across 36 question-runs

`Pooled 36 pop. SD` includes stable differences between questions as well as
run-to-run variation. `Run-mean pop. SD` isolates the spread of the three
12-question aggregate means.

| Metric | Run 1 mean | Run 2 mean | Run 3 mean | Pooled 36 mean | Pooled 36 pop. SD | Run-mean pop. SD |
|---|---:|---:|---:|---:|---:|---:|
| faithfulness | 0.9583 | 0.9167 | 0.9167 | 0.9306 | 0.2402 | 0.0196 |
| answer_relevancy | 0.6821 | 0.7025 | 0.6767 | 0.6871 | 0.2075 | 0.0111 |
| context_precision | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| context_recall | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| answer_correctness | 0.6268 | 0.5390 | 0.6117 | 0.5925 | 0.2391 | 0.0384 |
| concept_recall | 0.9306 | 0.8819 | 0.8333 | 0.8819 | 0.2007 | 0.0397 |

Context precision and recall are perfectly stable, confirming that the frozen
evidence boundary held. Aggregate answer correctness varies by `0.0878` between
its lowest and highest run mean, while individual unchanged-answer swings reach
`0.6000`. Fixed-matcher concept recall declines by `0.0972` from run 1 to run 3;
manual review removes the Enquirer false-negative portion but does not remove
the genuine QRNN-02 mechanism loss.

## Fallback investigation

Fallback judge metric calls were `0/0/0` for runs 1/2/3, total `0`. All 180
RAGAS metric values came from the configured primary judge, so the observed
spread cannot be attributed to mixing primary and fallback judges.

## Conclusion

**NO-GO for treating the current controlled-benchmark configuration as stable
and release-worthy, and NO-GO for trusting single-run RAGAS correctness deltas
in future sessions.**

The retrieval-bypassed evidence boundary is stable, and most deterministic
concept scores are stable. However, QRNN-02 loses required mechanisms across
runs, the frozen matcher has newly exposed Enquirer paraphrase brittleness, and
RAGAS correctness frequently moves independently of deterministic content.
Exact-identical-answer swings of `0.25` and `0.60` are too large to treat a
single correctness score as a reliable change signal.

A separate follow-up should address QRNN-02 generation completeness and review
the documented Enquirer matcher false negatives before another release gate.
NSM-02 and Video-02 remain accepted limitations and are not re-opened by this
conclusion. No fix is made in this task.
