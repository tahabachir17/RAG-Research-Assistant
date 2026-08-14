# Controlled generation RAGAS analysis

## Executive result

The controlled generation run completed all 12 reviewed questions and all 60
RAGAS metric values. Retrieval was bypassed: every answer used the exact reviewed
gold chunk or chunks. This isolates generation quality from retrieval failure.

Generator: Groq `llama-3.3-70b-versatile`  
Primary judge: Gemini `gemini-3.5-flash-lite`  
Fallback judge: Groq `llama-3.1-8b-instant`  
Fallback calls: **0**

| Metric | Mean | Interpretation |
|---|---:|---|
| Faithfulness | 1.000 | Generated claims were supported by the supplied evidence. |
| Answer relevancy | 0.688 | Below the 0.80 target; several short, correct answers scored unexpectedly low. |
| Context precision | 1.000 | Expected because only reviewed gold evidence was supplied. |
| Context recall | 1.000 | Expected because the frozen context equals the reviewed reference context. |
| Answer correctness | 0.621 | Main actionable weakness: answers often omit important parts of the reference. |

Deterministic generation checks were strong: citation validity and claim-level
citation coverage were both 1.000, no answers were truncated, no repair retry was
needed, and average generation latency was 3.77 seconds.

## Difficulty slices

| Difficulty | Faithfulness | Answer relevancy | Context precision | Context recall | Answer correctness |
|---|---:|---:|---:|---:|---:|
| Easy (6) | 1.000 | 0.588 | 1.000 | 1.000 | 0.712 |
| Moderate (6) | 1.000 | 0.788 | 1.000 | 1.000 | 0.531 |

Moderate questions have the expected correctness drop: their reference answers
contain several required concepts, while the generator often returns only the
most prominent one. The lower easy-question answer-relevancy score does not match
human inspection as cleanly and should not be interpreted as an easy-question
quality regression without judge calibration.

## Most actionable misses

1. `controlled-qrnn-02` — answer correctness 0.155. The generated answer says
   QRNNs combine parallelism and context, but omits the actual mechanisms:
   convolutional parallel feature computation and recurrent pooling for order and
   long-range context.
2. `controlled-submodular-02` — answer correctness 0.402. The answer reports
   DUC improvement, many-feature capacity, and overfitting control, but omits
   directly learning scoring parameters and the coverage/redundancy trade-off.
3. `controlled-video-02` — answer correctness 0.496. The answer correctly says
   averaging loses temporal order but omits the incoherent mixing of distinct
   events and objects.
4. `controlled-enquirer-02` — answer correctness 0.541. It correctly identifies
   stacked executors and end-to-end query-answer training, but leaves out the
   layered memory of intermediate table annotations.

These are generation completeness failures, not hallucinations: all 12 answers
received faithfulness 1.000.

## Answer-relevancy caveat

Some answer-relevancy values appear harsher than human inspection. For example,
`controlled-reddit-01` directly names the best bidirectional-LSTM architecture
and has answer correctness 0.805, yet answer relevancy is only 0.292. Likewise,
the concise extractive-summarization answer scores 0.464 relevancy. RAGAS answer
relevancy uses generated-question/embedding similarity, so wording and long paper
titles can affect it even when an answer directly addresses the question.

Treat the 0.688 aggregate as a diagnostic signal pending a small human-vs-judge
calibration set. It should not outweigh the direct correctness and faithfulness
evidence.

## Recommended remediation

1. Add a prompt rule for multi-part explanatory questions: cover every distinct
   mechanism, reason, limitation, or future-work item supported by the evidence.
2. Add explicit answer requirements to the golden records for moderate questions
   and measure requirement coverage deterministically alongside RAGAS.
3. Keep this frozen-context test as the generation unit test, then run the same
   questions through real retrieval for an end-to-end RAG test. The score gap will
   quantify how much quality is lost specifically at retrieval.
4. Hand-label 5–10 answer-relevancy cases and compare Gemini judgments before
   using the 0.80 relevancy target as a release gate.

## Artifacts

- Golden set: `evaluation/data/controlled_generation_qa.json`
- Full JSON: `evaluation/data/eval_results/controlled_generation_ragas_20260813/report.json`
- Per-question CSV: `evaluation/data/eval_results/controlled_generation_ragas_20260813/per_question.csv`
- Generated summary: `evaluation/data/eval_results/controlled_generation_ragas_20260813/summary.md`
- Resumable metric cache: `evaluation/data/eval_results/controlled_generation_ragas_20260813/metric_cache.jsonl`
