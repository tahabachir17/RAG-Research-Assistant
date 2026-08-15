# Part 9l — Human-Style Live Query Benchmark

Date: 2026-08-15

## Design

This is an additive end-to-end benchmark of what a user receives from the live
application path. It contains 16 questions against papers already present in
the production corpus: eight vague/casual questions and eight questions that
name a method or concept without naming a paper title. Topics were selected
before answer-bearing chunks were reviewed. The reviewed IDs in the additive
question file are labels for manual review only and were never supplied as
context.

Every question was executed through:

```text
python -m generation.cli <question> --retrieve --live --provider groq
```

`--show-prompt` was added for the first five questions. The configured Groq
model was `llama-3.3-70b-versatile`. Retrieval and generation were live; there
was no golden-file lookup, paper filter, frozen context, or RAGAS run.

## Manual-review rubric

- **Retrieval relevance:** pass when at least one returned chunk contains
  evidence that materially answers the user's question. Alternate valid corpus
  evidence counts even when it is not one of the separately reviewed chunk IDs.
- **Faithfulness:** pass only when every material answer claim is entailed by
  the returned chunks. A correct abstention can pass faithfulness.
- **Completeness:** pass when the answer addresses every question clause at
  useful specificity. An abstention or materially partial answer fails.

The labels below were assigned by reading the returned chunk text and final
answer, not by RAGAS or an LLM judge.

## Completion status

The live run completed **3/16 questions** before Groq's rolling daily token
quota blocked further completions. At the user's direction, the remaining
**13/16 questions were skipped** rather than retried. All three completed
questions are from the vague/casual half of the set. Skipped questions are not
counted as failures and are excluded from every rate below.

Because only three questions completed, `--show-prompt` was captured for all
three completed questions rather than the requested first five. This is an
explicit incomplete requirement, not silently treated as satisfied.

## Manual results

| ID | Question | Retrieval | Faithfulness | Completeness | Failure note |
| --- | --- | --- | --- | --- | --- |
| `human-vague-01` | Why does predicting a whole translation at once lose information from earlier target words? | Pass | **Fail** | Pass | Rank 2 explains missing target history/collocations, but the answer then conflates this with autoregressive decoding, whose cited downside is latency rather than the information loss asked about. |
| `human-vague-02` | How can a chatbot ask a useful follow-up when a request is ambiguous? | **Fail** | Pass | **Fail** | Returned chunks discuss a values-elicitation chatbot and general dialogue work, not ambiguity-resolution or clarification-question generation; the answer faithfully abstains but does not answer the question. |
| `human-vague-03` | What tends to go wrong when you use a pretrained language model off the shelf for a new task? | Pass | Pass | Pass | No failure: rank 1 directly supports the answer that transfer degrades as the new task's inputs/outputs diverge from pretraining examples. |

## Explicit counts on completed questions

| Axis | Pass | Fail | Pass rate |
| --- | ---: | ---: | ---: |
| Retrieval relevance | 2 | 1 | 66.7% |
| Faithfulness | 2 | 1 | 66.7% |
| Completeness | 2 | 1 | 66.7% |
| Strict all-three-axis result | 1 | 2 | 33.3% |

These rates describe only three completed vague questions and must not be read
as the 16-question benchmark result. The 13 skipped rows comprise five
vague/casual and all eight method-named questions, so no style comparison is
possible from this partial run.

## Audit artifacts

The additive question definitions are in
`evaluation/data/human_style_generation_qa.json`. The full live result records
under
`evaluation/data/eval_results/part9l_human_query_benchmark_20260814/` retain,
for each question, the final answer and all five retrieved chunks with rank,
score, source, paper, section, chunk ID, and full text. All three completed
records also retain the CLI's complete `--show-prompt` output. Reconstruction
checks passed **3/3**: each saved ranking exactly matches the chunk IDs passed
to the live CLI. The result archive explicitly records partial status, three
completed questions, and 13 skipped questions.
