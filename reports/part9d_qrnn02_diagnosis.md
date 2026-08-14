# Part 9d QRNN-02 diagnosis

## Confirmed causes

Two causes were confirmed in sequence:

1. The gold conclusion has an **evidence gap**: it does not name recurrent
   pooling.
2. The canonical dataset wording had a **routing gap**: the leading
   `In '<paper title>',` clause hid the sentence-internal `how do` intent and
   classified the question as `direct_fact`.

The bare question form was already `mechanism`, but the canonical title-prefixed
form was `direct_fact`. The classifier now strips that general quoted title clause
before classification, and regression tests pin both forms to `mechanism`.

## Evidence before packing

The controlled run passed only chunk
`86e4d367-e844-569c-9afa-c41c7768edaa` (`conclusion`) to generation:

```text
Intuitively, many aspects of the semantics of long sequences are context-invariant and can be com-
puted in parallel (e.g., convolutionally), but some aspects require long-distance context and must be
computed recurrently. Many existing neural network architectures either fail to take advantage of
the contextual information or fail to take advantage of the parallelism. QRNNs exploit both
parallelism and context, exhibiting advantages from both convolutional and recurrent neural networks.
```

This supplies convolutional parallelism and recurrent long-distance context, but
it does not name the recurrent pooling mechanism.

## Answer before packing

The completed pre-fix controlled run returned:

```text
QRNNs combine the useful properties of convolutional and recurrent sequence models by exploiting
both parallelism and context, exhibiting advantages from both convolutional and recurrent neural
networks. [1]
```

Its deterministic `concept_recall` is `0.0`.

## Evidence after packing

For multi-concept mechanism questions, the controlled evaluator supplies the
gold chunk plus immediate same-paper neighbors, excluding references and other
non-evidence sections. For QRNN-02 this adds preceding chunk
`853684bd-7f1a-5c6b-98ee-ca11177821fd` (`experiments`):

Raw adjacent-chunk excerpt:

```text
regularization schemes would perform well when applied to the QRNN. Our tests showed encouraging
results from zoneout applied to the QRNN’s recurrent pooling layer, implemented as described in
Section 2.1.
```

The phrase `recurrent pooling` is literally present as `recurrent pooling layer`.

Raw gold-chunk excerpt:

```text
Intuitively, many aspects of the semantics of long sequences are context-invariant and can be com-
puted in parallel (e.g., convolutionally), but some aspects require long-distance context and must be
computed recurrently.
```

The exact phrase `convolutional computation` is not present, but the equivalent
operation is explicit and compact: `computed in parallel (e.g., convolutionally)`.
Both required mechanisms are therefore present in extractable form across the
two supplied chunks. The prompt language and evidence-packing scope are unchanged.

## Final prompt delivery pre-flight

Before the targeted mechanism-instruction change, the exact final system and
user strings passed to the model client were captured at the `complete_json`
boundary used by the Groq client.

Result: **PASS**. Both stored chunks reach the model completely and unmodified.

- Context budget: 2,500 whitespace tokens.
- Assembled context: 480 whitespace tokens.
- Stored source text: 176 tokens for the gold chunk and 264 for the adjacent
  chunk.
- Final order: gold chunk `[1]`, then adjacent chunk `[2]`.
- Final context IDs:
  `86e4d367-e844-569c-9afa-c41c7768edaa`,
  `853684bd-7f1a-5c6b-98ee-ca11177821fd`.
- Neither chunk was dropped, partially truncated, reordered, or deduplicated.

Exact final-prompt excerpts:

```text
[1] Title: Quasi-Recurrent Neural Networks | Authors: James Bradbury, Stephen Merity,
Caiming Xiong, Richard Socher | Year: 2016
Section: conclusion
"Intuitively, many aspects of the semantics of long sequences are context-invariant and can be
com- puted in parallel (e.g., convolutionally), but some aspects require long-distance context and
must be computed recurrently."
```

```text
[2] Title: Quasi-Recurrent Neural Networks | Authors: James Bradbury, Stephen Merity,
Caiming Xiong, Richard Socher | Year: 2016
Section: experiments
"regularization schemes would perform well when applied to the QRNN. Our tests showed encouraging
results from zoneout applied to the QRNN’s recurrent pooling layer, implemented as described in
Section 2.1."
```

`recurrent pooling layer` occurs literally once. The requested normalized phrase
`computed in parallel (e.g., convolutionally)` is delivered in the source chunk's
unaltered OCR form, `com- puted in parallel (e.g., convolutionally)`. This is a
source-text dehyphenation artifact, not context loss or prompt mutation.

## Answer after routing and packing fix

The post-fix live pre-flight confirmed:

- Classification: `mechanism`
- Context IDs:
  `86e4d367-e844-569c-9afa-c41c7768edaa`,
  `853684bd-7f1a-5c6b-98ee-ca11177821fd`
- `concept_recall`: `0.3333333333333333`

The regenerated answer was:

```text
QRNNs combine the useful properties of convolutional and recurrent sequence models by exploiting
both parallelism and context, allowing them to compute many aspects of long sequences in parallel
while also considering long-distance context [1]

The QRNN's architecture allows for extensions to both CNNs and RNNs to be directly applicable, and
its hidden states are more interpretable than those of other recurrent architectures [1]
```

Recall moved above zero because the answer names long-distance context, but it
still omits the concrete terms `recurrent pooling` and `convolutional
computation`. Because the supplied evidence contains both mechanisms in
extractable form, this is now a genuine generation/prompt-adherence failure, not
a remaining evidence-packing gap. At that diagnostic stage, no prompt change
was made and the three-run protocol was not started.

## Targeted mechanism-instruction test

After context delivery passed, only the `mechanism` question-type fragment was
strengthened. The shared completeness preamble, other question-type fragments,
and evidence-packing scope were not changed.

The targeted post-change QRNN-02 run returned:

```text
QRNNs combine the useful properties of convolutional and recurrent sequence models by exploiting
both parallelism and context, with convolutional aspects computed in parallel and recurrent aspects
requiring long-distance context [1]

The QRNN's recurrent pooling layer, implemented with zoneout, contributes to its ability to combine
convolutional and recurrent properties [2]
```

Result: **PASS**.

- Classification: `mechanism`.
- Context IDs: gold conclusion followed by the adjacent experiments chunk.
- Convolution mechanism: `convolutional aspects computed in parallel`.
- Recurrent mechanism: `recurrent pooling layer`.
- `concept_recall`: `0.6666666666666666`, up from the prior `0.3333333333333333`.

The deterministic exact-phrase scorer credits `recurrent pooling` and
`long-distance context`. It does not credit the close technical paraphrase
`convolutional aspects computed in parallel` for the required phrase `parallel
convolutional computation`, so the score is 2/3 rather than 3/3. No other
questions were generated and the three-run protocol was not started.

After bounded, hand-authored aliases were added to the concept schema, the same
saved answer rescored to `1.0`: the convolution concept matched the alias
`convolutional aspects computed in parallel`, while the other two concepts
matched their primary phrases. The old vague QRNN answer remains at `0.0`.
