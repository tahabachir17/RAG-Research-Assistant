# Part 9d concept-recall alias calibration

## Previous matching logic

`concept_recall` previously normalized each required concept and the answer by:

1. converting punctuation and other non-alphanumeric characters to spaces;
2. case-folding; and
3. collapsing whitespace.

It then performed an exact normalized substring check:

```text
normalized_required_concept in normalized_answer
```

There was no token reordering, stemming, synonym handling, fuzzy distance,
embedding similarity, or LLM judgment.

For QRNN-02, the required phrase normalized to:

```text
parallel convolutional computation
```

The targeted answer wording normalized to:

```text
convolutional aspects computed in parallel
```

Neither string is a substring of the other: the word order differs,
`computation`/`computed` differ morphologically, and `aspects` intervenes. This
is why a valid technical paraphrase was not credited.

## Bounded alias schema

A required concept may remain a string or use an object:

```json
{
  "concept": "parallel convolutional computation",
  "aliases": [
    "convolutional aspects computed in parallel",
    "computed in parallel (e.g., convolutionally)",
    "convolutional computation"
  ]
}
```

Aliases are hand-authored and capped at five per concept. The matcher tests the
primary phrase first, then aliases in authored order, using exactly the same
normalized substring rule. It introduces no automatic, fuzzy, semantic,
embedding, or model-based expansion.

Evaluation rows now include `concept_recall_details`, recording the primary
concept, whether it matched, `primary` versus `alias` attribution, and the exact
authored phrase that matched.

## QRNN-02 rescore

Targeted post-instruction answer score before aliases: `0.6666666666666666`.

Score with the bounded QRNN alias: `1.0`.

| Required concept | Credited by | Matched phrase |
|---|---|---|
| parallel convolutional computation | alias | convolutional aspects computed in parallel |
| recurrent pooling | primary | recurrent pooling |
| long-distance context | primary | long-distance context |

The alias list was derived from the actual targeted answer and the two supplied
QRNN chunks. No alias was added for the old generic wording `parallelism and
context`, because it does not identify either concrete mechanism.

## Over-crediting spot-check

Three historical incomplete answers from
`controlled_generation_ragas_completeness_20260814` were rescored:

| Answer | Incompleteness | Alias-aware score | False-positive change? |
|---|---|---:|---|
| QRNN-02 old vague summary | only says `parallelism and context` | 0.0 | no |
| Enquirer-02 old partial answer | omits conditioned operations and layered intermediate annotations | 0.0 | no |
| Video-02 old partial answer | omits incoherent fusion of distinct events and objects | 0.0 | no |

The alias widens only the auditable phrasing of the QRNN convolution concept; it
does not lower the coverage threshold for the generic or incomplete answers.
