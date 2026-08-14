# Part 9d canonical question-type audit

## Attribution shapes found

All 12 canonical questions were inspected. Their paper attribution appears in
four shapes:

- Leading `In '<title>', ...`: 8 questions.
- Leading `According to the paper '<title>', ...`: QRNN-01.
- Leading `According to '<title>', ...`: Video-02.
- No leading attribution clause: Submodular-02 and Video-01 refer to the paper
  later in the question.

The classifier now removes a leading non-question clause when that clause
contains a quoted paper title and ends at the comma following the title. It does
not enumerate literal lead-in phrases, so the same logic also covers forms such
as `As described in '<title>', ...`.

Plural `mechanisms` is recognized alongside singular `mechanism`. Quantitative
stems such as `how much`, `how many`, and `how long` are classified as direct
facts before the general `how` mechanism rule, keeping QRNN-01 correctly routed.

## Full before/after audit

`Original` is the output before any attribution normalization. `Title-only` is
the intermediate `In '<title>',` implementation. `Generalized` is the final
output in this pass.

| Question | Original | Title-only | Generalized | Intended |
|---|---|---|---|---|
| `controlled-submodular-01` | `direct_fact` | `direct_fact` | `direct_fact` | direct fact |
| `controlled-submodular-02` | `causes_evidence` | `causes_evidence` | `causes_evidence` | causal/evidence |
| `controlled-enquirer-01` | `direct_fact` | `direct_fact` | `direct_fact` | direct fact |
| `controlled-enquirer-02` | `direct_fact` | `mechanism` | `mechanism` | mechanism |
| `controlled-qrnn-01` | `direct_fact` | `direct_fact` | `direct_fact` | quantitative direct fact |
| `controlled-qrnn-02` | `direct_fact` | `mechanism` | `mechanism` | mechanism |
| `controlled-nsm-01` | `direct_fact` | `direct_fact` | `direct_fact` | direct fact |
| `controlled-nsm-02` | `direct_fact` | `mechanism` | `mechanism` | mechanism |
| `controlled-video-01` | `direct_fact` | `direct_fact` | `mechanism` | mechanism |
| `controlled-video-02` | `direct_fact` | `direct_fact` | `causes_evidence` | causal/evidence |
| `controlled-reddit-01` | `direct_fact` | `direct_fact` | `direct_fact` | direct fact |
| `controlled-reddit-02` | `limitations_future_work` | `limitations_future_work` | `limitations_future_work` | limitations/future work |

All 12 generalized outputs match the intended structures. Enquirer-02,
QRNN-02, and NSM-02 remain correctly classified; Video-01 and Video-02 are now
corrected without regressions.

## Regression coverage

Tests cover every leading attribution shape present in the controlled set, a
shape-equivalent `As described in` form, singular and plural mechanism wording,
the five supported output structures, the exact canonical QRNN-02 string, and
an exact expected classification map for all 12 controlled questions. Before
the generalized fix, four tests failed: the `According to` causal case, the
shape-equivalent `As described in` case, plural `mechanisms`, and the all-12
audit assertion.
