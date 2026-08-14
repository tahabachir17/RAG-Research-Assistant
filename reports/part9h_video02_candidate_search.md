# Part 9h follow-up: Video-02 second causal-example search

## Method

The 75-question run's recorded question order was joined to the existing QASA
and QASPER assets. Questions with literal causal/reason/failure wording were
inspected, including compound and conditional forms. Quantitative/comparative
questions routed as causal by the heuristic were retained in the table so the
literal-scope rejection is visible. The controlled set was separately checked
and contributed Submodular-02 and Video-02.

Criteria columns are:

- **C1:** already-existing evaluation asset;
- **C2:** literal wording asks for a cause, reason, or failure mode;
- **C3:** explicit contextual cause was omitted and is traceable as an unmatched
  authored required concept;
- **C4:** an existing generation output demonstrates the omission.

## Candidate table

| Question ID | C1 | C2 | C3 | C4 | Verdict / evidence |
|---|---|---|---|---|---|
| qasa-1907.11692-16 | Pass | Pass | Fail | Pass | Generated in the smoke and sparse-baseline reports, but the asset has no `required_concepts`; concept recall is null, so no auditable unmatched causal concept exists. |
| qasa-1804.06655-10 | Pass | Pass | Fail | Pass | Generated in the smoke report, but required concepts and concept recall are absent. |
| qasa-1612.08242-9 | Pass | Pass | Fail | Fail | Asks for the reason for exponential `Bw`/`Bh`; no authored concept trace and no generated answer in the 75-question checkpoint. |
| qasa-1502.04681-11 | Pass | Pass | Fail | Fail | Literal “why” question; no concept trace and no prior generated answer. |
| qasa-2208.01626-15 | Pass | Pass | Fail | Fail | Asks whether a stated diffusion-step reason is correct; no concept trace and no prior generated answer. |
| qasa-1704.07813-15 | Pass | Pass | Fail | Fail | Asks why the model suffers on close objects, but its reviewed reference says the reason is not discussed; there is no explicit contextual cause or prior answer. |
| qasa-2202.03036-17 | Pass | Pass | Fail | Fail | Asks why RWPE was chosen; no concept trace and no prior generated answer. |
| qasa-1605.06409-19 | Pass | Pass | Fail | Fail | Asks why R-FCN could not converge; no concept trace and no prior generated answer. |
| qasa-2108.13530-9 | Pass | Pass | Fail | Fail | Asks why the model is described as end-to-end; no concept trace and no prior generated answer. |
| qasa-1704.04861-3 | Pass | Pass | Fail | Fail | Compound question asks what distillation is and why it is used; no concept trace and no prior generated answer. |
| qasa-1508.06615-9 | Pass | Pass | Fail | Fail | Asks why a particular LSTM input was used; no concept trace and no prior generated answer. |
| qasa-2212.10560-7 | Pass | Pass | Fail | Fail | Asks why instruction quality/usefulness was not evaluated; no concept trace and no prior generated answer. |
| qasa-1801.06146-7 | Pass | Pass | Fail | Fail | Literal “why” question, although its reference rejects the premise; no unmatched concept trace or prior answer. |
| qasa-2301.00122-16 | Pass | Pass | Fail | Fail | Conditionally asks why more epochs would not improve validation accuracy; no concept trace and no prior generated answer. |
| qasa-2210.01504-8 | Pass | Pass | Fail | Fail | Asks why established privacy metrics are not used; no concept trace and no prior generated answer. |
| qasper-73738e42d488b32c9db89ac8adefc75403fa2653 | Pass | Fail | Fail | Fail | Asks how much improvement, not why; no prior generated answer. |
| qasper-234ccc1afcae4890e618ff2a7b06fc1e513ea640 | Pass | Fail | Fail | Fail | Asks how large an improvement is, not for its cause; no prior generated answer. |
| qasper-1d6c42e3f545d55daa86bea6fabf0b1c52a93bbb | Pass | Fail | Fail | Fail | Asks whether objectives perform better, not why; no prior generated answer. |
| controlled-submodular-02 | Pass | Pass | Fail | Pass | Previously generated, but final alias-aware manual scoring credits every required concept (`1.0`); there is no omitted causal concept. |
| controlled-video-02 | Pass | Pass | Pass | Pass | Existing Part 9g answer omits the explicit incoherent fusion of temporally distinct events/objects; this is the seed example, not a second candidate. |

## Result

**No valid second causal example was found.** Video-02 remains the only question
that passes all four criteria. The QASA/QASPER candidates cannot establish the
pattern because they have no authored `required_concepts` or deterministic
concept-recall traces; only two had prior generated answers, and neither has an
auditable unmatched concept. The other controlled causal question is complete
under the final alias-aware scorer.

Video-02 therefore remains a documented single-example open gap. The causal
fragment stays unchanged until a real second example surfaces.

No new question, generation call, judge call, golden edit, prompt change, or
source-code change was made for this search.
