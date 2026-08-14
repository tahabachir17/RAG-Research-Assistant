# Part 9h omission root-cause analysis

## Scope

This pass classified the five Part 9g omissions against the literal wording of
their questions, edited the golden set for the over-scoped case, and tested only
the affected direct-fact and mechanism questions. It did not run the full
12-question set, RAGAS, an LLM judge, or the three-run protocol. No routing,
evidence-packing, retrieval, ingestion, shared completeness preamble, causal
instruction, API, frontend, or CI code changed.

## Question-versus-concept classification

| Question | Literal scope test | Classification | Result |
|---|---|---|---|
| Enquirer-01 | “What does the model execute natural-language queries against to obtain answers?” asks for the object against which queries execute. `knowledge-base tables` answers that. The additional fact that the returned answers are table values is true, but answers a different question about output form. | **A — required-concept over-scope** | **Resolved by rescoping:** removed `table values as answers` from `required_concepts`; no generation-code change was made for this finding. |
| Enquirer-02 | “How does the model handle multi-step compositional queries without predefined logical operations?” asks for the process. Intermediate annotations stored in layered memory are one of the process's explicit state-transfer mechanisms, not optional background. | **B — genuine completeness gap** | **Original omission resolved:** the strengthened mechanism-only process trace produced annotations encoding intermediate results “stored in the memory of each layer.” |
| QRNN-01 | “How much faster can QRNNs be than stacked LSTMs?” asks for a speed claim. `at train and test time` is the source's directly attached scope for that measurement; omitting it leaves the comparison underspecified. | **B — genuine completeness gap** | **Resolved:** the direct-fact-only qualifier instruction produced “up to 16 times faster ... at train and test time”; concept recall is `1.0`. |
| NSM-02 | “How is training stabilized when only weak supervision is available?” asks how the stated training procedure works. The context presents REINFORCE's direct optimization of task reward as the purpose of that named training mechanism, alongside pseudo-gold bootstrapping. | **B — genuine completeness gap** | **Still open:** both targeted mechanism regenerations continued to name REINFORCE without its task-reward objective. No question-specific instruction was added. |
| Video-02 | “Why can averaging all frame features be inadequate for video description?” directly asks for the failure modes. Incoherent fusion of temporally distinct events and objects is an explicit reason in the supplied context. | **B — genuine completeness gap** | **Still open pending a second causal example:** this is the only causal question exhibiting the pattern, so the causal fragment was not changed and Video-02 was not regenerated. |

## Golden-set rescope

Enquirer-01 now requires only `knowledge-base tables`. The previous reference
answer may remain more informative than the minimum required answer, but the
deterministic completeness contract no longer penalizes a response for omitting
an output detail the question did not request. Its unchanged targeted answer
scores `1.0` after the rescope.

Because this leaves one correctly scoped concept, the controlled-golden
validation test now permits one to five required concepts rather than requiring
an artificial minimum of two.

## Type-isolated prompt tests

### Direct fact

The direct-fact fragment retains its short-answer requirement and now adds:

> Keep it short, but include any qualifying detail the context directly attaches
> to the requested fact, such as a condition, scope, or comparison.

Only Enquirer-01 and QRNN-01 were regenerated in the initial targeted run.
Enquirer-01 remained concise and correctly answered the now-rescoped question.
QRNN-01 changed from:

> QRNNs can be up to 16 times faster than stacked LSTMs.

to:

> QRNNs can be up to 16 times faster than stacked LSTMs at train and test time.

Both QRNN concepts match their primary phrases, raising its score from `0.5` to
`1.0`. This supports the hypothesis that the former “short, direct answer”
instruction suppressed a directly attached scope qualifier.

### Mechanism

Enquirer-02 and NSM-02 provided two independent mechanism examples, so a
mechanism-only process-completeness addition was justified. The first narrow
version did not recover either target omission. It was tightened, still only in
the mechanism fragment, to require a context-supported trace through
intermediate results and to state stored state and an explicitly specified
optimization objective.

Only Enquirer-02 and NSM-02 were regenerated after that tightening.

- Enquirer-02 now states that executors output annotations encoding intermediate
  results which are stored in each layer's memory. The exact answer phrasing was
  added as a bounded alias for `intermediate table annotations in layered
  memory`. That target concept is now credited by alias. Its answer scores
  `0.75`: the new sample stochastically omitted `operations conditioned on the
  query`, which remains uncredited.
- NSM-02 still says only that training uses REINFORCE, pseudo-gold programs, an
  iterative ML process, and bootstrapping. It does not say that REINFORCE
  optimizes task reward, so the concept remains uncredited and the score remains
  `0.75`.

The mechanism change is retained because it recovered the layered-state failure
seen in one of the two confirming examples. It is not treated as a solution for
NSM-02, and no stronger blanket or question-specific instruction was added.

### Causal

Video-02 is the sole causal example with this omission pattern. Generalizing a
causal instruction from one sample would violate the stated calibration rule,
so the causal fragment is unchanged. The fusion omission remains open until a
second causal example confirms the pattern or a separately scoped decision is
made.

## Deterministic matching and over-crediting

The targeted Enquirer-02 answer also used `series of executors` and a long-form
end-to-end/query-answer paraphrase. These exact, concept-specific strings were
added as aliases, keeping every concept below the five-alias cap. No fuzzy,
semantic, embedding, or LLM matching was introduced.

The Part 9f/9g incomplete-answer checks still preserve their missing concepts:

- the generic old QRNN-02 summary remains `0.0`;
- the old Enquirer-02 partial answer now receives credit only for its genuinely
  present stacked-executor and end-to-end-training concepts (`0.5`), while
  conditioned operations and layered intermediate state remain unmatched;
- the old Video-02 answer remains `0.667`, with incoherent event/object fusion
  unmatched.

The new Enquirer-02 mechanism retry scores `0.75`; the newly recovered layered
state is credited, while the concept absent from that answer remains uncredited.

## Artifacts and decision

Targeted artifacts:

- `evaluation/data/eval_results/controlled_generation_part9h_targeted_20260814/report.json`
- `evaluation/data/eval_results/controlled_generation_part9h_mechanism_retry_20260814/report.json`

**The system remains NO-GO for the full 12-question verification pass and the
three-run protocol.** Enquirer-01 was resolved by rescoping, QRNN-01 was resolved
by the direct-fact qualifier instruction, and the original Enquirer-02 omission
was recovered. NSM-02 remains an open mechanism completeness failure, and
Video-02 remains open pending a second confirming causal example.
