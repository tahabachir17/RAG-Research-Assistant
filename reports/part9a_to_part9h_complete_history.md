# Part 9a–9h complete interaction, implementation, diagnostics, and evaluation history

Date consolidated: 2026-08-14  
Repository: `RAG-AI_Reasearch_Papers`  
Current decision: **NO-GO for the three-run controlled-generation protocol**

## 1. Purpose and provenance

This document consolidates the work from Part 9a through the latest Part 9h
follow-up into one auditable narrative. It covers:

- the requests and decisions made during the interaction;
- implementation changes and regression coverage;
- retrieval, generation, RAGAS, concept-recall, and prompt diagnostics;
- live and offline runs, including what data was sent externally;
- the controlled 12-question results and per-question failure analysis;
- every durable report and important machine-readable artifact;
- resolved issues, rejected hypotheses, and remaining blockers.

Parts 9c–9h are supported by direct reports and the interaction history. The
exact conversational prompts for Parts 9a–9b are not present in the current
task context; those two sections are reconstructed from Git commits, run
artifacts, README documentation, and the Part 9c audit's description of the
earlier work. They should be read as an evidence-backed project history, not a
verbatim transcript. From Part 9d onward, the sequence of user requests is
preserved explicitly below.

No three-run controlled protocol has been executed. The user authorized the
public ArXiv-derived controlled data flow to Groq and the configured
Gemini-compatible judge, but later prompts repeatedly and explicitly required
pre-flight checks and targeted fixes before spending those runs.

## 2. Executive summary

The Part 9 work evolved from building an evaluation foundation into separating
three different sources of apparent answer-quality failure:

1. **Retrieval failure:** the answer-bearing chunk never reaches generation.
2. **Generation completeness failure:** the evidence is present, but the answer
   omits an in-scope concept.
3. **Measurement failure:** the answer conveys a concept, but an LLM judge or a
   brittle deterministic matcher fails to credit it.

The most important findings are:

- The reviewed 75-question external benchmark contains 55 QASA and 20 QASPER
  questions. Production-style hybrid retrieval is operational, but top-four
  evidence coverage is weak, particularly for QASPER.
- At top four, the Part 9c hybrid-plus-reranker path achieved overall recall
  `0.2486` and Hit@4 `0.3600`; QASPER Hit@4 was only `0.2000`.
- The reranker is the dominant latency cost and can demote QASPER evidence.
  MMR at small cutoffs was actively harmful and is now bypassed at the
  generator's top-four boundary.
- A separate 12-question controlled benchmark with frozen reviewed evidence
  isolated generation from retrieval. Its baseline RAGAS run was perfectly
  faithful but incomplete: answer correctness was `0.6214`.
- The first completeness remediation moved correctness only to `0.6264`, well
  inside observed judge noise. Two unchanged answers moved by approximately
  `±0.25`, demonstrating why RAGAS correctness could not be trusted alone.
- `concept_recall` existed in code but all 12 live golden records initially had
  empty `required_concepts`, so the deterministic cross-check returned null.
- All controlled questions now have reviewed concept requirements. Matching
  supports at most five hand-authored aliases per concept and records whether a
  match came from the primary phrase or an alias. It remains deterministic
  normalized-substring matching—no embeddings, fuzzy distance, or LLM scoring.
- QRNN-02 exposed a chain of independent failures: title-prefix
  misclassification, insufficient single-chunk evidence, prompt adherence, and
  then matcher brittleness. Each layer was diagnosed separately and fixed with
  scoped tests.
- The first full post-fix 12-question generation pass reached mean
  alias-aware concept recall `0.8472`, versus `0.7014` when the previous live
  answers were rescored with the same final matcher.
- Five remaining omissions were root-caused. Enquirer-01 was an over-scoped
  golden concept; QRNN-01 and the original Enquirer-02 omission were resolved
  by type-specific prompt changes; NSM-02 and Video-02 remain open.
- NSM-02's context was verified as **Explicit**: the supplied chunk directly
  links REINFORCE to optimizing task reward. It is not a retrieval gap.
- A search of the existing 75-question and controlled assets found no valid
  second causal example for Video-02. The causal fragment therefore remains
  unchanged rather than being generalized from one sample.

Current blocking state:

- **NSM-02:** genuine mechanism completeness gap remains open.
- **Video-02:** genuine causal completeness gap remains open, but it is still a
  single-example pattern and has not justified a causal prompt change.
- **Three-run protocol:** not started.

## 3. Evaluation architecture established across Parts 9a–9c

The early Part 9 work established a provider-neutral generation and evaluation
stack with the following capabilities:

- Groq/OpenAI-compatible provider routing and bounded rate-limit retries;
- structured JSON answer contracts and schema validation;
- deterministic citation validation and claim-level citation coverage;
- finish-reason and truncation metadata;
- one bounded repair attempt for invalid output;
- optional faithfulness verification;
- resumable per-question generation checkpoints;
- resumable, per-question/per-metric RAGAS caches;
- external QASA/QASPER benchmark construction and evidence alignment;
- separate manual, QASA, and QASPER reporting layers;
- dense, sparse, reciprocal-rank fusion, cross-encoder reranking, and MMR
  diagnostics;
- deterministic generation metrics independent of RAGAS.

Relevant foundation commits include:

| Commit | Main contribution |
|---|---|
| `7ad9070` | Provider routing, rate limiting, structured answers, stronger generation validation, and expanded evaluation orchestration. |
| `c9633e3` | Additional generation evaluator, judge, RAGAS, structured-answer, and chunker hardening. |
| `7de59ea` | Citation and faithfulness improvements plus the isolated context-count experiment. |
| `d8c1a15` | Atomic generation checkpoints, resumability, and larger evaluator/RAGAS reliability changes. |
| `2318973` | External QASA/QASPER assets, full RAGAS runner, layered dataset reporting, retrieval-stack diagnostics, judge diagnostics, and reranker profiling. |

## 4. Part 9a — evaluation foundation (reconstructed)

Part 9a's durable outcome was the transition from a simple generation test into
a robust, provider-neutral evaluation system. The implementation added:

- normalized generation result records;
- structured narrative and tabular answer parsing;
- retry-aware provider clients;
- offline generation-quality metrics;
- RAGAS integration that can rescore saved generation outputs;
- separation between generation cost and judge cost;
- citation-grounding and validation signals;
- checkpointed execution so long live runs could resume.

The context-count diagnostic tested 4, 5, and 8 chunks on one question. Its
Gemini-scored results were:

| Chunks | Faithfulness | Answer relevancy | Context utilization |
|---:|---:|---:|---:|
| 4 | 0.9231 | 0.4120 | 1.0000 |
| 5 | 1.0000 | 0.4066 | 0.7000 |
| 8 | 1.0000 | 0.4120 | 0.7000 |

This single unreviewed example did not justify changing the production context
count. It did establish the discipline later used throughout Part 9: preserve
the default, isolate one variable, and require reviewed multi-question evidence
before changing global behavior.

## 5. Part 9b — external benchmark and retrieval-stack diagnostic (reconstructed)

Part 9b built and inspected the reviewed external benchmark:

- 55 QASA questions;
- 20 QASPER questions;
- 75 reviewed questions total;
- 2,777 aligned external chunks in the benchmark Qdrant collection;
- SciDQA normalized to 2,937 rows but emitted zero evidence-grounded evaluation
  records because the source export lacked evidence passages.

The retrieval diagnostic compared dense, sparse BM25, hybrid RRF,
hybrid-plus-reranker, and hybrid-plus-reranker-plus-MMR configurations. It also
recorded per-query ranks and failure correlations.

The Part 9c audit later quantified two important Part 9b observations:

- MMR after reranking reduced reviewed Recall@4 from `0.4571` to `0.1800`
  (`-60.6%`) and Recall@8 from `0.6146` to `0.3108` (`-49.4%`).
- The historical 50-candidate cross-encoder path raised mean latency to roughly
  `28.44 s/query`, with reranking representing about `99.6%` of retrieval
  latency.

The judge-reliability diagnostic also separated malformed/truncated judge
responses from valid but semantically noncommittal responses. This distinction
later drove JSON-schema validation, token-cap escalation, and the policy that
answer relevancy remain diagnostic until calibrated.

## 6. Part 9c — audit and eight remediation items

### 6.1 Phase 0 audit-before-edit

The audit checked six suspected failure areas before changing code:

| Area | Finding |
|---|---|
| Qdrant collection | Both external collections existed with 2,777 points; current dense retrieval fails loudly rather than silently falling back. |
| Judge schema anomalies | One case combined truncation and schema mismatch; three hard-zero relevancy cases were valid noncommittal judgments, not malformed JSON. |
| Full retrieval stack | The 75-question ablation exercised dense, sparse, RRF, reranking, and MMR, but the earlier smoke path was BM25-only and the full runner did not apply MMR at the real top-four boundary. |
| Small-k MMR | Confirmed harmful at k=4 and k=8. |
| Reranker latency | Confirmed as the dominant cost. |
| Corpus limitations | `qasper-bf0080...` is upstream-unanswerable, and SciDQA lacks grounding evidence; neither is truthfully fixable in the evaluation adapter. |

### 6.2 The eight Part 9c implementation items

The repository commit sequence records the eight scoped remediations:

1. **Complete evidence-backed answers** (`4f80c8e`)  
   Added a shared completeness instruction requiring all supported mechanisms,
   reasons, limitations, and future-work items rather than a vague summary.

2. **Deterministic required-concept coverage** (`e46fe8d`)  
   Added `concept_recall` to score required concept phrases independently of an
   LLM judge.

3. **Question-type routing** (`51f3953`)  
   Added distinct answer fragments for direct facts, mechanisms,
   causes/evidence, limitations/future work, and comparisons.

4. **Multi-concept evidence-packing comparison** (`955dae6`)  
   Added gold/adjacent/section packing infrastructure and tests.

5. **Balanced golden distribution** (`5d83172`)  
   Added fixtures and validation for the intended question mix.

6. **Answer-relevancy calibration** (`b0a2b86`)  
   Kept answer relevancy diagnostic because its generated-question/embedding
   method did not align reliably with human inspection on short correct answers.

7. **Disputed-correctness adjudication** (`a00c918`)  
   Added adjudication only for genuinely disputed correctness scores rather than
   using blanket secondary judgment.

8. **Layered reporting** (`47b61cb`)  
   Separated dataset tiers and deterministic, judged, and retrieval signals in
   final reports.

### 6.3 Retrieval profiling and live local scoring

Part 9c profiled the current 20-candidate reranker and a 10-candidate
alternative:

| Candidate k | Recall@4 | Mean rerank | p50 | p95 |
|---:|---:|---:|---:|---:|
| 10 | 0.2841 | 570.2 ms | 530.9 ms | 857.2 ms |
| 20 | 0.2486 | 991.3 ms | 973.0 ms | 1,244.7 ms |

Candidate-k 10 reduced median reranking latency by `45.4%` and did not lose
Recall@4 on frozen candidates, but the default remained 20 pending a fresh
end-to-end validation.

The production-style local retrieval run over all 75 reviewed questions used
dense plus BM25 RRF, 20 reranker candidates, and four final contexts. MMR was
configured but bypassed at k=4:

| Tier | N | Recall@4 | Precision@4 | Hit@4 | MRR | nDCG@4 | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 75 | 0.2486 | 0.0933 | 0.3600 | 0.2400 | 0.2136 | 868.9 ms | 1,117.3 ms |
| QASA | 55 | 0.2965 | 0.1091 | 0.4182 | 0.2985 | 0.2655 | 810.8 ms | 1,180.5 ms |
| QASPER | 20 | 0.1167 | 0.0500 | 0.2000 | 0.0792 | 0.0710 | 931.1 ms | 1,043.9 ms |

Only 27 of 75 questions retrieved any reviewed evidence in the top four. The
full 75-question generation/RAGAS run was not completed: the first Groq calls
failed with connection errors, and a later unsandboxed probe was denied because
third-party data transfer had not yet been explicitly authorized. The run
stopped with zero generated answers rather than presenting partial metrics.

## 7. Part 9d/9e retrieval diagnostics

Two retrieval-focused reports followed the Part 9c score analysis.

### 7.1 QASPER evidence coverage

The evidence audit checked whether missed QASPER gold text existed in the
processed corpus. For the reviewed misses, evidence generally existed but was
not retrieved into the top four. This ruled out a broad ingestion-loss theory
and focused attention on candidate generation and ranking.

### 7.2 Retriever configuration audit

The run path was traced end to end:

`MMR wrapper → reranker → hybrid RRF → dense + sparse`

There was no silent BM25 fallback. Dense and sparse search both returned
results, the cross-encoder reranked 20 candidates, and MMR was intentionally
bypassed because the final request was top four.

### 7.3 QASPER stage ablation

Of 16 QASPER top-four misses:

- 11 were candidate-generation misses;
- 5 were reranker-demotion misses.

At k=4, BM25 outperformed hybrid RRF and hybrid-plus-rerank on QASPER. At k=8,
BM25 and hybrid RRF tied on recall (`0.2667`), while BM25 retained better Hit,
MRR, nDCG, and latency. The diagnostic made no retrieval-code changes.

## 8. Controlled retrieval and generation baselines

### 8.1 Controlled retrieval benchmark

A 12-question controlled benchmark was authored across six papers already in
the 1,195-paper production corpus. Each question includes the paper title and
exact answer-bearing chunk IDs.

Production BM25 results were:

| Group | N | Recall@4 | Hit@4 | MRR | nDCG@4 | Recall@8 | Hit@8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 12 | 0.7917 | 0.8333 | 0.4125 | 0.4906 | 0.9167 | 0.9167 |
| Easy | 6 | 0.7500 | 0.8333 | 0.4306 | 0.5043 | 0.8333 | 0.8333 |
| Moderate | 6 | 0.8333 | 0.8333 | 0.3944 | 0.4769 | 1.0000 | 1.0000 |

This showed that the retriever is not generally nonfunctional. The original
external questions often assume a known paper and are underspecified for
open-corpus retrieval; controlled title-aware questions perform much better.

### 8.2 Controlled frozen-context RAGAS baseline

The 12 controlled questions were then answered from exact reviewed evidence,
bypassing retrieval. Generator and judge configuration:

- Groq `llama-3.3-70b-versatile` generator;
- Gemini `gemini-3.5-flash-lite` primary RAGAS judge;
- Groq `llama-3.1-8b-instant` fallback only for failed/missing scores;
- zero fallback calls in the completed baseline.

| Metric | Mean |
|---|---:|
| Faithfulness | 1.000 |
| Answer relevancy | 0.6879 |
| Context precision | 1.000 |
| Context recall | 1.000 |
| Answer correctness | 0.6214 |

The answers were grounded but incomplete. Moderate-question correctness was
`0.531`, versus `0.712` for easy questions. QRNN-02, Submodular-02, Video-02,
and Enquirer-02 were the most actionable completeness misses.

### 8.3 First completeness remediation and judge-noise finding

The follow-up controlled run moved answer correctness only from `0.621435` to
`0.626357` (`+0.004923`). Faithfulness stayed `1.0`; answer relevancy moved
from `0.687934` to `0.673252`.

The per-question deltas were unstable. QRNN-02's unchanged answer gained
`+0.25`, while Reddit-01's unchanged answer lost `-0.225`; QRNN-01 lost
`-0.394`. This established that a single small-sample RAGAS correctness delta
could not demonstrate improvement. It motivated the deterministic concept
cross-check and the eventual three-run mean-and-spread protocol.

## 9. Interaction chronology from the Part 9d implementation request onward

This section records the sequence of user requests and the corresponding
actions/outcomes.

The explicit interaction ledger in this task was:

| Order | User request |
|---:|---|
| 1 | **Implementation Prompt — Part 9d: Populate `required_concepts` + Debug QRNN-02** |
| 2 | **Approve Live Runs + Pre-Flight Checks** |
| 3 | **Fix Classifier Title-Prefix Bug (blocks Part 9d live runs)** |
| 4 | **Generalize Prefix Fix + Verify QRNN-02 Evidence Text** |
| 5 | **Verify Context Delivery + Targeted Prompt Fix, Single-Question Test Only** |
| 6 | **Calibrate `concept_recall` Matching** |
| 7 | **Full 12-Question Single Pass (last step before the three-run protocol)** |
| 8 | **Root-Cause the Five Remaining Omissions** |
| 9 | Clarify the meanings of QRNN, NSM, and the Video benchmark shorthand. |
| 10 | **Part 9h follow-up: NSM-02 context-sufficiency check and Video-02 second-example criteria** |
| 11 | The same Part 9h follow-up was attached again; SHA-256 equality was confirmed and duplicate work was avoided. |
| 12 | Create this consolidated Part 9a-to-current Markdown history. |

### 9.1 Populate `required_concepts` and debug QRNN-02

The user identified that every controlled golden question had
`required_concepts: []`, meaning the newly added scorer never ran. The request
required 2–5 context-verified concepts per question and a root-cause diagnosis
for QRNN-02 before changing prompts.

Outcome:

- all 12 controlled records received reviewed required concepts;
- concepts were checked against production chunk text;
- QRNN-02 was found to have both a routing problem and a single-chunk evidence
  gap;
- no three-run protocol was started.

### 9.2 Approve live runs, but require two pre-flights

The user explicitly approved the controlled public-ArXiv data flow to Groq and
the Gemini-compatible judge, but required:

1. proof that `concept_recall` varies with content;
2. a regenerated QRNN-02 answer showing old context → old answer → new context
   → new answer.

This approval increased authorized request volume but did not remove the
pre-flight gates.

### 9.3 Fix the title-prefix classifier bug

The canonical QRNN question began with `In '<paper title>', ...`, causing the
classifier to return `direct_fact`; the bare question correctly returned
`mechanism`. The user required a general fix and canonical regression cases for
all five question types.

Outcome:

- leading quoted-title attribution clauses are ignored before classification;
- canonical cases were added for mechanism, direct fact, causal,
  limitations, and comparison;
- all 12 questions were audited.

### 9.4 Generalize beyond literal prefixes and plural forms

Further testing found Video-01 (`mechanisms`) and Video-02 (`According to ...,
why`) still misrouted. The user correctly rejected one-string-at-a-time fixes.

Outcome:

- attribution stripping now handles any leading non-question clause containing
  a quoted title and ending at its comma;
- plural `mechanisms` and singular/plural limitation forms are handled;
- quantitative `how much/many/long/fast/often` questions route to direct fact
  before the general `how` mechanism rule;
- all 12 classifications now match intent.

The final before/after audit identified originally silent misroutes for
Enquirer-02, QRNN-02, NSM-02, Video-01, and Video-02.

### 9.5 Verify QRNN evidence text directly

The user required raw excerpts rather than relying on concept scores.

Findings:

- gold chunk `86e4d367-e844-569c-9afa-c41c7768edaa` contains the equivalent of
  parallel convolutional computation and long-distance recurrent context;
- adjacent chunk `853684bd-7f1a-5c6b-98ee-ca11177821fd` literally contains
  `recurrent pooling layer`;
- both concepts were extractable after adjacent packing.

### 9.6 Verify final context delivery, then make a mechanism-only prompt change

The exact assembled prompt boundary was checked. Both QRNN chunks reached Groq
in full, in the intended order, within a 2,500-token budget. The assembled
context used only 480 whitespace tokens; no truncation, deduplication, or
reordering occurred.

Only the mechanism fragment was strengthened to require named technical terms
and individually named mechanisms. The shared completeness preamble and all
other question types remained unchanged.

QRNN-02 moved from a vague answer to one naming:

- `convolutional aspects computed in parallel`;
- `recurrent pooling layer`;
- `long-distance context`.

Raw concept recall moved from `0.3333` to `0.6667` because the deterministic
matcher did not recognize the convolution paraphrase.

### 9.7 Calibrate concept-recall aliases

The user explicitly prohibited fuzzy, embedding, semantic, or LLM matching and
requested bounded hand-authored aliases.

Outcome:

- a concept may be a string or `{concept, aliases}` object;
- aliases are capped at five;
- primary and alias matches use the same punctuation-normalized,
  case-folded substring rule;
- `concept_recall_details` records match source and phrase;
- the QRNN answer rescored from `0.6667` to `1.0`;
- the old generic `parallelism and context` answer remained `0.0`.

### 9.8 Full 12-question single pass

The user authorized one full generation pass—but explicitly not the three-run
protocol. A dedicated checkpointed generation-only runner was added so this
verification could not accidentally invoke RAGAS judges.

All 12 Groq generations completed. Transient token-per-minute 429s were handled
within bounded retries. Eighteen additional aliases were added across nine
questions only where manual review confirmed a real paraphrase.

Final Part 9g table:

| Question | Previous answer recall | New answer recall | Previous RAGAS correctness | Material change |
|---|---:|---:|---:|---|
| Submodular-01 | 1.000 | 1.000 | 0.581 | No |
| Submodular-02 | 1.000 | 1.000 | 0.234 | No |
| Enquirer-01 | 0.500 | 0.500 | 0.551 | No |
| Enquirer-02 | 0.250 | 0.750 | 0.556 | Yes |
| QRNN-01 | 0.500 | 0.500 | 0.569 | No |
| QRNN-02 | 0.000 | 1.000 | 0.405 | Yes |
| NSM-01 | 1.000 | 1.000 | 0.862 | No |
| NSM-02 | 0.750 | 0.750 | 0.748 | No |
| Video-01 | 0.750 | 1.000 | 0.950 | Yes |
| Video-02 | 0.667 | 0.667 | 0.550 | No |
| Reddit-01 | 1.000 | 1.000 | 0.580 | No |
| Reddit-02 | 1.000 | 1.000 | 0.930 | No |

Mean concept recall was `0.8472` for the new answers versus `0.7014` for the
previous live answers under the same final matcher. Manual review and the final
matcher agreed for all 12.

Five concepts were genuinely absent despite being present in delivered
evidence: Enquirer-01 table values, Enquirer-02 layered annotations, QRNN-01
train/test scope, NSM-02 task reward, and Video-02 incoherent fusion.

### 9.9 Root-cause the five omissions

The user required each omission to be classified as:

- **A: required-concept over-scope**, or
- **B: genuine completeness gap**.

Results:

| Question | Classification | Resolution state |
|---|---|---|
| Enquirer-01 | A | `table values as answers` removed because the literal question asks only what queries run against. Resolved by rescoping. |
| Enquirer-02 | B | Mechanism-only process-trace instruction recovered layered intermediate annotations. Original omission resolved. |
| QRNN-01 | B | Direct-fact qualifier instruction recovered `at train and test time`; recall `0.5 → 1.0`. Resolved. |
| NSM-02 | B | Two targeted mechanism answers still omitted the task-reward objective. Open. |
| Video-02 | B | Only one causal example showed this pattern, so no causal prompt change was made. Open. |

The direct-fact fragment now keeps answers short while requiring any condition,
scope, or comparison directly attached to the requested fact. The mechanism
fragment now asks for a context-supported process trace through intermediate
results and explicitly stated objectives. These additions are type-specific;
the shared preamble was not rewritten.

The targeted run generated only Enquirer-01, QRNN-01, Enquirer-02, and NSM-02.
A second mechanism-only retry generated only Enquirer-02 and NSM-02. No judge
calls or full-set run occurred.

### 9.10 NSM-02 context-sufficiency follow-up

Both Part 9h NSM runs used the same recorded context IDs:

1. `26af9bea-0d86-5b50-b866-7e4617e98db4`;
2. `cfbf42cc-a301-5f3a-987b-6bd384f5af23`.

The first chunk says, in one clause, that the method applies REINFORCE “to
directly optimize the task reward.” The raw PDF extraction transmits the token
as `REIN-\nFORCE`, so a literal contiguous search for `REINFORCE` is false until
the line-break hyphen is normalized. The semantic relationship is nevertheless
direct and explicit.

Bucket: **Explicit**.  
Routing decision: generation-completeness gap; no retrieval/context change.

No third NSM regeneration was authorized or attempted.

### 9.11 Video-02 second-example search

The user required a real existing example satisfying all four criteria:

1. existing asset;
2. literal causal wording;
3. explicit contextual reason omitted and traceable as an unmatched authored
   concept;
4. existing generated answer demonstrating the omission.

Eighteen external candidate questions plus the two controlled causal candidates
were inspected. Only Video-02 itself passed all four criteria. External
QASA/QASPER candidates lacked authored required-concept traces; most also had no
prior generated answer. Controlled Submodular-02 was already complete after
alias calibration.

No second causal example was found. No synthetic question was created, no model
call was made, and the causal fragment remains unchanged.

### 9.12 Duplicate follow-up attachment

The same NSM/Video follow-up prompt was attached a second time. The two files had
the same SHA-256 hash. The completed investigation was not repeated, and no
additional changes or calls were made.

### 9.13 Terminology clarification requested during the interaction

- **QRNN:** Quasi-Recurrent Neural Network, combining parallel convolutional
  computation with recurrent pooling for sequence context.
- **NSM:** Neural Symbolic Machine, combining a neural sequence-to-sequence
  programmer with key-variable memory and a symbolic Lisp interpreter.
- **Video:** shorthand in this benchmark for *Describing Videos by Exploiting
  Temporal Structure*, which uses a 3-D CNN for local temporal structure and
  temporal attention for globally relevant segments.

## 10. Final question-type routing map

The generalized classifier's intended controlled-set routing is:

| Question | Type |
|---|---|
| controlled-submodular-01 | direct fact |
| controlled-submodular-02 | causes/evidence |
| controlled-enquirer-01 | direct fact |
| controlled-enquirer-02 | mechanism |
| controlled-qrnn-01 | quantitative direct fact |
| controlled-qrnn-02 | mechanism |
| controlled-nsm-01 | direct fact |
| controlled-nsm-02 | mechanism |
| controlled-video-01 | mechanism |
| controlled-video-02 | causes/evidence |
| controlled-reddit-01 | direct fact |
| controlled-reddit-02 | limitations/future work |

Attribution shapes covered by regression tests include `In '<title>',`,
`According to '<title>',`, `According to the paper '<title>',`, and a
shape-equivalent `As described in '<title>',` form.

## 11. Concept-recall design and safeguards

The final deterministic scorer:

1. converts non-alphanumeric characters to spaces;
2. case-folds;
3. collapses whitespace;
4. checks the normalized primary phrase as a substring;
5. if needed, checks up to five authored aliases in order.

It does **not** perform stemming, token reordering, fuzzy distance, embedding
similarity, or LLM judgment.

Each detail row records:

- canonical concept;
- matched boolean;
- `primary` or `alias` match type;
- exact authored matching phrase.

Known-incomplete spot-checks protect against over-crediting:

- the old generic QRNN-02 answer remains `0.0`;
- the old Enquirer-02 partial answer receives credit only for genuinely present
  stacked-executor and end-to-end concepts;
- the old Video-02 answer still fails the fusion concept.

## 12. Evidence-packing behavior

The controlled evaluator always starts from reviewed frozen gold chunks.
Mechanism questions may add immediate same-paper adjacent chunks, while
excluding references, bibliography, acknowledgements, and front matter.

This is not retrieval expansion in the controlled benchmark: chunk selection is
deterministic and paper-local. The QRNN diagnosis demonstrated why the packing
trigger matters—the gold conclusion contained convolutional parallelism and
long-distance context, while the adjacent experiments chunk named the recurrent
pooling layer.

No evidence-packing scope was expanded during the later Part 9h work.

## 13. Live-run and data-transfer ledger

The controlled data consists of public ArXiv-derived questions and chunks. The
user explicitly approved transmission to Groq for generation and the configured
Gemini-compatible judge for RAGAS, matching the earlier controlled-run data
flow.

Documented run classes:

| Run | External activity | Outcome |
|---|---|---|
| Controlled baseline RAGAS | 12 Groq generations plus 60 RAGAS metric values | Completed; correctness `0.6214`. |
| Controlled completeness RAGAS | 12 generations plus RAGAS judging | Completed; correctness `0.6264`; judge-noise concern confirmed. |
| Part 9c reviewed 75-question run | Retrieval local; first Groq calls attempted | Stopped with zero generated answers after connection failures/authorization boundary. |
| QRNN pre-flights | One-question Groq regenerations only | Diagnosed routing, evidence, prompt, and matcher layers. |
| Part 9g single pass | 12 Groq generations; no judge | Completed with bounded 429 retries. |
| Part 9h targeted pass | Four Groq generations; no judge | Enquirer-01, Enquirer-02, QRNN-01, NSM-02 only. |
| Part 9h mechanism retry | Two Groq generations; no judge | Enquirer-02 and NSM-02 only. |
| NSM context and Video candidate follow-ups | No model calls | Read-only. |
| Three-run protocol | None | **Not started.** |

## 14. Code and data improvements

The main durable implementation areas are:

| Area | Files | Improvement |
|---|---|---|
| Prompt routing | `generation/prompt_manager.py` | General attribution-prefix handling; plural forms; quantitative direct facts; scoped direct-fact and mechanism completeness instructions. |
| Prompt assembly | `generation/cli.py`, `config/prompts/qa_prompt.yaml` | Injects question-type fragments while preserving the shared completeness preamble. |
| Evidence packing | `generation/context_assembler.py`, `evaluation/evidence_packing.py` | Gold/adjacent/section comparison and adjacent mechanism packing. |
| Golden concepts | `evaluation/data/controlled_generation_qa.json` | Context-reviewed concepts and bounded authored aliases; Enquirer-01 rescope. |
| Concept schema | `evaluation/concept_requirements.py`, `evaluation/generation_golden.py` | String-or-object concept parsing and five-alias validation. |
| Deterministic scoring | `evaluation/generation_metrics.py` | Alias-aware concept recall with auditable match details. |
| Controlled evaluation | `evaluation/run_controlled_generation_ragas.py` | Frozen-context generation plus resumable RAGAS judging. |
| Single-pass safety | `evaluation/run_controlled_generation_single_pass.py` | Generation-only, checkpointed, question-filterable controlled runner with no judge stage. |
| Retrieval diagnostics | `evaluation/run_part9c_retrieval_scores.py`, `evaluation/run_part9e_ablation.py`, `evaluation/audit_part9d_qasper_coverage.py` | Production-path retrieval scores, stage isolation, and evidence coverage. |
| Reporting | `evaluation/full_ragas_evaluation.py`, `evaluation/layered_reporting.py` | Layered deterministic/retrieval/judged reporting and resumable artifacts. |

Regression coverage includes canonical routing for all 12 questions, every
attribution shape, the exact QRNN question, evidence packing, alias caps,
primary-versus-alias traces, over-crediting fixtures, controlled-run defaults,
and repeated question filters.

Test milestones recorded during the interaction:

- 291 tests passed after the early alias/calibration work;
- 293 tests passed after the Part 9g runner and full-set calibration;
- 296 tests passed after the Part 9h scoped prompt and golden changes;
- the final read-only follow-up again passed all 296 unit tests, Ruff, and
  `git diff --check` (with only existing Windows line-ending warnings).

## 15. Naming note: two “Part 9e” strands

Two work streams used the Part 9e label:

1. `reports/part9e_ablation_analysis.md` is the QASPER retrieval-stage ablation
   conducted after Part 9d retrieval audits.
2. Later interaction prompts referred to the QRNN mechanism-specific technical
   term instruction as the Part 9e prompt fix.

They are independent: the former diagnoses retrieval candidate/reranker stages;
the latter changes only generation prompt behavior for mechanism questions.

## 16. Complete report and diagnostic index

| Artifact | Contents |
|---|---|
| `reports/part9c_phase0_audit.md` | Audit-before-edit evidence and root causes. |
| `reports/part9c_latency_report.md` | Reranker k=10/20 latency and recall profile. |
| `reports/part9c_phase6_status.md` | Blocked 75-question generation/RAGAS status and authorization boundary. |
| `reports/part9c_retrieval_score_analysis.md` | Live local 75-question production-style retrieval metrics. |
| `reports/part9d_qasper_evidence_coverage.md` | QASPER gold-text presence versus retrieval misses. |
| `reports/part9d_retriever_config_audit.md` | Runtime proof of dense+sparse+RRF+reranker and MMR bypass. |
| `reports/part9e_ablation_analysis.md` | Candidate-generation versus reranker-demotion analysis and k/config comparison. |
| `reports/controlled_retrieval_benchmark_analysis.md` | Controlled 12-question BM25 benchmark. |
| `reports/controlled_generation_ragas_analysis.md` | Frozen-context controlled RAGAS baseline. |
| `reports/answer_completeness_ragas_overall_20260814.csv` | Baseline/remediation aggregate comparison. |
| `reports/answer_completeness_ragas_question_deltas_20260814.csv` | Per-question correctness deltas and same-answer flags. |
| `reports/part9d_question_type_audit.md` | All-12 before/title-only/generalized routing table. |
| `reports/part9d_qrnn02_diagnosis.md` | QRNN old/new context, answers, final prompt delivery, and targeted prompt result. |
| `reports/part9d_concept_recall_alias_calibration.md` | Alias schema, QRNN rescore, and over-crediting checks. |
| `reports/part9g_controlled_single_pass_analysis.md` | Full single-pass 12-question results and go/no-go. |
| `reports/part9h_omission_root_cause.md` | A/B classification of five remaining omissions and targeted tests. |
| `reports/part9h_nsm02_context_check.md` | Explicit NSM task-reward/REINFORCE context proof. |
| `reports/part9h_video02_candidate_search.md` | Twenty-question causal-candidate criteria table. |

Important machine-readable run directories:

- `evaluation/data/eval_results/retrieval_stack_diagnostic_20260812/`
- `evaluation/data/eval_results/part9c_20260813/`
- `evaluation/data/eval_results/part9c_retrieval_20260813/`
- `evaluation/data/eval_results/part9e_ablation_20260813/`
- `evaluation/data/eval_results/controlled_retrieval_20260813/`
- `evaluation/data/eval_results/controlled_generation_ragas_20260813/`
- `evaluation/data/eval_results/controlled_generation_ragas_completeness_20260814/`
- `evaluation/data/eval_results/controlled_generation_part9f_single_pass_20260814/`
- `evaluation/data/eval_results/controlled_generation_part9h_targeted_20260814/`
- `evaluation/data/eval_results/controlled_generation_part9h_mechanism_retry_20260814/`

## 17. Resolved, rejected, and open hypotheses

### Resolved

- Empty required-concept data: populated and validated.
- Canonical title-prefix routing: generalized and regression-tested.
- Video plural-mechanism and `According to` causal routing: fixed.
- QRNN recurrent-pooling evidence gap: addressed through adjacent packing.
- QRNN final context delivery: verified clean.
- QRNN technical-term omission: resolved by mechanism-only instruction.
- QRNN paraphrase matcher miss: resolved by bounded alias.
- Full-set paraphrase brittleness: calibrated with authored aliases and
  over-crediting checks.
- Enquirer-01 over-scoped golden concept: removed.
- QRNN-01 missing train/test scope: resolved by direct-fact qualifier rule.
- Original Enquirer-02 layered-memory omission: recovered by mechanism process
  tracing, although a later sample omitted a different mechanism concept.

### Rejected or ruled out

- Silent BM25 fallback in the Part 9c retrieval run.
- Missing Qdrant population as the cause of Part 9c scores.
- Broad QASPER ingestion loss; reviewed text generally exists in indexed chunks.
- QRNN prompt-context truncation, reordering, or deduplication.
- Fuzzy/semantic concept scoring as an acceptable solution.
- A global causal prompt change based only on Video-02.
- A synthetic second causal question.
- Partial or single-run RAGAS correctness as sufficient evidence of improvement.

### Still open

1. **NSM-02:** context explicitly states that REINFORCE directly optimizes task
   reward, but repeated targeted answers omit that objective. A decision is
   needed between a third narrowly authorized mechanism attempt and documenting
   the miss as an accepted open limitation.
2. **Video-02:** the answer omits incoherent fusion of temporally distinct events
   and objects. No second real causal example currently supports changing the
   causal fragment.
3. **Enquirer-02 stochastic completeness:** the process-trace retry recovered
   layered state but omitted conditioned operations, leaving overall recall
   `0.75`. The original target omission is resolved, but the sample demonstrates
   remaining generation variability.
4. **External retrieval:** QASPER remains weak, with most misses occurring before
   reranking and some additional reranker demotions.

## 18. Current go/no-go and next authorized decision

The current state is **NO-GO** for both:

- another full 12-question verification pass; and
- the three-run controlled-generation protocol.

The next decision should be explicit and narrowly scoped:

- either authorize a third NSM-02 mechanism-only attempt or accept/document the
  NSM gap;
- decide how long Video-02 should remain an isolated open causal limitation
  while waiting for a real second example.

Only after those findings are resolved or explicitly accepted should the system
repeat the Part 9g-style full 12-question pass. If that pass is clean enough for
the agreed standard, the separate three-run protocol can then proceed with:

- three full controlled generations;
- per-question mean and spread;
- RAGAS answer-correctness deltas cross-checked against deterministic
  concept-recall movement;
- no prompt, routing, or evidence changes mid-protocol.
