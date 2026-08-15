# Part 9n — Correct-Paper, Wrong-Chunk Verification

Date: 2026-08-15

## Technical summary

The diagnostic reproduced **14 cases: 11 vague and 3 topic-named**. This exactly matches the expected 11 + 3 split.

Only **6/14 (42.9%)** of the correct-paper/wrong-reviewed-chunk cases contain a retrieved same-paper passage that would independently let a generator answer the literal question. The remaining **8/14 (57.1%)** stay on the same topic but omit the requested mechanism, comparison, or number. No case was wholly unrelated to the question.

The 0.55 lexical-overlap check passes **0/14** cases. It therefore misses all 6 human-usable alternate passages and agrees with the binary human judgment only on the 8 non-answering cases (57.1% overall agreement).

## Most same-paper alternates do not answer the question

| Tier | Cases | Answers it | Same topic, wrong evidence | Unrelated section |
| --- | ---: | ---: | ---: | ---: |
| Vague/casual | 11 | 5 (45.5%) | 6 (54.5%) | 0 (0.0%) |
| Topic-named | 3 | 1 (33.3%) | 2 (66.7%) | 0 (0.0%) |
| Overall | 14 | 6 (42.9%) | 8 (57.1%) | 0 (0.0%) |

The correct-paper framing was therefore optimistic for a majority of these misses. It identifies useful paper selection, but not sufficient passage selection.

## Per-case verification

The overlap score is directional: reviewed-gold content tokens covered by the concatenated same-paper passages retrieved in the top four. `Pass` means score ≥ 0.55.

| Tier | Question ID | Same-paper ranks | Sections | Overlap | Pass | Manual label |
| --- | --- | --- | --- | ---: | --- | --- |
| vague | `qasa-1911.03814-7::vague` | 1, 3 | methodology, methodology | 0.316 | No | **answers it** |
| vague | `qasa-1703.10593-13::vague` | 1, 2, 3, 4 | introduction, abstract, results, results | 0.429 | No | **answers it** |
| vague | `qasa-1703.06870-15::vague` | 1, 2, 4 | abstract, related_work, references | 0.417 | No | **same topic, wrong specific evidence** |
| vague | `qasa-1503.04069-9::vague` | 1, 4 | front_matter, references | 0.199 | No | **answers it** |
| vague | `qasa-1610.06475-8::vague` | 2 | introduction | 0.176 | No | **same topic, wrong specific evidence** |
| vague | `qasa-1511.07247-12::vague` | 1, 2, 3, 4 | abstract, introduction, introduction, experiments | 0.545 | No | **answers it** |
| vague | `qasper-bc473c5bd0e1a8be9b2037aa7006fd68217c3f47::vague` | 1, 2, 3, 4 | related_work, abstract, related_work, related_work | 0.487 | No | **answers it** |
| vague | `qasper-fb5ce11bfd74e9d7c322444b006a27f2ff32a0cf::vague` | 1, 2, 3, 4 | results, references, related_work, introduction | 0.538 | No | **same topic, wrong specific evidence** |
| vague | `qasper-b0799e26152197aeb3aa3b11687a6cc9f6c31011::vague` | 1, 2, 3, 4 | abstract, conclusion, introduction, methodology | 0.266 | No | **same topic, wrong specific evidence** |
| vague | `qasper-73738e42d488b32c9db89ac8adefc75403fa2653::vague` | 3, 4 | conclusion, abstract | 0.116 | No | **same topic, wrong specific evidence** |
| vague | `qasper-234ccc1afcae4890e618ff2a7b06fc1e513ea640::vague` | 1, 2, 4 | introduction, front_matter, conclusion | 0.392 | No | **same topic, wrong specific evidence** |
| topic_named | `qasa-1511.04587-13::topic_named` | 1, 2, 3, 4 | abstract, conclusion, introduction, introduction | 0.388 | No | **same topic, wrong specific evidence** |
| topic_named | `qasa-1503.04069-9::topic_named` | 1, 2, 3, 4 | introduction, conclusion, introduction, introduction | 0.395 | No | **answers it** |
| topic_named | `qasa-1706.02413-8::topic_named` | 1, 2, 3, 4 | methodology, conclusion, introduction, methodology | 0.508 | No | **same topic, wrong specific evidence** |

## Manual evidence notes

### `qasa-1911.03814-7::vague` — answers it

**Question:** why does testing on completely new websites count as zero-shot?

A retrieved methodology passage states that no test mention-entity pairs were observed in training and identifies evaluation on held-out test domains; that is enough to explain the zero-shot designation.

### `qasa-1703.10593-13::vague` — answers it

**Question:** what changes when image translation training has paired examples instead of separate collections?

The retrieved introduction and abstract directly contrast aligned pairs with separate unpaired domain collections and explain adversarial mapping plus cycle consistency.

### `qasa-1703.06870-15::vague` — same topic, wrong specific evidence

**Question:** how can an instance segmentation model be adapted to predict body joints?

The abstract says Mask R-CNN generalizes to pose/keypoint detection, but it omits the requested adaptation mechanism: one mask per keypoint type with one-hot targets and a spatial softmax loss.

### `qasa-1503.04069-9::vague` — answers it

**Question:** how can you compare several recurrent-network variants fairly?

The retrieved front matter independently states that every LSTM variant was optimized separately for each task using random search, which answers how the comparison was kept fair.

### `qasa-1610.06475-8::vague` — same topic, wrong specific evidence

**Question:** how does a visual mapping system recognize that it returned to an earlier place and verify it?

The introduction says place recognition detects returns and closes loops, but it does not give the actual loop-detection and geometric-validation procedure requested.

### `qasa-1511.07247-12::vague` — answers it

**Question:** how do you make the classic place-recognition aggregation layer trainable inside a CNN?

Retrieved passages describe NetVLAD as a differentiable generalized VLAD pooling layer whose parameters are learned by backpropagation inside an end-to-end CNN.

### `qasper-bc473c5bd0e1a8be9b2037aa7006fd68217c3f47::vague` — answers it

**Question:** if someone claims machine translation is as good as humans, how should the evaluation be run?

Retrieved passages independently support human evaluation with expert raters, document-level context, ranking/significance analysis, and attention to references—enough to answer how a parity claim should be tested.

### `qasper-fb5ce11bfd74e9d7c322444b006a27f2ff32a0cf::vague` — same topic, wrong specific evidence

**Question:** how often did the robot succeed when instructions referred to color or shape?

The retrieved results passage describes the color/shape task and success criterion but omits the requested success rates (97.6%, 96.0%, and 79.0% for shape alone).

### `qasper-b0799e26152197aeb3aa3b11687a6cc9f6c31011::vague` — same topic, wrong specific evidence

**Question:** what ways can text and images be combined to detect hateful posts?

The passages establish multimodal hate detection and its challenges but do not describe the requested FCM, SCM, and TKM ways of combining visual and textual representations.

### `qasper-73738e42d488b32c9db89ac8adefc75403fa2653::vague` — same topic, wrong specific evidence

**Question:** how much did adapting the question-answering model improve its score?

The abstract and conclusion say adaptation improves over a baseline but contain none of the EM/F1 values needed to answer how much it improved.

### `qasper-234ccc1afcae4890e618ff2a7b06fc1e513ea640::vague` — same topic, wrong specific evidence

**Question:** how much did the proposed data augmentation improve robustness to noisy dialog inputs?

The introduction and conclusion say augmentation improves robustness, but omit the requested clean/adversarial accuracy values and improvement magnitude.

### `qasa-1511.04587-13::topic_named` — same topic, wrong specific evidence

**Question:** what loss does VDSR use for image super-resolution training?

The retrieved passages discuss residual learning, high learning rates, and gradient clipping but never state the Euclidean/mean-squared residual reconstruction loss asked for.

### `qasa-1503.04069-9::topic_named` — answers it

**Question:** how were the nine LSTM variants tuned for a fair comparison?

A retrieved hyperparameter-search passage states that 27 separate random searches covered every nine-variant/three-dataset combination, with 200 trials each and independently sampled settings.

### `qasa-1706.02413-8::topic_named` — same topic, wrong specific evidence

**Question:** why is PointNet++ MRG cheaper than MSG?

The retrieved passages discuss MSG/MRG accuracy and general computation, but not the reason MRG is cheaper: it avoids large-neighborhood feature extraction at the lowest levels.

## Scope and methodology

The source is the saved Part 9m retrieval trace; retrieval was not rerun. For every query, the diagnostic selected hybrid+rerank top-four results from the reviewed source paper only when none of the exact reviewed chunk IDs appeared in the top four. It loaded both retrieved and reviewed-gold text from the unchanged external benchmark BM25 index.

The automated check imports `generation.citation_handler.lexical_overlap_score`, the same content-token overlap function used by `evaluation/audit_part9d_qasper_coverage.py`. For multiple gold chunks, it mirrors that audit's max-over-gold approach against combined candidate evidence.

Manual labels were assigned by reading the literal question, every same-paper passage actually retrieved in the top four, and the reviewed gold passages. `Answers it` requires at least one retrieved passage to contain sufficient evidence on its own; merely naming the method, task, or improvement direction does not qualify when the question asks for a mechanism or number.

## Limitations and robustness

The manual classification is a single-reviewer judgment, not a new golden label or disposition. It is used only for this diagnostic. The lexical check measures token coverage between passages, not semantic entailment, and its complete lack of threshold passes shows that 0.55 is unsuitable as a standalone detector of alternate answer-bearing chunks in this sample.

## Part D implication

The exact-chunk metric does understate quality in **6/14** same-paper misses, but a majority (**8/14**) would still not produce a usable answer. Part D should therefore keep the reranker/final-boundary work **and** explicitly measure within-paper section/passage ranking. Correct-paper Hit@4 should not replace exact-chunk or answerability review as the success criterion.

A useful next diagnostic is to score section-aware reranking on these eight non-answering cases while separately retaining the six answer-bearing alternates as label-completeness cases. No existing golden, alias, prompt, matcher, retriever configuration, or disposition was changed.

## Further question

Should the six verified alternate answer passages be proposed for a separate human-review queue as possible additional acceptable evidence labels? This report does not add them to any golden set.
