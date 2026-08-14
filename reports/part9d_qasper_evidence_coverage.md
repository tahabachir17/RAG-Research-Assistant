# Part 9d — QASPER Evidence-Coverage Audit

Date: 2026-08-13

Of 16 QASPER questions with Recall@4 = 0, **0** are bucket (a) evidence absent and **16** are bucket (b) evidence present but not retrieved.

| Question ID | Bucket | Indexed gold chunks | Best overlap | Evidence location / cause |
| --- | --- | ---: | ---: | --- |
| `qasper-bc473c5bd0e1a8be9b2037aa7006fd68217c3f47` | (b) Evidence present, not retrieved | 1/1 | 0.616 | Reviewed evidence is indexed in related_work; this is a top-4 ranking miss. |
| `qasper-32a232310babb92991c4b1b75f7aa6b4670ec447` | (b) Evidence present, not retrieved | 1/1 | 0.921 | Reviewed evidence is indexed in front_matter; this is a top-4 ranking miss. |
| `qasper-fb5ce11bfd74e9d7c322444b006a27f2ff32a0cf` | (b) Evidence present, not retrieved | 1/1 | 0.963 | Reviewed evidence is indexed in results; this is a top-4 ranking miss. |
| `qasper-559c1307610a15427caeb8aff4d2c01ae5c9de20` | (b) Evidence present, not retrieved | 1/1 | 0.792 | Reviewed evidence is indexed in introduction; this is a top-4 ranking miss. |
| `qasper-83f567489da49966af3dc5df2d9d20232bb8cb1e` | (b) Evidence present, not retrieved | 2/2 | 1.000 | Reviewed evidence is indexed in conclusion; this is a top-4 ranking miss. |
| `qasper-efc65e5032588da4a134d121fe50d49fe8fe5e8c` | (b) Evidence present, not retrieved | 3/3 | 1.000 | Reviewed evidence is indexed in introduction, methodology; this is a top-4 ranking miss. |
| `qasper-36a9230fadf997d3b0c5fc8af8d89bd48bf04f12` | (b) Evidence present, not retrieved | 1/1 | 0.833 | Reviewed evidence is indexed in related_work; this is a top-4 ranking miss. |
| `qasper-9651fbd887439bf12590244c75e714f15f50f73d` | (b) Evidence present, not retrieved | 1/1 | 0.811 | Reviewed evidence is indexed in introduction; this is a top-4 ranking miss. |
| `qasper-71fca845edd33f6e227eccde10db73b99a7e157b` | (b) Evidence present, not retrieved | 2/2 | 0.900 | Reviewed evidence is indexed in related_work, results; this is a top-4 ranking miss. |
| `qasper-994ac7aa662d16ea64b86510fcf9efa13d17b478` | (b) Evidence present, not retrieved | 1/1 | 0.875 | Reviewed evidence is indexed in experiments; this is a top-4 ranking miss. |
| `qasper-f2c5da398e601e53f9f545947f61de5f40ede1ee` | (b) Evidence present, not retrieved | 2/2 | 0.938 | Reviewed evidence is indexed in introduction; this is a top-4 ranking miss. |
| `qasper-0fa81adf00662694e1dc74475ae2b9283c50748c` | (b) Evidence present, not retrieved | 1/1 | 0.842 | Reviewed evidence is indexed in conclusion; this is a top-4 ranking miss. |
| `qasper-b3fcab006a9e51a0178a1f64d1d084a895bd8d5c` | (b) Evidence present, not retrieved | 2/2 | 0.923 | Reviewed evidence is indexed in methodology; this is a top-4 ranking miss. |
| `qasper-9508e9ec675b6512854e830fa89fa6a747b520c5` | (b) Evidence present, not retrieved | 1/1 | 0.714 | Reviewed evidence is indexed in introduction; this is a top-4 ranking miss. |
| `qasper-234ccc1afcae4890e618ff2a7b06fc1e513ea640` | (b) Evidence present, not retrieved | 2/2 | 0.889 | Reviewed evidence is indexed in methodology; this is a top-4 ranking miss. |
| `qasper-2ff3898fbb5954aa82dd2f60b37dd303449c81ba` | (b) Evidence present, not retrieved | 2/2 | 1.000 | Reviewed evidence is indexed in experiments, methodology; this is a top-4 ranking miss. |

## Method

The audit reconstructed original evidence from `evaluation/data/external_cache/qasper.json`, loaded all chunks from `external_bm25_index.pkl`, and checked normalized exact substrings plus the same lexical-overlap threshold (0.55) used by benchmark evidence alignment. Stored reviewed chunk IDs were independently checked for presence in the current BM25 corpus.

This is a coverage diagnosis only; no retrieval or chunking logic was changed.
