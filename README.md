# RAG Research Assistant

Ask questions about AI research papers in plain English and get grounded, cited answers â€” not a keyword search, not a hallucinated summary.

RAG Research Assistant ingests papers from ArXiv, indexes them section-by-section, and answers questions using hybrid retrieval (semantic + keyword search), cross-encoder reranking, and an LLM that's required to cite its sources. Every answer traces back to the exact paper and section it came from.

---

## Why this exists

Keeping up with AI research means wading through hundreds of papers. Most RAG demos chunk text into arbitrary fixed-size windows, which means a question like *"what dataset did they use?"* can retrieve a random paragraph instead of the actual Experiments section. This project takes a more deliberate approach: retrieval quality is treated as the core problem, not an afterthought.

## Features

- **ArXiv ingestion pipeline** â€” search, download, and parse papers by topic or category
- **Section-aware chunking** â€” documents are split by abstract, introduction, methodology, results, and conclusion, not by arbitrary token windows
- **Hybrid retrieval** â€” dense (semantic) search fused with BM25 (keyword) search via Reciprocal Rank Fusion
- **Cross-encoder reranking** â€” a second, more precise pass re-scores the top candidates before they reach the LLM
- **Cited, grounded answers** â€” every answer references the specific papers and sections it draws from, and the model is instructed to say so when it doesn't know
- **Retrieval-stage evaluation** â€” dense, sparse, hybrid, reranked, and MMR stages are measured independently from generation using Recall, Precision, MRR, nDCG, and latency
- **Chat + explore + evaluate UI** â€” a Streamlit app for asking questions, browsing the corpus, and viewing quality metrics
- **One-command deployment** â€” the full stack runs via Docker Compose

## How it works

```
User query
    â”‚
    â–¼
Query Processor        clean + expand the query (HyDE)
    â”‚
    â–¼
Hybrid Retriever        dense (Qdrant) + sparse (BM25) â†’ RRF fusion
    â”‚
    â–¼
Cross-Encoder Reranker   re-score top candidates for precision
    â”‚
    â–¼
Context Assembler        build a prompt from the retrieved chunks
    â”‚
    â–¼
LLM provider adapter    Claude / OpenAI / Ollama, sync or streaming
    â”‚
    â–¼
Streamlit UI / API       display the answer with source cards
```


## Follow-up

The work completed so far was developed in eight parts: ingestion, improved discovery, section-aware chunking, dense/sparse indexing, the retrieval stack, scalable resumable ingestion, retrieval-stage evaluation, and generation. Retrieval and generation remain independently testable so failures can be attributed to the correct layer.

### Part 1: Ingestion - 06/07/2026

This part implements a reproducible pipeline using the ArXiv API, PDF parsing, regular expressions, and text-processing heuristics, without requiring an LLM or MCP server.

| File | Summary |
|---|---|
| `ingestion/arxiv_scraper.py` | Searches ArXiv and normalizes results into `Paper` objects with identifiers, metadata, abstracts, categories, and URLs. |
| `ingestion/pdf_downloader.py` | Downloads PDFs into category-based directories, skips existing files, and reports download failures. |
| `ingestion/pdf_parser.py` | Extracts page-by-page text and metadata from PDFs with PyMuPDF and produces a `RawDocument`. |
| `ingestion/section_detector.py` | Detects research-paper headings and groups text into labelled sections such as abstract, introduction, method, experiments, conclusion, and references. |
| `ingestion/data_cleaner.py` | Repairs common PDF text artifacts, joins broken words and lines, removes isolated page numbers, and normalizes whitespace. |
| `ingestion/citation_extractor.py` | Parses references into structured citations and extracts ArXiv IDs and DOIs when available. |
| `ingestion/metadata_extractor.py` | Produces consistent paper metadata for persistence and later indexing stages. |
| `ingestion/pipeline.py` | Orchestrates download, parsing, section detection, cleaning, citation extraction, and JSON persistence while collecting per-paper errors. |
| `ingestion/__init__.py` | Exposes the public ingestion classes and helper functions. |

**Part 1 summary:** papers can be downloaded, transformed into clean section-labelled text, enriched with metadata and citations, and saved as structured JSON without generative AI dependencies.

### Part 2: Improvement â€” discovery before ingestion using LLM and MCP - 09/07/2026

This part adds a dedicated discovery stage to improve result relevance, support multiple sources, remove duplicates, and retain a reliable ArXiv fallback.

| File | Summary |
|---|---|
| `ingestion/paper_discovery.py` | Defines the discovery interface and implements ArXiv, alphaXiv MCP, Feyman, research-API, and automatic alphaXiv-to-ArXiv fallback providers. It normalizes responses, deduplicates papers, and ranks candidates. |
| `ingestion/pipeline.py` | Runs the selected discovery provider before downloading and records provider, discovery, and fallback metadata in saved records. |
| `ingestion/arxiv_scraper.py` | Provides direct ArXiv search and remains the dependable fallback when enhanced discovery is unavailable. |
| `tests/unit/test_ingestion.py` | Tests normalization, scoring, deduplication, ingestion helpers, and automatic provider fallback without live external services. |

**Part 2 summary:** ingestion is no longer tied to one search mechanism. Richer discovery can be attempted while a deterministic ArXiv-only path remains available for local testing and fallback.

### Part 3: Chunking - 14/07/2026

This part transforms processed papers into retrieval-ready chunks while retaining section structure and source provenance.

| File | Summary |
|---|---|
| `processing/chunker.py` | Defines the `Chunk` schema and section-aware chunker. It uses bounded overlapping windows, prefers natural text boundaries, preserves exact offsets, creates deterministic UUIDs, and supports raw-text fallback. |
| `processing/__init__.py` | Exposes `Chunk`, `SectionAwareChunker`, and `chunk_document` as the processing package API. |
| `tests/unit/test_chunker.py` | Verifies labels, offsets, window sizes, overlap, stable IDs, configuration validation, and unsectioned-document fallback. |

**Part 3 summary:** processed papers can now become stable, traceable chunks for embedding, indexing, retrieval filtering, and section-specific citation. Defaults are 512 tokens with an 80-token overlap and natural-boundary cuts where possible.

### Part 4: Dense and sparse indexing - 16/07/2026

This part completes the first retrieval-ready processing pipeline by adding dense embeddings, BM25 sparse indexing, persistence, and local end-to-end validation without remote discovery.

| File | Summary |
|---|---|
| `processing/chunker.py` | Adds the configurable 40-token minimum, duplicate prevention, malformed-input handling, deterministic chunk IDs, section labels, metadata, and exact character offsets. |
| `processing/embedder.py` | Loads `sentence-transformers/all-MiniLM-L6-v2` once and converts text or `Chunk` objects into JSON-ready 384-dimensional dense embeddings. Empty inputs and blank text are handled safely. |
| `processing/bm25_indexer.py` | Builds a `BM25Okapi` sparse index from chunks, performs ranked keyword search, and saves or restores the index and chunk records with pickle. |
| `local_pipeline_test.py` | Runs the existing local PDFs through ingestion, chunking, BM25 indexing, and optional embedding without external paper discovery or new downloads. |
| `tests/unit/test_chunker.py` | Tests normal, long, empty, missing, and malformed section inputs together with offsets, overlap, labels, and unique IDs. |
| `tests/unit/test_embedder.py` | Tests model reuse, empty inputs, blank-text filtering, metadata preservation, and numeric embedding output using a mocked transformer model. |
| `tests/unit/test_bm25_indexer.py` | Tests index construction, ranking, `top_k`, empty inputs, persistence, restoration, and post-load search. |

**Part 4 summary:** locally ingested papers can now be transformed into stable section-aware chunks, dense MiniLM embeddings, and a persistent BM25 sparse index for later retrieval.

### Part 5: Retrieval and evaluation - 27/07/2026

This part implements the production retrieval stages and adds reproducible notebooks for integration testing and retrieval-quality evaluation.

| File | Summary |
|---|---|
| `retrieval/dense_retriever.py` | Embeds queries and searches a cosine Qdrant collection with payload filters, score thresholds, health checks, and dimension validation. |
| `retrieval/sparse_retriever.py` | Loads trusted project-generated BM25 artifacts and returns filtered, backend-neutral retrieval results. |
| `retrieval/hybrid_retriever.py` | Combines dense and sparse rankings using Reciprocal Rank Fusion, deduplicates chunks, and preserves result provenance. |
| `retrieval/reranker.py` | Reranks candidate chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2`; injected models support deterministic offline tests. |
| `retrieval/mmr_sampler.py` | Applies Maximal Marginal Relevance to balance query relevance and result diversity. |
| `retrieval/retriever_factory.py` | Builds validated dense, sparse, or nested hybrid retrievers from configuration and injected runtime dependencies. |
| `notebooks/06_retrieval_stack_testbench.ipynb` | Runs all production retrieval components end to end with deterministic offline dependencies. |
| `notebooks/07_retrieval_evaluation.ipynb` | Runs local PDFs through ingestion, processing, MiniLM/Qdrant and BM25 retrieval, then reports Hit@K, Precision@K, Recall@K, MRR, nDCG, latency, and query-level diagnostics. |
| `tests/unit/test_advanced_retrieval.py` | Tests RRF fusion, cross-encoder reranking, MMR selection, validation behavior, and factory construction without external services. |

The original notebook evaluation covers 5 local papers, 530 section-aware chunks, and 10 manually labelled queries. All tested configurations ranked the relevant paper first for every query. This confirms the pipeline works, but that benchmark is too small and easy to select a winning configuration. The larger persisted-corpus ablation is documented in Part 7.

**Part 5 summary:** the project can now evaluate retrieval independently of generation across dense, sparse, hybrid, reranked, and diversity-aware configurations. The next evaluation milestone is a larger independently labelled corpus with paraphrased questions, hard negatives, multi-relevant queries, chunk-level judgments, and the real MS MARCO cross-encoder.

### Part 6: Scalable ingestion and retrieval-triggered enrichment - 01/08/2026

This part makes the corpus resumable and lets retrieval expand it only when the static indexes are insufficient.

| File | Summary |
|---|---|
| `ingestion/identity.py` | Normalizes modern, versioned, URL, and legacy ArXiv identifiers into one canonical corpus identity. |
| `ingestion/corpus_registry.py` | Uses SQLite as the authoritative paper/checkpoint registry and atomically exports `papers.json`. |
| `ingestion/pdf_downloader.py` | Stores PDFs in a flat canonical-ID layout, retries transient HTTP failures, validates PDFs, and atomically promotes `.part` downloads. |
| `ingestion/pipeline.py` | Supports `--resume`, direct selected-paper ingestion, durable per-paper states, flat processed output, and atomic JSON writes. |
| `retrieval/fallback_retriever.py` | Runs static retrieval first; when its relevance gate fails, it discovers and ingests papers, rebuilds BM25 from every registered processed paper, upserts dense vectors, reloads sparse retrieval, and retries the query. |
| `tests/unit/test_ingestion_registry.py` | Tests registry state transitions, checkpoint persistence, resume behavior, identity normalization, and recovery after partial ingestion failures. |

`CorpusEnrichmentRetriever` accepts either `min_results`/`min_score` or a custom `relevance_gate(query, results)`. For hybrid RRF output, use a calibrated custom gate based on reranker or dense evidence because RRF scores are rank-fusion values rather than confidence probabilities. Set `bm25_index_path` so the rebuilt sparse corpus is saved and reloaded before the retry.

**Part 6 summary:** a 1000+ paper baseline can be resumed safely, paper identity is independent of category, and missing coverage can be added through the same ingestion and processing code instead of a duplicate live-search path.

### Part 7: Retrieval-stage ablation on the persisted corpus - 03/08/2026

This part isolates retrieval quality from answer generation and evaluates every retrieval stage against the same ranked relevance judgments. It runs dense-only, sparse-only, hybrid RRF, hybrid plus the real MS MARCO cross-encoder, and hybrid plus reranking and MMR. Rankings and results are cached so failures can be inspected without invoking an LLM.

| File | Summary |
|---|---|
| `evaluation/data/golden_retrieval.json` | Contains 50 retrieval questions with paper relevance labels and fields for reviewed chunk IDs. Ten questions come from the earlier notebook; 40 title-derived bootstrap questions are explicitly marked as requiring review. |
| `evaluation/bootstrap_retrieval_golden.py` | Rebuilds the reviewable 50-question seed set from the ingestion registry when a chunk-labelled golden set does not exist yet. |
| `evaluation/label_retrieval.py` | Creates a top-20 candidate dump and provides a lightweight CLI for accepting or rejecting relevant chunks. Progress is saved after each question. |
| `evaluation/metrics.py` | Implements stage-agnostic Hit@k, Precision@k, Recall@k, reciprocal rank, and binary nDCG@k over any ranked identifier list. |
| `evaluation/retrieval_evaluator.py` | Executes and times the five-stage ablation, measures reranker lift, identifies universal failures, and writes timestamped JSON, CSV, Markdown, and cache artifacts. |
| `tests/unit/test_retrieval_evaluator.py` | Verifies stage-agnostic metric calculations and the targeted reranker-promotion measurement. |

The 03/08/2026 run evaluated 50 questions against the persisted corpus (99,141 BM25 chunks plus the Qdrant dense index) using `cross-encoder/ms-marco-MiniLM-L-6-v2` rather than the notebook proxy. Latency is cumulative for each named configuration.

| Configuration | Recall@5 | Recall@8 | Recall@20 | MRR | nDCG@20 | Average latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.820 | 0.820 | 0.820 | 0.820 | 0.820 | 524 ms |
| Sparse | 0.820 | **0.840** | **0.840** | 0.780 | 0.795 | 1,428 ms |
| Hybrid RRF | 0.820 | 0.820 | 0.820 | 0.820 | 0.820 | 1,952 ms |
| Hybrid + rerank | 0.820 | 0.820 | 0.820 | 0.820 | 0.820 | 10,679 ms |
| Hybrid + rerank + MMR | 0.820 | 0.820 | 0.820 | 0.820 | 0.820 | 14,012 ms |

Sparse retrieval produced the best Recall@8 and Recall@20. Hybrid did not strictly beat both single retrievers on any reported metric. Reranking produced no Recall@8 lift (`0.820 -> 0.820`) while adding about 8.7 seconds per query, and MMR added another 3.3 seconds without improving these paper-level metrics. Eight questions failed in every configuration because their relevant papers had no chunks in the BM25 artifact, identifying an ingestion/index synchronization gap rather than a ranking failure.

The current run is a diagnostic baseline, not a publication-quality comparison: none of the 50 questions has completed chunk-level review, and the 40 title-derived questions make the benchmark easier than independently authored paraphrases would be. Use the review CLI before making architecture decisions from these numbers.

```powershell
# Run the full retrieval ablation and save timestamped artifacts.
.\venv\Scripts\python.exe -m evaluation.retrieval_evaluator --candidate-k 20

# Generate top-20 candidates and review chunk relevance interactively.
.\venv\Scripts\python.exe -m evaluation.label_retrieval
```

**Part 7 summary:** on the current paper-level diagnostic set, sparse retrieval provides the highest recall, while hybrid fusion, the cross-encoder, and MMR do not earn their additional latency. The next valid comparison requires reviewed chunk labels, independently written questions, hard negatives, and reindexing the eight missing papers.

### Part 8: Grounded answer generation - 03/08/2026

This part turns ranked `RetrievalResult` chunks into bounded prompts and structured cited answers without coupling provider SDK objects to the rest of the application.

| File | Summary |
|---|---|
| `config/settings.py` | Centralizes provider, model, API-key, token-limit, and temperature settings with Pydantic `BaseSettings`. |
| `config/prompts/*.yaml` | Defines strict Jinja templates for cited Q&A, summarization, paper comparison, and HyDE retrieval expansion. |
| `generation/prompt_manager.py` | Loads and caches YAML prompts, rejects unsafe names and malformed templates, and fails clearly when required variables are missing. |
| `generation/context_assembler.py` | Converts ranked retrieval results into numbered, whole-chunk context blocks with a configurable token budget and citation map. |
| `generation/citation_handler.py` | Validates numeric citation markers, reports structured lexical claim-support flags, hard-rejects provider-native markup, and builds an ordered source list. |
| `generation/faithfulness_verifier.py` | Optionally performs one evidence-only auxiliary audit and appends verdicts to the same claim-support flags. |
| `generation/llm_client.py` | Provides injectable synchronous and asynchronous clients for Claude, OpenAI, Gemini, Groq, LM Studio, and Ollama; every completion carries provider finish metadata and token counts when available. |
| `generation/provider_router.py` | Routes non-streaming requests through the configured zero-cost provider order, defaulting to Groq, Gemini, and then LM Studio. |
| `generation/structured_answer.py` | Parses cited JSON fields, rejects uncited or out-of-range factual cells, supports explicit insufficient-evidence answers, and renders Markdown deterministically. |
| `generation/streaming_handler.py` | Produces incremental text or SSE-ready token/done events while buffering the complete answer for final citation validation. |
| `generation/response_formatter.py` | Owns the chat response contract, including finish reason, validation failures, and whether the original or repaired attempt produced the answer. |
| `generation/response_validator.py` | Applies deterministic citation, truncation, required-field, table-completeness, and max-item checks, then supports one failure-specific repair attempt. |
| `generation/cli.py` | Provides an offline-by-default smoke harness, optional evaluator-result loading, streamed event output, and opt-in live provider calls. |
| Focused `tests/unit/test_*.py` generation test modules | Exercise prompt errors, context limits, citation failures, structured output, provider routing and rate limits, streaming interruption, and response formatting without network calls. |

The validated local flow is: `RetrievalResult` â†’ numbered context â†’ rendered prompt â†’ injected LLM response â†’ citation validation â†’ `GeneratedAnswer.to_dict()`. The complete test suite currently passes 186 tests.

Known limitations are explicit: API and frontend integration are not part of this milestone; provider tests use injected clients rather than live services; streaming interruptions propagate without a false completion event; and unknown citations are reported rather than silently removed or rewritten. Source fields unavailable in `RetrievalResult` remain `null` or empty until a richer metadata lookup is connected.

```powershell
# Deterministic offline smoke test.
.\venv\Scripts\python.exe -m generation.cli

# Inspect SSE-ready token and completion events.
.\venv\Scripts\python.exe -m generation.cli --events

# Test generation from one saved retrieval-evaluation ranking.
.\venv\Scripts\python.exe -m generation.cli --results-json evaluation\data\eval_results\retrieval_eval_<timestamp>.json --config hybrid_rerank --query-id <query_id>

# Full local RAG test: prompt -> BM25 retrieval -> Groq -> cited full chunks.
.\venv\Scripts\python.exe -m generation.cli "How does scaled dot-product attention work, and why is scaling needed?" --retrieve --live --provider groq

# Zero-cost failover: Groq -> Gemini -> LM Studio.
.\venv\Scripts\python.exe -m generation.cli "How does scaled dot-product attention work?" --retrieve --live --provider router
```

The RAG CLI retrieves 30 BM25 candidates by default, excludes references,
bibliography, front matter, and acknowledgements, then selects 5 while allowing
at most 2 chunks per normalized paper ID and 1 chunk per paper section. Override
the diversity controls with `--candidate-k`, `--top-k`,
`--max-chunks-per-paper`, and `--max-chunks-per-section`. For diagnostics,
`--include-low-information-sections` restores the excluded sections.

The interactive default performs one literal BM25 search. Use
`--evidence-query-expansion` only for difficult offline analysis; it performs
four facet searches and therefore adds latency. Cross-encoder reranking remains
opt-in because the persisted-corpus baseline showed no recall lift.

Non-streaming answers use cited JSON internally and deterministic Markdown
rendering externally. Ordinary answers are limited to a small set of atomic,
individually cited claims; required-field evaluation questions use the same
mechanism with a strict table schema. This keeps the API-facing answer readable
without trusting a provider to produce or repair Markdown formatting.

For higher semantic precision, add `--rerank`. This loads the cached
`cross-encoder/ms-marco-MiniLM-L-6-v2` model, reranks the BM25 candidate pool
before diversity selection, and can add substantial CPU latency:

```powershell
.\venv\Scripts\python.exe -m generation.cli "YOUR QUESTION" --retrieve --rerank --candidate-k 20 --live --provider groq

# Opt into a real configured provider.
.\venv\Scripts\python.exe -m generation.cli --live --provider ollama --model llama3.1 --stream
```
#### Generation answer-quality evaluation and runtime gates - 05/08/2026

Generation now uses an independent evaluator rather than importing retrieval-evaluator internals. `evaluation/generation_metrics.py` is separate from the retrieval metrics module because claim grounding, citation coverage, field completeness, truncation, and qualifying-item classification are generation-stage concepts; keeping them separate avoids turning the stage-agnostic ranking module into mixed-purpose code. `evaluation/generation_evaluator.py` exercises the same `run_generation` validation-and-repair path used by the CLI and writes timestamped JSON, CSV, and Markdown breakdowns with latency and an injectable provider-cost estimate. Required-field evaluations now request cited JSON and render the final table in code. `evaluation/llm_judge.py` judges only semantic support, qualification, limitation attribution, and item distinctness from supplied evidence; deterministic format checks never spend judge calls. Judge failures remain `unjudged`, and verdict caches now include the evidence, subjects, provider, model, and rubric version so changed evidence cannot reuse a stale verdict.

Every new generation-evaluation JSON artifact records SHA-256 hashes for the
golden file and QA prompt together with the BM25 path, context budget, output
budget, and evaluation schema version. This makes cross-model runs comparable
and exposes accidental changes to the benchmark.

The first release-quality model comparison has **not** been run. The local candidate file contains 20 frozen-context hard questions, but 0 are marked human-reviewed and no human calibration verdicts have been supplied. The loader deliberately rejects these candidates when `require_reviewed=True`; consequently there are no honest groundedness, precision/recall, latency, or cost numbers to report yet, and none of the proposed quality thresholds is claimed as passing. This is a data-review blocker, not a model result. Exact-label judge/human calibration agreement is fixed at **at least 80%** before judge-based metrics may gate a release.

The deterministic implementation is covered by the full offline suite: **186 tests pass**. These tests cover citation mapping and lexical claim-support flags, optional verifier behavior, compound-question handling, structured output, provider routing, bounded repair, evaluation metrics, RAGAS projection, and artifact generation. They do not establish answer quality because provider calls are mocked. Cost remains `null` unless the evaluator is given a provider-specific cost estimator, preventing an unknown price from being presented as zero.

Suggested release gates remain starting hypotheses: no truncated final answers, 100% valid project citation syntax, at least 95% claim-level citation coverage, no unsupported qualifying items, at least 90% judge-supported claims, and 100% required-field completeness. They should be tuned only after human review and the first real run.

### Run generation evaluation

Generation is intentionally limited to Groq or Gemini. The semantic LLM judge
and RAGAS use one shared evaluator target: Groq, Gemini, or local Qwen through
LM Studio. The command prints the fully resolved matrix before making a model
call, preventing a provider switch from silently retaining another provider's
model.

Validate the default matrix without making model calls:

```powershell
.\venv\Scripts\python.exe -m evaluation.run_generation_eval --dry-run
```

Run one answer with Groq 70B and evaluate it with the Groq 8B semantic judge
and RAGAS:

```powershell
.\venv\Scripts\python.exe -m evaluation.run_generation_eval --limit 1 --provider groq --model llama-3.3-70b-versatile --judge-provider groq --judge-model llama-3.1-8b-instant
```

Switch both evaluation layers to Gemini without changing the generator:

```powershell
.\venv\Scripts\python.exe -m evaluation.run_generation_eval --limit 1 --provider groq --model llama-3.3-70b-versatile --judge-provider gemini --judge-model gemini-3.5-flash-lite
```

Use local Qwen for both evaluation layers after starting LM Studio:

```powershell
.\venv\Scripts\python.exe -m evaluation.run_generation_eval --limit 1 --provider groq --model llama-3.3-70b-versatile --judge-provider qwen --judge-model qwen/qwen3-4b-2507 --request-timeout 120 --ragas-timeout 300
```

Results are written as timestamped JSON, CSV, and Markdown files under `evaluation/data/eval_results`. By default, the run combines deterministic validation, the evidence-only LLM judge, and RAGAS scoring. `--judge-provider` and `--judge-model` control both semantic judging and RAGAS. Use `--no-ragas` only when deliberately isolating generation/judge behavior. Reference-dependent RAGAS metrics are marked unavailable until reviewed reference answers exist. Add `--require-reviewed` only after the golden questions have been human-reviewed; the current candidates intentionally fail that release gate.

The resumable full-set command defaults to Groq generation and a distinct
Gemini judge. It atomically checkpoints every generated question and every
RAGAS metric value as it completes:

```powershell
.\venv\Scripts\python.exe -m evaluation.run_generation_eval --dataset evaluation\data\golden_generation_qa.json --generator-provider groq --judge-provider gemini --workers 1 --requests-per-second 0.05

# Resume the newest compatible run, or name an exact JSON checkpoint.
.\venv\Scripts\python.exe -m evaluation.run_generation_eval --dataset evaluation\data\golden_generation_qa.json --generator-provider groq --judge-provider gemini --workers 1 --requests-per-second 0.05 --resume
.\venv\Scripts\python.exe -m evaluation.run_generation_eval --dataset evaluation\data\golden_generation_qa.json --generator-provider groq --judge-provider gemini --resume-from evaluation\data\eval_results\generation_eval_<timestamp>.json
```

Use `--chunk-count N` (or `--top-k N`) to rerun a single A5-style context
count through the same durable command. The full metric schema includes
faithfulness, answer relevancy, context precision, context recall, context
utilization, and answer correctness. Reference-dependent values are explicitly
recorded as unavailable when a golden record lacks `reference_answer`; they are
never synthesized from the evidence passages.

To add or retry RAGAS scoring without paying the generation cost again:

```powershell
.\venv\Scripts\python.exe -m evaluation.run_ragas_eval evaluation\data\eval_results\generation_eval_<timestamp>.json
```

Run the isolated nested 4/5/8-chunk generation ablation without changing the
production default:

```powershell
.\venv\Scripts\python.exe -m evaluation.run_context_count_experiment --limit 1 --counts 4 5 8 --provider groq --model llama-3.3-70b-versatile --judge-provider groq --judge-model llama-3.1-8b-instant
```

The complete 11/08/2026 retry used Gemini 3.5 Flash Lite as the RAGAS judge and
produced these raw one-question values:

| chunks | faithfulness | answer relevancy | context utilization |
|---:|---:|---:|---:|
| 4 | 0.9231 | 0.4120 | 1.0000 |
| 5 | 1.0000 | 0.4066 | 0.7000 |
| 8 | 1.0000 | 0.4120 | 0.7000 |

This is diagnostic rather than a basis for changing `RERANK_TOP_K`: it uses
one unreviewed question and answer relevancy is effectively tied. Four chunks
used all supplied context, while five and eight used only 70%, so the production
default remains unchanged pending a reviewed multi-question run.

RAGAS uses provider-compatible answer relevancy (`strictness=1`, therefore `n=1`) and throttles Groq calls to `RAGAS_REQUESTS_PER_SECOND=0.05` by default. Increase that value only when the judge account has a larger token-per-minute allowance. The re-score path evaluates the exact context chunks seen by generation when the result contains `context_chunk_ids`.

Rebuild and audit the evidence-aligned candidate set after replacing the BM25 artifact:

```powershell
.\venv\Scripts\python.exe -m evaluation.bootstrap_generation_golden
.\venv\Scripts\python.exe -m evaluation.bootstrap_generation_golden --audit-only
```

The bootstrapper guarantees that frozen chunks exist and belong to the declared source paper. It deliberately leaves every record `reviewed=false`; human review and calibration labels are still required before release gating.

**Part 8 summary:** generation is now a self-contained, provider-neutral layer with finish metadata, deterministic runtime validation, one bounded repair attempt, and an independent offline quality evaluator. The immediate next step is human review of the 20 candidate questions plus 5â€“10 calibration cases, followed by the first real cross-model run; until then, release-gate quality remains unmeasured.
