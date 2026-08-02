# RAG Research Assistant

Ask questions about AI research papers in plain English and get grounded, cited answers — not a keyword search, not a hallucinated summary.

RAG Research Assistant ingests papers from ArXiv, indexes them section-by-section, and answers questions using hybrid retrieval (semantic + keyword search), cross-encoder reranking, and an LLM that's required to cite its sources. Every answer traces back to the exact paper and section it came from.

---

## Why this exists

Keeping up with AI research means wading through hundreds of papers. Most RAG demos chunk text into arbitrary fixed-size windows, which means a question like *"what dataset did they use?"* can retrieve a random paragraph instead of the actual Experiments section. This project takes a more deliberate approach: retrieval quality is treated as the core problem, not an afterthought.

## Features

- **ArXiv ingestion pipeline** — search, download, and parse papers by topic or category
- **Section-aware chunking** — documents are split by abstract, introduction, methodology, results, and conclusion, not by arbitrary token windows
- **Hybrid retrieval** — dense (semantic) search fused with BM25 (keyword) search via Reciprocal Rank Fusion
- **Cross-encoder reranking** — a second, more precise pass re-scores the top candidates before they reach the LLM
- **Cited, grounded answers** — every answer references the specific papers and sections it draws from, and the model is instructed to say so when it doesn't know
- **RAGAS evaluation** — retrieval and generation quality are measured against a golden Q&A set, not eyeballed
- **Chat + explore + evaluate UI** — a Streamlit app for asking questions, browsing the corpus, and viewing quality metrics
- **One-command deployment** — the full stack runs via Docker Compose

## How it works

```
User query
    │
    ▼
Query Processor        clean + expand the query (HyDE)
    │
    ▼
Hybrid Retriever        dense (Qdrant) + sparse (BM25) → RRF fusion
    │
    ▼
Cross-Encoder Reranker   re-score top candidates for precision
    │
    ▼
Context Assembler        build a prompt from the retrieved chunks
    │
    ▼
LLM (Groq API)         generate a grounded, cited answer
    │
    ▼
Streamlit UI / API       display the answer with source cards
```


## Follow-up

The work completed so far was developed in five parts: ingestion, improved discovery, section-aware chunking, dense/sparse indexing, and an evaluated hybrid retrieval stack. The generation layer is still planned and is not required for retrieval evaluation.

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

### Part 2: Improvement — discovery before ingestion using LLM and MCP - 09/07/2026

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

The executed evaluation currently covers 5 local papers, 530 section-aware chunks, and 10 manually labelled queries. All tested configurations ranked the relevant paper first for every query. This confirms the pipeline works, but the benchmark is too small and easy to select a winning configuration.

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

`CorpusEnrichmentRetriever` accepts either `min_results`/`min_score` or a custom `relevance_gate(query, results)`. For hybrid RRF output, use a calibrated custom gate based on reranker or dense evidence because RRF scores are rank-fusion values rather than confidence probabilities. Set `bm25_index_path` so the rebuilt sparse corpus is saved and reloaded before the retry.

**Part 6 summary:** a 1000+ paper baseline can be resumed safely, paper identity is independent of category, and missing coverage can be added through the same ingestion and processing code instead of a duplicate live-search path.
