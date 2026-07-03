# RAG Research Assistant — Full Architecture

> A production-grade RAG system for querying AI research papers.
> Stack: Python · FastAPI · Qdrant · sentence-transformers · Claude API · Streamlit · Docker

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Full Folder Structure](#2-full-folder-structure)
3. [Layer-by-Layer Breakdown](#3-layer-by-layer-breakdown)
   - [Ingestion](#31-ingestion)
   - [Processing](#32-processing)
   - [Retrieval](#33-retrieval)
   - [Generation](#34-generation)
   - [Evaluation](#35-evaluation)
   - [API](#36-api)
   - [Frontend](#37-frontend)
   - [Config & Infra](#38-config--infra)
4. [Data Flow](#4-data-flow)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Environment Variables](#6-environment-variables)
7. [Docker Services](#7-docker-services)
8. [CI/CD Pipeline](#8-cicd-pipeline)

---

## 1. Project Overview

```
User query
    │
    ▼
[Query Processor]  ← expand + rewrite the query
    │
    ▼
[Hybrid Retriever]  ← dense (Qdrant) + sparse (BM25) → RRF fusion
    │
    ▼
[Cross-Encoder Reranker]  ← re-score top-k chunks
    │
    ▼
[Context Assembler]  ← build prompt with retrieved chunks + citations
    │
    ▼
[LLM (Claude API)]  ← generate grounded answer with source attribution
    │
    ▼
[Streamlit UI / FastAPI]  ← display answer + cited papers
```

The system ingests AI research papers from ArXiv, chunks them by section
(abstract, introduction, methodology, results, conclusion), embeds each
chunk, and stores them in Qdrant. At query time it runs hybrid retrieval
(semantic + BM25), reranks with a cross-encoder, assembles a context
window, and calls the LLM.

---

## 2. Full Folder Structure

```
rag-research-assistant/
│
├── data/                               # All data artifacts (gitignored)
│   ├── raw/                            # Original downloaded PDFs
│   │   └── arxiv/                      # Organized by ArXiv category
│   │       ├── cs.AI/
│   │       ├── cs.CL/
│   │       └── cs.LG/
│   ├── processed/                      # Cleaned text chunks (JSON/JSONL)
│   │   ├── chunks/                     # One JSONL file per paper
│   │   └── embeddings/                 # Cached numpy embedding arrays
│   └── metadata/                       # Paper-level metadata
│       └── papers.json                 # Master metadata registry
│
├── ingestion/                          # Data ingestion layer
│   ├── __init__.py
│   ├── arxiv_scraper.py                # ArXiv API client
│   ├── pdf_downloader.py               # PDF fetcher with dedup
│   ├── pdf_parser.py                   # PyMuPDF-based text extractor
│   ├── section_detector.py             # Detects paper sections by heading
│   ├── metadata_extractor.py           # Parses title, authors, year, DOI
│   ├── citation_extractor.py           # Extracts reference list
│   ├── data_cleaner.py                 # Text normalization & noise removal
│   └── pipeline.py                     # Orchestrates full ingestion run
│
├── processing/                         # Chunking, embedding & indexing layer
│   ├── __init__.py
│   ├── chunker.py                      # Section-aware + sliding-window chunker
│   ├── embedder.py                     # Wraps sentence-transformers / OpenAI
│   ├── bm25_indexer.py                 # Builds & serializes BM25 index
│   ├── qdrant_indexer.py               # Upserts vectors into Qdrant
│   ├── metadata_tagger.py              # Attaches payload metadata to vectors
│   └── pipeline.py                     # Orchestrates full processing run
│
├── retrieval/                          # Retrieval & reranking layer
│   ├── __init__.py
│   ├── query_processor.py              # Query cleaning + HyDE expansion
│   ├── dense_retriever.py              # Qdrant semantic search
│   ├── sparse_retriever.py             # BM25 keyword search
│   ├── hybrid_retriever.py             # RRF fusion of dense + sparse
│   ├── reranker.py                     # Cross-encoder reranker (SBERT)
│   ├── mmr_sampler.py                  # Maximal Marginal Relevance diversity
│   └── retriever_factory.py            # Returns configured retriever instance
│
├── generation/                         # LLM generation layer
│   ├── __init__.py
│   ├── context_assembler.py            # Builds prompt from retrieved chunks
│   ├── prompt_manager.py               # Loads & renders YAML prompt templates
│   ├── llm_client.py                   # Abstraction over Claude / Ollama / OpenAI
│   ├── citation_handler.py             # Injects & validates source citations
│   ├── streaming_handler.py            # Handles streaming LLM responses
│   └── response_formatter.py          # Formats final answer + source cards
│
├── evaluation/                         # RAG evaluation layer
│   ├── __init__.py
│   ├── ragas_eval.py                   # RAGAS metrics runner
│   ├── test_dataset_builder.py         # Generates golden Q&A pairs
│   ├── metrics.py                      # Custom metric helpers
│   ├── batch_evaluator.py              # Runs eval across full test set
│   └── data/
│       ├── golden_qa.json              # 50+ curated question-answer pairs
│       └── eval_results/               # Timestamped evaluation outputs
│
├── api/                                # FastAPI backend
│   ├── __init__.py
│   ├── main.py                         # App factory, middleware, startup hooks
│   ├── dependencies.py                 # Shared DI (retriever, llm, qdrant client)
│   ├── schemas.py                      # Pydantic v2 request/response models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── search.py                   # GET /search — semantic paper search
│   │   ├── chat.py                     # POST /chat — RAG Q&A endpoint
│   │   ├── papers.py                   # CRUD for paper metadata
│   │   └── ingest.py                   # POST /ingest — trigger ingestion run
│   └── middleware/
│       ├── __init__.py
│       ├── logging.py                  # Request/response logging
│       └── rate_limiter.py             # Simple token-bucket rate limiter
│
├── frontend/                           # Streamlit UI
│   ├── app.py                          # Entry point & page router
│   ├── components/
│   │   ├── __init__.py
│   │   ├── chat_interface.py           # Chat input + message history
│   │   ├── source_card.py              # Citation card component
│   │   ├── paper_card.py               # Paper metadata display card
│   │   └── eval_chart.py              # RAGAS metric bar charts
│   └── pages/
│       ├── 1_chat.py                   # Main Q&A chat page
│       ├── 2_explore.py                # Paper browser + filter UI
│       └── 3_evaluate.py               # Evaluation dashboard
│
├── config/                             # Configuration layer
│   ├── __init__.py
│   ├── settings.py                     # Pydantic Settings (reads .env)
│   └── prompts/
│       ├── qa_prompt.yaml              # Q&A system + user prompt
│       ├── summary_prompt.yaml         # Summarization prompt
│       ├── compare_prompt.yaml         # Multi-paper comparison prompt
│       └── hyde_prompt.yaml            # HyDE query expansion prompt
│
├── tests/                              # Test suite
│   ├── conftest.py                     # Shared fixtures
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_query_processor.py
│   │   ├── test_hybrid_retriever.py
│   │   ├── test_citation_handler.py
│   │   └── test_prompt_manager.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_retrieval_pipeline.py
│   │   └── test_api_endpoints.py
│   └── e2e/
│       └── test_full_rag_flow.py       # End-to-end query → answer test
│
├── notebooks/                          # Experimentation (not in CI)
│   ├── 01_arxiv_exploration.ipynb      # Explore ArXiv API & paper structure
│   ├── 02_chunking_experiments.ipynb   # Compare chunking strategies
│   ├── 03_embedding_comparison.ipynb   # Compare embedding models
│   ├── 04_retrieval_analysis.ipynb     # Analyze retrieval quality
│   └── 05_ragas_analysis.ipynb         # Deep-dive into RAGAS scores
│
├── docker/                             # Container configuration
│   ├── Dockerfile                      # Multi-stage: builder + runtime
│   ├── Dockerfile.frontend             # Streamlit app container
│   └── docker-compose.yml              # Orchestrates all services
│
├── .github/
│   └── workflows/
│       ├── ci.yml                      # PR checks: lint + test + type-check
│       └── build.yml                   # Build & push Docker images on merge
│
├── scripts/                            # One-off utility scripts
│   ├── ingest_arxiv.py                 # CLI: scrape & index papers by topic
│   ├── rebuild_index.py                # CLI: wipe & rebuild Qdrant collection
│   └── export_eval_report.py           # CLI: export RAGAS report to PDF
│
├── .env.example                        # Template for environment variables
├── .pre-commit-config.yaml             # Pre-commit hooks config
├── pyproject.toml                      # Project metadata, ruff, mypy config
├── requirements.txt                    # Production dependencies
├── requirements-dev.txt                # Dev/test dependencies
└── README.md                           # Project overview + quickstart
```

---

## 3. Layer-by-Layer Breakdown

---

### 3.1 Ingestion

**Purpose**: Pull papers from ArXiv, download PDFs, extract clean text, and
detect document sections.

```
ingestion/
├── arxiv_scraper.py          ← entry point for paper discovery
├── pdf_downloader.py         ← fetches PDFs, skips already-downloaded
├── pdf_parser.py             ← extracts raw text per page via PyMuPDF
├── section_detector.py       ← maps page text → {abstract, intro, method...}
├── metadata_extractor.py     ← title, authors, year, category, DOI
├── citation_extractor.py     ← parses reference list
├── data_cleaner.py           ← removes headers/footers, LaTeX artifacts
└── pipeline.py               ← wires all steps together; called by CLI
```

**Key classes & functions:**

| File | Key export | What it does |
|------|-----------|--------------|
| `arxiv_scraper.py` | `ArxivScraper.search(query, max_results)` | Calls `arxiv` lib, returns list of `Paper` dicts |
| `pdf_parser.py` | `PDFParser.extract(path) → RawDocument` | Page-by-page text via `fitz.open()` |
| `section_detector.py` | `SectionDetector.detect(raw_doc) → SectionedDoc` | Regex + heading heuristics to label sections |
| `metadata_extractor.py` | `extract_metadata(arxiv_result) → PaperMeta` | Parses ArXiv API response fields |
| `data_cleaner.py` | `clean_text(text) → str` | Strips line artifacts, normalizes whitespace |
| `pipeline.py` | `run_ingestion(query, max_results)` | Full run: scrape → download → parse → clean → save |

---

### 3.2 Processing

**Purpose**: Chunk documents by section, generate embeddings, build BM25
index, and upsert everything into Qdrant.

```
processing/
├── chunker.py           ← produces Chunk objects with section labels
├── embedder.py          ← encodes chunks via sentence-transformers
├── bm25_indexer.py      ← builds BM25Okapi index, saves to disk
├── qdrant_indexer.py    ← upserts PointStruct into Qdrant collection
├── metadata_tagger.py   ← enriches each vector with paper payload
└── pipeline.py          ← orchestrates chunk → embed → index
```

**Chunk schema** (what flows between processing steps):

```python
@dataclass
class Chunk:
    chunk_id:   str         # uuid
    paper_id:   str         # ArXiv ID
    section:    str         # "abstract" | "introduction" | "method" | ...
    text:       str         # raw chunk text
    start_char: int
    end_char:   int
    metadata:   PaperMeta   # author, year, title, url
```

**Chunking strategy** (in `chunker.py`):

```
1. Section-aware primary split
   ├── Each detected section becomes its own unit
   ├── If section > MAX_TOKENS → sliding window (512 tokens, 50% overlap)
   └── If section < MIN_TOKENS → merge with adjacent section

2. Fallback: sliding window on unsectioned text
   ├── Window size: 400 tokens
   └── Overlap: 80 tokens
```

**Qdrant collection schema** (payload per vector):

```json
{
  "paper_id":  "2310.12345",
  "title":     "Attention Is All You Need",
  "authors":   ["Vaswani", "Shazeer"],
  "year":      2017,
  "section":   "method",
  "chunk_id":  "uuid",
  "text":      "...",
  "url":       "https://arxiv.org/abs/2310.12345"
}
```

---

### 3.3 Retrieval

**Purpose**: Given a user query, retrieve the most relevant chunks using
hybrid search, then rerank for precision.

```
retrieval/
├── query_processor.py      ← clean + expand the query
├── dense_retriever.py      ← Qdrant cosine similarity search
├── sparse_retriever.py     ← BM25 keyword search
├── hybrid_retriever.py     ← RRF fusion of both
├── reranker.py             ← cross-encoder (ms-marco-MiniLM)
├── mmr_sampler.py          ← diversity via Maximal Marginal Relevance
└── retriever_factory.py    ← builds the right retriever from config
```

**Retrieval pipeline:**

```
User query
    │
    ▼
query_processor.py
  ├── normalize & lowercase
  ├── remove stop words for BM25
  └── HyDE: generate hypothetical answer → use as expanded query
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
dense_retriever.py                        sparse_retriever.py
  └── Qdrant top-50 by cosine             └── BM25 top-50 by score
    │                                              │
    └──────────────────────────────────────────────┘
                        │
                        ▼
              hybrid_retriever.py
              └── RRF: score(d) = Σ 1/(k + rank_i(d))
                  k=60, merge & sort → top-20
                        │
                        ▼
                  reranker.py
                  └── cross-encoder scores all 20
                      → return top-5 or top-8
                        │
                        ▼
                  mmr_sampler.py (optional)
                  └── ensure topic diversity
```

**HyDE (Hypothetical Document Embeddings):**
Instead of embedding the raw query, the system asks the LLM to generate
a short hypothetical answer, then embeds that. This dramatically improves
recall for technical queries.

---

### 3.4 Generation

**Purpose**: Assemble retrieved chunks into a prompt, call the LLM, and
return a cited, grounded answer.

```
generation/
├── context_assembler.py     ← formats chunks into a prompt context block
├── prompt_manager.py        ← loads YAML templates, renders with Jinja2
├── llm_client.py            ← unified interface: Claude / Ollama / OpenAI
├── citation_handler.py      ← validates citations, builds source list
├── streaming_handler.py     ← AsyncGenerator for streaming tokens
└── response_formatter.py   ← structures final output
```

**Prompt structure** (from `prompts/qa_prompt.yaml`):

```
System:
  You are a research assistant specializing in AI papers. Answer the
  user's question using ONLY the provided context. Cite sources using
  [1], [2], etc. If the answer is not in the context, say so.

Context:
  [1] Title: Attention Is All You Need | Authors: Vaswani et al. | Year: 2017
  Section: method
  "The encoder maps an input sequence of symbol representations..."

  [2] Title: BERT | Authors: Devlin et al. | Year: 2019
  Section: abstract
  "We introduce a new language representation model called BERT..."

User: {user_query}
```

**LLM Client abstraction** (in `llm_client.py`):

```python
class LLMClient:
    # Switches provider based on config.LLM_PROVIDER
    def complete(prompt, stream=False) → str | AsyncGenerator
    def embed(text) → list[float]

# Supported providers
class ClaudeClient(LLMClient): ...    # anthropic SDK
class OllamaClient(LLMClient): ...   # local models via ollama
class OpenAIClient(LLMClient): ...   # optional GPT-4 fallback
```

---

### 3.5 Evaluation

**Purpose**: Measure RAG quality using RAGAS metrics on a golden test set.

```
evaluation/
├── ragas_eval.py            ← runs RAGAS on a batch of Q&A pairs
├── test_dataset_builder.py  ← semi-automated golden Q&A generation
├── metrics.py               ← custom metric helpers (MRR, Hit@k)
├── batch_evaluator.py       ← runs full eval loop, saves timestamped results
└── data/
    ├── golden_qa.json       ← 50+ curated question/answer/context triples
    └── eval_results/        ← JSON + CSV outputs per eval run
```

**RAGAS metrics tracked:**

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Faithfulness | Answer contains only facts from context | > 0.85 |
| Answer Relevancy | Answer addresses the question | > 0.80 |
| Context Precision | Retrieved docs are relevant | > 0.75 |
| Context Recall | All needed docs were retrieved | > 0.70 |
| Answer Correctness | Answer matches ground truth | > 0.75 |

**Golden Q&A format** (`golden_qa.json`):

```json
[
  {
    "question": "What attention mechanism does the Transformer use?",
    "ground_truth": "Scaled dot-product attention with multi-head attention",
    "reference_paper_ids": ["1706.03762"]
  }
]
```

---

### 3.6 API

**Purpose**: Expose all functionality over HTTP for the frontend and
external consumers.

```
api/
├── main.py              ← FastAPI app, lifespan hooks, CORS, middleware
├── dependencies.py      ← DI providers (qdrant client, retriever, llm)
├── schemas.py           ← Pydantic v2 models for all endpoints
├── routes/
│   ├── search.py        ← GET  /api/v1/search?q=...&limit=10
│   ├── chat.py          ← POST /api/v1/chat  { question, history }
│   ├── papers.py        ← GET  /api/v1/papers/{paper_id}
│   └── ingest.py        ← POST /api/v1/ingest { query, max_results }
└── middleware/
    ├── logging.py       ← logs request id, latency, status
    └── rate_limiter.py  ← 60 req/min per IP
```

**Key endpoints:**

```
GET  /api/v1/search
  Query params: q (str), limit (int=5), year_from (int), category (str)
  Response: { results: [{ chunk, paper_meta, score }] }

POST /api/v1/chat
  Body:    { question: str, history: [Message], top_k: int=5 }
  Response: { answer: str, sources: [PaperMeta], latency_ms: int }

POST /api/v1/ingest
  Body:    { query: str, max_results: int=50, category: str }
  Response: { ingested: int, indexed: int, skipped: int }

GET  /api/v1/health
  Response: { status: "ok", qdrant: bool, llm: bool }
```

---

### 3.7 Frontend

**Purpose**: User-facing Streamlit app with three pages.

```
frontend/
├── app.py                  ← sets page config, nav, shared state
├── components/
│   ├── chat_interface.py   ← renders message history + input box
│   ├── source_card.py      ← expandable card: title, authors, section, excerpt
│   ├── paper_card.py       ← paper overview: title, year, abstract snippet
│   └── eval_chart.py      ← Plotly bar chart for RAGAS metrics
└── pages/
    ├── 1_chat.py           ← main Q&A page, calls /api/v1/chat
    ├── 2_explore.py        ← paper browser, filter by year/category/author
    └── 3_evaluate.py       ← trigger eval run + display RAGAS dashboard
```

**Page: Chat (1_chat.py)**
```
┌──────────────────────────────────────────────────────┐
│  Ask anything about AI research papers               │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │ User: What is RLHF?                            │  │
│  │                                                │  │
│  │ Assistant: RLHF (Reinforcement Learning from  │  │
│  │ Human Feedback) is a training technique...    │  │
│  │                                                │  │
│  │ Sources:                                       │  │
│  │  [1] Ouyang et al., 2022 — InstructGPT        │  │
│  │  [2] Stiennon et al., 2020 — Summarization    │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  [____ Ask a question about AI papers ____________]  │
└──────────────────────────────────────────────────────┘
```

---

### 3.8 Config & Infra

**`config/settings.py`** — Pydantic BaseSettings, reads from `.env`:

```python
class Settings(BaseSettings):
    # LLM
    ANTHROPIC_API_KEY:  str
    LLM_PROVIDER:       str = "claude"          # claude | ollama | openai
    LLM_MODEL:          str = "claude-sonnet-4-6"

    # Embeddings
    EMBEDDING_MODEL:    str = "all-MiniLM-L6-v2"  # sentence-transformers
    EMBEDDING_DIM:      int = 384

    # Qdrant
    QDRANT_HOST:        str = "localhost"
    QDRANT_PORT:        int = 6333
    QDRANT_COLLECTION:  str = "ai_papers"

    # Retrieval
    DENSE_TOP_K:        int = 50
    SPARSE_TOP_K:       int = 50
    HYBRID_TOP_K:       int = 20
    RERANK_TOP_K:       int = 8

    # Chunking
    MAX_CHUNK_TOKENS:   int = 512
    CHUNK_OVERLAP:      int = 80

    # API
    API_HOST:           str = "0.0.0.0"
    API_PORT:           int = 8000
    RATE_LIMIT_RPM:     int = 60

    class Config:
        env_file = ".env"
```

---

## 4. Data Flow

### Ingestion flow (run once / on-demand)

```
ArXiv API
    │  arxiv.search(query, max_results)
    ▼
[arxiv_scraper.py]  →  List[Paper]
    │  paper.download_pdf(dirpath)
    ▼
[pdf_downloader.py]  →  data/raw/arxiv/{category}/{paper_id}.pdf
    │  fitz.open(pdf_path)
    ▼
[pdf_parser.py]  →  RawDocument { pages: [str] }
    │  heading detection heuristics
    ▼
[section_detector.py]  →  SectionedDoc { abstract, intro, method... }
    │  regex + field parsing
    ▼
[metadata_extractor.py]  →  PaperMeta { title, authors, year, doi }
    │  clean & normalize text
    ▼
[data_cleaner.py]  →  cleaned SectionedDoc
    │  save to data/processed/chunks/
    ▼
[chunker.py]  →  List[Chunk]
    │  encode with sentence-transformers
    ▼
[embedder.py]  →  List[Chunk + vector]
    │  upsert to Qdrant
    ├─▶ [qdrant_indexer.py]  →  Qdrant collection "ai_papers"
    └─▶ [bm25_indexer.py]   →  data/processed/bm25_index.pkl
```

### Query flow (real-time)

```
User query (str)
    │
    ▼
[query_processor.py]
  ├── clean query
  ├── HyDE: LLM → hypothetical answer → embed
  └── returns: clean_query (str) + hyde_vector (List[float])
    │
    ├─────────────────────────────────┐
    ▼                                 ▼
[dense_retriever.py]         [sparse_retriever.py]
  Qdrant.search(                BM25.get_scores(
    vector=hyde_vector,           tokenized_query
    limit=50                    )
  )                             top-50 chunk ids
  top-50 chunks
    │                                 │
    └─────────────────────────────────┘
                    │
                    ▼
          [hybrid_retriever.py]
            RRF fusion → top-20
                    │
                    ▼
             [reranker.py]
             cross-encoder → top-8
                    │
                    ▼
         [context_assembler.py]
         formats: numbered context blocks
                    │
                    ▼
           [prompt_manager.py]
           renders qa_prompt.yaml
                    │
                    ▼
            [llm_client.py]
            ClaudeClient.complete()
                    │
                    ▼
          [citation_handler.py]
          validates [1][2] references
                    │
                    ▼
         [response_formatter.py]
         { answer, sources, latency }
                    │
                    ▼
           FastAPI /api/v1/chat
           or Streamlit page
```

---

## 5. Key Design Decisions

### Section-aware chunking

Most RAG projects chunk by fixed token windows. This project chunks
by detected section instead:

- Abstract (always kept as one chunk — it's the most dense summary)
- Introduction
- Related Work
- Methodology / Method
- Experiments / Results
- Conclusion

**Why**: A question like "what dataset did they use?" should retrieve
the Experiments section, not a random page slice. Section labels are
stored in the Qdrant payload and used as metadata filters.

### Hybrid retrieval with RRF

Dense search alone misses exact keyword matches (model names, acronyms,
paper titles). BM25 alone misses semantic similarity. RRF combines both
without requiring score normalization:

```
score(doc) = Σ  1 / (k + rank_i(doc))
             i
k = 60  (prevents domination by very high-ranked docs)
```

### Cross-encoder reranking

Bi-encoder retrieval (dense) is fast but imprecise. The cross-encoder
sees the full (query, chunk) pair and produces a much more accurate
relevance score. Running it on only the top-20 candidates keeps latency
manageable (~300 ms for 20 pairs).

### RAGAS evaluation

Rather than eyeballing outputs, the project uses RAGAS to score every
retrieval configuration. This lets you compare:
- Chunking strategy A vs B
- Dense-only vs hybrid vs hybrid+rerank
- Prompt template variations

---

## 6. Environment Variables

Create a `.env` file from `.env.example`:

```env
# LLM
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=claude
LLM_MODEL=claude-sonnet-4-6

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=ai_papers

# Retrieval tuning
DENSE_TOP_K=50
SPARSE_TOP_K=50
HYBRID_TOP_K=20
RERANK_TOP_K=8

# Chunking
MAX_CHUNK_TOKENS=512
CHUNK_OVERLAP=80

# API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 7. Docker Services

`docker/docker-compose.yml` runs 4 services:

```yaml
services:

  qdrant:                       # Vector database
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrant_data:/qdrant/storage"]

  api:                          # FastAPI backend
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [qdrant]
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000

  frontend:                     # Streamlit UI
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports: ["8501:8501"]
    env_file: .env
    depends_on: [api]
    command: streamlit run frontend/app.py --server.port 8501

  redis:                        # Rate limiter / session cache
    image: redis:7-alpine
    ports: ["6379:6379"]

volumes:
  qdrant_data:
```

**Start everything:**
```bash
docker compose up --build
# API   → http://localhost:8000/docs
# UI    → http://localhost:8501
# Qdrant dashboard → http://localhost:6333/dashboard
```

---

## 8. CI/CD Pipeline

`.github/workflows/ci.yml` runs on every pull request:

```
Steps:
  1. Checkout code
  2. Set up Python 3.11
  3. Install dependencies (pip install -r requirements-dev.txt)
  4. ruff check .               ← linting
  5. black --check .            ← formatting
  6. mypy .                     ← type checking
  7. pytest tests/unit/         ← fast unit tests
  8. pytest tests/integration/  ← integration tests (with Qdrant mock)
```

`.pre-commit-config.yaml` runs on every local commit:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks: [ruff, ruff-format]
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks: [mypy]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks: [trailing-whitespace, end-of-file-fixer, check-yaml]
```

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/your-username/rag-research-assistant
cd rag-research-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your API keys

# 2. Start infrastructure
docker compose up qdrant redis -d

# 3. Ingest some papers
python scripts/ingest_arxiv.py --query "retrieval augmented generation" --max 100

# 4. Start the API
uvicorn api.main:app --reload

# 5. Start the UI (in another terminal)
streamlit run frontend/app.py
```

---

*Architecture version 1.0 — RAG Research Assistant*
