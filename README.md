# RAG AI Research Papers Assistant

This project is a research-paper ingestion system for a future RAG assistant. It discovers papers from a user research query, downloads PDFs, parses and cleans their content, detects paper sections, extracts metadata and citations, and saves structured JSON artifacts for later chunking, embedding, indexing, and retrieval.

## Current Default Architecture

```text
User research query
  -> Groq-backed alphaXiv query planner when GROQ_API_KEY is set
  -> alphaXiv MCP paper discovery
  -> if alphaXiv MCP fails: direct arXiv scraper fallback
  -> normalized Paper metadata / arXiv IDs / PDF URLs
  -> PDF download
  -> PDF parsing with PyMuPDF
  -> section detection
  -> metadata extraction
  -> citation extraction
  -> data cleaning
  -> processed JSON saved for future chunking / embedding / RAG
```

The default discovery provider is `auto`, implemented by `AlphaXivThenArxivDiscovery`. It tries `AlphaXivMCPPaperDiscovery` first and falls back to `ArxivPaperDiscovery` if alphaXiv MCP is unavailable, returns no parseable results, times out, or raises an integration error.

## Main Modules

| File | Purpose |
|---|---|
| `ingestion/paper_discovery.py` | Discovery providers, Groq alphaXiv planning, alphaXiv MCP integration, arXiv fallback, scoring, deduplication |
| `ingestion/pipeline.py` | End-to-end ingestion orchestration and CLI |
| `ingestion/arxiv_scraper.py` | Direct arXiv API client and normalized `Paper` schema |
| `ingestion/pdf_downloader.py` | PDF download and local raw PDF storage |
| `ingestion/pdf_parser.py` | PyMuPDF text extraction into page-level raw documents |
| `ingestion/section_detector.py` | Heuristic section detection for abstract, introduction, method, experiments, conclusion, references |
| `ingestion/metadata_extractor.py` | Paper-level metadata normalization |
| `ingestion/citation_extractor.py` | Reference list and citation extraction |
| `ingestion/data_cleaner.py` | PDF text cleanup and normalization |
| `tests/unit/test_ingestion.py` | Unit tests for ingestion utilities and discovery fallback behavior |

## Discovery Providers

- `auto`: default production path. alphaXiv MCP first, direct arXiv fallback.
- `alphaxiv-mcp`: alphaXiv MCP only.
- `arxiv`: direct arXiv API only.
- `research-apis`: arXiv plus optional Semantic Scholar and OpenAlex enrichment.
- `feyman`: optional legacy/local Feyman integration.

## alphaXiv MCP Setup

The alphaXiv provider expects a local MCP bridge command. By default it uses:

```bash
npx -y mcp-remote https://api.alphaxiv.org/mcp/v1
```

You can override it with:

```bash
set ALPHAXIV_MCP_COMMAND=npx -y mcp-remote https://api.alphaxiv.org/mcp/v1
```

If `GROQ_API_KEY` is set, the alphaXiv provider uses Groq to produce better `discover_papers` arguments: concise keywords, a richer semantic question, and difficulty tuning. If Groq is not configured or fails, local query-planning heuristics are used.

## Run Ingestion

Default architecture:

```bash
python -m ingestion.pipeline --query "retrieval augmented generation" --max-results 5
```

Explicit providers:

```bash
python -m ingestion.pipeline --query "retrieval augmented generation" --max-results 5 --discovery-provider auto
python -m ingestion.pipeline --query "retrieval augmented generation" --max-results 5 --discovery-provider alphaxiv-mcp
python -m ingestion.pipeline --query "retrieval augmented generation" --max-results 5 --discovery-provider arxiv
```

Outputs are written to:

- `data/raw/arxiv/{category}/{paper_id}.pdf`
- `data/metadata/papers.json`
- `data/processed/raw_text/{category}/{paper_id}.json`

## Processed JSON Shape

Each processed paper includes:

- `paper_id`
- normalized `metadata`
- `raw_document` with page text
- cleaned `sections`
- `section_spans`
- extracted `citations`

These artifacts are intentionally ready for the next layer: section-aware chunking, embeddings, vector indexing, and RAG retrieval.

## Tests

```bash
python -m pytest tests\unit\test_ingestion.py
```

Current local verification: `13 passed`. Pytest may emit a Windows cache warning if the sandbox cannot write `.pytest_cache`; that warning does not indicate test failure.

## Current Scope

Implemented now:

- discovery orchestration
- alphaXiv MCP integration
- Groq-assisted alphaXiv query planning
- arXiv fallback discovery
- PDF download and parsing
- section, metadata, citation extraction
- data cleaning
- structured output saving

Planned later:

- section-aware chunking
- embeddings
- BM25 and vector indexing
- retrieval and reranking
- grounded generation
- API and UI layers
