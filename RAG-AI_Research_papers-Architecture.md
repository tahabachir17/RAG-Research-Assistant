# RAG AI Research Papers Assistant Architecture

This document describes the architecture implemented in the current repository and the intended next layers for the RAG assistant.

## Implemented Ingestion Flow

```text
User research query
  -> LLM query planner using Groq API when GROQ_API_KEY is configured
  -> alphaXiv MCP paper discovery
  -> if alphaXiv MCP fails: fallback to arXiv scraper
  -> selected paper metadata / paper IDs / PDF URLs
  -> ingestion pipeline
  -> PDF download
  -> PDF parsing
  -> section detection
  -> metadata extraction
  -> citation extraction
  -> data cleaning
  -> save processed output for later chunking / embedding / RAG
```

## Code Map

```text
ingestion/
  arxiv_scraper.py        Direct arXiv API client and Paper schema
  paper_discovery.py      Discovery providers and fallback orchestration
  pdf_downloader.py       PDF downloader
  pdf_parser.py           PyMuPDF parser
  section_detector.py     Section heading detection
  metadata_extractor.py   Paper metadata normalization
  citation_extractor.py   Reference extraction
  data_cleaner.py         Text cleanup
  pipeline.py             End-to-end ingestion CLI
```

## Discovery Layer

The default provider is `auto`, which maps to `AlphaXivThenArxivDiscovery`.

`AlphaXivThenArxivDiscovery` performs:

1. Validate the user query and result limit.
2. Call `AlphaXivMCPPaperDiscovery`.
3. Use Groq planning inside the alphaXiv provider when `GROQ_API_KEY` is available.
4. Normalize alphaXiv MCP result payloads into the shared `Paper` dataclass.
5. Score and deduplicate papers locally.
6. If any alphaXiv step fails, call `ArxivPaperDiscovery` as a fallback.
7. Add discovery provenance to each paper record.

Fallback papers include metadata fields such as `discovery_pipeline`, `discovery_provider_used`, and `fallback_reason`.

## Ingestion Pipeline

`IngestionPipeline.run()` receives normalized `Paper` records and executes the document-processing path:

1. `PDFDownloader.download()` stores PDFs under `data/raw/arxiv/{category}/`.
2. `PDFParser.extract()` uses PyMuPDF to extract page text.
3. `SectionDetector.detect()` builds section text and page spans.
4. `clean_sections()` normalizes PDF artifacts and noisy whitespace.
5. `extract_citations()` parses references from the references section.
6. `extract_metadata()` creates normalized paper metadata.
7. `_write_processed_document()` saves JSON under `data/processed/raw_text/{category}/`.
8. `data/metadata/papers.json` stores metadata for all processed papers in the run.

## Data Contracts

### Paper

The shared discovery schema is `ingestion.arxiv_scraper.Paper`:

```text
paper_id, title, authors, summary, published, updated,
primary_category, categories, pdf_url, entry_id,
doi, journal_ref, comment, metadata
```

### Processed Paper JSON

```json
{
  "paper_id": "2005.11401v4",
  "metadata": {},
  "raw_document": {},
  "sections": {},
  "section_spans": {},
  "citations": []
}
```

This output is the handoff point for future chunking, embedding, indexing, and RAG retrieval.

## Planned RAG Layers

The following packages are currently architectural targets rather than completed implementation:

```text
processing/   section-aware chunking, embeddings, BM25, vector indexing
retrieval/    dense retrieval, sparse retrieval, hybrid RRF, reranking
generation/   prompt assembly, LLM answer generation, citation validation
api/          FastAPI endpoints
frontend/     Streamlit or web UI
evaluation/   retrieval and answer quality evaluation
```

## Environment Variables

```env
ALPHAXIV_MCP_COMMAND=npx -y mcp-remote https://api.alphaxiv.org/mcp/v1
GROQ_API_KEY=optional
GROQ_MODEL=llama-3.1-70b-versatile
SEMANTIC_SCHOLAR_API_KEY=optional
OPENALEX_EMAIL=optional
PAPER_DISCOVERY_ENRICH_LIMIT=25
PAPER_DISCOVERY_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## CLI

```bash
python -m ingestion.pipeline --query "retrieval augmented generation" --max-results 5
```

Equivalent explicit default:

```bash
python -m ingestion.pipeline --query "retrieval augmented generation" --max-results 5 --discovery-provider auto
```

Other providers remain available for diagnostics and experiments:

```bash
--discovery-provider alphaxiv-mcp
--discovery-provider arxiv
--discovery-provider research-apis
--discovery-provider feyman
```

## Verification

Run:

```bash
python -m pytest tests\unit\test_ingestion.py
```

The current unit suite covers section detection, citation extraction, cleaning, discovery normalization, scoring, deduplication, alphaXiv argument planning, and alphaXiv-to-arXiv fallback behavior.
