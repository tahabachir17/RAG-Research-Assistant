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
LLM (Claude API)         generate a grounded, cited answer
    │
    ▼
Streamlit UI / API       display the answer with source cards
```

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Vector store | Qdrant |
| Embeddings | sentence-transformers |
| LLM | Claude API |
| UI | Streamlit |
| Containerization | Docker / Docker Compose |
| CI/CD | GitHub Actions |

## Quick start

```bash
# 1. Clone and set up the environment
git clone https://github.com/<your-username>/rag-research-assistant
cd rag-research-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your Anthropic API key

# 2. Start infrastructure
docker compose up qdrant redis -d

# 3. Ingest some papers
python scripts/ingest_arxiv.py --query "retrieval augmented generation" --max 100

# 4. Start the API
uvicorn api.main:app --reload

# 5. Start the UI (in another terminal)
streamlit run frontend/app.py
```

Or run everything at once:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Qdrant dashboard | http://localhost:6333/dashboard |

## Usage

**Ask a question via the UI:** open the Chat page and ask anything about the ingested corpus. Answers include inline citations `[1]`, `[2]` linking to the source papers.

**Ask a question via the API:**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What attention mechanism does the Transformer use?"}'
```

**Ingest more papers:**

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"query": "large language model alignment", "max_results": 50}'
```

## Project structure

```
rag-research-assistant/
├── ingestion/      # ArXiv scraping, PDF parsing, section detection
├── processing/     # Chunking, embedding, BM25 + Qdrant indexing
├── retrieval/      # Hybrid search, RRF fusion, reranking
├── generation/      # Prompt assembly, LLM client, citation validation
├── evaluation/      # RAGAS metrics, golden Q&A set, batch evaluation
├── api/            # FastAPI backend
├── frontend/        # Streamlit UI (chat, explore, evaluate)
├── config/          # Settings + prompt templates
├── tests/           # Unit, integration, and end-to-end tests
└── docker/           # Dockerfiles + docker-compose.yml
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full layer-by-layer breakdown, and [`CAHIER_DES_CHARGES.md`](CAHIER_DES_CHARGES.md) for the project specification.

## Evaluation

Retrieval and generation quality are tracked with [RAGAS](https://github.com/explodinggradients/ragas) against a curated set of question/answer/context triples.

| Metric | Target |
|---|---|
| Faithfulness | > 0.85 |
| Answer Relevancy | > 0.80 |
| Context Precision | > 0.75 |
| Context Recall | > 0.70 |
| Answer Correctness | > 0.75 |

Run the evaluation suite:

```bash
python -m evaluation.batch_evaluator
```

Results are saved to `evaluation/data/eval_results/` and viewable on the Evaluate page of the UI.

## Roadmap

Deprioritized for v1, planned for later:

- Diversity-aware retrieval (MMR sampling)
- Citation graph visualization
- Support for non-ArXiv sources (manual PDF upload, other repositories)
- Multi-user auth and saved conversations

## Contributing

Issues and pull requests are welcome. Before opening a PR, please make sure:

```bash
ruff check .
black --check .
mypy .
pytest tests/unit/
```

all pass — these are enforced in CI on every pull request.

## License

[MIT](LICENSE)

---

Built by [Taha](https://github.com/<your-username>) as an open-source exploration of production-grade RAG systems.
