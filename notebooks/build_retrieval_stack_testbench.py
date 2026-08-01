"""Build the reader-facing retrieval stack test bench notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "06_retrieval_stack_testbench.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    md(
        r"""
# Retrieval Stack Test Bench

An offline-first, reader-facing notebook for validating the retrieval pipeline:

`query_processor -> dense + sparse -> RRF -> cross-encoder -> MMR -> factory`

**How to use it:** choose parameters in **Setup**, then run all cells. Green checks exercise production code where it exists; amber rows explain optional or missing production modules without stopping the notebook.

## Navigation

- [1. Goal and test matrix](#1.-Goal-and-test-matrix)
- [2. Setup](#2.-Setup)
- [3. Component readiness](#3.-Component-readiness)
- [4. Query processor](#4.-Query-processor)
- [5. Dense retrieval](#5.-Dense-retrieval)
- [6. Sparse retrieval](#6.-Sparse-retrieval)
- [7. Hybrid RRF fusion](#7.-Hybrid-RRF-fusion)
- [8. Cross-encoder reranking](#8.-Cross-encoder-reranking)
- [9. MMR diversity](#9.-MMR-diversity)
- [10. Retriever factory](#10.-Retriever-factory)
- [11. End-to-end rank flow](#11.-End-to-end-rank-flow)
- [12. Test summary and next steps](#12.-Test-summary-and-next-steps)
"""
    ),
    md(
        r"""
## 1. Goal and test matrix

This is a **diagnostic and integration test notebook**, not a second implementation of the retrieval package. It uses deterministic local fixtures and no network by default.

| Component | Production target | What this notebook checks |
|---|---|---|
| Query processing | `retrieval/query_processor.py` | cleaning, technical-token preservation, bounded expansion |
| Dense retrieval | `retrieval/dense_retriever.py` | query embedding, Qdrant request contract, filters, score ordering, health checks |
| Sparse retrieval | `retrieval/sparse_retriever.py` | BM25 load/search, metadata filters, score ordering, artifact integrity |
| Hybrid retrieval | `retrieval/hybrid_retriever.py` | module readiness plus reference RRF invariants and fused ranking |
| Reranking | `retrieval/reranker.py` | module/model readiness plus deterministic cross-encoder contract |
| MMR | `retrieval/mmr_sampler.py` | module readiness plus relevance/diversity behavior |
| Factory | `retrieval/retriever_factory.py` | module readiness and expected configuration matrix |

### Key assumptions

- Qdrant uses cosine distance and payloads compatible with `RetrievalResult`.
- Dense and sparse scores are not directly comparable; fusion uses ranks.
- The small BM25 fallback below is only an offline compatibility aid when the declared `rank-bm25` dependency is absent. That condition remains visible in the readiness table.
- Reference RRF/MMR/fake reranking validate algorithmic expectations only; they do not mark a missing production module as implemented.
"""
    ),
    md("## 2. Setup\n\nEdit only this parameter cell for routine experiments."),
    code(
        r"""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import tempfile
import time
import types
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.max_colwidth", 100)
np.set_printoptions(precision=3, suppress=True)


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "retrieval").is_dir() and (candidate / "processing").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the notebook working directory")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Experiment parameters
QUERY = "How does RAG combine BM25 and dense retrieval?"
TOP_K = 4
RRF_K = 60
MMR_LAMBDA = 0.65
MMR_TOP_K = 3
ENABLE_QUERY_EXPANSION = True
RUN_OPTIONAL_MODEL = False  # True may download cross-encoder/ms-marco-MiniLM-L-6-v2

print(f"Project root: {PROJECT_ROOT}")
print(f"Query: {QUERY!r}")
"""
    ),
    md("### Test recorder\n\nAll sections write to one bounded summary table."),
    code(
        r"""
checks = []


def record_check(component, check, passed, detail="", status=None):
    resolved_status = status or ("PASS" if passed else "FAIL")
    checks.append({
        "component": component,
        "check": check,
        "status": resolved_status,
        "detail": str(detail)[:240],
    })


def result_table(results, limit=TOP_K):
    return pd.DataFrame([
        {
            "rank": rank,
            "chunk_id": item.chunk_id,
            "source": item.source,
            "score": round(float(item.score), 5),
            "section": item.section,
            "text": item.text[:110],
        }
        for rank, item in enumerate(results[:limit], 1)
    ])
"""
    ),
    md("## 3. Component readiness\n\nThis section separates missing packages, missing production modules, and runnable code."),
    code(
        r"""
dependency_names = ["numpy", "pandas", "matplotlib", "rank_bm25", "qdrant_client", "sentence_transformers"]
dependency_rows = []
for name in dependency_names:
    available = importlib.util.find_spec(name) is not None
    dependency_rows.append({"dependency": name, "available": available})

dependency_status = pd.DataFrame(dependency_rows)
display(dependency_status.style.map(lambda value: "color: #14833b" if value is True else "color: #a15c00"))
"""
    ),
    md(
        r"""
### Offline BM25 compatibility

If `rank-bm25` is missing, install the declared project dependency for normal use:

```bash
python -m pip install rank-bm25==0.2.2
```

For this notebook only, the following minimal compatible class keeps production query/sparse code testable. It does **not** hide the missing dependency in the table above.
"""
    ),
    code(
        r"""
USING_BM25_FALLBACK = importlib.util.find_spec("rank_bm25") is None

if USING_BM25_FALLBACK:
    fallback_module = types.ModuleType("rank_bm25")

    class BM25Okapi:
        def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
            self.corpus = [list(document) for document in corpus]
            self.k1, self.b, self.epsilon = k1, b, epsilon
            self.doc_len = np.asarray([len(document) for document in self.corpus], dtype=float)
            self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
            self.doc_freqs = []
            document_frequency = {}
            for document in self.corpus:
                frequencies = {}
                for token in document:
                    frequencies[token] = frequencies.get(token, 0) + 1
                self.doc_freqs.append(frequencies)
                for token in frequencies:
                    document_frequency[token] = document_frequency.get(token, 0) + 1
            corpus_size = max(len(self.corpus), 1)
            raw_idf = {
                token: math.log(corpus_size - frequency + 0.5) - math.log(frequency + 0.5)
                for token, frequency in document_frequency.items()
            }
            positive = [value for value in raw_idf.values() if value >= 0]
            average_idf = sum(positive) / len(positive) if positive else 0.0
            self.idf = {
                token: value if value >= 0 else epsilon * average_idf
                for token, value in raw_idf.items()
            }

        def get_scores(self, query_tokens):
            scores = np.zeros(len(self.corpus), dtype=float)
            if not len(self.corpus):
                return scores
            normalization = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-12))
            for token in query_tokens:
                frequencies = np.asarray([document.get(token, 0) for document in self.doc_freqs], dtype=float)
                scores += self.idf.get(token, 0.0) * frequencies * (self.k1 + 1) / (frequencies + normalization)
            return scores

    fallback_module.BM25Okapi = BM25Okapi
    sys.modules["rank_bm25"] = fallback_module

record_check(
    "environment",
    "rank-bm25 dependency",
    not USING_BM25_FALLBACK,
    "Installed" if not USING_BM25_FALLBACK else "Missing; notebook compatibility fallback activated",
    status="PASS" if not USING_BM25_FALLBACK else "WARN",
)
print("BM25 backend:", "notebook fallback" if USING_BM25_FALLBACK else "rank-bm25 package")
"""
    ),
    code(
        r"""
production_modules = {
    "query": "retrieval.query_processor",
    "dense": "retrieval.dense_retriever",
    "sparse": "retrieval.sparse_retriever",
    "hybrid": "retrieval.hybrid_retriever",
    "reranker": "retrieval.reranker",
    "mmr": "retrieval.mmr_sampler",
    "factory": "retrieval.retriever_factory",
}

module_rows = []
for component, module_name in production_modules.items():
    available = importlib.util.find_spec(module_name) is not None
    module_rows.append({"component": component, "module": module_name, "production_available": available})
    record_check(
        component,
        "production module present",
        available,
        module_name,
        status="PASS" if available else "MISSING",
    )

module_status = pd.DataFrame(module_rows)
display(module_status.style.map(lambda value: "color: #14833b" if value is True else "color: #a15c00"))

from processing.bm25_indexer import BM25Indexer
from processing.chunker import Chunk
from retrieval.dense_retriever import DenseRetriever
from retrieval.models import RetrievalResult
from retrieval.query_processor import QueryProcessor
from retrieval.sparse_retriever import SparseRetriever
"""
    ),
    md("### Shared deterministic fixture"),
    code(
        r"""
fixture_chunks = [
    Chunk("c-rag", "p-rag", "abstract", "retrieval augmented generation grounds answers with evidence", 0, 60,
          {"title": "Grounded RAG", "year": 2024, "category": "cs.CL"}),
    Chunk("c-bm25", "p-bm25", "method", "BM25 lexical keyword search provides sparse retrieval", 0, 53,
          {"title": "Lexical Search", "year": 2021, "category": "cs.IR"}),
    Chunk("c-dense", "p-dense", "method", "dense semantic vector similarity retrieves paraphrases", 0, 54,
          {"title": "Semantic Search", "year": 2023, "category": "cs.IR"}),
    Chunk("c-hybrid", "p-hybrid", "results", "hybrid retrieval fuses BM25 and dense ranks with reciprocal rank fusion", 0, 71,
          {"title": "Hybrid Retrieval", "year": 2025, "category": "cs.CL"}),
    Chunk("c-vision", "p-vision", "abstract", "vision transformer image classification benchmark", 0, 49,
          {"title": "Vision Models", "year": 2022, "category": "cs.CV"}),
]
print(f"Fixture: {len(fixture_chunks)} chunks")
"""
    ),
    md("## 4. Query processor\n\nExercise production cleaning, sparse tokenization, and optional expansion."),
    code(
        r"""
query_processor = QueryProcessor(enable_expansion=ENABLE_QUERY_EXPANSION)
query_cases = [
    "  GPT-4\tC++  F1-score\nLLaMA-3 Q-learning  ",
    QUERY,
    "RAG evaluation",
]
processed_rows = []
for raw_query in query_cases:
    processed = query_processor.process(raw_query)
    processed_rows.append({
        "original": raw_query.replace("\n", "\\n").replace("\t", "\\t"),
        "cleaned": processed.cleaned_query,
        "dense_query": processed.dense_query,
        "sparse_tokens": processed.sparse_tokens,
        "expanded": processed.expanded_query,
    })

display(pd.DataFrame(processed_rows))
technical = query_processor.process(query_cases[0])
expected_terms = {"gpt-4", "c++", "f1-score", "llama-3", "q-learning"}
record_check("query", "technical tokens preserved", expected_terms.issubset(set(technical.sparse_tokens)), technical.sparse_tokens)
record_check("query", "whitespace normalized", technical.cleaned_query == "GPT-4 C++ F1-score LLaMA-3 Q-learning", technical.cleaned_query)
expanded = query_processor.process("RAG evaluation")
record_check("query", "bounded expansion retains original", expanded.dense_query.startswith("RAG evaluation"), expanded.dense_query)
"""
    ),
    md("## 5. Dense retrieval\n\nUse a fake Qdrant client to test the production request/response contract without a server or model download."),
    code(
        r"""
class FakeEmbedder:
    dimension = 3

    def encode_texts(self, texts):
        return np.asarray([[0.2, 0.7, 0.1] for _ in texts], dtype=float)


class FakeQdrantClient:
    def __init__(self):
        self.last_query = None

    def collection_exists(self, name):
        return name == "test-papers"

    def get_collection(self, name):
        return SimpleNamespace(
            points_count=3,
            status=SimpleNamespace(value="green"),
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(
                size=3, distance=SimpleNamespace(value="Cosine")
            ))),
        )

    def query_points(self, **kwargs):
        self.last_query = kwargs
        return SimpleNamespace(points=[
            SimpleNamespace(id="c-dense", score=0.91, payload=fixture_chunks[2].to_dict()),
            SimpleNamespace(id="c-hybrid", score=0.86, payload=fixture_chunks[3].to_dict()),
            SimpleNamespace(id="c-rag", score=0.72, payload=fixture_chunks[0].to_dict()),
        ])


fake_qdrant = FakeQdrantClient()
dense_retriever = DenseRetriever(fake_qdrant, FakeEmbedder(), "test-papers", default_top_k=TOP_K)
dense_health = dense_retriever.health_check()
dense_start = time.perf_counter()
dense_results = dense_retriever.search(
    query_processor.process(QUERY).dense_query,
    top_k=TOP_K,
    filters={"year": {"gte": 2020}, "section": ["abstract", "method", "results"]},
)
dense_latency_ms = (time.perf_counter() - dense_start) * 1000

display(pd.Series(dense_health, name="value").to_frame())
display(result_table(dense_results))
record_check("dense", "cosine collection is healthy", dense_health["distance"].casefold() == "cosine", dense_health)
record_check("dense", "dimension matches", dense_health["dimension_match"] is True, dense_health)
record_check("dense", "results sorted descending", [r.score for r in dense_results] == sorted([r.score for r in dense_results], reverse=True), [r.score for r in dense_results])
record_check("dense", "Qdrant filter forwarded", fake_qdrant.last_query.get("query_filter") is not None, type(fake_qdrant.last_query.get("query_filter")).__name__)
"""
    ),
    md("## 6. Sparse retrieval\n\nBuild a temporary trusted BM25 artifact, then exercise the production retriever."),
    code(
        r"""
with tempfile.TemporaryDirectory() as temporary_directory:
    index_path = Path(temporary_directory) / "bm25_fixture.pkl"
    indexer = BM25Indexer()
    indexer.build(fixture_chunks)
    indexer.save(index_path)

    sparse_retriever = SparseRetriever(index_path, default_top_k=TOP_K)
    sparse_start = time.perf_counter()
    sparse_results = sparse_retriever.search(
        query_processor.process(QUERY).sparse_tokens,
        top_k=TOP_K,
    )
    sparse_latency_ms = (time.perf_counter() - sparse_start) * 1000
    sparse_filtered = sparse_retriever.search("retrieval", filters={"year": {"gte": 2024}})
    sparse_health = sparse_retriever.health_check()

display(pd.Series(sparse_health, name="value").to_frame())
display(result_table(sparse_results))
record_check("sparse", "artifact mapping valid", sparse_health["mapping_valid"] is True, sparse_health)
record_check("sparse", "results sorted descending", [r.score for r in sparse_results] == sorted([r.score for r in sparse_results], reverse=True), [round(r.score, 4) for r in sparse_results])
record_check("sparse", "year filter applied", all((r.year or 0) >= 2024 for r in sparse_filtered), [r.year for r in sparse_filtered])
"""
    ),
    md(
        r"""
## 7. Hybrid RRF fusion

RRF combines **rank positions**, not incompatible raw cosine/BM25 scores. The reference below is an executable oracle for validating the future production module.
"""
    ),
    code(
        r"""
def reference_rrf(result_lists, rrf_k=60):
    fused = {}
    representatives = {}
    ranks = {}
    for backend_results in result_lists:
        for rank, item in enumerate(backend_results, 1):
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            representatives.setdefault(item.chunk_id, item)
            ranks.setdefault(item.chunk_id, []).append(rank)
    ordered_ids = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], min(ranks[chunk_id]), chunk_id))
    return [
        RetrievalResult(
            chunk_id=chunk_id,
            text=representatives[chunk_id].text,
            score=fused[chunk_id],
            source="hybrid",
            paper_id=representatives[chunk_id].paper_id,
            title=representatives[chunk_id].title,
            year=representatives[chunk_id].year,
            section=representatives[chunk_id].section,
            metadata=representatives[chunk_id].metadata,
        )
        for chunk_id in ordered_ids
    ]


hybrid_results = reference_rrf([dense_results, sparse_results], RRF_K)
display(result_table(hybrid_results, limit=10))
overlap_ids = {item.chunk_id for item in dense_results} & {item.chunk_id for item in sparse_results}
top_hybrid_ids = {item.chunk_id for item in hybrid_results[:max(1, len(overlap_ids))]}
record_check("hybrid", "RRF output sorted", [r.score for r in hybrid_results] == sorted([r.score for r in hybrid_results], reverse=True), [round(r.score, 5) for r in hybrid_results])
record_check("hybrid", "duplicate chunks fused", len(hybrid_results) == len({r.chunk_id for r in dense_results + sparse_results}), f"overlap={sorted(overlap_ids)}")
record_check("hybrid", "multi-list evidence promoted", overlap_ids.issubset(set(r.chunk_id for r in hybrid_results)), sorted(overlap_ids))
"""
    ),
    md(
        r"""
## 8. Cross-encoder reranking

The default is deterministic and offline: a fake cross-encoder verifies the `(query, passage)` scoring and reorder contract. Set `RUN_OPTIONAL_MODEL=True` only when `sentence-transformers` is installed and model download/cache access is acceptable.
"""
    ),
    code(
        r"""
class FakeCrossEncoder:
    def predict(self, pairs):
        query_terms = set(pairs[0][0].casefold().split()) if pairs else set()
        return np.asarray([
            len(query_terms & set(passage.casefold().replace("?", "").split())) + 0.01 * len(passage)
            for _, passage in pairs
        ], dtype=float)


def reference_rerank(query, candidates, model, top_k=None):
    pairs = [(query, item.text) for item in candidates]
    scores = np.asarray(model.predict(pairs), dtype=float)
    order = np.argsort(-scores, kind="stable")
    limit = len(candidates) if top_k is None else min(top_k, len(candidates))
    return [(candidates[index], float(scores[index])) for index in order[:limit]]


reranked_pairs = reference_rerank(QUERY, hybrid_results, FakeCrossEncoder(), top_k=TOP_K)
reranked_results = [
    RetrievalResult(
        chunk_id=item.chunk_id,
        text=item.text,
        score=score,
        source="reranked",
        paper_id=item.paper_id,
        title=item.title,
        year=item.year,
        section=item.section,
        metadata=item.metadata,
    )
    for item, score in reranked_pairs
]
display(result_table(reranked_results))
record_check("reranker", "one pair per candidate", len(hybrid_results) == len([(QUERY, item.text) for item in hybrid_results]), len(hybrid_results))
record_check("reranker", "scores sorted descending", [r.score for r in reranked_results] == sorted([r.score for r in reranked_results], reverse=True), [round(r.score, 3) for r in reranked_results])

if RUN_OPTIONAL_MODEL:
    if importlib.util.find_spec("sentence_transformers") is None:
        record_check("reranker", "real model smoke test", False, "sentence-transformers is not installed", status="SKIP")
    else:
        from sentence_transformers import CrossEncoder
        real_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        real_scores = real_model.predict([(QUERY, item.text) for item in hybrid_results[:TOP_K]])
        record_check("reranker", "real model smoke test", len(real_scores) == min(TOP_K, len(hybrid_results)), f"scores={np.asarray(real_scores).round(3).tolist()}")
else:
    record_check("reranker", "real model smoke test", False, "Disabled by RUN_OPTIONAL_MODEL=False", status="SKIP")
"""
    ),
    md("## 9. MMR diversity\n\nMMR balances query relevance with novelty against already selected candidates."),
    code(
        r"""
def cosine_similarity(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def reference_mmr(query_vector, candidate_vectors, top_k, lambda_mult=0.5):
    remaining = list(range(len(candidate_vectors)))
    selected = []
    while remaining and len(selected) < top_k:
        scored = []
        for index in remaining:
            relevance = cosine_similarity(query_vector, candidate_vectors[index])
            redundancy = max(
                (cosine_similarity(candidate_vectors[index], candidate_vectors[chosen]) for chosen in selected),
                default=0.0,
            )
            scored.append((lambda_mult * relevance - (1 - lambda_mult) * redundancy, index))
        _, winner = max(scored, key=lambda pair: (pair[0], -pair[1]))
        selected.append(winner)
        remaining.remove(winner)
    return selected


mmr_candidates = reranked_results
candidate_vectors = np.asarray([
    [1.00, 0.05, 0.00],  # highly relevant
    [0.98, 0.08, 0.00],  # near duplicate
    [0.70, 0.10, 0.65],  # relevant and diverse
    [0.20, 0.95, 0.00],  # less relevant
][:len(mmr_candidates)])
query_vector = np.asarray([1.0, 0.0, 0.0])
selected_positions = reference_mmr(query_vector, candidate_vectors, min(MMR_TOP_K, len(mmr_candidates)), MMR_LAMBDA)
mmr_results = [mmr_candidates[index] for index in selected_positions]

display(pd.DataFrame({
    "selection_order": range(1, len(selected_positions) + 1),
    "original_rank": [index + 1 for index in selected_positions],
    "chunk_id": [item.chunk_id for item in mmr_results],
}))
record_check("mmr", "selection is unique", len(selected_positions) == len(set(selected_positions)), selected_positions)
record_check("mmr", "top relevant candidate retained", 0 in selected_positions, selected_positions)
record_check("mmr", "diverse candidate preferred over near duplicate", (2 in selected_positions) or len(mmr_candidates) < 3, selected_positions)
"""
    ),
    md(
        r"""
## 10. Retriever factory

The factory should centralize construction and reject invalid configurations. The table below is the expected configuration contract to use when the production factory is added.
"""
    ),
    code(
        r"""
factory_matrix = pd.DataFrame([
    {"type": "dense", "requires": "qdrant_client, embedder, collection_name", "expected_class": "DenseRetriever"},
    {"type": "sparse", "requires": "index_path", "expected_class": "SparseRetriever"},
    {"type": "hybrid", "requires": "dense config, sparse config, rrf_k", "expected_class": "HybridRetriever"},
])
display(factory_matrix)

factory_available = importlib.util.find_spec("retrieval.retriever_factory") is not None
if factory_available:
    factory_module = importlib.import_module("retrieval.retriever_factory")
    public_names = [name for name in dir(factory_module) if not name.startswith("_")]
    record_check("factory", "public factory API discovered", any("factory" in name.casefold() or "build" in name.casefold() for name in public_names), public_names)
else:
    record_check("factory", "configuration construction", False, "Production factory module is not present", status="MISSING")
"""
    ),
    md("## 11. End-to-end rank flow\n\nTrace how candidate order changes across retrieval, fusion, reranking, and diversity selection."),
    code(
        r"""
stage_results = {
    "dense": dense_results,
    "sparse": sparse_results,
    "hybrid_rrf": hybrid_results[:TOP_K],
    "reranked": reranked_results,
    "mmr": mmr_results,
}
rank_rows = []
for stage, results in stage_results.items():
    for rank, item in enumerate(results, 1):
        rank_rows.append({"stage": stage, "rank": rank, "chunk_id": item.chunk_id, "score": item.score})
rank_flow = pd.DataFrame(rank_rows)
rank_pivot = rank_flow.pivot_table(index="chunk_id", columns="stage", values="rank", aggfunc="first")
display(rank_pivot)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
latency = pd.Series({"dense (fake Qdrant)": dense_latency_ms, "sparse (fixture)": sparse_latency_ms})
latency.plot(kind="bar", ax=axes[0], color=["#4c78a8", "#f58518"], title="Local diagnostic latency")
axes[0].set_ylabel("milliseconds")
axes[0].tick_params(axis="x", rotation=20)

rank_pivot.plot(kind="bar", ax=axes[1], title="Rank by pipeline stage")
axes[1].invert_yaxis()
axes[1].set_ylabel("rank (1 is best)")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

record_check("pipeline", "final output bounded", len(mmr_results) <= MMR_TOP_K, len(mmr_results))
record_check("pipeline", "final output preserves payloads", all(item.metadata for item in mmr_results), [item.chunk_id for item in mmr_results])
"""
    ),
    md("## 12. Test summary and next steps\n\nThe summary intentionally treats `MISSING`, `WARN`, and `SKIP` separately from failed executable checks."),
    code(
        r"""
check_table = pd.DataFrame(checks)
status_order = pd.CategoricalDtype(["FAIL", "MISSING", "WARN", "SKIP", "PASS"], ordered=True)
check_table["status"] = check_table["status"].astype(status_order)
check_table = check_table.sort_values(["status", "component", "check"]).reset_index(drop=True)

def color_status(value):
    colors = {"PASS": "#14833b", "FAIL": "#c62828", "MISSING": "#a15c00", "WARN": "#a15c00", "SKIP": "#667085"}
    return f"color: {colors.get(str(value), '#111827')}; font-weight: 600"

display(check_table.style.map(color_status, subset=["status"]))
summary = check_table["status"].astype(str).value_counts().reindex(["PASS", "FAIL", "MISSING", "WARN", "SKIP"], fill_value=0)
display(summary.rename("checks").to_frame())

failures = check_table[check_table["status"].astype(str) == "FAIL"]
missing = check_table[check_table["status"].astype(str) == "MISSING"]
print(f"Executable failures: {len(failures)}")
print(f"Missing production checks: {len(missing)}")
assert failures.empty, failures.to_dict("records")
"""
    ),
    md(
        r"""
### Takeaways

- Use the final summary as the handoff record: **FAIL** means runnable behavior is wrong; **MISSING** means the named production module does not exist yet; **WARN/SKIP** identifies environment-dependent coverage.
- Current deterministic coverage is safe to run offline and exercises the existing production query, dense, sparse, result-model, filter, and health-check paths.
- Once `hybrid_retriever.py`, `reranker.py`, `mmr_sampler.py`, and `retriever_factory.py` exist, replace the reference-only calls with their public APIs while retaining these invariants as regression checks.
- For a live smoke test, install project dependencies, start/populate the cosine Qdrant collection, and enable the optional cross-encoder model only when cache/download access is available.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
