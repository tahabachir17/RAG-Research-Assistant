"""Sparse BM25 indexing for research-paper chunks."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk

_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


class BM25Indexer:
    """Build, query, save, and restore a BM25 index over chunks."""

    def __init__(self) -> None:
        self.index: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase `text` and return its word-like tokens."""

        if not isinstance(text, str):
            return []
        normalized = " ".join(text.lower().split())
        return _WORD_PATTERN.findall(normalized)

    def build(self, chunks: list[Chunk]) -> None:
        """Build an index from non-empty chunks, replacing any existing index."""

        valid_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.text, str) and chunk.text.strip()
        ]
        self.chunks = [chunk.to_dict() for chunk in valid_chunks]
        corpus = [self.tokenize(chunk["text"]) for chunk in self.chunks]
        self.index = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return up to `top_k` chunks ranked by BM25 score."""

        query_tokens = self.tokenize(query)
        if self.index is None or not self.chunks or not query_tokens or top_k <= 0:
            return []

        scores = np.asarray(self.index.get_scores(query_tokens), dtype=float)
        result_count = min(top_k, len(self.chunks))
        ranked_indices = np.argsort(-scores, kind="stable")[:result_count]

        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "metadata": dict(chunk.get("metadata") or {}),
                    "score": float(scores[index]),
                }
            )
        return results

    def save(self, path: str | Path) -> None:
        """Serialize the BM25 index and its chunk records to `path`."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"index": self.index, "chunks": self.chunks}
        with output_path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Indexer":
        """Restore an index saved by :meth:`save`.

        Pickle files must only be loaded from trusted sources.
        """

        input_path = Path(path)
        with input_path.open("rb") as file:
            payload = pickle.load(file)

        if (
            not isinstance(payload, dict)
            or "index" not in payload
            or "chunks" not in payload
        ):
            raise ValueError("Invalid BM25 index file")

        instance = cls()
        instance.index = payload["index"]
        instance.chunks = list(payload["chunks"])
        return instance
