"""Sparse BM25 indexing for research-paper chunks."""

from __future__ import annotations

import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk

_LEGACY_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_TECHNICAL_TOKEN_PATTERN = re.compile(
    r"(?:[^\W_]+(?:[._#-][^\W_]+)*(?:\+\+)?|\d+(?:\.\d+)*)", re.UNICODE
)
DEFAULT_PREPROCESSING_CONFIG: dict[str, Any] = {
    "tokenizer": "technical_terms_v2",
    "lowercase": True,
    "stop_words_removed": False,
}
LEGACY_PREPROCESSING_CONFIG: dict[str, Any] = {
    "tokenizer": "word_v1",
    "lowercase": True,
    "stop_words_removed": False,
}
INDEX_VERSION = "2.0"


class BM25Indexer:
    """Build, query, save, and restore a BM25 index over chunks."""

    def __init__(self) -> None:
        self.index: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []
        self.documents: list[list[str]] = []
        self.preprocessing_config = dict(DEFAULT_PREPROCESSING_CONFIG)
        self.version = INDEX_VERSION
        self.created_at: str | None = None

    @staticmethod
    def tokenize(
        text: str, preprocessing_config: Mapping[str, Any] | None = None
    ) -> list[str]:
        """Normalize text using the indexing-time tokenizer configuration."""

        if not isinstance(text, str):
            return []
        config = dict(preprocessing_config or DEFAULT_PREPROCESSING_CONFIG)
        normalized = " ".join(text.split())
        if config.get("lowercase", True):
            normalized = normalized.casefold()
        tokenizer = config.get("tokenizer", "technical_terms_v2")
        if tokenizer == "word_v1":
            return _LEGACY_WORD_PATTERN.findall(normalized)
        if tokenizer != "technical_terms_v2":
            raise ValueError(f"Unsupported BM25 tokenizer: {tokenizer!r}")
        return _TECHNICAL_TOKEN_PATTERN.findall(normalized)

    def build(self, chunks: list[Chunk]) -> None:
        """Build an index from non-empty chunks, replacing any existing index."""

        valid_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.text, str) and chunk.text.strip()
        ]
        self.chunks = [chunk.to_dict() for chunk in valid_chunks]
        self.documents = [
            self.tokenize(chunk["text"], self.preprocessing_config)
            for chunk in self.chunks
        ]
        self.index = BM25Okapi(self.documents) if self.documents else None
        self.created_at = datetime.now(timezone.utc).isoformat()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return up to `top_k` chunks ranked by BM25 score."""

        query_tokens = self.tokenize(query, self.preprocessing_config)
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
        """Serialize the index, corpus, ordered chunk mapping, and settings."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            # Keep the v1 keys so older project code can still read new files.
            "index": self.index,
            "bm25": self.index,
            "documents": self.documents,
            "chunks": self.chunks,
            "chunk_ids": [str(chunk.get("chunk_id", "")) for chunk in self.chunks],
            "metadata": [dict(chunk.get("metadata") or {}) for chunk in self.chunks],
            "preprocessing_config": dict(self.preprocessing_config),
            "version": self.version,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }
        with output_path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Indexer":
        """Restore a trusted project-generated index saved by :meth:`save`.

        Pickle files can execute arbitrary code and must only be loaded from
        trusted local sources.
        """

        input_path = Path(path)
        with input_path.open("rb") as file:
            payload = pickle.load(file)

        if (
            not isinstance(payload, dict)
            or not ({"index", "bm25"} & payload.keys())
            or "chunks" not in payload
        ):
            raise ValueError("Invalid BM25 index file")

        instance = cls()
        instance.index = payload.get("bm25", payload.get("index"))
        instance.chunks = list(payload["chunks"])
        instance.version = str(payload.get("version", "1.0"))
        instance.created_at = payload.get("created_at")
        config = payload.get("preprocessing_config")
        instance.preprocessing_config = dict(
            config if isinstance(config, Mapping) else LEGACY_PREPROCESSING_CONFIG
        )
        documents = payload.get("documents")
        instance.documents = (
            [list(tokens) for tokens in documents]
            if isinstance(documents, list)
            else [
                instance.tokenize(
                    str(chunk.get("text", "")), instance.preprocessing_config
                )
                for chunk in instance.chunks
            ]
        )
        if len(instance.documents) != len(instance.chunks):
            raise ValueError(
                "Invalid BM25 index file: corpus and chunk mapping lengths differ"
            )
        return instance
