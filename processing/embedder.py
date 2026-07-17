"""Dense embedding utilities for research-paper chunks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Encode text and :class:`Chunk` objects with a sentence transformer.

    The model is constructed once and reused for every call. A prebuilt model
    may be supplied for tests or dependency injection.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model if model is not None else SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for non-empty strings in `texts`.

        Blank strings are skipped rather than sent to the model. Empty input
        returns an empty two-dimensional NumPy array.
        """

        clean_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        if not clean_texts:
            return np.empty((0, 0), dtype=float)

        embeddings = self.model.encode(
            clean_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def encode_chunks(self, chunks: list[Chunk]) -> list[dict[str, Any]]:
        """Encode non-empty chunks and return JSON-ready records."""

        valid_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.text, str) and chunk.text.strip()
        ]
        if not valid_chunks:
            return []

        embeddings = self.encode_texts([chunk.text for chunk in valid_chunks])
        return [
            {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "section": chunk.section,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
                "embedding": embedding.tolist(),
            }
            for chunk, embedding in zip(valid_chunks, embeddings, strict=True)
        ]
