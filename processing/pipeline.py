"""End-to-end processing pipeline from ingestion JSON to retrieval indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .bm25_indexer import BM25Indexer
from .chunker import Chunk, SectionAwareChunker
from .embedder import Embedder
from .metadata_tagger import MetadataTagger
from .qdrant_indexer import QdrantIndexer


class ProcessingPipeline:
    def __init__(
        self,
        *,
        chunker: SectionAwareChunker | None = None,
        embedder: Embedder | None = None,
        metadata_tagger: MetadataTagger | None = None,
        bm25_indexer: BM25Indexer | None = None,
        qdrant_indexer: QdrantIndexer | None = None,
    ) -> None:
        self.chunker = chunker or SectionAwareChunker()
        self.embedder = embedder
        self.metadata_tagger = metadata_tagger or MetadataTagger()
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.qdrant_indexer = qdrant_indexer

    def process_documents(
        self,
        documents: Iterable[dict[str, Any]],
        *,
        index_dense: bool = True,
        recreate_qdrant: bool = False,
    ) -> dict[str, Any]:
        chunks: list[Chunk] = []
        for document in documents:
            raw_chunks = self.chunker.chunk(document)
            chunks.extend(
                self.metadata_tagger.tag_chunks(
                    raw_chunks, document.get("metadata", {})
                )
            )
        self.bm25_indexer.build(chunks)
        embedding_records: list[dict[str, Any]] = []
        indexed = 0
        if index_dense:
            if self.embedder is None:
                self.embedder = Embedder()
            embedding_records = self.embedder.encode_chunks(chunks)
            if self.qdrant_indexer is not None:
                indexed = self.qdrant_indexer.index_embeddings(
                    embedding_records, recreate=recreate_qdrant
                )
        return {
            "chunks": chunks,
            "embedding_records": embedding_records,
            "bm25_documents": len(self.bm25_indexer.chunks),
            "qdrant_points": indexed,
        }

    def process_paths(
        self, paths: Iterable[str | Path], **kwargs: Any
    ) -> dict[str, Any]:
        documents = [
            json.loads(Path(path).read_text(encoding="utf-8")) for path in paths
        ]
        return self.process_documents(documents, **kwargs)
