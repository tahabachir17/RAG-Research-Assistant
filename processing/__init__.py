"""Text processing primitives used by the indexing pipeline."""

from .chunker import Chunk, SectionAwareChunker, chunk_document

__all__ = ["Chunk", "SectionAwareChunker", "chunk_document"]
