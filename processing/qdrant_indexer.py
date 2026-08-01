"""Qdrant dense-vector indexing for embedding records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:  # Module remains importable for lightweight/unit-test environments.
    QdrantClient = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]


@dataclass(slots=True)
class QdrantPayload:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "section": self.section,
            "text": self.text,
            "metadata": self.metadata,
        }


class QdrantIndexer:
    """Create a collection, upsert dense vectors, and query their payloads."""

    def __init__(
        self,
        collection_name: str = "research_papers",
        *,
        client: Any | None = None,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        distance: str = "cosine",
        batch_size: int = 128,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if client is None:
            if QdrantClient is None:
                raise ImportError("qdrant-client is required for Qdrant indexing")
            client = QdrantClient(url=url, api_key=api_key)
        self.client, self.collection_name = client, collection_name
        self.distance, self.batch_size = distance.lower(), batch_size

    def ensure_collection(self, vector_size: int, *, recreate: bool = False) -> None:
        if vector_size < 1:
            raise ValueError("vector_size must be positive")
        exists = self._collection_exists()
        config = _vector_params(vector_size, self.distance)
        if recreate and exists:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name, vectors_config=config
            )

    def index_embeddings(
        self, records: Iterable[Mapping[str, Any]], *, recreate: bool = False
    ) -> int:
        prepared = [dict(record) for record in records if _valid_record(record)]
        if not prepared:
            return 0
        vector_size = len(prepared[0]["embedding"])
        if any(len(record["embedding"]) != vector_size for record in prepared):
            raise ValueError("All embeddings must have the same dimension")
        self.ensure_collection(vector_size, recreate=recreate)
        for start in range(0, len(prepared), self.batch_size):
            points = [
                _point(record) for record in prepared[start : start + self.batch_size]
            ]
            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )
        return len(prepared)

    def index(
        self,
        chunks: Sequence[Any],
        embeddings: Sequence[Sequence[float]],
        *,
        recreate: bool = False,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal lengths")
        records = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            record = chunk.to_dict() if hasattr(chunk, "to_dict") else dict(chunk)
            record["embedding"] = list(embedding)
            records.append(record)
        return self.index_embeddings(records, recreate=recreate)

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        query_filter: Any | None = None,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or not vector:
            return []
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [
            {
                "id": str(getattr(hit, "id", "")),
                "score": float(getattr(hit, "score", 0.0)),
                "payload": dict(getattr(hit, "payload", {}) or {}),
                **({"vector": getattr(hit, "vector", None)} if with_vectors else {}),
            }
            for hit in hits
        ]

    def delete_collection(self) -> bool:
        if not self._collection_exists():
            return False
        return bool(self.client.delete_collection(self.collection_name))

    def _collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.collection_name))
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception as exc:
            if exc.__class__.__name__ in {
                "UnexpectedResponse",
                "NotFoundError",
            } or "404" in str(exc):
                return False
            raise


def build_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = (
        record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    )
    return QdrantPayload(
        chunk_id=str(record.get("chunk_id", "")),
        paper_id=str(record.get("paper_id", "")),
        section=str(record.get("section", "other")),
        text=str(record.get("text", "")),
        metadata=dict(metadata),
    ).to_dict()


def _valid_record(record: Mapping[str, Any]) -> bool:
    embedding = record.get("embedding")
    return bool(
        record.get("chunk_id") and isinstance(embedding, (list, tuple)) and embedding
    )


def _point(record: Mapping[str, Any]) -> Any:
    point = {
        "id": str(record["chunk_id"]),
        "vector": list(record["embedding"]),
        "payload": build_payload(record),
    }
    return models.PointStruct(**point) if models is not None else point


def _vector_params(size: int, distance: str) -> Any:
    names = {
        "cosine": "COSINE",
        "dot": "DOT",
        "euclid": "EUCLID",
        "manhattan": "MANHATTAN",
    }
    if distance not in names:
        raise ValueError(f"Unsupported distance: {distance}")
    if models is None:
        return {"size": size, "distance": distance}
    return models.VectorParams(
        size=size, distance=getattr(models.Distance, names[distance])
    )
