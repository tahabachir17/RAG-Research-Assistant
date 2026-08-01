from processing.chunker import Chunk
from processing.metadata_tagger import MetadataTagger
from processing.qdrant_indexer import QdrantIndexer, build_payload


class FakeClient:
    def __init__(self):
        self.exists = False
        self.created = []
        self.upserts = []

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.exists = True

    def delete_collection(self, name):
        self.exists = False
        return True

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def search(self, **kwargs):
        return []


def test_metadata_tagger_normalizes_and_preserves_metadata():
    chunk = Chunk(
        "id",
        "paper",
        "Proposed Method",
        "text",
        0,
        4,
        {"year": "2023-01", "authors": "Ada"},
    )
    tagged = MetadataTagger().tag(chunk, {"title": "Paper", "categories": ["cs.AI"]})
    assert tagged.metadata["year"] == 2023
    assert tagged.metadata["authors"] == ["Ada"]
    assert tagged.metadata["categories"] == ["cs.AI"]
    assert tagged.metadata["section_family"] == "method"


def test_qdrant_indexer_creates_batches_and_payload():
    client = FakeClient()
    indexer = QdrantIndexer(client=client, batch_size=1)
    records = [
        {
            "chunk_id": "00000000-0000-0000-0000-000000000001",
            "paper_id": "p",
            "section": "abstract",
            "text": "a",
            "metadata": {"year": 2020},
            "embedding": [0.1, 0.2],
        }
    ]
    assert indexer.index_embeddings(records) == 1
    assert len(client.created) == len(client.upserts) == 1
    assert build_payload(records[0])["metadata"]["year"] == 2020


def test_qdrant_rejects_mixed_dimensions_and_empty_input():
    client = FakeClient()
    indexer = QdrantIndexer(client=client)
    assert indexer.index_embeddings([]) == 0
    import pytest

    with pytest.raises(ValueError):
        indexer.index_embeddings(
            [
                {"chunk_id": "a", "embedding": [1.0]},
                {"chunk_id": "b", "embedding": [1.0, 2.0]},
            ]
        )
