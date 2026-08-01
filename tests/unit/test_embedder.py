import numpy as np
import pytest

from processing.chunker import Chunk
from processing.embedder import DEFAULT_MODEL_NAME, Embedder


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.asarray(
            [[float(len(text)), float(index)] for index, text in enumerate(texts)]
        )


@pytest.fixture
def embedder():
    return Embedder(model=FakeModel())


def make_chunk(text="Dense retrieval text."):
    return Chunk(
        chunk_id="chunk-1",
        paper_id="paper-1",
        section="method",
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={"title": "Example"},
    )


def test_default_model_is_loaded_once(monkeypatch):
    loaded_names = []

    class ModelFactory(FakeModel):
        def __init__(self, model_name):
            super().__init__()
            loaded_names.append(model_name)

    monkeypatch.setattr("processing.embedder.SentenceTransformer", ModelFactory)

    instance = Embedder()
    instance.encode_texts(["one"])
    instance.encode_texts(["two"])

    assert loaded_names == [DEFAULT_MODEL_NAME]


def test_encode_chunks_empty_input_returns_empty_list(embedder):
    assert embedder.encode_chunks([]) == []


def test_encode_texts_empty_input_is_safe(embedder):
    result = embedder.encode_texts([])

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 0)
    assert embedder.model.calls == []


def test_encode_texts_returns_one_vector_per_non_empty_text(embedder):
    result = embedder.encode_texts(["first", "second"])

    assert result.shape == (2, 2)
    assert embedder.model.calls[0][0] == ["first", "second"]


def test_blank_text_is_not_encoded(embedder):
    result = embedder.encode_texts(["first", " ", "second"])

    assert result.shape[0] == 2
    assert embedder.model.calls[0][0] == ["first", "second"]


def test_encode_chunks_preserves_fields_and_returns_numeric_list(embedder):
    chunk = make_chunk()

    results = embedder.encode_chunks([chunk])

    assert len(results) == 1
    assert results[0]["chunk_id"] == chunk.chunk_id
    assert results[0]["paper_id"] == chunk.paper_id
    assert results[0]["section"] == chunk.section
    assert results[0]["text"] == chunk.text
    assert results[0]["metadata"] == chunk.metadata
    assert isinstance(results[0]["embedding"], list)
    assert results[0]["embedding"]
    assert all(isinstance(value, float) for value in results[0]["embedding"])


def test_encode_chunks_skips_empty_chunk_text(embedder):
    assert embedder.encode_chunks([make_chunk("  ")]) == []
    assert embedder.model.calls == []
