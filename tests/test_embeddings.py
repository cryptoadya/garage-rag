import numpy as np
import pytest

from garage_rag.embeddings import E5Embeddings


def test_e5_uses_distinct_prefixes_and_normalizes(tokenizer, monkeypatch):
    class Encoder:
        def __init__(self):
            self.inputs = []

        def encode(self, texts, **kwargs):
            self.inputs.extend(texts)
            assert kwargs["normalize_embeddings"]
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

    encoder = Encoder()
    embedding = E5Embeddings()
    monkeypatch.setattr(embedding, "_model", encoder)
    monkeypatch.setattr(embedding, "_tokenizer", tokenizer)
    assert embedding.embed_documents(["cold tires"])[0] == [1.0, 0.0, 0.0]
    assert embedding.embed_query("давление") == [1.0, 0.0, 0.0]
    assert encoder.inputs == ["passage: cold tires", "query: давление"]


def test_e5_rejects_silent_truncation(tokenizer, monkeypatch):
    embedding = E5Embeddings()
    monkeypatch.setattr(embedding, "_tokenizer", tokenizer)
    with pytest.raises(ValueError, match="512"):
        embedding.embed_query("word " * 513)
