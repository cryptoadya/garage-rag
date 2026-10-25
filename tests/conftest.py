import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def corpus():
    return json.loads((Path(__file__).parent / "fixtures/corpus.json").read_text())


@pytest.fixture
def pdf_path(tmp_path, corpus):
    path = tmp_path / "aurora.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.setTitle(corpus["title"])
    for page in corpus["pages"]:
        text = pdf.beginText(40, 780)
        for line in page.splitlines():
            text.textLine(line)
        pdf.drawText(text)
        pdf.showPage()
    pdf.save()
    return path


@pytest.fixture
def tokenizer():
    # Real fast tokenizer with offsets; entirely local, no model download in CI.
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


class TinyEmbeddings:
    """Deterministic test vectors; real Qdrant persists and ranks them."""

    def __init__(self, fail_after=None):
        self.document_calls = 0
        self.fail_after = fail_after

    def embed_documents(self, texts):
        if self.fail_after is not None and self.document_calls >= self.fail_after:
            raise RuntimeError("simulated interruption")
        self.document_calls += 1
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        return self.vector(text)

    @staticmethod
    def vector(text):
        text = text.lower()
        if any(word in text for word in ["tire", "reifen", "шин"]):
            return [1.0, 0.0, 0.0]
        if any(word in text for word in ["coolant", "kühl", "охлажд"]):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.fixture
def embeddings():
    return TinyEmbeddings()


@pytest.fixture
def config():
    from garage_rag.config import IndexConfig

    return IndexConfig(
        model="test-only", revision="test-v1", dimension=3, chunk_tokens=64, overlap_tokens=8
    )
