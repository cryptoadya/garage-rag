"""Lazy local E5 adapter: stats and unchanged indexing never load the model."""

from langchain_core.embeddings import Embeddings

from .config import IndexConfig


class E5Embeddings(Embeddings):
    def __init__(self, config=None, device="cpu", batch_size=16):
        self.config = config or IndexConfig()
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                revision=self.config.revision,
                use_fast=True,
                trust_remote_code=False,
            )
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.model,
                revision=self.config.revision,
                device=self.device,
                trust_remote_code=False,
            )
            if self._model.get_embedding_dimension() != self.config.dimension:
                raise ValueError("Embedding model dimension does not match index configuration")
            if self._model.max_seq_length < self.config.max_sequence_tokens:
                raise ValueError("Embedding model sequence limit does not match configuration")
        return self._model

    def _encode(self, texts, prefix):
        inputs = [prefix + text for text in texts]
        for text in inputs:
            if len(self.tokenizer.encode(text, add_special_tokens=True)) > 512:
                raise ValueError("Input exceeds the E5 limit of 512 tokens; shorten the question")
        if not inputs:
            return []
        return self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

    def embed_documents(self, texts):
        return self._encode(texts, self.config.passage_prefix)

    def embed_query(self, text):
        return self._encode([text], self.config.query_prefix)[0]
