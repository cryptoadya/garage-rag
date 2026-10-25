"""Local experimental ranking adapters. Dense search delegates to the original Index."""

import math
import re
from collections import Counter

from langchain_core.documents import Document
from qdrant_client import models

from .index import COLLECTION, SearchHit
from .profiles import RERANK_MODEL, RERANK_REVISION


def reciprocal_rank_fusion(dense, lexical, constant=60):
    scores = {}
    for ranking in (dense, lexical):
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0) + 1 / (constant + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


def tokens(text):
    return re.findall(r"\w+", text.casefold())


class LocalReranker:
    def __init__(self, device="cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            RERANK_MODEL,
            revision=RERANK_REVISION,
            device=device,
            trust_remote_code=False,
            max_length=512,
        )

    def predict(self, pairs, **kwargs):
        for question, passage in pairs:
            ids = self.model.tokenizer(question, passage, truncation=False)["input_ids"]
            if len(ids) > 512:
                raise ValueError("Question and passage exceed the reranker 512-token limit")
        return self.model.predict(pairs, **kwargs)


class Retriever:
    """Small-corpus BM25, equal-weight RRF or dense top-20 cross-encoder reranking.

    Lexical state is process-local and rebuilt only from active generations. No
    secondary on-disk index, source rewriting, language dictionary or remote API.
    """

    def __init__(self, index, method="dense", reranker=None):
        if method not in {"dense", "bm25", "hybrid", "rerank"}:
            raise ValueError("Unknown retrieval method")
        self.index = index
        self.method = method
        self.reranker = reranker
        self._active = None

    def __getattr__(self, name):
        return getattr(self.index, name)

    @staticmethod
    def key(hit):
        m = hit.document.metadata
        return (m["content_sha256"], m["document"], m["pdf_page"], m["chunk_index"])

    def _lexical(self, question):
        active = tuple(sorted(r["generation"] for r in self.manifest["documents"].values()))
        if active != self._active:
            points = []
            cursor = None
            while active:
                batch, cursor = self.client.scroll(
                    COLLECTION,
                    limit=256,
                    offset=cursor,
                    with_vectors=False,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="generation", match=models.MatchAny(any=list(active))
                            )
                        ]
                    ),
                )
                points.extend(batch)
                if cursor is None:
                    break
            if len(points) != sum(r["chunks"] for r in self.manifest["documents"].values()):
                raise ValueError("Incomplete active lexical corpus")
            self._hits = sorted(
                [
                    SearchHit(
                        Document(page_content=p.payload["text"], metadata=p.payload["metadata"]), 0
                    )
                    for p in points
                ],
                key=self.key,
            )
            self._terms = [Counter(tokens(h.document.page_content)) for h in self._hits]
            self._lengths = [sum(c.values()) for c in self._terms]
            self._df = Counter(t for c in self._terms for t in c)
            self._avg = sum(self._lengths) / max(1, len(points))
            self._active = active
        scores = []
        query_terms = sorted(set(tokens(question)))
        for hit, terms, length in zip(self._hits, self._terms, self._lengths, strict=True):
            score = 0
            for term in query_terms:
                tf = terms[term]
                if tf:
                    idf = math.log(
                        1 + (len(self._hits) - self._df[term] + 0.5) / (self._df[term] + 0.5)
                    )
                    score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / self._avg))
            if score > 0:
                scores.append(SearchHit(hit.document, score))
        return sorted(scores, key=lambda h: (-h.score, self.key(h)))[:20]

    def search(self, question, embeddings, top_k=5, min_score=None):
        if self.method == "dense":
            return self.index.search(question, embeddings, top_k, min_score)
        if not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain 1–4000 characters")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if min_score is not None:
            raise ValueError("min_score is supported only by dense cosine retrieval")
        if self.method == "bm25":
            return self._lexical(question)[:top_k]
        dense = self.index.search(question, embeddings, top_k=20)
        if self.method == "hybrid":
            lexical = self._lexical(question)
            lookup = {self.key(h): h for h in [*dense, *lexical]}
            rankings = [[self.key(h) for h in hits] for hits in [dense, lexical]]
            keys = reciprocal_rank_fusion(*rankings)
            scores = {
                key: sum(
                    1 / (60 + ranking.index(key) + 1) for ranking in rankings if key in ranking
                )
                for key in keys
            }
            return [SearchHit(lookup[key].document, scores[key]) for key in keys[:top_k]]
        if not dense:
            return []
        if self.reranker is None:
            raise ValueError("Load a pinned local reranker before search")
        values = self.reranker.predict(
            [(question, h.document.page_content) for h in dense],
            batch_size=8,
            show_progress_bar=False,
        )
        if len(values) != len(dense) or not all(math.isfinite(float(s)) for s in values):
            raise ValueError("Invalid reranker scores")
        hits = [SearchHit(h.document, float(s)) for h, s in zip(dense, values, strict=True)]
        return sorted(hits, key=lambda h: (-h.score, self.key(h)))[:top_k]
