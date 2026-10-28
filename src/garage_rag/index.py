"""Single-process Qdrant Local storage with atomic document revision activation."""

import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from .config import IndexConfig, digest
from .ingestion import PDFValidationError, chunk_pages, extract_pdf

COLLECTION = "garage_rag"


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float


class Index:
    def __init__(
        self, path: Path, config: IndexConfig | None = None, *, allow_cross_thread: bool = False
    ):
        self.path = Path(path)
        self.config = config or IndexConfig()
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        # The HTTP adapter serializes all vector/embedding work with its own lock.
        # SQLite on macOS otherwise rejects writes from FastAPI's worker thread.
        self.client = QdrantClient(
            path=str(self.path / "vectors"), force_disable_check_same_thread=allow_cross_thread
        )
        try:
            if self.manifest_path.exists():
                self.manifest = json.loads(self.manifest_path.read_text())
                if self.manifest.get("fingerprint") != self.config.fingerprint:
                    raise ValueError("Index configuration is incompatible; use a new --index-dir")
            else:
                self.manifest = {
                    "fingerprint": self.config.fingerprint,
                    "config": self.config.as_dict(),
                    "documents": {},
                }
                if self.client.collection_exists(COLLECTION):
                    raise ValueError("Index manifest missing; use a new --index-dir")
                self._save_manifest(self.manifest)
            if not self.client.collection_exists(COLLECTION):
                if self.manifest["documents"]:
                    raise ValueError("Index is incomplete: vector collection is missing")
                self.client.create_collection(
                    COLLECTION,
                    vectors_config=models.VectorParams(
                        size=self.config.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
            vector_config = self.client.get_collection(COLLECTION).config.params.vectors
            if vector_config.size != self.config.dimension:
                raise ValueError("Index vector dimension is incompatible")
        except BaseException:
            self.client.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def _save_manifest(self, value):
        # Temporary file + fsync + replace: interruption leaves old or new JSON, never half JSON.
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path, delete=False) as out:
                name = out.name
                json.dump(value, out, ensure_ascii=False, indent=2)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, self.manifest_path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    def ingest(self, path, embeddings, tokenizer=None, *, metadata=None, batch_size=16):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        started = time.perf_counter()
        path = Path(path)
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = metadata or {}
        generation = digest(
            {
                "file": path.name,
                "content": content_hash,
                "metadata": metadata,
                "config": self.config.fingerprint,
            }
        )
        previous = self.manifest["documents"].get(path.name)
        stored_count = (
            self.client.count(
                COLLECTION,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="generation", match=models.MatchValue(value=generation)
                        ),
                    ]
                ),
            ).count
            if previous
            else 0
        )
        if previous and previous["generation"] == generation and stored_count == previous["chunks"]:
            return {
                "status": "unchanged",
                "document": path.name,
                "chunks": previous["chunks"],
                "embedded_chunks": 0,
                "reused_chunks": previous["chunks"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        pages, diagnostics = extract_pdf(path, metadata)
        if pages[0].metadata["content_sha256"] != content_hash:
            raise ValueError("Source PDF changed during ingestion; run index again")
        if tokenizer is None:
            tokenizer = embeddings.tokenizer
        chunks, chunk_diagnostics = chunk_pages(
            pages,
            tokenizer,
            self.config.chunk_tokens,
            self.config.overlap_tokens,
        )
        diagnostics.update(chunk_diagnostics)
        if not chunks:
            raise PDFValidationError("PDF produced no text chunks; OCR may be needed")
        extraction_seconds = time.perf_counter() - started
        embedded = reused = 0
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            ids = [
                str(uuid.uuid5(uuid.NAMESPACE_URL, f"{generation}:{offset + i}"))
                for i in range(len(batch))
            ]
            existing = {
                point.id
                for point in self.client.retrieve(
                    COLLECTION,
                    ids,
                    with_payload=False,
                    with_vectors=False,
                )
            }
            pending = [
                (point_id, chunk)
                for point_id, chunk in zip(ids, batch, strict=True)
                if point_id not in existing
            ]
            reused += len(existing)
            if not pending:
                continue
            vectors = embeddings.embed_documents([chunk.page_content for _, chunk in pending])
            self._validate_vectors(vectors, len(pending))
            self.client.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "generation": generation,
                            "text": chunk.page_content,
                            "metadata": chunk.metadata,
                        },
                    )
                    for (point_id, chunk), vector in zip(pending, vectors, strict=True)
                ],
                wait=True,
            )
            embedded += len(pending)
        record = {
            "generation": generation,
            "content_sha256": content_hash,
            "pages": len(pages),
            "chunks": len(chunks),
            "metadata": pages[0].metadata,
            "diagnostics": diagnostics,
            "embedded_chunks": embedded,
            "reused_chunks": reused,
            "extraction_chunking_seconds": round(extraction_seconds, 3),
            "embedding_storage_seconds": round(
                time.perf_counter() - started - extraction_seconds, 3
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        updated = {**self.manifest, "documents": {**self.manifest["documents"], path.name: record}}
        self._save_manifest(updated)
        self.manifest = updated
        return {"status": "indexed", "document": path.name, **record}

    def _validate_vectors(self, vectors, count):
        if len(vectors) != count:
            raise ValueError("Embedding vector count does not match chunk count")
        for vector in vectors:
            if (
                len(vector) != self.config.dimension
                or not all(math.isfinite(value) for value in vector)
                or sum(value * value for value in vector) == 0
            ):
                raise ValueError("Invalid embedding vector dimension, values or norm")

    def search(self, question, embeddings, top_k=5, min_score=None):
        if not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain 1–4000 characters")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if min_score is not None and not -1 <= min_score <= 1:
            raise ValueError("min_score must be between -1 and 1")
        active = [record["generation"] for record in self.manifest["documents"].values()]
        if not active:
            return []
        vector = embeddings.embed_query(question)
        self._validate_vectors([vector], 1)
        result = self.client.query_points(
            COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="generation",
                        match=models.MatchAny(any=active),
                    )
                ]
            ),
            score_threshold=min_score,
            with_payload=True,
        )
        return [
            SearchHit(
                document=Document(
                    page_content=point.payload["text"], metadata=point.payload["metadata"]
                ),
                score=point.score,
            )
            for point in result.points
        ]

    def stats(self):
        documents = self.manifest["documents"]
        return {
            "documents": len(documents),
            "pages": sum(d["pages"] for d in documents.values()),
            "chunks": sum(d["chunks"] for d in documents.values()),
            **self.manifest["config"],
            "document_details": documents,
        }
