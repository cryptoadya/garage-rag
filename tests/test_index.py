import pytest

from garage_rag.index import Index


def test_persistent_search_and_unchanged_skip(tmp_path, pdf_path, tokenizer, embeddings, config):
    store = tmp_path / "index"
    with Index(store, config) as index:
        result = index.ingest(pdf_path, embeddings, tokenizer, batch_size=1)
        assert result["status"] == "indexed"
        assert index.stats()["pages"] == 4
        assert index.stats()["dimension"] == 3
    calls = embeddings.document_calls
    with Index(store, config) as index:
        assert index.ingest(pdf_path, embeddings, tokenizer)["status"] == "unchanged"
        hits = index.search("Давление в шинах?", embeddings, top_k=2)
        assert hits[0].document.metadata["pdf_page"] == 1
        assert hits[0].score == pytest.approx(1)
        assert "2.4 bar" in hits[0].document.page_content
    assert embeddings.document_calls == calls


def test_resume_reuses_completed_batches_and_hides_partial_revision(
    tmp_path, pdf_path, tokenizer, config
):
    from conftest import TinyEmbeddings

    broken = TinyEmbeddings(fail_after=1)
    with Index(tmp_path / "index", config) as index:
        with pytest.raises(RuntimeError, match="interruption"):
            index.ingest(pdf_path, broken, tokenizer, batch_size=1)
        assert index.search("tire", broken) == []
        assert index.stats()["chunks"] == 0
    working = TinyEmbeddings()
    with Index(tmp_path / "index", config) as index:
        result = index.ingest(pdf_path, working, tokenizer, batch_size=1)
        assert result["reused_chunks"] == 1
        assert working.document_calls == result["chunks"] - 1
        assert index.search("tire", working)[0].document.metadata["pdf_page"] == 1


def test_incompatible_configuration_is_rejected(tmp_path, config):
    from dataclasses import replace

    with Index(tmp_path / "index", config):
        pass
    with pytest.raises(ValueError, match="incompatible"):
        with Index(tmp_path / "index", replace(config, revision="different")):
            pass
    with Index(tmp_path / "index", config):
        pass  # Rejected open must release Qdrant's lock.


def test_metadata_change_invalidates_generation(tmp_path, pdf_path, tokenizer, embeddings, config):
    with Index(tmp_path / "index", config) as index:
        index.ingest(pdf_path, embeddings, tokenizer, metadata={"language": "en"})
        result = index.ingest(pdf_path, embeddings, tokenizer, metadata={"language": "de"})
        assert result["status"] == "indexed"
        hits = index.search("tire", embeddings)
        assert all(hit.document.metadata["language"] == "de" for hit in hits)


def test_invalid_vector_batch_never_activates_document(
    tmp_path, pdf_path, tokenizer, embeddings, config, monkeypatch
):
    monkeypatch.setattr(embeddings, "embed_documents", lambda texts: [[0.0, 0.0, 0.0]])
    with Index(tmp_path / "index", config) as index:
        with pytest.raises(ValueError, match="vector"):
            index.ingest(pdf_path, embeddings, tokenizer)
        assert index.stats()["chunks"] == 0


def test_resume_initialization_after_collection_creation_error(tmp_path, config, monkeypatch):
    from qdrant_client import QdrantClient

    create = QdrantClient.create_collection

    def fail(*args, **kwargs):
        create(*args, **kwargs)
        raise RuntimeError("simulated collection creation failure")

    monkeypatch.setattr(QdrantClient, "create_collection", fail)
    with pytest.raises(RuntimeError):
        Index(tmp_path / "index", config)
    monkeypatch.setattr(QdrantClient, "create_collection", create)
    with Index(tmp_path / "index", config) as index:
        assert index.stats()["chunks"] == 0


def test_previous_revision_remains_searchable_after_failed_update(
    tmp_path, pdf_path, tokenizer, embeddings, config
):
    from conftest import TinyEmbeddings

    with Index(tmp_path / "index", config) as index:
        index.ingest(pdf_path, embeddings, tokenizer, metadata={"title": "Original"})
        with pytest.raises(RuntimeError):
            index.ingest(
                pdf_path,
                TinyEmbeddings(fail_after=1),
                tokenizer,
                metadata={"title": "Updated"},
                batch_size=1,
            )
        hits = index.search("tire", embeddings)
        assert hits
        assert all(hit.document.metadata["title"] == "Original" for hit in hits)
        assert index.stats()["documents"] == 1


def test_unchanged_document_repairs_a_missing_point(
    tmp_path, pdf_path, tokenizer, embeddings, config
):
    from qdrant_client import models

    with Index(tmp_path / "index", config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        points, _ = index.client.scroll("garage_rag", limit=1)
        index.client.delete("garage_rag", models.PointIdsList(points=[points[0].id]), wait=True)
        result = index.ingest(pdf_path, embeddings, tokenizer)
        assert result["status"] == "indexed"
        assert result["embedded_chunks"] == 1
        assert index.stats()["chunks"] == index.client.count("garage_rag").count
