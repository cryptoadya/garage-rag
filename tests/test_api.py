import json
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from garage_rag.index import Index


@pytest.fixture
def api_setup(tmp_path, pdf_path, tokenizer, embeddings, config, monkeypatch):
    from garage_rag import api, cli

    directory = tmp_path / "index"
    with Index(directory, config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
    models, adapters = [], []

    def make_llm(**kwargs):
        def invoke(messages):
            model.calls += 1
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "decision": "accept",
                        "complete": True,
                        "checks": [
                            {
                                "claim_index": 0,
                                "entailed": True,
                                "applicable": True,
                                "numbers_safe": True,
                            }
                        ],
                    }
                    if "evidence auditor" in messages[0].content
                    else {
                        "supported": True,
                        "claims": [
                            {
                                "text": model.claim,
                                "evidence": [
                                    {
                                        "source_id": "S1",
                                        "quote": "Check tire pressure when the tires are cold.",
                                    }
                                ],
                            }
                        ],
                    }
                ),
                response_metadata={"finish_reason": "stop"},
            )

        def close(*args):
            model.closed = True

        model = SimpleNamespace(
            claim=models[-1].claim if models else "Check the tires when they are cold.",
            calls=0,
            closed=False,
            invoke=invoke,
            client=SimpleNamespace(__exit__=close),
        )
        models.append(model)
        return model

    def adapter(*args, **kwargs):
        adapters.append(embeddings)
        return embeddings

    embeddings.tokenizer = tokenizer
    for module in (api, cli):
        monkeypatch.setattr(module, "make_llm", make_llm)
        monkeypatch.setattr(module, "E5Embeddings", adapter)
        monkeypatch.setattr(module, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "IndexConfig", lambda **kwargs: config)
    return SimpleNamespace(
        app=api.create_app(index_dir=directory, config=config, pdf_dir=tmp_path / "pdfs"),
        directory=directory,
        models=models,
        adapters=adapters,
        config=config,
        pdf_dir=tmp_path / "pdfs",
    )


def without_timings(result):
    result["workflow"]["elapsed_ms"] = 0
    for event in result["workflow"]["events"]:
        event["elapsed_ms"] = 0
    return result


@pytest.mark.parametrize("mode", ["direct", "grounded"])
@pytest.mark.parametrize(
    ("question", "language", "claim"),
    [
        ("How do I check tires?", "auto", "Check the tires when they are cold."),
        ("Wie prüfe ich die Reifen?", "auto", "Prüfen Sie die Reifen im kalten Zustand."),
        ("Как проверить шины?", "auto", "Проверяйте давление на холодных шинах."),
        ("How do I check tires?", "de", "Prüfen Sie die Reifen im kalten Zustand."),
    ],
)
def test_http_matches_cli(api_setup, capsys, mode, question, language, claim):
    from garage_rag.cli import main

    setup = api_setup
    with TestClient(setup.app) as client:
        setup.models[-1].claim = claim
        response = client.post(
            "/ask",
            json={
                "question": question,
                "mode": mode,
                "language": language,
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["supported"] is True
        assert set(result) == {"supported", "text", "reason", "language", "workflow", "claims"}
        assert result.pop("claims") == [
            {
                "text": claim,
                "citations": [
                    {
                        "source_id": "S1",
                        "document": "aurora.pdf",
                        "pdf_page": 1,
                        "quote": "Check tire pressure when the tires are cold.",
                    }
                ],
            }
        ]
        assert "[" in result["text"]  # Verified PDF references survive HTTP serialization.
        assert "candidate" not in response.text
        assert "context" not in result
    # CLI and HTTP use the same retrieval, generation, validation and public metadata.
    assert (
        main(
            [
                "--index-dir",
                str(setup.directory),
                "ask",
                question,
                "--workflow",
                mode,
                "--answer-language",
                language,
                "--json",
            ]
        )
        == 0
    )
    cli_result = json.loads(capsys.readouterr().out)
    assert without_timings(result) == without_timings(cli_result)


def test_health_reuse_and_shutdown_release_index(api_setup):
    setup = api_setup
    with TestClient(setup.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert setup.models[0].calls == 0
        for _ in range(2):
            assert client.post("/ask", json={"question": "tires?"}).status_code == 200
        assert len(setup.models) == len(setup.adapters) == 1
        with pytest.raises(RuntimeError):
            Index(setup.directory, setup.config)
    assert setup.models[0].closed
    with Index(setup.directory, setup.config):
        pass


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 4001},
        {"question": 123},
        {"question": "tires?", "mode": "agentic"},
        {"question": "tires?", "language": "fr"},
        {"question": "tires?", "provider": "secret-key"},
    ],
)
def test_invalid_request_makes_no_llm_call(api_setup, body):
    with TestClient(api_setup.app) as client:
        response = client.post("/ask", json=body)
        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid request."}
        assert api_setup.models[0].calls == 0


def test_malformed_json_is_sanitized(api_setup):
    with TestClient(api_setup.app) as client:
        response = client.post(
            "/ask",
            content='{"question":"secret-key"',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert "secret-key" not in response.text


@pytest.mark.parametrize("mode", ["direct", "grounded"])
def test_provider_failure_is_safe_and_next_request_succeeds(api_setup, mode):
    with TestClient(api_setup.app) as client:
        model = api_setup.models[0]
        invoke = model.invoke

        def fail(messages):
            raise RuntimeError("secret-key private prompt /private/source.pdf")

        model.invoke = fail
        response = client.post("/ask", json={"question": "tires?", "mode": mode})
        assert response.status_code == 502
        assert response.json() == {"detail": "Answer generation failed."}
        model.invoke = invoke
        assert client.post("/ask", json={"question": "tires?"}).status_code == 200


@pytest.mark.parametrize("mode", ["direct", "grounded"])
def test_no_evidence_is_a_normal_answer(api_setup, monkeypatch, mode):
    monkeypatch.setattr(Index, "search", lambda *a, **kw: [])
    with TestClient(api_setup.app) as client:
        response = client.post("/ask", json={"question": "tires?", "mode": mode})
        assert response.status_code == 200
        assert response.json()["supported"] is False
        assert response.json()["reason"] == "no_context"
        assert response.json()["claims"] == []
        assert api_setup.models[0].calls == 0


def test_citations_preserve_claim_order_multiple_sources_and_exact_quotes(api_setup, monkeypatch):
    from langchain_core.documents import Document

    from garage_rag.index import SearchHit

    quote = 'Check tire pressure\nwhen the tires are cold. "Before driving."'
    other_quote = "Check the tires before driving."
    documents = [
        Document(page_content=quote, metadata={
            "document": "manual [EN], edition.pdf", "pdf_page": 7, "page_label": "3",
            "source_url": "file:///private/manual.pdf", "private_field": "hidden",
        }),
        Document(page_content=other_quote, metadata={"document": "other.pdf", "pdf_page": 2}),
    ]
    monkeypatch.setattr(Index, "search", lambda *a, **kw: [SearchHit(d, 1) for d in documents])
    with TestClient(api_setup.app) as client:
        api_setup.models[0].invoke = lambda messages: SimpleNamespace(
            content=json.dumps({"supported": True, "claims": [
                {"text": "Check the tires when they are cold.", "evidence": [
                    {"source_id": "S1", "quote": quote},
                    {"source_id": "S2", "quote": other_quote},
                ]},
                {"text": "Check the tires before driving.", "evidence": [
                    {"source_id": "S1", "quote": quote},
                ]},
            ]}),
        )
        response = client.post("/ask", json={"question": "tires?", "mode": "direct"})
    assert response.status_code == 200
    result = response.json()
    citation = {
        "source_id": "S1", "document": "manual [EN], edition.pdf", "pdf_page": 7,
        "quote": quote,
    }
    assert result["claims"] == [
        {"text": "Check the tires when they are cold.", "citations": [citation, {
            "source_id": "S2", "document": "other.pdf", "pdf_page": 2, "quote": other_quote,
        }]},
        {"text": "Check the tires before driving.", "citations": [citation]},
    ]
    assert "private" not in response.text
    assert "page_label" not in response.text


@pytest.mark.parametrize("rejection", ["audit", "invalid_quote"])
def test_refusals_do_not_expose_rejected_draft_citations(api_setup, rejection):
    with TestClient(api_setup.app) as client:
        model = api_setup.models[0]
        invoke = model.invoke

        def reject(messages):
            response = invoke(messages)
            value = json.loads(response.content)
            if rejection == "audit" and "decision" in value:
                value["decision"] = "unsupported"
            elif rejection == "invalid_quote" and "claims" in value:
                value["claims"][0]["evidence"][0]["quote"] = "Rejected private draft quote."
            response.content = json.dumps(value)
            return response

        model.invoke = reject
        response = client.post("/ask", json={"question": "tires?"})
    assert response.status_code == 200
    assert response.json()["supported"] is False
    assert response.json()["claims"] == []
    assert "Check the tires" not in response.text
    assert "Rejected private draft" not in response.text


def test_local_failure_is_safe(api_setup, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("/private/source.pdf secret-key")

    monkeypatch.setattr(Index, "search", fail)
    with TestClient(api_setup.app) as client:
        response = client.post("/ask", json={"question": "tires?"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Local retrieval failed."}


def test_busy_request_does_not_overlap_and_health_remains_available(api_setup):
    entered, release = Event(), Event()
    with TestClient(api_setup.app) as client:
        model = api_setup.models[0]
        invoke = model.invoke

        def slow(messages):
            entered.set()
            assert release.wait(10)
            return invoke(messages)

        model.invoke = slow
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(client.post, "/ask", json={"question": "tires?"})
            try:
                assert entered.wait(10)
                busy = client.post("/ask", json={"question": "tires?"})
                assert busy.status_code == 503
                assert busy.headers["retry-after"] == "1"
                assert client.get("/health").status_code == 200
            finally:
                release.set()
            assert pending.result(timeout=10).status_code == 200


def test_empty_index_can_start_and_receive_first_pdf(api_setup, tmp_path, pdf_path):
    from garage_rag import api

    directory = tmp_path / "missing"
    app = api.create_app(index_dir=directory, config=api_setup.config, pdf_dir=api_setup.pdf_dir)
    with TestClient(app) as client:
        assert client.get("/documents").json() == {"documents": []}
        assert client.post("/ask", json={"question": "tires?"}).json()["reason"] == "no_context"
        response = client.post("/documents?filename=aurora.pdf", content=pdf_path.read_bytes(),
                               headers={"content-type": "application/pdf"})
        assert response.status_code == 200
        assert response.json()["status"] == "indexed"
        assert client.get("/documents").json()["documents"][0]["document"] == "aurora.pdf"
        result = client.post("/ask", json={"question": "tires?"}).json()
        assert result["supported"] is True
        assert result["claims"][0]["citations"][0]["document"] == "aurora.pdf"


@pytest.mark.parametrize("failure", ["credentials", "locked", "embeddings"])
def test_startup_failure_is_sanitized_and_cleans_up(api_setup, monkeypatch, failure):
    from garage_rag import api

    def fail(*args, **kwargs):
        raise RuntimeError("secret-key /private/source.pdf")

    setup = api_setup
    if failure == "credentials":
        monkeypatch.setattr(api, "make_llm", fail)
    elif failure == "embeddings":
        monkeypatch.setattr(api, "E5Embeddings", fail)
    with Index(setup.directory, setup.config) if failure == "locked" else nullcontext():
        with pytest.raises(RuntimeError) as caught:
            with TestClient(setup.app):
                pass
    assert "RAG startup failed" in str(caught.value)
    assert "secret-key" not in str(caught.value)
    assert all(model.closed for model in setup.models)
    with Index(setup.directory, setup.config):
        pass


def upload(client, content, filename="new manual.pdf", content_type="application/pdf"):
    return client.post("/documents", params={"filename": filename}, content=content,
                       headers={"content-type": content_type})


def test_upload_list_reuse_storage_and_ask(api_setup, pdf_path):
    setup = api_setup
    content = pdf_path.read_bytes()
    with TestClient(setup.app) as client:
        before = client.get("/documents")
        assert before.status_code == 200
        assert set(before.json()["documents"][0]) == {"document", "pages", "chunks"}
        response = upload(client, content)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "indexed"
        assert body["document"]["document"] == "new manual.pdf"
        assert body["document"]["pages"] == len(PdfReader(pdf_path).pages)
        assert body["document"]["chunks"] > 0
        assert len(client.get("/documents").json()["documents"]) == 2
        calls = setup.adapters[0].document_calls
        assert upload(client, content).json()["status"] == "unchanged"
        assert setup.adapters[0].document_calls == calls
        for mode in ("direct", "grounded"):
            result = client.post("/ask", json={"question": "tires?", "mode": mode}).json()
            assert result["supported"] is True
            assert result["claims"][0]["citations"][0]["quote"]
        assert "/" not in str(body)
        assert "source_url" not in before.text
    stored = list(setup.pdf_dir.rglob("*.pdf"))
    assert stored and all(path.read_bytes() == content for path in stored)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in stored)
    assert all(stat.S_IMODE(path.parent.stat().st_mode) == 0o700 for path in stored)
    with TestClient(setup.app) as client:
        assert len(client.get("/documents").json()["documents"]) == 2
        assert client.get("/new manual.pdf").status_code == 404


@pytest.mark.parametrize("filename", ["../escape.pdf", "..\\escape.pdf", "/tmp/a.pdf",
                                      "bad\nname.pdf", "x.txt", "😀" * 100 + ".pdf"])
def test_unsafe_filename_is_rejected_without_storage(api_setup, pdf_path, filename):
    with TestClient(api_setup.app) as client:
        assert upload(client, pdf_path.read_bytes(), filename).status_code == 400
    assert not list(api_setup.pdf_dir.rglob("*.pdf"))


@pytest.mark.parametrize("kind", ["empty", "not_pdf", "broken", "encrypted", "scan"])
def test_invalid_pdfs_leave_no_active_document_or_file(api_setup, pdf_path, kind):
    content = {"empty": b"", "not_pdf": b"not a PDF", "broken": b"%PDF-1.7\nbroken"}.get(kind)
    if kind in {"encrypted", "scan"}:
        from io import BytesIO

        writer = PdfWriter()
        writer.add_blank_page(300, 300)
        if kind == "encrypted":
            writer.encrypt("private-password")
        out = BytesIO()
        writer.write(out)
        content = out.getvalue()
    with TestClient(api_setup.app) as client:
        response = upload(client, content)
        assert response.status_code == 422
        assert "private-password" not in response.text
        assert len(client.get("/documents").json()["documents"]) == 1
        assert client.post("/ask", json={"question": "tires?"}).json()["supported"]
    assert not list(api_setup.pdf_dir.rglob("*.pdf"))


def test_upload_media_type_and_stream_size_limit(api_setup, monkeypatch, pdf_path):
    from garage_rag import api

    with TestClient(api_setup.app) as client:
        assert upload(client, pdf_path.read_bytes(), content_type="text/plain").status_code == 415
        monkeypatch.setattr(api, "MAX_PDF_BYTES", 16)
        assert upload(client, pdf_path.read_bytes()).status_code == 413
        response = upload(client, iter([b"%PDF-1.7\n", b"x" * 20]))
        assert response.status_code == 413  # Also bounded without Content-Length.
    assert not list(api_setup.pdf_dir.rglob("*.pdf"))


def test_indexing_failure_is_sanitized_and_preserves_previous_revision(api_setup, monkeypatch,
                                                                     pdf_path):
    def fail(*args, **kwargs):
        raise RuntimeError("secret-key /private/source.pdf")

    with TestClient(api_setup.app) as client:
        before = client.get("/documents").json()
        with monkeypatch.context() as patch:
            patch.setattr(Index, "ingest", fail)
            response = upload(client, pdf_path.read_bytes(), "aurora.pdf")
            assert response.status_code == 500
            assert "secret-key" not in response.text and "/private" not in response.text
        assert client.get("/documents").json() == before
        assert client.post("/ask", json={"question": "tires?"}).json()["supported"]
    assert not list(api_setup.pdf_dir.rglob("*.pdf"))


@pytest.mark.parametrize("operation", ["ask", "index"])
def test_index_operations_are_exclusive_but_health_and_list_stay_responsive(api_setup, pdf_path,
                                                                          monkeypatch, operation):
    entered, release = Event(), Event()
    with TestClient(api_setup.app) as client:
        original = Index.ingest if operation == "index" else api_setup.models[0].invoke

        def slow(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)

        if operation == "index":
            monkeypatch.setattr(Index, "ingest", slow)
        else:
            api_setup.models[0].invoke = slow
        with ThreadPoolExecutor() as pool:
            pending = (pool.submit(upload, client, pdf_path.read_bytes()) if operation == "index"
                       else pool.submit(client.post, "/ask", json={"question": "tires?"}))
            try:
                assert entered.wait(10)
                for response in (client.post("/ask", json={"question": "tires?"}),
                                 upload(client, pdf_path.read_bytes(), "overlap.pdf")):
                    assert response.status_code == 503
                    assert response.headers["retry-after"] == "1"
                assert client.get("/health").status_code == 200
                assert len(client.get("/documents").json()["documents"]) == 1
            finally:
                release.set()
            assert pending.result(timeout=10).status_code == 200
