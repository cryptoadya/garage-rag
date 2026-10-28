import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

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

    for module in (api, cli):
        monkeypatch.setattr(module, "make_llm", make_llm)
        monkeypatch.setattr(module, "E5Embeddings", adapter)
        monkeypatch.setattr(module, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "IndexConfig", lambda **kwargs: config)
    return SimpleNamespace(
        app=api.create_app(index_dir=directory, config=config),
        directory=directory,
        models=models,
        adapters=adapters,
        config=config,
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


def test_missing_index_does_not_create_storage(tmp_path, monkeypatch):
    from garage_rag import api

    monkeypatch.setattr(api, "load_dotenv", lambda *a, **kw: None)
    directory = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="RAG startup failed"):
        with TestClient(api.create_app(index_dir=directory)):
            pass
    assert not directory.exists()


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
