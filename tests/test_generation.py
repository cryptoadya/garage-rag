import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from garage_rag.generation import GenerationError, answer, build_context, make_llm
from garage_rag.index import SearchHit


def hits():
    return [
        SearchHit(
            document=Document(
                page_content="Check tire pressure when the tires are cold. Use 2.4 bar.",
                metadata={"document": "aurora.pdf", "pdf_page": 1, "title": "Aurora A1"},
            ),
            score=0.9,
        )
    ]


class Reply:
    def __init__(self, value):
        self.value = value
        self.messages = None

    def invoke(self, messages, **kwargs):
        self.messages = messages
        return SimpleNamespace(content=json.dumps(self.value, ensure_ascii=False))


def valid_reply():
    return {
        "supported": True,
        "claims": [
            {
                "text": "Проверяйте давление на холодных шинах.",
                "evidence": [
                    {"source_id": "S1", "quote": "Check tire pressure when the tires are cold."}
                ],
            }
        ],
    }


def test_context_budget_and_stable_source_map():
    context = build_context(hits(), max_chars=2000)
    assert context.sources["S1"].metadata["pdf_page"] == 1
    assert "2.4 bar" in context.text
    assert build_context(hits(), max_chars=10).sources == {}


def test_russian_answer_contains_locally_verified_page_and_quote():
    llm = Reply(valid_reply())
    result = answer("Как проверить шины?", hits(), llm)
    assert result.supported
    assert "Проверяйте" in result.text
    assert "aurora.pdf, PDF стр. 1" in result.text
    assert "Check tire pressure when the tires are cold." in result.text
    assert llm.messages[0].type == "system"
    assert len(llm.messages) == 2


@pytest.mark.parametrize("change", ["wrong_page", "fake_quote", "english", "uncited"])
def test_unverifiable_output_fails_closed(change):
    value = valid_reply()
    claim = value["claims"][0]
    if change == "wrong_page":
        claim["evidence"][0]["source_id"] = "S99"
    elif change == "fake_quote":
        claim["evidence"][0]["quote"] = "Cylinder head torque is 100 Nm."
    elif change == "english":
        claim["text"] = "Check the tires."
    else:
        claim["evidence"] = []
    result = answer("Как проверить шины?", hits(), Reply(value))
    assert not result.supported
    assert "недостаточно" in result.text.lower()


def test_no_context_does_not_call_model():
    assert not answer("torque?", [], None).supported


def test_explicit_abstention():
    assert not answer("torque?", hits(), Reply({"supported": False, "claims": []})).supported


def test_malformed_json_is_not_displayed():
    class BadReply:
        def invoke(self, *args, **kwargs):
            return SimpleNamespace(content="unsupported answer with secret")

    assert not answer("torque?", hits(), BadReply()).supported


def test_api_error_is_sanitized():
    class Broken:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("secret-token request body")

    with pytest.raises(GenerationError) as exc:
        answer("tires?", hits(), Broken())
    assert "secret-token" not in str(exc.value)


def test_missing_key_and_paid_model_are_rejected(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="OPENROUTER_API_KEY"):
        make_llm()
    with pytest.raises(GenerationError, match="free"):
        make_llm(model="paid/model", api_key="test-not-a-real-key")


@pytest.mark.parametrize("status", [401, 429, 503])
def test_real_sdk_wire_disables_paid_fallback_and_hidden_retries(monkeypatch, status):
    import httpx
    import openrouter

    requests = []

    def transport(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            status, json={"error": {"code": status, "message": "secret-token provider error"}}
        )

    constructor = openrouter.OpenRouter
    http = httpx.Client(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(
        openrouter, "OpenRouter", lambda **kwargs: constructor(client=http, **kwargs)
    )
    llm = make_llm(api_key="test-not-a-real-key")
    with llm.client, pytest.raises(GenerationError) as exc:
        answer("Как проверить шины?", hits(), llm)
    assert "secret-token" not in str(exc.value)
    if status == 429:
        assert "rate limit" in str(exc.value)
    if status == 401:
        assert "authentication" in str(exc.value)
    assert len(requests) == 1
    body = requests[0]
    assert body["model"] == "openrouter/free"
    assert body["provider"]["max_price"] == {"prompt": "0", "completion": "0", "request": "0"}
    assert body["provider"]["allow_fallbacks"] is False
    assert "models" not in body
    assert "messages" in body
