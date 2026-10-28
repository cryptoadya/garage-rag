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
    assert "aurora.pdf, PDF page 1" in result.text
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


def test_default_provider_is_groq_and_requires_free_plan(monkeypatch):
    monkeypatch.delenv("GARAGE_RAG_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_FREE_PLAN_CONFIRMED", raising=False)
    with pytest.raises(GenerationError, match="Free Plan"):
        make_llm(api_key="test-only")
    monkeypatch.setenv("GROQ_FREE_PLAN_CONFIRMED", "true")
    with make_llm(api_key="test-only").client as client:
        assert str(client.base_url) == "https://api.groq.com/openai/v1/"


@pytest.mark.parametrize("provider", ["removed-provider", "paid-provider"])
def test_other_providers_fail_closed(monkeypatch, provider):
    monkeypatch.setenv("GARAGE_RAG_PROVIDER", provider)
    with pytest.raises(GenerationError, match="Groq"):
        make_llm(api_key="test-only")


@pytest.mark.parametrize("content", ["", "truncated JSON", json.dumps(valid_reply())])
def test_token_limit_never_accepts_partial_output_or_retries(content):
    calls = []

    class Truncated:
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content=content, response_metadata={"finish_reason": "length"})

    result = answer("Как проверить шины?", hits(), Truncated())
    assert not result.supported
    assert result.reason == "output_token_limit"
    assert len(calls) == 1


@pytest.mark.parametrize("configured,expected", [(None, 2048), ("512", 512), ("4096", 4096)])
def test_groq_completion_budget_preserves_reasoning_allowance(monkeypatch, configured, expected):
    monkeypatch.setenv("GROQ_FREE_PLAN_CONFIRMED", "true")
    monkeypatch.delenv("GROQ_MAX_TOKENS", raising=False)
    if configured is not None:
        monkeypatch.setenv("GROQ_MAX_TOKENS", configured)
    llm = make_llm(api_key="test-only")
    with llm.client:
        assert llm.max_tokens == expected


@pytest.mark.parametrize("configured", ["secret-token", "0", "511", "4097", "1.5"])
def test_groq_invalid_completion_budget_is_rejected_safely(monkeypatch, configured):
    monkeypatch.setenv("GROQ_FREE_PLAN_CONFIRMED", "true")
    monkeypatch.setenv("GROQ_MAX_TOKENS", configured)
    with pytest.raises(GenerationError, match="GROQ_MAX_TOKENS") as caught:
        make_llm(api_key="test-only")
    assert "secret-token" not in str(caught.value)
