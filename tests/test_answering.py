import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from garage_rag.index import Index


class Replies:
    def __init__(self, *values):
        self.values = iter(values)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            content=json.dumps(value, ensure_ascii=False),
            usage_metadata={"input_tokens": 10, "output_tokens": 5},
        )


def tire_draft(text):
    return {"supported": True, "claims": [{"text": text, "evidence": [{
        "source_id": "S1", "quote": "Check tire pressure when the tires are cold.",
    }]}]}


@pytest.mark.parametrize("mode", ["direct", "grounded"])
@pytest.mark.parametrize(("question", "claim"), [
    ("Как проверить шины?", "Проверяйте давление на холодных шинах."),
    ("Wie prüfe ich die Reifen?", "Prüfen Sie die Reifen im kalten Zustand."),
])
def test_translated_query_retrieves_english_evidence_but_answers_original_question(
    tmp_path, pdf_path, tokenizer, embeddings, config, mode, question, claim,
):
    from garage_rag.answering import run_answer

    # An English-only retrieval vector reproduces a cross-language recall miss.
    class EnglishQueries:
        def embed_query(self, text):
            return embeddings.embed_query(text) if text.isascii() else [0.0, 1.0, 0.0]

    replies = [
        {"query": "How do I check tire pressure?"}, tire_draft(claim),
        {"decision": "accept", "complete": True, "checks": [{
            "claim_index": 0, "entailed": True, "applicable": True, "numbers_safe": True,
        }]},
    ]
    llm = Replies(*replies)
    with Index(tmp_path / "index", config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        def search(query, k):
            return index.search(query, EnglishQueries(), k)

        assert search(question, 1)[0].document.metadata["pdf_page"] == 2
        result = run_answer(question, search, llm, mode=mode, top_k=1, translate_query=True)

    assert result.answer.supported
    assert claim in result.answer.text
    assert "aurora.pdf, PDF page 1" in result.answer.text
    assert result.context.sources["S1"].metadata["pdf_page"] == 1
    assert json.loads(llm.messages[1][1].content)["question"] == question
    assert result.trace.llm_calls == (2 if mode == "direct" else 3)
    assert result.trace.llm_call_limit == (2 if mode == "direct" else 5)
    assert result.trace.input_tokens == 10 * result.trace.llm_calls
    assert result.trace.public_dict()["token_usage_complete"]
    assert result.trace.events[0].node == "translate_query"
    assert question not in json.dumps(result.trace.public_dict(), ensure_ascii=False)


@pytest.mark.parametrize("translate_query", [False, True])
def test_english_query_skips_translation_even_with_russian_answer(translate_query):
    from garage_rag.answering import run_answer
    from garage_rag.index import SearchHit

    llm = Replies(tire_draft("Проверяйте давление на холодных шинах."))
    seen = []
    def search(query, k):
        seen.append(query)
        return [SearchHit(Document(
            page_content="Check tire pressure when the tires are cold.",
            metadata={"document": "aurora.pdf", "pdf_page": 1},
        ), 0.9)]

    result = run_answer("How do I check tires?", search, llm, mode="direct",
                        answer_language="ru", translate_query=translate_query)
    assert result.answer.supported
    assert seen == ["How do I check tires?"]
    assert result.trace.llm_calls == result.trace.llm_call_limit == 1


@pytest.mark.parametrize("value", [
    {"query": ""}, {"query": "Как открыть капот?"}, {"query": "How to open?", "extra": 1},
    {"query": "How do I check the BMW E46 oil at 50 degrees?"},
    {"query": "How do I check BMW E39 oil at 50 degrees?"},
    {"query": "How do I check BMW E39 DSC oil at 20 degrees?"},
    "not a translation object",
    RuntimeError("secret-token"),
])
def test_bad_translation_stops_before_search_without_retry_or_private_errors(value):
    from garage_rag.answering import run_answer

    llm = Replies(value)
    def search(query, k):
        pytest.fail("Invalid translation must not reach retrieval")

    result = run_answer("Как проверить масло BMW E39 при 20 градусах?", search, llm,
                        mode="direct", translate_query=True)
    assert not result.answer.supported
    assert result.answer.reason == "query_translation_error"
    assert result.trace.llm_calls == 1
    assert result.trace.search_calls == 0
    assert len(llm.messages) == 1
    assert "secret-token" not in json.dumps(result.public_dict())


def test_translation_preserves_numbers_and_identifiers_without_inventing_evidence():
    from garage_rag.answering import run_answer

    query = "How do I check BMW E39 oil at 20 degrees?"
    seen = []
    def search(text, k):
        seen.append(text)
        return []

    result = run_answer("Как проверить масло BMW E39 при 20 градусах?", search,
                        Replies({"query": query}), mode="direct")
    assert seen == [query]
    assert not result.answer.supported
    assert result.answer.reason == "no_context"
    assert result.trace.llm_calls == result.trace.search_calls == 1


def test_disabled_translation_keeps_original_russian_query():
    from garage_rag.answering import run_answer

    seen = []
    question = "Как открыть капот?"
    def search(query, k):
        seen.append(query)
        return []

    result = run_answer(question, search, Replies(), mode="direct", translate_query=False)
    assert seen == [question]
    assert result.trace.llm_calls == 0
    assert result.trace.llm_call_limit == 1


def test_translation_setting_defaults_on_and_rejects_typos(monkeypatch):
    from garage_rag.answering import query_translation_enabled

    monkeypatch.delenv("GARAGE_RAG_TRANSLATE_QUERY", raising=False)
    assert query_translation_enabled()
    monkeypatch.setenv("GARAGE_RAG_TRANSLATE_QUERY", "false")
    assert not query_translation_enabled()
    monkeypatch.setenv("GARAGE_RAG_TRANSLATE_QUERY", "maybe")
    with pytest.raises(ValueError, match="GARAGE_RAG_TRANSLATE_QUERY"):
        query_translation_enabled()


def test_truncated_translation_is_rejected_without_search():
    from garage_rag.answering import run_answer

    llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(
        content='{"query":"How do I open the hood?"}',
        response_metadata={"finish_reason": "length"},
    ))
    result = run_answer("Как открыть капот?", lambda q, k: pytest.fail("search"), llm)
    assert result.answer.reason == "query_translation_error"
    assert result.trace.llm_calls == 1
    assert result.trace.search_calls == 0


@pytest.mark.parametrize(("original", "translated"), [
    ("-20", "20"), ("-20", "+20"), ("20", "-20"), ("−20.5", "20.5"),
])
def test_translation_cannot_drop_or_change_numeric_signs(original, translated):
    from garage_rag.answering import run_answer

    seen = []
    def search(query, k):
        seen.append(query)
        return []

    result = run_answer(f"Как проверить шины при {original} градусах?", search,
                        Replies({"query": f"How do I check tires at {translated} degrees?"}),
                        mode="direct")
    assert result.answer.reason == "query_translation_error"
    assert seen == []
    assert result.trace.llm_calls == 1


@pytest.mark.parametrize(("question", "query"), [
    ("Как проверить шины при −20 градусах?", "How do I check tires at -20 degrees?"),
    ("Как проверить масло bmw e39?", "How do I check BMW E39 oil?"),
    ("Как проверить индикатор ABS?", "How do I check the abs warning light?"),
])
def test_translation_accepts_equivalent_minus_and_identifier_casing(question, query):
    from garage_rag.answering import run_answer

    seen = []
    def search(text, k):
        seen.append(text)
        return []

    result = run_answer(question, search, Replies({"query": query}), mode="direct")
    assert result.answer.reason == "no_context"
    assert seen == [query]
