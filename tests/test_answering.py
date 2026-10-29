import json
from types import SimpleNamespace

import pytest

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
@pytest.mark.parametrize("question", [
    "Как проверить шины BMW E39 при −20 градусах?",
    "Wie prüfe ich die Reifen beim BMW E39 bei -20 Grad?",
    "How do I check BMW E39 tires at -20 degrees?",
])
@pytest.mark.parametrize("translate_query", [False, True])
def test_original_query_retrieves_from_reopened_legacy_index(
    tmp_path, pdf_path, tokenizer, embeddings, config, mode, question, translate_query,
):
    from garage_rag.answering import run_answer

    directory = tmp_path / "index"
    with Index(directory, config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
    manifest = (directory / "manifest.json").read_bytes()
    llm = Replies(tire_draft("Проверяйте давление на холодных шинах."), {
        "decision": "accept", "complete": True, "checks": [{
            "claim_index": 0, "entailed": True, "applicable": True, "numbers_safe": True,
        }],
    })
    queries = []
    with Index(directory, config) as index:
        def search(query, k):
            queries.append(query)
            return index.search(query, embeddings, k)

        result = run_answer(question, search, llm, mode=mode, top_k=1,
                            answer_language="ru", translate_query=translate_query)

    assert result.answer.supported
    assert queries == [question]
    assert "aurora.pdf, PDF page 1" in result.answer.text
    assert json.loads(llm.messages[0][1].content)["question"] == question
    assert result.trace.llm_calls == (1 if mode == "direct" else 2)
    assert result.trace.llm_call_limit == (1 if mode == "direct" else 4)
    assert result.trace.input_tokens == 10 * result.trace.llm_calls
    assert result.trace.public_dict()["token_usage_complete"]
    assert result.trace.search_calls == 1
    assert (directory / "manifest.json").read_bytes() == manifest
    assert question not in json.dumps(result.trace.public_dict(), ensure_ascii=False)


@pytest.mark.parametrize("question", [
    "Как открыть капот?", "Wie öffne ich die Motorhaube?", "How do I open the hood?",
])
def test_empty_retrieval_needs_no_provider_call(question):
    from garage_rag.answering import run_answer

    seen = []
    def search(query, k):
        seen.append(query)
        return []

    result = run_answer(question, search, Replies(), mode="direct")
    assert seen == [question]
    assert result.answer.reason == "no_context"
    assert result.trace.llm_calls == 0


@pytest.mark.parametrize("setting", [None, "true", "false"])
def test_legacy_translation_setting_never_enables_english_rewriting(monkeypatch, setting):
    from garage_rag.answering import query_translation_enabled

    if setting is None:
        monkeypatch.delenv("GARAGE_RAG_TRANSLATE_QUERY", raising=False)
    else:
        monkeypatch.setenv("GARAGE_RAG_TRANSLATE_QUERY", setting)
    assert not query_translation_enabled()


def test_translation_setting_rejects_typos(monkeypatch):
    from garage_rag.answering import query_translation_enabled

    monkeypatch.setenv("GARAGE_RAG_TRANSLATE_QUERY", "maybe")
    with pytest.raises(ValueError, match="GARAGE_RAG_TRANSLATE_QUERY"):
        query_translation_enabled()
