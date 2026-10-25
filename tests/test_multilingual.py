"""Offline multilingual contracts using the independent Aurora corpus."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from garage_rag.generation import answer
from garage_rag.index import SearchHit

QUESTIONS = {
    "en": "How do I check BMW tire pressure?",
    "de": "Wie prüfe ich den Reifendruck beim BMW?",
    "ru": "Как проверить давление в шинах BMW?",
}
CLAIMS = {
    "en": "Check tire pressure when the tires are cold.",
    "de": "Prüfen Sie den Reifendruck bei kalten Reifen.",
    "ru": "Проверяйте давление на холодных шинах.",
}
REFUSALS = {
    "en": "There is not enough evidence in the retrieved context to answer this question.",
    "de": "Der abgerufene Kontext enthält nicht genügend Belege, um diese Frage zu beantworten.",
    "ru": "В извлечённом контексте недостаточно подтверждений для ответа на этот вопрос.",
}


class Model:
    def __init__(self, value):
        self.value = value
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=json.dumps(self.value, ensure_ascii=False))


def evidence_hits(corpus):
    return [
        SearchHit(
            document=Document(
                page_content=corpus["pages"][0],
                metadata={"document": "aurora.pdf", "pdf_page": 1, "page_label": "99"},
            ),
            score=0.9,
        )
    ]


def draft(language):
    return {
        "supported": True,
        "claims": [
            {
                "text": CLAIMS[language],
                "evidence": [
                    {"source_id": "S1", "quote": "Check tire pressure when the tires are cold."}
                ],
            }
        ],
    }


@pytest.mark.parametrize("language", QUESTIONS)
def test_auto_language_and_verbatim_evidence(language, corpus):
    llm = Model(draft(language))
    result = answer(QUESTIONS[language], evidence_hits(corpus), llm)
    assert result.supported
    assert result.language == language
    assert CLAIMS[language] in result.text
    assert "aurora.pdf, PDF page 1" in result.text
    assert "PDF page 99" not in result.text
    assert '"Check tire pressure when the tires are cold."' in result.text
    name = {"en": "English", "de": "German", "ru": "Russian"}[language]
    assert f"Write every claim in {name}" in llm.messages[0].content
    payload = json.loads(llm.messages[1].content)
    assert payload["question"] == QUESTIONS[language]
    assert payload["sources"][0]["text"] == corpus["pages"][0]


@pytest.mark.parametrize("question_language", QUESTIONS)
@pytest.mark.parametrize("answer_language", QUESTIONS)
def test_explicit_override(question_language, answer_language, corpus):
    result = answer(
        QUESTIONS[question_language],
        evidence_hits(corpus),
        Model(draft(answer_language)),
        answer_language=answer_language,
    )
    assert result.supported
    assert result.language == answer_language


@pytest.mark.parametrize("language", QUESTIONS)
@pytest.mark.parametrize("mode", ["no_context", "refused", "invalid"])
def test_localized_refusals(language, mode, corpus):
    llm = Model({"supported": False, "claims": []} if mode == "refused" else {"bad": "json"})
    result = answer(QUESTIONS[language], [] if mode == "no_context" else evidence_hits(corpus), llm)
    assert not result.supported
    assert result.text == REFUSALS[language]
    assert result.language == language
    if mode == "no_context":
        assert not llm.messages


@pytest.mark.parametrize("language", QUESTIONS)
@pytest.mark.parametrize("corruption", ["source", "quote", "translated_quote", "page", "language"])
def test_multilingual_validation_fails_closed(language, corruption, corpus):
    value = draft(language)
    claim = value["claims"][0]
    if corruption == "source":
        claim["evidence"][0]["source_id"] = "S99"
    elif corruption == "quote":
        claim["evidence"][0]["quote"] = "The invented tire pressure is 9.9 bar."
    elif corruption == "translated_quote":
        claim["evidence"][0]["quote"] = CLAIMS["de"]
    elif corruption == "page":
        claim["pdf_page"] = 99
    else:
        claim["text"] = CLAIMS["de" if language == "en" else "en"]
    result = answer(QUESTIONS[language], evidence_hits(corpus), Model(value))
    assert not result.supported
    assert result.text == REFUSALS[language]
    assert result.reason == "invalid_model_output"


@pytest.mark.parametrize("question", ["BMW", "DDE ABS Steptronic", "E39 525i 2.4 bar", "???"])
def test_technical_or_uncertain_questions_fall_back_to_english(question):
    result = answer(question, [], None)
    assert result.text == REFUSALS["en"]


def test_invalid_override_is_rejected_before_model_call(corpus):
    with pytest.raises(ValueError, match="answer_language"):
        answer("tires?", evidence_hits(corpus), None, answer_language="fr")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("BMW DDE ABS Steptronic: how does it work?", "en"),
        ("BMW DDE ABS Steptronic: Wie funktioniert das?", "de"),
        ("BMW DDE ABS Steptronic: как это работает?", "ru"),
        ("Reifendruck?", "de"),
        ("Ölstand?", "de"),
        ("Давление?", "ru"),
        ("Tire pressure?", "en"),
        ("Motor?", "en"),
        ("BMW DDE ABS", "en"),
        ("bmw dde abs steptronic", "en"),
    ],
)
def test_short_questions_and_technical_terms(question, expected):
    result = answer(question, [], None)
    assert result.text == REFUSALS[expected]


@pytest.mark.parametrize("language", QUESTIONS)
def test_empty_claims_and_bracket_citations_are_rejected(language, corpus):
    for value in [{"supported": True, "claims": []}, draft(language)]:
        if value["claims"]:
            value["claims"][0]["text"] += " [PDF page 99]"
        result = answer(QUESTIONS[language], evidence_hits(corpus), Model(value))
        assert not result.supported
        assert result.reason == "invalid_model_output"


@pytest.mark.parametrize("language", QUESTIONS)
def test_override_applies_to_no_context_refusal(language):
    result = answer(QUESTIONS["en"], [], None, answer_language=language)
    assert result.text == REFUSALS[language]
    assert result.language == language
