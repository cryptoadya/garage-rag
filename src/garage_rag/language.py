"""Local language selection; technical identifiers are not language evidence."""

import re
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

LANGUAGES = {"en": "English", "de": "German", "ru": "Russian"}
REFUSALS = {
    "en": "There is not enough evidence in the retrieved context to answer this question.",
    "de": "Der abgerufene Kontext enthält nicht genügend Belege, um diese Frage zu beantworten.",
    "ru": "В извлечённом контексте недостаточно подтверждений для ответа на этот вопрос.",
}
# Shared names and units can otherwise dominate very short technical questions.
TECHNICAL_TERMS = frozenset(
    "bmw dde dme abs dsc asc ews obd obdii ecu can ibus vanos steptronic "
    "tiptronic valvetronic xdrive pdf bar psi nm km kph rpm mph mm cm kg kw".split()
)


@lru_cache(maxsize=1)
def _detector():
    return LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.GERMAN, Language.RUSSIAN
    ).build()


def detect_language(text: str) -> str | None:
    """Return a confident EN/DE/RU guess, or None for ambiguous linguistic input.

    Confidence is relative to these three candidates, not a calibrated probability.
    This is a heuristic for supported languages, not an arbitrary-language classifier.
    """
    words = re.findall(r"[^\W_]+", text.casefold())
    linguistic = " ".join(word for word in words if word not in TECHNICAL_TERMS and word.isalpha())
    if len(linguistic) < 3:
        return None
    scores = _detector().compute_language_confidence_values(linguistic)
    if scores[0].value < 0.7 or scores[0].value - scores[1].value < 0.25:
        return None
    return scores[0].language.iso_code_639_1.name.lower()


def resolve_answer_language(question: str, requested: str = "auto") -> str:
    if requested not in (*LANGUAGES, "auto"):
        raise ValueError("answer_language must be auto, en, de or ru")
    return (detect_language(question) or "en") if requested == "auto" else requested
