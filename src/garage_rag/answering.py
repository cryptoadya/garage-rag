"""Production Direct/Grounded entry point with original multilingual queries."""

import os

from .workflow import WorkflowLimits, run_direct, run_workflow


def query_translation_enabled():
    """Accept the legacy environment setting without enabling query rewriting.

    Existing .env files may contain true or false. Both now use the original
    question with the pinned multilingual embeddings; typos still fail startup.
    """
    value = os.environ.get("GARAGE_RAG_TRANSLATE_QUERY", "false").lower()
    if value not in {"true", "false"}:
        raise ValueError("GARAGE_RAG_TRANSLATE_QUERY must be true or false")
    return False


def run_answer(question, search, llm, *, mode="grounded", top_k=5,
               answer_language="auto", limits=None, translate_query=False):
    """Use the original question for local search, writing and auditing.

    The legacy translate_query boolean remains accepted for caller compatibility
    but has no effect. No language metadata, migration or translation call is needed.
    """
    if not question.strip() or len(question) > 4000:
        raise ValueError("Question must contain 1–4000 characters")
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if mode not in {"direct", "grounded"}:
        raise ValueError("mode must be direct or grounded")
    if type(translate_query) is not bool:
        raise ValueError("translate_query must be a boolean")
    runner = run_direct if mode == "direct" else run_workflow
    kwargs = {"top_k": top_k, "answer_language": answer_language}
    if mode == "grounded":
        kwargs["limits"] = limits or WorkflowLimits()
    return runner(question, lambda k: search(question, k), llm, **kwargs)
