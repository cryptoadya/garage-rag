"""Bounded LangGraph answering. Only missing evidence can trigger one local expansion."""

import time
from dataclasses import asdict, dataclass, replace
from typing import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import ValidationError

from .generation import (
    Answer,
    Candidate,
    Context,
    GenerationError,
    build_context,
    generate_candidate,
)
from .grounding import audit_candidate, numeric_evidence_safe
from .index import SearchHit
from .language import REFUSALS, resolve_answer_language
from .telemetry import MeteredLLM, NodeEvent, RunTrace


@dataclass(frozen=True)
class WorkflowLimits:
    max_llm_calls: int = 4
    max_searches: int = 2

    def __post_init__(self):
        if type(self.max_llm_calls) is not int or not 1 <= self.max_llm_calls <= 4:
            raise ValueError("max_llm_calls must be between 1 and 4")
        if type(self.max_searches) is not int or not 1 <= self.max_searches <= 2:
            raise ValueError("max_searches must be 1 or 2")


@dataclass(frozen=True)
class WorkflowResult:
    answer: Answer
    trace: RunTrace
    candidate: Candidate | None = None  # Private, never serialized by public_dict.
    context: Context | None = None
    error: str | None = None

    def public_dict(self):
        return {**asdict(self.answer), "workflow": self.trace.public_dict()}


class State(TypedDict, total=False):
    context: Context
    candidate: Candidate
    reason: str
    error: str | None
    route: str
    answer: Answer


def _identity(document):
    return (document.metadata["document"], document.metadata["pdf_page"], document.page_content)


def run_workflow(question, search, llm, *, top_k=5, answer_language="auto", limits=None):
    """search(k) must run local retrieval of the unchanged question and filters.

    No query rewriting, retries, checkpoints, parallel nodes or remote tracing.
    The first draft and direct baseline use the exact same generation primitive.
    """
    if not question.strip() or len(question) > 4000:
        raise ValueError("Question must contain 1–4000 characters")
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    limits = limits or WorkflowLimits()
    language = resolve_answer_language(question, answer_language)
    trace = RunTrace(llm_call_limit=limits.max_llm_calls, search_call_limit=limits.max_searches)
    model = MeteredLLM(llm, trace)
    started = time.perf_counter()

    def refusal(reason):
        return Answer(False, REFUSALS[language], reason, language)

    def can_expand():
        return top_k < 20 and trace.search_calls < limits.max_searches

    def missing(reason):
        if not can_expand():
            return {"reason": reason, "route": "finish"}
        if limits.max_llm_calls - trace.llm_calls < 2:
            return {"reason": "llm_budget_exhausted", "route": "finish"}
        return {"reason": reason, "route": "expand"}

    def retrieve(state):
        trace.search_calls += 1
        try:
            context = build_context(search(top_k))
        except Exception:
            return {
                "reason": "retrieval_error",
                "error": "Local retrieval failed.",
                "route": "finish",
            }
        return {
            "context": context,
            "route": "draft" if context.sources else "finish",
            "reason": "context_available" if context.sources else "no_context",
        }

    def expand(state):
        if not can_expand():
            return {"route": "finish"}
        trace.search_calls += 1
        try:
            hits = search(20)
            old = {_identity(d) for d in state["context"].sources.values()}
            novel = [h for h in hits if _identity(h.document) not in old]
            # Keep ALL previously seen constraints. An expansion must not make
            # the auditor forget a warning/conflict that the first draft saw.
            previous = [SearchHit(d, 0) for d in state["context"].sources.values()]
            context = build_context([*previous, *novel])
            # Extra returned hits are useless unless they fit in the actual sent context.
            if not any(_identity(d) not in old for d in context.sources.values()):
                return {"reason": "no_new_context", "route": "finish"}
        except Exception:
            return {
                "reason": "retrieval_error",
                "error": "Local retrieval failed.",
                "route": "finish",
            }
        return {"context": context, "route": "draft"}

    def draft(state):
        # Reserve both generation and audit before spending another completion.
        if limits.max_llm_calls - trace.llm_calls < 2:
            return {"reason": "llm_budget_exhausted", "route": "finish"}
        try:
            candidate = generate_candidate(question, state["context"], model, language)
        except GenerationError as exc:
            return {"reason": "generation_error", "error": str(exc), "route": "finish"}
        if not candidate.answer.supported:
            reason = candidate.answer.reason
            return (
                {"candidate": candidate, **missing(reason)}
                if reason == "insufficient_evidence"
                else {"candidate": candidate, "reason": reason, "route": "finish"}
            )
        if not numeric_evidence_safe(candidate.draft, state["context"]):
            return {"candidate": candidate, "reason": "unsafe_numeric_evidence", "route": "finish"}
        return {"candidate": candidate, "route": "audit"}

    def audit(state):
        if trace.llm_calls >= limits.max_llm_calls:
            return {"reason": "llm_budget_exhausted", "route": "finish"}
        try:
            review = audit_candidate(question, state["candidate"], state["context"], model)
        except GenerationError as exc:
            return {"reason": "generation_error", "error": str(exc), "route": "finish"}
        except (ValidationError, ValueError, TypeError) as exc:
            reason = (
                "output_token_limit" if str(exc) == "output_token_limit" else "invalid_audit_output"
            )
            return {"reason": reason, "route": "finish"}
        if review.decision == "accept":
            return {
                "answer": replace(state["candidate"].answer, reason="audited_evidence"),
                "reason": "audited_evidence",
                "route": "finish",
            }
        if review.decision == "insufficient":
            return missing("insufficient_evidence")
        reasons = {
            "conflicting": "conflicting_sources",
            "ambiguous": "ambiguous_question",
            "out_of_scope": "out_of_scope",
            "unsafe_numeric": "unsafe_numeric_evidence",
            "unsupported": "unsupported_claims",
        }
        return {"reason": reasons[review.decision], "route": "finish"}

    def finish(state):
        return {"answer": state.get("answer") or refusal(state["reason"]), "route": "done"}

    def observed(name, fn):
        def node(state):
            began = time.perf_counter()
            update = fn(state)
            context = update.get("context", state.get("context"))
            trace.events.append(
                NodeEvent(
                    name,
                    update.get("reason", update.get("route", "done")),
                    (time.perf_counter() - began) * 1000,
                    len(context.sources) if context else 0,
                )
            )
            return update

        return node

    graph = StateGraph(State)
    for name, fn in [
        ("retrieve", retrieve),
        ("expand", expand),
        ("draft", draft),
        ("audit", audit),
        ("finish", finish),
    ]:
        graph.add_node(name, observed(name, fn))
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", lambda s: s["route"], ["draft", "finish"])
    graph.add_conditional_edges("expand", lambda s: s["route"], ["draft", "finish"])
    graph.add_conditional_edges("draft", lambda s: s["route"], ["audit", "expand", "finish"])
    graph.add_conditional_edges("audit", lambda s: s["route"], ["expand", "finish"])
    graph.add_edge("finish", END)
    try:
        # Disable tracing for the WHOLE graph, including private state and all LLM calls.
        with tracing_context(enabled=False):
            state = graph.compile().invoke(
                {"reason": "workflow_error", "error": None},
                config={"recursion_limit": 16, "callbacks": []},
            )
    except GraphRecursionError:
        state = {"answer": refusal("step_limit"), "error": "Workflow step limit reached."}
    except Exception:
        state = {"answer": refusal("workflow_error"), "error": "Answer workflow failed."}
    trace.elapsed_ms = (time.perf_counter() - started) * 1000
    return WorkflowResult(
        state["answer"], trace, state.get("candidate"), state.get("context"), state.get("error")
    )


def run_direct(question, search, llm, *, top_k=5, answer_language="auto"):
    """Instrument the unchanged retrieve -> generate baseline for paired evaluation."""
    if not question.strip() or len(question) > 4000:
        raise ValueError("Question must contain 1–4000 characters")
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    language = resolve_answer_language(question, answer_language)
    trace = RunTrace(mode="direct", llm_call_limit=1, search_call_limit=1)
    started = time.perf_counter()
    candidate, context, error = None, None, None
    reason = "retrieval_error"
    try:
        trace.search_calls += 1
        context = build_context(search(top_k))
        trace.events.append(
            NodeEvent(
                "retrieve",
                "context" if context.sources else "no_context",
                (time.perf_counter() - started) * 1000,
                len(context.sources),
            )
        )
        reason = "generation_error"
        began = time.perf_counter()
        candidate = generate_candidate(question, context, MeteredLLM(llm, trace), language)
        result = candidate.answer
        trace.events.append(
            NodeEvent(
                "draft", result.reason, (time.perf_counter() - began) * 1000, len(context.sources)
            )
        )
    except GenerationError as exc:
        error = str(exc)
        result = Answer(False, REFUSALS[language], reason, language)
    except Exception:
        error = "Direct answering failed."
        result = Answer(False, REFUSALS[language], reason, language)
    trace.elapsed_ms = (time.perf_counter() - started) * 1000
    return WorkflowResult(result, trace, candidate, context, error)
