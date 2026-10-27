"""Strict planner boundary; transport replay is not provider-quality evidence."""

import json
from pathlib import Path

import httpx
import pytest
from test_groq import model, success
from test_workflow import audit, draft, hit

from garage_rag.agentic import run_agentic
from garage_rag.answer_evaluation import BudgetedPacedLLM
from garage_rag.source_evaluation import ObservedEvaluationLLM


def decision(action, **kwargs):
    return json.dumps(
        dict(action=action, method=None, query=None, top_k=None, reason=None) | kwargs
    )


def test_structured_second_planner_keeps_session_capture_and_writer(monkeypatch, tmp_path):
    requests, searches = [], []

    def transport(request):
        b = json.loads(request.content)
        requests.append(b)
        if b["response_format"]["type"] == "json_schema":
            assert b["response_format"]["json_schema"]["strict"] is True
            assert not {"tools", "tool_choice", "parallel_tool_calls"} & b.keys()
            assert all(
                m["role"] in {"system", "user", "assistant"} and "tool_calls" not in m
                for m in b["messages"]
            )
            if len(requests) == 1:
                return success(
                    decision("search", method="bm25", query="cold tire pressure", top_k=1)
                )
            assert json.loads(b["messages"][-1]["content"])["sources"]
            return success(decision("finalize"))
        return success(
            json.dumps(audit() if "evidence auditor" in b["messages"][0]["content"] else draft())
        )

    llm = model(monkeypatch, transport)
    # No real sleep or remote transport; retain the actual shared session guard.
    monkeypatch.setattr("garage_rag.answer_evaluation.time.sleep", lambda s: None)
    guard = BudgetedPacedLLM(llm, max_api_calls=4, max_total_tokens=24000)
    guard.requests_per_minute = 10**9
    observed = ObservedEvaluationLLM(guard)
    with llm.client:
        result = run_agentic(
            "When should I check pressure?",
            lambda *a: searches.append(a) or [hit()],
            observed,
            answer_language="en",
        )
    assert result.answer.supported
    assert len(requests) == guard.calls == result.trace.llm_calls == 4
    assert searches == [("bm25", "cold tire pressure", 1)]
    assert guard.reserved_tokens == result.trace.reserved_tokens
    assert observed.calls[0]["response_format"]["json_schema"]["strict"] is True
    assert json.loads(observed.calls[1]["response"])["action"] == "finalize"
    assert observed.public_calls[1]["phase"] == "planner"
    from dataclasses import asdict

    from garage_rag.answer_evaluation import _row
    from garage_rag.review_packet import write_packet

    row = _row(result, {"id": "fixture", "kind": "answerable"}, "en", "agentic", 1, scripted=False)
    row.update(sent_contexts=observed.sent_contexts, model_calls=observed.public_calls)
    capture = {
        "key": {k: row[k] for k in ("id", "language", "mode", "trial", "answer_sha256")},
        "answer": asdict(result.answer),
        "draft": result.candidate.draft.model_dump(),
        "context": json.loads(result.context.text),
        "model_calls": observed.calls,
    }
    report = {"dataset_sha256": "fixture", "rows": [row]}
    write_packet(report, [capture], tmp_path / "review")
    capture["model_calls"][0]["response_format"]["json_schema"]["strict"] = False
    with pytest.raises(ValueError, match="native tools"):
        write_packet(report, [capture], tmp_path / "tampered")


@pytest.mark.parametrize(
    "bad",
    [
        {"action": "finalize"},
        json.loads(decision("finalize", query="hidden search")),
        json.loads(decision("refuse", reason="invented")),
        json.loads(decision("search", method="shell", query="x", top_k=1)),
        json.loads(decision("search", method="dense", query=" ", top_k=1)),
        json.loads(decision("search", method="dense", query="x", top_k=True)),
    ],
)
def test_strict_local_validation_never_dispatches_bad_decisions(monkeypatch, bad):
    llm = model(monkeypatch, lambda r: success(json.dumps(bad)))
    with llm.client:
        result = run_agentic("Question", lambda *a: pytest.fail("unexpected search"), llm)
    assert not result.answer.supported
    assert result.trace.search_calls == 0 and result.trace.llm_calls == 1


@pytest.mark.parametrize(
    "directory",
    [
        "phase4b-live-executed-20261027",
        "phase4b-real-live-executed-20261027",
        "phase4b-final-live-65da46f-20261027",
    ],
)
def test_saved_failure_still_stops_without_retry(monkeypatch, directory):
    path = Path("output/private") / directory / "wire.json"
    if not path.exists():
        pytest.skip("Private historical capture is not distributed")
    error = json.loads(path.read_text())[-1]["response"]
    sent = []

    def transport(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return success(decision("search", method="dense", query="cold tire pressure", top_k=1))
        return httpx.Response(400, json=error)

    llm = model(monkeypatch, transport)
    with llm.client:
        result = run_agentic("Question", lambda *a: [hit()], llm)
    assert len(sent) == result.trace.llm_calls == 2
    assert result.error and result.trace.search_calls == 1
    assert llm._history[-1][1] > llm.max_tokens


@pytest.mark.parametrize(
    "status,code",
    [(400, "output_parse_failed"), (400, "tool_use_failed"), (429, "rate_limit_exceeded")],
)
def test_public_failure_classes_stop_structured_continuation(monkeypatch, status, code):
    sent = []

    def transport(request):
        sent.append(request)
        if len(sent) == 1:
            return success(decision("search", method="dense", query="cold tire pressure", top_k=1))
        return httpx.Response(
            status,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": code,
                    "message": "Recorded failure class",
                }
            },
        )

    llm = model(monkeypatch, transport)
    with llm.client:
        result = run_agentic("Question", lambda *a: [hit()], llm)
    assert result.error and result.trace.provider_errors[-1]["http_status"] == status
    assert len(sent) == 2 and llm._history[-1][1] > llm.max_tokens


@pytest.mark.parametrize("action", ["refuse", "search"])
def test_structured_followup_preserves_autonomy_and_local_score_filter(monkeypatch, action):
    sent = []
    searches = []

    def transport(request):
        sent.append(request)
        return success(
            decision("refuse", reason="conflicting_sources")
            if action == "refuse"
            else decision("search", method="bm25", query="cold tire pressure", top_k=1)
        )

    llm = model(monkeypatch, transport)
    with llm.client:
        result = run_agentic("Question", lambda *a: searches.append(a), llm, allow_lexical=False)
    assert result.answer.reason == (
        "conflicting_sources" if action == "refuse" else "invalid_tool_call"
    )
    assert searches == [] and len(sent) == 1


def test_structured_binding_cannot_bypass_session_limit(monkeypatch):
    from langchain_core.messages import HumanMessage

    from garage_rag.generation import GenerationError

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success(decision("finalize")))
    guard = BudgetedPacedLLM(llm, max_api_calls=1, max_total_tokens=8000)
    with llm.client:
        bound = ObservedEvaluationLLM(guard).bind_planner()
        bound.invoke([HumanMessage(content="question")])
        held = guard.reserved_tokens
        with pytest.raises(GenerationError, match="API budget"):
            bound.invoke([HumanMessage(content="question")])
    assert len(requests) == guard.calls == 1 and guard.reserved_tokens == held


def test_structured_schema_reserved_and_native_history_rejected_before_http(monkeypatch):
    from langchain_core.messages import HumanMessage, ToolMessage
    from test_agentic import tool

    from garage_rag.generation import GenerationError
    from garage_rag.planner import PLANNER_FORMAT

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    messages = [HumanMessage(content="question")]
    with llm.client:
        assert (
            llm.request_reservation(messages, PLANNER_FORMAT)
            > llm.request_reservation(messages) + 500
        )
        with pytest.raises(GenerationError, match="cannot use native"):
            llm.bind_planner().invoke(
                [*messages, tool(), ToolMessage(content="{}", tool_call_id="c1")]
            )
    assert requests == []
