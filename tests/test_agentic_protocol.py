"""Regression contracts for JSON-answer instructions leaking into native tool turns."""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from test_agentic import Model, control, reply, tool
from test_groq import model, success
from test_workflow import audit, draft, hit

from garage_rag.agentic import AgenticLimits, run_agentic


@pytest.mark.parametrize("action", ["finalize", "refuse", "lexical_search"])
def test_second_native_planner_action_after_real_tool_result(monkeypatch, action):
    """Catch mixed text/tool control, forced extra searches and broken writer history."""
    requests, searches = [], []

    def transport(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "tools" not in payload:
            assert "tool_choice" not in payload
            assert payload["response_format"] == {"type": "json_object"}
            if "evidence auditor" in payload["messages"][0]["content"]:
                return success(json.dumps(audit()))
            # A terminal control is consumed locally; the writer keeps the last search pair.
            assert payload["messages"][-2]["tool_calls"][0]["function"]["name"] in {
                "dense_search",
                "lexical_search",
            }
            assert (
                payload["messages"][-2]["tool_calls"][0]["id"]
                == payload["messages"][-1]["tool_call_id"]
            )
            return success(json.dumps(draft()))
        assert payload["tool_choice"] == "required"
        assert payload["parallel_tool_calls"] is False
        assert "response_format" not in payload
        assert {t["function"]["name"] for t in payload["tools"]} == {
            "dense_search",
            "lexical_search",
            "finalize",
            "refuse",
        }
        first = len(requests) == 1
        if not first:
            assert [m["role"] for m in payload["messages"]] == [
                "system",
                "user",
                "assistant",
                "tool",
            ]
            assert payload["messages"][-1]["tool_call_id"] == "p1"
            assert json.loads(payload["messages"][-1]["content"])["sources"]
        name = "dense_search" if first else action
        args = (
            {"query": "first query" if first else "evidence followup", "top_k": 1}
            if name.endswith("search")
            else {"reason": "conflicting_sources"}
            if name == "refuse"
            else {}
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"p{len(requests)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
            },
        )

    def search(method, query, k):
        searches.append((method, query, k))
        return [hit()]

    llm = model(monkeypatch, transport)
    with llm.client:
        result = run_agentic(
            "When should I check pressure?",
            search,
            llm,
            answer_language="en",
            limits=AgenticLimits(max_searches=2),
        )
    assert result.error is None
    assert result.answer.supported == (action != "refuse")
    assert result.answer.reason == (
        "conflicting_sources" if action == "refuse" else "audited_evidence"
    )
    assert len(requests) == result.trace.llm_calls == (2 if action == "refuse" else 4)
    assert searches == (
        [("dense", "first query", 1), ("bm25", "evidence followup", 1)]
        if action == "lexical_search"
        else [("dense", "first query", 1)]
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("finalize", {"query": "must not run"}),
        ("refuse", {"reason": "invented"}),
        ("refuse", {"reason": "out_of_scope", "path": "/tmp/x"}),
    ],
)
def test_invalid_native_terminal_actions_stop_without_search_or_writer(name, args):
    response = AIMessage(content="", tool_calls=[{"id": "c1", "name": name, "args": args}])
    result = run_agentic("Question", lambda *a: pytest.fail("unexpected search"), Model(response))
    assert result.answer.reason == "invalid_tool_call"
    assert result.trace.llm_calls == 1 and result.trace.search_calls == 0


@pytest.mark.parametrize("text", ["FINALIZE", "REFUSE out_of_scope", "Need more detail."])
def test_plain_text_is_not_a_planner_control(text):
    result = run_agentic(
        "Question", lambda *a: pytest.fail("unexpected search"), Model(AIMessage(content=text))
    )
    assert result.answer.reason == "invalid_planner_output"
    assert result.trace.llm_calls == 1 and result.trace.search_calls == 0


def test_native_finalize_without_context_does_not_spend_writer_calls():
    response = AIMessage(content="", tool_calls=[{"id": "done", "name": "finalize", "args": {}}])
    result = run_agentic("Question", lambda *a: pytest.fail("unexpected search"), Model(response))
    assert result.answer.reason == "no_context"
    assert result.trace.llm_calls == 1 and result.trace.search_calls == 0


def test_score_filter_keeps_native_terminal_actions_available():
    class Checked(Model):
        def bind_tools(self, tools, **kwargs):
            assert {t["function"]["name"] for t in tools} == {"dense_search", "finalize", "refuse"}
            assert kwargs["tool_choice"] == "required"
            return self

    response = AIMessage(
        content="",
        tool_calls=[{"id": "stop", "name": "refuse", "args": {"reason": "out_of_scope"}}],
    )
    result = run_agentic(
        "Question",
        lambda *a: pytest.fail("unexpected search"),
        Checked(response),
        allow_lexical=False,
    )
    assert result.answer.reason == "out_of_scope"


def test_native_control_cannot_reuse_a_search_call_id():
    done = AIMessage(content="", tool_calls=[{"id": "c1", "name": "finalize", "args": {}}])
    result = run_agentic("Question", lambda *a: [hit()], Model(tool(), done))
    assert result.answer.reason == "invalid_tool_call"
    assert result.trace.llm_calls == 2 and result.trace.search_calls == 1


def test_mixed_plain_control_and_native_action_are_rejected():
    mixed = AIMessage(content="FINALIZE", tool_calls=tool().tool_calls)
    result = run_agentic("Question", lambda *a: pytest.fail("unexpected search"), Model(mixed))
    assert result.answer.reason == "invalid_tool_call"
    assert result.trace.llm_calls == 1 and result.trace.search_calls == 0


def test_required_binding_cannot_force_search_without_terminal_actions(monkeypatch):
    from garage_rag.generation import GenerationError
    from garage_rag.local_tools import TOOLS

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    with llm.client, pytest.raises(GenerationError, match="finalize and refuse"):
        llm.bind_tools(TOOLS, tool_choice="required")
    assert requests == []


def test_required_action_missing_from_200_response_blocks_without_retry(monkeypatch):
    from langchain_core.messages import HumanMessage

    from garage_rag.generation import GenerationError
    from garage_rag.local_tools import PLANNER_TOOLS

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success("FINALIZE"))
    with llm.client:
        bound = llm.bind_tools(PLANNER_TOOLS, tool_choice="required")
        with pytest.raises(GenerationError, match="invalid completion"):
            bound.invoke([HumanMessage(content="Question")])
        with pytest.raises(GenerationError, match="quota/error stop"):
            bound.invoke([HumanMessage(content="Must not retry")])
    assert len(requests) == 1


def test_one_search_three_calls_separates_tool_and_json_wire_contracts(monkeypatch):
    requests = []

    def transport(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            prompt = payload["messages"][0]["content"]
            assert '"supported"' not in prompt and '"claims"' not in prompt
            assert "return JSON" not in prompt
            assert "response_format" not in payload
            assert payload["tool_choice"] == "required"
            assert {t["function"]["name"] for t in payload["tools"]} == {
                "dense_search",
                "lexical_search",
                "finalize",
                "refuse",
            }
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "dense_search",
                                            "arguments": '{"query":"cold tire pressure","top_k":1}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                },
            )
        assert "tools" not in payload and "tool_choice" not in payload
        assert payload["response_format"] == {"type": "json_object"}
        if len(requests) == 2:
            assert payload["messages"][-2]["tool_calls"][0]["id"] == "c1"
            assert payload["messages"][-1]["tool_call_id"] == "c1"
            return success(json.dumps(draft()))
        return success(json.dumps(audit()))

    monkeypatch.setenv("GROQ_MAX_TOKENS", "512")
    llm = model(monkeypatch, transport)
    with llm.client:
        result = run_agentic(
            "When should I check tire pressure?",
            lambda *a: [hit()],
            llm,
            top_k=1,
            answer_language="en",
            limits=AgenticLimits(max_llm_calls=3, max_searches=1, max_total_tokens=16000),
        )
    assert result.answer.supported
    assert len(requests) == 3
    assert result.trace.reserved_tokens <= 16000


def test_default_early_finish_has_one_control_turn_and_keeps_native_pair():
    llm = Model(tool(), control(), reply(draft()), reply(audit()))
    result = run_agentic(
        "When should I check tire pressure?", lambda *a: [hit()], llm, answer_language="en"
    )
    assert result.answer.supported
    assert result.trace.llm_calls == 4
    assert llm.messages[1][-1].type == "tool"
    assert llm.messages[2][-1].tool_call_id == "c1"
    assert all("FINALIZE" not in m.content for m in llm.messages[2][1:])


@pytest.mark.parametrize(
    "reason", ["insufficient_evidence", "out_of_scope", "ambiguous_question", "conflicting_sources"]
)
def test_planner_refusal_is_a_closed_control_response_not_answer_json(reason):
    llm = Model(control(f"REFUSE {reason}"))
    result = run_agentic(
        "A question", lambda *a: pytest.fail("unexpected search"), llm, answer_language="en"
    )
    assert result.answer.reason == reason
    assert not result.answer.supported
    assert result.trace.llm_calls == 1


@pytest.mark.parametrize(
    "bad",
    [
        reply({"supported": False, "claims": [], "stop_reason": "out_of_scope"}),
        control("Some arbitrary answer"),
        control("REFUSE invent_reason"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "name": "json",
                    "args": {"supported": False, "claims": [], "stop_reason": "out_of_scope"},
                }
            ],
        ),
    ],
)
def test_bad_planner_output_never_becomes_a_draft_or_executes_a_tool(bad):
    result = run_agentic(
        "When should I check tires?",
        lambda *a: pytest.fail("unexpected tool"),
        Model(bad),
        answer_language="en",
    )
    assert result.answer.reason in {"invalid_planner_output", "invalid_tool_call"}
    assert result.trace.llm_calls == 1
    assert result.trace.search_calls == 0


def test_historical_tool_arguments_are_reserved_without_advertising_tools(monkeypatch):
    from langchain_core.messages import HumanMessage, ToolMessage

    llm = model(monkeypatch, lambda r: success())
    messages = [
        HumanMessage(content="question"),
        tool(query="x" * 500),
        ToolMessage(content='{"sources":[]}', tool_call_id="c1"),
    ]
    with llm.client:
        reserve = llm.request_reservation(messages)
        assert reserve > llm.max_tokens + 600
        assert llm.invoke(messages).content


def test_captured_error_class_still_stops_without_repair_or_retry(monkeypatch):
    requests = []
    error = {
        "message": "Tool call validation failed: attempted to call tool 'json' "
        "which was not in request.tools",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": json.dumps(
            {
                "name": "json",
                "arguments": {"supported": False, "claims": [], "stop_reason": "out_of_scope"},
            }
        ),
    }
    llm = model(
        monkeypatch, lambda r: requests.append(r) or httpx.Response(400, json={"error": error})
    )
    with llm.client:
        result = run_agentic(
            "When should I check tire pressure?", lambda *a: [hit()], llm, answer_language="en"
        )
    assert result.answer.reason == "generation_error"
    assert len(requests) == result.trace.llm_calls == 1
    assert result.trace.search_calls == 0
    assert "failed_generation" not in json.dumps(result.public_dict())


@pytest.mark.parametrize("calls,searches,expected_calls", [(3, 4, 3), (6, 1, 3)])
def test_budget_boundary_finalizes_without_an_extra_planning_call(calls, searches, expected_calls):
    llm = Model(tool(), reply(draft()), reply(audit()))
    result = run_agentic(
        "When should I check tire pressure?",
        lambda *a: [hit()],
        llm,
        answer_language="en",
        limits=AgenticLimits(max_llm_calls=calls, max_searches=searches),
    )
    assert result.answer.supported
    assert result.trace.llm_calls == expected_calls
    assert result.trace.search_calls == 1


def test_finalize_without_evidence_refuses_without_paying_for_writer_or_audit():
    result = run_agentic(
        "A question",
        lambda *a: pytest.fail("unexpected search"),
        Model(control()),
        answer_language="en",
    )
    assert result.answer.reason == "no_context"
    assert result.trace.llm_calls == 1


@pytest.mark.parametrize("name", ["json", "shell", "browser_search"])
def test_no_tools_writer_still_rejects_unapproved_historical_functions(monkeypatch, name):
    from langchain_core.messages import HumanMessage, ToolMessage

    from garage_rag.generation import GenerationError

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": name, "args": {}}]),
        ToolMessage(content="untrusted result", tool_call_id="c1"),
    ]
    with llm.client, pytest.raises(GenerationError):
        llm.invoke(messages)
    assert requests == []


def test_writer_cannot_return_a_new_search_even_if_function_is_allowlisted():
    result = run_agentic(
        "When should I check pressure?",
        lambda *a: [hit()],
        Model(tool(), tool(query="another query", call_id="c2")),
        answer_language="en",
        limits=AgenticLimits(max_searches=1),
    )
    assert result.answer.reason == "invalid_tool_call"
    assert result.trace.search_calls == 1
    assert result.trace.llm_calls == 2


def test_separate_writer_preserves_prior_warning_and_uses_only_latest_native_pair():
    llm = Model(
        tool(),
        tool(query="warnings", call_id="c2"),
        control(),
        reply(draft()),
        reply(audit("conflicting")),
    )
    result = run_agentic(
        "When should I check pressure?",
        lambda method, query, k: (
            [hit()]
            if "cold" in query
            else [hit("Do not check tire pressure when the tires are cold.", 2)]
        ),
        llm,
        answer_language="en",
    )
    assert result.answer.reason == "conflicting_sources"
    writer_messages = llm.messages[-2]
    assert len(writer_messages) == 4  # fresh system, question, last native call/result
    assert writer_messages[-1].tool_call_id == "c2"
    assert len(json.loads(writer_messages[-1].content)["sources"]) == 2
