"""Token headroom is separate from spent reservations and call slots."""

import json

import pytest
from conftest import Model, control, reply, tool
from test_workflow import TEXT, audit, draft, hit

from garage_rag.agentic import AgenticLimits, run_agentic


class PricedModel(Model):
    """Deterministic request prices, independent of reported/missing usage."""

    def __init__(self, *responses, audit_price=500):
        super().__init__(*responses)
        self.audit_price = audit_price

    def request_reservation(self, messages, tools=None):
        if tools:
            return 100 if len(messages) == 2 else 300
        if messages[0].content.startswith("You are the evidence auditor"):
            payload = json.loads(messages[-1].content)
            return self.audit_price if payload["draft"]["claims"] else 200
        return 400


def run(model, cap=2600, search=None):
    return run_agentic(
        "How should I check tire pressure?",
        search or (lambda *a: [hit()]),
        model,
        answer_language="en",
        limits=AgenticLimits(max_total_tokens=cap),
    )


def test_handoff_skips_control_turn_and_only_actual_calls_are_charged():
    model = PricedModel(tool(), reply(draft()), reply(audit()))
    result = run(model)
    assert result.answer.supported
    assert result.trace.llm_calls == 3
    assert result.trace.search_calls == 1
    assert result.trace.reserved_tokens == 1000  # 100 planner + 400 writer + 500 audit
    assert result.trace.handoffs[-1]["decision"] == "handoff"
    assert TEXT in model.messages[-1][-1].content


def test_planner_keeps_autonomy_when_extra_turn_and_terminal_pair_fit():
    result = run(PricedModel(tool(), control(), reply(draft()), reply(audit())), cap=5000)
    assert result.answer.supported
    assert result.trace.llm_calls == 4
    assert result.trace.reserved_tokens == 1300
    assert result.trace.handoffs[0]["decision"] == "continue"


def test_structured_planner_handoff_preserves_local_pair_and_citation_validation():
    from conftest import decision

    class Structured(PricedModel):
        structured_planner_supported = True

        def bind_planner(self):
            return self

    model = Structured(
        reply(json.loads(decision("search", method="dense", query="cold tire pressure", top_k=1))),
        reply(draft()),
        reply(audit()),
    )
    result = run(model)
    assert result.answer.supported
    assert result.trace.llm_calls == 3 and result.trace.reserved_tokens == 1000
    assert model.messages[1][-1].tool_call_id == "local_1"
    assert "aurora.pdf, PDF page 1" in result.answer.text


def test_pair_that_cannot_fit_refuses_before_writer_without_charging_forecasts():
    result = run(PricedModel(tool()), cap=600)
    assert result.answer.reason == "token_budget_exhausted"
    assert result.trace.llm_calls == 1
    assert result.trace.reserved_tokens == 100
    assert result.trace.handoffs[-1]["decision"] == "refuse"


def test_no_evidence_does_not_handoff_to_writer_to_save_tokens():
    result = run(PricedModel(tool()), cap=2600, search=lambda *a: [])
    assert result.answer.reason == "insufficient_evidence"
    assert result.trace.llm_calls == 1
    assert result.trace.reserved_tokens == 100


def test_navigation_only_context_refuses_when_another_planner_turn_threatens_audit():
    navigation = "Contents\nTire pressure95\nRadio85\nLighting90\nWindows110\nMirrors120"
    result = run(PricedModel(tool()), search=lambda *a: [hit(navigation)])
    assert result.answer.reason == "insufficient_evidence"
    assert result.trace.llm_calls == 1
    assert not result.trace.handoffs[-1]["has_substantive_evidence"]
    assert result.trace.handoffs[-1]["decision"] == "refuse"


def test_failed_writer_keeps_its_spent_reservation_without_refund_or_retry():
    from garage_rag.generation import GenerationError

    result = run(PricedModel(tool(), GenerationError("private provider failure")))
    assert result.answer.reason == "generation_error"
    assert result.trace.llm_calls == 2
    assert result.trace.reserved_tokens == 500


def test_new_retrieval_can_exhaust_terminal_capacity_without_dropping_its_warning():
    class GrowingContext(PricedModel):
        def request_reservation(self, messages, tools=None):
            if not tools and not messages[0].content.startswith("You are the evidence auditor"):
                if len(json.loads(messages[-1].content)["sources"]) == 2:
                    return 3000
            return super().request_reservation(messages, tools)

    batches = iter([[hit()], [hit("Warning: check tire pressure only when cold.", 2)]])
    result = run(
        GrowingContext(tool(), tool(query="tire pressure warning", call_id="c2")),
        cap=5000,
        search=lambda *a: next(batches),
    )
    assert result.answer.reason == "token_budget_exhausted"
    assert result.trace.llm_calls == result.trace.search_calls == 2
    assert result.trace.reserved_tokens == 400
    assert (
        result.context.sources["S2"].page_content == "Warning: check tire pressure only when cold."
    )


def test_large_actual_audit_cannot_escape_its_allocation_or_retry():
    result = run(PricedModel(tool(), reply(draft()), audit_price=10000))
    assert result.answer.reason == "token_budget_exhausted"
    assert not result.answer.supported
    assert result.trace.llm_calls == 2
    assert result.trace.reserved_tokens == 500


def test_actual_audit_can_use_spare_unspent_capacity_without_a_second_charge():
    result = run(PricedModel(tool(), reply(draft()), reply(audit()), audit_price=2000))
    assert result.answer.supported
    assert result.trace.reserved_tokens == 2500  # Forecast headroom was never charged.
    assert result.trace.terminal_budget["audit_capacity"] == 2100


def test_large_valid_draft_is_not_returned_when_its_actual_audit_does_not_fit():
    class LargeDraft(PricedModel):
        max_tokens = 2048

        def request_reservation(self, messages, tools=None):
            if not tools and messages[0].content.startswith("You are the evidence auditor"):
                payload = json.loads(messages[-1].content)["draft"]
                return 200 + len(json.dumps(payload).encode()) if payload["claims"] else 200
            return super().request_reservation(messages, tools)

    value = draft("Check the tire pressure when the tires are cold. " * 24)
    value["claims"] *= 6
    result = run(LargeDraft(tool(), control(), reply(value)), cap=4500)
    assert result.answer.reason == "token_budget_exhausted"
    assert not result.answer.supported
    assert len(result.candidate.draft.claims) == 6
    assert result.trace.llm_calls == 3
    assert result.trace.reserved_tokens == 800


@pytest.mark.parametrize("question", ["Tire pressure?", 'Как проверить шины?\n"context"'])
def test_audit_sizing_request_matches_unchanged_auditor_protocol(question):
    from garage_rag.agentic_budget import audit_messages
    from garage_rag.generation import Answer, Candidate, Context, Draft
    from garage_rag.grounding import audit_candidate

    context = Context(json.dumps([{"source_id": "S1", "text": TEXT}]), {"S1": hit().document})
    empty = Draft(supported=False, claims=[])
    candidate = Candidate(Answer(False, "", "insufficient_evidence", "en"), empty)
    observed = Model(reply({"decision": "insufficient", "complete": False, "checks": []}))
    audit_candidate(question, candidate, context, observed)
    expected = audit_messages(question, context, empty.model_dump())
    assert [(m.type, m.content) for m in expected] == [
        (m.type, m.content) for m in observed.messages[0]
    ]


def test_reservation_quotes_include_schema_but_never_mutate_shared_quota_or_trace(monkeypatch):
    from langchain_core.messages import HumanMessage, SystemMessage
    from test_groq import model

    from garage_rag.agentic import AGENT_PROMPT, AgenticTrace, _Budgeted
    from garage_rag.answer_evaluation import BudgetedPacedLLM
    from garage_rag.local_tools import PLANNER_TOOLS
    from garage_rag.planner import PLANNER_FORMAT

    def no_http(request):
        pytest.fail("A reservation quote must not invoke HTTP")

    llm = model(monkeypatch, no_http)
    trace = AgenticTrace()
    guard = BudgetedPacedLLM(llm, max_api_calls=6, max_total_tokens=24000)
    messages = [SystemMessage(content=AGENT_PROMPT), HumanMessage(content='{"question":"Tires?"}')]
    with llm.client:
        planner = _Budgeted(guard, trace, PLANNER_TOOLS)
        first = planner.estimate(messages)
        assert first == planner.estimate(messages)
        assert first == llm.request_reservation(messages, PLANNER_FORMAT)
        assert first > llm.request_reservation(messages)
    assert trace.llm_calls == trace.reserved_tokens == 0
    assert guard.calls == guard.reserved_tokens == 0
    assert llm._history == []


@pytest.mark.parametrize(
    "decision,reason",
    [
        ("conflicting", "conflicting_sources"),
        ("ambiguous", "ambiguous_question"),
        ("unsupported", "unsupported_claims"),
    ],
)
def test_handoff_keeps_mandatory_conflict_applicability_and_support_checks(decision, reason):
    result = run(PricedModel(tool(), reply(draft()), reply(audit(decision))))
    assert not result.answer.supported
    assert result.answer.reason == reason
    assert result.trace.llm_calls == 3


def test_handoff_does_not_allow_a_citation_outside_the_selected_source_map():
    result = run(PricedModel(tool(), reply(draft(source="S9"))))
    assert result.answer.reason == "invalid_model_output"
    assert result.trace.llm_calls == 2
