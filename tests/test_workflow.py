import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from garage_rag.index import SearchHit

TEXT = "Check tire pressure when the tires are cold."


def hit(text=TEXT, page=1):
    return SearchHit(
        Document(
            page_content=text,
            metadata={
                "document": "aurora.pdf",
                "pdf_page": page,
                "chunk_index": 0,
            },
        ),
        0.8,
    )


def draft(text="Check the tire pressure when the tires are cold.", quote=TEXT, source="S1"):
    return {
        "supported": True,
        "claims": [
            {
                "text": text,
                "evidence": [{"source_id": source, "quote": quote}],
            }
        ],
    }


def audit(decision="accept", *, entailed=True, complete=True):
    return {
        "decision": decision,
        "complete": complete,
        "checks": [
            {
                "claim_index": 0,
                "entailed": entailed,
                "applicable": True,
                "numbers_safe": True,
            }
        ],
    }


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            content=json.dumps(value),
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
            },
        )


def run(llm, search=None, **kwargs):
    from garage_rag.workflow import run_workflow

    kwargs.setdefault("answer_language", "en")
    return run_workflow(
        "How should I check tire pressure?",
        search or (lambda k: [hit()]),
        llm,
        **kwargs,
    )


def test_valid_answer_requires_audit_and_retains_local_citations():
    result = run(Scripted(draft(), audit()))
    assert result.answer.supported
    assert "aurora.pdf, PDF page 1" in result.answer.text
    assert result.answer.reason == "audited_evidence"
    assert result.trace.llm_calls == 2
    assert result.trace.search_calls == 1
    assert result.trace.input_tokens == 20
    assert [e.node for e in result.trace.events] == ["retrieve", "draft", "audit", "finish"]


def test_valid_citation_does_not_make_an_unsupported_claim_true():
    result = run(
        Scripted(
            draft("Check the tire pressure when the tires are hot."),
            audit("unsupported", entailed=False),
        )
    )
    assert not result.answer.supported
    assert result.answer.reason == "unsupported_claims"
    assert result.trace.llm_calls == 2


@pytest.mark.parametrize(
    "decision,reason",
    [
        ("conflicting", "conflicting_sources"),
        ("ambiguous", "ambiguous_question"),
        ("out_of_scope", "out_of_scope"),
        ("unsafe_numeric", "unsafe_numeric_evidence"),
    ],
)
def test_nonrecoverable_audit_never_expands(decision, reason):
    result = run(Scripted(draft(), audit(decision)))
    assert not result.answer.supported
    assert result.answer.reason == reason
    assert result.trace.search_calls == 1


def test_insufficient_context_expands_once_without_discarding_originals():
    searches = []

    def search(k):
        searches.append(k)
        return [hit()] if k == 5 else [hit(), hit("The spare key is in the blue box.", 2)]

    llm = Scripted(
        {"supported": False, "claims": []},
        draft(
            "The spare key is in the blue box.", "The spare key is in the blue box.", source="S2"
        ),
        audit(),
    )
    result = run(llm, search)
    assert result.answer.supported
    assert searches == [5, 20]
    assert "PDF page 2" in result.answer.text
    assert result.trace.llm_calls == 3


def test_audit_can_request_one_expansion_before_accepting_complete_answer():
    llm = Scripted(draft(), audit("insufficient", complete=False), draft(), audit())
    result = run(
        llm, lambda k: [hit()] if k == 5 else [hit(), hit("Extra passage for context.", 2)]
    )
    assert result.answer.supported
    assert result.trace.llm_calls == 4
    assert result.trace.search_calls == 2


def test_no_new_sent_context_stops_without_another_generation():
    result = run(Scripted({"supported": False, "claims": []}))
    assert not result.answer.supported
    assert result.answer.reason == "no_new_context"
    assert result.trace.llm_calls == 1
    assert result.trace.search_calls == 2


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_budgets_never_allow_unaudited_answer(budget):
    from garage_rag.workflow import WorkflowLimits

    llm = Scripted(draft(), audit("insufficient", complete=False), draft(), audit())
    result = run(
        llm,
        lambda k: [hit()] if k == 5 else [hit(), hit("Extra passage.", 2)],
        limits=WorkflowLimits(max_llm_calls=budget),
    )
    assert result.trace.llm_calls <= budget
    assert result.answer.supported == (budget == 4)


def test_empty_context_uses_no_model_and_reports_no_context():
    result = run(None, lambda k: [])
    assert not result.answer.supported
    assert result.answer.reason == "no_context"
    assert result.trace.llm_calls == 0
    assert result.trace.search_calls == 1


@pytest.mark.parametrize("change", ["quote", "source", "language", "brackets"])
def test_local_validation_blocks_before_audit(change):
    value = draft()
    claim = value["claims"][0]
    if change == "quote":
        claim["evidence"][0]["quote"] = "A completely fabricated quotation."
    elif change == "source":
        claim["evidence"][0]["source_id"] = "S99"
    elif change == "language":
        claim["text"] = "Проверяйте давление на холодных шинах."
    else:
        claim["text"] += " [S1]"
    result = run(Scripted(value))
    assert result.answer.reason == "invalid_model_output"
    assert result.trace.llm_calls == 1


@pytest.mark.parametrize(
    "review",
    [
        audit(entailed=False),
        audit(complete=False),
        {"decision": "accept", "complete": True, "checks": []},
        {"decision": "accept", "complete": True, "checks": audit()["checks"] * 2},
        {**audit(), "extra": "untrusted"},
    ],
)
def test_invalid_or_inconsistent_audit_fails_closed(review):
    result = run(Scripted(draft(), review))
    assert not result.answer.supported
    assert result.trace.llm_calls == 2


@pytest.mark.parametrize(
    "text,claim",
    [
        ("Pressure table\nFront Rear\nA1 2.2 2.8\nA2 2.5 3.0", "The tire pressure is 2.2 bar."),
        ("The tire pressure is 2.\ufffd bar.", "The tire pressure is 2.4 bar."),
        ("The tightening torque is 90 Nm.", "The tightening torque is 100 Nm."),
    ],
)
def test_numeric_guard_rejects_tables_corruption_and_number_substitution(text, claim):
    result = run(Scripted(draft(claim, text)), lambda k: [hit(text)])
    assert not result.answer.supported
    assert result.answer.reason == "unsafe_numeric_evidence"
    assert result.trace.llm_calls == 1


def test_explicit_numeric_prose_can_be_audited():
    text = "The tightening torque is 100 Nm."
    result = run(Scripted(draft(text, text), audit()), lambda k: [hit(text)])
    assert result.answer.supported


def test_provider_failure_is_sanitized_counted_and_not_retried():
    result = run(Scripted(RuntimeError("secret-key private-source")))
    assert result.answer.reason == "generation_error"
    assert result.trace.llm_calls == 1
    assert "secret" not in json.dumps(result.public_dict())
    assert "private-source" not in str(result.error)


def test_retrieval_failure_is_sanitized_without_calling_model():
    def broken(k):
        raise RuntimeError("private-path private-text")

    result = run(None, broken)
    assert result.answer.reason == "retrieval_error"
    assert result.trace.search_calls == 1
    assert "private" not in json.dumps(result.public_dict())


def test_telemetry_contains_no_question_passages_claims_or_filenames():
    result = run(Scripted(draft(), audit()))
    serialized = json.dumps(result.public_dict()["workflow"])
    assert not any(s in serialized for s in ["aurora", "tire", "S1", TEXT])
    assert result.trace.elapsed_ms >= sum(e.elapsed_ms for e in result.trace.events)


def test_top_twenty_has_no_redundant_expansion():
    result = run(Scripted({"supported": False, "claims": []}), top_k=20)
    assert result.trace.search_calls == 1
    assert not result.answer.supported


def test_limits_and_question_are_validated_before_any_operation():
    from garage_rag.workflow import WorkflowLimits, run_workflow

    with pytest.raises(ValueError):
        WorkflowLimits(max_llm_calls=5)
    with pytest.raises(ValueError):
        WorkflowLimits(max_searches=3)
    with pytest.raises(ValueError):
        run_workflow(" ", None, None)


def test_spelled_out_number_cannot_bypass_flattened_table_guard():
    text = "Pressure table\nFront Rear\nA1 2.2 2.8\nA2 2.5 3.0"
    value = draft("The tire pressure is two point two bar.", text)
    result = run(Scripted(value, audit()), lambda k: [hit(text)])
    assert result.answer.reason == "unsafe_numeric_evidence"
    assert result.trace.llm_calls == 1


def test_orphaned_numeric_cells_cannot_be_interpreted_as_prose():
    text = "The tire pressure is shown below.\nFront Rear\n2.2\n2.8\n2.5\n3.0"
    value = draft("The tire pressure is 2.2 bar.", text)
    result = run(Scripted(value, audit()), lambda k: [hit(text)])
    assert result.answer.reason == "unsafe_numeric_evidence"


def test_truncated_audit_is_never_accepted_or_retried():
    llm = Scripted(draft(), audit())
    invoke = llm.invoke

    def truncated(messages):
        response = invoke(messages)
        if "evidence auditor" in messages[0].content:
            response.response_metadata = {"finish_reason": "length"}
        return response

    llm.invoke = truncated
    result = run(llm)
    assert not result.answer.supported
    assert result.answer.reason == "output_token_limit"
    assert result.trace.llm_calls == 2


def test_audit_checks_every_claim_exactly_once():
    value = draft()
    value["claims"] *= 2
    result = run(Scripted(value, audit()))
    assert result.answer.reason == "invalid_audit_output"


def test_search_cap_prevents_second_retrieval():
    from garage_rag.workflow import WorkflowLimits

    result = run(
        Scripted({"supported": False, "claims": []}), limits=WorkflowLimits(max_searches=1)
    )
    assert result.answer.reason == "insufficient_evidence"
    assert result.trace.search_calls == 1


def test_exhausted_graph_recursion_is_a_safe_terminal_failure(monkeypatch):
    from langgraph.errors import GraphRecursionError
    from langgraph.graph.state import CompiledStateGraph

    def fail(*args, **kwargs):
        raise GraphRecursionError("secret private graph state")

    monkeypatch.setattr(CompiledStateGraph, "invoke", fail)
    result = run(None)
    assert result.answer.reason == "step_limit"
    assert result.trace.llm_calls == 0
    assert "secret" not in str(result.error)


def test_whole_graph_disables_inherited_langsmith_tracing(monkeypatch):
    from langsmith.run_helpers import get_tracing_context

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    llm = Scripted(draft(), audit())
    invoke = llm.invoke

    def checked(messages):
        assert get_tracing_context()["enabled"] is False
        return invoke(messages)

    llm.invoke = checked
    assert run(llm).answer.supported


def test_groq_success_wire_has_two_fixed_model_requests_without_hidden_calls(monkeypatch):
    import httpx

    from garage_rag.generation import make_llm

    requests = []
    responses = [draft(), audit()]

    def transport(request):
        requests.append(json.loads(request.content))
        assert request.extensions["timeout"]["read"] == 60
        return httpx.Response(
            200,
            json={
                "id": "synthetic-response",
                "object": "chat.completion",
                "system_fingerprint": None,
                "model": "openai/gpt-oss-120b",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(responses[len(requests) - 1]),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    constructor = httpx.Client
    monkeypatch.setenv("GROQ_FREE_PLAN_CONFIRMED", "true")
    monkeypatch.setenv("GROQ_MAX_TOKENS", "512")
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: constructor(
            transport=httpx.MockTransport(transport), **kwargs
        )
    )
    llm = make_llm(api_key="test-not-a-real-key")
    with llm.client:
        result = run(llm)
    assert result.answer.supported
    assert result.trace.llm_calls == len(requests) == 2
    assert result.trace.input_tokens == 20
    assert result.trace.output_tokens == 10
    assert result.trace.public_dict()["token_usage_complete"]
    for body in requests:
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["service_tier"] == "on_demand"
        assert "models" not in body


def test_provider_failure_keeps_safe_status_in_trace_without_payload(monkeypatch):
    import httpx

    from garage_rag.generation import make_llm

    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(
            502, json={"error": {"code": 502, "message": "private provider payload"}}
        )

    constructor = httpx.Client
    monkeypatch.setenv("GROQ_FREE_PLAN_CONFIRMED", "true")
    monkeypatch.setenv("GROQ_MAX_TOKENS", "512")
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: constructor(
            transport=httpx.MockTransport(transport), **kwargs
        )
    )
    llm = make_llm(api_key="test-not-a-real-key")
    with llm.client:
        result = run(llm)
    assert result.error
    assert len(requests) == result.trace.llm_calls == 1
    assert result.trace.public_dict()["provider_errors"] == [
        {"http_status": 502, "category": "http_error"}
    ]
    assert not result.trace.public_dict()["token_usage_complete"]
    assert "private provider payload" not in json.dumps(result.public_dict())


@pytest.mark.parametrize(
    "claim",
    [
        "The tightening torque is 100Nm.",
        "The tightening torque is one hundred newton metres.",
        "The tightening torque is twenty newton metres.",
        "Das Anzugsdrehmoment beträgt zwanzig Newtonmeter.",
        "Момент затяжки составляет двадцать ньютон-метров.",
    ],
)
def test_alternative_number_spelling_cannot_bypass_numeric_guard(claim):
    text = "The tightening torque is 90 Nm."
    from garage_rag.language import detect_language

    result = run(
        Scripted(draft(claim, text), audit()),
        lambda k: [hit(text)],
        answer_language=detect_language(claim),
    )
    assert result.answer.reason == "unsafe_numeric_evidence"
    assert result.trace.llm_calls == 1


def test_expansion_retains_previously_retrieved_safety_constraints():
    warning = "Do not open the coolant cap while the engine is hot."
    novel = [
        hit(f"The coolant cap is on top of the expansion tank. Item {i}.", i + 2) for i in range(8)
    ]
    llm = Scripted(
        {"supported": False, "claims": []},
        draft(
            "Open the coolant cap when the engine is hot.",
            "The coolant cap is on top of the expansion tank.",
            source="S2",
        ),
        audit("unsupported", entailed=False),
    )
    result = run(llm, lambda k: [hit(warning)] if k == 5 else [hit(warning), *novel])
    assert not result.answer.supported
    final_prompt = json.loads(llm.calls[-1][1].content)
    assert any(warning in s["text"] for s in final_prompt["sources"])


@pytest.mark.parametrize(
    "text,claim",
    [
        (
            "Do not exceed a\nspeed of 20 mph (32 km/h) while\nusing the accessory.",
            "Do not exceed 20 mph (32 km/h) while using the accessory.",
        ),
        (
            "The clearance must remain\nat 8 in (20 cm) above\nthe equipment.",
            "The clearance must remain at 8 in (20 cm) above the equipment.",
        ),
    ],
)
def test_wrapped_dual_unit_prose_reaches_semantic_audit(text, claim):
    result = run(Scripted(draft(claim, " ".join(text.split())), audit()), lambda k: [hit(text)])
    assert result.answer.supported
    assert result.trace.llm_calls == 2


def test_nonnumeric_claim_is_audited_despite_unrelated_numeric_source_rows():
    text = "The release lever is below the dashboard.\nClosing height 8 in (20 cm)"
    result = run(
        Scripted(draft("The release lever is below the dashboard.", text.splitlines()[0]), audit()),
        lambda k: [hit(text)],
    )
    assert result.answer.supported
    assert result.trace.llm_calls == 2


def test_dual_unit_table_still_cannot_confirm_numeric_claim():
    text = "Accessory limits\nA1 20 mph (32 km/h)\nA2 40 mph (64 km/h)"
    result = run(Scripted(draft("The limit is 20 mph.", text), audit()), lambda k: [hit(text)])
    assert result.answer.reason == "unsafe_numeric_evidence"
    assert result.trace.llm_calls == 1


def test_dual_unit_prose_still_requires_conditions_and_supported_numbers():
    text = "Do not exceed a\nspeed of 20 mph (32 km/h) while\nusing the accessory."
    result = run(
        Scripted(
            draft("Always drive at 20 mph.", " ".join(text.split())),
            audit("unsupported", entailed=False),
        ),
        lambda k: [hit(text)],
    )
    assert result.answer.reason == "unsupported_claims"
    result = run(
        Scripted(draft("The limit is 25 mph.", " ".join(text.split())), audit()),
        lambda k: [hit(text)],
    )
    assert result.answer.reason == "unsafe_numeric_evidence"


def test_expansion_uses_budget_for_passages_preserving_applicability_and_warnings():
    from garage_rag.generation import build_context

    passages = []
    for i in range(8):
        value = hit(
            ("Do not open while hot. " if i == 0 else "Background information. ") + "x" * 1200,
            i + 1,
        )
        value.document.metadata.update(
            title="Synthetic owner manual 2040",
            language="en",
            vehicle_models=["Aurora X"],
            document_type="owner_manual",
            source_url="https://example.invalid/" + "a" * 220,
            page_label=str(i + 1),
        )
        passages.append(value)
    passages[-1].document.page_content += " The release lever is below the dashboard."
    assert len(build_context(passages, max_chars=20000).text) > 12000
    llm = Scripted(
        {"supported": False, "claims": []},
        draft(
            "The release lever is below the dashboard.",
            "The release lever is below the dashboard.",
            source="S8",
        ),
        audit(),
    )
    result = run(llm, lambda k: passages[:5] if k == 5 else passages)
    assert result.answer.supported
    assert len(result.context.text) <= 12000
    assert len(result.context.sources) == 8
    for original, entry in zip(passages, json.loads(result.context.text), strict=True):
        assert entry["text"] == original.document.page_content
        for field in ("title", "vehicle_models", "language", "document_type"):
            assert entry["metadata"][field] == original.document.metadata[field]
    assert "Do not open while hot." in llm.calls[-1][1].content


def test_dual_unit_axle_rows_with_prose_header_remain_unsafe():
    text = (
        "The maximum speed is shown below.\nfront axle 20 mph (32 km/h)\nrear axle 40 mph (64 km/h)"
    )
    result = run(
        Scripted(draft("The maximum speed is 20 mph.", text), audit()), lambda k: [hit(text)]
    )
    assert result.answer.reason == "unsafe_numeric_evidence"
    assert result.trace.llm_calls == 1
