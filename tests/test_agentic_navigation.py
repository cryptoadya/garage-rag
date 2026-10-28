"""Planner-only navigation leads; real graph/selector with controlled model replies."""

import json

import pytest
from conftest import Model, control, reply, tool
from test_workflow import TEXT, audit, draft, hit

from garage_rag.agentic import run_agentic

INDEX = "Reserve handle 42\nHood latch 51\nBattery care 63\nFuse box 77"


class PlannerModel(Model):
    def __init__(self, structured):
        super().__init__(
            reply(
                dict(action="search", method="dense", query="reserve handle", top_k=5, reason=None)
            )
            if structured
            else tool(query="reserve handle", top_k=5),
            reply(
                dict(
                    action="refuse",
                    method=None,
                    query=None,
                    top_k=None,
                    reason="insufficient_evidence",
                )
            )
            if structured
            else control("REFUSE insufficient_evidence"),
        )
        self.structured_planner_supported = structured

    def bind_planner(self):
        return self


def planner_payload(hits, *, structured=True, question="Where is the reserve handle?"):
    model = PlannerModel(structured)
    result = run_agentic(question, lambda *a: hits, model, answer_language="en")
    assert result.answer.reason == "insufficient_evidence"
    return json.loads(model.messages[1][-1].content), result


@pytest.mark.parametrize("structured", [False, True])
def test_planner_receives_topic_lead_without_guessing_physical_page(structured):
    source = hit(INDEX, 3)
    source.document.metadata.update(content_sha256="revision-a", page_label="iii")
    payload, result = planner_payload([source], structured=structured)
    assert payload.get("navigation_hints") == [
        {"source_id": "S1", "heading": "Reserve handle", "target_label": "42"}
    ]
    assert payload["sources"][0]["metadata"]["pdf_page"] == 3
    assert payload["sources"][0]["text"] == INDEX
    assert not payload["selection"]["has_substantive_evidence"]
    assert result.context.sources["S1"].metadata["content_sha256"] == "revision-a"
    assert result.trace.llm_calls == 2 and result.trace.search_calls == 1


def test_many_matching_references_cannot_flood_planner_or_rewrite_source():
    text = "\n".join(f"Reserve handle component {n}" for n in range(10, 50))
    payload, result = planner_payload([hit(text, 3)])
    hints = payload.get("navigation_hints", [])
    assert len(hints) == 3
    assert len(json.dumps(hints)) < 400
    assert payload["sources"][0]["text"] == text
    assert result.context.sources["S1"].page_content == text


@pytest.mark.parametrize(
    "text",
    [
        "Reserve handle " + "extended " * 30 + "42\nHood latch 51\nBattery care 63\nFuse box 77",
        "Clock 42\nHood latch 51\nBattery care 63\nFuse box 77",
        INDEX + "\nNever open the reserve handle while the mechanism is moving.",
    ],
)
def test_unusable_or_unsafe_navigation_is_not_made_into_a_hint(text):
    payload, result = planner_payload([hit(text, 3)])
    assert not payload.get("navigation_hints")
    # Omitting a lead must not remove/truncate its full warning-bearing passage.
    if payload["sources"]:
        assert payload["sources"][0]["text"] == text
    assert result.evidence_store.capture()["searches"][0]["hits"][0]["text"] == text


def test_navigation_hints_do_not_leak_into_writer_or_auditor():
    navigation = hit("Tire pressure 42\nHood latch 51\nBattery care 63\nFuse box 77", 3)
    model = Model(tool(), control(), reply(draft(source="S2")), reply(audit()))
    result = run_agentic(
        "When should I check tire pressure?",
        lambda *a: [navigation, hit(TEXT, 42)],
        model,
        answer_language="en",
    )
    assert result.answer.supported and result.answer.reason == "audited_evidence"
    assert json.loads(model.messages[1][-1].content).get("navigation_hints") == [
        {"source_id": "S1", "heading": "Tire pressure", "target_label": "42"}
    ]
    assert "navigation_hints" not in json.loads(model.messages[2][-1].content)
    assert all("navigation_hints" not in m.content for m in model.messages[3])
    assert result.trace.llm_calls == 4 and result.trace.search_calls == 1


@pytest.mark.parametrize(
    "question,index,heading,label",
    [
        (
            "Wo ist das Bordwerkzeug?",
            "Bordwerkzeug 84\nSicherungen 92\nReifendruck 104\nMotorhaube 121",
            "Bordwerkzeug",
            "84",
        ),
        (
            "Где находится набор инструментов?",
            "Набор инструментов 24\nЗамок капота 38\nАккумулятор 52\nБлок предохранителей 96",
            "Набор инструментов",
            "24",
        ),
    ],
)
def test_heading_hints_are_not_bound_to_english_or_a_vehicle(question, index, heading, label):
    payload, _ = planner_payload([hit(index, 3)], question=question)
    assert payload["navigation_hints"] == [
        {"source_id": "S1", "heading": heading, "target_label": label}
    ]


def test_same_heading_in_different_revisions_keeps_both_source_bindings():
    sources = [hit(INDEX, 3), hit(INDEX, 3)]
    for source, revision in zip(sources, ["revision-a", "revision-b"], strict=True):
        source.document.metadata["content_sha256"] = revision
    payload, result = planner_payload(sources)
    assert [h["source_id"] for h in payload["navigation_hints"]] == ["S1", "S2"]
    assert result.context.sources["S1"].metadata["content_sha256"] == "revision-a"
    assert result.context.sources["S2"].metadata["content_sha256"] == "revision-b"


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("method", ["dense", "bm25"])
def test_navigation_does_not_force_or_rewrite_planners_next_search(structured, method):
    model = PlannerModel(structured)
    query = "reserve handle release"
    model.responses.insert(
        1,
        reply(dict(action="search", method=method, query=query, top_k=1, reason=None))
        if structured
        else tool(
            "dense_search" if method == "dense" else "lexical_search",
            query=query,
            top_k=1,
            call_id="c2",
        ),
    )
    searches = []

    def search(*args):
        searches.append(args)
        return [hit(INDEX, 3)]

    result = run_agentic("Where is the reserve handle?", search, model, answer_language="en")
    assert searches == [("dense", "reserve handle", 5), (method, query, 1)]
    assert not result.answer.supported
