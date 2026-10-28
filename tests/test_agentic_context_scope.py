"""Scope relevance and dependencies without page-specific or live fixtures."""

import json

import pytest
from conftest import Model, control, tool
from test_workflow import hit

from garage_rag.agentic import run_agentic
from garage_rag.agentic_context import EvidenceStore, select_context

QUESTION = "Where is the onboard tool kit and how do I access it?"
PROCEDURE = "The onboard tool kit is in the rear compartment. Turn the latch to access it."


def add(store, hits, query="onboard tool kit", method="dense"):
    store.add(hits, method=method, query=query, top_k=5)


def index_text(target="42"):
    return (
        f"Contents\nOnboard tool kit{target}\nRadio85\nLighting90\n"
        "Tire pressure95\nWarning lamps100\nWindows110"
    )


def test_generic_query_and_semantic_lead_do_not_pin_unrelated_conditional_prose():
    store = EvidenceStore()
    generic = hit("If you own the vehicle, it is not only yours before and after use. " * 240, 8)
    generic.document.metadata["vehicle_models"] = ["Example Alpha"]
    add(store, [generic], "Example Alpha vehicle manual")
    add(store, [hit(PROCEDURE, 42)])
    selection = select_context(store, QUESTION)
    assert selection.reason is None
    assert list(selection.context.sources) == ["S2"]
    assert selection.diagnostics["protection"]["S1"] == []
    assert selection.diagnostics["has_substantive_evidence"]


def test_weak_conditions_attach_to_same_procedure_not_all_adjacent_pages():
    store = EvidenceStore()
    add(
        store,
        [
            hit(PROCEDURE, 42),
            hit("Only if it is stationary; otherwise wait.", 42),
            hit("If the radio is on, turn it off before leaving. " * 300, 43),
        ],
    )
    selection = select_context(store, QUESTION)
    assert selection.reason is None
    assert list(selection.context.sources) == ["S1", "S2"]
    assert "same_procedure_page" in selection.diagnostics["protection"]["S2"]
    assert selection.diagnostics["protection"]["S3"] == []


def test_explicit_procedure_dependency_keeps_remote_condition_and_refuses_its_overflow():
    store = EvidenceStore()
    procedure = hit(PROCEDURE, 42)
    procedure.document.metadata["procedure_id"] = "rear-latch"
    condition = hit("If it is moving, wait until it stops. " * 400, 90)
    condition.document.metadata["applies_to"] = ["rear-latch"]
    add(store, [procedure, condition])
    selection = select_context(store, QUESTION)
    assert selection.reason == "context_budget_exhausted"
    assert "procedure_dependency" in selection.diagnostics["protection"]["S2"]


def test_explicitly_other_procedure_warning_is_excluded_but_unknown_scope_stays_protected():
    procedure = hit(PROCEDURE, 42)
    procedure.document.metadata["procedure_id"] = "rear-latch"
    warning = hit("Warning: disconnect the radio power before repair. " * 300, 90)
    warning.document.metadata["applies_to"] = ["radio-repair"]
    store = EvidenceStore()
    add(store, [procedure, warning])
    selection = select_context(store, QUESTION)
    assert selection.reason is None
    assert list(selection.context.sources) == ["S1"]
    warning.document.metadata.pop("applies_to")
    store = EvidenceStore()
    add(store, [procedure, warning])
    assert select_context(store, QUESTION).reason == "context_budget_exhausted"


def test_navigation_is_deduplicated_by_reference_and_never_counts_as_answer_evidence():
    store = EvidenceStore()
    add(store, [hit(index_text(), 5)])
    add(store, [hit(index_text() + "\nMirrors120", 190)])
    selection = select_context(store, QUESTION)
    assert selection.reason is None
    assert list(selection.context.sources) == ["S1"]
    assert not selection.diagnostics["has_substantive_evidence"]
    assert selection.diagnostics["roles"] == {"S1": "navigation", "S2": "navigation"}
    assert selection.diagnostics["excluded"] == [
        {"source_id": "S2", "reason": "redundant_navigation"}
    ]


def test_navigation_only_finalization_refuses_without_writer_or_auditor():
    model = Model(tool(query="onboard tool kit"), control())
    result = run_agentic(QUESTION, lambda *a: [hit(index_text(), 5)], model, answer_language="en")
    assert result.answer.reason == "insufficient_evidence"
    assert result.trace.llm_calls == 2
    assert len(result.trace.context_deliveries) == 2


def test_late_procedure_and_its_conditions_survive_navigation_and_unrelated_overflow():
    store = EvidenceStore()
    add(store, [hit(index_text(), 5), hit("If the radio is off, presets remain. " * 450, 80)])
    add(
        store,
        [hit(index_text(), 5), hit(PROCEDURE + " Only with the vehicle stationary.", 42)],
        method="bm25",
    )
    selection = select_context(store, QUESTION)
    assert selection.reason is None
    assert selection.diagnostics["found"] == 4
    assert selection.diagnostics["stored"] == 3
    assert "S3" in selection.context.sources
    assert "Only with the vehicle stationary." in selection.context.sources["S3"].page_content
    assert selection.diagnostics["has_substantive_evidence"]
    assert len(selection.context.text) <= 12000


def test_topic_conflicts_from_different_revisions_are_preserved_even_with_other_warnings():
    store = EvidenceStore()
    a = hit("The onboard tool kit is in the rear compartment. Only for the sedan.", 42)
    b = hit("The onboard tool kit is in the front compartment. Only for the sedan.", 9)
    b.document.metadata.update(document="supplement.pdf", content_sha256="new")
    add(store, [a, hit("If the radio is off, presets remain. " * 450, 80)])
    add(store, [b], method="bm25")
    selection = select_context(store, QUESTION)
    assert selection.reason is None
    assert list(selection.context.sources) == ["S1", "S3"]
    assert [e["text"] for e in json.loads(selection.context.text)] == [
        a.document.page_content,
        b.document.page_content,
    ]


def test_warning_beside_navigation_cannot_be_demoted_to_a_page_reference():
    store = EvidenceStore()
    add(store, [hit(index_text() + "\nWarning: do not proceed while it is moving. " * 400, 5)])
    assert select_context(store, QUESTION).reason == "context_budget_exhausted"


def test_number_inside_reference_title_is_not_a_warning_or_substantive_evidence():
    store = EvidenceStore()
    add(store, [hit(index_text() + "\nChapter 7 Warning130", 5)])
    selection = select_context(store, QUESTION)
    assert selection.diagnostics["roles"]["S1"] == "navigation"
    assert not selection.diagnostics["has_substantive_evidence"]


def test_imperative_warning_with_numeric_suffix_is_not_demoted_to_navigation():
    store = EvidenceStore()
    add(store, [hit(index_text() + "\nNever loosen the latch100" * 600, 5)])
    assert select_context(store, QUESTION).reason == "context_budget_exhausted"


@pytest.mark.parametrize(
    "text",
    [
        "Wenn das Radio nicht läuft, bleiben nur die Sender gespeichert. ",
        "Если радио не работает, сохраняются только станции. ",
    ],
)
def test_foreign_bare_condition_words_do_not_globally_pin_other_procedures(text):
    store = EvidenceStore()
    add(store, [hit(PROCEDURE, 42), hit(text * 400, 90)])
    selection = select_context(store, QUESTION)
    assert selection.reason is None
    assert list(selection.context.sources) == ["S1"]


@pytest.mark.parametrize(
    "text",
    [
        "The support must remain locked. ",
        "Nicht öffnen, wenn es heiß ist. ",
        "Не открывайте горячую крышку. ",
    ],
)
def test_explicit_unscoped_prohibitions_and_obligations_remain_protected(text):
    store = EvidenceStore()
    add(store, [hit(PROCEDURE, 42), hit(text * 500, 90)])
    assert select_context(store, QUESTION).reason == "context_budget_exhausted"


@pytest.mark.parametrize(
    "text",
    [
        "Only when the engine is cold. ",
        "Nur wenn der Motor kalt ist. ",
        "Только при холодном двигателе. ",
    ],
)
def test_unlinked_explicit_applicability_clause_with_unknown_scope_remains_protected(text):
    store = EvidenceStore()
    add(store, [hit(PROCEDURE, 42), hit(text * 600, 90)])
    assert select_context(store, QUESTION).reason == "context_budget_exhausted"


def test_multi_search_graph_retains_linked_condition_and_refuses_real_topic_conflict():
    from conftest import reply
    from test_workflow import audit, draft

    a = hit(PROCEDURE, 42)
    a.document.metadata["procedure_id"] = "rear-latch"
    condition = hit("Only if it is stationary; otherwise wait.", 95)
    condition.document.metadata["applies_to"] = ["rear-latch"]
    conflicting = hit("The onboard tool kit is in the front compartment.", 9)
    conflicting.document.metadata["document"] = "supplement.pdf"
    batches = iter([[a, condition], [conflicting]])
    model = Model(
        tool(query="onboard tool kit"),
        tool(query="tool kit compartment", call_id="c2"),
        control(),
        reply(draft(PROCEDURE, PROCEDURE)),
        reply(audit("conflicting")),
    )
    result = run_agentic(QUESTION, lambda *a: next(batches), model, answer_language="en")
    assert result.answer.reason == "conflicting_sources"
    assert not result.answer.supported
    assert list(result.context.sources) == ["S1", "S2", "S3"]
    assert result.context.sources["S2"].page_content == condition.document.page_content
    assert result.trace.context_deliveries[-1]["phase"] == "auditor"


def test_private_bmw_three_search_context_preserves_every_procedure_and_warning():
    from pathlib import Path

    import pytest
    from langchain_core.documents import Document

    from garage_rag.index import SearchHit

    path = Path("output/private/agentic-context-scoped-20261027/selection-after.json")
    if not path.exists():
        pytest.skip("Private additional local BMW search is not distributed")
    capture = json.loads(path.read_text())
    store = EvidenceStore()
    for batch in capture["evidence_store"]["searches"]:
        store.add(
            [
                SearchHit(Document(page_content=h["text"], metadata=h["metadata"]), h["score"])
                for h in batch["hits"]
            ],
            **{k: batch[k] for k in ("method", "query", "top_k")},
        )
    selection = select_context(store, QUESTION)
    assert selection.reason is None and len(selection.context.text) <= 12000
    assert selection.diagnostics["found"] == 15 and selection.diagnostics["stored"] == 13
    assert selection.diagnostics["has_substantive_evidence"]
    for sid, reasons in selection.diagnostics["protection"].items():
        if reasons:
            assert (
                selection.context.sources[sid].page_content
                == store.entries[sid].document.page_content
            )
