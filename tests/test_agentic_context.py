"""Bounded evidence selection: real store/selector/graph, no remote models."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_agentic import Model, control, reply, tool
from test_workflow import TEXT, audit, draft, hit

from garage_rag.agentic import AgenticLimits, run_agentic


def store_for(*batches):
    from garage_rag.agentic_context import EvidenceStore

    store = EvidenceStore()
    for i, batch in enumerate(batches):
        store.add(batch, method="dense", query=f"tire pressure search {i}", top_k=5)
    return store


def select(store, **kwargs):
    from garage_rag.agentic_context import select_context

    return select_context(store, "How should I check tire pressure?", **kwargs)


def test_overlapping_searches_preserve_scores_origins_and_original_metadata():
    first = hit()
    first.document.metadata.update(source_url="https://example.org/manual", page_label="i")
    store = store_for([first], [replace(first, score=0.95), hit("Tire pressure guidance.", 8)])
    # The store owns snapshots, even if retrieval reuses mutable Documents.
    first.document.page_content = "mutated"
    first.document.metadata["pdf_page"] = 999
    selection = select(store)
    assert selection.reason is None
    assert list(selection.context.sources) == ["S1", "S2"]
    assert selection.context.sources["S1"].page_content == TEXT
    assert selection.context.sources["S1"].metadata["page_label"] == "i"
    capture = store.capture()
    assert [s["hits"][0]["score"] for s in capture["searches"]] == [0.8, 0.95]
    assert [s["hits"][0]["source_id"] for s in capture["searches"]] == ["S1", "S1"]
    assert capture["searches"][0]["hits"][0]["metadata"]["pdf_page"] == 1
    assert selection.diagnostics["found"] == 3
    assert selection.diagnostics["stored"] == 2
    assert selection.diagnostics["duplicates"] == 1


def test_irrelevant_overflow_does_not_displace_second_search_evidence_or_renumber_ids():
    unrelated = hit("Radio presets are stored in memory. " * 500, 30)
    important = hit("Check tire pressure before driving with cold tires.", 8)
    store = store_for([hit(), unrelated], [hit(), important])
    selection = select(store)
    assert selection.reason is None
    assert list(selection.context.sources) == ["S1", "S3"]
    assert selection.context.sources["S3"].page_content == important.document.page_content
    assert len(selection.context.text) <= 12000
    assert selection.diagnostics["excluded"] == [
        {"source_id": "S2", "reason": "low_relevance_budget"}
    ]
    assert "Radio presets" in json.dumps(store.capture())


@pytest.mark.parametrize(
    "text,metadata",
    [
        ("Warning: keep clear of moving parts. " * 500, {}),
        ("Only for vehicles equipped with the blue module. " * 400, {}),
        ("Never open the enclosure while it is hot. " * 400, {}),
        ("Do not check tire pressure cold. " * 500, {}),
        ("Achtung: nur bei stehendem Motor. " * 500, {}),
        ("Предупреждение: только при холодном двигателе. " * 400, {}),
        ("Unrelated prose. " * 1000, {"requires_layout": True}),
        ("W" * 13000, {}),
    ],
    ids=["warning", "equipment", "prohibition", "conflict", "de", "ru", "layout", "unreadable"],
)
def test_potential_warning_condition_conflict_or_unreadable_evidence_cannot_be_dropped(
    text, metadata
):
    guarded = hit(text, 40)
    guarded.document.metadata.update(metadata)
    selection = select(store_for([hit(), guarded]))
    assert selection.reason == "context_budget_exhausted"
    assert selection.context.sources == {}
    assert selection.diagnostics["blocked"]
    assert "S2" in selection.diagnostics["blocked"]


def test_continuation_and_semantic_top_hit_are_retained_without_shared_words():
    store = store_for(
        [hit(), hit("The blue enclosure contains the required item. " * 400, 1)],
    )
    assert select(store).reason == "context_budget_exhausted"
    # A semantic top hit is not discarded just because lexical matching fails.
    store = store_for([hit("The blue enclosure contains the required item. " * 400, 30)])
    assert select(store).reason == "context_budget_exhausted"


def test_conflicting_applicability_and_revisions_keep_distinct_sources_and_exact_text():
    source = hit("Tire pressure is checked cold.\nOnly for the sedan.")
    source.document.metadata.update(vehicle_models=["BMW E39"], content_sha256="revision-a")
    other = deepcopy(source)
    other.document.metadata.update(vehicle_models=["other vehicle"], content_sha256="revision-b")
    other.document.page_content = "Tire pressure is checked warm.\nOnly for the touring."
    selection = select(store_for([source], [other]))
    assert selection.reason is None
    entries = json.loads(selection.context.text)
    assert [e["text"] for e in entries] == [
        source.document.page_content,
        other.document.page_content,
    ]
    assert entries[0]["metadata"]["vehicle_models"] == ["BMW E39"]
    assert entries[1]["metadata"]["vehicle_models"] == ["other vehicle"]
    # Same page/text with different applicability is not an exact duplicate.
    other.document.page_content = source.document.page_content
    assert select(store_for([source, other])).diagnostics["stored"] == 2


def test_chunk_ceiling_cannot_hide_relevant_evidence_and_selection_is_deterministic():
    store = store_for([hit(), hit("Tire pressure supplementary instructions.", 8)])
    a, b = select(store, max_chunks=1), select(store, max_chunks=1)
    assert a.reason == b.reason == "context_budget_exhausted"
    assert a.diagnostics == b.diagnostics


def test_unmatched_multilingual_overflow_stops_when_relevance_cannot_be_assessed():
    from garage_rag.agentic_context import EvidenceStore, select_context

    store = EvidenceStore()
    store.add(
        [hit("The blue enclosure holds the item."), hit("Radio memory presets. " * 900, 30)],
        method="dense",
        query="давление шин",
        top_k=2,
    )
    selection = select_context(store, "Как проверить шины?")
    assert selection.reason == "context_relevance_uncertain"
    assert selection.context.sources == {}
    assert {x["reason"] for x in selection.diagnostics["excluded"]} == {"unassessed_relevance"}


def test_evicted_evidence_reenters_with_original_id_when_new_query_makes_it_material():
    from garage_rag.agentic_context import EvidenceStore

    store = EvidenceStore()
    store.add(
        [hit(), hit("Radio presets are stored in memory. " * 15, 30)],
        method="dense",
        query="tire pressure",
        top_k=2,
    )
    store.add(
        [hit("Tire pressure guidelines. " * 15, 8)],
        method="bm25",
        query="tire pressure guidelines",
        top_k=1,
    )
    # Force a narrower budget: all relevant evidence fits, optional S2 does not.
    selection = select(store, max_chars=1200)
    assert selection.reason is None and list(selection.context.sources) == ["S1", "S3"]
    store.add(
        [hit("Radio presets are stored in memory. " * 15, 30)],
        method="bm25",
        query="radio memory",
        top_k=1,
    )
    selection = select(store)
    assert selection.reason is None and list(selection.context.sources) == ["S1", "S2", "S3"]
    assert selection.diagnostics["stored"] == 3


@pytest.mark.parametrize(
    "sentence",
    [
        "Keep hands clear of moving components.",
        "Disconnect power before opening the cover.",
        "Applies to vehicles fitted with the special module.",
    ],
)
def test_unlinked_safety_imperatives_and_applicability_are_protected(sentence):
    selection = select(store_for([hit(), hit((sentence + " ") * 500, 30)]))
    assert selection.reason == "context_budget_exhausted"


def test_graph_reselects_after_second_search_captures_raw_hits_and_bounds_planner_history():
    batches = iter(
        [
            [hit(), hit("Radio presets are stored in memory. " * 500, 30)],
            [hit(), hit("Tire pressure instructions apply to cold tires.", 8)],
        ]
    )
    model = Model(
        tool(),
        tool(query="tire pressure instructions", call_id="c2"),
        control(),
        reply(draft(source="S3", quote="Tire pressure instructions apply to cold tires.")),
        reply(audit()),
    )
    result = run_agentic(
        "How should I check tire pressure?",
        lambda *a: next(batches),
        model,
        answer_language="en",
        limits=AgenticLimits(max_total_tokens=48000),
    )
    assert result.answer.supported
    assert "PDF page 8" in result.answer.text
    assert len(model.messages[2]) == 4  # system, question, latest call/result
    assert len([m for m in model.messages[2] if m.type == "tool"]) == 1
    assert model.messages[2][-1].tool_call_id == "c2"
    payload = json.loads(model.messages[2][-1].content)
    assert [s["source_id"] for s in payload["sources"]] == ["S1", "S3"]
    assert len(payload["search_history"]) == 2
    assert [d["phase"] for d in result.trace.context_deliveries] == [
        "planner",
        "planner",
        "planner",
        "writer",
        "auditor",
    ]
    assert result.trace.context_deliveries[-2]["source_ids"] == ["S1", "S3"]
    public = json.dumps(result.public_dict())
    assert "Radio presets" not in public and "tire pressure instructions" not in public
    assert "Radio presets" in json.dumps(result.evidence_store.capture())


def test_dropped_source_id_cannot_be_cited_and_failed_selection_is_captured():
    model = Model(tool(), reply(draft(source="S2")))
    result = run_agentic(
        "Tire pressure?",
        lambda *a: [hit(), hit("Radio presets are stored in memory. " * 500, 30)],
        model,
        answer_language="en",
        limits=AgenticLimits(max_llm_calls=3, max_searches=1),
    )
    assert result.answer.reason == "invalid_model_output"
    result = run_agentic(
        "Tire pressure?",
        lambda *a: [hit(), hit("Warning: hot parts. " * 900, 30)],
        Model(tool()),
        answer_language="en",
    )
    assert result.answer.reason == "context_budget_exhausted"
    assert result.trace.llm_calls == 1
    assert "Warning: hot parts" in json.dumps(result.evidence_store.capture())
    assert result.trace.context_selections[-1]["blocked"] == ["S2"]


def test_multibyte_request_budget_stops_before_audit_without_refunding():
    from test_agentic import run

    passage = hit(TEXT + " Шины холодные." * 100)
    baseline = run(Model(tool(), reply(draft()), reply(audit())), lambda *a: [passage])
    assert baseline.answer.supported
    cap = baseline.trace.reserved_tokens - 1
    result = run(
        Model(tool(), reply(draft()), reply(audit())),
        lambda *a: [passage],
        limits=AgenticLimits(max_llm_calls=3, max_searches=1, max_total_tokens=cap),
    )
    assert result.answer.reason == "token_budget_exhausted"
    assert result.trace.reserved_tokens <= cap
    assert result.trace.llm_calls == 2


def test_evaluation_captures_include_store_without_publishing_passages():
    from pathlib import Path

    from garage_rag.answer_evaluation import evaluate_synthetic
    from garage_rag.config import digest

    dataset = json.loads(Path("benchmarks/aurora-comparative.v1.json").read_text())
    report, captures = evaluate_synthetic(
        dataset, modes=["agentic"], languages=["en"], case_ids=["simple"]
    )
    assert captures[0]["evidence_store"]["searches"][0]["hits"]
    assert report["rows"][0]["trace"]["context_selections"][-1]["evidence_sha256"] == digest(
        captures[0]["evidence_store"]
    )
    assert "evidence_store" not in report["rows"][0]
    assert "Aurora jack" not in json.dumps(report)


def test_private_bmw_e39_offline_hits_remain_available_without_mistaking_pointers_for_evidence():
    from pathlib import Path

    from langchain_core.documents import Document

    from garage_rag.agentic_context import EvidenceStore, select_context
    from garage_rag.index import SearchHit

    path = Path("output/private/agentic-context-offline-20261027/retrieval-capture.json")
    if not path.exists():
        pytest.skip("Private fresh offline BMW retrieval is not distributed")
    capture = json.loads(path.read_text())
    store = EvidenceStore()
    for search in capture["evidence_store"]["searches"]:
        store.add(
            [
                SearchHit(Document(page_content=h["text"], metadata=h["metadata"]), h["score"])
                for h in search["hits"]
            ],
            **{k: search[k] for k in ("method", "query", "top_k")},
        )
    selection = select_context(
        store, "Where is the onboard tool kit located and how do I access it?"
    )
    assert selection.reason is None
    assert selection.diagnostics["found"] == 10
    assert selection.diagnostics["stored"] == 9
    assert not selection.diagnostics["blocked"]
    assert not selection.diagnostics["has_substantive_evidence"]
    assert selection.diagnostics["roles"]["S1"] == "navigation"
    assert selection.diagnostics["roles"]["S9"] == "navigation"
    assert sum(len(s["hits"]) for s in store.capture()["searches"]) == 10
