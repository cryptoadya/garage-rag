"""Full private evidence, compact model diagnostics, unchanged request budgets."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from test_agentic import Model, control, tool
from test_workflow import hit

from garage_rag.agentic import run_agentic
from garage_rag.agentic_context import EvidenceStore, select_context


def test_internal_index_metadata_cannot_displace_intact_applicable_passages():
    store = EvidenceStore()
    hits = [hit("Check tire pressure cold.", p) for p in range(1, 5)]
    for h in hits:
        h.document.metadata.update(
            document_id="a" * 64,
            content_sha256="b" * 64,
            char_start=0,
            char_end=24,
            tokens=10,
            market="US",
            conditions="cold",
        )
    store.add(hits, method="dense", query="tire pressure", top_k=4)
    selection = select_context(store, "Tire pressure?", max_chars=800)
    assert selection.reason is None
    assert len(selection.context.sources) == 4
    assert len(selection.context.text) <= 800
    for entry in json.loads(selection.context.text):
        assert entry["metadata"]["market"] == "US"
        assert entry["metadata"]["conditions"] == "cold"
        assert entry["text"] == "Check tire pressure cold."
        assert selection.context.sources[entry["source_id"]].metadata["content_sha256"] == "b" * 64
    assert store.capture()["searches"][0]["hits"][0]["metadata"]["document_id"] == "a" * 64


def test_full_diagnostics_stay_in_trace_without_inflating_each_model_turn():
    model = Model(tool(), control("REFUSE insufficient_evidence"))
    result = run_agentic(
        "Tire pressure?",
        lambda *a: [hit(page=p) for p in range(1, 4)],
        model,
        answer_language="en",
    )
    payload = json.loads(model.messages[-1][-1].content)
    assert len(json.dumps(payload["selection"])) < 700
    assert len(payload["sources"]) == 3
    assert payload["selection"]["has_substantive_evidence"]
    assert result.trace.context_selections[-1]["protection"]
    assert result.trace.context_selections[-1]["evidence_sha256"]


def test_private_default_route_refuses_without_unaudited_answer_when_audit_cannot_fit():
    from transformers import AutoTokenizer

    from garage_rag.groq import MODEL, TOKENIZER_REVISION, GroqLLM
    from garage_rag.index import SearchHit

    root = Path("output/private/agentic-context-scoped-20261027")
    if not (root / "workflow-after-three.json").exists():
        pytest.skip("Private controlled BMW graph capture is not distributed")
    capture = json.loads((root / "workflow-after-three.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=TOKENIZER_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    replies = [call["response"] for call in capture["model_calls"]]
    # Restore the normal fourth planner (finalize) turn; no lowered search ceiling.
    replies.insert(
        3, json.dumps(dict(action="finalize", method=None, query=None, top_k=None, reason=None))
    )

    class Fixture:
        provider = "groq"
        structured_planner_supported = True
        max_tokens = 1024
        request_reservation = GroqLLM.request_reservation
        reserve_tokens = GroqLLM.reserve_tokens

        def __init__(self):
            self.tokenizer = tokenizer
            self.replies = iter(replies)

        def bind_planner(self):
            return self

        def invoke(self, messages):
            return AIMessage(
                content=next(self.replies), response_metadata={"finish_reason": "stop"}
            )

    batches = iter(capture["evidence_store"]["searches"])

    def search(method, query, top_k):
        from langchain_core.documents import Document

        batch = next(batches)
        assert (method, query, top_k) == (batch["method"], batch["query"], batch["top_k"])
        return [
            SearchHit(Document(page_content=h["text"], metadata=h["metadata"]), h["score"])
            for h in batch["hits"]
        ]

    result = run_agentic(
        "Where is the onboard tool kit located and how do I access it?",
        search,
        Fixture(),
        answer_language="en",
    )
    assert result.answer.reason == "token_budget_exhausted"
    assert not result.answer.supported
    assert result.candidate is not None  # Writer ran; mandatory audit was not bypassed.
    assert result.trace.llm_calls == 5
    assert result.trace.search_calls == 3
    assert result.trace.reserved_tokens <= result.trace.token_budget == 24000
    assert [d["phase"] for d in result.trace.context_deliveries] == [
        "planner",
        "planner",
        "planner",
        "planner",
        "writer",
    ]
