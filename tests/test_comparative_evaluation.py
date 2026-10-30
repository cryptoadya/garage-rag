"""Phase 4B contracts: actual workflows with fictional, explicitly scripted replies."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from garage_rag.answer_evaluation import (
    BudgetedPacedLLM,
    apply_reviews,
    evaluate_synthetic,
    main,
    paired_summary,
    validate_live_budget,
)
from garage_rag.generation import GenerationError

MODES = ["direct", "grounded", "agentic"]
DATASET = Path("benchmarks/datasets/aurora-comparative.v1.json")


def dataset():
    return json.loads(DATASET.read_text())


def test_three_modes_run_complete_multilingual_matrix_and_keep_audits_out_of_truth():
    report, captures = evaluate_synthetic(dataset(), modes=MODES, seed=23)
    assert report["complete"]
    assert report["planned_runs"] == len(captures) == 63
    assert report["paired"]["pairs"] == 21
    assert len(report["pairwise"]) == 3
    assert set(report["by_language"]) == {"en", "de", "ru"}
    assert len(report["by_kind"]) == 7
    for mode, cap in zip(MODES, [1, 4, 6], strict=True):
        assert report["summary"][mode]["llm_calls"]["max"] <= cap
        assert not report["summary"][mode]["token_usage_complete"]
        assert report["summary"][mode]["independent_reviewed_answers"] == 0
    for mode in MODES:
        row = next(r for r in report["rows"] if r["id"] == "warning" and r["mode"] == mode)
        assert row["supported"] and row["claim_support"] == [False]
        assert row["warnings_preserved"] is False
        assert row["citation_support"] == [False]
        if mode != "direct":
            assert row["model_calls"][-1]["audit_decision"] == "accept"
    public = json.dumps(report)
    assert "Aurora jack" not in public and '"tool_calls":' not in public
    assert "Aurora jack" in json.dumps(captures)


def test_targeted_retrieval_and_compound_completeness_are_distinct():
    report, _ = evaluate_synthetic(dataset(), modes=MODES, languages=["ru"])
    rows = {(r["id"], r["mode"]): r for r in report["rows"]}
    assert rows["follow_reference", "grounded"]["supported"] is False
    assert rows["follow_reference", "agentic"]["complete_answer"] is True
    assert rows["compound", "direct"]["claim_support"] == [True]
    assert rows["compound", "direct"]["complete_answer"] is False
    assert rows["compound", "grounded"]["complete_answer"] is True
    assert rows["compound", "agentic"]["complete_answer"] is True
    assert rows["compound", "direct"]["warnings_preserved"] is False
    assert rows["simple", "agentic"]["trace"]["llm_calls"] > 1


def test_all_mode_matching_does_not_hide_missing_or_failed_agentic_rows():
    report, _ = evaluate_synthetic(dataset(), modes=MODES, languages=["en"], case_ids=["simple"])
    rows = report["rows"]
    assert paired_summary(rows, modes=MODES)["pairs"] == 1
    agent = next(r for r in rows if r["mode"] == "agentic")
    agent["error"] = True
    assert paired_summary(rows, modes=MODES)["pairs"] == 0
    assert paired_summary([r for r in rows if r is not agent], modes=MODES)["pairs"] == 0


def test_private_review_packet_preserves_native_pairs_and_binds_tool_arguments(tmp_path):
    from garage_rag.review_packet import write_packet

    report, captures = evaluate_synthetic(
        dataset(), modes=MODES, languages=["en"], case_ids=["follow_reference"]
    )
    write_packet(report, captures, tmp_path / "valid")
    template = json.loads((tmp_path / "valid/claim-review.template.json").read_text())
    assert all("citation_support" in r and "warnings_preserved" in r for r in template["rows"])
    capture = next(c for c in captures if c["key"]["mode"] == "agentic")
    writer = next(
        c
        for c in capture["model_calls"]
        if c["messages"][-1]["role"] == "tool"
        and c["messages"][0]["content"].startswith("You are GarageRAG")
    )
    writer["messages"][-2]["tool_calls"][0]["args"]["query"] = "tampered"
    with pytest.raises(ValueError, match="request"):
        write_packet(report, captures, tmp_path / "invalid")


def test_live_outputs_remain_unscored_until_atomic_hash_bound_external_review():
    from garage_rag.answer_evaluation import ScriptedScenarioLLM

    data = dataset()
    model = ScriptedScenarioLLM(data["scenarios"][0], "en")
    report, _ = evaluate_synthetic(
        data, modes=MODES, languages=["en"], case_ids=["simple"], llm=model
    )
    assert all(
        r["claim_support"] is None
        and r["citation_support"] is None
        and r["warnings_preserved"] is None
        for r in report["rows"]
    )
    reviews = {
        "report_dataset_sha256": report["dataset_sha256"],
        "reviewer": "External reviewer",
        "independent_human_review": True,
        "rows": [
            {
                **{k: r[k] for k in ("id", "language", "mode", "trial", "answer_sha256")},
                "claim_support": [True] * r["returned_claims"],
                "complete_answer": True,
                "citation_support": [True] * r["returned_claims"],
                "warnings_preserved": None,
            }
            for r in report["rows"]
        ],
    }
    before = deepcopy(report)
    reviews["rows"][-1]["citation_support"] = []
    with pytest.raises(ValueError, match="review"):
        apply_reviews(report, reviews)
    assert report == before
    reviews["rows"][-1]["citation_support"] = [True]
    apply_reviews(report, reviews)
    assert report["paired"]["pairs"] == 1
    assert all(s["independent_reviewed_answers"] == 1 for s in report["summary"].values())


def test_source_sufficiency_matches_all_three_modes_without_using_audits():
    from garage_rag.source_evaluation import apply_source_reviews

    report, _ = evaluate_synthetic(dataset(), languages=["en"], case_ids=["simple"])
    reviews = {
        "schema_version": 1,
        "report_dataset_sha256": report["dataset_sha256"],
        "reviewer": "Fixture reviewer",
        "independent_human_review": False,
        "rows": [
            {
                **{k: r[k] for k in ("id", "language", "mode", "trial", "answer_sha256")},
                "contexts": [
                    {"sources_sha256": c["sources_sha256"], "sufficiency": "sufficient"}
                    for c in r["sent_contexts"]
                ],
            }
            for r in report["rows"]
        ],
    }
    apply_source_reviews(report, reviews)
    paired = report["source_sufficiency"]["paired"]
    assert paired["pairs"] == 1 and set(paired["summary"]) == set(MODES)
    assert all(s["sufficient_context_runs"] == 1 for s in paired["summary"].values())
    assert report["source_sufficiency"]["independent_human_review"] is False


def test_failed_native_attempt_consumes_shared_call_and_token_budget(monkeypatch):
    from garage_rag.local_tools import TOOLS

    class Model:
        def bind_tools(self, tools, **kwargs):
            return self

        def request_reservation(self, messages, tools=None):
            return 600

        def invoke(self, messages):
            raise GenerationError("Provider failed; PRIVATE payload")

    model = BudgetedPacedLLM(Model(), max_api_calls=1, max_total_tokens=1000)
    bound = model.bind_tools(TOOLS)
    with pytest.raises(GenerationError, match="Provider failed"):
        bound.invoke([])
    assert model.calls == 1 and model.reserved_tokens == 600
    with pytest.raises(GenerationError, match="API budget exhausted"):
        model.invoke([])
    assert model.calls == 1 and model.reserved_tokens == 600


def test_tool_binding_cannot_bypass_session_calls_tokens_pacing_or_observation(monkeypatch):
    import garage_rag.answer_evaluation as evaluation
    from garage_rag.local_tools import TOOLS
    from garage_rag.source_evaluation import ObservedEvaluationLLM

    clock = [0.0]
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(evaluation.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    starts = []

    class Model:
        provider = "groq"
        max_tokens = 512

        def request_reservation(self, messages, tools=None):
            return 1000 if tools else 600

        def bind_tools(self, tools, **kwargs):
            return SimpleNamespace(invoke=self.invoke)

        def invoke(self, messages):
            starts.append(clock[0])
            return AIMessage(content="FINALIZE")

    budget = BudgetedPacedLLM(Model(), max_api_calls=3, max_total_tokens=1600)
    observed = ObservedEvaluationLLM(budget)
    bound = observed.bind_tools(TOOLS, tool_choice="auto", parallel_tool_calls=False)
    messages = [HumanMessage(content='{"question":"private"}')]
    bound.invoke(messages)
    observed.invoke(messages)
    with pytest.raises(GenerationError, match="token budget"):
        bound.invoke(messages)
    assert budget.calls == 2 and budget.reserved_tokens == 1600
    assert starts[1] - starts[0] >= 4
    assert len(observed.public_calls) == 3  # Includes locally blocked attempt.
    assert "private" not in json.dumps(observed.public_calls)


def test_three_mode_budget_and_offline_cli_never_construct_remote_client(tmp_path, monkeypatch):
    validate_live_budget(pairs=2, max_api_calls=22, modes=MODES)
    with pytest.raises(ValueError, match="22"):
        validate_live_budget(pairs=2, max_api_calls=21, modes=MODES)
    monkeypatch.setattr(
        "garage_rag.generation.make_llm", lambda **kw: pytest.fail("No API client permitted")
    )
    output = tmp_path / "report.json"
    assert main(["--dataset", str(DATASET), "--modes", *MODES, "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["complete"] and report["provenance"]["generation"] is None
    assert (
        main(
            [
                "--dataset",
                str(DATASET),
                "--modes",
                *MODES,
                "--live",
                "--provider",
                "groq",
                "--max-api-calls",
                "231",
                "--output",
                str(tmp_path / "rejected.json"),
            ]
        )
        == 2
    )  # Token cap required.


@pytest.mark.parametrize("modes", [["agentic", "agentic"], ["unknown"], []])
def test_invalid_modes_reject_before_execution(modes):
    with pytest.raises(ValueError, match="mode"):
        evaluate_synthetic(dataset(), modes=modes)
