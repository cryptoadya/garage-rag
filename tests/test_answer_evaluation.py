import json
from pathlib import Path

import pytest


def dataset():
    return json.loads(Path("benchmarks/aurora-grounded-answering.v1.json").read_text())


def test_scripted_paired_evaluation_distinguishes_refusals_and_grounding():
    from garage_rag.answer_evaluation import evaluate_synthetic

    report, captures = evaluate_synthetic(
        dataset(), languages=["en", "de", "ru"], trials=1, seed=17
    )
    assert report["complete"]
    assert len(report["rows"]) == 84
    assert len(captures) == 84
    by_mode = report["summary"]
    assert by_mode["grounded"]["unsupported_answers"] < by_mode["direct"]["unsupported_answers"]
    assert by_mode["grounded"]["false_refusals"] > 0
    assert by_mode["grounded"]["llm_calls"]["max"] <= 4
    assert by_mode["direct"]["llm_calls"]["max"] <= 1
    serialized = json.dumps(report)
    assert "Check tire pressure" not in serialized
    assert "blue box" not in serialized
    assert "claims" not in report["rows"][0]  # No answer text in public reports.


def test_live_budget_preflight_prevents_unplanned_api_calls():
    from garage_rag.answer_evaluation import validate_live_budget

    with pytest.raises(ValueError, match="budget"):
        validate_live_budget(pairs=2, max_api_calls=9)
    validate_live_budget(pairs=2, max_api_calls=10)


def test_error_is_separate_from_correct_refusal_and_false_refusal():
    from garage_rag.answer_evaluation import summarize

    row = {
        "mode": "grounded",
        "supported": False,
        "expected_answerable": True,
        "error": True,
        "claim_support": None,
        "complete_answer": None,
        "trace": {
            "llm_calls": 1,
            "search_calls": 1,
            "elapsed_ms": 12,
            "input_tokens": 0,
            "output_tokens": 0,
            "usage_records": 0,
        },
    }
    summary = summarize([row])["grounded"]
    assert summary["errors"] == 1
    assert summary["false_refusals"] == 0
    assert summary["scored_runs"] == 0


def test_review_must_match_answer_hash_and_claim_count():
    from garage_rag.answer_evaluation import apply_reviews, evaluate_synthetic

    report, _ = evaluate_synthetic(dataset(), languages=["en"], case_ids=["cold_pressure"])
    row = report["rows"][0]
    review = {k: row[k] for k in ["id", "language", "mode", "trial", "answer_sha256"]}
    review.update(claim_support=[True], complete_answer=True)
    reviews = {
        "report_dataset_sha256": report["dataset_sha256"],
        "reviewer": "Codex agent",
        "independent_human_review": False,
        "rows": [review],
    }
    apply_reviews(report, reviews)
    reviews["rows"][0]["answer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="review"):
        apply_reviews(report, reviews)


def test_unknown_case_and_duplicate_languages_are_rejected():
    from garage_rag.answer_evaluation import evaluate_synthetic

    with pytest.raises(ValueError):
        evaluate_synthetic(dataset(), case_ids=["typo"])
    with pytest.raises(ValueError):
        evaluate_synthetic(dataset(), languages=["en", "en"])


def test_paired_summary_excludes_unmatched_runs_and_errors():
    from garage_rag.answer_evaluation import paired_summary

    def row(id, mode, error=False):
        return {
            "id": id,
            "language": "en",
            "trial": 1,
            "mode": mode,
            "supported": False,
            "expected_answerable": False,
            "error": error,
            "claim_support": [],
            "complete_answer": False,
            "trace": {
                "llm_calls": 1,
                "search_calls": 1,
                "elapsed_ms": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "usage_records": 0,
            },
        }

    result = paired_summary(
        [
            row("ok", "direct"),
            row("ok", "grounded"),
            row("unmatched", "direct"),
            row("failed", "direct"),
            row("failed", "grounded", True),
        ]
    )
    assert result["pairs"] == 1
    assert result["summary"]["direct"]["runs"] == 1
    assert result["summary"]["grounded"]["runs"] == 1


def test_offline_command_does_not_construct_generation_client(tmp_path, monkeypatch):
    from garage_rag.answer_evaluation import main

    def forbidden(**kwargs):
        pytest.fail("Offline evaluation must not construct a remote client")

    monkeypatch.setattr("garage_rag.generation.make_llm", forbidden)
    output = tmp_path / "report.json"
    assert main(["--output", str(output), "--cases", "cold_pressure", "--languages", "en"]) == 0
    report = json.loads(output.read_text())
    assert report["execution"] == "scripted"
    assert report["paired"]["pairs"] == 1
    assert report["provenance"]["generation"] is None
