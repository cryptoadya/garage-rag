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


def test_live_limiter_paces_all_calls_and_stops_at_budget(monkeypatch):
    import garage_rag.answer_evaluation as evaluation
    from garage_rag.generation import GenerationError

    clock = [0.0]
    starts = []
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        evaluation.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )

    class Model:
        def invoke(self, messages):
            starts.append(clock[0])
            if len(starts) == 2:
                raise RuntimeError("provider failed")
            return "response"

    model = evaluation.BudgetedPacedLLM(Model(), max_api_calls=3, requests_per_minute=15)
    assert model.invoke([]) == "response"
    with pytest.raises(RuntimeError, match="provider failed"):
        model.invoke([])
    assert model.invoke([]) == "response"
    with pytest.raises(GenerationError, match="budget exhausted"):
        model.invoke([])
    assert len(starts) == model.calls == 3
    assert all(b - a >= 4 for a, b in zip(starts, starts[1:], strict=False))


@pytest.mark.parametrize("rate", [0, 16, True])
def test_live_limiter_rejects_unsafe_rates(rate):
    from garage_rag.answer_evaluation import BudgetedPacedLLM

    with pytest.raises(ValueError, match="1.*15"):
        BudgetedPacedLLM(None, max_api_calls=5, requests_per_minute=rate)


def test_checkpoint_resume_skips_completed_modes_and_preserves_error_history():
    from garage_rag.answer_evaluation import _evaluate, _synthetic_search
    from garage_rag.generation import GenerationError

    data = dataset()
    cases = [c for c in data["scenarios"] if c["id"] == "cold_pressure"]
    checkpoints = []

    class Interrupted:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 2:
                raise GenerationError("Groq quota stop")
            from types import SimpleNamespace

            return SimpleNamespace(content='{"supported": false, "claims": []}')

    first = Interrupted()
    report, _ = _evaluate(
        data,
        cases,
        ["en"],
        1,
        17,
        first,
        _synthetic_search,
        scripted=False,
        checkpoint=lambda r, c: checkpoints.append(json.loads(json.dumps(r))),
    )
    assert len(checkpoints) >= 2
    assert len(checkpoints[-1]["rows"]) == 2
    assert not report["complete"]
    completed = next(r for r in report["rows"] if not r["error"])
    second = Interrupted()
    resumed, _ = _evaluate(
        data, cases, ["en"], 1, 17, second, _synthetic_search, scripted=False, previous=report
    )
    assert resumed["complete"]
    assert second.calls == 1
    assert len(resumed["rows"]) == 2
    assert completed in resumed["rows"]
    assert len(resumed["previous_errors"]) == 1


def test_resume_rejects_model_or_selection_drift_before_api(tmp_path, monkeypatch):
    from garage_rag.answer_evaluation import main

    output = tmp_path / "result.json"
    args = ["--output", str(output), "--cases", "cold_pressure", "--languages", "en"]
    assert main(args) == 0

    def forbidden(**kwargs):
        pytest.fail("Resume mismatch must be caught before constructing client")

    monkeypatch.setattr("garage_rag.generation.make_llm", forbidden)
    assert (
        main(
            [
                *args,
                "--resume",
                "--live",
                "--provider",
                "groq",
                "--model",
                "openai/gpt-oss-120b",
                "--max-api-calls",
                "5",
            ]
        )
        == 2
    )
    assert json.loads(output.read_text())["execution"] == "scripted"


def test_live_cli_checkpoints_failure_and_resumes_only_unfinished_mode(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from garage_rag.answer_evaluation import main
    from garage_rag.generation import GenerationError

    calls = []

    class Client:
        def __exit__(self, *args):
            pass

    class Model:
        client = Client()
        max_tokens = 8192
        temperature = 0

        def invoke(self, messages):
            calls.append(messages)
            if len(calls) == 2:
                raise GenerationError("OpenRouter request failed")
            return SimpleNamespace(content='{"supported":false,"claims":[]}')

    monkeypatch.setattr("garage_rag.generation.make_llm", lambda **kw: Model())
    monkeypatch.setattr("garage_rag.answer_evaluation.time.sleep", lambda delay: None)
    # Disable pacing only in this test; production limiter has a separate fake-clock regression.
    clock = iter(range(0, 1000, 60))
    monkeypatch.setattr("garage_rag.answer_evaluation.time.monotonic", lambda: next(clock))
    output = tmp_path / "live.json"
    args = [
        "--output",
        str(output),
        "--cases",
        "cold_pressure",
        "--languages",
        "en",
        "--live",
        "--model",
        "test/model:free",
        "--provider",
        "openrouter",
        "--max-api-calls",
        "5",
    ]
    assert main(args) == 2
    partial = json.loads(output.read_text())
    assert len(partial["rows"]) == 2
    assert not partial["complete"]
    assert partial["provenance"]["model"] == "test/model:free"
    assert main([*args, "--resume"]) == 0
    resumed = json.loads(output.read_text())
    assert len(calls) == 3
    assert resumed["complete"]
    assert len(resumed["previous_errors"]) == 1
    assert resumed["provenance"]["generation"]["previous_attempted_api_calls"] == 2
    assert main([*args, "--resume"]) == 0
    assert len(calls) == 3
