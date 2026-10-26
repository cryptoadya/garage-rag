import json
from pathlib import Path

import pytest


def test_packet_binds_answers_and_exact_sources_and_escapes_private_text(tmp_path):
    from garage_rag.answer_evaluation import evaluate_synthetic
    from garage_rag.review_packet import write_packet

    data = json.loads(Path("benchmarks/aurora-grounded-answering.v1.json").read_text())
    report, captures = evaluate_synthetic(data, languages=["en"], case_ids=["recovery"])
    captures[0]["answer"]["text"] += "<script>private</script>"
    # Tampered captures are rejected before anything is written.
    with pytest.raises(ValueError, match="capture"):
        write_packet(report, captures, tmp_path)
    for case in data["scenarios"]:
        if case["id"] == "recovery":
            case["initial"] = [t + "<script>private</script>" for t in case["initial"]]
            case["expanded"] = [t + "<script>private</script>" for t in case["expanded"]]
    report, captures = evaluate_synthetic(data, languages=["en"], case_ids=["recovery"])
    write_packet(report, captures, tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert "not independent human review" in page
    assert "blue box" in page
    assert "<script>private</script>" not in page
    assert "&lt;script&gt;private&lt;/script&gt;" in page
    template = json.loads((tmp_path / "source-review.template.json").read_text())
    assert template["independent_human_review"] is False
    assert all(c["sufficiency"] is None for r in template["rows"] for c in r["contexts"])
    with pytest.raises(FileExistsError):
        write_packet(report, captures, tmp_path)


def test_packet_rejects_changed_question_even_with_unchanged_sources(tmp_path):
    from garage_rag.answer_evaluation import evaluate_synthetic
    from garage_rag.review_packet import write_packet

    data = json.loads(Path("benchmarks/aurora-grounded-answering.v1.json").read_text())
    report, captures = evaluate_synthetic(data, languages=["en"], case_ids=["recovery"])
    message = captures[0]["model_calls"][0]["messages"][1]
    payload = json.loads(message["content"])
    payload["question"] = "Tampered question"
    message["content"] = json.dumps(payload)
    with pytest.raises(ValueError, match="capture"):
        write_packet(report, captures, tmp_path)


def test_packet_includes_failed_calls_without_inventing_response_or_review(tmp_path):
    from garage_rag.answer_evaluation import evaluate_synthetic
    from garage_rag.generation import GenerationError
    from garage_rag.review_packet import write_packet

    class Broken:
        def invoke(self, messages):
            raise GenerationError("Groq request failed")

    data = json.loads(Path("benchmarks/aurora-grounded-answering.v1.json").read_text())
    report, captures = evaluate_synthetic(
        data, languages=["en"], case_ids=["cold_pressure"], llm=Broken()
    )
    assert report["rows"][0]["error"]
    write_packet(report, captures, tmp_path)
    assert json.loads((tmp_path / "source-review.template.json").read_text())["rows"] == []
