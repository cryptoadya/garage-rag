import json
from types import SimpleNamespace

import pytest

from garage_rag.cli import main


def test_missing_index_is_reported_without_creating_one(tmp_path, capsys):
    directory = tmp_path / "missing"
    assert main(["--index-dir", str(directory), "stats"]) == 2
    assert not directory.exists()
    assert "run garage-rag index first" in capsys.readouterr().err


def test_stats_and_search_need_no_openrouter_key(
    tmp_path, pdf_path, tokenizer, embeddings, config, monkeypatch, capsys
):
    from garage_rag.index import Index

    directory = tmp_path / "index"
    with Index(directory, config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
    monkeypatch.setattr("garage_rag.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("garage_rag.cli.IndexConfig", lambda **kw: config)
    monkeypatch.setattr("garage_rag.cli.E5Embeddings", lambda *a, **kw: embeddings)
    assert main(["--index-dir", str(directory), "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["pages"] == 4
    assert main(["--index-dir", str(directory), "search", "шины", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["metadata"]["pdf_page"] == 1


def test_missing_key_never_echoes_environment(tmp_path, monkeypatch, capsys, config):
    from garage_rag.index import Index

    directory = tmp_path / "index"
    with Index(directory, config):
        pass
    monkeypatch.setattr("garage_rag.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-print-this-value")
    assert main(["--index-dir", str(directory), "ask", "шины?"]) == 2
    stderr = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in stderr
    assert "do-not-print-this-value" not in stderr


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize(
    ("question", "language", "option", "claim"),
    [
        ("How do I check tire pressure?", "en", [], "Check the tires when they are cold."),
        ("Wie prüfe ich den Reifendruck?", "de", [], "Prüfen Sie die Reifen im kalten Zustand."),
        ("Как проверить шины?", "ru", [], "Проверяйте давление на холодных шинах."),
        (
            "How do I check tire pressure?",
            "de",
            ["--answer-language", "de"],
            "Prüfen Sie die Reifen im kalten Zustand.",
        ),
    ],
)
def test_ask_language_through_cli(
    tmp_path,
    pdf_path,
    tokenizer,
    embeddings,
    config,
    monkeypatch,
    capsys,
    question,
    language,
    option,
    claim,
    truncated,
):
    from garage_rag.index import Index

    directory = tmp_path / "index"
    with Index(directory, config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
    response = {
        "supported": True,
        "claims": [
            {
                "text": claim,
                "evidence": [
                    {"source_id": "S1", "quote": "Check tire pressure when the tires are cold."}
                ],
            }
        ],
    }
    llm = SimpleNamespace(
        invoke=lambda messages: SimpleNamespace(
            content=json.dumps(response),
            response_metadata={"finish_reason": "length" if truncated else "stop"},
        ),
        client=SimpleNamespace(__exit__=lambda *a: None),
    )
    monkeypatch.setattr("garage_rag.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr("garage_rag.cli.make_llm", lambda **kw: llm)
    monkeypatch.setattr("garage_rag.cli.IndexConfig", lambda **kw: config)
    monkeypatch.setattr("garage_rag.cli.E5Embeddings", lambda *a, **kw: embeddings)
    assert main(["--index-dir", str(directory), "ask", question, "--json", *option]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    if truncated:
        assert not result["supported"]
        assert result["reason"] == "output_token_limit"
        assert "OPENROUTER_MAX_TOKENS" in captured.err
        return
    assert result["supported"]
    assert result["language"] == language
    assert "aurora.pdf, PDF page 1" in result["text"]


def test_unsupported_answer_language_is_a_cli_usage_error():
    from garage_rag.cli import parser

    with pytest.raises(SystemExit) as exc:
        parser().parse_args(["ask", "tires?", "--answer-language", "fr"])
    assert exc.value.code == 2
