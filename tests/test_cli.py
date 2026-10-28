import json
from types import SimpleNamespace

import pytest

from garage_rag.cli import main


def test_missing_index_is_reported_without_creating_one(tmp_path, capsys):
    directory = tmp_path / "missing"
    assert main(["--index-dir", str(directory), "stats"]) == 2
    assert not directory.exists()
    assert "run garage-rag index first" in capsys.readouterr().err


@pytest.mark.parametrize("encrypted", [False, True])
def test_pdf_validation_keeps_existing_cli_diagnostics(
    tmp_path, tokenizer, embeddings, config, monkeypatch, capsys, encrypted
):
    from pypdf import PdfWriter

    path = tmp_path / "manual.pdf"
    writer = PdfWriter()
    writer.add_blank_page(300, 300)
    if encrypted:
        writer.encrypt("private-password")
    writer.write(path)
    embeddings.tokenizer = tokenizer
    monkeypatch.setattr("garage_rag.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr("garage_rag.cli.IndexConfig", lambda **kw: config)
    monkeypatch.setattr("garage_rag.cli.E5Embeddings", lambda *a, **kw: embeddings)
    assert main(["--index-dir", str(tmp_path / "index"), "index", str(path)]) == 2
    output = capsys.readouterr()
    expected = ("Password-protected PDF is not supported" if encrypted
                else "PDF produced no text chunks; OCR may be needed")
    assert output.err == (
        "Indexing locally; private text is stored only in the index.\n"
        f"Error: {expected}\n"
    )
    assert output.out == ""


def test_stats_and_search_need_no_groq_key(
    tmp_path, pdf_path, tokenizer, embeddings, config, monkeypatch, capsys
):
    from garage_rag.index import Index

    directory = tmp_path / "index"
    with Index(directory, config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
    monkeypatch.setattr("garage_rag.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
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
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_FREE_PLAN_CONFIRMED", "true")
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-print-this-value")
    assert main(["--index-dir", str(directory), "ask", "шины?"]) == 2
    stderr = capsys.readouterr().err
    assert "GROQ_API_KEY" in stderr
    assert "do-not-print-this-value" not in stderr


@pytest.mark.parametrize("workflow", ["grounded", "direct"])
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
    workflow,
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
            content=json.dumps(
                response
                if "evidence auditor" not in messages[0].content
                else {
                    "decision": "accept",
                    "complete": True,
                    "checks": [
                        {
                            "claim_index": 0,
                            "entailed": True,
                            "applicable": True,
                            "numbers_safe": True,
                        }
                    ],
                }
            ),
            response_metadata={"finish_reason": "length" if truncated else "stop"},
        ),
        client=SimpleNamespace(__exit__=lambda *a: None),
    )
    monkeypatch.setattr("garage_rag.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr("garage_rag.cli.make_llm", lambda **kw: llm)
    monkeypatch.setattr("garage_rag.cli.IndexConfig", lambda **kw: config)
    monkeypatch.setattr("garage_rag.cli.E5Embeddings", lambda *a, **kw: embeddings)
    assert (
        main(
            [
                "--index-dir",
                str(directory),
                "ask",
                question,
                "--json",
                "--workflow",
                workflow,
                *option,
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    if truncated:
        assert not result["supported"]
        assert result["reason"] == "output_token_limit"
        assert "GROQ_MAX_TOKENS" in captured.err
        return
    assert result["supported"]
    assert result["language"] == language
    assert "aurora.pdf, PDF page 1" in result["text"]


def test_unsupported_answer_language_is_a_cli_usage_error():
    from garage_rag.cli import parser

    with pytest.raises(SystemExit) as exc:
        parser().parse_args(["ask", "tires?", "--answer-language", "fr"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "explicit", [[], ["--provider", "groq"]]
)
@pytest.mark.parametrize("workflow", ["direct", "grounded", "agentic"])
def test_all_modes_default_to_groq(
    tmp_path, monkeypatch, capsys, explicit, workflow
):
    from garage_rag.generation import GenerationError

    directory = tmp_path / "index"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    monkeypatch.setattr("garage_rag.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("GARAGE_RAG_PROVIDER", raising=False)
    calls = []

    def make(**kw):
        calls.append(kw)
        raise GenerationError("intentional preflight stop")

    monkeypatch.setattr("garage_rag.cli.make_llm", make)
    assert (
        main(["--index-dir", str(directory), "ask", "tires", "--workflow", workflow, *explicit])
        == 2
    )
    assert calls[0]["provider"] == "groq"
    capsys.readouterr()


def test_grounded_expanded_caps_rejected_before_provider(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "index"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    monkeypatch.setattr("garage_rag.cli.make_llm", lambda **kw: pytest.fail("provider constructed"))
    assert main(["--index-dir", str(directory), "ask", "tires", "--max-llm-calls", "6"]) == 2
    assert "max_llm_calls" in capsys.readouterr().err


def test_agentic_cli_real_local_index(
    tmp_path, pdf_path, tokenizer, embeddings, config, monkeypatch, capsys
):
    from contextlib import nullcontext

    from conftest import Model, control, reply, tool
    from test_workflow import audit, draft

    from garage_rag.index import Index

    directory = tmp_path / "index"
    with Index(directory, config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
    llm = Model(tool("lexical_search"), control(), reply(draft()), reply(audit()))
    llm.client = nullcontext()
    monkeypatch.setattr("garage_rag.cli.make_llm", lambda **kw: llm)
    monkeypatch.setattr("garage_rag.cli.IndexConfig", lambda **kw: config)
    monkeypatch.setattr("garage_rag.cli.E5Embeddings", lambda *a, **kw: embeddings)
    assert (
        main(
            [
                "--index-dir",
                str(directory),
                "ask",
                "How to check tire pressure?",
                "--workflow",
                "agentic",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["supported"]
    assert result["workflow"]["mode"] == "agentic"
    assert result["workflow"]["llm_calls"] == 4
