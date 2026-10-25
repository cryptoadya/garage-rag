import json

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
