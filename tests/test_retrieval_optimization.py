"""Synthetic-only regression checks for Phase 2B; no model downloads."""

import pytest

from garage_rag.config import IndexConfig
from garage_rag.index import Index


def test_profiles_preserve_baseline_and_isolate_base():
    from garage_rag.profiles import profile_config

    assert profile_config("baseline") == IndexConfig()
    base = profile_config("e5-base")
    assert base.dimension == 768
    assert base.revision == "d128750597153bb5987e10b1c3493a34e5a4502a"
    assert base.fingerprint != IndexConfig().fingerprint


def test_wrong_profile_cannot_open_baseline(tmp_path):
    from garage_rag.profiles import profile_config

    with Index(tmp_path / "index"):
        pass
    with pytest.raises(ValueError, match="incompatible"):
        Index(tmp_path / "index", profile_config("e5-base"))


def test_bm25_reads_only_active_revisions(tmp_path, pdf_path, tokenizer, embeddings, config):
    from garage_rag.retrieval import Retriever

    with Index(tmp_path / "index", config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        before = index.manifest_path.read_bytes()
        retriever = Retriever(index, method="bm25")
        hits = retriever.search("coolant", None, top_k=5)
        assert hits[0].document.metadata["pdf_page"] == 2
        assert index.manifest_path.read_bytes() == before
        assert retriever.search("абракадабра", None) == []
        # A replacement generation must invalidate the lexical cache.
        index.ingest(pdf_path, embeddings, tokenizer, metadata={"title": "replacement"})
        assert retriever.search("coolant", None)[0].document.metadata["title"] == "replacement"


@pytest.mark.parametrize("method", ["bm25", "hybrid", "rerank"])
def test_noncosine_methods_reject_cosine_threshold(tmp_path, config, method):
    from garage_rag.retrieval import Retriever

    with Index(tmp_path / "index", config) as index:
        with pytest.raises(ValueError, match="min_score"):
            Retriever(index, method=method).search("tires", None, min_score=0.8)


def test_rrf_empty_lexical_branch_preserves_dense_order():
    from garage_rag.retrieval import reciprocal_rank_fusion

    assert reciprocal_rank_fusion(["b", "a"], []) == ["b", "a"]
    assert reciprocal_rank_fusion(["b", "a"], ["a", "b"]) == ["a", "b"]


def test_reranker_reorders_candidates_and_rejects_nonfinite(
    tmp_path, pdf_path, tokenizer, embeddings, config
):
    from garage_rag.retrieval import Retriever

    class Reranker:
        def predict(self, pairs, **kwargs):
            return [float(i) for i in range(len(pairs))]

    with Index(tmp_path / "index", config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        r = Retriever(index, method="rerank", reranker=Reranker())
        dense = index.search("tires", embeddings, top_k=20)
        assert r.search("tires", embeddings)[0].document == dense[-1].document
        r.reranker.predict = lambda pairs, **kw: [float("nan")] * len(pairs)
        with pytest.raises(ValueError, match="reranker"):
            r.search("tires", embeddings)


def test_evidence_metrics_distinguish_wrong_chunk_and_layout():
    from garage_rag.optimization import evidence_summary, validate_evidence

    data = {
        "cases": [
            {
                "id": "a",
                "gold_pages": [1],
                "evidence": {
                    "sufficient_chunks": [{"pdf_page": 1, "chunk_index": 1}],
                    "requires_layout": False,
                },
            },
            {
                "id": "table",
                "gold_pages": [2],
                "evidence": {"sufficient_chunks": [], "requires_layout": True},
            },
        ]
    }
    validate_evidence(data)

    def row(id, hits):
        return {
            "id": id,
            "language": "de",
            "review_status": "agent_verified_pdf",
            "trials": [{"hits": hits}],
        }

    result = {
        "rows": [
            row("a", [{"pdf_page": 1, "chunk_index": 0}]),
            row("table", [{"pdf_page": 2, "chunk_index": 0}]),
        ]
    }
    metrics = evidence_summary(result, data)["de"]
    assert metrics["n"] == 1
    assert metrics["layout_required_excluded"] == 1
    assert metrics["hit@5"] == 0
    result["rows"][0]["trials"][0]["hits"].append({"pdf_page": 1, "chunk_index": 1})
    assert evidence_summary(result, data)["de"]["mrr@10"] == 0.5
    data["cases"][1]["evidence"]["sufficient_chunks"] = [{"pdf_page": 2, "chunk_index": 0}]
    with pytest.raises(ValueError, match="Layout"):
        validate_evidence(data)


def test_public_splits_are_disjoint_and_frozen():
    import json
    from pathlib import Path

    from garage_rag.evaluation import validate_dataset
    from garage_rag.optimization import validate_evidence

    datasets = []
    for name in [
        "bmw_e39_2002.v1.json",
        "bmw_e39_2002.phase2b-dev.v1.json",
        "bmw_e39_2002.phase2b-heldout.v1.json",
    ]:
        data = validate_dataset(
            json.loads((Path(__file__).parents[1] / "benchmarks" / "datasets" / name).read_bytes())
        )
        validate_evidence(data)
        datasets.append({c["id"] for c in data["cases"]})
    assert all(not a & b for i, a in enumerate(datasets) for b in datasets[i + 1 :])


def test_cli_profile_requires_separate_index(tmp_path, capsys):
    from garage_rag.cli import main

    with Index(tmp_path / "index"):
        pass
    assert main(["--profile", "e5-base", "--index-dir", str(tmp_path / "index"), "stats"]) == 2
    assert "incompatible" in capsys.readouterr().err


@pytest.mark.parametrize(
    "question,language,claim",
    [
        ("How do I check tire pressure?", "en", "Check the tires when they are cold."),
        ("Wie prüfe ich den Reifendruck?", "de", "Prüfen Sie die Reifen im kalten Zustand."),
        ("Как проверить шины?", "ru", "Проверяйте давление на холодных шинах."),
    ],
)
def test_selected_profile_ask_preserves_language_and_verified_citations(
    tmp_path,
    pdf_path,
    tokenizer,
    embeddings,
    config,
    monkeypatch,
    capsys,
    question,
    language,
    claim,
):
    import json
    from types import SimpleNamespace

    from garage_rag.cli import main

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
            response_metadata={"finish_reason": "stop"},
        ),
        client=SimpleNamespace(__exit__=lambda *a: None),
    )
    monkeypatch.setattr("garage_rag.cli.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("GARAGE_RAG_TRANSLATE_QUERY", "false")
    monkeypatch.setattr("garage_rag.cli.profile_config", lambda *a, **kw: config)
    monkeypatch.setattr("garage_rag.cli.E5Embeddings", lambda *a, **kw: embeddings)
    monkeypatch.setattr("garage_rag.cli.make_llm", lambda **kw: llm)
    assert (
        main(["--profile", "bge-m3", "--index-dir", str(directory), "ask", question, "--json"]) == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["supported"]
    assert result["language"] == language
    assert "aurora.pdf, PDF page 1" in result["text"]


def test_bge_profile_has_no_e5_prefix_and_is_isolated():
    from garage_rag.profiles import profile_config

    c = profile_config("bge-m3")
    assert c.model == "BAAI/bge-m3"
    assert c.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert c.dimension == 1024
    assert c.query_prefix == c.passage_prefix == ""
    assert c.fingerprint != IndexConfig().fingerprint


@pytest.mark.parametrize(
    "chunks,layout",
    [
        ([{"pdf_page": True, "chunk_index": 0}], False),
        ([{"pdf_page": 1, "chunk_index": True}], False),
        ([{"pdf_page": 1, "chunk_index": 0}] * 2, False),
        ([], False),
    ],
)
def test_malformed_evidence_labels_fail_closed(chunks, layout):
    from garage_rag.optimization import validate_evidence

    with pytest.raises(ValueError, match="evidence|chunk"):
        validate_evidence(
            {
                "cases": [
                    {
                        "gold_pages": [1],
                        "evidence": {"sufficient_chunks": chunks, "requires_layout": layout},
                    }
                ]
            }
        )


def test_bge_encoder_inputs_never_get_e5_prefixes(tokenizer, monkeypatch):
    import numpy as np

    from garage_rag.embeddings import E5Embeddings
    from garage_rag.profiles import profile_config

    inputs = []

    class Encoder:
        def encode(self, texts, **kwargs):
            inputs.extend(texts)
            return np.ones((len(texts), 1024))

    encoder = E5Embeddings(profile_config("bge-m3"))
    monkeypatch.setattr(encoder, "_model", Encoder())
    monkeypatch.setattr(encoder, "_tokenizer", tokenizer)
    encoder.embed_documents(["cold tires"])
    encoder.embed_query("давление")
    assert inputs == ["cold tires", "давление"]
