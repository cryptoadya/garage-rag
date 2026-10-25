"""Evaluation contracts use synthetic ranks and real local Qdrant, never BMW text."""

import hashlib
import json
from pathlib import Path

import pytest

from garage_rag import evaluation as ev
from garage_rag.index import Index


def dataset(pdf_path):
    return {
        "schema_version": 1,
        "id": "synthetic",
        "source": {"sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(), "pages": 4},
        "cases": [
            {
                "id": "tires",
                "kind": "answerable",
                "gold_pages": [1],
                "questions": {"en": "tire pressure", "de": "Reifendruck", "ru": "давление шин"},
                "review": {"status": "agent_verified_pdf", "human_verified": False},
            },
            {
                "id": "unsupported",
                "kind": "unanswerable",
                "gold_pages": [],
                "questions": {"en": "unknown", "de": "unbekannt", "ru": "неизвестно"},
                "review": {"status": "agent_scope_checked", "human_verified": False},
            },
        ],
        "diagnostics": [],
    }


@pytest.mark.parametrize(
    "pages,gold,want",
    [
        ([8, 8, 3], [3], {"hit@1": 0, "hit@5": 1, "hit@10": 1, "mrr@10": 1 / 3}),
        ([3, 9], [3, 4], {"hit@1": 1, "hit@5": 1, "hit@10": 1, "mrr@10": 1}),
        ([8] * 9 + [3], [3], {"hit@1": 0, "hit@5": 0, "hit@10": 1, "mrr@10": 0.1}),
        ([8] * 10 + [3], [3], {"hit@1": 0, "hit@5": 0, "hit@10": 0, "mrr@10": 0}),
        ([], [3], {"hit@1": 0, "hit@5": 0, "hit@10": 0, "mrr@10": 0}),
    ],
)
def test_metrics_use_chunk_rank_with_ten_cutoff(pages, gold, want):
    assert ev.ranking_metrics(pages, gold) == want


def test_empty_gold_cannot_be_scored_as_a_retrieval_miss():
    with pytest.raises(ValueError, match="gold"):
        ev.ranking_metrics([1], [])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["cases"][0]["questions"].pop("ru"),
        lambda d: d["cases"][0].update(gold_pages=[0]),
        lambda d: d["cases"][0].update(gold_pages=[5]),
        lambda d: d["cases"][0].update(gold_pages=[True]),
        lambda d: d["cases"][0].update(gold_pages=[]),
        lambda d: d["cases"][1].update(gold_pages=[1]),
        lambda d: d["cases"][1].update(id="tires"),
        lambda d: d["cases"][0]["review"].update(status="manual"),
        lambda d: d["source"].update(sha256="broken"),
    ],
)
def test_malformed_labels_fail_closed(pdf_path, mutation):
    data = dataset(pdf_path)
    mutation(data)
    with pytest.raises(ValueError):
        ev.validate_dataset(data)


def test_unverified_labels_and_negatives_are_excluded_from_accuracy():
    def row(id, kind, status, pages, gold):
        return {
            "id": id,
            "language": "en",
            "kind": kind,
            "review_status": status,
            "gold_pages": gold,
            "trials": [
                {"hits": [{"pdf_page": p, "score": 0.8} for p in pages], "latency_ms": 10.0}
            ],
        }

    rows = [
        row("a", "answerable", "agent_verified_pdf", [9, 3], [3]),
        row("b", "answerable", "agent_verified_pdf", [8], [3]),
        row("c", "answerable", "unverified", [3], [3]),
        row("d", "unanswerable", "agent_scope_checked", [3], []),
    ]
    result = ev.summarize(rows)
    assert result["en"]["answerable"]["n"] == 2
    assert result["en"]["answerable"]["hit@1"] == 0
    assert result["en"]["answerable"]["hit@5"] == 0.5
    assert result["en"]["answerable"]["mrr@10"] == 0.25
    assert result["en"]["unanswerable"]["n"] == 1
    assert result["en"]["unanswerable"]["nonempty_rate"] == 1
    assert result["en"]["excluded_unverified"] == 1
    assert result["en"]["latency_ms"]["n"] == 4


def test_real_index_evaluation_is_reproducible_and_text_free(
    tmp_path, pdf_path, tokenizer, embeddings, config
):
    with Index(tmp_path / "index", config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        before = index.manifest_path.read_bytes()
        calls = embeddings.document_calls
        result = ev.run_benchmark(index, embeddings, dataset(pdf_path), repeats=2, seed=7)
        again = ev.run_benchmark(index, embeddings, dataset(pdf_path), repeats=2, seed=7)
        assert result["summary"]["ru"]["answerable"]["hit@1"] == 1
        assert result["summary"]["en"]["answerable"]["n"] == 1
        assert result["summary"]["de"]["unanswerable"]["nonempty_rate"] == 1
        assert result["query_order"] == again["query_order"]
        assert len(result["query_order"]) == 12
        assert all(len(r["trials"]) == 2 for r in result["rows"])
        serialized = json.dumps(result)
        assert "page_content" not in serialized
        assert "2.4 bar" not in serialized
        assert str(tmp_path) not in serialized
        assert embeddings.document_calls == calls
        assert index.manifest_path.read_bytes() == before


def test_wrong_corpus_rejected_before_any_query(tmp_path, pdf_path, tokenizer, embeddings, config):
    with Index(tmp_path / "index", config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        data = dataset(pdf_path)
        data["source"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="corpus"):
            ev.run_benchmark(index, embeddings, data)
        data = dataset(pdf_path)
        data["source"]["pages"] = 5
        with pytest.raises(ValueError, match="corpus"):
            ev.run_benchmark(index, embeddings, data)


def test_missing_index_cli_does_not_create_storage(tmp_path, pdf_path, capsys):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(dataset(pdf_path)))
    missing = tmp_path / "missing"
    assert (
        ev.main(
            [
                "--dataset",
                str(labels),
                "--pdf",
                str(pdf_path),
                "--index-dir",
                str(missing),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )
    assert not missing.exists()
    assert "index" in capsys.readouterr().err.lower()


def test_public_dataset_has_valid_labels():
    path = Path(__file__).parents[1] / "benchmarks/bmw_e39_2002.v1.json"
    data = ev.validate_dataset(json.loads(path.read_text()))
    assert len([c for c in data["cases"] if c["kind"] == "answerable"]) >= 15
    assert all(not c["review"]["human_verified"] for c in data["cases"])


def test_diagnostics_cannot_silently_add_scored_labels(pdf_path):
    data = dataset(pdf_path)
    data["diagnostics"] = [{**data["cases"][0], "id": "diagnostic_injection"}]
    with pytest.raises(ValueError, match="dataset"):
        ev.validate_dataset(data)


def test_latency_quantiles_use_linear_interpolation():
    assert ev.distribution([30, 10, 20]) == {
        "n": 3,
        "min": 10,
        "p50": 20,
        "p95": 29,
        "max": 30,
    }
