"""Offline retrieval evaluation; source text and index payloads never enter reports."""

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

LANGUAGES = ("en", "de", "ru")
VERIFIED = {"agent_verified_pdf", "human_verified_pdf"}


def _require(condition):
    if not condition:
        raise ValueError("Invalid benchmark field")


def validate_dataset(data):
    """Reject ambiguous labels; unverified answerable labels remain explicitly unscored."""
    try:
        _require(data["schema_version"] == 1)
        _require(isinstance(data["id"], str) and data["id"].strip())
        source = data["source"]
        _require(re.fullmatch(r"[0-9a-f]{64}", source["sha256"]))
        _require(type(source["pages"]) is int and source["pages"] > 0)
        _require(isinstance(data["cases"], list) and data["cases"])
        seen = set()
        for case in [*data["cases"], *data.get("diagnostics", [])]:
            _require(isinstance(case["id"], str) and case["id"].strip())
            _require(case["id"] not in seen)
            seen.add(case["id"])
            _require(set(case["questions"]) == set(LANGUAGES))
            _require(
                all(
                    isinstance(q, str) and 0 < len(q.strip()) <= 4000
                    for q in case["questions"].values()
                )
            )
        for probe in data.get("diagnostics", []):
            _require(not {"kind", "gold_pages", "review"}.intersection(probe))
        for case in data["cases"]:
            _require(case["kind"] in {"answerable", "unanswerable"})
            pages = case["gold_pages"]
            _require(isinstance(pages, list) and len(pages) == len(set(pages)))
            _require(all(type(p) is int and 1 <= p <= source["pages"] for p in pages))
            _require(bool(pages) == (case["kind"] == "answerable"))
            review = case["review"]
            _require(type(review["human_verified"]) is bool)
            if case["kind"] == "answerable":
                _require(review["status"] in VERIFIED | {"unverified"})
                _require(review["human_verified"] == (review["status"] == "human_verified_pdf"))
            else:
                _require(review["status"] in {"agent_scope_checked", "unverified"})
                _require(not review["human_verified"])
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid benchmark dataset or review labels") from exc
    return data


def ranking_metrics(ranked_pages, gold_pages):
    """Binary page relevance at original chunk ranks, with reciprocal rank censored at 10."""
    if not gold_pages:
        raise ValueError("Cannot score an empty gold set")
    rank = next((i for i, p in enumerate(ranked_pages[:10], 1) if p in gold_pages), None)
    return {
        **{f"hit@{k}": int(rank is not None and rank <= k) for k in (1, 5, 10)},
        "mrr@10": 1 / rank if rank else 0,
    }


def distribution(values):
    """Linear-interpolated sample quantiles; timings are descriptive, not confidence bounds."""
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)

    def percentile(q):
        index = (len(ordered) - 1) * q
        left = int(index)
        return ordered[left] + (ordered[math.ceil(index)] - ordered[left]) * (index - left)

    return {
        "n": len(values),
        "min": ordered[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def summarize(rows):
    """Accuracy uses one first trial per verified need/language; repeats do not inflate n."""
    result = {}
    for lang in LANGUAGES:
        selected = [r for r in rows if r["language"] == lang]
        positive = [
            r for r in selected if r["kind"] == "answerable" and r["review_status"] in VERIFIED
        ]
        negative = [
            r
            for r in selected
            if r["kind"] == "unanswerable" and r["review_status"] == "agent_scope_checked"
        ]
        metrics = [
            ranking_metrics([h["pdf_page"] for h in r["trials"][0]["hits"]], r["gold_pages"])
            for r in positive
        ]
        result[lang] = {
            "answerable": {
                "n": len(positive),
                **{
                    key: statistics.mean(m[key] for m in metrics) if metrics else None
                    for key in ("hit@1", "hit@5", "hit@10", "mrr@10")
                },
            },
            "unanswerable": {
                "n": len(negative),
                "nonempty_rate": statistics.mean(bool(r["trials"][0]["hits"]) for r in negative)
                if negative
                else None,
                "top1_score": distribution(
                    [r["trials"][0]["hits"][0]["score"] for r in negative if r["trials"][0]["hits"]]
                ),
            },
            "answerable_top1_score": distribution(
                [r["trials"][0]["hits"][0]["score"] for r in positive if r["trials"][0]["hits"]]
            ),
            "excluded_unverified": sum(r["review_status"] == "unverified" for r in selected),
            "latency_ms": distribution([t["latency_ms"] for r in selected for t in r["trials"]]),
            "ranking_stable_queries": sum(
                all(
                    [(h["pdf_page"], h.get("chunk_index")) for h in t["hits"]]
                    == [(h["pdf_page"], h.get("chunk_index")) for h in r["trials"][0]["hits"]]
                    for t in r["trials"]
                )
                for r in selected
            ),
            "total_queries": len(selected),
        }
    return result


def validate_corpus(index, data):
    from qdrant_client import models

    from .index import COLLECTION

    records = list(index.manifest["documents"].values())
    if (
        len(records) != 1
        or records[0]["content_sha256"] != data["source"]["sha256"]
        or records[0]["pages"] != data["source"]["pages"]
    ):
        raise ValueError("Benchmark corpus must contain exactly the matching PDF revision")
    record = records[0]
    count = index.client.count(
        COLLECTION,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="generation", match=models.MatchValue(value=record["generation"])
                )
            ]
        ),
        exact=True,
    ).count
    if count != record["chunks"]:
        raise ValueError("Benchmark corpus has incomplete active chunks; restore the index first")
    return record


def run_benchmark(index, embeddings, data, *, repeats=3, seed=20261025):
    """Use the unchanged search entry point: top_k=10, no score threshold, no deduplication."""
    validate_dataset(data)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    record = validate_corpus(index, data)
    before = index.manifest_path.read_bytes()
    rows = []
    for case in [*data["cases"], *data.get("diagnostics", [])]:
        for lang in LANGUAGES:
            rows.append(
                {
                    "id": case["id"],
                    "language": lang,
                    "kind": case.get("kind", "diagnostic"),
                    "question": case["questions"][lang],
                    "gold_pages": case.get("gold_pages", []),
                    "review_status": case.get("review", {}).get("status", "unscored"),
                    "trials": [],
                }
            )
    warmup_start = time.perf_counter()
    index.search(rows[0]["question"], embeddings, top_k=10, min_score=None)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    rng = random.Random(seed)
    orders = []
    for repeat in range(repeats):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in order:
            row = rows[i]
            start = time.perf_counter()
            hits = index.search(row["question"], embeddings, top_k=10, min_score=None)
            elapsed = (time.perf_counter() - start) * 1000
            safe_hits = []
            for hit in hits:
                meta = hit.document.metadata
                if (
                    meta["content_sha256"] != data["source"]["sha256"]
                    or not 1 <= meta["pdf_page"] <= data["source"]["pages"]
                    or not math.isfinite(hit.score)
                ):
                    raise ValueError("Invalid corpus hit metadata or score")
                safe_hits.append(
                    {
                        "pdf_page": meta["pdf_page"],
                        "chunk_index": meta["chunk_index"],
                        "score": hit.score,
                    }
                )
            row["trials"].append({"latency_ms": elapsed, "hits": safe_hits})
            orders.append({"repeat": repeat + 1, "id": row["id"], "language": row["language"]})
    if index.manifest_path.read_bytes() != before:
        raise ValueError("Index manifest changed during evaluation")
    return {
        "schema_version": 1,
        "dataset_id": data["id"],
        "source": {
            "sha256": data["source"]["sha256"],
            "pages": record["pages"],
            "chunks": record["chunks"],
            "generation": record["generation"],
        },
        "protocol": {
            "top_k": 10,
            "min_score": None,
            "rank_unit": "chunk",
            "relevance_unit": "physical_pdf_page_1_based",
            "mrr_cutoff": 10,
            "accuracy_trial": 1,
            "repeats": repeats,
            "seed": seed,
            "latency_scope": "warm Index.search: query tokenization + embedding + Qdrant",
            "quantiles": "linear interpolation",
            "warmup_queries_excluded": 1,
        },
        "warmup_ms": warmup_ms,
        "summary": summarize(rows),
        "rows": rows,
        "query_order": orders,
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline GarageRAG retrieval benchmark")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; no source text")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    # Do not read .env or initialize any generation client. Cached model only.
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ[key] = value
    os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache/huggingface"))
    logging.disable(logging.CRITICAL)
    try:
        from pypdf import PdfReader

        from .config import IndexConfig
        from .embeddings import E5Embeddings
        from .index import Index

        if not (args.index_dir / "manifest.json").is_file():
            raise ValueError("Existing index required; run garage-rag index first")
        if args.output.exists():
            raise ValueError("Output already exists; choose a new result file")
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
        raw = args.dataset.read_bytes()
        data = validate_dataset(json.loads(raw))
        if _sha(args.pdf) != data["source"]["sha256"]:
            raise ValueError("Original PDF hash does not match the benchmark corpus")
        if len(PdfReader(args.pdf).pages) != data["source"]["pages"]:
            raise ValueError("Original PDF page count does not match the benchmark corpus")
        config = IndexConfig()
        start = time.perf_counter()
        with Index(args.index_dir, config) as index:
            opening_ms = (time.perf_counter() - start) * 1000
            validate_corpus(index, data)
            embeddings = E5Embeddings(config, device=args.device)
            start = time.perf_counter()
            _ = embeddings.tokenizer, embeddings.model
            model_load_ms = (time.perf_counter() - start) * 1000
            result = run_benchmark(index, embeddings, data, repeats=args.repeats, seed=args.seed)
        import torch

        result["run"] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "configuration": config.as_dict(),
            "fingerprint": config.fingerprint,
            "device": args.device,
            "python": platform.python_version(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "libraries": {
                name: version(name)
                for name in ("garage-rag", "torch", "numpy", "qdrant-client", "huggingface-hub")
            },
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "startup_ms": {
                "index_open": opening_ms,
                "cached_model_load": model_load_ms,
                "first_warmup_search": result["warmup_ms"],
            },
            "code_sha256": {
                name: _sha(Path(__file__).with_name(name))
                for name in (
                    "evaluation.py",
                    "index.py",
                    "embeddings.py",
                    "ingestion.py",
                    "config.py",
                )
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            out.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        if type(exc) in (ValueError, FileNotFoundError):
            # Filesystem exception strings can include private local paths.
            message = str(exc) if type(exc) is ValueError else "Required local file is missing"
            print(f"Error: {message}", file=sys.stderr)
        else:
            print("Error: invalid local benchmark data", file=sys.stderr)
        return 2
    except Exception:
        print(
            "Error: offline evaluation failed; check model cache, index lock and local files",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
