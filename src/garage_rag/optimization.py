"""Reproducible Phase 2B experiments using the unmodified Phase 2A evaluator.

No ingestion, .env loading, generation client or network model download here.
Reports contain authored questions, ranks and numeric metadata, never passages.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from .evaluation import LANGUAGES, run_benchmark, validate_corpus, validate_dataset
from .profiles import RERANK_MODEL, RERANK_REVISION, profile_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_weights_bytes(model, revision):
    from huggingface_hub import try_to_load_from_cache

    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = try_to_load_from_cache(model, filename, revision=revision)
        if isinstance(path, str):
            return Path(path).stat().st_size
    return None


def chunk_digest(index):
    from qdrant_client import models

    from .index import COLLECTION

    active = [r["generation"] for r in index.manifest["documents"].values()]
    rows, cursor = [], None
    while active:
        points, cursor = index.client.scroll(
            COLLECTION,
            limit=256,
            offset=cursor,
            with_vectors=False,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="generation", match=models.MatchAny(any=active))]
            ),
        )
        for point in points:
            meta = point.payload["metadata"]
            rows.append(
                (
                    meta["pdf_page"],
                    meta["chunk_index"],
                    hashlib.sha256(point.payload["text"].encode()).hexdigest(),
                )
            )
        if cursor is None:
            break
    return hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()


def validate_evidence(data):
    for case in data["cases"]:
        if "evidence" not in case:
            continue
        evidence = case["evidence"]
        if type(evidence.get("requires_layout")) is not bool or not isinstance(
            evidence.get("sufficient_chunks"), list
        ):
            raise ValueError("Invalid evidence labels")
        seen = set()
        for chunk in evidence["sufficient_chunks"]:
            if (
                type(chunk.get("pdf_page")) is not int
                or chunk.get("pdf_page") not in case["gold_pages"]
                or type(chunk.get("chunk_index")) is not int
                or chunk["chunk_index"] < 0
            ):
                raise ValueError("Invalid sufficient chunk label")
            key = (chunk["pdf_page"], chunk["chunk_index"])
            if key in seen:
                raise ValueError("Duplicate sufficient chunk label")
            seen.add(key)
        if not evidence["requires_layout"] and not seen:
            raise ValueError("Text-sufficient evidence needs a chunk label")
        if evidence["requires_layout"] and evidence["sufficient_chunks"]:
            raise ValueError("Layout-dependent evidence cannot be scored as text-sufficient")


def evidence_summary(result, data):
    """A stricter, separately reported metric; layout cases stay out of this denominator."""
    cases = {c["id"]: c for c in data["cases"]}
    output = {}
    for language in LANGUAGES:
        rows = []
        layout = 0
        for row in result["rows"]:
            if row["language"] != language:
                continue
            evidence = cases.get(row["id"], {}).get("evidence")
            if evidence is None or row["review_status"] not in {
                "agent_verified_pdf",
                "human_verified_pdf",
            }:
                continue
            if evidence["requires_layout"]:
                layout += 1
                continue
            keys = {(h["pdf_page"], h["chunk_index"]) for h in evidence["sufficient_chunks"]}
            rank = next(
                (
                    i
                    for i, h in enumerate(row["trials"][0]["hits"], 1)
                    if (h["pdf_page"], h["chunk_index"]) in keys
                ),
                None,
            )
            rows.append(rank)
        output[language] = {
            "n": len(rows),
            "layout_required_excluded": layout,
            **{
                f"hit@{k}": sum(r is not None and r <= k for r in rows) / len(rows)
                if rows
                else None
                for k in [1, 5, 10]
            },
            "mrr@10": sum(1 / r for r in rows if r is not None) / len(rows) if rows else None,
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["baseline", "e5-base", "bge-m3"], default="baseline")
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "rerank"], default="dense")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args(argv)
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
        if args.output.exists():
            raise ValueError("Output already exists; choose a new result file")
        data = validate_dataset(json.loads(args.dataset.read_bytes()))
        validate_evidence(data)
        from pypdf import PdfReader

        if sha(args.pdf) != data["source"]["sha256"]:
            raise ValueError("Original PDF hash does not match the benchmark corpus")
        if len(PdfReader(args.pdf).pages) != data["source"]["pages"]:
            raise ValueError("Original PDF page count does not match the benchmark corpus")
        if not (args.index_dir / "manifest.json").is_file():
            raise ValueError("Existing compatible index required")
        import torch

        from .embeddings import E5Embeddings
        from .index import Index
        from .retrieval import LocalReranker, Retriever

        torch.set_num_threads(4)
        torch.set_num_interop_threads(8)
        config = profile_config(args.profile)
        start = time.perf_counter()
        with Index(args.index_dir, config) as index:
            opening_ms = (time.perf_counter() - start) * 1000
            validate_corpus(index, data)
            signature = chunk_digest(index)
            expected = data.get("design", {}).get("chunks_sha256")
            if expected and expected != signature:
                raise ValueError("Chunk text differs from the frozen evidence-label corpus")
            embeddings = E5Embeddings(config, device="cpu")
            start = time.perf_counter()
            if args.method != "bm25":
                _ = embeddings.tokenizer, embeddings.model
            reranker = LocalReranker() if args.method == "rerank" else None
            model_load_ms = (time.perf_counter() - start) * 1000
            retriever = Retriever(index, method=args.method, reranker=reranker)
            result = run_benchmark(
                retriever, embeddings, data, repeats=args.repeats, seed=args.seed
            )
            disk = sum(p.stat().st_size for p in args.index_dir.rglob("*") if p.is_file())
        result["protocol"]["latency_scope"] = (
            "warm Retriever.search: embedding + all ranking stages"
        )
        result["evidence_summary"] = evidence_summary(result, data)
        result["run"] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_sha256": sha(args.dataset),
            "chunks_sha256": signature,
            "profile": args.profile,
            "method": args.method,
            "configuration": config.as_dict(),
            "fingerprint": config.fingerprint,
            "ranking": {
                "candidate_k": 20,
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
                "rrf_constant": 60,
                "reranker_model": RERANK_MODEL if reranker else None,
                "reranker_revision": RERANK_REVISION if reranker else None,
            },
            "device": "cpu",
            "python": platform.python_version(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "libraries": {
                n: version(n)
                for n in [
                    "torch",
                    "numpy",
                    "qdrant-client",
                    "sentence-transformers",
                    "transformers",
                    "huggingface-hub",
                ]
            },
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "startup_ms": {"index_open": opening_ms, "cached_model_load": model_load_ms},
            "resources": {
                "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024**2 if sys.platform == "darwin" else 1024),
                "index_bytes": disk,
                "encoder_weights_bytes": cached_weights_bytes(config.model, config.revision)
                if args.method != "bm25"
                else 0,
                "reranker_weights_bytes": cached_weights_bytes(RERANK_MODEL, RERANK_REVISION)
                if reranker
                else 0,
                "rss_scope": "whole process high-water RSS: imports, cached models, warmup",
            },
            "code_sha256": {
                n: sha(Path(__file__).with_name(n))
                for n in [
                    "evaluation.py",
                    "index.py",
                    "ingestion.py",
                    "embeddings.py",
                    "config.py",
                    "profiles.py",
                    "retrieval.py",
                    "optimization.py",
                ]
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    lang: {
                        **s["answerable"],
                        "p50_ms": s["latency_ms"]["p50"],
                        "p95_ms": s["latency_ms"]["p95"],
                    }
                    for lang, s in result["summary"].items()
                },
                indent=2,
            )
        )
        return 0
    except Exception:
        print(
            "Error: offline experiment failed; check dataset, PDF, profile, index and cache",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
