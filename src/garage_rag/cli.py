"""Four explicit commands, no network generation in index/search/stats."""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .config import IndexConfig
from .embeddings import E5Embeddings
from .generation import GenerationError, answer, make_llm
from .index import Index


def parser():
    cli = argparse.ArgumentParser(description="GarageRAG: local automotive document retrieval")
    cli.add_argument(
        "--index-dir", type=Path, default=Path(os.environ.get("GARAGE_RAG_INDEX", "data/qdrant"))
    )
    cli.add_argument("--chunk-tokens", type=int, default=384)
    cli.add_argument("--overlap-tokens", type=int, default=48)
    cli.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    commands = cli.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("index", help="Index a PDF or resume its interrupted revision")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--title")
    ingest.add_argument("--language", default="en")
    ingest.add_argument("--models", default="", help="Comma-separated model names")
    ingest.add_argument("--kind", default="owner_manual")
    ingest.add_argument("--source-url")
    ingest.add_argument("--batch-size", type=int, default=16)
    for name in ("search", "ask"):
        command = commands.add_parser(
            name,
            help=(
                "Local retrieval only"
                if name == "search"
                else "EN/DE/RU answer with verified citations"
            ),
        )
        command.add_argument("question")
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--min-score", type=float)
        command.add_argument("--json", action="store_true")
        if name == "ask":
            command.add_argument("--model", help="openrouter/free or provider/model:free")
            command.add_argument(
                "--answer-language",
                choices=["auto", "en", "de", "ru"],
                default="auto",
                help="Answer language (default: auto detects the question; English fallback)",
            )
    commands.add_parser("stats", help="Read active index statistics without loading E5")
    return cli


def _json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv=None):
    # Explicit local .env only: do not search parent directories for credentials.
    load_dotenv(Path.cwd() / ".env", override=False)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Third-party debug output can contain request payloads and credentials.
    logging.disable(logging.CRITICAL)
    args = parser().parse_args(argv)
    try:
        config = IndexConfig(chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens)
        if args.command != "index" and not (args.index_dir / "manifest.json").exists():
            raise ValueError("No index found; run garage-rag index first")
        # Check credentials/model before loading E5 or opening storage for ask.
        llm = make_llm(model=args.model) if args.command == "ask" else None
        try:
            with Index(args.index_dir, config) as index:
                if args.command == "stats":
                    _json(index.stats())
                    return 0
                embeddings = E5Embeddings(config, args.device, getattr(args, "batch_size", 16))
                if args.command == "index":
                    metadata = {
                        "language": args.language,
                        "document_type": args.kind,
                        "vehicle_models": [m.strip() for m in args.models.split(",") if m.strip()],
                    }
                    if args.title:
                        metadata["title"] = args.title
                    if args.source_url:
                        metadata["source_url"] = args.source_url
                    print(
                        "Indexing locally; private text is stored only in the index.",
                        file=sys.stderr,
                    )
                    _json(
                        index.ingest(
                            args.pdf, embeddings, metadata=metadata, batch_size=args.batch_size
                        )
                    )
                    return 0
                hits = index.search(args.question, embeddings, args.top_k, args.min_score)
                if args.command == "search":
                    if args.json:
                        _json(
                            [
                                {
                                    "score": h.score,
                                    "text": h.document.page_content,
                                    "metadata": h.document.metadata,
                                }
                                for h in hits
                            ]
                        )
                    else:
                        for i, hit in enumerate(hits, 1):
                            meta = hit.document.metadata
                            print(
                                f"\n#{i} score={hit.score:.4f} | {meta['document']} | "
                                f"PDF page {meta['pdf_page']} (label {meta['page_label']})"
                            )
                            print(hit.document.page_content)
                        if not hits:
                            print("No matching passages in the active index.")
                else:
                    result = answer(args.question, hits, llm, answer_language=args.answer_language)
                    if args.json:
                        _json(asdict(result))
                    else:
                        print(result.text)
                        if result.reason == "invalid_model_output":
                            print(
                                "Model output failed evidence/format validation.", file=sys.stderr
                            )
                    if result.reason == "output_token_limit":
                        print(
                            "Model reached its output token limit. Increase OPENROUTER_MAX_TOKENS "
                            "or choose another explicit :free model; no automatic retry was made.",
                            file=sys.stderr,
                        )
        finally:
            if llm is not None:
                llm.client.__exit__(None, None, None)
        return 0
    except GenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        # These errors come from local validation. Never print third-party request errors.
        if isinstance(exc, ValueError) and type(exc) is not ValueError:
            print(
                "Error: invalid data or configuration; check PDF/index settings.", file=sys.stderr
            )
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; rerun the identical index command to resume.", file=sys.stderr)
        return 130
    except Exception:
        print(
            "Error: local operation failed. Check PDF access, model cache, disk space and "
            "that no other process has the index open. Re-run indexing to resume.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
