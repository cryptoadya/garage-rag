"""Offline RU/EN query comparison on the frozen BMW corpus; no provider calls.

Run from the repository root with the pinned local embedding cache. English
counterparts are controlled translations from the existing datasets, not newly
sampled Groq translations. Only rankings/metrics, never PDF text, are exported.
"""

import argparse
import hashlib
import json
from pathlib import Path

from garage_rag.embeddings import E5Embeddings
from garage_rag.evaluation import ranking_metrics
from garage_rag.index import Index
from garage_rag.retrieval import AnswerSearch

DATASETS = (
    "benchmarks/bmw_e39_2002.v1.json",
    "benchmarks/bmw_e39_2002.phase2b-dev.v1.json",
    "benchmarks/bmw_e39_2002.phase2b-heldout.v1.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/qdrant"))
    parser.add_argument("--pdf", type=Path, default=Path("data/private/bmw_e39_2002_manual.pdf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sha = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    rows = []
    embeddings = E5Embeddings()
    with Index(args.index_dir) as index:
        before = index.manifest_path.read_bytes()
        documents = list(index.manifest["documents"].values())
        if len(documents) != 1 or documents[0]["content_sha256"] != sha:
            raise ValueError("Use an unchanged baseline index of only the specified PDF")
        search = AnswerSearch(index, embeddings)
        languages = search.source_languages
        for path in DATASETS:
            dataset = json.loads(Path(path).read_text())
            if dataset["source"]["sha256"] != sha:
                raise ValueError("Benchmark PDF hash mismatch")
            for case in dataset["cases"]:
                if case["kind"] != "answerable":
                    continue
                ru, en = (case["questions"][lang] for lang in ("ru", "en"))
                variants = {"old_en": en, "pr_original_ru": ru, "expanded": (ru, en)}
                rankings = {}
                metrics = {}
                for name, queries in variants.items():
                    hits = search(queries, 20)
                    rankings[name] = [[h.document.metadata["pdf_page"],
                                       h.document.metadata["chunk_index"], round(h.score, 8)]
                                      for h in hits]
                    metrics[name] = ranking_metrics([h[0] for h in rankings[name]],
                                                    case["gold_pages"])
                rows.append({"dataset": path, "id": case["id"],
                             "gold_pages": case["gold_pages"], "rankings": rankings,
                             "metrics": metrics})
        if before != index.manifest_path.read_bytes():
            raise ValueError("Index manifest changed")
        config = index.config.as_dict()
    summary = {name: {metric: round(sum(r["metrics"][name][metric] for r in rows) / len(rows), 8)
                      for metric in ("hit@1", "hit@5", "hit@10", "mrr@10")}
               for name in variants}
    losses = {name: [r["id"] for r in rows if r["metrics"]["old_en"]["hit@5"]
                    and not r["metrics"][name]["hit@5"]]
              for name in ("pr_original_ru", "expanded")}
    result = {"schema_version": 1, "pdf_sha256": sha, "index_config": config,
              "source_languages": languages, "n": len(rows),
              "english_queries": "Frozen benchmark equivalents; not fresh Groq output",
              "ranking_entry": ["physical_pdf_page", "chunk_index", "raw_cosine"],
              "provider_calls": 0, "summary": summary, "lost_old_hit_at_5": losses}
    # One row per line keeps the complete, source-text-free rankings reviewable.
    prefix = json.dumps(result, ensure_ascii=False, indent=2)[:-2]
    contents = prefix + ',\n  "rows": [\n' + ',\n'.join(
        "    " + json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
    ) + '\n  ]\n}\n'
    args.output.write_text(contents)
    print(json.dumps({"n": len(rows), "source_languages": languages,
                      "summary": summary, "lost_old_hit_at_5": losses}, indent=2))


if __name__ == "__main__":
    main()
