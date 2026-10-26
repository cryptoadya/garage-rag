"""Build a PRIVATE, self-contained answer/source review packet without API calls."""

import argparse
import html
import json
from pathlib import Path

from .config import digest


def write_packet(report, captures, directory):
    keys = ("id", "language", "mode", "trial", "answer_sha256")
    lookup = {tuple(c["key"][k] for k in keys): c for c in captures}
    sections, claim_rows, source_rows = [], [], []
    for row in report["rows"]:
        capture = lookup.get(tuple(row[k] for k in keys))
        if capture is None or digest(capture["answer"]) != row["answer_sha256"]:
            raise ValueError("Mismatched private capture")
        calls = capture.get("model_calls", [])
        observed = []
        for call, public in zip(calls, row.get("model_calls", []), strict=True):
            if digest(call["messages"]) != public["request_sha256"]:
                raise ValueError("Mismatched private capture request")
            payload = json.loads(call["messages"][1]["content"])
            if digest(payload["sources"]) != public["sources_sha256"]:
                raise ValueError("Mismatched private capture sources")
            if (
                "response_sha256" in public
                and digest(call.get("response")) != public["response_sha256"]
            ) or ("response_sha256" not in public and "response" in call):
                raise ValueError("Mismatched private capture response")
            sources = {
                "sources_sha256": digest(payload["sources"]),
                "source_count": len(payload["sources"]),
            }
            if "draft" not in payload and sources not in observed:
                observed.append(sources)
        if observed != row.get("sent_contexts", []):
            raise ValueError("Mismatched private capture contexts")
        name = " / ".join(str(row[k]) for k in keys[:4])

        def pre(value):
            return "<pre>" + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"

        sections.append(
            "<section><h2>"
            + html.escape(name)
            + "</h2><h3>Final answer</h3>"
            + pre(capture["answer"])
            + "<h3>Final candidate and context</h3>"
            + pre({k: capture[k] for k in ("draft", "context")})
            + "<details><summary>Every actual model input and raw reply</summary>"
            + pre(calls)
            + "</details><h3>Public observations and hashes</h3>"
            + pre({
                k: row[k]
                for k in (
                    *keys, "supported", "reason", "error", "error_category", "returned_claims",
                    "trace", "sent_contexts", "model_calls",
                )
                if k in row
            })
            + "</section>"
        )
        if not row["error"]:
            key = {k: row[k] for k in keys}
            claim_rows.append(
                {**key, "claim_support": [None] * row["returned_claims"], "complete_answer": None}
            )
            if row.get("sent_contexts"):
                source_rows.append(
                    {
                        **key,
                        "contexts": [
                            {"sources_sha256": c["sources_sha256"], "sufficiency": None}
                            for c in row["sent_contexts"]
                        ],
                    }
                )
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [
        directory / n
        for n in ("index.html", "claim-review.template.json", "source-review.template.json")
    ]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Review packet already exists")
    page = """<!doctype html><meta charset="utf-8"><title>Private GarageRAG review</title>
<style>body{font:16px system-ui;max-width:1050px;margin:32px auto;padding:0 20px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f3f5;padding:16px}
section{border-top:2px solid #555;margin-top:40px}summary{cursor:pointer}</style>
<h1>Private answer/source review</h1>
<p>This packet contains private passages and model replies. Keep it local.
Existing agent judgments are not independent human review. The reviewer must
make their own judgments before viewing any agent-scored report.</p>
<ol><li>Read the question and each actual draft source set. Decide whether it
supports a complete, unambiguous answer with all relevant conditions. A page hit
alone is insufficient; missing table layout can make a set insufficient.</li>
<li>For every returned claim, check entailment, applicability, numbers, units,
translation, exact quotation and source/page mapping against the original PDF.
Judge completeness separately. For a refusal, distinguish missing evidence from
over-abstention or policy/validation rejection. Inspect earlier drafts separately.</li>
<li>Fill claim-review.template.json booleans and source-review.template.json
sufficient/insufficient values. Remove unresolved rows rather than guessing.
Set reviewer identity and independent_human_review=true ONLY after actual
independent human review. Preserve all hashes. Rescore offline using
--report-input, --review-file and --source-review-file.</li></ol>
<p>PDF page numbers are physical, one-based. Obtain the original privately from
the corpus owner and verify the report corpus SHA256. No vehicle applicability
or vehicle test is established by this review.</p>"""
    outputs[0].write_text(page + "".join(sections))
    base = {
        "report_dataset_sha256": report["dataset_sha256"],
        "reviewer": "",
        "independent_human_review": False,
    }
    outputs[1].write_text(json.dumps({**base, "rows": claim_rows}, indent=2) + "\n")
    outputs[2].write_text(
        json.dumps({"schema_version": 1, **base, "rows": source_rows}, indent=2) + "\n"
    )


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report", type=Path, required=True)
    cli.add_argument("--captures", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args(argv)
    if not args.output_dir.resolve().is_relative_to(Path("output/private").resolve()):
        cli.error("Review packets must stay under ignored output/private/")
    write_packet(
        json.loads(args.report.read_text()), json.loads(args.captures.read_text()), args.output_dir
    )


if __name__ == "__main__":
    main()
