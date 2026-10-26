"""Separate paired answer evaluation. Offline by default; live calls require a budget.

Public reports allowlist numeric observations and decisions, never source/answer
text. Scripted labels exercise software behavior; live claim quality is unscored
until explicit hash-bound review, never scored by the workflow's own auditor.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from .config import digest
from .evaluation import distribution
from .index import SearchHit
from .source_evaluation import ObservedEvaluationLLM, apply_source_reviews
from .workflow import run_direct, run_workflow

DEFAULT_DATASET = Path("benchmarks/aurora-grounded-answering.v1.json")


class BudgetedPacedLLM:
    """Sequential live evaluator guard, shared by drafts and audits in both modes."""

    def __init__(self, llm, *, max_api_calls, requests_per_minute=15):
        if type(requests_per_minute) is not int or not 1 <= requests_per_minute <= 15:
            raise ValueError("requests_per_minute must be between 1 and 15")
        if type(max_api_calls) is not int or max_api_calls < 1:
            raise ValueError("max_api_calls must be a positive integer")
        self.llm = llm
        self.max_api_calls = max_api_calls
        self.requests_per_minute = requests_per_minute
        self.calls = 0
        self.last_started = None

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def invoke(self, messages):
        from .generation import GenerationError

        if self.calls >= self.max_api_calls:
            raise GenerationError("Live evaluation API budget exhausted")
        wait_started = time.monotonic()
        if self.last_started is not None:
            remaining = (
                60 / self.requests_per_minute + 0.01 - (time.monotonic() - self.last_started)
            )
            while remaining > 0:
                time.sleep(remaining)
                remaining = (
                    60 / self.requests_per_minute + 0.01 - (time.monotonic() - self.last_started)
                )
        self.last_started = time.monotonic()
        self.calls += 1  # Failed provider attempts also consume the session budget.
        waited = (self.last_started - wait_started) * 1000
        response = self.llm.invoke(messages)
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            metadata["limiter_wait_ms"] = waited
        return response


def validate_live_budget(*, pairs, max_api_calls):
    if type(max_api_calls) is not int or max_api_calls < pairs * 5:
        raise ValueError(
            f"Explicit live budget must cover at least {pairs * 5} API calls; "
            "one direct and at most four grounded calls per pair"
        )


def _selection(cases, languages, case_ids, trials):
    if (
        not languages
        or len(set(languages)) != len(languages)
        or not set(languages) <= {"en", "de", "ru"}
    ):
        raise ValueError("Choose unique EN/DE/RU languages")
    if type(trials) is not int or not 1 <= trials <= 10:
        raise ValueError("trials must be between 1 and 10")
    if case_ids and (
        len(set(case_ids)) != len(case_ids) or not set(case_ids) <= {c["id"] for c in cases}
    ):
        raise ValueError("Unknown or duplicate scenario ID")
    return [c for c in cases if not case_ids or c["id"] in case_ids]


def _normal(text):
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


class ScriptedScenarioLLM:
    """Adversarial fixture, not an LLM-quality simulation or token-cost estimate."""

    def __init__(self, scenario, language):
        self.script = scenario["script"]
        self.language = language
        self.drafts = 0

    def invoke(self, messages):
        data = json.loads(messages[1].content)
        if "evidence auditor" in messages[0].content:
            decision = self.script["audit"]
            result = {
                "decision": decision,
                "complete": decision != "insufficient",
                "checks": [
                    {
                        "claim_index": i,
                        "entailed": decision != "unsupported",
                        "applicable": True,
                        "numbers_safe": decision != "unsafe_numeric",
                    }
                    for i in range(len(data["draft"]["claims"]))
                ],
            }
        else:
            expanded = self.drafts > 0 and self.script["expanded"] is not None
            text = self.script["expanded" if expanded else "first"]
            quote = self.script["expanded_quote" if expanded else "quote"]
            self.drafts += 1
            result = {"supported": text is not None, "claims": []}
            if text is not None:
                source = next(s["source_id"] for s in data["sources"] if quote in s["text"])
                result["claims"] = [
                    {
                        "text": text[self.language],
                        "evidence": [{"source_id": source, "quote": quote}],
                    }
                ]
        return SimpleNamespace(content=json.dumps(result, ensure_ascii=False))


def _synthetic_search(scenario, language=None):
    all_text = list(dict.fromkeys([*scenario["initial"], *scenario["expanded"]]))

    def search(k):
        texts = scenario["initial"] if k == 5 else scenario["expanded"]
        return [
            SearchHit(
                Document(
                    page_content=text,
                    metadata={
                        "document": "aurora-synthetic.pdf",
                        "pdf_page": all_text.index(text) + 1,
                        "chunk_index": 0,
                        "document_type": "owner_manual",
                    },
                ),
                1 / (i + 1),
            )
            for i, text in enumerate(texts)
        ]

    return search


def _row(result, case, language, mode, trial, *, scripted):
    claims = (
        result.candidate.draft.claims
        if result.answer.supported and result.candidate and result.candidate.draft
        else []
    )
    support, complete, reference_match = None, None, None
    if "expected" in case:
        answerable = case["expected"] == "answer"
        reference = (case.get("reference") or {}).get(language)
        reference_match = [_normal(c.text) == _normal(reference or "") for c in claims]
        if scripted:
            # Only fixture outputs have exhaustive, exact reference labels.
            support = reference_match
            complete = bool(claims) and all(support)
    else:
        evidence = case.get("evidence")
        answerable = (
            False
            if case["kind"] == "unanswerable"
            else not evidence["requires_layout"]
            if evidence
            else None
        )
    row = {
        "id": case["id"],
        "language": language,
        "mode": mode,
        "trial": trial,
        "supported": result.answer.supported,
        "reason": result.answer.reason,
        "error": result.error is not None,
        "error_category": _error_category(result.error),
        "expected_answerable": answerable,
        "returned_claims": len(claims),
        "claim_support": support,
        "complete_answer": complete,
        "exact_reference_match": reference_match,
        "answer_sha256": digest(asdict(result.answer)),
        "trace": result.trace.public_dict(),
    }
    if "gold_pages" in case:
        metadata = [
            result.context.sources[e.source_id].metadata for c in claims for e in c.evidence
        ]
        sufficient = {
            (c["pdf_page"], c["chunk_index"])
            for c in case.get("evidence", {}).get("sufficient_chunks", [])
        }
        row["citation_page_matches"] = [m["pdf_page"] in case["gold_pages"] for m in metadata]
        row["citation_chunk_matches"] = (
            [(m["pdf_page"], m["chunk_index"]) in sufficient for m in metadata]
            if sufficient
            else None
        )
    return row


def summarize(rows):
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        runs = [r for r in rows if r["mode"] == mode]
        scored = [r for r in runs if not r["error"]]
        reviewed = [r for r in scored if r["supported"] and r["claim_support"] is not None]
        supported_claims = sum(sum(r["claim_support"]) for r in reviewed)
        reviewed_claims = sum(len(r["claim_support"]) for r in reviewed)
        summary[mode] = {
            "runs": len(runs),
            "errors": len(runs) - len(scored),
            "scored_runs": len(scored),
            "answerable_runs": sum(r["expected_answerable"] is True for r in scored),
            "unanswerable_runs": sum(r["expected_answerable"] is False for r in scored),
            "answers": sum(r["supported"] for r in scored),
            "correct_refusals": sum(
                not r["supported"] and r["expected_answerable"] is False for r in scored
            ),
            "false_refusals": sum(
                not r["supported"] and r["expected_answerable"] is True for r in scored
            ),
            "reviewed_answers": len(reviewed),
            "reviewed_claims": reviewed_claims,
            "supported_claims": supported_claims,
            "claim_support_rate": supported_claims / reviewed_claims if reviewed_claims else None,
            "unsupported_answers": sum(not all(r["claim_support"]) for r in reviewed),
            "complete_supported_answers": sum(
                all(r["claim_support"]) and r["complete_answer"] is True for r in reviewed
            ),
            "locally_validated_citation_answers": sum(r["supported"] for r in scored),
            **{
                key: distribution([r["trace"][key] for r in runs])
                for key in ("llm_calls", "search_calls", "elapsed_ms")
            },
            "total_llm_calls": sum(r["trace"]["llm_calls"] for r in runs),
            "reported_input_tokens": sum(r["trace"]["input_tokens"] for r in runs),
            "reported_output_tokens": sum(r["trace"]["output_tokens"] for r in runs),
            "token_usage_complete": all(
                r["trace"]["usage_records"] == r["trace"]["llm_calls"] for r in runs
            ),
        }
    return summary


def paired_summary(rows):
    pairs = {}
    for row in rows:
        key = (row["id"], row["language"], row["trial"])
        pairs.setdefault(key, {})[row["mode"]] = row
    complete = [
        row
        for pair in pairs.values()
        if set(pair) == {"direct", "grounded"} and not any(r["error"] for r in pair.values())
        for row in pair.values()
    ]
    return {"pairs": len(complete) // 2, "summary": summarize(complete)}


def _error_category(message):
    if not message:
        return None
    if "rate limit" in message or message.startswith(("Groq local", "Groq reported", "Groq quota")):
        return "rate_limit"
    if "authentication" in message:
        return "authentication"
    if "free model is unavailable" in message or "Groq model unavailable" in message:
        return "free_model_unavailable"
    if "OpenRouter request failed" in message or message.startswith("Groq request failed"):
        return "provider_or_network_error"
    return "local_or_workflow_error"


def apply_reviews(report, reviews):
    """Apply external judgments bound to exact outputs, without publishing answer text."""
    if (
        reviews.get("report_dataset_sha256") != report["dataset_sha256"]
        or type(reviews.get("independent_human_review")) is not bool
        or not isinstance(reviews.get("reviewer"), str)
        or not reviews["reviewer"].strip()
        or not isinstance(reviews.get("rows"), list)
    ):
        raise ValueError("Invalid review provenance")
    lookup = {(r["id"], r["language"], r["mode"], r["trial"]): r for r in report["rows"]}
    seen = set()
    updates = []
    for review in reviews["rows"]:
        key = tuple(review[k] for k in ("id", "language", "mode", "trial"))
        row = lookup.get(key)
        if (
            row is None
            or key in seen
            or row["answer_sha256"] != review["answer_sha256"]
            or row["error"]
            or type(review.get("complete_answer")) is not bool
            or not isinstance(review.get("claim_support"), list)
            or len(review["claim_support"]) != row["returned_claims"]
            or not all(type(v) is bool for v in review["claim_support"])
        ):
            raise ValueError("Invalid or mismatched review row")
        seen.add(key)
        updates.append((row, review))
    for row, review in updates:
        row["claim_support"] = review["claim_support"]
        row["complete_answer"] = review["complete_answer"]
        row["claim_review_provenance"] = {
            k: reviews[k] for k in ("reviewer", "independent_human_review")
        }
    report["claim_scoring"] = "Hash-bound external review; unreviewed claims remain unscored"
    report["review"] = {k: reviews[k] for k in ("reviewer", "independent_human_review")}
    scored = [r for r in report["rows"] if not r["error"] and r["claim_support"] is not None]
    report["review"]["independent_human_review"] = bool(scored) and all(
        r.get("claim_review_provenance", {}).get("independent_human_review") is True
        for r in scored
    )
    report["review"]["rows_reviewed"] = len(seen)
    report["summary"] = summarize(report["rows"])
    report["paired"] = paired_summary(report["rows"])
    return report


def _evaluate(
    data,
    cases,
    languages,
    trials,
    seed,
    llm,
    search_factory,
    *,
    scripted,
    previous=None,
    previous_captures=None,
    checkpoint=None,
):
    previous = previous or {}
    rows = [r for r in previous.get("rows", []) if not r["error"]]
    captures = list(previous_captures or [])
    previous_errors = [
        *previous.get("previous_errors", []),
        *[r for r in previous.get("rows", []) if r["error"]],
    ]
    done = {(r["id"], r["language"], r["mode"], r["trial"]) for r in rows}
    schedule = [
        (c, language, trial)
        for c in cases
        for language in languages
        for trial in range(1, trials + 1)
    ]
    rng = random.Random(seed)
    rng.shuffle(schedule)
    stopped = False

    def snapshot():
        report = {
            "schema_version": 1,
            "dataset_id": data["id"],
            "dataset_sha256": digest(data),
            "execution": "scripted" if scripted else "live",
            "complete": len(rows) == len(schedule) * 2 and not any(r["error"] for r in rows),
            "planned_runs": len(schedule) * 2,
            "seed": seed,
            "trials": trials,
            "languages": languages,
            "independent_human_review": False,
            "claim_scoring": (
                "exhaustive scripted references" if scripted else "unscored until review"
            ),
            "summary": summarize(rows),
            "paired": paired_summary(rows),
            "rows": list(rows),
            "previous_errors": previous_errors,
        }
        if checkpoint:
            checkpoint(report, captures)
        return report

    snapshot()  # Durable empty checkpoint before the first paid/quota-consuming operation.
    for case, language, trial in schedule:
        modes = ["direct", "grounded"]
        rng.shuffle(modes)
        for mode in modes:
            if (case["id"], language, mode, trial) in done:
                continue
            model = ScriptedScenarioLLM(case, language) if scripted else llm
            runner = run_direct if mode == "direct" else run_workflow
            observed = ObservedEvaluationLLM(model)
            result = runner(
                case["questions"][language],
                search_factory(case, language),
                observed,
                answer_language=language,
            )
            row = _row(result, case, language, mode, trial, scripted=scripted)
            row["sent_contexts"] = observed.sent_contexts
            row["model_calls"] = observed.public_calls
            rows.append(row)
            captures.append(
                {
                    "key": {
                        k: row[k] for k in ("id", "language", "mode", "trial", "answer_sha256")
                    },
                    "model_calls": observed.calls,
                    "answer": asdict(result.answer),
                    "error": result.error,
                    "context": json.loads(result.context.text) if result.context else None,
                    "draft": (
                        result.candidate.draft.model_dump()
                        if result.candidate and result.candidate.draft
                        else None
                    ),
                }
            )
            if not scripted:
                print(
                    f"{case['id']}/{language}/{mode}: {result.answer.reason}; "
                    f"{result.trace.llm_calls} calls"
                    + (f"; {_error_category(result.error)}" if result.error else ""),
                    file=sys.stderr,
                    flush=True,
                )
            snapshot()  # Persist each completed mode, including the first failure.
            if result.error:
                stopped = True  # Never hammer a provider after a rate limit/failure.
                break
        if stopped:
            break
    return snapshot(), captures


def evaluate_synthetic(
    data, *, languages=None, case_ids=None, trials=1, seed=20261025, llm=None, **execution
):
    languages = languages or ["en", "de", "ru"]
    if data.get("suite") != "synthetic_grounded_answering" or data.get("schema_version") != 1:
        raise ValueError("Invalid synthetic answer dataset")
    cases = _selection(data["scenarios"], languages, case_ids, trials)
    return _evaluate(
        data,
        cases,
        languages,
        trials,
        seed,
        llm,
        _synthetic_search,
        scripted=llm is None,
        **execution,
    )


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_atomic(path, value):
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    cli.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmarks/answers.json"),
        help="New local report (default: output/benchmarks/answers.json)",
    )
    cli.add_argument("--languages", nargs="+", default=["en", "de", "ru"])
    cli.add_argument("--cases", nargs="+")
    cli.add_argument("--trials", type=int, default=1)
    cli.add_argument("--seed", type=int, default=20261025)
    cli.add_argument("--live", action="store_true")
    cli.add_argument("--max-api-calls", type=int, default=0)
    cli.add_argument("--requests-per-minute", type=int, default=15, choices=range(1, 16))
    cli.add_argument("--provider", choices=["openrouter", "groq"])
    cli.add_argument("--model", help="Explicit model; Groq defaults to openai/gpt-oss-120b")
    cli.add_argument(
        "--resume", action="store_true", help="Resume identical checkpoint; skip successes"
    )
    cli.add_argument(
        "--index-dir", type=Path, help="Existing matching corpus index for Phase 2 datasets"
    )
    cli.add_argument("--profile", choices=["baseline", "e5-base", "bge-m3"], default="baseline")
    cli.add_argument("--private-captures", type=Path, help="Optional path under output/private/")
    cli.add_argument(
        "--report-input", type=Path, help="Rescore an existing report without API calls"
    )
    cli.add_argument("--review-file", type=Path, help="Hash-bound claim reviews for --report-input")
    cli.add_argument("--source-review-file", type=Path, help="Additive sent-source reviews")
    args = cli.parse_args(argv)
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    logging.disable(logging.CRITICAL)
    for k, value in {
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ[k] = value
    os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache/huggingface"))
    try:
        if not args.resume and (
            args.output.exists() or (args.private_captures and args.private_captures.exists())
        ):
            raise ValueError("Output already exists; choose new result paths")
        if args.resume and (
            not args.output.exists()
            or args.report_input
            or args.review_file
            or args.source_review_file
        ):
            raise ValueError("Resume requires an existing checkpoint and prohibits rescoring")
        if args.report_input:
            if not (args.review_file or args.source_review_file) or args.live:
                raise ValueError("Rescoring requires reviews and prohibits live calls")
            report = json.loads(args.report_input.read_text())
            if args.review_file:
                apply_reviews(report, json.loads(args.review_file.read_text()))
            if args.source_review_file:
                apply_source_reviews(report, json.loads(args.source_review_file.read_text()))
            _write_new(args.output, report)
            return 0
        if args.review_file or args.source_review_file:
            raise ValueError("Reviews require --report-input")
        if args.private_captures and not args.private_captures.resolve().is_relative_to(
            Path("output/private").resolve()
        ):
            raise ValueError("Private captures must stay under ignored output/private/")
        if args.private_captures and args.output.resolve() == args.private_captures.resolve():
            raise ValueError("Report and captures must use separate paths")
        data = json.loads(args.dataset.read_text())
        synthetic = data.get("suite") == "synthetic_grounded_answering"
        cases = _selection(
            data["scenarios" if synthetic else "cases"], args.languages, args.cases, args.trials
        )
        pairs = len(cases) * len(args.languages) * args.trials
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env", override=False)
        provider = args.provider or os.environ.get("GARAGE_RAG_PROVIDER", "openrouter")
        selected_model = (
            args.model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
            if provider == "groq"
            else args.model
        )
        identity = {
            "dataset_sha256": digest(data),
            "cases": [c["id"] for c in cases],
            "languages": args.languages,
            "trials": args.trials,
            "seed": args.seed,
            "execution": "live" if args.live else "scripted",
            "provider": provider if args.live else None,
            "model": selected_model if args.live else None,
            "profile": args.profile,
            "source_sha256": source_hashes,
            "index_sha256": (
                digest(json.loads((args.index_dir / "manifest.json").read_text()))
                if args.index_dir and not synthetic
                else None
            ),
            "groq_token_accounting": os.environ.get("GROQ_TOKEN_ACCOUNTING", "bytes")
            if args.live and provider == "groq"
            else None,
            "max_tokens": (
                os.environ.get("GROQ_MAX_TOKENS", "2048")
                if provider == "groq"
                else os.environ.get("OPENROUTER_MAX_TOKENS", "8192")
            )
            if args.live
            else None,
            "private_captures": str(args.private_captures.resolve())
            if args.private_captures
            else None,
            "requests_per_minute": args.requests_per_minute if args.live else None,
        }
        previous = json.loads(args.output.read_text()) if args.resume else None
        if previous and previous.get("identity") != identity:
            raise ValueError(
                "Resume configuration/model/dataset/code mismatch; choose separate output"
            )
        previous_captures = []
        if previous and args.private_captures:
            previous_captures = json.loads(args.private_captures.read_text())
            capture_keys = {
                tuple(c["key"][k] for k in ("id", "language", "mode", "trial", "answer_sha256"))
                for c in previous_captures
            }
            if any(
                tuple(r[k] for k in ("id", "language", "mode", "trial", "answer_sha256"))
                not in capture_keys
                for r in previous["rows"]
            ):
                raise ValueError("Resume captures do not match checkpoint")
        if previous and previous["complete"]:
            print("Checkpoint already complete; no API calls")
            return 0
        done_pairs = {
            (r["id"], r["language"], r["trial"])
            for r in (previous or {}).get("rows", [])
            if not r["error"]
        }
        # Reserve five calls for every still incomplete pair, even if one mode is done.
        completed_pairs = (
            sum(
                sum(
                    not r["error"] and (r["id"], r["language"], r["trial"]) == key
                    for r in previous["rows"]
                )
                == 2
                for key in done_pairs
            )
            if previous
            else 0
        )
        llm = index = None
        if args.live:
            validate_live_budget(pairs=pairs - completed_pairs, max_api_calls=args.max_api_calls)
            if provider == "openrouter" and (
                not selected_model or not selected_model.endswith(":free")
            ):
                raise ValueError(
                    "Choose one explicit :free model for a reproducible live comparison"
                )
            from .generation import make_llm

            llm = BudgetedPacedLLM(
                make_llm(model=selected_model, provider=provider),
                max_api_calls=args.max_api_calls,
                requests_per_minute=args.requests_per_minute,
            )
            if provider == "groq" and previous:
                llm.restore_quota(previous.get("provenance", {}).get("quota_state", {}))
            print(
                f"Live {provider}/{selected_model}: <= {args.max_api_calls} attempts; "
                "no retries/fallback; stop on first error.",
                file=sys.stderr,
                flush=True,
            )
        elif not synthetic:
            raise ValueError("Corpus answer evaluation requires --live and an explicit API budget")
        extra = {}

        def checkpoint(report, captures):
            report.update(extra)
            report["identity"] = identity
            report["provenance"] = {
                "created_at": datetime.now(UTC).isoformat(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "provider": provider if args.live else None,
                "model": selected_model if args.live else None,
                "max_api_calls": args.max_api_calls if args.live else 0,
                "generation": {
                    "max_tokens": llm.max_tokens,
                    "temperature": llm.temperature,
                    "network_timeout_ms": 60000,
                    "free_only": True,
                    "retries": 0,
                    "free_plan_attested": provider == "groq",
                    "requests_per_minute": llm.requests_per_minute,
                    "attempted_api_calls": llm.calls,
                    "previous_attempted_api_calls": (
                        (previous or {}).get("provenance", {}).get("generation") or {}
                    ).get("attempted_api_calls", 0)
                    + ((previous or {}).get("provenance", {}).get("generation") or {}).get(
                        "previous_attempted_api_calls", 0
                    ),
                }
                if llm
                else None,
                "quota_state": llm.quota_state() if llm and provider == "groq" else None,
                "libraries": {n: version(n) for n in ("langgraph", "langchain-core", "garage-rag")},
                "source_sha256": source_hashes,
            }
            if args.private_captures:
                _write_atomic(args.private_captures, captures)
            _write_atomic(args.output, report)
            if llm and provider == "groq":
                llm.llm.quota_observer = lambda: checkpoint(report, captures)

        execution = dict(
            previous=previous, previous_captures=previous_captures, checkpoint=checkpoint
        )
        try:
            if synthetic:
                report, captures = evaluate_synthetic(
                    data,
                    languages=args.languages,
                    case_ids=args.cases,
                    trials=args.trials,
                    seed=args.seed,
                    llm=llm,
                    **execution,
                )
            else:
                from .embeddings import E5Embeddings
                from .evaluation import validate_corpus, validate_dataset
                from .index import Index
                from .optimization import chunk_digest, validate_evidence
                from .profiles import profile_config

                validate_dataset(data)
                validate_evidence(data)
                if not args.index_dir or not (args.index_dir / "manifest.json").exists():
                    raise ValueError("Existing corpus index required")
                config = profile_config(args.profile)
                index = Index(args.index_dir, config)
                validate_corpus(index, data)
                if chunk_digest(index) != data["design"]["chunks_sha256"]:
                    raise ValueError("Frozen evidence chunk digest does not match")
                embeddings = E5Embeddings(config, "cpu")
                # Excluded LOCAL warmup, no LLM or query text rewriting.
                index.search(cases[0]["questions"][args.languages[0]], embeddings)

                def corpus_search(case, language):
                    def search(k):
                        return index.search(case["questions"][language], embeddings, top_k=k)

                    return search

                extra.update(
                    profile=args.profile,
                    index_fingerprint=config.fingerprint,
                    corpus_sha256=data["source"]["sha256"],
                    chunks_sha256=data["design"]["chunks_sha256"],
                )
                report, captures = _evaluate(
                    data,
                    cases,
                    args.languages,
                    args.trials,
                    args.seed,
                    llm,
                    corpus_search,
                    scripted=False,
                    **execution,
                )
                report["profile"] = args.profile
                report["index_fingerprint"] = config.fingerprint
                report["corpus_sha256"] = data["source"]["sha256"]
                report["chunks_sha256"] = data["design"]["chunks_sha256"]
        finally:
            if index is not None:
                index.__exit__(None, None, None)
            if llm is not None:
                llm.client.__exit__(None, None, None)
        print(json.dumps({"complete": report["complete"], "summary": report["summary"]}, indent=2))
        return 0 if report["complete"] else 2
    except ValueError as exc:
        message = str(exc) if type(exc) is ValueError else "Invalid dataset or result JSON"
        print(f"Answer evaluation failed: {message}", file=sys.stderr)
        return 2
    except Exception:
        print(
            "Answer evaluation failed; check dataset, budgets, output paths and local "
            "index/model configuration. Provider data is not printed.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
