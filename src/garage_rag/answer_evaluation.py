"""Matched answer evaluation. Offline by default; live calls require explicit budgets.

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
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from .config import digest
from .evaluation import distribution
from .index import SearchHit
from .source_evaluation import (
    EvaluationBinding,
    ObservedEvaluationLLM,
    apply_source_reviews,
)
from .workflow import run_direct, run_workflow

DEFAULT_DATASET = Path("benchmarks/aurora-grounded-answering.v1.json")
MODE_CALL_CAPS = {"direct": 1, "grounded": 4, "agentic": 6}


def _modes(modes=None):
    modes = ["direct", "grounded"] if modes is None else list(modes)
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= MODE_CALL_CAPS.keys():
        raise ValueError("Choose unique direct/grounded/agentic modes")
    return modes


class BudgetedPacedLLM:
    """Sequential live guard, shared by planning, writing and auditing in every mode."""

    def __init__(self, llm, *, max_api_calls, requests_per_minute=15, max_total_tokens=None):
        if type(requests_per_minute) is not int or not 1 <= requests_per_minute <= 15:
            raise ValueError("requests_per_minute must be between 1 and 15")
        if type(max_api_calls) is not int or max_api_calls < 1:
            raise ValueError("max_api_calls must be a positive integer")
        self.llm = llm
        self.max_api_calls = max_api_calls
        self.requests_per_minute = requests_per_minute
        self.calls = 0
        self.last_started = None
        if max_total_tokens is not None and (
            type(max_total_tokens) is not int or max_total_tokens < 1
        ):
            raise ValueError("max_total_tokens must be a positive integer")
        self.max_total_tokens = max_total_tokens
        self.reserved_tokens = 0

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def invoke(self, messages):
        return self._invoke(messages, self.llm)

    def bind_planner(self):
        from .planner import PLANNER_FORMAT

        return EvaluationBinding(self, self.llm.bind_planner(), PLANNER_FORMAT)

    def bind_tools(self, tools, **kwargs):
        return EvaluationBinding(self, self.llm.bind_tools(tools, **kwargs), tools)

    def _invoke(self, messages, target, tools=None):
        from .generation import GenerationError

        if self.calls >= self.max_api_calls:
            raise GenerationError("Live evaluation API budget exhausted")
        reserve = 0
        if self.max_total_tokens is not None:
            from .local_tools import byte_reservation, wire_messages

            reserve_fn = getattr(self.llm, "request_reservation", None)
            reserve = (
                reserve_fn(messages, tools)
                if reserve_fn
                else byte_reservation(wire_messages(messages, tools), self.llm.max_tokens, tools)
            )
            if type(reserve) is not int or reserve <= 0:
                raise GenerationError("Invalid live token reservation")
            if self.reserved_tokens + reserve > self.max_total_tokens:
                raise GenerationError("Live evaluation token budget exhausted")
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
        self.reserved_tokens += reserve  # No refunds, including failed calls/reasoning.
        waited = (self.last_started - wait_started) * 1000
        response = target.invoke(messages)
        usage = getattr(response, "usage_metadata", None)
        if reserve and usage and usage.get("total_tokens", 0) > reserve:
            raise GenerationError("Provider usage exceeded conservative token reservation")
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            metadata["limiter_wait_ms"] = waited
        return response


def validate_live_budget(*, pairs, max_api_calls, modes=None):
    per_case = sum(MODE_CALL_CAPS[m] for m in _modes(modes))
    if type(max_api_calls) is not int or max_api_calls < pairs * per_case:
        raise ValueError(
            f"Explicit live budget must cover at least {pairs * per_case} API calls; "
            f"worst-case {per_case} per selected case/language/trial"
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
        self.scenario = scenario
        self.script = scenario["script"]
        self.language = language
        self.drafts = 0

    provider = "scripted"
    max_tokens = 512

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        from .source_evaluation import capture_messages, request_context

        data = json.loads(messages[1].content)
        phase, sources = request_context(capture_messages(messages))
        if phase == "planner":
            if not any(m.type == "tool" for m in messages):
                name, query, call_id = "dense_search", data["question"], "first"
            elif self.scenario.get("followup") and not all(
                any(t in s["text"] for s in sources) for t in self.scenario["targeted"]
            ):
                name, query, call_id = "lexical_search", self.scenario["followup"], "followup"
            else:
                return AIMessage(
                    content="", tool_calls=[{"id": "finalize", "name": "finalize", "args": {}}]
                )
            return AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": name, "args": {"query": query, "top_k": 5}}],
            )
        if sources:
            data["sources"] = sources
        if "evidence auditor" in messages[0].content:
            decision = self.script["audit"]
            if self.script["expanded"] and any(
                c["text"] in self.script["expanded"].values() for c in data["draft"]["claims"]
            ):
                decision = self.script.get("expanded_audit", decision)
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
            expanded = self.script["expanded"] is not None and (
                all(
                    any(t in s["text"] for s in sources)
                    for t in (self.scenario.get("targeted") or [self.script["expanded_quote"]])
                )
                if "labels" in self.script
                else self.drafts > 0
            )
            text = self.script["expanded" if expanded else "first"]
            quote = self.script["expanded_quote" if expanded else "quote"]
            self.drafts += 1
            result = {"supported": text is not None, "claims": []}
            if text is not None:
                quotes = self.script.get("expanded_quotes", [quote]) if expanded else [quote]
                result["claims"] = [
                    {
                        "text": text[self.language],
                        "evidence": [
                            {
                                "source_id": next(
                                    s["source_id"] for s in data["sources"] if q in s["text"]
                                ),
                                "quote": q,
                            }
                            for q in quotes
                        ],
                    }
                ]
        return SimpleNamespace(content=json.dumps(result, ensure_ascii=False))


def _synthetic_search(scenario, language=None):
    all_text = list(
        dict.fromkeys([*scenario["initial"], *scenario["expanded"], *scenario.get("targeted", [])])
    )

    def search(*args):
        if len(args) == 1:
            texts = scenario["initial"] if args[0] == 5 else scenario["expanded"]
        else:
            method, query, k = args
            texts = (
                scenario["targeted"]
                if method == "bm25" and query == scenario.get("followup")
                else scenario["initial"]
            )[:k]
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
        "error_category": _error_category(result.error, result.trace.provider_errors),
        "expected_answerable": answerable,
        "returned_claims": len(claims),
        "claim_support": support,
        "complete_answer": complete,
        "exact_reference_match": reference_match,
        "answer_sha256": digest(asdict(result.answer)),
        "trace": result.trace.public_dict(),
    }
    if "labels" in case.get("script", {}):
        row.update(
            kind=case["kind"],
            warnings_required=case["warnings_required"],
            citation_support=None,
            warnings_preserved=None,
        )
        if scripted:
            row.update(claim_support=[], complete_answer=False, citation_support=[])
            if claims:
                key = (
                    "expanded"
                    if case["script"]["expanded"]
                    and (claims[0].text == case["script"]["expanded"][language])
                    else "first"
                )
                row.update(case["script"]["labels"][key])
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
            "independent_reviewed_answers": sum(
                r.get("claim_review_provenance", {}).get("independent_human_review") is True
                for r in reviewed
            ),
            "reviewed_citation_claims": sum(len(r.get("citation_support") or []) for r in scored),
            "supported_citation_claims": sum(sum(r.get("citation_support") or []) for r in scored),
            "warning_answers_reviewed": sum(
                r.get("warnings_required") is True and r.get("warnings_preserved") is not None
                for r in scored
                if r["supported"]
            ),
            "missed_warning_answers": sum(
                r.get("warnings_required") is True and r.get("warnings_preserved") is False
                for r in scored
                if r["supported"]
            ),
            "stop_reasons": {
                reason: sum(r.get("reason", "unknown") == reason for r in runs)
                for reason in sorted({r.get("reason", "unknown") for r in runs})
            },
            "timing_ms": {
                key: sum(c.get(key, 0) for r in runs for c in r.get("model_calls", []))
                for key in ("http_elapsed_ms", "limiter_wait_ms", "quota_wait_ms")
            },
            **{
                key: distribution([r["trace"][key] for r in runs])
                for key in ("llm_calls", "search_calls", "elapsed_ms")
            },
            "total_llm_calls": sum(r["trace"]["llm_calls"] for r in runs),
            "total_search_calls": sum(r["trace"]["search_calls"] for r in runs),
            "reported_input_tokens": sum(r["trace"]["input_tokens"] for r in runs),
            "reported_output_tokens": sum(r["trace"]["output_tokens"] for r in runs),
            "token_usage_complete": all(
                r["trace"]["usage_records"] == r["trace"]["llm_calls"] for r in runs
            ),
        }
    return summary


def paired_summary(rows, modes=None):
    modes = set(_modes(modes))
    pairs = {}
    for row in rows:
        key = (row["id"], row["language"], row["trial"])
        pairs.setdefault(key, {})[row["mode"]] = row
    complete = [
        row
        for pair in pairs.values()
        if modes <= set(pair) and not any(pair[m]["error"] for m in modes)
        for mode, row in pair.items()
        if mode in modes
    ]
    return {"pairs": len(complete) // len(modes), "summary": summarize(complete)}


def _refresh_summaries(report):
    rows, modes = report["rows"], report.get("modes", ["direct", "grounded"])
    report["summary"] = summarize(rows)
    report["paired"] = paired_summary(rows, modes)
    if report.get("comparative"):
        report["attempt_summary"] = summarize([*rows, *report.get("previous_errors", [])])
        report["pairwise"] = {
            f"{a}/{b}": paired_summary(rows, [a, b]) for a, b in combinations(modes, 2)
        }
        for field in ("language", "kind"):
            report[f"by_{field}"] = {
                value: summarize([r for r in rows if r[field] == value])
                for value in sorted({r[field] for r in rows})
            }


def _error_category(message, provider_errors=None):
    if not message:
        return None
    # Workflows deliberately replace provider details with a safe generic message.
    # Prefer the structured observation rather than misclassifying that message.
    for error in reversed(provider_errors or []):
        status = error.get("http_status")
        if status == 429:
            return "rate_limit"
        if status in (401, 403):
            return "authentication"
        if status == 404:
            return "free_model_unavailable"
        if status is not None or error.get("category") in {"timeout", "transport_error"}:
            return "provider_or_network_error"
    if "rate limit" in message or message.startswith(("Groq local", "Groq reported", "Groq quota")):
        return "rate_limit"
    if "authentication" in message:
        return "authentication"
    if "free model is unavailable" in message or "Groq model unavailable" in message:
        return "free_model_unavailable"
    if message.startswith("Groq request failed"):
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
            or report.get("comparative")
            and (
                not isinstance(review.get("citation_support"), list)
                or len(review["citation_support"]) != row["returned_claims"]
                or not all(type(v) is bool for v in review["citation_support"])
                or (
                    type(review.get("warnings_preserved")) is not bool
                    if row["supported"] and row["warnings_required"]
                    else review.get("warnings_preserved") is not None
                )
            )
        ):
            raise ValueError("Invalid or mismatched review row")
        seen.add(key)
        updates.append((row, review))
    for row, review in updates:
        row["claim_support"] = review["claim_support"]
        row["complete_answer"] = review["complete_answer"]
        if report.get("comparative"):
            for field in ("citation_support", "warnings_preserved"):
                row[field] = review[field]
        row["claim_review_provenance"] = {
            k: reviews[k] for k in ("reviewer", "independent_human_review")
        }
    report["claim_scoring"] = "Hash-bound external review; unreviewed claims remain unscored"
    report["review"] = {k: reviews[k] for k in ("reviewer", "independent_human_review")}
    scored = [r for r in report["rows"] if not r["error"] and r["claim_support"] is not None]
    report["review"]["independent_human_review"] = bool(scored) and all(
        r.get("claim_review_provenance", {}).get("independent_human_review") is True for r in scored
    )
    report["review"]["rows_reviewed"] = len(seen)
    _refresh_summaries(report)
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
    modes=None,
    previous=None,
    previous_captures=None,
    checkpoint=None,
):
    modes = _modes(modes)
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
            "complete": len(rows) == len(schedule) * len(modes)
            and not any(r["error"] for r in rows),
            "planned_runs": len(schedule) * len(modes),
            "modes": modes,
            "comparative": data.get("comparative", False),
            "seed": seed,
            "trials": trials,
            "languages": languages,
            "independent_human_review": False,
            "claim_scoring": (
                "exhaustive scripted references" if scripted else "unscored until review"
            ),
            "rows": list(rows),
            "previous_errors": previous_errors,
        }
        _refresh_summaries(report)
        if checkpoint:
            checkpoint(report, captures)
        return report

    snapshot()  # Durable empty checkpoint before the first paid/quota-consuming operation.
    for case, language, trial in schedule:
        order = list(modes)
        rng.shuffle(order)
        for mode in order:
            if (case["id"], language, mode, trial) in done:
                continue
            model = ScriptedScenarioLLM(case, language) if scripted else llm
            if mode == "agentic":
                from .agentic import run_agentic

                runner = run_agentic
            else:
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
                    **(
                        {"evidence_store": result.evidence_store.capture()}
                        if getattr(result, "evidence_store", None) is not None
                        else {}
                    ),
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
                    + (f"; {row['error_category']}" if result.error else ""),
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
    if "modes" not in execution and data.get("comparative"):
        execution["modes"] = list(MODE_CALL_CAPS)
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
    cli.add_argument("--modes", nargs="+", choices=list(MODE_CALL_CAPS))
    cli.add_argument("--max-api-calls", type=int, default=0)
    cli.add_argument(
        "--max-total-tokens", type=int, help="Hard session input+output reservation cap"
    )
    cli.add_argument("--requests-per-minute", type=int, default=15, choices=range(1, 16))
    cli.add_argument("--provider", choices=["groq"])
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
        modes = _modes(
            args.modes
            if args.modes is not None
            else (list(MODE_CALL_CAPS) if data.get("comparative") else None)
        )
        cases = _selection(
            data["scenarios" if synthetic else "cases"], args.languages, args.cases, args.trials
        )
        pairs = len(cases) * len(args.languages) * args.trials
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env", override=False)
        provider = args.provider or os.environ.get("GARAGE_RAG_PROVIDER", "groq")
        selected_model = args.model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        if args.live and (data.get("comparative") or "agentic" in modes):
            if type(args.max_total_tokens) is not int or args.max_total_tokens < 1:
                raise ValueError("Comparative/agentic live evaluation requires --max-total-tokens")
            if "agentic" in modes and provider != "groq":
                raise ValueError("Agentic comparison requires the native Groq tool protocol")
            if not args.private_captures:
                raise ValueError("Comparative live evaluation requires --private-captures")
        identity = {
            "dataset_sha256": digest(data),
            "cases": [c["id"] for c in cases],
            "languages": args.languages,
            "trials": args.trials,
            "modes": modes,
            "max_total_tokens": args.max_total_tokens,
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
            "max_tokens": os.environ.get("GROQ_MAX_TOKENS", "2048") if args.live else None,
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
        # Conservatively reserve every selected mode for any incomplete comparison.
        completed_pairs = (
            sum(
                sum(
                    not r["error"] and (r["id"], r["language"], r["trial"]) == key
                    for r in previous["rows"]
                )
                == len(modes)
                for key in done_pairs
            )
            if previous
            else 0
        )
        llm = index = None
        if args.live:
            validate_live_budget(
                pairs=pairs - completed_pairs, max_api_calls=args.max_api_calls, modes=modes
            )
            from .generation import make_llm

            llm = BudgetedPacedLLM(
                make_llm(model=selected_model, provider=provider),
                max_api_calls=args.max_api_calls,
                requests_per_minute=args.requests_per_minute,
                max_total_tokens=args.max_total_tokens,
            )
            if previous:
                llm.reserved_tokens = (previous.get("provenance", {}).get("generation") or {}).get(
                    "reserved_tokens", 0
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
                    "max_total_tokens": llm.max_total_tokens,
                    "reserved_tokens": llm.reserved_tokens,
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
            previous=previous,
            previous_captures=previous_captures,
            checkpoint=checkpoint,
            modes=modes,
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
                    from .agentic import LocalSearch

                    adaptive = LocalSearch(index, embeddings)

                    def search(*args):
                        if len(args) == 3:
                            return adaptive(*args)
                        return index.search(case["questions"][language], embeddings, top_k=args[0])

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
