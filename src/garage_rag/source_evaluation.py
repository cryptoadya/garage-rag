"""Additive source-relative evaluation; never use the workflow auditor as ground truth."""

import json
import time
from copy import deepcopy

from .config import digest


class EvaluationBinding:
    """Keep native tool-bound requests inside the owning observer/session guard."""

    def __init__(self, owner, target, tools):
        self.owner, self.target, self.tools = owner, target, tools

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def invoke(self, messages):
        return self.owner._invoke(messages, self.target, self.tools)


def capture_messages(messages):
    result = []
    for message in messages:
        item = {"role": message.type, "content": message.content}
        if getattr(message, "tool_calls", []):
            item["tool_calls"] = deepcopy(message.tool_calls)
        if message.type == "tool":
            item["tool_call_id"] = message.tool_call_id
        result.append(item)
    return result


def request_context(messages):
    """Recognize the existing draft/audit protocol and native planner/writer turns."""
    payloads = []
    for message in messages:
        if message["role"] in {"human", "tool"}:
            try:
                payload = json.loads(message["content"])
                if isinstance(payload, dict):
                    payloads.append(payload)
            except (ValueError, TypeError):
                pass
    sources = next((p["sources"] for p in reversed(payloads) if "sources" in p), [])
    phase = (
        "audit"
        if any("draft" in p for p in payloads)
        else "planner"
        if len(messages) < 2
        or messages[0]["content"].startswith("You control local retrieval")
        or not payloads
        or "sources" not in payloads[-1]
        else "draft"
    )
    return phase, sources


class ObservedEvaluationLLM:
    """Observe exact model inputs/outputs privately without altering requests or replies."""

    def __init__(self, llm):
        self.llm = llm
        self.calls = []
        self.public_calls = []
        self.sent_contexts = []

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def invoke(self, messages):
        return self._invoke(messages, self.llm)

    def bind_tools(self, tools, **kwargs):
        return EvaluationBinding(self, self.llm.bind_tools(tools, **kwargs), tools)

    def _invoke(self, messages, target, tools=None):
        captured = capture_messages(messages)
        phase, sources = request_context(captured)
        context = {"sources_sha256": digest(sources), "source_count": len(sources)}
        if (phase == "draft" or phase == "planner" and sources) and (
            not self.sent_contexts or context != self.sent_contexts[-1]
        ):
            self.sent_contexts.append(context)
        public = {
            "phase": phase,
            **context,
            "request_sha256": digest(captured),
        }
        private = {"messages": captured}
        if tools:
            private["tools"] = deepcopy(tools)
            public["tools_sha256"] = digest(tools)
        self.calls.append(private)
        self.public_calls.append(public)
        started = time.perf_counter()
        try:
            response = target.invoke(messages)
            private["response"] = response.content
            public["response_sha256"] = digest(response.content)
            if getattr(response, "tool_calls", []):
                private["response_tool_calls"] = deepcopy(response.tool_calls)
                public["response_tool_calls_sha256"] = digest(response.tool_calls)
            metadata = getattr(response, "response_metadata", {})
            for key in ("http_elapsed_ms", "limiter_wait_ms", "quota_wait_ms"):
                value = metadata.get(key)
                if type(value) in (int, float) and value >= 0:
                    public[key] = value
            if phase == "draft":
                try:
                    value = json.loads(response.content).get("supported")
                    if type(value) is bool:
                        public["draft_supported"] = value
                except (ValueError, TypeError, AttributeError):
                    pass
            if phase == "audit":
                try:
                    value = json.loads(response.content).get("decision")
                    if isinstance(value, str) and value in {
                        "accept",
                        "insufficient",
                        "ambiguous",
                        "conflicting",
                        "out_of_scope",
                        "unsafe_numeric",
                        "unsupported",
                    }:
                        public["audit_decision"] = value
                except (ValueError, TypeError, AttributeError):
                    pass
            return response
        finally:
            public["call_elapsed_ms"] = (time.perf_counter() - started) * 1000


def source_summary(rows):
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        runs = [r for r in rows if r["mode"] == mode and not r["error"]]
        reviewed = [r for r in runs if r.get("source_review")]
        sufficient = [r for r in reviewed if r["source_review"][-1]["sufficiency"] == "sufficient"]
        refusals = [r for r in sufficient if not r["supported"]]
        generator = []
        for row in refusals:
            drafts = [c for c in row.get("model_calls", []) if c["phase"] == "draft"]
            if drafts and drafts[-1].get("draft_supported") is False:
                generator.append(row)
        summary[mode] = {
            "reviewed_runs": len(reviewed),
            "unreviewed_runs": len(runs) - len(reviewed),
            "sufficient_context_runs": len(sufficient),
            "insufficient_context_runs": len(reviewed) - len(sufficient),
            "refusals_reviewed": sum(not r["supported"] for r in reviewed),
            "false_refusals_with_sufficient_context": len(refusals),
            "generator_false_refusals": len(generator),
            "other_sufficient_context_refusals": len(refusals) - len(generator),
            "false_refusal_rate_given_sufficient_context": (
                len(refusals) / len(sufficient) if sufficient else None
            ),
            "justified_context_refusals": sum(
                not r["supported"] and r not in sufficient for r in reviewed
            ),
            "answers_with_insufficient_context": sum(
                r["supported"] and r not in sufficient for r in reviewed
            ),
        }
    return summary


def apply_source_reviews(report, reviews):
    """Atomically validate answer AND every exact sent draft source-set judgment.

    Unknown/ambiguous judgments must remain unreviewed, never infer a label from
    gold-page hits, corpus answerability, the model's answer or its own audit.
    Legacy claim/refusal summaries are untouched.
    """
    if (
        reviews.get("schema_version") != 1
        or reviews.get("report_dataset_sha256") != report["dataset_sha256"]
        or type(reviews.get("independent_human_review")) is not bool
        or not isinstance(reviews.get("reviewer"), str)
        or not reviews["reviewer"].strip()
        or not isinstance(reviews.get("rows"), list)
    ):
        raise ValueError("Invalid source review provenance")
    keys = ("id", "language", "mode", "trial")
    lookup = {tuple(r[k] for k in keys): r for r in report["rows"]}
    seen, updates = set(), []
    for review in reviews["rows"]:
        key = tuple(review.get(k) for k in keys)
        row = lookup.get(key)
        contexts = review.get("contexts")
        if (
            row is None
            or key in seen
            or row["error"]
            or review.get("answer_sha256") != row["answer_sha256"]
            or not isinstance(contexts, list)
            or not contexts
            or len(contexts) != len(row.get("sent_contexts", []))
            or any(
                not isinstance(c, dict)
                or c.get("sources_sha256") != sent["sources_sha256"]
                or c.get("sufficiency") not in ("sufficient", "insufficient")
                for c, sent in zip(contexts, row.get("sent_contexts", []), strict=True)
            )
        ):
            raise ValueError("Invalid or mismatched source review row")
        seen.add(key)
        updates.append(
            (row, [{k: c[k] for k in ("sources_sha256", "sufficiency")} for c in contexts])
        )
    for row, contexts in updates:
        row["source_review"] = contexts
        row["source_review_provenance"] = {
            k: reviews[k] for k in ("reviewer", "independent_human_review")
        }
    pairs = {}
    for row in report["rows"]:
        pairs.setdefault(tuple(row[k] for k in ("id", "language", "trial")), {})[row["mode"]] = row
    modes = set(report.get("modes", ["direct", "grounded"]))
    paired = [
        r
        for p in pairs.values()
        if set(p) == modes and all(not r["error"] and r.get("source_review") for r in p.values())
        for r in p.values()
    ]
    reviewed = [r for r in report["rows"] if not r["error"] and r.get("source_review")]
    report["source_sufficiency"] = {
        "schema_version": 1,
        "reviewer": reviews["reviewer"],
        "independent_human_review": bool(reviewed)
        and all(
            r.get("source_review_provenance", {}).get("independent_human_review") is True
            for r in reviewed
        ),
        "scope": "Last sent draft/planner context; generator subset uses raw draft abstention",
        "summary": source_summary(report["rows"]),
        "paired": {"pairs": len(paired) // len(modes), "summary": source_summary(paired)},
    }
    return report
