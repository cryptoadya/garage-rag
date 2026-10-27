"""Private evidence snapshots and deterministic, whole-passage Agentic selection.

Lexical relevance is a heuristic, never a semantic proof. On overflow we protect
topic matches, each search's semantic lead, safety cues and page companions. Only
lower-ranked, unconnected, cue-free nonmatches may be omitted. Uncertain or
oversized protected evidence stops the workflow instead of silently disappearing.
"""

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .config import digest
from .generation import Context

# General question/function words, not a vehicle, page or task-specific dictionary.
_STOP = set(
    """a an and are as at be before by can check could do does for from how i
in is it its me my of on or please should tell that the their these this to use was
what when where which while with you your search find
der die das den dem des ein eine einer einen einem und oder im in am an auf aus
bei mit von zu zum zur wie wo was wann ich sie bitte ist sind
а в во где для и из как когда мне на о по при с со что это я
""".split()
)
_SAFETY = re.compile(
    r"\b(warn\w*|caution\w*|danger\w*|risk\w*|never|not|must|only|unless|except\w*|"
    r"if|without|equipped|applicab\w*|require\w*|prohibit\w*|avoid\w*|"
    r"keep|ensure|always|before|after|during|disconnect\w*|applies|fitted|provided|"
    r"achtung|vorsicht|gefahr\w*|nicht|nur|wenn|sofern|ohne|ausgenommen|"
    r"предупрежд\w*|опас\w*|запрещ\w*|нельзя|не|только|если|кроме|без)\b",
    re.IGNORECASE,
)


def _terms(text):
    return {t for t in re.findall(r"[^\W_]+", text.casefold()) if len(t) > 1 and t not in _STOP}


@dataclass
class StoredEvidence:
    document: Document
    origins: list[tuple[int, int]] = field(default_factory=list)


class EvidenceStore:
    """Per-run private snapshots. Never include this object in public trace/asdict.

    Exact text AND metadata identify duplicates. Thus revisions and differing
    applicability cannot be collapsed. Source IDs are allocated once, never reused.
    Production callers are bounded by four searches of at most five hits.
    """

    def __init__(self):
        self.entries = {}
        self.searches = []
        self._identities = {}

    def add(self, hits, *, method, query, top_k):
        if len(self.searches) >= 4 or len(hits) > top_k or not 1 <= top_k <= 5:
            raise ValueError("Evidence store search/hit limit exceeded")
        raw = []
        number = len(self.searches) + 1
        for rank, hit in enumerate(hits, 1):
            document = deepcopy(hit.document)
            metadata = document.metadata
            if (
                not isinstance(metadata.get("document"), str)
                or not metadata["document"]
                or type(metadata.get("pdf_page")) is not int
                or metadata["pdf_page"] < 1
                or not document.page_content.strip()
                or not math.isfinite(hit.score)
            ):
                raise ValueError("Invalid evidence provenance")
            key = digest({"metadata": metadata, "text": document.page_content})
            source_id = self._identities.get(key)
            if source_id is None:
                source_id = f"S{len(self.entries) + 1}"
                self._identities[key] = source_id
                self.entries[source_id] = StoredEvidence(document)
            self.entries[source_id].origins.append((number, rank))
            raw.append(
                {
                    "source_id": source_id,
                    "rank": rank,
                    "score": hit.score,
                    "document_id": document.id,
                    "metadata": metadata,
                    "text": document.page_content,
                }
            )
        self.searches.append(
            {
                "search": number,
                "method": method,
                "query": query,
                "top_k": top_k,
                "hits": raw,
            }
        )

    def history(self):
        return [
            {
                **{k: s[k] for k in ("search", "method", "query", "top_k")},
                "source_ids": [h["source_id"] for h in s["hits"]],
            }
            for s in self.searches
        ]

    def capture(self):
        return deepcopy({"searches": self.searches})


@dataclass(frozen=True)
class Selection:
    context: Context
    diagnostics: dict
    reason: str | None = None


def select_context(store, question, *, max_chars=12000, max_chunks=20):
    """Reconsider the complete store; no accumulated-context feedback or text cuts.

    Scores from dense/BM25 are not comparable: use query overlap and reciprocal
    rank with discovery order as a final tie-break. Safety is a *retention* rule,
    not a claim that a regex can understand all warnings or resolve contradictions.
    """
    if not 2 <= max_chars <= 12000 or not 1 <= max_chunks <= 20:
        raise ValueError("Invalid context bounds")
    entries = {}
    protected = {}
    relevance = {}
    question_terms = _terms(question)
    query_terms = [_terms(s["query"]) for s in store.searches]
    for sid, item in store.entries.items():
        document = item.document
        metadata = deepcopy(document.metadata)
        # Display URL stays in the private map; never remove applicability fields.
        metadata.pop("source_url", None)
        if str(metadata.get("pdf_page")) == str(metadata.get("page_label")):
            metadata.pop("page_label", None)
        entries[sid] = {"source_id": sid, "metadata": metadata, "text": document.page_content}
        words = _terms(document.page_content)
        overlap = len(words & question_terms)
        query_overlap = max((len(words & q) for q in query_terms), default=0)
        relevance[sid] = (overlap, query_overlap, sum(1 / (60 + rank) for _, rank in item.origins))
        reasons = []
        if overlap or query_overlap:
            reasons.append("topic_match")
        if any(rank == 1 for _, rank in item.origins):
            reasons.append("semantic_lead")
        if _SAFETY.search(document.page_content):
            reasons.append("safety_or_condition")
        if any(document.metadata.get(k) for k in ("requires_layout", "warnings", "conditions")):
            reasons.append("safety_metadata")
        if "\ufffd" in document.page_content or re.search(r"\S{100,}", document.page_content):
            reasons.append("uncertain_extraction")
        protected[sid] = reasons

    # Keep retrieved continuations even when their own words have no topic overlap.
    # Compare against primary evidence only, avoiding an unbounded transitive chain.
    pages = {
        (e["metadata"]["document"], e["metadata"].get("content_sha256"), e["metadata"]["pdf_page"])
        for sid, e in entries.items()
        if protected[sid]
    }
    for sid, entry in entries.items():
        m = entry["metadata"]
        if any(
            (m["document"], m.get("content_sha256"), m["pdf_page"] + d) in pages for d in (-1, 0, 1)
        ):
            protected[sid].append("page_companion")

    def serialize(ids):
        return json.dumps([entries[sid] for sid in ids], ensure_ascii=False, separators=(",", ":"))

    found = sum(len(s["hits"]) for s in store.searches)
    diagnostics = {
        "search": len(store.searches),
        "found": found,
        "stored": len(entries),
        "duplicates": found - len(entries),
        "selected": [],
        "excluded": [],
        "blocked": [],
        "selected_count": 0,
        "excluded_count": len(entries),
        "context_chars": 2,
        "max_chars": max_chars,
        "max_chunks": max_chunks,
        "protection": protected,
        "evidence_sha256": digest(store.capture()),
    }
    all_ids = list(entries)
    if len(all_ids) <= max_chunks and len(serialize(all_ids)) <= max_chars:
        selected = all_ids  # Preserve the historical order when nothing needs to be omitted.
    else:
        # All potentially material passages must fit together. No cherry-picking
        # the cheap/positive evidence after a larger warning/negative fails to fit.
        required = [sid for sid in entries if protected[sid]]
        if len(required) > max_chunks or len(serialize(required)) > max_chars:
            packed = []
            for sid in sorted(required, key=lambda s: relevance[s], reverse=True):
                if len(packed) >= max_chunks or len(serialize([*packed, sid])) > max_chars:
                    diagnostics["blocked"].append(sid)
                else:
                    packed.append(sid)
            diagnostics["excluded"] = [
                {
                    "source_id": sid,
                    "reason": "protected_budget"
                    if sid in diagnostics["blocked"]
                    else "selection_aborted",
                }
                for sid in entries
            ]
            return Selection(Context("[]", {}), diagnostics, "context_budget_exhausted")
        if not any(a or b for a, b, _ in relevance.values()):
            diagnostics["excluded"] = [
                {"source_id": sid, "reason": "unassessed_relevance"} for sid in entries
            ]
            return Selection(Context("[]", {}), diagnostics, "context_relevance_uncertain")
        selected = required[:]
        for sid in sorted(entries, key=lambda s: relevance[s], reverse=True):
            if sid not in selected and len(selected) < max_chunks:
                if len(serialize([*selected, sid])) <= max_chars:
                    selected.append(sid)
        # IDs and order remain stable across turns, including after re-entry.
        selected = [sid for sid in entries if sid in selected]
    text = serialize(selected)
    diagnostics.update(
        selected=selected,
        selected_count=len(selected),
        excluded_count=len(entries) - len(selected),
        excluded=[
            {"source_id": sid, "reason": "low_relevance_budget"}
            for sid in entries
            if sid not in selected
        ],
        context_chars=len(text),
    )
    return Selection(
        Context(text, {sid: deepcopy(store.entries[sid].document) for sid in selected}), diagnostics
    )
