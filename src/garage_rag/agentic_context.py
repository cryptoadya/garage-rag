"""Private evidence snapshots and deterministic, whole-passage Agentic selection.

Lexical relevance is a heuristic, never a semantic proof. Navigation is separate
from answer evidence. Conditions attach to procedures; explicit warnings without
a proven scope remain protected. No passage is clipped or summarized.
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
located location access instruction instructions information manual owner owners
vehicle vehicles document documentation model models guide guidance
der die das den dem des ein eine einer einen einem und oder im in am an auf aus
bei mit von zu zum zur wie wo was wann ich sie bitte ist sind
а в во где для и из как когда мне на о по при с со что это я
""".split()
)
_WARNING = re.compile(
    r"\b(warn\w*|caution\w*|danger\w*|risk\w*|injur\w*|precaution\w*|never|"
    r"prohibit\w*|disconnect\w*|must|should\s+not|do\s+not|don't|keep\s+(?:hands|clear|away)|"
    r"only\s+(?:for|when|if|with)|unless|except|provided\s+that|"
    r"applies\s+to|equipped\s+with|fitted\s+with|"
    r"achtung|vorsicht|gefahr\w*|nur\s+(?:für|wenn|bei|mit)|gilt\s+für|"
    r"nicht\s+(?:öffnen|entfernen|lösen|berühren|verwenden|betätigen)|"
    r"предупрежд\w*|опас\w*|запрещ\w*|нельзя|только\s+(?:для|при|если|с)|"
    r"не\s+(?:открыва\w*|снима\w*|использ\w*|ослабля\w*|трога\w*))\b",
    re.IGNORECASE,
)
_DIRECTIVE = re.compile(
    r"\b(never|do\s+not|don't|disconnect\w*|keep\s+(?:hands|clear|away)|"
    r"only\s+(?:if|when|with)|must|"
    r"nicht\s+(?:öffnen|entfernen|lösen|berühren|verwenden|betätigen)|"
    r"нельзя|запрещ\w*|не\s+(?:открыва\w*|снима\w*|использ\w*|ослабля\w*|трога\w*))\b",
    re.IGNORECASE,
)


def _terms(text):
    words = re.findall(r"[^\W\d_]+", text.casefold())
    return {
        t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t
        for t in words
        if len(t) > 1 and t not in _STOP
    }


def _focus_match(text, focus):
    """Require a local pair for compound topics, not scattered shared modifiers."""
    words = [(m.start(), _terms(m.group())) for m in re.finditer(r"[^\W\d_]+", text)]
    matches = [(pos, terms & focus) for pos, terms in words if terms & focus]
    return max(
        (
            len(set().union(*(terms for pos, terms in matches if start <= pos <= start + 160)))
            for start, _ in matches
        ),
        default=0,
    )


def _navigation(text):
    """Recognize reference lists structurally, not by document name or page number.

    Decimal cells and sentence prose do not qualify. A real warning alongside a
    list prevents demotion. Reference labels stay opaque: printed != PDF pages.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    refs, rest = [], []
    for line in lines:
        match = re.fullmatch(r"(.+?[^\d\s])\s*(\d{1,4})(?:[a-z])?", line, re.IGNORECASE)
        if match and not re.search(r"[.!?;:]", match[1]):
            refs.append((match[1].strip(), match[2]))
        else:
            rest.append(line)
    if (
        len(refs) < 4
        or len(refs) / max(1, len(lines)) < 0.35
        or any(len(line) > 75 or re.search(r"[.!?;]$", line) for line in rest)
        or _WARNING.search(" ".join(rest))
        or _DIRECTIVE.search(text)
    ):
        return []
    return refs


def _scope(metadata, field):
    value = metadata.get(field)
    if isinstance(value, str) and value:
        return {value}
    if isinstance(value, list) and value and all(isinstance(x, str) and x for x in value):
        return set(value)
    return set()


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
    """Select whole evidence plus scoped dependencies; budget navigation separately."""
    if not 2 <= max_chars <= 12000 or not 1 <= max_chunks <= 20:
        raise ValueError("Invalid context bounds")
    entries, protected, relevance, roles, references = {}, {}, {}, {}, {}
    # Vehicle qualifiers constrain applicability, not the subject of every procedure.
    qualifiers = set().union(
        *(
            _terms(" ".join(item.document.metadata.get("vehicle_models", [])))
            for item in store.entries.values()
        )
    )
    question_terms = _terms(question) - qualifiers
    query_terms = [_terms(s["query"]) - qualifiers for s in store.searches]
    focuses = [f for f in [question_terms, *query_terms] if f]
    topical = set()
    for sid, item in store.entries.items():
        document = item.document
        metadata = deepcopy(document.metadata)
        metadata.pop("source_url", None)
        if str(metadata.get("pdf_page")) == str(metadata.get("page_label")):
            metadata.pop("page_label", None)
        entries[sid] = {"source_id": sid, "metadata": metadata, "text": document.page_content}
        counts = [_focus_match(document.page_content, f) for f in focuses]
        match = any(n >= min(2, len(f)) for n, f in zip(counts, focuses, strict=True))
        relevance[sid] = (
            _focus_match(document.page_content, question_terms),
            max(counts, default=0),
            sum(1 / (60 + rank) for _, rank in item.origins),
        )
        refs = _navigation(document.page_content)
        references[sid] = {
            label
            for title, label in refs
            if any(len(_terms(title) & f) >= min(2, len(f)) for f in focuses)
        }
        roles[sid] = "navigation" if refs else "evidence" if match else "unrelated"
        protected[sid] = []
        if match and not refs:
            topical.add(sid)
            protected[sid].append("procedure_topic")

    def revision(sid):
        m = entries[sid]["metadata"]
        return m["document"], m.get("content_sha256")

    def scopes(sid):
        m = entries[sid]["metadata"]
        return _scope(m, "procedure_id") | _scope(m, "section_id")

    dependencies = {}
    for sid, entry in entries.items():
        m, text = entry["metadata"], entry["text"]
        links = [
            other
            for other in topical
            if other != sid
            and revision(other) == revision(sid)
            and (scopes(other) & (scopes(sid) | _scope(m, "applies_to")))
        ]
        same_page = [
            other
            for other in topical
            if other != sid
            and revision(other) == revision(sid)
            and entries[other]["metadata"]["pdf_page"] == m["pdf_page"]
        ]
        if links:
            protected[sid].append("procedure_dependency")
        if same_page:
            protected[sid].append("same_procedure_page")
        dependencies[sid] = sorted(set(links + same_page))
        # Exclusion by explicit scope needs complete primary scope information.
        # Missing/malformed metadata, cross-revision/global or ambiguous warnings
        # are never treated as proof of irrelevance.
        same_revision = [other for other in topical if revision(other) == revision(sid)]
        other_scope = bool(
            same_revision
            and all(scopes(other) for other in same_revision)
            and _scope(m, "applies_to")
            and not any(_scope(m, "applies_to") & scopes(other) for other in same_revision)
            and not m.get("global_warning")
        )
        # A reference to a warning in an index is navigation, not the warning itself.
        if roles[sid] != "navigation" and _WARNING.search(text) and not other_scope:
            protected[sid].append(
                "unscoped_warning"
                if sid not in topical and not links and not same_page
                else "procedure_warning"
            )
        if (
            any(m.get(k) for k in ("requires_layout", "warnings", "conditions", "global_warning"))
            and not other_scope
        ):
            protected[sid].append("safety_metadata")
        if "\ufffd" in text or re.search(r"\S{100,}", text):
            protected[sid].append("uncertain_extraction")
        if protected[sid] and roles[sid] == "unrelated":
            roles[sid] = "safety"

    navigation = [sid for sid in entries if references[sid]]
    # With no usable lexical/navigation anchor keep semantic leads for inspection,
    # rather than assuming that a multilingual/paraphrased result is irrelevant.
    fallback = not topical and not navigation
    if fallback:
        for sid, item in store.entries.items():
            if any(rank == 1 for _, rank in item.origins):
                protected[sid].append("unassessed_semantic_lead")
                if roles[sid] == "unrelated":
                    roles[sid] = "semantic_candidate"

    def serialize(ids):
        return json.dumps([entries[sid] for sid in ids], ensure_ascii=False, separators=(",", ":"))

    required = [sid for sid in entries if protected[sid]]
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
        "protected_count": len(required),
        "protected_chars": len(serialize(required)),
        "context_chars": 2,
        "max_chars": max_chars,
        "max_chunks": max_chunks,
        "protection": protected,
        "roles": roles,
        "dependencies": dependencies,
        "relevance": {
            sid: {"question_overlap": a, "focus_overlap": b} for sid, (a, b, _) in relevance.items()
        },
        "has_substantive_evidence": False,
        "evidence_sha256": digest(store.capture()),
    }
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
    if fallback and len(serialize(list(entries))) > max_chars:
        diagnostics["excluded"] = [
            {"source_id": sid, "reason": "unassessed_relevance"} for sid in entries
        ]
        return Selection(Context("[]", {}), diagnostics, "context_relevance_uncertain")

    selected, exclusion = required[:], {}
    seen_targets = set()
    for sid in sorted(navigation, key=lambda s: relevance[s], reverse=True):
        targets = {(revision(sid), label) for label in references[sid]}
        if targets <= seen_targets:
            exclusion[sid] = "redundant_navigation"
        elif len(selected) < max_chunks and len(serialize([*selected, sid])) <= max_chars:
            selected.append(sid)
            seen_targets.update(targets)
        else:
            exclusion[sid] = "navigation_budget"
    # Discovery order keeps citation display stable. Only the priority decisions
    # above affect membership; adding new evidence never reassigns an ID.
    selected = [sid for sid in entries if sid in selected]
    text = serialize(selected)
    diagnostics.update(
        selected=selected,
        selected_count=len(selected),
        excluded_count=len(entries) - len(selected),
        excluded=[
            {
                "source_id": sid,
                "reason": exclusion.get(
                    sid,
                    "low_relevance_budget" if len(serialize([sid])) > max_chars else "off_topic",
                ),
            }
            for sid in entries
            if sid not in selected
        ],
        context_chars=len(text),
        has_substantive_evidence=bool(topical or fallback and selected),
    )
    return Selection(
        Context(text, {sid: deepcopy(store.entries[sid].document) for sid in selected}), diagnostics
    )
