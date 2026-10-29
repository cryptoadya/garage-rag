"""Unspent terminal capacity, independent of the monotonic request ledger."""

import json
from dataclasses import asdict, dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from .generation import Draft, GenerationError
from .grounding import AUDIT_PROMPT, audit_payload


@dataclass(frozen=True)
class TerminalBudget:
    writer_tokens: int
    audit_tokens: int
    draft_allowance: int

    @property
    def total(self):
        return self.writer_tokens + self.audit_tokens

    def public_dict(self):
        return {**asdict(self), "total": self.total}


def audit_messages(question, context, draft):
    """Mirror the frozen auditor's request format; contract-tested against it."""
    return [
        SystemMessage(content=AUDIT_PROMPT),
        HumanMessage(
            content=json.dumps(
                audit_payload(question, context, draft),
                ensure_ascii=False,
            )
        ),
    ]


def terminal_budget(question, context, messages, writer, auditor):
    """Allocate audit growth without assuming English characters per token.

    Capacity includes all writer completion/reasoning tokens and a byte-level
    allowance for the largest JSON frame permitted by the unchanged draft schema.
    It is an allocation, not a promise that arbitrary normalized JSON will fit:
    the actual audit request must fit the protected terminal capacity before it
    can be invoked. Once retrieval ends, audit may also use all spare headroom.
    Projections never enter reserved_tokens; actual requests are charged once.
    """
    completion = getattr(writer, "max_tokens", None)
    if type(completion) is not int or completion <= 0:
        raise GenerationError("A finite completion budget is required")
    schema = Draft.model_json_schema()
    claims = schema["properties"]["claims"]["maxItems"]
    evidence = schema["$defs"]["Claim"]["properties"]["evidence"]["maxItems"]
    id_chars = schema["$defs"]["Evidence"]["properties"]["source_id"]["maxLength"]
    # Sizing only: this frame is never sent to a model or treated as a candidate.
    frame = {
        "supported": False,
        "claims": [
            {
                "text": "",
                "evidence": [{"source_id": "X" * id_chars, "quote": ""} for _ in range(evidence)],
            }
            for _ in range(claims)
        ],
    }
    allowance = completion + len(json.dumps(frame, ensure_ascii=False).encode("utf-8"))
    base = auditor.estimate(audit_messages(question, context, {"supported": False, "claims": []}))
    return TerminalBudget(writer.estimate(messages), base + allowance, allowance)
