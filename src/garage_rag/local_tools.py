"""Closed local retrieval protocol. No caller-supplied function registry."""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .generation import GenerationError


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(ge=1, le=5)


class FinalizeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RefuseArgs(FinalizeArgs):
    reason: Literal[
        "insufficient_evidence", "ambiguous_question", "conflicting_sources", "out_of_scope"
    ]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": SearchArgs.model_json_schema(),
        },
    }
    for name, description in [
        ("dense_search", "Read local active Qdrant passages by multilingual semantic similarity."),
        (
            "lexical_search",
            "Read local active passages by BM25 exact terms, e.g. English identifiers.",
        ),
    ]
]
METHODS = {"dense_search": "dense", "lexical_search": "bm25"}
CONTROL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": args.model_json_schema(),
        },
    }
    for name, description, args in [
        (
            "finalize",
            "Finish retrieval with existing evidence; separate writer and audit follow.",
            FinalizeArgs,
        ),
        (
            "refuse",
            "Stop with a justified closed refusal reason, without another search.",
            RefuseArgs,
        ),
    ]
]
PLANNER_TOOLS = [*TOOLS, *CONTROL_TOOLS]
ARGUMENTS = {
    "dense_search": SearchArgs,
    "lexical_search": SearchArgs,
    "finalize": FinalizeArgs,
    "refuse": RefuseArgs,
}


def checked_tools(tools):
    if not isinstance(tools, list) or not tools or any(t not in PLANNER_TOOLS for t in tools):
        raise GenerationError("Only predefined local retrieval tools are allowed")
    names = [t["function"]["name"] for t in tools]
    if len(names) != len(set(names)):
        raise GenerationError("Duplicate tool definitions")
    return tools


def checked_call(call, tools=TOOLS):
    try:
        if call["name"] not in {t["function"]["name"] for t in tools}:
            raise ValueError
        if not isinstance(call["id"], str) or not re.fullmatch(
            r"[\w-]{1,128}", call["id"], re.ASCII
        ):
            raise ValueError
        args = ARGUMENTS[call["name"]].model_validate(call["args"])
        if isinstance(args, SearchArgs) and not args.query.strip():
            raise ValueError
        return args
    except (KeyError, TypeError, ValueError):
        raise GenerationError("Invalid local tool call") from None


def wire_messages(messages, tools=None):
    """Lossless native call/result pairing, excluding reasoning and arbitrary metadata."""
    payload, pending, used = [], None, set()
    for message in messages:
        if not isinstance(message.content, str):
            raise GenerationError("Only text chat messages are allowed")
        role = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}.get(
            message.type
        )
        if role is None:
            raise GenerationError("Unsupported chat role")
        entry = {"role": role, "content": message.content}
        if role == "tool":
            if pending is None or message.tool_call_id != pending:
                raise GenerationError("Orphan local tool result")
            entry["tool_call_id"] = pending
            pending = None
        else:
            if pending is not None:
                raise GenerationError("Missing local tool result")
            calls = getattr(message, "tool_calls", [])
            if getattr(message, "invalid_tool_calls", []):
                raise GenerationError("Invalid local tool call")
            if calls:
                if role != "assistant" or len(calls) != 1:
                    raise GenerationError("Only serial local tool calls are allowed")
                call = calls[0]
                # No active tools is valid for a writer receiving a historical
                # native pair. The same closed allowlist still validates it.
                checked_call(call, TOOLS if tools is None else tools)
                pending = call["id"]
                if pending in used:
                    raise GenerationError("Duplicate local tool call ID")
                used.add(pending)
                entry["tool_calls"] = [
                    {
                        "id": pending,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"], ensure_ascii=False),
                        },
                    }
                ]
        payload.append(entry)
    if pending is not None:
        raise GenerationError("Missing local tool result")
    return payload


def byte_reservation(payload, max_tokens, tools=None):
    # Byte-level input bound, chat framing, schema/function framing, full output
    # including invisible reasoning. Never assumes English chars/token ratios.
    return (
        len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        + 32 * len(payload)
        + max_tokens
        + (len(json.dumps(tools, ensure_ascii=False).encode("utf-8")) + 256 if tools else 0)
    )
