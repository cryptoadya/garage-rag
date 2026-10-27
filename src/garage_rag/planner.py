"""Closed structured planner decisions; no provider tools or executable arguments."""

import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from .local_tools import SearchArgs


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["search", "finalize", "refuse"]
    method: Literal["dense", "bm25"] | None
    query: str | None
    top_k: int | None
    reason: (
        Literal[
            "insufficient_evidence", "ambiguous_question", "conflicting_sources", "out_of_scope"
        ]
        | None
    )


PLANNER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "garage_rag_planner",
        "strict": True,
        "schema": Decision.model_json_schema(),
    },
}


def parse_decision(content):
    d = Decision.model_validate_json(content)
    if d.action == "search":
        if d.method is None or d.reason is not None:
            raise ValueError("invalid_search_arguments")
        SearchArgs.model_validate({"query": d.query, "top_k": d.top_k})
    elif any(v is not None for v in (d.method, d.query, d.top_k)) or (
        (d.action == "finalize" and d.reason is not None)
        or (d.action == "refuse" and d.reason is None)
    ):
        raise ValueError("incompatible_terminal_arguments")
    return d


def planner_history(messages):
    """Serialize local search history as data, never native provider tool messages."""
    output = []
    for m in messages:
        if m.type == "tool":
            output.append(HumanMessage(content=m.content))
        elif m.type == "ai" and m.tool_calls:
            call = m.tool_calls[0]
            output.append(
                AIMessage(
                    content=json.dumps(
                        {
                            "action": "search",
                            "method": "dense" if call["name"] == "dense_search" else "bm25",
                            **call["args"],
                            "reason": None,
                        }
                    )
                )
            )
        else:
            output.append(m)
    return output
