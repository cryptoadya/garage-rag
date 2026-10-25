"""Allowlisted in-memory observations: no prompts, passages, paths or provider errors."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class NodeEvent:
    node: str
    outcome: str
    elapsed_ms: float
    source_count: int


@dataclass
class RunTrace:
    mode: str = "grounded"
    llm_call_limit: int = 4
    search_call_limit: int = 2
    llm_calls: int = 0
    search_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_records: int = 0
    elapsed_ms: float = 0
    events: list[NodeEvent] = field(default_factory=list)

    def public_dict(self):
        return {**asdict(self), "token_usage_complete": self.usage_records == self.llm_calls}


class MeteredLLM:
    def __init__(self, llm, trace):
        self.llm = llm
        self.trace = trace

    def invoke(self, messages):
        self.trace.llm_calls += 1  # Include failed attempts, not only successful responses.
        response = self.llm.invoke(messages)
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict) and all(
            type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens")
        ):
            self.trace.usage_records += 1
            self.trace.input_tokens += usage["input_tokens"]
            self.trace.output_tokens += usage["output_tokens"]
        return response
