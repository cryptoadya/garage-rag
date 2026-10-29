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
    quota_wait_ms: float = 0
    events: list[NodeEvent] = field(default_factory=list)
    provider_errors: list[dict] = field(default_factory=list)
    rate_limits: list[dict] = field(default_factory=list)

    def public_dict(self):
        return {**asdict(self), "token_usage_complete": self.usage_records == self.llm_calls}


class MeteredLLM:
    def __init__(self, llm, trace):
        self.llm = llm
        self.trace = trace

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def invoke(self, messages):
        self.trace.llm_calls += 1  # Include failed attempts, not only successful responses.
        try:
            response = self.llm.invoke(messages)
        except Exception as exc:
            import httpx

            status = getattr(exc, "status_code", None)
            status = status if type(status) is int and 100 <= status <= 599 else None
            self.trace.provider_errors.append(
                {
                    "http_status": status,
                    "category": (
                        "http_error"
                        if status is not None
                        else "timeout"
                        if isinstance(exc, httpx.TimeoutException)
                        else "transport_error"
                        if isinstance(exc, httpx.TransportError)
                        else "unknown_error"
                    ),
                }
            )
            limits = getattr(exc, "rate_limits", {})
            if isinstance(limits, dict):
                self.trace.provider_errors[-1].update(safe_limits(limits))
            raise
        limits = getattr(response, "response_metadata", {}).get("rate_limits")
        wait = getattr(response, "response_metadata", {}).get("quota_wait_ms", 0)
        if isinstance(wait, (int, float)) and 0 <= wait < float("inf"):
            self.trace.quota_wait_ms += wait
        if isinstance(limits, dict):
            self.trace.rate_limits.append(safe_limits(limits))
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict) and all(
            type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens")
        ):
            self.trace.usage_records += 1
            self.trace.input_tokens += usage["input_tokens"]
            self.trace.output_tokens += usage["output_tokens"]
        return response


def safe_limits(values):
    import math

    allowed = {
        "rpd_limit",
        "rpd_remaining",
        "tpm_limit",
        "tpm_remaining",
        "rpd_reset_seconds",
        "tpm_reset_seconds",
        "retry_after_seconds",
    }
    return {
        k: v
        for k, v in values.items()
        if k in allowed and type(v) in (int, float) and math.isfinite(v) and v >= 0
    }
