"""Fixed Groq chat endpoint; one HTTP attempt, conservative Free Plan guards.

No SDK retries, alternate endpoints, remote tools, routing or tier upgrades.
The API cannot prove billing tier: an explicit local Free Plan attestation is required.
"""

import json
import math
import os
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from langchain_core.messages import AIMessage

from .generation import GenerationError
from .local_tools import CONTROL_TOOLS, byte_reservation, checked_call, checked_tools, wire_messages

MODEL = "openai/gpt-oss-120b"
TOKENIZER_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def _seconds(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            result = (date - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, result) if math.isfinite(result) else None


def _duration(value):
    if not value or not re.fullmatch(r"(?:\d+(?:\.\d+)?[hms])+", value):
        return None
    return sum(
        float(n) * {"h": 3600, "m": 60, "s": 1}[unit]
        for n, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", value)
    )


def quota_headers(headers):
    """Only numeric fields; requests headers mean RPD, token headers mean TPM."""
    result = {}
    for suffix, name in (
        ("limit-requests", "rpd_limit"),
        ("remaining-requests", "rpd_remaining"),
        ("limit-tokens", "tpm_limit"),
        ("remaining-tokens", "tpm_remaining"),
    ):
        value = headers.get(f"x-ratelimit-{suffix}", "")
        if value.isascii() and value.isdigit() and len(value) <= 12:
            result[name] = int(value)
    for suffix, name in (
        ("reset-requests", "rpd_reset_seconds"),
        ("reset-tokens", "tpm_reset_seconds"),
    ):
        value = _duration(headers.get(f"x-ratelimit-{suffix}"))
        if value is not None:
            result[name] = value
    retry = _seconds(headers.get("retry-after"))
    if retry is not None:
        result["retry_after_seconds"] = retry
    return result


class GroqLLM:
    provider = "groq"
    structured_planner_supported = True
    temperature = 0

    def __init__(self, model=None, api_key=None):
        if os.environ.get("GROQ_FREE_PLAN_CONFIRMED", "").lower() != "true":
            raise GenerationError(
                "Verify Groq organization is on Free Plan, then set GROQ_FREE_PLAN_CONFIRMED=true; "
                "Groq has no per-request zero-price cap"
            )
        self.model = model or os.environ.get("GROQ_MODEL") or MODEL
        if self.model != MODEL:
            raise GenerationError("Groq Free Plan integration currently allows openai/gpt-oss-120b")
        try:
            self.max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", "2048"))
        except ValueError:
            raise GenerationError("GROQ_MAX_TOKENS must be an integer from 512 to 4096") from None
        if not 512 <= self.max_tokens <= 4096:
            raise GenerationError("GROQ_MAX_TOKENS must be an integer from 512 to 4096")
        try:
            self.max_quota_wait = float(os.environ.get("GROQ_MAX_QUOTA_WAIT_SECONDS", "65"))
            if not math.isfinite(self.max_quota_wait) or not 0 <= self.max_quota_wait <= 120:
                raise ValueError
        except ValueError:
            raise GenerationError("GROQ_MAX_QUOTA_WAIT_SECONDS must be from 0 to 120") from None
        key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        if not key:
            raise GenerationError("Set GROQ_API_KEY in .env; local search needs no API key")
        self.token_accounting = os.environ.get("GROQ_TOKEN_ACCOUNTING", "bytes")
        self.tokenizer = None
        if self.token_accounting not in {"bytes", "tokenizer"}:
            raise GenerationError("GROQ_TOKEN_ACCOUNTING must be bytes or tokenizer")
        if self.token_accounting == "tokenizer":
            from transformers import AutoTokenizer

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL,
                    revision=TOKENIZER_REVISION,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception:
                raise GenerationError("Pinned Groq tokenizer unavailable in local cache") from None
        self.client = httpx.Client(
            base_url="https://api.groq.com/openai/v1/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
            follow_redirects=False,
        )
        self.rate_limits = {}
        self._history = []
        self._blocked = False
        self._rate_limit_stop = False
        self._headers_at = 0.0
        self._resume_after = 0.0
        self.quota_observer = None

    def bind_planner(self):
        return _PlannerLLM(self)

    def bind_tools(self, tools, *, tool_choice="auto", parallel_tool_calls=False):
        tools = checked_tools(tools)
        if tool_choice not in ("auto", "required") or parallel_tool_calls is not False:
            raise GenerationError("Groq agent requires serial local actions")
        if tool_choice == "required" and any(t not in tools for t in CONTROL_TOOLS):
            raise GenerationError("Required local actions must allow finalize and refuse")
        return _ToolLLM(self, tools, tool_choice)

    def request_reservation(self, messages, tools=None):
        from .planner import PLANNER_FORMAT

        if isinstance(tools, dict):
            if tools != PLANNER_FORMAT:
                raise GenerationError("Only the closed planner schema is allowed")
            payload = wire_messages(messages)
            if any(m["role"] == "tool" or m.get("tool_calls") for m in payload):
                raise GenerationError("Structured planner cannot use native tools")
            return self.reserve_tokens(payload) + len(json.dumps(tools).encode()) + 256
        return self.reserve_tokens(wire_messages(messages, tools), tools)

    def reserve_tokens(self, payload, tools=None):
        if self.tokenizer is not None:
            # Actual pinned Harmony template plus a margin for provider JSON-mode framing.
            tokens = self.tokenizer.apply_chat_template(
                payload,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **({"tools": tools} if tools else {}),
            )
            return len(tokens) + 256 + self.max_tokens
        if tools or any(m.get("tool_calls") or m["role"] == "tool" for m in payload):
            return byte_reservation(payload, self.max_tokens, tools)
        return sum(len(m["content"].encode("utf-8")) + 32 for m in payload) + self.max_tokens

    def quota_state(self):
        now = time.monotonic()
        return {
            "saved_at": time.time(),
            "history": [[now - started, tokens] for started, tokens in self._history],
            "rate_limits": dict(self.rate_limits),
            "headers_age": now - self._headers_at,
            "resume_after": self._resume_after,
        }

    def restore_quota(self, state):
        if not state:
            return
        elapsed = max(0, time.time() - state["saved_at"])
        now = time.monotonic()
        self._history = [
            [now - age - elapsed, tokens]
            for age, tokens in state["history"]
            if age + elapsed < 86400
        ]
        from .telemetry import safe_limits

        self.rate_limits = safe_limits(state["rate_limits"])
        self._headers_at = now - state["headers_age"] - elapsed
        self._resume_after = state["resume_after"]

    def _observe(self):
        if self.quota_observer:
            self.quota_observer()

    def invoke(self, messages):
        return self._invoke(messages)

    def _quota_error(self, name, delay):
        error = GenerationError(f"Groq {name} quota reached; try again after the cooldown")
        error.status_code = 429
        error.rate_limits = {**self.rate_limits, "retry_after_seconds": math.ceil(delay)}
        return error

    def _wait_for_quota(self, reserve):
        """Pace BEFORE sending; never retry a failed HTTP request or wait for a day."""
        began = time.monotonic()
        while True:
            now = time.monotonic()
            self._history = [e for e in self._history if now - e[0] < 86400]
            if self._blocked:
                if not self._rate_limit_stop:
                    raise GenerationError(
                        "Groq quota/error stop; resume manually after checking limits"
                    )
                if time.time() >= self._resume_after:
                    self._blocked = False
            delays = []
            if time.time() < self._resume_after:
                delays.append((self._resume_after - time.time(), "reported"))
            # Check daily windows first, so a daily exhaustion is never mistaken
            # for a short TPM wait. Failed/unknown usage keeps its reservation.
            for name, window, cap, tokens in (
                ("RPD", 86400, min(1000, self.rate_limits.get("rpd_limit", 1000)), False),
                ("TPD", 86400, 200000, True),
                ("RPM", 60, 30, False),
                ("TPM", 60, min(8000, self.rate_limits.get("tpm_limit", 8000)), True),
            ):
                required = reserve if tokens else 1
                if required > cap:
                    error = GenerationError(f"Groq single-request {name} allowance exceeded")
                    error.status_code = 413
                    raise error
                entries = sorted(e for e in self._history if now - e[0] < window)
                used = sum(e[1] for e in entries) if tokens else len(entries)
                for started, count in entries:
                    if used + required <= cap:
                        break
                    used -= count if tokens else 1
                    delay = max(0, started + window - now) + 0.05
                    # Last expired entry needed to provide sufficient headroom.
                    if used + required <= cap:
                        if window > 60:
                            raise self._quota_error(name, delay)
                        delays.append((delay, name))
            for name, required in (("rpd", 1), ("tpm", reserve)):
                remaining = self.rate_limits.get(f"{name}_remaining")
                reset = self.rate_limits.get(
                    f"{name}_reset_seconds", 86400 if name == "rpd" else 60,
                )
                delay = reset - (now - self._headers_at)
                if remaining is not None and remaining < required and delay > 0:
                    delays.append((delay + 0.05, name.upper()))
            if not delays:
                return (time.monotonic() - began) * 1000
            delay, name = max(delays)
            if delay > self.max_quota_wait - (now - began):
                raise self._quota_error(name, delay)
            time.sleep(min(delay, 30))

    def _invoke(self, messages, tools=None, tool_choice="auto", response_format=None):
        payload = wire_messages(messages, tools)
        # UTF-8 bytes bound byte-level tokenizer text tokens conservatively. Include
        # chat framing and reserve the full completion budget, including reasoning.
        if response_format and tools:
            raise GenerationError("Structured outputs cannot be combined with native tools")
        reserve = self.request_reservation(messages, response_format or tools)
        quota_wait_ms = self._wait_for_quota(reserve)
        now = time.monotonic()
        entry = [now, reserve]
        self._history.append(entry)  # Failed requests keep their conservative reservation.
        self._observe()  # Save reservation before sending, even if the process is interrupted.
        # A recovered 429 must not make a later transport/format error resumable.
        self._rate_limit_stop = False
        try:
            http_started = time.perf_counter()
            response = self.client.post(
                "chat/completions",
                json={
                    "model": self.model,
                    "messages": payload,
                    "temperature": self.temperature,
                    "max_completion_tokens": self.max_tokens,
                    "service_tier": "on_demand",
                    **(
                        {
                            "tools": tools,
                            "tool_choice": tool_choice,
                            "parallel_tool_calls": False,
                            "reasoning_effort": "low",
                            "include_reasoning": False,
                        }
                        if tools
                        else {
                            "response_format": response_format or {"type": "json_object"},
                            **(
                                {"reasoning_effort": "low", "include_reasoning": False}
                                if response_format
                                else {}
                            ),
                        }
                    ),
                },
            )
        except httpx.TransportError:
            self._blocked = True
            self._resume_after = time.time() + 60
            self._observe()
            raise
        http_elapsed_ms = (time.perf_counter() - http_started) * 1000
        self.rate_limits = quota_headers(response.headers)
        self._headers_at = time.monotonic()
        if response.status_code != 200:
            self._blocked = True
            self._rate_limit_stop = response.status_code == 429
            delay = self.rate_limits.get("retry_after_seconds", 60)
            for kind in ("rpd", "tpm"):
                if self.rate_limits.get(f"{kind}_remaining") == 0:
                    delay = max(delay, self.rate_limits.get(f"{kind}_reset_seconds", 60))
            self._resume_after = time.time() + delay
            self._observe()
            category = (
                "rate limit reached"
                if response.status_code == 429
                else "authentication rejected"
                if response.status_code in (401, 403)
                else "model unavailable"
                if response.status_code == 404
                else "billing rejected; verify Free Plan"
                if response.status_code == 402
                else "invalid request"
                if response.status_code == 400
                else "request failed"
            )
            error = GenerationError(
                f"Groq {category} (HTTP {response.status_code}); no retry or fallback; "
                "check limits/key/model before manual resume"
            )
            error.status_code = response.status_code
            error.rate_limits = dict(self.rate_limits)
            if response.status_code == 429:
                error.rate_limits["retry_after_seconds"] = math.ceil(delay)
            raise error
        try:
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
            valid_usage = all(type(n) is int and n >= 0 for n in counts)
            if valid_usage:
                entry[1] = sum(counts)
            self._observe()
            message = choice["message"]
            calls = message.get("tool_calls") or []
            finish = choice["finish_reason"]
            normalized = []
            if finish == "tool_calls":
                if not tools or len(calls) != 1:
                    raise ValueError
                call = calls[0]
                if call["type"] != "function":
                    raise ValueError
                normalized = [
                    {
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "args": json.loads(call["function"]["arguments"]),
                    }
                ]
                checked_call(normalized[0], tools)
            elif (
                finish not in {"stop", "length"}
                or calls
                or (tools and tool_choice == "required" and finish != "length")
            ):
                raise ValueError
            content = message.get("content") or ""
            if not isinstance(content, str) or len(content) > 20000:
                raise ValueError
            return AIMessage(
                content=content,
                tool_calls=normalized,
                response_metadata={
                    "quota_wait_ms": quota_wait_ms,
                    "http_elapsed_ms": http_elapsed_ms,
                    "finish_reason": choice["finish_reason"],
                    "provider": "groq",
                    "model": self.model,
                    "rate_limits": dict(self.rate_limits),
                },
                usage_metadata={
                    "input_tokens": counts[0],
                    "output_tokens": counts[1],
                    "total_tokens": sum(counts),
                }
                if valid_usage
                else None,
            )
        except (ValueError, KeyError, IndexError, TypeError, GenerationError):
            self._blocked = True
            raise GenerationError("Groq returned invalid completion data; no retry") from None


class _ToolLLM:
    """A bound view sharing the client's quota history and single-attempt transport."""

    def __init__(self, llm, tools, tool_choice):
        self.llm, self.tools, self.tool_choice = llm, tools, tool_choice

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def invoke(self, messages):
        return self.llm._invoke(messages, self.tools, self.tool_choice)


class _PlannerLLM:
    def __init__(self, llm):
        self.llm = llm

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def invoke(self, messages):
        from .planner import PLANNER_FORMAT

        return self.llm._invoke(messages, response_format=PLANNER_FORMAT)
