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
from .local_tools import byte_reservation, checked_call, checked_tools, wire_messages

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
        self._headers_at = 0.0
        self._resume_after = 0.0
        self.quota_observer = None

    def bind_tools(self, tools, *, tool_choice="auto", parallel_tool_calls=False):
        if tool_choice != "auto" or parallel_tool_calls is not False:
            raise GenerationError("Groq agent requires serial auto local tools")
        return _ToolLLM(self, checked_tools(tools))

    def request_reservation(self, messages, tools=None):
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

    def _invoke(self, messages, tools=None):
        payload = wire_messages(messages, tools)
        # UTF-8 bytes bound byte-level tokenizer text tokens conservatively. Include
        # chat framing and reserve the full completion budget, including reasoning.
        reserve = self.reserve_tokens(payload, tools)
        now = time.monotonic()
        self._history = [entry for entry in self._history if now - entry[0] < 86400]
        if self._blocked or time.time() < self._resume_after:
            raise GenerationError("Groq quota/error stop; resume manually after checking limits")
        for name, window, cap, tokens in (
            ("RPM", 60, 30, False),
            ("RPD", 86400, min(1000, self.rate_limits.get("rpd_limit", 1000)), False),
            ("TPM", 60, min(8000, self.rate_limits.get("tpm_limit", 8000)), True),
            ("TPD", 86400, 200000, True),
        ):
            entries = [e for e in self._history if now - e[0] < window]
            used = sum(e[1] for e in entries) if tokens else len(entries)
            if used + (reserve if tokens else 1) > cap:
                raise GenerationError(f"Groq local {name} budget reached; wait/resume manually")
        for name, required in (("rpd", 1), ("tpm", reserve)):
            remaining = self.rate_limits.get(f"{name}_remaining")
            reset = self.rate_limits.get(f"{name}_reset_seconds", 86400 if name == "rpd" else 60)
            if remaining is not None and now - self._headers_at < reset and remaining < required:
                raise GenerationError(
                    f"Groq reported {name.upper()} budget reached; resume manually"
                )
        entry = [now, reserve]
        self._history.append(entry)  # Failed requests keep their conservative reservation.
        self._observe()  # Save reservation before sending, even if the process is interrupted.
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
                            "tool_choice": "auto",
                            "parallel_tool_calls": False,
                            "reasoning_effort": "low",
                            "include_reasoning": False,
                        }
                        if tools
                        else {"response_format": {"type": "json_object"}}
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
            elif finish not in {"stop", "length"} or calls:
                raise ValueError
            content = message.get("content") or ""
            if not isinstance(content, str) or len(content) > 20000:
                raise ValueError
            return AIMessage(
                content=content,
                tool_calls=normalized,
                response_metadata={
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

    def __init__(self, llm, tools):
        self.llm, self.tools = llm, tools

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def invoke(self, messages):
        return self.llm._invoke(messages, self.tools)
