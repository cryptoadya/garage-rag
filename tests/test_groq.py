import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from garage_rag.generation import GenerationError, invoke_model, make_llm
from garage_rag.telemetry import MeteredLLM, RunTrace

HTTP_CLIENT = httpx.Client


def model(monkeypatch, handler):
    monkeypatch.setenv("GROQ_FREE_PLAN_CONFIRMED", "true")
    constructor = HTTP_CLIENT
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: constructor(transport=httpx.MockTransport(handler), **kw)
    )
    return make_llm(provider="groq", api_key="private-test-key")


def success(content='{"supported":false,"claims":[]}'):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
        },
        headers={
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-requests": "999",
            "x-ratelimit-remaining-tokens": "7950",
        },
    )


def test_groq_wire_and_usage(monkeypatch):
    requests = []

    def transport(request):
        requests.append(request)
        return success()

    llm = model(monkeypatch, transport)
    with llm.client:
        reply = invoke_model(
            llm, [SystemMessage(content="Return JSON"), HumanMessage(content="Hi")]
        )
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["service_tier"] == "on_demand"
    assert body["response_format"] == {"type": "json_object"}
    assert "tools" not in body and "models" not in body
    assert reply.usage_metadata["input_tokens"] == 20
    assert reply.usage_metadata["output_tokens"] == 30
    assert llm.rate_limits["rpd_limit"] == 1000
    assert llm.rate_limits["tpm_limit"] == 8000


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_errors_are_safe_single_attempt_with_quota_metadata(monkeypatch, status):
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={"error": {"message": "private-test-key payload"}},
            headers={"Retry-After": "12", "x-ratelimit-limit-tokens": "8000"},
        )

    llm = model(monkeypatch, transport)
    trace = RunTrace()
    with llm.client, pytest.raises(GenerationError) as exc:
        invoke_model(MeteredLLM(llm, trace), [HumanMessage(content="JSON")])
    assert len(requests) == 1
    assert "private-test-key" not in str(exc.value)
    assert trace.provider_errors[0]["http_status"] == status
    if status == 429:
        assert trace.provider_errors[0]["retry_after_seconds"] == 12
        with pytest.raises(GenerationError):
            llm.invoke([HumanMessage(content="JSON")])
        assert len(requests) == 1


def test_free_plan_requires_explicit_confirmation(monkeypatch):
    monkeypatch.delenv("GROQ_FREE_PLAN_CONFIRMED", raising=False)
    with pytest.raises(GenerationError, match="Free"):
        make_llm(provider="groq", api_key="private-test-key")


def test_observed_groq_timeout_keeps_provider_identity_and_safe_trace(monkeypatch):
    from garage_rag.source_evaluation import ObservedEvaluationLLM

    requests = []

    def transport(request):
        requests.append(request)
        raise httpx.ReadTimeout("private-test-key payload", request=request)

    llm = model(monkeypatch, transport)
    trace = RunTrace()
    with llm.client, pytest.raises(GenerationError, match="Groq request failed") as exc:
        invoke_model(
            MeteredLLM(ObservedEvaluationLLM(llm), trace),
            [SystemMessage(content="JSON"), HumanMessage(content='{"sources": []}')],
        )
    assert "private-test-key" not in str(exc.value)
    assert len(requests) == trace.llm_calls == 1
    assert trace.provider_errors == [{"http_status": None, "category": "timeout"}]


@pytest.mark.parametrize(
    ("setting", "value"), [("GROQ_MODEL", "other/model"), ("GROQ_MAX_TOKENS", "bad")]
)
def test_groq_rejects_unsupported_configuration(monkeypatch, setting, value):
    monkeypatch.setenv("GROQ_FREE_PLAN_CONFIRMED", "true")
    monkeypatch.setenv(setting, value)
    with pytest.raises(GenerationError):
        make_llm(provider="groq", api_key="private-test-key")


def test_quota_preflight_never_sends_oversized_request(monkeypatch):
    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    with llm.client, pytest.raises(GenerationError, match="TPM"):
        llm.invoke([HumanMessage(content="x" * 8001)])
    assert requests == []


def test_retry_after_http_date_and_header_semantics():
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    from garage_rag.groq import quota_headers

    headers = httpx.Headers(
        {
            "Retry-After": format_datetime(datetime.now(UTC) + timedelta(seconds=60)),
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-reset-requests": "2h3m4.5s",
            "x-ratelimit-reset-tokens": "7.66s",
            "x-secret": "private-test-key",
        }
    )
    result = quota_headers(headers)
    assert 58 <= result["retry_after_seconds"] <= 60
    assert result["rpd_reset_seconds"] == 7384.5
    assert result["tpm_reset_seconds"] == 7.66
    assert "rpm_limit" not in result and "tpd_limit" not in result
    assert "private-test-key" not in json.dumps(result)


@pytest.mark.parametrize(
    ("count", "tokens", "name"),
    [(30, 1, "RPM"), (1000, 1, "RPD"), (1, 8000, "TPM"), (1, 200000, "TPD")],
)
def test_all_four_local_quota_windows(monkeypatch, count, tokens, name):
    import time

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    age = 61 if name in ("RPD", "TPD") else 0
    llm._history = [[time.monotonic() - age, tokens] for _ in range(count)]
    with llm.client, pytest.raises(GenerationError, match=name):
        llm.invoke([HumanMessage(content="JSON")])
    assert requests == []


def test_resume_restores_quota_and_retry_after_without_network(monkeypatch):
    requests = []
    llm = model(
        monkeypatch,
        lambda r: requests.append(r) or httpx.Response(429, headers={"retry-after": "60"}),
    )
    with pytest.raises(GenerationError):
        llm.invoke([HumanMessage(content="JSON")])
    state = llm.quota_state()
    restored = model(monkeypatch, lambda r: requests.append(r) or success())
    restored.restore_quota(state)
    llm.client.close()
    with restored.client, pytest.raises(GenerationError):
        restored.invoke([HumanMessage(content="JSON")])
    assert len(requests) == 1
    assert "private-test-key" not in json.dumps(state)


@pytest.mark.parametrize("mode", ["direct", "grounded"])
@pytest.mark.parametrize(
    ("language", "claim"),
    [
        ("en", "Check the tires when they are cold."),
        ("de", "Prüfen Sie die Reifen im kalten Zustand."),
        ("ru", "Проверяйте давление на холодных шинах."),
    ],
)
def test_groq_wire_preserves_workflow_languages_and_citations(monkeypatch, mode, language, claim):
    from langchain_core.documents import Document

    from garage_rag.index import SearchHit
    from garage_rag.workflow import run_direct, run_workflow

    quote = "Check tire pressure when the tires are cold."
    hit = SearchHit(
        Document(page_content=quote, metadata={"document": "manual.pdf", "pdf_page": 7}), 0.9
    )
    calls = []

    def transport(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if "evidence auditor" in payload["messages"][0]["content"]:
            content = {
                "decision": "accept",
                "complete": True,
                "checks": [
                    {"claim_index": 0, "entailed": True, "applicable": True, "numbers_safe": True}
                ],
            }
        else:
            content = {
                "supported": True,
                "claims": [{"text": claim, "evidence": [{"source_id": "S1", "quote": quote}]}],
            }
        return success(json.dumps(content, ensure_ascii=False))

    llm = model(monkeypatch, transport)
    runner = run_direct if mode == "direct" else run_workflow
    with llm.client:
        result = runner("How to check tires?", lambda k: [hit], llm, answer_language=language)
    assert not result.error
    assert result.answer.supported
    assert result.answer.language == language
    assert "manual.pdf, PDF page 7" in result.answer.text
    assert quote in result.answer.text
    assert len(calls) == (1 if mode == "direct" else 2)


def test_pinned_tokenizer_reservation_is_opt_in_offline_and_keeps_output_budget(monkeypatch):
    from types import SimpleNamespace

    import transformers

    seen = []
    monkeypatch.setenv("GROQ_TOKEN_ACCOUNTING", "tokenizer")

    def load(name, **kwargs):
        seen.append((name, kwargs))
        return SimpleNamespace(
            apply_chat_template=lambda payload, **kw: (
                list(range(100))
                if kw.get("return_dict") is False
                else {"input_ids": list(range(100))}
            )
        )

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    requests = []
    llm = model(monkeypatch, lambda request: requests.append(request) or success())
    payload = [{"role": "user", "content": "long context " * 1000}]
    assert llm.reserve_tokens(payload) == 100 + 256 + 2048
    assert seen[0][0] == "openai/gpt-oss-120b"
    assert seen[0][1]["revision"] == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    assert seen[0][1]["local_files_only"] is True
    assert seen[0][1]["trust_remote_code"] is False
    with llm.client:
        llm.invoke([HumanMessage(content=payload[0]["content"])])
    assert json.loads(requests[0].content)["max_completion_tokens"] == 2048


def test_groq_http_time_excludes_pacing_and_local_observers(monkeypatch):
    import garage_rag.groq as groq

    clock = [0.0]
    monkeypatch.setattr(groq.time, "perf_counter", lambda: clock[0])

    def handler(request):
        clock[0] += 0.025
        return success()

    llm = model(monkeypatch, handler)
    llm.quota_observer = lambda: clock.__setitem__(0, clock[0] + 1)
    with llm.client:
        reply = llm.invoke([HumanMessage(content="JSON")])
    assert reply.response_metadata["http_elapsed_ms"] == pytest.approx(25)
