import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from garage_rag.generation import GenerationError, invoke_model, make_llm
from garage_rag.telemetry import MeteredLLM, RunTrace

HTTP_CLIENT = httpx.Client


@pytest.fixture(autouse=True)
def default_byte_accounting(monkeypatch):
    # Wire/quota fixtures must not inherit the developer's local tokenizer opt-in.
    # Tokenizer-specific tests override this explicitly below.
    monkeypatch.setenv("GROQ_TOKEN_ACCOUNTING", "bytes")
    monkeypatch.setenv("GROQ_MAX_QUOTA_WAIT_SECONDS", "0")


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


@pytest.mark.parametrize(
    "status, category",
    [(400, "provider_or_network_error"), (401, "authentication"), (429, "rate_limit")],
)
def test_agentic_provider_failure_after_tool_result_is_not_a_local_error(
    monkeypatch, status, category
):
    from langchain_core.documents import Document

    from garage_rag.agentic import run_agentic
    from garage_rag.answer_evaluation import _row
    from garage_rag.index import SearchHit

    monkeypatch.setenv("GROQ_MAX_TOKENS", "1024")
    requests = []

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        from conftest import decision

        assert body["response_format"]["type"] == "json_schema"
        assert "tools" not in body and "tool_choice" not in body
        if len(requests) == 1:
            return success(
                decision("search", method="dense", query="Aurora release location", top_k=5)
            )
        assert json.loads(body["messages"][-1]["content"])["sources"]
        return httpx.Response(
            status,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "code": "output_parse_failed",
                    "message": "Parsing failed.",
                    "failed_generation": "Need more detail.",
                }
            },
        )

    llm = model(monkeypatch, transport)
    hits = [
        SearchHit(
            Document(
                page_content=(
                    "For the Aurora manual tailgate release, "
                    "see the emergency release instructions."
                ),
                metadata={"document": "aurora-synthetic.pdf", "pdf_page": 1},
            ),
            1.0,
        )
    ]
    with llm.client:
        result = run_agentic(
            "Where is the Aurora manual tailgate release?",
            lambda *args: hits,
            llm,
            answer_language="en",
        )
        with pytest.raises(GenerationError):
            llm.invoke([HumanMessage(content="Must not retry")])
    assert len(requests) == result.trace.llm_calls == 2
    assert result.trace.search_calls == 1
    assert result.trace.provider_errors[-1]["http_status"] == status
    assert not result.trace.public_dict()["token_usage_complete"]
    assert llm._history[-1][1] > 1024  # Unknown failed usage keeps the full reservation.
    assert "Need more detail" not in result.error
    row = _row(
        result, {"id": "observed_failure", "expected": "answer"}, "en", "agentic", 1, scripted=False
    )
    assert row["error"] and row["error_category"] == category


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
    monkeypatch.setenv("GROQ_TOKEN_ACCOUNTING", "bytes")
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
@pytest.mark.parametrize("quote", [
    "Check tire pressure when the tires are cold.",
    "Prüfen Sie den Reifendruck bei kalten Reifen.",
    "Проверяйте давление на холодных шинах.",
])
def test_groq_wire_preserves_workflow_languages_and_citations(
    monkeypatch, mode, language, claim, quote,
):
    from langchain_core.documents import Document

    from garage_rag.index import SearchHit
    from garage_rag.workflow import run_direct, run_workflow

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


def test_native_tools_roundtrip_serializes_ids_arguments_and_results(monkeypatch):
    from langchain_core.messages import ToolMessage

    from garage_rag.local_tools import TOOLS

    requests = []

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning": "private reasoning",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "dense_search",
                                            "arguments": '{"query":"Reifendruck","top_k":2}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                },
            )
        return success()

    llm = model(monkeypatch, transport)
    bound = llm.bind_tools(TOOLS, tool_choice="auto", parallel_tool_calls=False)
    messages = [SystemMessage(content="Search or return JSON"), HumanMessage(content="Tires?")]
    with llm.client:
        first = bound.invoke(messages)
        second = bound.invoke(
            [*messages, first, ToolMessage(content='{"sources":[]}', tool_call_id="call_1")]
        )
    assert first.tool_calls[0]["args"] == {"query": "Reifendruck", "top_k": 2}
    assert second.content == '{"supported":false,"claims":[]}'
    assert requests[0]["tools"] == TOOLS
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["include_reasoning"] is False
    assert requests[0]["reasoning_effort"] == "low"
    assert "response_format" not in requests[0]
    assert "reasoning_format" not in requests[0]
    sent = requests[1]["messages"]
    assert sent[2]["tool_calls"][0]["id"] == sent[3]["tool_call_id"] == "call_1"
    assert sent[3]["role"] == "tool"
    assert "private reasoning" not in json.dumps(sent)


@pytest.mark.parametrize(
    "call",
    [
        {"id": "x", "type": "browser_search"},
        {"id": "x", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
        {"id": "x", "type": "function", "function": {"name": "dense_search", "arguments": "{"}},
    ],
)
def test_native_tools_reject_malformed_or_unauthorized_responses(monkeypatch, call):
    from garage_rag.local_tools import TOOLS

    requests = []
    llm = model(
        monkeypatch,
        lambda r: (
            requests.append(r)
            or httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": None, "tool_calls": [call]},
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        ),
    )
    with llm.client, pytest.raises(GenerationError):
        llm.bind_tools(TOOLS).invoke([HumanMessage(content="hello")])
    assert len(requests) == 1


def test_native_tools_reject_remote_or_modified_definitions_before_http(monkeypatch):
    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    with llm.client, pytest.raises(GenerationError):
        llm.bind_tools([{"type": "browser_search"}])
    assert requests == []


def test_native_tool_quota_reserves_schema_arguments_and_results(monkeypatch):
    monkeypatch.setenv("GROQ_TOKEN_ACCOUNTING", "bytes")
    from langchain_core.messages import AIMessage, ToolMessage

    from garage_rag.local_tools import TOOLS

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    messages = [
        HumanMessage(content="hello"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "dense_search", "args": {"query": "tire", "top_k": 1}}
            ],
        ),
        ToolMessage(content="x" * 8000, tool_call_id="c1"),
    ]
    with llm.client, pytest.raises(GenerationError, match="TPM"):
        llm.bind_tools(TOOLS).invoke(messages)
    assert requests == []


def test_orphan_tool_result_rejected_without_http(monkeypatch):
    from langchain_core.messages import ToolMessage

    from garage_rag.local_tools import TOOLS

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    with llm.client, pytest.raises(GenerationError):
        llm.bind_tools(TOOLS).invoke([ToolMessage(content="secret", tool_call_id="unknown")])
    assert requests == []


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 429, 500, 503])
def test_tool_requests_share_one_attempt_error_stop(monkeypatch, status):
    from garage_rag.local_tools import TOOLS

    requests = []
    llm = model(
        monkeypatch,
        lambda r: (
            requests.append(r)
            or httpx.Response(status, json={"error": {"message": "secret key and tool args"}})
        ),
    )
    with llm.client:
        for _ in range(2):
            with pytest.raises(GenerationError) as error:
                llm.bind_tools(TOOLS).invoke([HumanMessage(content="search")])
            assert "secret key" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "count,tokens,name", [(30, 1, "RPM"), (1000, 1, "RPD"), (1, 8000, "TPM"), (1, 200000, "TPD")]
)
def test_tool_requests_share_all_four_quota_guards(monkeypatch, count, tokens, name):
    import time

    from garage_rag.local_tools import TOOLS

    requests = []
    llm = model(monkeypatch, lambda r: requests.append(r) or success())
    age = 61 if name in {"RPD", "TPD"} else 0
    llm._history = [[time.monotonic() - age, tokens] for _ in range(count)]
    with llm.client, pytest.raises(GenerationError, match=name):
        llm.bind_tools(TOOLS).invoke([HumanMessage(content="search")])
    assert requests == []


def test_missing_usage_retains_whole_tool_reservation(monkeypatch):
    from garage_rag.local_tools import TOOLS

    response = success()
    data = response.json()
    del data["usage"]
    llm = model(monkeypatch, lambda r: httpx.Response(200, json=data))
    messages = [HumanMessage(content="search")]
    reserve = llm.request_reservation(messages, TOOLS)
    assert reserve > llm.request_reservation(messages)
    with llm.client:
        reply = llm.bind_tools(TOOLS).invoke(messages)
    assert reply.usage_metadata is None
    assert llm._history[0][1] == reserve


def test_tool_tokenizer_receives_schemas_and_native_results(monkeypatch):
    from types import SimpleNamespace

    import transformers
    from langchain_core.messages import AIMessage, ToolMessage

    from garage_rag.local_tools import TOOLS

    seen = []

    def template(messages, **kwargs):
        seen.append((messages, kwargs))
        return list(range(100))

    monkeypatch.setenv("GROQ_TOKEN_ACCOUNTING", "tokenizer")
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(apply_chat_template=template),
    )
    llm = model(monkeypatch, lambda r: success())
    messages = [
        HumanMessage(content="search"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "dense_search", "args": {"query": "tire", "top_k": 1}}
            ],
        ),
        ToolMessage(content='{"sources": []}', tool_call_id="c1"),
    ]
    with llm.client:
        assert llm.request_reservation(messages, TOOLS) == 2404
    assert seen[0][1]["tools"] == TOOLS
    assert seen[0][0][-1]["tool_call_id"] == "c1"
    assert seen[0][0][-2]["tool_calls"][0]["function"]["name"] == "dense_search"


def test_tool_response_with_multiple_calls_cannot_trigger_parallel_search(monkeypatch):
    from garage_rag.local_tools import TOOLS

    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "dense_search", "arguments": '{"query":"tire","top_k":1}'},
    }
    llm = model(
        monkeypatch,
        lambda r: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": None, "tool_calls": [call, {**call, "id": "c2"}]},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ),
    )
    with llm.client, pytest.raises(GenerationError):
        llm.bind_tools(TOOLS).invoke([HumanMessage(content="search")])


@pytest.mark.parametrize("initial_empty", [False, True])
def test_agentic_groq_structured_loop_changes_strategy_from_local_evidence(
    monkeypatch, initial_empty
):
    from test_workflow import audit, draft, hit

    from garage_rag.agentic import AgenticLimits, run_agentic

    requests, searches = [], []

    def transport(request):
        payload = json.loads(request.content)
        requests.append(payload)
        from conftest import decision

        # User sources for structured planner; native local pair for unchanged writer.
        results = [
            m
            for m in payload["messages"]
            if m["role"] in {"user", "tool"} and "sources" in json.loads(m["content"])
        ]
        if "evidence auditor" in payload["messages"][0]["content"]:
            return success(json.dumps(audit()))
        if payload["response_format"]["type"] != "json_schema":
            return success(json.dumps(draft()))
        assert "tools" not in payload
        if results and json.loads(results[-1]["content"])["sources"]:
            return success(decision("finalize"))
        return success(
            decision(
                "search",
                method="bm25" if results else "dense",
                query="cold tire pressure" if results else "Reifendruck prüfen",
                top_k=1,
            )
        )

    def search(method, query, k):
        searches.append((method, query))
        return [] if initial_empty and len(searches) == 1 else [hit()]

    if not initial_empty:
        monkeypatch.setenv("GROQ_MAX_TOKENS", "512")
    llm = model(monkeypatch, transport)
    limits = (
        AgenticLimits(max_total_tokens=48000)
        if initial_empty
        else AgenticLimits(max_llm_calls=3, max_searches=1, max_total_tokens=16000)
    )
    with llm.client:
        result = run_agentic(
            "Wie prüfe ich die Reifen?",
            search,
            llm,
            answer_language="en",
            limits=limits,
            top_k=1,
        )
    assert result.answer.supported
    assert len(requests) == result.trace.llm_calls == (5 if initial_empty else 3)
    assert result.trace.input_tokens == (100 if initial_empty else 60)
    assert searches == (
        [("dense", "Reifendruck prüfen"), ("bm25", "cold tire pressure")]
        if initial_empty
        else [("dense", "Reifendruck prüfen")]
    )
    assert all("models" not in r and r["model"] == "openai/gpt-oss-120b" for r in requests)


def test_groq_cross_language_expansion_shares_transport_and_usage(monkeypatch):
    from test_multilingual import CLAIMS, QUESTIONS
    from test_workflow import draft, hit

    from garage_rag.answering import run_answer

    requests, queries = [], []
    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        data = json.loads(body['messages'][1]['content'])
        if 'languages' in data:
            assert data == {'question': QUESTIONS['en'], 'languages': ['ru']}
            value = {'queries': [{'language': 'ru', 'query': QUESTIONS['ru']}]}
        else:
            assert data['question'] == QUESTIONS['en']
            value = draft(CLAIMS['de'], CLAIMS['ru'])
        return success(json.dumps(value, ensure_ascii=False))
    llm = model(monkeypatch, transport)
    with llm.client:
        result = run_answer(
            QUESTIONS['en'], lambda q, k: queries.append(q) or [hit(CLAIMS['ru'])], llm,
            mode='direct', answer_language='de', source_languages=('ru',),
        )
    assert result.answer.supported
    assert result.answer.language == 'de'
    assert queries == [(QUESTIONS['en'], QUESTIONS['ru'])]
    assert len(requests) == result.trace.llm_calls == result.trace.llm_call_limit == 2
    assert result.trace.input_tokens == 40
    assert result.trace.output_tokens == 60
    assert result.trace.public_dict()['token_usage_complete']
