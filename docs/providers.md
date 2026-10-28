# Generation providers

`ask --provider openrouter|groq` overrides `GARAGE_RAG_PROVIDER`. Indexing and
search need neither provider. Both transports share the generation prompt,
JSON schemas, local validator, language handling and bounded LangGraph workflow.

## OpenRouter

Set `OPENROUTER_API_KEY` in the local `.env`. The default is `openrouter/free`;
`OPENROUTER_MODEL` or `ask --model` can select an explicit `provider/model:free`.
The client rejects paid model IDs, requires provider parameter support and sets
prompt/completion/request price caps to zero. SDK retries and provider fallbacks
are disabled. Free routing still depends on availability and quotas; it does
not pin an upstream model when using the free router.

`OPENROUTER_MAX_TOKENS` defaults to 8192 (512–32768), shared by reasoning and
visible output. Truncated responses refuse even if their JSON looks complete.
The fixed endpoint is `https://openrouter.ai/api/v1`; each completion has a
60-second network-operation timeout, not a total deadline. See OpenRouter's
[routing controls](https://openrouter.ai/docs/guides/routing/provider-selection)
and [reasoning budget](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## Groq Cloud Free Plan

Groq is an additional transport for the existing direct and bounded LangGraph
workflows. It does not change retrieval baselines, prompts, JSON answer/draft/audit
schemas, evidence quotations, citation validation, EN/DE/RU language handling,
case selection, paired summaries or hash-bound review methodology.

## Setup and billing boundary

1. In [Groq Billing](https://console.groq.com/settings/billing/plans), verify the
   organization used by your key shows **Free / Current Plan**. Do not upgrade.
2. Put `GROQ_API_KEY` in the existing local `.env` without printing or committing it.
3. Set `GROQ_FREE_PLAN_CONFIRMED=true` only after verification. This local
   attestation is required before constructing the client. Recheck if you change
   key/organization or billing plan. It cannot prevent an external account upgrade.

```bash
garage-rag ask 'How do I check tire pressure?' --provider groq --workflow grounded
# Direct uses the same transport and the unchanged single-completion baseline:
garage-rag ask 'Как проверить давление в шинах?' --provider groq --workflow direct
```

`GARAGE_RAG_PROVIDER` selects `openrouter` (default) or `groq`; `--provider` wins.
`GROQ_MODEL` defaults to `openai/gpt-oss-120b`, currently the only allowlisted Groq
model. `GROQ_MAX_TOKENS` is 2,048 by default, integer 512–4096, including reasoning.
There is no automatic output repair/retry. Truncated JSON remains a refusal.

The HTTP adapter uses the fixed `https://api.groq.com/openai/v1/chat/completions`
endpoint, JSON object mode, temperature zero and `service_tier=on_demand`.
It never uses `auto`, Flex, Batch, remote/built-in tools, provider fallback or redirects.
Network-operation timeout is 60 seconds, not a whole-workflow deadline.
No new LLM SDK is needed; `httpx` is an explicit pinned dependency.

Unlike OpenRouter's unchanged zero-price caps, Groq has no request parameter that
forces zero billing. Free usage depends on the account's plan. Attestation is a
fail-closed setup guard, not an API-enforced price guarantee. This distinction
follows Groq's [billing documentation](https://console.groq.com/docs/billing-faqs)
and [service tiers](https://console.groq.com/docs/service-tiers). No account or
billing settings are changed by GarageRAG.

## Quotas and errors

On October 26, 2026, this organization's Console showed Free Plan and these
`openai/gpt-oss-120b` limits: **30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD**.
These are the adapter's local caps, not a promise of current account allowance.
Check your Console and the [rate-limit documentation](https://console.groq.com/docs/rate-limits)
before live runs; organization limits can change.

The adapter guards all four windows conservatively: rolling 60 seconds and
24 hours, reserving UTF-8 input bytes plus chat framing and the entire output
budget before sending. Successful responses replace the token reservation with
reported prompt+completion usage; failed/interrupted calls retain it. This upper
estimate can stop a long prompt before the provider would; it never trims sources,
reduces top-k or changes retrieval to fit. Lowering the explicitly chosen output
budget or waiting is manual. A prompt larger than the conservative single-request
TPM allowance needs a separately reviewed token-accounting improvement.

Groq headers have specific meanings: `*-requests` describes **RPD**, while
`*-tokens` describes **TPM**. They do not prove RPM/TPD balances. Numeric remaining,
limit and reset observations are retained in traces. Server limits can lower local
caps. RPM/TPD use conservative local accounting; other processes/organizations'
usage is governed by the API, not known to this process. Separate `ask` invocations
have separate local histories; run comparisons sequentially in one evaluator.

Every API request has one HTTP attempt. A 429 stops the run, retains HTTP status
and numeric `Retry-After` (seconds or HTTP date) and does not sleep/retry/switch.
Quota reset durations are also observed. Resume preserves a known cooldown and
quota reservations; a missing Retry-After uses a conservative 60-second stop,
not a claim about the actual daily reset. Authentication, request/model, billing,
server and transport failures likewise stop without raw response bodies or keys.

## Separate, resumable comparisons

Do not run a matrix just to confirm connectivity. Start with a single synthetic
supported grounded question (normally two calls: draft and audit), inspect its
JSON/citations and the live headers, then explicitly agree on a comparison scope.
A smoke is operational evidence, not a quality improvement claim.

Example one-pair EN comparison, at most five attempts; run only within an
explicitly agreed live scope:

```bash
GROQ_FREE_PLAN_CONFIRMED=true python -m garage_rag.answer_evaluation \
  --live --provider groq --model openai/gpt-oss-120b \
  --cases cold_pressure --languages en --trials 1 \
  --max-api-calls 5 --requests-per-minute 6 \
  --output output/groq/gpt-oss-120b/cold-pressure-en.json \
  --private-captures output/private/groq/gpt-oss-120b/cold-pressure-en.json
```

Use a different result path for every provider/model/selection. Reports identify
provider and model, generation settings, dataset/code/index hashes and observed
quota state; they cannot merge OpenRouter and Groq into one resumed report.
`--max-api-calls` is a per-invocation attempt ceiling; failed attempts consume it.
The existing five-attempt worst-case reservation per unfinished pair remains.
With `--modes direct grounded agentic`, the bound is eleven per unfinished
case/language/trial. Comparative runs additionally require a session token cap
and private captures; see the [Phase 4B comparison](benchmarks/phase4b-comparison.md).
Shared pacing is unchanged (1–15 RPM); Groq additionally checks token/day windows.

Results are atomically replaced after each completed **mode**, including errors.
Groq reservations are checkpointed before every request and updated after its
response. Optional private captures are saved first; the public report determines
which rows committed. Ctrl-C or process termination preserves earlier completed
modes. An interrupted in-flight mode can have unknown provider outcome and must
be rerun manually; there is no provider idempotency guarantee.

To continue, rerun the identical command with **`--resume`**, after the limit clears.
Successful rows (including valid refusals) are skipped. Failed rows move to
`previous_errors` before their explicit new attempt; their traces/costs remain
available. Existing summaries score the current rows using unchanged rules;
errors are excluded from quality denominators. Cumulative prior attempt counts
are retained separately. A completed checkpoint makes zero further calls.
Changing provider/model, data, case/language/trial/seed selection, source code,
profile/index manifest, output budget, pacing or capture path is rejected before
client construction; start a separate report. Historical pre-checkpoint reports
remain readable for review but cannot be resumed without the new identity.

Private captures remain under ignored `output/private/`; public reports contain
hashes/counts/decisions, never source passages or generated claim text. As before,
semantic quality requires explicit external hash-bound review. The same-model
LangGraph auditor is not independent evaluation ground truth.

## Optional offline tokenizer accounting for real corpus contexts

`GROQ_TOKEN_ACCOUNTING=tokenizer` opts into the pinned
[`openai/gpt-oss-120b` tokenizer](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a).
Download only `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `config.json` and `chat_template.jinja` to the existing
Hugging Face cache before the run. No model weights are needed. The runtime
loads revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` with
`local_files_only=True` and `trust_remote_code=False`; unavailable files fail
closed. Existing `transformers` dependencies suffice.

The reservation counts the pinned Harmony chat template, adds 256 input tokens
for provider/JSON-mode framing differences, and retains the full configured
completion/reasoning budget. This is a conservative estimate, not a guarantee
of identical provider framing; compare against returned input usage in live
reports. The default `bytes` mode, all four quota caps, fixed endpoint, output
budget, sources, prompts and retrieval are unchanged. This option addresses a
measured transport-accounting limitation: ordinary real BMW top-5 contexts can
exceed the byte reservation's 8,000-TPM guard while fitting under the tokenizer
reservation. It does not trim context to improve evaluation results.

The evaluator records RPM-limiter wait independently. Groq response metadata
records wall-clock duration of the synchronous HTTP call, excluding local quota
observers and sleeps. An experiment can proactively wait for token-window
headroom **before** invoking the provider; that wait must be reported separately
as `quota_wait_ms`. No automatic retry follows a local/provider error.

## Agentic planner transport

Agentic defaults to Groq even when `GARAGE_RAG_PROVIDER=openrouter`. An explicit
`--provider openrouter` remains compatible, with zero-price limits and no fallback;
Phase 4A makes no OpenRouter live/quality comparison. Other workflows keep their
provider defaults and existing JSON completion wire format.

The experimental Groq planner uses strict `json_schema` Structured Outputs,
without native tools, tool choice or parallel-tool options. Writer/auditor retain
JSON-object mode. Planner requests keep `reasoning_effort=low` and
`include_reasoning=false`; hidden reasoning consumes the completion allowance.
Full schemas plus framing are reserved alongside input/output, including in
pinned-tokenizer mode. Bound requests share session and RPM/TPM/RPD/TPD guards.
See [planner validation and dispatch](architecture.md#agentic-retrieval-phase-4a)
for the closed contract and optional non-Groq native path. Neither protocol is a
fallback for the other. Captures bind exact formats and sent sources; offline
transport checks do not establish inference reliability.

Different `ask` processes do not share quota history, and other clients' usage is
unknown until headers/errors; run one experiment sequentially and verify available
organization quota first. API-reported RPD/TPM reductions are enforced. Neither
mode can enforce a Free Plan if someone externally upgrades the account.

## Phase 4A live acceptance

Historical October 26 synthetic Aurora runs confirmed native tool/result delivery,
a RU dense → EN BM25 change, cited writing and mandatory audit. The completed
RU pair used Agentic 4 calls / 4,177 actual tokens versus Grounded 3 / 3,904;
both answered, both omitted the explicit hot-system warning. A prior 512-token
Grounded attempt failed output validation; the completed pair used 1,024 tokens.
This was AI source checking, not independent human or vehicle acceptance, and
showed no Agentic quality/resource advantage. Full measurements remain in the
[immutable Phase 4A record](https://github.com/cryptoadya/garage-rag/blob/de6a96577322feffcb2ea59c649070402f3ac1f1/docs/providers.md#phase-4a-live-acceptance).
For subsequent real-corpus results and current experimental status, see the
[Phase 4B comparison](benchmarks/phase4b-comparison.md).
