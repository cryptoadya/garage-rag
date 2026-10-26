# Groq Cloud Free Plan

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
It never uses `auto`, Flex, Batch, tools, model/provider fallback or redirects.
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
They match the current [published Free Plan table](https://console.groq.com/docs/rate-limits).
Limits apply to the organization and can change; check the Console before runs.

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

Bounded one-pair EN command, at most five attempts; execute only within an
explicitly agreed scope. Its completed result is recorded below:

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

## Verified smoke on October 26, 2026

The [separate public artifact](benchmarks/groq-20261026-smoke.v1.json) records one
English `cold_pressure` grounded run, with a two-attempt cap and one local search.
Both completions succeeded; the returned claim retained the cold-tire condition
and its exact source quote. Local citation/language validation and the semantic
audit accepted it (`audited_evidence`). Agent inspection matched the claim to
this synthetic source; independent human review was not performed.

Reported usage was **1,054 input + 372 output tokens** across two calls. The
10.824-second workflow elapsed time includes deliberate 6-RPM pacing and is not
a provider latency estimate. Headers exposed RPD limit 1,000, TPM limit 8,000,
and final remaining balances 998 RPD / 7,174 TPM. RPM/TPD were read from Console,
not inferred from those headers. There were no errors, retries or fallbacks.
No direct comparison, live DE/RU matrix or corpus/profile run followed.

Offline verification passed 243 tests and Ruff; wheel/sdist builds succeeded.
Five frozen Phase 2 dataset/result files matched their pre-change bytes.
After that smoke, the proposed scope was the one EN pair shown above, with at
most five new attempts, stop on first error and explicit resume. Its subsequent
result is recorded below; wider quality conclusions remain outstanding.

## Completed English pair on October 26, 2026

The subsequently authorized one-pair `cold_pressure` comparison is complete;
see the [Phase 3 comparison and explicit source review](benchmarks/grounded-answering.md#groq-one-english-cold-pressure-pair-october-26-2026)
and [separate Groq result](benchmarks/groq-20261026-cold-pressure-en.v1.json).
It used three of five allowed attempts at 6 RPM: direct one call, grounded two.
Both returned the same complete condition-bearing answer and exact source quote;
there were no errors or false refusals. Agent review is hash-bound and separate
from Groq's audit, with no independent human-quality claim.

The original ignored checkpoint now contains both completed modes. Repeating
the exact comparison command with `--resume` was verified to make no API calls.
Do not start a new EN report merely to repeat this pair. At that point the next
proposed stage was DE/RU only, maximum ten new attempts at 6 RPM after quota
verification; its subsequent completion is recorded below.

## Completed DE/RU follow-up and multilingual summary

The authorized DE/RU pairs are now complete, using six of ten new attempts at
6 RPM, after rechecking Free Plan and limits. EN was not rerun. All six EN/DE/RU
answer rows preserve the cold-tire condition and exact English evidence, with no
observed errors or false refusals under explicit agent source/translation review.
The [combined result and next proposal](benchmarks/grounded-answering.md#groq-deru-completion-and-combined-enderu-result-october-26-2026)
separates HTTP/provider-reported time from limiter waits for DE/RU; that split was
not recorded for EN. Current production code is unchanged. The next scope
proposed then was four harder EN cases (unsupported entailment, conflicting
sources, missing-source recovery and false-refusal control), maximum 20 attempts
in separate five-attempt pairs. Its subsequent results are recorded below.
No general quality gain or independent human review is claimed.


## Completed harder English comparisons

The four authorized EN cases have now completed, using 11 of 20 new attempts
at 6 RPM. [Detailed results](benchmarks/groq-hard-en-20261026.md) show one useful
missing-evidence recovery through expansion and three source-relative quality
ties. Grounded added three calls, three searches and 2,398 tokens overall.
No cold-pressure case was rerun, no unsupported draft reached audit, and no
extra false refusal was observed given the actual sources. The report preserves
the legacy recovery false-refusal metric alongside an explicit context-relative
agent review; independent human review and general quality improvement remain
outstanding. No further live stage is scheduled or executed by this report.
