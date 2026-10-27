# Phase 4B: direct / grounded / agentic

**Phase 4B remains open.** The authorized real-corpus run below stopped on a
provider error after four attempts. One complete Grounded answer received a
separate AI source review; no successful quality pair/triple or human review exists.

Offline software acceptance, 2026-10-26, based on `main` / `v0.4.0` (`da4b3cc`).
The earlier authorized Aurora attempt on 2026-10-27 stopped on a provider error
(see below); no independent human review was performed. **The offline results
do not measure LLM quality or establish Agentic superiority.** The existing
answer evaluator, captures and review packet are extended; historical Phase 2/3
datasets, baselines and direct/grounded runtime behavior are unchanged.

## Fixed selection and reproduction

`benchmarks/aurora-comparative.v1.json` contains seven fictional English-source
needs, each asked in EN/DE/RU: simple cold-pressure guidance, compound jack
preparation/safety, omitted warning, missing vehicle qualifier, absent repair
specification, conflicting equally applicable documents, and following a release
instruction reference. Translations and annotations are implementation-authored,
not independently validated. The canonical dataset SHA256 is
`1e7814cf8186ba26397d0febb15b0dd9e9ea4a345ae95c3e74462d1d024ce4f5`.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= GROQ_API_KEY= \
  python -m garage_rag.answer_evaluation \
  --dataset benchmarks/aurora-comparative.v1.json \
  --modes direct grounded agentic --trials 3 --seed 20261026 \
  --output output/benchmarks/phase4b.json \
  --private-captures output/private/phase4b.json
```

Use new paths for every run; existing outputs are not overwritten. The new suite
defaults to all three modes; legacy datasets retain the original two-mode default.
Only this summary and the necessary fixture/tests are published. JSON results,
raw answers, passages, prompts and review packets stay in ignored local output.

Each mode executes its real workflow independently. Direct and Grounded retain
the identical first context; Grounded can increase the fixed ranking depth.
Agentic starts empty and uses its existing native local-search protocol. Rankings
and model replies here are **controlled fixtures**, including a targeted passage
outside fixed-depth expansion and an auditor deliberately accepting an unsafe
omission. Neither natural retrieval difficulty nor LLM decision quality is tested.
Seeded case/mode order, dataset/code hashes, settings and private captures make
the run reproducible. Repeated trials are repetitions, not independent samples.

## Observations and denominators

There are 189 successful workflow executions: 63 per mode, 63 matched triples,
and 63 matched comparisons for each mode pair. There are only **seven distinct
needs**, four answerable and three requiring refusal. No significance or
representative generalization follows from these counts.

| Scripted observation | Direct | Grounded | Agentic |
| --- | ---: | ---: | ---: |
| Complete supported answers / answerable runs | 9/36 | 18/36 | 27/36 |
| Unsupported positive answers / positive answers | 27/45 | 9/27 | 9/36 |
| Corpus-relative false refusals / answerable runs | 9/36 | 9/36 | 0/36 |
| Correct refusals / unanswerable runs | 9/27 | 27/27 | 27/27 |
| Claims with sufficient cited evidence / reviewed positive claims | 18/45 | 18/27 | 27/36 |
| Missed warnings / reviewed positive answers requiring warnings | 18/18 | 9/18 | 9/18 |
| Answers passing local source/quote/page validation | 45 | 27 | 36 |
| Workflow model invocations (zero actual API calls) | 63 | 135 | 261 |
| Local fixture searches | 63 | 90 | 81 |
| Workflow latency p50 / p95, ms | 0.29 / 0.40 | 3.06 / 7.62 | 4.35 / 9.97 |

Latency is local orchestration with scripted responses, **not inference latency**.
Token usage is unavailable: zero reported tokens and `token_usage_complete=false`
are not a zero-token LLM-cost measurement. Live reports retain reported input/output
tokens and missing-usage flags, workflow latency, HTTP time, pacing and quota waits
separately. Failed attempts retain resources; `attempt_summary` includes failures
from earlier checkpoints, while matched quality summaries exclude failed or missing
members. Per-language and per-kind strata and all three pairwise summaries remain
in local JSON; identical fixture translations cannot prove multilingual parity.

Correctness checks entailment, applicability and essential conditions; completeness
checks every requested part separately. Citation support judges the entire claim
against its cited evidence, not merely quote existence. Thus a valid partial jack
instruction can lack completeness and its warning. A locally valid quotation does
not certify safety. Corpus-relative false refusals can reflect missing sent context;
they do not automatically mean generator over-abstention.

## Independent evidence review

Script annotations establish only expected fixture behavior. Same-model
`audit_decision` is a recorded workflow observation, never an evaluation label.
Live claim support, completeness, citation support and warning preservation remain
null until an external reviewer supplies answer-hash-bound labels. Uncertain rows
must stay unreviewed. Per-row reviewer identity and human independence are retained;
an implementation-agent review is not independent human review.

```bash
python -m garage_rag.review_packet --report output/benchmarks/phase4b.json \
  --captures output/private/phase4b.json --output-dir output/private/phase4b-review
# Fill templates independently; then rescore offline into a new report:
python -m garage_rag.answer_evaluation --report-input output/benchmarks/phase4b.json \
  --review-file output/private/phase4b-review/claim-review.template.json \
  --source-review-file output/private/phase4b-review/source-review.template.json \
  --output output/benchmarks/phase4b-reviewed.json
```

Packets verify native call arguments, tool/result IDs, requests, responses and
ordered source-set hashes. Source review labels every context actually sent for
writing or an evidence-based planner decision; retrieved-but-unsent material
cannot establish sufficiency. A refusal before any context is sent remains
unreviewed. Existing source-relative refusal metrics distinguish raw-generator
abstention from validation/audit/planner rejection. Citation/warning review is
atomic with claim review; auditor acceptance is not copied into these labels.

## Where the extra work can pay off

In the controlled reference-following case, both workflows search twice, but only
Agentic changes the query and obtains the missing release passage: five model
calls versus Grounded's two-call refusal. This demonstrates a recoverable route,
not a measured probability that a real model will choose it.

Grounded suffices when the first context or fixed expansion contains the necessary
evidence. On the compound fixture both finish correctly, with four calls for
Grounded versus five for Agentic; on simple guidance the counts are two versus
four (Direct needs one). Ambiguity/conflict refusals also tie, at two versus four.
Both audited modes miss the deliberately accepted warning omission: extra planning
does not itself fix an unreliable auditor. Agentic's 126 additional invocations
over Grounded yield nine extra complete fixture answers, all repetitions of the
one targeted-retrieval need. Prefer Grounded for routine evidence already within
reach; consider Agentic for reference following, terminology/translation repair
or cross-document retrieval only when independent review confirms enough recovered
complete answers to justify its additional calls, tokens and latency. No such
general model-quality or production-cost advantage is established offline.

## One authorized live run — partial, stopped on provider error

On **2026-10-27 16:51–16:54 UTC / 17:51–17:54 Europe/Berlin**, at PR head
`30f6307`, the existing evaluator selected `follow_reference`, EN, one planned trial per mode,
Groq `openai/gpt-oss-120b`, temperature zero, 1,024 completion tokens **including
reasoning**. Hard ceilings were 11 API attempts and 64,000 reserved input+output
tokens, with at least 61 seconds between actual HTTP starts. Seed `20261026`
selected the order **Grounded → Agentic → Direct**. Workflows, prompts, dataset,
retrieval fixtures and production guards were unchanged.

The morning preflight had correctly refused to start with only 37,127 local
tokens available. On this second, explicitly requested check, the latest
cumulative quota state retained **16 requests / 18,156 tokens** in the rolling
24-hour window, leaving **181,844 tokens**: enough for the full reservation.
Entries were aged, not cleared at midnight; overlapping older states were not
added again. Historical Free Plan evidence matched the same key. Current account
plan/organization-wide balance was not independently verified; no browser,
Console or extra API probe was used. Provider quota guards remained active.

An ignored one-shot runner restored that history and retained the evaluator's
shared no-refund attempt/token guard across every mode and native tool binding.
It added execution-only HTTP spacing and a durable start marker preventing
reruns. All four attempts used the specified model/token cap; starts were
**61.063 / 61.068 / 61.066 seconds apart**. There were no retries, fallback,
automatic resume or subsequent experiment.

| Observed resource / outcome | Direct | Grounded | Agentic |
| --- | --- | --- | --- |
| Outcome | Not run after stop | Positive but incomplete answer | Provider error before writer/audit |
| API attempts / local fixture searches | 0 / 0 | 2 / 1 | 2 / 1 |
| Held reservations | 0 | 6,390 | 7,422 |
| Reported input + completion tokens | Not run | 1,078 + 477 = 1,555 | 464 + 48 = 512; failed-call usage unknown |
| Workflow elapsed, including pacing | Not run | 61.873 s | 121.982 s |
| Wire HTTP time, including failed requests | Not run | 1.785 s | 1.149 s |

Total: **4/11 attempts**, **13,812/64,000 held tokens**, **2,067 reported actual
tokens plus unknown failed-call usage**, two searches and three HTTP 200 replies.
The fourth response was **HTTP 400**, `invalid_request_error` /
`output_parse_failed`, with failed generation `Need more detail.` The run stopped
immediately, exit status 2; Direct remained unexecuted and there are **zero
successful matched pairs/triples**. The failed attempt keeps its full 4,040-token
reservation. Actual usage is incomplete, not zero for that attempt. Completion
usage includes reported reasoning; HTTP error alone does not establish truncation.
Wire timing includes the failed call; normal successful-response telemetry omits
that call's HTTP/pacing fields, so it is not a complete failure-cost measurement.

### Separate source check versus same-model auditor

The original source here is the fictional JSON benchmark, not a real vehicle
manual. Its target/reference says the release is **under the rear seat**. The
initial source merely directs the reader to emergency release instructions.
Grounded returned: “The Aurora manual tailgate release is found in the emergency
release instructions.” Its quote exactly matches that initial source and its
`aurora-synthetic.pdf`, page 1 mapping is correct **fixture metadata**, not an
independently inspected physical PDF page. The claim/citation are supported only
as a documentary pointer; they do not provide the requested physical location.
The supplied context is insufficient and the answer is **incomplete**. There are
no case-required warnings to score. Grounded did not expand its retrieval.

The **same-model auditor** nevertheless returned `accept`, `complete=true`, all
claim checks true. A separate post-run implementation-assistant source check
assigned pointer claim/citation support but `complete_answer=false` and source
`sufficiency=insufficient`, bound to the exact answer/context hashes. These are
**not independent human judgments** and were not copied from the auditor.
Grounded thus has 0/1 complete answers despite locally valid citations.

Agentic made one dense-tool query for the tailgate release location and received
the same reference pointer. Its next planner call failed at the provider. It did
not issue the follow-reference lexical query, retrieve the target, reach the
writer or invoke its auditor. The generic user-facing refusal represents a
provider failure, not a reviewed semantic refusal. **No Agentic recovery of
information beyond Grounded was observed.** This establishes no quality ranking:
Agentic failed and Direct is missing.

Retrieval remains controlled: the target is returned only for the benchmark's
exact BM25 follow-up query. The run tests live model behavior on this fixture,
not natural Qdrant/BM25 retrieval quality, multilingual parity or vehicle safety.
Private requests, raw replies (including the 400), cumulative quota state, review
packet, hash-bound labels and verification remain under ignored
`output/private/phase4b-live-executed-20261027/`. Dataset/source hashes match the
unmodified inputs and runtime; packet integrity and resource limits were checked.
**24 existing offline quota/reservation/tool-binding/pacing tests passed** before
the run. No full test suite or package build was rerun for this report-only update.

The software's existing offline acceptance remains intact. This live experiment
exposed **two real blockers to successful live acceptance**: provider output
parsing stopped Agentic, and the same-model auditor accepted an incomplete
Grounded answer. One case does not establish their frequency or a general model
advantage. The offline checks are not a Phase 4B completion decision: the required
real-corpus three-mode comparison is **not complete** and Phase 4B remains open.
No production tuning, extra live run, merge, tag or Release was performed.

## Offline diagnosis and real-corpus preparation, 2026-10-27

The preserved failed request is a native planner continuation, not an answer-JSON
request. Its assistant call has the permitted function and valid JSON arguments;
its following tool result uses the same call ID. `tool_choice=auto`, serial tools,
`reasoning_effort=low`, `include_reasoning=false`, and absent `response_format` /
`reasoning_format` match the local protocol and documented Groq options. Earlier
successful local traces use the same continuation structure. See Groq's
[local tool-calling protocol](https://console.groq.com/docs/tool-use/local-tool-calling)
and [reasoning parameters](https://console.groq.com/docs/reasoning). No confirmed
malformed-request or call-pairing defect was found; no production prompt,
reasoning option or auditor behavior was changed to improve the benchmark.

Offline replay used `httpx.MockTransport`, with **both serialized request bodies
exactly equal to the saved Agentic requests**. Replaying the recorded first 200
and then the 400 reproduces the safe two-attempt stop, one search, retained
unknown-usage reservation and no retry. Returning the same `Need more detail.`
text as a hypothetical 200 instead produces `invalid_planner_output`: it is not
one of the allowed planner commands. This is a local response-handling test,
**not reproduction of Groq inference or its server parser**. A model/parser
formatting failure is plausible; a transient backend issue cannot be excluded.
There is no usage/finish metadata proving completion-cap truncation. Root cause
and provider-side reproducibility remain unproven; no speculative fix or new
live request was made.

One separate defect **was confirmed and fixed**: the report categorized the
Agentic safe generic error as `local_or_workflow_error` despite a structured
HTTP 400 in its trace. New regression cases first failed for 400/401/429, then
passed when classification used the structured provider observation; console
progress now uses the same row category. Stop/retry/reservation and answer/audit
behavior are unchanged. The historical raw report remains intact, including
its original category; it is not silently rewritten after the fact.

The measured Grounded `accept` / `complete=true` on the incomplete documentary
pointer remains unchanged in this report and the preserved hash-bound review.
Its genuine quote is not reinterpreted as evidence of the missing location.

The next proposal is preregistered in
[the real-corpus protocol](phase4b-real-live-protocol.md): two fixed existing EN
questions, the original BMW E39 PDF and existing baseline Qdrant/E5/BM25 index,
six planned outputs, at most **22 attempts / 128,000 held tokens**, 1,024
completion tokens including reasoning and actual HTTP spacing of at least
61 seconds. It uses the existing evaluator, capture/review packet and rescoring
path, not a new evaluation framework. Gold passages, questions, independent
source-review criteria and resource ceilings are fixed before future generation.
Local real-index contract replies are controlled refusals, not model-quality
results. **At preparation commit `a6fbe15`, only offline diagnosis/preparation had
occurred. The subsequent user-authorized real run is recorded below; the frozen
protocol is retained unchanged. Phase 4B remains open.**

Preparation validation: **384 tests passed** with model downloads disabled and
both API keys blank; Ruff and diff whitespace checks passed. The pinned-tokenizer
real-index contract completed six local workflow executions / two matched
triples with controlled refusals, checked identical initial Direct/Grounded
contexts and generated an integrity-checked review packet. It made zero API
calls and does not establish live answer quality. No package build was needed
for the reporting-only runtime correction.

## Authorized real-corpus run — partial, stopped on first error

On **2026-10-27 17:59:52–18:02:56 UTC / 18:59:52–19:02:56 Europe/Berlin**, the
user-authorized one-shot experiment ran exactly at `a6fbe15`, using the frozen
[real-corpus protocol](phase4b-real-live-protocol.md). Two unchanged EN needs,
`hood_release` and `toolkit_access`, one trial per Direct/Grounded/Agentic mode,
were planned. Groq `openai/gpt-oss-120b`, temperature zero, completion cap 1,024
including reasoning, existing prompts/reasoning settings and serial native tools
were unchanged. Seed `20261027` scheduled toolkit Grounded → Agentic → Direct,
then hood Grounded → Agentic → Direct. There was no retry, fallback, resume,
additional experiment, production change, merge, tag or Release.

### Integrity, quotas and execution controls

Immediately before execution, original PDF SHA256, 186 pages, canonical dataset
SHA256, baseline index fingerprint, 241 active chunks, logical chunk digest,
frozen protocol hash, clean code commit and locally cached pinned tokenizer
revision all matched the protocol. The latest cumulative quota checkpoint was
selected after scanning saved checkpoints; its historical failed 4,040-token
reservation was preserved. Aging the rolling 86,400-second window retained
**11 requests / 14,188 reserved tokens**, leaving **185,812 tokens / 989 local
requests**: sufficient for the full **128,000-token / 22-attempt** ceiling.
No calendar-midnight reset or overlapping snapshot addition was used; no known
cooldown remained. The same historically verified Free Plan key matched.
Current organization-wide balance and billing plan were not independently
verified; stale headers were not represented as a current balance.

The adapted, ignored execution-only runner restored that history, retained the
existing shared no-refund guard across all modes/tool bindings, checked the
actual request model/cap and used a durable one-shot marker. Actual HTTP start
intervals were **61.091 / 61.074 / 61.083 seconds**. Provider/local quota guards
remained active. The final response headers reported RPD 996/1,000 remaining
and TPM 5,363/8,000 remaining; these are response-time snapshots, not TPD or
organization-wide guarantees. The failed 3,847-token reservation remains held.

### Observed outcomes and retrieval

| Case / mode | Outcome | API attempts / workflow searches | Held tokens | Reported input + completion | Workflow wall time | Wire HTTP time, including failure |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Toolkit / Grounded | Complete, source-supported answer | 2 / 1 | 6,636 | 4,132 + 632 = 4,764 | 62.440 s | 2.278 s |
| Toolkit / Agentic | HTTP 400 before writer/audit | 2 / 1 | 5,513 | 468 + 49 = 517; failed-call usage unknown | 121.373 s | 0.908 s |
| Toolkit / Direct | Not run after stop | 0 / 0 | 0 | Not run | — | — |
| Hood / all three modes | Not run after stop | 0 / 0 | 0 | Not run | — | — |

Total: **4/22 attempts**, **12,149/128,000 held tokens**, **5,281 reported actual
tokens plus unknown failed-call usage**: 4,600 input and 681 completion tokens,
including 496 reported reasoning tokens. Missing usage is not zero. There were
three HTTP 200 replies; attempt four returned HTTP 400,
`invalid_request_error` / `output_parse_failed`. Its failed generation expressed
an intention to search further, rather than a valid native tool call. Execution
stopped immediately with exit status 2. This repeats the provider parsing failure
class seen in Aurora; its backend cause remains unproven. No live diagnosis or
repair was attempted. Agentic's generic user-facing refusal is a provider failure,
not a scored semantic refusal or corpus-relative false refusal.

Grounded used genuine dense top-5 pages **23, 180, 85, 31, 100**. Page 23 already
states the tool kit's location on the inside of the luggage compartment
lid/tailgate and the wing-screw opening procedure. Both requested facts were
present in the actual sent context. It produced a complete answer citing that
page and its matching quote, then received same-model auditor acceptance.
There was no top-20 expansion. **Preparation's inference that absent gold page
150 meant missing necessary evidence was incorrect.** Page 150 is a valid source,
but page 23 independently supplies equivalent facts. Frozen gold-page/chunk
match flags are false for this answer; they are an incomplete gold-set measure,
not evidence that its citation is invalid. Frozen labels were not changed.

Agentic started without context and chose one native `dense_search`, query
`onboard tool kit access location vehicle manual`, top-k 5. Actual sent pages
were **180, 182, 181, 8, 31**: index pointers/manual notes, insufficient for both
requested facts. Its next planner call failed before another search, writing or
auditing. **No BM25 call or retrieval recovery occurred.** Grounded already had
the necessary evidence, so merely retrieving page 150 would not have established
recovery beyond Grounded. Genuine BM25 remained available through unchanged
local tools; the model did not select it before the failure.

### Separate source assessment and limits

A **separate Codex AI source-review agent**, distinct from the execution agent
and workflow generator/auditor, fixed judgments before consulting auditor
verdicts or costs. It inspected original physical pages 137/150 and freshly
rendered cited page 23, checked all ten sent source/page entries against
the original text, and saved answer/context-hash-bound labels plus per-claim
citation/applicability/warning notes. The sole non-error answer has **1/1 supported
claims, complete_answer=true, 1/1 supported citations**, correct quote/document/
physical-page mapping and no omitted warning relevant to the stated location/
access claim. It volunteers no servicing/closing instructions or specifications.
The same-model auditor agrees here; its verdict was not an evaluation label.
The earlier Aurora false completeness acceptance remains a measured weakness.

Offline rescoring through the existing evaluator succeeded. Agentic's error row
remains unscored; missing modes remain missing. There are **zero successful
matched pairs/triples** (`paired_summary` has zero matched pairs), so there is
no quality ranking or evidence of Agentic superiority. Review was **AI-based,
unblinded and not independent human evaluation**. Two preselected, previously
used EN needs, with only one usable final answer, establish no multilingual,
representative, statistical or vehicle-safety conclusion.

Whole execution was **187.615 s**, including **3.636 s before first HTTP** for
initialization/local warmup. Exactly **one local warmup search** is excluded from
the two metered workflow searches. Total wire HTTP time including failure was
**3.187 s**; workflow non-HTTP time was **180.627 s**, mostly pacing plus local
retrieval/tokenization/orchestration. Successful-call telemetry records 119.758 s
of limiter wait; failed-call pacing metadata is missing, so it is not a complete
wait total. Provider-reported successful completion generation totaled 1.435 s;
failed inference duration is unknown. Wire time includes transport/queue work
and is not pure inference. No billed USD total was verified; historical Free
Plan attestation is not an invoice measurement.

All raw requests/replies, partial captures, cumulative quota state, one-shot
marker, review packet, separate AI reviews, rescored report and hash-bound
resource verification remain ignored under
`output/private/phase4b-real-live-executed-20261027/`. **35 existing selected
offline quota/reservation/tool-binding/pacing/tokenizer tests passed** before
execution; packet integrity, bounds/spacing, preserved failed reservations and
unchanged runtime-source hashes were checked afterward. No full suite/package
build was rerun for this report-only update. **Phase 4B remains open because the
real three-mode comparison is incomplete.**
