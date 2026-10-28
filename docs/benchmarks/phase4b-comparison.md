# Phase 4B: direct / grounded / agentic

**Phase 4B comparison is ready for closure, with a negative Agentic result.**
The final Structured Outputs run recorded all six workflows: Direct 2/2,
Grounded 2/2 and Agentic 1/2 complete supported answers under separate unblinded
AI source review, not human review. All 12 HTTP calls succeeded; Agentic toolkit
stopped safely at the context guard. No Agentic retrieval/quality advantage was
observed. This completes the comparison boundary, not all-answer-success or
production/vehicle-readiness acceptance. See the
[final Structured Outputs result](#final-structured-outputs-comparison--completed-with-bounded-stop).

PR #5 also contains a subsequent, **offline-only** Agentic context-management
change. Its [separate validation](#post-comparison-agentic-context-management--offline-only)
does not change any historical score or establish successful toolkit recovery.

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

## Planner protocol correction — offline acceptance

The two preserved live captures fail at the same boundary: the second Agentic
planner request contains a valid first native search/result pair, then Groq
returns HTTP 400 / `output_parse_failed` with free-form generated prose. Both
requests advertised only search functions with `tool_choice=auto`, while the
prompt also allowed plain-text `FINALIZE` / `REFUSE`. This establishes a repeated
format failure, not a proven internal Groq parser cause. No retry, new model,
reasoning-setting change or new live inference was used to diagnose this change.

The minimal selected correction keeps the two existing search functions and adds
strict native **`finalize({})` / `refuse({reason})`** actions. The planner now uses
`tool_choice=required` with serial calls and no plain-text control path. Groq's
[local tool documentation](https://console.groq.com/docs/tool-use/local-tool-calling#controlling-tool-use-behavior)
specifies that required mode requests tool use; it can still return HTTP 400 if
the model emits no tool. This is a protocol correction, not proof that Groq can
no longer fail.

Prompt-only changes with `auto` retain two output grammars. Required search-only
tools prevent a clean early stop; that option was rejected. Replacing native
retrieval with JSON planner output would require a different execution protocol.
Native terminal actions keep one planner grammar while preserving autonomous
method/query/top-k choice and completion/refusal without an extra search.

Terminal actions are consumed locally: they do not execute retrieval, create
sources, emit an answer or bypass the separate JSON writer and mandatory audit.
The writer receives the last validated search/result pair with all accumulated
evidence, excluding the terminal call. No-context finalize refuses. Unknown/
extra arguments, batched actions, reused call IDs, mixed text/actions, repeated
searches and provider failures stop without repair/retry. The score-filtered
path removes BM25 but keeps both terminal actions. Existing call/search/token/
step, context, numeric, citation, applicability and quota/session guards remain.
Direct/Grounded prompts and wire formats, frozen datasets and historical reports
are unchanged. The existing scripted evaluator emits a native finalize action
for the new protocol; that remains fixture contract data, not quality evidence.

Offline regressions exercise the actual graph and Groq adapter through
`httpx.MockTransport`: a successful second planner turn after a local fixture
result may search again, finalize or refuse. Finalize/refuse consume no extra
search; writer/audit keep JSON mode without active tools. The preserved HTTP 400
class still stops immediately, retaining failed reservations and missing usage.
Transport fixtures cannot establish provider inference reliability.

The toolkit result remains complete on original physical PDF page **23**, already
sent to Grounded. The correction does not turn that case into an Agentic
retrieval advantage or alter its frozen gold labels. Phase 4B remains open.

Offline acceptance for this correction: **399 tests passed** with both provider
keys blank and model downloads disabled; `ruff check .`, diff whitespace checks
and sdist/wheel builds passed. The focused new tests first failed on the missing
native-control behavior, then passed with the implementation. Zero new live API
attempts; offline acceptance does not certify Groq's server-side parser fix.

### Preregistered smoke specification — subsequently authorized below

After offline acceptance, propose **one protocol-only two-request smoke**, not a
new benchmark or quality comparison. Use the existing exact EN `hood_release`
question, original BMW E39 PDF/index and baseline local retrieval; Groq
`openai/gpt-oss-120b`, temperature zero, unchanged reasoning settings, pinned local
tokenizer and **1,024 completion tokens including reasoning per request**.
Hard ceilings: **2 API attempts / 16,000 held input+output tokens / 1 local
search**, actual HTTP starts at least **61 seconds apart**. Require fresh binding,
rolling cumulative quota/cooldown and full-budget checks from the latest real-run
checkpoint (including both historical failed reservations); no overlapping
snapshot addition, retries, fallback, resume, writer or auditor calls.

A bounded execution-only probe would use the production planner prompt, closed
native schemas, request reservation and call/result validation. First call starts
empty; the model freely chooses its search method/query/top-k. Execute that one
validated real local search and deliver its exact accumulated result under its
actual call ID to the second planner call, with the same native action protocol.
**Success requires an HTTP 200 second response containing exactly one valid
native search/finalize/refuse action after consuming that real tool result.**
Stop there; a proposed second search is recorded but not executed. This measures
the previously failing continuation boundary, not answer quality or a completed
Agentic workflow. If the first response is terminal, no sources are obtained,
any guard/error occurs, or the second response is invalid, stop and preserve
partial data; no extra request to obtain the desired path. Record raw request/
response hashes, call IDs, sent sources, reservations, reported/missing usage
and HTTP spacing. **No API request, live runner or scheduled launch is installed
by this proposal.**


## Authorized native planner smoke — continuation passed

On **2026-10-27 18:21:30–18:22:32 UTC / 19:21:30–19:22:32 Europe/Berlin**, at
**`e7626c8`**, the one explicitly authorized diagnostic ran the existing EN
`hood_release` question against the original BMW E39 PDF and real local index.
It used an ignored isolated runner with the production `AGENT_PROMPT`, closed
native schemas, `_Budgeted` planner adapter, shared attempt/token/pacing guard,
`LocalSearch`, context builder and native call/result validators. It did **not**
invoke the full workflow. Production code, prompts, model/reasoning settings and
benchmark data were not changed for execution.

Preflight verified clean pinned code, PDF SHA256/186 pages, canonical dataset
hash, baseline index fingerprint/241 active chunks, logical chunk digest and
cached pinned tokenizer. The latest cumulative checkpoint was restored, not
combined with overlapping snapshots. Rolling 86,400-second aging retained
**15 requests / 23,316 tokens**, including both historical failed reservations
(**4,040 and 3,847**), leaving **985 requests / 176,684 tokens** locally. The full
**2-attempt / 16,000-token** allowance fit and no known cooldown remained.
Historical same-key Free Plan attestation matched; current organization-wide
balance/plan was not independently verified. No extra API quota probe was sent.

The first planner freely selected `dense_search`, query
`lever that unlocks the hood location`, top-k **5**. Exactly one real Qdrant/E5
search returned physical PDF pages **137, 73, 66, 116, 37**. No warmup, BM25 call,
fixed-query override, gold injection or ingestion occurred. BM25 remained
available through the production native tool schema but was not selected.
The complete actual source set was sent in a native tool result with the first
assistant call's unchanged ID. Production prompt bounds reflected this probe:
searches remaining **1**, then **0**, top-k cap **5**.

**The second planner returned HTTP 200, `finish_reason=tool_calls`, and exactly
one valid native `finalize({})` with a fresh call ID and no mixed text.** The
request contained `system/user/assistant/tool` messages, the matching first
call/result ID, the exact saved five-source set and unchanged native schemas.
Both calls used Groq `openai/gpt-oss-120b`, temperature zero,
`tool_choice=required`, `parallel_tool_calls=false`, existing reasoning settings
and **1,024 completion tokens including reasoning**. The diagnostic stopped
immediately after validating that response: **no second search, writer or audit**.

| Resource | First planner | Second planner | Total / limit |
| --- | ---: | ---: | ---: |
| HTTP status | 200 | 200 | 2 attempts / 2 |
| Held input+output reservation | 1,745 | 3,954 | **5,699 / 16,000** |
| Reported input tokens | 565 | 2,558 | **3,123** |
| Reported completion tokens | 51 | 42 | **93** |
| Reasoning tokens, included in completion | 9 | 24 | **33** |
| Reported actual total tokens | 616 | 2,600 | **3,216** |
| Wire HTTP time | 0.800 s | 0.707 s | **1.506 s** |

Actual HTTP starts were **61.080 seconds apart**. The diagnostic body took
**61.830 s**, including **57.454 s** of execution-only pacing and **2.739 s** for
the one cold local search. Initialization/preflight before that body is excluded;
there is no full-workflow latency measurement. Provider-reported completion
generation totaled 0.197 s; wire time also includes transport/queue work.
Both usage records are present, so `token_usage_complete=true`. Session held
reservations stayed **5,699** without refunds; provider-local rolling history
records known actual usage, totaling **26,532 tokens**, while preserving both
historical unknown-usage failure reservations. Final response headers reported
RPD 998/1,000 and TPM 5,391/8,000 remaining; these snapshots do not verify TPD or
organization-wide balance. No billed USD amount was measured.

**Result: the preregistered continuation criterion passed once.** This is live
evidence that this revised planner consumed a real tool result and returned a
valid native completion action on its second request. It is not a complete
answer/workflow, independent source review, statistical reliability estimate,
proof of the old backend cause, or guarantee against future parser failures.
Phase 4B's three-mode quality comparison remains incomplete. Grounded's toolkit
answer remains supported by its already-sent page 23; this smoke establishes no
Agentic retrieval-quality advantage.

Exactly two API attempts and one local search occurred. There were no errors,
retries, fallback, automatic resume or subsequent experiments. A durable one-shot
marker prevents rerunning this authorization. The partial/final checkpoints,
raw wire replies, first/second planner actions, actual sent sources, quota state
and hash-bound verification remain ignored under
`output/private/phase4b-native-smoke-e7626c8-20261027/`.
**40 existing selected offline quota/reservation/pacing/native-protocol tests
passed** before execution. Post-run checks verified the two HTTP 200s, closed
action schemas, native pairing, source-set identity, spacing, limits, unchanged
runtime-source hashes and retained historical failed reservations. No full suite
or package build was rerun for this report-only update. No merge, tag or Release.


## Final authorized comparison — partial, not ready for closure

The user authorized one further full comparison after the native smoke, with
**`hood_release`, then `toolkit_access`; Direct → Grounded → Agentic for each**.
This explicitly changes the original randomized schedule to preserve a usable
Direct/Grounded comparison if Agentic fails. The six-row order and reason were
saved in a timestamped/hash-bound private protocol amendment **before generation**.
An execution-only scheduling reference orders the existing evaluator; no
production workflow, prompt, question, retrieval rule or frozen benchmark changed.
Seed `20261027` is retained as metadata, **not a randomized-order claim**. The
original frozen protocol and earlier measurements remain preserved.

The run used PR code **`65da46f`** (native runtime from `e7626c8`) on
**2026-10-27 18:28:46–18:32:51 UTC / 19:28:46–19:32:51 Europe/Berlin**.
Groq `openai/gpt-oss-120b`, temperature zero, existing reasoning settings and
**1,024 completion tokens including reasoning** were unchanged. Hard session
caps were **22 attempts / 128,000 no-refund reserved input+output tokens**, with
actual HTTP starts at least **61 seconds apart**, no retry/fallback/resume and
stop on the first error.

### Preflight and actual stop

Original PDF SHA256/186 pages, canonical dataset hash, baseline index fingerprint,
241 active chunks, frozen logical chunk digest, local pinned tokenizer and clean
code bindings matched. Latest cumulative history came from the successful native
smoke, including historical failed reservations **4,040 and 3,847**. Rolling
86,400-second aging retained **17 requests / 26,532 tokens** and left
**983 local requests / 173,468 tokens**, enough for the full session ceiling.
No overlapping histories were added or midnight reset applied; no known cooldown
remained. The same historically verified Free Plan key matched. Current
organization-wide balance/plan was not independently verified, and no extra API
quota probe was sent. Both provider/local guards and a one-shot marker remained.

The order actually reached was **hood Direct → Grounded → Agentic**. Attempts
1–4 returned HTTP 200. Attempt **5** returned **HTTP 400**, error type
`invalid_request_error`, code **`tool_use_failed`**, message that required tool
choice was unmet. Saved failed generation was **`{"action":"finalize"}`** as
plain JSON text, not a native `tool_calls` object. The experiment stopped
immediately, exit status 2. **All three toolkit modes were not run.** Failed
usage is unknown; its full **3,467-token reservation** remains held.
This error differs from the earlier `output_parse_failed` responses but occurs
at the same second-planner-after-tool-result boundary. No local repair, retry,
conversion of that JSON into a tool call, or further live diagnosis was attempted.

### Source quality and actual retrieval

Direct and Grounded each used the original dense top-5: physical PDF pages
**137, 73, 66, 37, 68**. Their initial **entire HTTP request bodies**, sent source
hashes, drafts, final answer text and cited quote are identical. The sole claim
correctly locates the lever under the left-hand side of the instrument panel,
with a valid exact quote/document/physical-page-137 mapping. Grounded performed
its audit without retrieval expansion.

A separate Codex AI reviewer fixed source judgments before reading auditor
verdicts or costs, inspecting the original PDF, a fresh page-137 render and all
**15 sent chunk/page checks**. For **each** of the two non-error answers:
**1/1 supported claims, complete_answer=true, 1/1 supported citations**, preserved
applicability and no relevant warning omission for the location-only question.
Answer hashes differ because the workflow reason metadata differs. Grounded's
same-model `accept`/`complete=true` agrees here, but did not supply the labels.
Review is **unblinded separate AI assessment, not independent human evaluation**.
Hash-bound claim/source labels and per-claim citation/warning notes were saved
and the existing offline rescoring path succeeded.

Agentic autonomously selected **`lexical_search` (real BM25)**, query
`lever that unlocks the hood`, top-k **5**, returning/sending pages
**137, 33, 6, 73, 102**. Its actual planner context already contained sufficient
page-137 evidence. The following request preserved the native search/result ID
and advertised all four required serial actions, but produced the provider error
above before writer/audit. Its generic user-facing refusal is **not a scored
semantic false refusal**. It obtained no necessary fact absent from Grounded;
**no retrieval recovery or quality advantage was observed**.

The previous diagnostic smoke used one remaining search then zero and dense
results; the full workflow used its unchanged normal remaining-search budget
(four then three) and model-selected BM25 results. The continuation inputs were
therefore different. The smoke's valid native finalize remains a real single
success; it did not demonstrate reliable native termination under the normal
full workflow. This run demonstrates a remaining output-contract failure, without
proving the provider's internal cause or warranting a speculative production fix.

Toolkit was not measured in this run. Its earlier complete Grounded result on
already-sent **page 23** remains intact; it cannot be counted as a matched row
here or described as evidence of an Agentic retrieval advantage.

### Resources and matched comparison

| Hood mode | Outcome | API / searches | Held tokens | Reported input + completion | Paced workflow wall | Wire HTTP, including failure |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Direct | Complete, source-supported | 1 / 1 | 3,879 | 2,627 + 222 = **2,849** | 0.937 s | 0.850 s |
| Grounded | Same complete answer | 2 / 1 | 7,914 | 5,410 + 492 = **5,902** | 122.154 s | 1.460 s |
| Agentic | Provider error before writer/audit | 2 / 1 | 5,212 | 565 + 55 = **620**, plus unknown failed usage | 121.586 s | 1.066 s |
| Toolkit, all modes | Not run after stop | 0 / 0 | 0 | Not run | — | — |

Total: **5/22 API attempts**, **17,005/128,000 held tokens**, **9,371 reported
actual tokens plus unknown failed-call usage** (8,602 input + 769 completion,
including **548 reasoning tokens**). Overall usage is incomplete, not zero for
the failed request. Three workflow searches plus **one separately excluded local
warmup** occurred. Actual HTTP intervals were **61.082 / 61.087 / 61.078 / 61.087 s**.

Whole execution was **248.383 s**, including **3.730 s** initialization before
first HTTP. Wire time totaled **3.376 s** including failure; execution-only pacing
was **240.955 s**, also including the wait before the failure (0 / 120.584 /
120.371 s by Direct/Grounded/Agentic). These paced, serial-order workflow wall
times do not measure intrinsic latency differences. Known successful provider
completion generation totaled **1.611 s**; failed inference time is unknown.
No billed USD total was verified. Latest cumulative quota checkpoint retains
all historical unknown-usage reservations plus the new failed 3,467 reservation;
response-time remaining headers do not establish organization-wide/TPD balance.

Using existing `paired_summary` separately for each pair gives **1 matched
Direct/Grounded pair, 0 Direct/Agentic pairs, 0 Grounded/Agentic pairs and 0 full
triples**. The source-reviewed D/G pair is a quality tie. Grounded used one extra
audit call and **5,902 versus 2,849** reported tokens for the same answer; that
single easy, preselected case does not establish a general mode ranking. Agentic
failed with incomplete cost data and no quality-scored final answer; no quality
pair or efficiency superiority can be inferred from it. Missing/error rows are
not promoted to semantic refusals, and historical runs are not pooled into a
completed six-row comparison.

**Closure decision: NOT READY.** Only two of six planned answers completed;
there is no successful full-workflow Agentic answer or matched three-mode
comparison. Native termination remains unreliable in the observed full-workflow
continuation despite offline acceptance and one passed smoke. The bounded stop
and preservation of the Direct/Grounded pair worked; they do not satisfy the
remaining live acceptance boundary. No representative, statistical, multilingual
or vehicle-safety conclusion follows from this small selected sample.

Private amendment/order events, requests/replies, captures, reviews, pairwise
rescoring, resource verification, cumulative quota state and one-shot marker are
ignored under `output/private/phase4b-final-live-65da46f-20261027/`.
**41 selected existing offline guard/protocol/pairing tests passed** before the
run. Packet integrity, fixed actual order, full-request D/G identity, unchanged
runtime source hashes, resource caps/spacing and failed-reservation retention
were verified afterward. No full suite/build was rerun for this report-only
update. There was no additional experiment, production change, merge or Release.


## Structured planner revision — offline acceptance, live not authorized

The latest preserved failure emitted text `{"action":"finalize"}` under required
native tools. Groq planner now uses a **single strict JSON decision schema**,
without tools/tool_choice in those requests. `search` chooses dense/BM25, query
and top-k; finalize/refuse have no search side effect. All five schema fields
are required; unused fields must be null. Local validation rejects incompatible
combinations, unknown/extra arguments and invalid searches. The old captured
minimal JSON is rejected for missing required fields, not silently repaired.

Local search history becomes ordinary assistant/user JSON for the planner.
The last locally generated search/result correlation pair supplies the existing
writer path, preserving accumulated sources and warning/conflict checks. It is
not represented as a provider-generated native call. Writer/auditor, Direct/
Grounded, search/context/step/call ceilings and shared quota/session guards stay
in place. Schema overhead is reserved alongside messages and full completion
allowance; exact response-format hashes are captured and verified in review
packets. Groq never falls back to the retained optional non-Groq native path.
No model answers, questions, corpus, retrieval rankings or BMW gold labels changed.

Offline tests cover the actual graph and transport: adaptive local searches,
second-planner finalize/refuse, schema incompatibilities, capture integrity,
shared no-refund budget enforcement and preservation of errors/unknown usage.
All three preserved private 400 responses are replayed exactly where locally
available; portable error-class cases also run without private captures. These
are contract regressions, not live inference or quality measurements. Earlier
failed runs, successful diagnostic and page-23 toolkit finding remain intact.
Phase 4B remains open pending a complete source-reviewed live comparison.

Offline acceptance: **416 local tests passed**, including three exact private
error-response replays; those three skip when private captures are absent, while
portable error-class regressions still run. Ruff, diff checks and sdist/wheel
builds passed. Zero live API requests were made for this revision. This does not
certify Groq inference reliability or complete Phase 4B.

Proposed **one final comparison, not authorized or launched here**: unchanged
hood_release then toolkit_access; **Direct → Grounded → Agentic** per question,
one trial, six planned outputs. Original PDF and frozen baseline Qdrant/BM25,
Groq GPT-OSS 120B, 1,024 completion tokens including reasoning, maximum **22 API
attempts / 128,000 held tokens**, actual starts **>=61 s** apart. Before any
separately authorized execution, verify the exact new code/schema/tokenizer and
corpus bindings, latest cumulative rolling quota history including every failed
reservation, full worst-case headroom and cooldown. Use one shared guard/marker,
stop the entire run on the first error; no retry/fallback/resume or tuning.
Save all actual contexts and resources including schema/pacing/failures, then
perform separate source-based AI review with explicit non-human provenance,
existing hash-bound rescoring and matched pair/triple summaries. The prior
native-protocol runner is incompatible and must not be reused unchanged; no
launch or automation is installed by this proposal.


## Final Structured Outputs comparison — completed with bounded stop

**Closure decision: READY for closure of Phase 4B as an evidence-collection and
comparison stage, with the negative result below.** Six planned workflow rows
were executed and separately source-reviewed. Five are complete positive answers;
one is a safe, incomplete Agentic toolkit stop. This is not six successful answers,
a claim that Agentic is production-ready, or authorization to merge/release.
Earlier HTTP failures, the diagnostic success, incorrect preparation inference
about toolkit evidence and all historical measurements remain preserved above.

### Frozen execution and integrity

The single user-authorized run used **`b41dac9`**, the new production Groq
Structured Outputs planner, on **2026-10-27 18:52:57–19:04:10 UTC /
19:52:57–20:04:10 Europe/Berlin**. Its protocol/order was durably recorded before
generation: **hood_release then toolkit_access, Direct → Grounded → Agentic for
each, one trial, six planned rows**. The existing evaluator's execution-only
schedule was fixed; the retained seed is metadata, not a randomization claim.
No workflow/prompt/retrieval tuning or benchmark change occurred during execution.

A newly written isolated Structured Outputs runner was used, not the old native
runner. HTTP hooks enforced no `tools`, `tool_choice` or parallel-tool options in
any request. All four planner requests used the exact hash-bound production
`json_schema` with `strict=true`, ordinary chat history and no native call/result
messages. All four returned locally valid structured decisions. Existing
JSON-object writer/auditor paths retained their validated **locally generated**
correlation pairs; these are not provider-generated native calls. The same Groq
GPT-OSS 120B model, temperature zero, reasoning settings and **1,024 completion
tokens including reasoning per call** were preserved.

Preflight verified clean pinned code, original PDF SHA256/186 pages, canonical
dataset hash, baseline fingerprint/241 active chunks, logical chunk digest and
pinned cached tokenizer. Latest cumulative quota history retained **15 requests /
31,289 tokens** after rolling 86,400-second aging, leaving **985 local requests /
168,711 tokens**, enough for the full **22-attempt / 128,000-token** allowance.
Historical failed reservations **4,040 / 3,847 / 3,467** remained intact; no
snapshots were added twice, midnight reset or known cooldown ignored. Historical
same-key Free Plan evidence matched; current organization-wide balance/plan was
not independently verified and no extra API quota probe was sent.

### Separate quality review

A separate Codex AI reviewer fixed criteria before generation and judgments before
consulting workflow auditors or costs. It checked original PDF text plus fresh
renders of physical pages 137 and 23, all **20 unique actually sent source objects**,
claim entailment/applicability, every requested part, quote/document/page mapping
and relevant warning preservation. The compatible claim/source labels and per-claim
citation/warning notes are bound to exact answer/context hashes. Existing packet
integrity and offline rescoring passed. This is **unblinded separate AI review,
not independent human evaluation**; no physical vehicle validation occurred.

| Question | Direct | Grounded | Agentic |
| --- | --- | --- | --- |
| Hood lever location | Complete, p137 | Same complete answer, p137 | Same complete answer, p137 |
| Toolkit location and access | Complete, p23 | Complete, p23 | Incomplete; `context_budget_exhausted` |
| Complete supported answers | **2/2** | **2/2** | **1/2** |
| Supported claims / reviewed positive claims | **2/2** | **2/2** | **1/1** |
| Supported citations / reviewed positive citations | **2/2** | **2/2** | **1/1** |

All hood answer texts are identical and correctly locate the lever under the
left-hand side of the instrument panel. Toolkit Direct/Grounded are semantically
equivalent, specifying the lid/tailgate-mounted kit and wing-screw opening.
Their original page **23** is a valid alternative to frozen gold page 150; those
p150-only match flags remain false without invalidating the source-reviewed
citations. Quotes match source wording/page after whitespace normalization;
Grounded's toolkit quote is **not a byte-identical multiline substring**, recorded
as a limitation rather than claimed exact-byte identity. No relevant location/
access warning or applicability condition was omitted by the five positive answers.
Three same-model audits agreed, but were consulted only after labels were fixed.

### Actual retrieval and bounded stop

Direct/Grounded initial entire HTTP bodies are identical for each question.
Both use dense top-5; neither Grounded execution expanded retrieval:

- Hood Direct/Grounded sent **137, 73, 66, 37, 68**. Agentic chose dense query
  `lever that unlocks the hood location`, top-k 5, sent **137, 73, 66, 116, 37**,
  then structured finalize, writer and audit. Page 137 was already available to
  Grounded. There is no recovered fact or quality improvement.
- Toolkit Direct/Grounded sent **23, 180, 85, 31, 100**, already sufficient.
  Agentic chose dense `onboard tool kit location access vehicle manual`, top-k 5,
  then sent **180, 182, 181, 8, 31** to its second planner: index pointers/general
  material insufficient for the requested location/access facts.
- Agentic's second toolkit decision was another dense search,
  `BMW E39 onboard tool kit location access`, top-k 5. The search executed, but
  the accumulated whole evidence could not fit the unchanged lossless context
  guard. Execution stopped before committing/sending that new context or invoking
  writer/auditor. **Rejected second-search raw hits were not saved** by the existing
  evaluator; their exact contents/ranks are unknown and were not reconstructed
  with another retrieval. No target-page recovery can be credited from that search.

All Agentic searches were model-selected dense searches. Genuine BM25 remained
available but was not chosen; no query equality fixture, gold injection or forced
route was used. The toolkit refusal is a **local context-guard stop**, not a
provider failure or raw model abstention. The legacy corpus-relative metric counts
one false refusal because the corpus is answerable; that must not be interpreted
as generator over-abstention. Its actually sent context was insufficient. The
lossless guard preserved evidence rather than silently dropping possible warnings
or conflicts. **No Agentic retrieval recovery or superiority was observed.**

### Calls, tokens and timing

| Case / mode | API / workflow searches | Held tokens | Reported input + completion = total | Paced workflow wall | Wire HTTP |
| --- | ---: | ---: | --- | ---: | ---: |
| Hood / Direct | 1 / 1 | 3,879 | 2,627 + 222 = **2,849** | 1.108 s | 1.014 s |
| Hood / Grounded | 2 / 1 | 7,914 | 5,410 + 491 = **5,901** | 122.622 s | 2.496 s |
| Hood / Agentic | 4 / 1 | 14,905 | 7,919 + 664 = **8,583** | 243.673 s | 2.984 s |
| Toolkit / Direct | 1 / 1 | 3,225 | 1,973 + 422 = **2,395** | 62.778 s | 2.534 s |
| Toolkit / Grounded | 2 / 1 | 6,636 | 4,132 + 608 = **4,740** | 120.677 s | 2.013 s |
| Toolkit / Agentic | 2 / 2 | 7,052 | 2,786 + 189 = **2,975** | 121.606 s | 1.057 s |

Totals: **12/22 API attempts, all HTTP 200; 43,611/128,000 held tokens** without
session refunds. Actual usage is complete at **27,443 tokens = 24,847 input +
2,596 completion**, including **1,854 reasoning tokens**. No current-call usage is
missing. Seven workflow searches plus **one separately excluded local warmup**
occurred. By mode: Direct **2 calls / 2 searches / 5,244 actual tokens**, Grounded
**4 / 2 / 10,641**, Agentic **6 / 3 / 11,558**. Agentic toolkit's lower token total
than Grounded reflects stopping without an answer, not an efficiency advantage.

Actual HTTP starts were **61.075–61.095 s apart**. Whole execution was
**676.203 s**, including **3.580 s** initialization before first HTTP, **659.448 s**
execution-only pacing and **12.098 s** wire time. Known provider completion
generation totaled **5.445 s**. Paced workflow times are dominated by the fixed
serial schedule/inter-mode waits and do not establish intrinsic latency rankings.
Wire time includes transport/queue work and is not pure inference. No billed USD
total was measured. Saved cumulative history ends at **27 requests / 58,732
local rolling tokens**, retaining all three historical unknown-usage failure
reservations; known current usage does not refund the separate session guard.
Response headers remain snapshots, not organization-wide/TPD guarantees.

### Matched result and closure scope

Existing `paired_summary` yields **two matched workflow triples and two matched
comparisons for each mode pair**, including the recorded bounded refusal. This
must not be described as two complete-positive Agentic answers. Hood ties across
all modes; toolkit Direct/Grounded are complete while Agentic is incomplete.
Direct/Grounded tie on both needs. Agentic spent more aggregate calls/tokens
without improving source-reviewed quality. Two previously used selected needs,
one trial and an unblinded AI review establish no significance, general ranking,
multilingual parity, safety or broad reliability claim.

**Phase 4B can close as a completed, reviewed comparison with these negative
findings.** The previous missing-matrix/provider-error boundary was completed in
this run: six recorded workflows, both strict planner continuations HTTP 200,
full resource accounting and separate source review. Agentic toolkit answer
success and a production-readiness claim remain unsupported. No requirement was
relaxed to turn the safe stop into a positive answer, and no subsequent experiment
or fix is proposed by this report. PR #5 remains unmerged; no Release.

Private protocol, order events, raw requests/replies, captures, review packet,
labels/notes, pairwise rescoring, resource verification, cumulative quota history
and one-shot marker remain ignored under
`output/private/phase4b-structured-final-b41dac9-20261027/`.
**47 selected existing structured-planner/quota/reservation/pacing/pairing tests
passed** before execution. Post-run verification checked fixed order, exact schema,
local validation of all four planner decisions, absence of native provider calls,
D/G request identity, context hashes, unchanged runtime source hashes, caps/spacing,
complete usage and historical reservation retention. No full suite/build was rerun
for this report-only update. No retries, fallback, resume, prompt/retrieval tuning,
additional experiments, merge or Release occurred.


## Post-comparison Agentic context management — offline only

Implemented on 2026-10-27 after the historical six-workflow comparison above.
**Historical outcomes remain Direct 2/2, Grounded 2/2, Agentic 1/2.** No provider
API requests, new live labels, merge, tag or Release are part of this change.

### Audit and cause

`generation.build_context` deduplicates document/page/text and packs intact
passages in hit order, skipping passages that exceed its serialized character
budget. Direct/Grounded retain this unchanged primitive and their original
limits. Agentic instead requested 20 passages / 12,000 characters, reconstructed
previous evidence from only its sent source map (discarding retrieval scores),
then compared the entire combined identity set with the packed result. Any
missing identity stopped the workflow, irrespective of relevance. A second
search could therefore trigger `context_budget_exhausted` before new evidence
was committed or sent. The rejected raw hits were not captured. Repeated complete
source arrays also accumulated in planner history and consumed token budget.

The existing final live capture cannot recover those missing raw second-search
hits. The fresh offline retrieval below is a new observation, not a reconstruction
of unavailable historical evidence.

### Implementation boundary and guarantees

`agentic_context.EvidenceStore` now owns full per-search snapshots and provenance
(method/query/top-k/rank/score, original text and metadata). Exact text plus full
metadata deduplicate repeated hits without merging different revisions or
applicability. IDs are assigned on discovery and survive exclusion/re-entry.
`select_context` rebuilds the LLM subset from the whole store after each search.
Whole passages and original source/page mappings are preserved; excluded source
IDs cannot pass the existing citation validator.

On overflow, all lexical question/query matches, every search's first semantic
hit, recognized EN/DE/RU warnings/conditions, explicit safety/layout metadata,
suspect extraction and retrieved same/adjacent pages are protected together.
Only lower-ranked, unconnected, cue-free nonmatches may be excluded for budget.
Priority is question overlap, query overlap and reciprocal retrieval rank, then
discovery order; raw dense and BM25 scores are not compared. A protected set that
does not fit stops explicitly, as does overflow with no usable lexical relevance
anchors. No passage is clipped, summarized or replaced with a generated extract.

Planner history contains one current source payload plus bounded search history.
Public diagnostics contain counts, IDs, selection/protection/exclusion reasons,
blocked IDs, sizes and hashes, and the actual attempted planner/writer/auditor
source sets after token reservation. Raw results, including selection-rejected
searches, are available in private evaluation captures. PDF passages, queries,
metadata and paths are absent from the public diagnostics/report.

The 12,000-character/20-passage Agentic context ceiling, six LLM calls, four
searches, twelve graph steps and 24,000 default / 48,000 maximum conservative
no-refund token budget are unchanged. Full messages/history/schema/output
reservations still apply before every call, including multibyte text and the
mandatory audit. Provider/free-plan/session guards remain in force. Groq's strict
Structured Outputs planner/schema and Direct/Grounded are unchanged; the writer's
single context-description sentence now says selected rather than accumulated.
Writer validation and auditor logic are unchanged. Package 0.5.0 remains the
coherent upcoming version; no version increment or release is needed here.

These are retention and orchestration guarantees, not a semantic proof. Lexical
matching and safety cues cannot identify every unmarked warning, paraphrased
conflict or missing page continuation. Same-model auditing remains fallible.
Conservative protection can retain irrelevant warnings and refuse an otherwise
answerable question. No general safety/completeness or Agentic-quality claim is
made, and no check is weakened to obtain a positive answer.

### Separate offline acceptance

- New store/selector/graph coverage: overlapping searches and scores/provenance,
  irrelevant overflow, essential second-search evidence, stable IDs through
  eviction/re-entry, rejected citations, intact conditions/conflicts, EN/DE/RU
  warning cues, page continuations, extraction/layout uncertainty, deterministic
  limits, multibyte token reservation, private captures and sanitized public data.
- BMW E39: fresh genuine local E5/Qdrant searches using the two recorded planner
  queries, offline weights from the existing cache, without invoking a model API.
  They returned 10 hits / 9 unique passages. The first context fits (5 passages,
  8,031 serialized characters); the second protected set still exceeds the limit
  and safely stops. All 10 raw hits are now captured. This establishes inspectable
  bounded refusal, **not toolkit answer recovery**. The index manifest is unchanged.
  Private files: `output/private/agentic-context-offline-20261027/`; no source text
  is copied into this report. The private regression is skipped when captures are
  absent from a public checkout.

Local acceptance: **439 pytest tests passed**, including 23 new context-management
cases (one uses the private BMW capture); Ruff and `pip check` passed. Wheel and sdist
builds passed, their allowlisted contents include the new runtime/tests/docs and
exclude private material, and an isolated wheel installation passed import and both
CLI/module `--help` checks using the existing local dependency environment.
CI results for the delivered commit are recorded in PR #5; these local checks do
not substitute for remote CI. All model HTTP tests use mocked transports.

### One proposed bounded live check — not executed

After separate authorization, run only Agentic on the original BMW E39 toolkit
question, English, one trial, same GPT-OSS 120B/temperature zero and 1,024 completion
tokens including reasoning. Keep the strict planner schema, top-k <=5, at most four
searches/six HTTP attempts, a 24,000-token no-refund workflow/session reservation
cap, free-plan preflight, >=61-second actual HTTP spacing and fresh private paths.
No retry, fallback, resume, forced query sequence or other mode is included.

Inspect every selection and actual source delivery, citation/source/page integrity,
conditions and warnings against the original PDF, and final writer/audit output if
reached. A model-selected route repeating the offline protected overflow must
remain a clearly diagnosed refusal; it is not a successful toolkit recovery.
Only an intact, supported, audited answer would establish recovery for this single
case. Either outcome is new evidence, separate from the frozen Phase 4B scores.


## Scoped context diagnosis after 1790e58 — offline only

This follow-up reuses the immutable two-search private capture from the preceding
change. No live model API, ingestion, model/index change, merge, tag or Release.
Historical Phase 4B remains **Direct 2/2, Grounded 2/2, Agentic 1/2**. The prior
1790e58 findings above describe that implementation and are not relabeled.

### Every formerly protected passage

Sizes below are text / compact JSON object characters, including original metadata
in the second number. `1:4;2:5` means search 1 rank 4 and search 2 rank 5. Topic
classifications come from inspection of the complete private text; no PDF excerpts
are published. Protection codes are the exact old diagnostic reasons:
T=`topic_match`, L=`semantic_lead`, W=`safety_or_condition`, P=`page_companion`.

| ID | PDF page | Text / JSON chars | Search:rank | Relevance to toolkit access | All old reasons | New treatment |
| --- | ---: | ---: | --- | --- | --- | --- |
| S1 | 180 | 1474 / 2018 | 1:1 | Useful index pointer, no procedure | T,L,P | Optional navigation, selected after two searches |
| S2 | 182 | 1425 / 1966 | 1:2 | Other index entries; scattered shared modifiers | T,W,P | Off-topic navigation, excluded |
| S3 | 181 | 771 / 1274 | 1:3 | Other index entries; shared modifier only | T,W,P | Off-topic navigation, excluded |
| S4 | 8 | 1635 / 2163 | 1:4;2:5 | Manual introduction / symbol precautions | T,W,P | Unknown-scope precautions retained |
| S6 | 10 | 1370 / 1871 | 2:1 | General accessory/parts safety, no access procedure | T,L,W,P | Unknown-scope warning retained |
| S7 | 9 | 1348 / 1846 | 2:2 | General model/equipment and publication notes | T,W,P | Off-topic prose, excluded |
| S8 | 146 | 1255 / 1741 | 2:3 | OBD procedure, different subject | T,W,P | Explicit warning scope not proven unrelated; retained conservatively |
| S9 | 6 | 616 / 1111 | 2:4 | Contents pointer to the same destination as S1 | T,W,P | Duplicate navigation excluded after two searches |

S5 (page 31, 147 / 604 characters, rank 1:5) was never protected and is a section
separator with no procedure. Total nine-object context: **14,604 characters**.
The eight protected objects plus JSON framing require **13,999**, exceeding
12,000 by **1,999**. The old greedy diagnostic lists S6/S7 as blocked; they are
packing-order casualties, not the sole semantic cause of overflow.

### Rule attribution and correction

- Broad query words (manual/vehicle and the manufacturer qualifier) promoted
  S4/S6/S7 despite zero substantive question overlap. A single shared modifier
  promoted unrelated index material and OBD. General request/document words and
  model qualifiers are now removed from topic scoring; compound topics require
  a local pair of substantive terms. This uses no BMW/page-specific rules.
- The old global safety regex matched `after` in S3 and warning-related words in
  index titles, as well as `if/not/only` and other generic language in boilerplate.
  `before` was not a trigger in this particular capture; synthetic cases exercise
  it. Recognized reference lists are navigation, so a warning title plus page
  label is not mistaken for the actual warning. Real warning prose next to an
  index prevents that demotion. Text inside selected chunks is never removed.
- In the baseline, P adds **zero** new passages: all eight already have another
  reason, including self-page matches. Ablations removing query promotion alone
  or the global safety regex alone both still protect eight and overflow. Jointly
  removing them lets packing proceed but drops S4's unknown-scope precautions.
  Scoped selection instead retains S4/S6/S8 while dropping unrelated weak matches.
  Blanket adjacency is replaced by same-procedure-page and explicit dependency
  links, with unknown-scope explicit warnings still protected.
- Useful navigation is optional and deduplicated by document revision plus opaque
  destination label. It cannot displace mandatory evidence or make an answer
  sufficient. `has_substantive_evidence=false` stops writer/auditor with
  `insufficient_evidence`; planner/schema and answer/auditor logic are unchanged.

### Measured before/after, with explicit workflow provenance

First two searches replay exactly the saved raw hits. A **separate new local BM25**
search for the toolkit topic supplies the third batch; this was an offline,
implementation-selected diagnostic continuation, not an autonomous model decision.
It naturally returned four substantive candidates and a previously seen pointer.
The complete candidate set includes the access procedure, body-style applicability,
related operations and intact warnings/conditions. No gold passage was injected
or forced into retrieval, and the index manifest is unchanged.

| Observation | Found / unique | Protected | Protected JSON chars | Selected / excluded | Final context chars | Controlled graph outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Before, saved two searches | 10 / 9 | 8 | 13999 | 0 / 9 | 2 (empty array, not sent) | context_budget_exhausted; 2 planner calls |
| After, identical two searches | 10 / 9 | 3 | 5779 | 4 / 5 | 7798 | insufficient_evidence at the same two-search ceiling; no writer/auditor |
| After, separately added local third search | 15 / 13 | 7 | 10350 | 8 / 5 | 11462 | audited_evidence in controlled offline graph; 3 planner + writer + audit |

After two searches selected IDs are **S1,S4,S6,S8**; excluded are S2,S3,S5,S7
(off-topic) and S9 (redundant pointer). All three unknown-scope warning passages
remain intact. At three searches selected IDs are **S4,S6,S8,S9,S10,S11,S12,S13**;
S1 becomes redundant after repeated retrieval improves S9's rank priority, and
S2,S3,S5,S7 remain off-topic. Existing IDs never change. All seven protected
passages are retained in full; navigation fills the remaining optional slot.
Before the old overflow, the last actually delivered planner context was the
first five sources (8,031 characters); the empty rejected second selection was
never sent. After two searches at the same search ceiling, the 7,798-character
selection is retained locally but writing is blocked because it has no procedure.

The graph replay uses strict-schema **fixed local planner replies**, fixed writer
and audit fixtures, and the pinned GPT-OSS Harmony tokenizer with the existing
production reservation functions. Both before/after two-search replays use the
same lowered search ceiling of two; the extended diagnostic uses three (below
the unchanged ceiling of four). The default **24,000** token cap, six-call maximum
and 1,024 completion-token allowance remain unchanged. Reservations are
7,784 before, 6,750 after two, and **23,312 / 24,000** for the five-call extended
path. These are conservative held tokens, not measured provider usage. There are
zero model HTTP requests. The accepted fixture proves orchestration and local
citation/gate contracts, **not independent semantic or live toolkit recovery**.

Private diagnostics, ablations, exact raw hits and graph captures are stored under
`output/private/agentic-context-scoped-20261027/`; the original capture is unchanged.
The tests cover generic-query/top-rank overprotection, weak unrelated conditionals,
linked distant conditions, unknown-scope warnings, reference-title pitfalls,
navigation-only refusal, late evidence, full graph conflict refusal, and complete
private three-search context retention. The original Direct/Grounded and planner/
auditor implementation files and benchmark datasets remain unchanged.

Final local acceptance: **460 pytest tests passed**, including 21 new scope cases;
Ruff passed. EN/DE/RU regressions distinguish unrelated bare condition words from
unscoped explicit prohibitions/applicability clauses; both linked and ambiguous
mandatory conditions retain fail-closed budget behavior. Wheel/sdist build results
and delivered-commit CI are recorded in PR #5. Private-capture tests are skipped
when their ignored local inputs are absent from a public checkout.

### Remaining boundary and smallest architectural extension

The original two searches contain no toolkit access procedure; no selector can
safely manufacture an answer from their index entries and general warnings. A
further retrieval is necessary, and its autonomous model selection remains
unvalidated. Scope heuristics cannot prove that every unmarked cross-page clause
is irrelevant. When mandatory or unknown-scope explicit warnings exceed the
budget, selection still refuses instead of silently discarding them.

The smallest stronger mechanism is reviewable ingestion metadata for procedure/
section and warning applicability, plus bounded local retrieval of referenced or
missing continuation passages. The selector already honors explicit scope links;
this change does not invent them for the BMW corpus or add an unbounded reader,
new reranker, larger budget, or alternate model. No live test is part of this work.


## Wire accounting and remaining default-route limit — 2026-10-28

An additional controlled replay restored the **default four-search ceiling** and
therefore the extra structured finalize turn. This exposed a distinct cumulative
budget limit after context packing was corrected: at the scoped-selector commit
`641b2b9`, the three-search/default-route fixture stops at `token_budget_exhausted`
after four planner calls, before writing. Holding 18,597 tokens leaves too little
for the writer and its audit. The five-call result in the preceding table used
an explicitly lowered three-search ceiling and is not default-route acceptance.

Model wire payloads now omit only known index bookkeeping fields while full raw
metadata stays in Evidence Store and the local source map. Revision matching still
uses the original content hash. All document/page references, applicability,
warning/layout metadata, unknown metadata fields and passage text are preserved.
Full protection/dependency/overlap diagnostics and hashes remain in trace/captures;
model inputs get compact selected roles, exclusions, found/stored counts and the
substantive-evidence flag. No source text or required warning was dropped to make
this optimization. No context/token/call/output ceiling was increased.

| Final wire observation | Found / unique | Protected | Protected chars | Selected / excluded | Context chars | Actual controlled graph result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Same saved two searches, default ceiling four | 10 / 9 | 3 | 5101 | 4 / 5 | 6895 | insufficient_evidence; three planner calls, no writer/auditor |
| Additional local third search, default ceiling four | 15 / 13 | 7 | 8772 | 8 / 5 | 9660 | token_budget_exhausted; four planner calls + writer, audit not invoked |
| Same three batches, explicitly lowered ceiling three | 15 / 13 | 7 | 8772 | 8 / 5 | 9660 | audited_evidence in five-call fixture only |

Membership and original text are identical to the preceding scoped-selection
observations: S1,S4,S6,S8 after two searches; S4,S6,S8,S9,S10,S11,S12,S13 after
three. Only the wire bookkeeping/diagnostic representation changed. Mandatory
unknown-scope S4/S6/S8 remain intact; none were silently declared irrelevant.

The default three-search-plus-finalize path holds **20,823** tokens through the
writer. The mandatory audit requires **4,373**, but only **3,177** remain. The full
fixture would need **25,196**, exceeding the unchanged **24,000** by **1,196**.
The audit request is rejected before invocation, and the candidate is never exposed
as a supported answer. The three-search-ceiling fixture holds **19,898** for its
five calls. These are pinned-tokenizer conservative reservations, not provider
usage or live outcomes; all model replies and audit judgments are local fixtures.
No HTTP request or live semantic recovery is claimed.

The initial desired positive default-route regression was not supported by those
measured reservations. Its acceptance contract instead verifies the required safe
stop, retained candidate/evidence, unchanged cap and absence of an unaudited answer.
Other wire regressions verify compact diagnostics, intact source text, preserved
applicability and complete original metadata in the store/local citation map.
Full local acceptance: **463 pytest tests passed**, Ruff passed; build/CI evidence
for the delivered revision is recorded in PR #5.

This leaves a concrete architectural boundary. Reserving two *call slots* does
not reserve enough *tokens* for writing and auditing after another planner turn.
The smallest follow-up is a token-aware handoff that reserves a conservative
writer/auditor envelope before spending the next planner/finalize turn, reusing
the existing budget-ceiling route to writing once substantive evidence is available.
If that envelope cannot fit, refuse. The existing configured three-search ceiling
illustrates the shorter route without raising limits. This policy is proposed,
not implemented or represented as a default-route success here. For reliable
pruning of unknown-scope warnings, explicit procedure/warning scope metadata remains
the separate minimal extension described above.

New exact default-route/wire captures are under ignored
`output/private/agentic-context-wire-20261027/`; earlier captures and all historical
Phase 4B outcomes remain unchanged. PR #5 stays open, with no merge/tag/Release
and no live API experiment.
