# Groq Phase 3: four harder English live pairs

On October 26, 2026, the existing evaluator ran `false_entailment`,
`conflicting_sources`, `recovery` and `persistent_false_refusal`, one English
trial per mode per scenario, on Groq `openai/gpt-oss-120b`. All four pairs
completed: **11 of 20 allowed new API attempts**, with no errors, retries,
fallback, paid/account changes or repeat of the completed `cold_pressure` runs.

The observed benefit is narrow but real: **grounded recovered a missing fact
through an additional local search in `recovery`**. Direct correctly refused
because that fact was absent from its actual context. The other three pairs
were equivalent in source-relative correctness. No unsupported live draft
reached an auditor, so these results do not demonstrate extra hallucination
prevention by the semantic audit. No additional false refusal was observed.

## Execution and quota boundary

Free / Current Plan was checked in Console before inference. Recorded model
limits remained 30 RPM, 1,000 RPD, 8,000 TPM and 200,000 TPD. The run used 6 RPM,
temperature zero, the unchanged 2,048-token reasoning/output budget and the fixed
`on_demand` endpoint. Quotas and actual usage from the preceding scenario were
inspected before starting each next scenario. Every scenario reserved at most
five attempts; the local wrapper additionally checked cumulative reservations
against the 20-attempt stage ceiling and prohibited reusing existing case files.

All inference used the existing CLI evaluator configuration:

```text
--live --provider groq --model openai/gpt-oss-120b
--cases <one authorized case> --languages en --trials 1 --seed 20261025
--max-api-calls 5 --requests-per-minute 6
--output output/groq/gpt-oss-120b/hard-<case>-en.json
--private-captures output/private/groq/gpt-oss-120b/hard-<case>-en.json
```

The runtime checkout was `cf0f54b`. Production code, schemas, prompts, graph,
retrieval, frozen scenario data and evaluation methodology were unchanged.
Independent seeded runs selected direct before grounded. A local observation
wrapper retained the previous timing measurements, restored known quota history,
and captured the actual sent source sets and live JSON responses privately.
It did not substitute scripted replies or change model inputs. Every first
draft's complete message list was identical between the two modes in its pair.
No dataset `script` object was present in model requests. The scenario names
refer to adversarial offline fixtures; they do not force Groq to make those
mistakes live. Source text selection remains the original controlled synthetic
retrieval, not an embedding/corpus benchmark.

| Before next scenario | Stage attempts already spent | Known local daily tokens | Previous remaining RPD / TPM | Seconds since previous scenario's last request start |
| --- | ---: | ---: | ---: | ---: |
| `false_entailment` | 0 | 8,020 | 994 / 7,197 (historical, expired) | prior window already clear |
| `conflicting_sources` | 2 | 9,066 | 998 / 7,476 | 117.274 |
| `recovery` | 4 | 10,305 | 997 / 7,389 | 105.866 |
| `persistent_false_refusal` | 8 | 12,974 | 994 / 7,122 | 125.906 |

All inter-scenario gaps cleared the conservative local 60-second TPM window.
Known history was carried from the prior RU run through every new checkpoint.
The final headers reported **992 RPD / 7,192 TPM remaining** against limits
1,000 / 8,000. No header exposes a live RPM or TPD balance. Including prior work,
this client observed **22 successful calls / 15,074 reported tokens** on this
date. Those local observations remain below the recorded daily caps; they are
not an organization-wide billing ledger. Header refill/reset behavior and
Console's up-to-15-minute lag prevent reconstructing exact daily usage by
subtracting remaining RPD from 1,000. No diagnostic completion was spent probing
quotas and no failure budget was reclaimed.

## Per-scenario source review and comparison

The implementation agent inspected the exact request sources, raw live replies,
final claims and citation mappings. This inspection is separate from Groq's
own audit and is hash-bound to each case/language/mode/trial/answer. It is
**not independent human review** and does not establish vehicle applicability.
All sources are the existing fictional Aurora owner's-manual passages.

| Scenario | Direct observation | Grounded observation | Technical conclusion |
| --- | --- | --- | --- |
| `false_entailment` | Refuses: the tire-pressure passage contains no spare-key location. | Same valid refusal, then one top-20 search returns no new evidence; ends `no_new_context`. No auditor call. | Equivalent correctness; no live unsupported claim was generated for the audit to prevent. Grounded adds one unproductive search. |
| `conflicting_sources` | Refuses to choose between two unqualified, contradictory torque statements, 100 and 90 Nm. | Same valid refusal; expansion returns only the same conflicting passages; ends `no_new_context`. No auditor call. | Equivalent correctness. Conflict avoidance was already achieved by the common generator prompt, not demonstrated as an audit benefit. One extra search. |
| `recovery` | Correctly refuses with only the tire-pressure source; no key location is available. | Same justified initial refusal, then expansion retains S1 and adds S2 containing the key location. A new draft states the supplied fact, with an exact S2/PDF-page-2 quote; audit accepts. | Better end-to-end information coverage through extra retrieval. The direct refusal is safe given its sources. This is not proof that an auditor reasoned better over identical evidence. |
| `persistent_false_refusal` | Answers correctly from the sufficient initial source, with exact S1/PDF-page-1 quote. | Same correct claim/quote, accepted by audit, no expansion. | Equivalent correctness. The live model did not reproduce the scripted persistent refusal; no refusal repair was demonstrated. Extra audit cost without a quality difference. |

No unsupported claim, lost condition, invalid quotation or source mismatch was
observed. The two accepted grounded audits cover supported drafts only. The
negative cases never reached audit. Grounded was not worse in observed quality;
it was more expensive overall and performed two no-novelty searches. These four
small, single-trial synthetic needs do not estimate general accuracy, error
prevalence, false-refusal rates or robustness against plausible bad drafts.

## Calls, tokens, searches and latency

Input/output counts include reasoning usage as reported by Groq. Additional
searches are local and add no HTTP request themselves. All attempts have usage
records. Values below are single observations, not distribution estimates.

| Case / mode | API calls | Local searches | Input / output tokens | Total tokens | Workflow seconds | HTTP seconds | Limiter wait seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `false_entailment` direct | 1 | 1 | 457 / 67 | 524 | 0.752 | 0.730 | 0.000 |
| `false_entailment` grounded | 1 | 2 | 457 / 65 | 522 | 10.837 | 1.544 | 9.243 |
| `conflicting_sources` direct | 1 | 1 | 508 / 103 | 611 | 0.367 | 0.350 | 0.000 |
| `conflicting_sources` grounded | 1 | 2 | 508 / 120 | 628 | 10.338 | 0.668 | 9.628 |
| `recovery` direct | 1 | 1 | 457 / 66 | 523 | 0.602 | 0.586 | 0.000 |
| `recovery` grounded | 3 | 2 | 1,609 / 537 | 2,146 | 30.300 | 1.897 | 28.281 |
| `persistent_false_refusal` direct | 1 | 1 | 457 / 213 | 670 | 0.632 | 0.589 | 0.000 |
| `persistent_false_refusal` grounded | 2 | 1 | 1,052 / 378 | 1,430 | 20.086 | 1.258 | 18.776 |
| **Direct total** | **4** | **4** | **1,879 / 449** | **2,328** | **2.353** | **2.254** | **0.000** |
| **Grounded total** | **7** | **7** | **3,626 / 1,100** | **4,726** | **71.561** | **5.367** | **65.928** |

Stage total: **11 API calls / 7,054 tokens**. Relative to direct, grounded used
three additional calls (+75%), three additional local searches (+75%) and 2,398
additional tokens (about +103%). In `recovery` alone the added cost was two calls,
one search and 1,623 tokens, for one newly supported final answer. The separate
audit in the already-answerable last case added one call and 760 tokens.

The shared limiter charges the wait before a grounded draft to that grounded
workflow because direct ran first. Thus the roughly ten-second delay for a
one-call grounded refusal is largely scheduling, not slower inference or graph
reasoning. HTTP duration measures the complete actual synchronous request,
including network, provider queue and generation, excluding limiter sleeps.
Workflow time additionally includes local graph, parsing, validation, checkpoint
and observation overhead. Inter-scenario quota pauses are excluded from the
per-answer times above.

The raw provider-reported prompt+completion (`total_time`) sums were 1.144s
for direct and 3.609s for grounded, with separate `queue_time` sums 0.642s and
1.117s. These fields are retained per request in the timing artifact; they are
not independent measurements of server execution. Do not describe HTTP latency,
limiter waiting and provider-reported processing as interchangeable or use the
paced totals to claim a model-speed ranking.

## Preserve legacy metrics; distinguish available evidence

The existing dataset marks `recovery` as globally answerable because its expanded
source set contains the fact. `_row` uses that dataset-level label for both modes;
`summarize` therefore records **one direct false refusal and zero grounded false
refusals**. Those exact legacy summaries and labels are preserved in the result.
They describe end-to-end task coverage, including retrieval; they must not be
presented as proof that the direct generator declined an answer its context
actually supported.

The separate source-relative agent review classifies direct/recovery as a
**correct refusal with missing evidence**, not model over-abstention. Under that
explicit interpretation, direct has one supported complete answer and three
correct refusals; grounded has two supported complete answers and two correct
refusals. Neither has an observed false refusal given its actual available
sources. The different number of answerable contexts means these are not
identical-denominator model-accuracy rates. Both views are published without
rewriting frozen labels or weakening tests.

## Confirmed limitations and justified follow-up proposals

No incorrect workflow result or confirmed routing defect was observed, so this
run does not justify changing production prompts, graph logic or retrieval to
make metrics positive. Two specific limitations merit follow-up:

1. **Evaluation interpretation:** add an explicitly versioned, additive source-
   sufficiency review field/metric to distinguish corpus-answerable retrieval
   misses from false refusals despite sufficient sent evidence. Keep the current
   end-to-end labels/summary for comparability. Bind the new annotations to sent
   source-set and answer hashes, reject missing/ambiguous review rather than
   infer sufficiency from the model's own audit, and add a regression covering
   direct/recovery as an evidence gap versus genuine over-abstention. This report
   contains the annotations, but the evaluator implementation was not changed.
2. **Unproductive expansion and limited audit evidence:** a draft refusal exposes
   only `supported=false`, so the graph cannot distinguish conflict from missing
   evidence and makes its permitted expansion attempt. Two no-novelty searches
   are a measured policy cost, not a demonstrated correctness failure. Avoid a
   speculative shortcut based on hit count or a prompt change to eliminate them:
   it could destroy the useful recovery path. Any future refusal-cause or
   retrieval-exhaustion optimization needs its own conservative design, baseline
   preservation and controlled evaluation. First obtain live cases with plausible
   incorrect drafts before claiming the audit reduces unsupported answers.

To finish Phase 3, obtain independently reviewed claim/source/applicability
judgments; repeat suitably diverse, complete live comparisons across harder
EN/DE/RU needs and real matching corpus/profiles in explicitly budgeted segments;
measure actual unsupported-answer prevention and false-refusal tradeoffs; and
resolve/document source-sufficiency scoring before relying on aggregate quality
rates. Existing controlled adversarial regressions remain useful software tests,
but cannot replace those live or independent observations. No additional live
stage was started. PR #3 remains open; no merge, tag or Release was performed.

## Artifacts and verification

- [Reviewed live results and unchanged legacy summaries](groq-20261026-hard-en-results.v1.json).
- [Separate source-relative agent review](groq-20261026-hard-en-review.v1.json).
- [HTTP, limiter, search and provider timing observations](groq-20261026-hard-en-timing.v1.json).

Public results retain checkpoint/capture/timing hashes and redact absolute local
capture paths. Original checkpoints, actual request/response captures and local
measurement code remain private/ignored or outside the repository. All three
completed `cold_pressure` checkpoints match their previously published hashes.
Runtime source hashes and the frozen scenario dataset are unchanged.

Before inference, all 243 offline tests and Ruff passed with both API keys empty
and Hugging Face/Transformers offline. JSON, artifact/source/answer hashes,
identical first-draft inputs, live request count/timing, absence of scripted
request data, private-data exclusion and the final diff were checked for the
published evidence. CI is the remote acceptance check for the report commit;
this results-only pass adds no runtime code or tests and does not create a new
package release.
