# Phase 4B: direct / grounded / agentic

Offline software acceptance, 2026-10-26, based on `main` / `v0.4.0` (`da4b3cc`).
The authorized live preflight on 2026-10-27 stopped before generation (see below).
No live generation or independent human review was performed. **These results
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

## Authorized live preflight — blocked before execution

Checked **2026-10-27 08:51 UTC / 09:51 Europe/Berlin** at PR head `f9bbd87`.
This supersedes the earlier 18-output proposal. The only authorized experiment
was `follow_reference`, EN, Direct/Grounded/Agentic, one trial each, Groq
`openai/gpt-oss-120b`, temperature zero, `max_completion_tokens=1024` including
reasoning; at most **11 API attempts**, **64,000 reserved input+output tokens**,
and **61 seconds between HTTP starts**. No retries, fallback, automatic restart,
additional experiment, merge or Release.

| Local 24-hour quota check | Result |
| --- | ---: |
| Requests retained / local RPD cap | 87 / 1,000 |
| Tokens retained / local TPD cap | 162,873 / 200,000 |
| Available local token headroom | 37,127 |
| Required full-run reservation | 64,000 |
| Reservation shortfall | 26,873 |
| New API attempts / actual tokens / reserved tokens | 0 / 0 / 0 |

The latest persisted cumulative quota state was saved on 2026-10-26 at
18:37:27 UTC. Its SHA256 is
`804e191fe97b162cb4cce86fc8217e6ebe93d92d2e51fbcf4e3f591d900dd085`.
Each history entry was aged by elapsed wall time exactly as `restore_quota`
requires; entries expire after 86,400 seconds. **Calendar midnight does not clear
this rolling history.** Older snapshots overlap and were not added again. The
140,300-token historical aggregate remains active until approximately
2026-10-27 15:45:59 UTC / 16:45:59 Europe/Berlin under this conservative local
accounting; this is not confirmation of the provider's reset or future balance.
The earlier unknown-usage failure remains charged conservatively. Stale RPD/TPM
headers and an elapsed cooldown do not prove available TPD. No browser, Console
or API quota probe was used; current organization-wide usage was not verified.

Source inspection confirms the existing session guard shares attempt/token
counts across all modes and native tool bindings, reserves before invocation,
keeps failed-call reservations and never refunds the session total on success.
All four local RPM/RPD/TPM/TPD guards remain applicable. The CLI's slowest built-in pacing
is 60.01 seconds, so an execution-only HTTP-start guard would be required to meet
61 seconds; none was installed or exercised because preflight failed. Production
logic, prompts, workflows, benchmark inputs and quota history were not changed.
The private preflight is retained under ignored
`output/private/phase4b-live-20261027/preflight.json`.

**Decision: do not launch.** The full authorized reservation does not fit the
available local budget. No mode ran, no local retrieval was executed, and no
automatic later run was scheduled. Actual answers, completeness, citation
entailment, Agentic evidence recovery, per-mode tokens, calls, searches and
latencies therefore remain **unmeasured**, not zero-quality results. There are
no new same-model auditor decisions or separate source-review judgments; the
controlled offline reference-following result above remains fixture evidence.

This is a real quota blocker for the requested live comparison, with no new
production defect established by the preflight. Existing offline software
acceptance remains intact: [CI for `f9bbd87` passed](https://github.com/cryptoadya/garage-rag/actions/runs/40385567263).
Tests were not rerun for this report-only update. Phase 4B meets its previously
stated offline completion boundary, but the requested live evaluation is
**blocked and incomplete**; no live-quality or Agentic-superiority conclusion is
supported. No merge, tag or Release was performed.
