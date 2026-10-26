# Phase 3: answer quality and workflow comparison

## What is being measured

This evaluation is separate from retrieval Hit@k/MRR. A relevant page, a genuine
quote and an auditor's approval are three different observations; none alone
establishes claim truth. The Phase 2A/2B datasets, evaluator, baseline artifacts,
embedding revisions and ranking implementations are unchanged.

The new [original Aurora scenarios](../../benchmarks/aurora-grounded-answering.v1.json)
cover supported EN/DE/RU prose, missing repair specifications, unrelated evidence,
negation, model mismatch, missing qualifications, conflicting sources, flattened
numeric tables, safe numerical prose, recoverable missing context, persistent
false refusals, an auditor false refusal, an auditor false acceptance and source
injection. All text is fictional and MIT-licensed. The same author created
questions, references, adversarial scripts and implementation: these are not an
independent sample or human-reviewed gold standard.

There are two distinct execution modes:

- **Scripted, offline:** fixed adversarial drafts and auditor verdicts exercise
  the real graph/validators/context builder. Exhaustive fixture labels can score
  returned claims exactly. This proves behavior under specified errors, not
  model-quality improvement or real-world error prevalence. Scripts are never
  included in live model prompts. Repeated passes measure runtime variability;
  they do not create new independent information needs.
- **Live, explicitly budgeted:** both modes independently call one explicit free
  model using the same generator, initial context and original question. Seeded
  shuffling alternates mode order and case order. There is no shared response,
  cherry-picking, output repair or automatic retry. Generation temperature is
  zero, but upstream routing and model output can still vary. Default limits are
  one completion for direct and at most four for grounded. The run stops at the
  first provider error and writes a partial result with a nonzero exit status.

Synthetic search is controlled context selection, not an embedding benchmark.
For real-corpus evaluation, `--dataset` can instead name an unchanged Phase 2
benchmark with an existing matching profile/index. Corpus SHA/page count, active
chunk count and Phase 2B's exact text/position chunk digest must match before
inference. Actual dense search is used; local cached models only, no reindexing.
Existing `requires_layout=true` cases are unsafe to answer numerically from
text, so answer-evaluation answerability differs intentionally from page-level
retrieval relevance. No original labels are rewritten.

## Scoring and denominators

Each row identifies case, language, mode and trial, the exact answer hash,
supported/refused/error status, reason, returned-claim count, local observations
and optional reviewed claim labels. Public JSON contains no questions, claims,
quotes, filenames or source passages. Optional local captures retain exact sent
context and validated drafts for review, under ignored `output/private/`.

- **Claim support:** whether each returned claim follows from the source with
  necessary conditions and applicability. Live values are `null` until explicit
  external review, never copied from the workflow's own auditor. Exact-reference
  matching is only a separately labelled lexical diagnostic for live answers;
  a correct paraphrase can fail it.
- **Completeness:** whether the supported claims answer all requested parts.
  Claim-level support can be correct while the answer is incomplete.
- **Citation validity:** every returned answer passed the shared local source-ID,
  verbatim-quote and language checks. This is a software invariant, not factual
  precision. For frozen corpus cases, citation page/chunk matches are recorded
  separately; an accepted page still does not establish entailment.
- **Correct refusal:** no answer on a case labelled unsafe/unanswerable in the
  supplied context. **False refusal:** no answer on a labelled answerable case.
  These are not complements of claim precision. A wrong positive answer on an
  answerable case is an unsupported answer, not a false refusal.
- **Provider/local errors:** reported separately, excluded from refusal and
  semantic-quality denominators. Their attempted calls and latency remain in
  resource totals. The `paired` summary includes only case/language/trial pairs
  with both modes completed without errors; unmatched successful rows remain in
  the full report, not in paired quality comparisons.
- **Cost:** all attempted completions, local searches, workflow p50/p95 and provider-reported tokens.
  Timing starts after answer-language resolution and excludes CLI/import/client
  setup and encoder loading, while including graph compilation, local search,
  draft/audit requests and validation. The first language-validator initialization
  is retained, and provider inference has no excluded warmup. Scripted milliseconds exclude network/model
  inference; corpus warmup excludes encoder loading and one local search.
  No paid warmup completion is made. Missing token usage is marked incomplete,
  rather than interpreted as zero tokens. Request latency has no hard wall-clock
  guarantee; SDK network timeouts are not a total request deadline.

Claim-review files bind labels to dataset SHA, case/language/mode/trial and
exact answer SHA, with one boolean per returned claim and a completeness label.
Mismatched hashes, duplicate reviews or wrong claim counts are rejected.
Review provenance states whether it is independent human review. Agent review
is still correlated with implementation and source-label authorship; this phase
does not acquire an independent human gold standard or vehicle validation.

## Reproduction

From a development installation:

```bash
# No key, network model or BMW material; actual graph and validators.
python -m garage_rag.answer_evaluation --trials 3 \
  --output output/answer-scripted.json

# At most 45 completion attempts; explicit live opt-in, model and budget.
python -m garage_rag.answer_evaluation --live \
  --model nvidia/nemotron-3.5-super-vl:free --max-api-calls 45 \
  --cases cold_pressure numeric_prose recovery \
  --output output/answer-live.json \
  --private-captures output/private/answer-live.json

# Real frozen held-out needs; repeat with bge-m3 and its separate index.
python -m garage_rag.answer_evaluation \
  --dataset benchmarks/bmw_e39_2002.phase2b-heldout.v1.json \
  --profile baseline --index-dir data/qdrant --live \
  --model nvidia/nemotron-3.5-super-vl:free --max-api-calls 45 \
  --cases seat_height toolkit_access engine_cylinders \
  --output output/answer-corpus-e5.json \
  --private-captures output/private/answer-corpus-e5.json

# Re-score exact captured outputs without any additional provider calls.
python -m garage_rag.answer_evaluation \
  --report-input output/answer-live.json --review-file output/answer-review.json \
  --output output/answer-reviewed.json
```

`--languages en de ru` is the default; `--trials` defaults to one and is capped at
ten. Live budget preflight requires capacity for five calls per selected pair,
even when most paths will spend fewer. Output paths must be new, preventing an
accidental overwrite of previous results. `.env` is read only in live mode;
never place credentials on the command line. Source snapshots, dependency
versions, model, seed, generation settings and dataset digest are recorded.
Private captures must stay private; they are not part of a package or PR.

Live evaluations now enforce `--requests-per-minute` (1–15, default 15) across
both modes, including every draft and audit. Attempts are paced sequentially,
failed requests consume the enforced `--max-api-calls` budget, and there are no
automatic retries. Set `--requests-per-minute 12` for additional headroom. Pacing
waits are included in workflow latency; a latency comparison must therefore
identify its rate setting. Each CLI invocation has its own budget: when chaining
runs, subtract actual attempts from an explicitly declared session ceiling before
starting the next run. The preflight reservation is not an extra account quota.

Review-file structure (replace hashes with the exact report values):

```json
{
  "report_dataset_sha256": "<dataset_sha256>",
  "reviewer": "Reviewer identity and method",
  "independent_human_review": false,
  "rows": [{
    "id": "cold_pressure", "language": "en", "mode": "grounded", "trial": 1,
    "answer_sha256": "<answer_sha256>",
    "claim_support": [true], "complete_answer": true
  }]
}
```

## Measured results

Results and interpretation are recorded below; the public numeric artifact
contains all attempts, including partial runs and provider failures. No
successful-only subset is presented as a complete live benchmark.

### Controlled software comparison

The [numeric artifact](phase3-results.v1.json) includes three seeded passes over
14 scenarios × three languages × two modes: 252 runs, zero provider calls.
Quality below uses the **first pass only**, 42 runs per mode (21 answerable,
21 unsafe/unanswerable). Later passes are not additional independent cases.

| Controlled outcome | Direct | Grounded |
| --- | ---: | ---: |
| Returned answers | 36 | 12 |
| Returned answers containing an unsupported claim | 27 | 3 |
| Complete supported answers | 9 | 9 |
| Correct refusals | 0 | 18 |
| False refusals | 6 | 12 |
| Logical model invocations | 42 | 78 |

The intended safety behavior is observed: exact but misleading quotations no
longer suffice when the supplied auditor rejects the inference, and damaged
numeric evidence fails locally. But the residual three false answers deliberately
have an auditor that falsely accepts them. Twelve false refusals include cases
where a useful source exists but the generator/auditor rejects it; there is no
unbounded repair loop. This is a safety/coverage tradeoff, not dominance: the
number of complete supported answers is unchanged. Scripted citation validity
is 100% among returned answers in **both** modes, despite their different
unsupported-claim counts.

Across all three passes, direct versus grounded invocation totals are 126 versus
234. The artifact retains individual timing samples; local graph execution is
on the order of a few milliseconds with scripted replies, versus a sub-millisecond
direct path. These timings exclude remote inference and are not live latency
estimates. Four-call recovery and lower-budget termination are separately covered
by regression tests; the fixture suite's observed maximum need not reach the cap.

### Live observations and unsuccessful attempts

All attempts used zero price caps, disabled SDK retries and no paid fallback.
The first four runs used `nvidia/nemotron-3.5-super-vl:free`. After repeated
failures, one **separate, explicitly selected** experiment used
`google/gemma-4-26b-a4b-it:free` and immediately received HTTP 429. No further
live calls were made. This was not automatic fallback in either answer mode.
Provider/network/response errors are not evidence that a model's factual answer
was wrong; the available observations cannot isolate every integration failure.

| Run | Planned pairs | Completed pairs | Attempted completions | Error runs |
| --- | ---: | ---: | ---: | ---: |
| Original English synthetic pilot | 14 | 7 | 18 | 1 |
| EN/DE/RU synthetic subset | 9 | 1 | 4 | 1 |
| Frozen held-out subset, E5 Small | 9 | 2 | 8 | 1 |
| Same subset, BGE-M3 | 9 | 0 | 1 | 1 |
| Separate Gemma synthetic subset | 9 | 0 | 1 | 1 |
| **All attempts** | **50** | **10** | **32** | **5** |

Every planned matrix stopped at its first error; none is a completed live
benchmark. The English pilot predates final numeric-guard/telemetry hardening;
its runtime source hashes are preserved explicitly instead of attributing it
to the final source snapshot. Later reports retain their own source snapshots.
The explicit 429 diagnostic logged only exception class/status, never payloads
or credentials. The first pilot's generation settings were less instrumented;
missing observations are not reconstructed from assumptions.

For the ten completed pairs, both modes returned five supported answers and
five refusals. Agent review found the five answers per mode supported and
complete, with locally valid citations. Four refusals per mode were appropriate;
one per mode was a false refusal on the German toolkit-access need in the real
manual. The expanded E5 context still lacked the required fact. This reuses
Phase 2B's source answerability labels; it does not infer document-wide absence
from a retrieval miss. **No live quality improvement is demonstrated.**

The ten pairs are observations from several overlapping/limited suites, not ten
independent random needs. Claim review was performed by the same Codex agent;
there is no independent human verification. Non-error answers outside complete
pairs are retained in the artifact but do not enter this paired comparison.
BGE's live output quality and a complete live EN/DE/RU matrix remain unmeasured.

| Completed paired subset | Direct calls | Grounded calls | Direct median s | Grounded median s |
| --- | ---: | ---: | ---: | ---: |
| English pilot, 7 pairs | 7 | 10 | 2.44 | 4.41 |
| Russian synthetic question, 1 pair | 1 | 2 | 2.74 | 10.25 |
| Real E5 corpus, 2 pairs | 2 | 4 | 11.10 | 47.37 |

These are descriptive workflow timings on a shared free service, not a latency
SLA or an embedding-speed comparison. The one-row entry is a single observation,
not an estimated latency distribution. The pilot p95 values were 19.32/9.75 s;
the apparent tail advantage for grounded is not credible evidence of lower
latency at this sample size. All error latencies and calls remain in full-run
resource totals, including a failed request lasting about 120 seconds.

Completed pairs used 10 versus 16 calls (+60%). Their provider-reported token
totals were 6,977 input / 3,070 output for direct and 14,165 input / 5,002 output
for grounded. Failed-call usage is incomplete, so these are **not** all-attempt
token totals or an independent billing audit. Price caps request zero cost; no
paid path was enabled. An ordinary accepted answer requires two completions;
worst-case recovery is four and can be substantially slower. There is no new
local encoder or local generative model to load.

### October 26, 2026: quota recovery and bounded recheck

The [dated numeric artifact](phase3-20261026-results.v1.json) preserves the new
probe, unreviewed live rows, hash-bound agent review, partial token usage, source
snapshots and fresh scripted summary separately from the October 25 results.

Before inference, `GET /api/v1/key` returned HTTP 200 and a free-model allowance
of **0 used / 50 limit / 50 remaining**. The current official
[limits documentation](https://openrouter.ai/docs/api_reference/limits) and
[pricing FAQ](https://openrouter.ai/pricing) describe 20 requests/minute and
50/day for accounts below the purchased-credit threshold, with a higher ceiling
for qualifying accounts. Failed requests can consume the allowance. No credits,
account settings or financial settings were changed.

The session ceiling was **45 attempted completions**, with no retries or paid
fallback. One minimal `Reply with OK.` request to the previously selected
`nvidia/nemotron-3.5-super-vl:free` returned **HTTP 200**, cost 0, and 20 input /
32 output tokens. All 32 output tokens were reasoning; `finish_reason=length`
and there was no visible answer. This established endpoint access, not usable
generation or provider stability. It was not repeated with a larger cap.

The subsequent functional smoke selected **cold_pressure, numeric_prose,
recovery**, English, both modes, seed 20261025, one trial, the normal 8192-token
cap, a 15-attempt reservation and **12 requests/minute**. Mode/case shuffling
was unchanged. It stopped at its first provider/network error (exit 2):

| Smoke outcome | Direct | Grounded |
| --- | ---: | ---: |
| Completed cold_pressure answers | 1 | 1 |
| Calls for that completed pair | 1 | 2 |
| Workflow latency for that pair, seconds | 5.313 | 31.453 |
| Input / output tokens for that pair | 387 / 271 | 917 / 404 |
| Additional recovery attempts before error | 0 | 3 |

Both returned claims say to check pressure when tires are cold, with an exact
quote carrying that condition. **Agent review** judged both supported and
complete; both passed local citation checks. There was no independent human
review. The sole non-error pair has no false refusal; that single observation
does not estimate refusal rates or demonstrate a quality advantage. Its timings
include pacing waits and first-use initialization, not just remote inference.

Recovery first returned insufficient evidence, expanded locally while retaining
the first passage, then produced a valid draft from the newly added source.
Its mandatory audit failed on the third workflow call. The graph correctly
withheld the unaudited draft. The failed row is an **error, not a false refusal**.
Its duration was 12.975 seconds. The exact HTTP status and original exception
were not retained by the earlier telemetry, so the cause cannot honestly be
identified as 429, 502 or timeout. The existing safe error category is
`provider_or_network_error`; no diagnostic inference request was made afterward.

The smoke completed **1/3 planned pairs**, with three rows (two answers and one
error) and **6 attempted completions**. Including the minimal probe, the session
used **7/45 attempts**. The post-stop read-only quota snapshot reported 6 used /
44 remaining; the difference from client attempts is retained, not used to
reclaim budget. Smoke usage reports 2,127 input / 967 output tokens across five
successful calls; failed-call usage is missing. These are partial resource
totals, not complete all-attempt usage or a billing audit.

No DE/RU or corpus/profile matrix was started after the error. All three cached
index manifests exist (E5 Small, E5 Base, BGE-M3), but their live quality was not
newly measured. A full held-out matrix of 12 needs × three languages × three
profiles has 108 pairs and requires a worst-case reservation of 540 calls under
the existing per-pair bounds. It must be segmented across explicit budgets and
quota windows; a 50/day allowance cannot reserve it in one day.

Two confirmed measurement defects were fixed: live runs now enforce request
pacing and runtime budgets, and workflow telemetry retains safe HTTP
status/category diagnostics before sanitizing provider exceptions. The latter
fix postdates this smoke and was checked with real SDK mock transport, including
HTTP 502; it cannot recover this run's missing status. No confirmed graph-routing
or claim-validation defect was demonstrated by this smoke. Retrieval, prompts,
frozen labels and baselines were not adjusted to improve metrics.

Fresh offline evaluation completed 252 scripted rows (three trials), with the
same controlled answer/refusal outcomes as before. All five previously recorded
Phase 2 file hashes match. Acceptance passed **216 offline tests**, Ruff, and
wheel/sdist builds. Remote CI is recorded on PR #3 for the pushed commit.
**Phase 3 acceptance remains open**: stable complete live EN/DE/RU/profile results
and independent human quality review are still missing.

To reproduce the attempted smoke, use a new output filename:

```bash
python -m garage_rag.answer_evaluation --live \
  --model nvidia/nemotron-3.5-super-vl:free --max-api-calls 15 \
  --requests-per-minute 12 --languages en --seed 20261025 --trials 1 \
  --cases cold_pressure numeric_prose recovery \
  --output output/answer-recheck-smoke.json \
  --private-captures output/private/answer-recheck-smoke.json
```

Provider behavior is nondeterministic: reproduction fixes inputs, scheduling,
budgets and code provenance; it does not promise the same answer or failure.

### Compatibility and checks (October 25)

Beyond mocked CLI tests, both **actual cached** E5 Small and BGE-M3 snapshots ran
the real graph on all nine selected case/language combinations per profile:
18/18 executions, matching index fingerprints/chunk digests, unchanged manifests
and zero API calls. A controlled refusal fixture supplied responses in this
check: it establishes runtime/profile/language compatibility, not model quality.
Its adapter-constructor timing is not model-load timing; the first retrieval
includes lazy encoder loading, so these check timings are not warm latency data.

Acceptance ran 211 offline tests, Ruff, wheel and sdist builds. SDK mock transport
checks enforce actual request count, zero price caps, no hidden retry/fallback,
60-second network-operation timeout configuration and reported-token accounting.
Package inspection found no private PDFs, extracted text, captures, credentials,
indexes, caches or model weights. The final PR's GitHub Actions status is the
remote acceptance gate; local checks do not stand in for CI.

Before merge, review the default two-call cost and false-refusal tradeoff,
rerun a larger **complete** live comparison after quota availability is restored,
and obtain independent human review of claims, source sufficiency, language
quality and table/applicability labels. Grounded mode is a tested bounded
workflow with stronger rejection contracts; it is not established as a generally
more accurate model-answering system by these partial live measurements.

## Groq: one English cold-pressure pair (October 26, 2026)

This is a separate Groq provider/model result, not an extension of the OpenRouter
sample or the earlier Groq grounded-only smoke. The existing evaluator ran the
unchanged Aurora `cold_pressure` case once in English on `openai/gpt-oss-120b`,
with seed 20261025, direct first then grounded. Free / Current Plan was reverified
in the organization's Console before inference. Runtime commit was `6c95612`;
retrieval, scenario, prompts, labels and evaluation methodology were unchanged.

The exact run used the existing checkpoint/resume infrastructure:

```bash
GROQ_FREE_PLAN_CONFIRMED=true GROQ_MAX_TOKENS=2048 \
python -m garage_rag.answer_evaluation \
  --live --provider groq --model openai/gpt-oss-120b \
  --cases cold_pressure --languages en --trials 1 --seed 20261025 \
  --max-api-calls 5 --requests-per-minute 6 \
  --output output/groq/gpt-oss-120b/cold-pressure-en.json \
  --private-captures output/private/groq/gpt-oss-120b/cold-pressure-en.json
```

It exited zero with **one complete pair / two rows and three attempted API calls**
out of the five-call ceiling. There were no retries, fallback, errors, repairs or
context expansions. A subsequent identical command with `--resume` reported
`Checkpoint already complete; no API calls`; no additional inference occurred.
Each completed mode was atomically saved. Raw checkpoints and answer/source
captures remain ignored locally. The [reviewed public result](groq-20261026-cold-pressure-en.v1.json)
retains runtime/source hashes and the original checkpoint hash, with the absolute
private-capture path removed; it is not the checkpoint used for resumption.

| Observation (one run per mode) | Direct | Grounded |
| --- | ---: | ---: |
| Supported, complete answers after explicit agent source review | 1 | 1 |
| False refusals / provider errors | 0 / 0 | 0 / 0 |
| Attempted API calls | 1 | 2 |
| Local searches / expansions | 1 / 0 | 1 / 0 |
| Input / output tokens, including reasoning | 458 / 188 | 1,054 / 401 |
| Total reported tokens | 646 | 1,455 |
| Workflow elapsed time, seconds | 0.873 | 19.983 |

Total usage for this pair was **1,512 input + 589 output = 2,101 tokens**, reported
for all three attempts. Grounded added one call and 809 tokens for this observation,
with no answer-quality difference. Its elapsed time includes the shared 6-RPM
limiter's wait before both its draft and its audit. Recorded request starts were
about 10.015 and 10.013 seconds apart. These are paced end-to-end workflow timings,
not comparable inference-only latency estimates or representative performance
statistics. This single supported timing question does not test difficult
abstention, recovery, contradiction or numeric/table behavior.

### Explicit source review, separate from the Groq audit

Both captured answers have identical claim/quote text and say to check tire
pressure **when the tires are cold**. Source `S1` states precisely that timing
condition. Both claims preserve the condition, add no pressure value, temperature
threshold, repair procedure or vehicle-specific applicability, and fully answer
this scenario's question. Their evidence is an exact, unmodified quotation from
the supplied synthetic owner's-manual passage; `S1` maps locally to PDF page 1.
No model/year/body-style restrictions are present in this fixture, so this result
cannot establish applicability to an actual BMW or the real manual. Neither mode
refused; no false refusal or unsupported claim was observed in this pair.

The implementation agent inspected both private captures against the supplied
passage and entered explicit per-claim support/completeness labels in a
[hash-bound review](groq-20261026-cold-pressure-en-review.v1.json). The existing
`--report-input` / `--review-file` path applied those labels without API calls.
The full answer hashes differ because their `reason` fields differ; the answer
texts themselves are identical. Grounded's own `audited_evidence` verdict was
not used as evaluation ground truth. **This is agent source review, not
independent human review.** No general quality advantage is demonstrated.

### Quota and billing observations

The confirmed Free Plan's recorded model limits are 30 RPM, 1,000 RPD,
8,000 TPM and 200,000 TPD. The run was paced at 6 RPM and used conservative
pre-request token/day guards. Live headers independently reported RPD limit
1,000 and TPM limit 8,000; final remaining fields were **997 RPD / 7,147 TPM**.
No 429, Retry-After, paid tier switch or billing action occurred. Groq has no
per-request zero-price cap; the free billing boundary is the verified account
plan and fixed non-upgrading request path described in [Groq setup](../groq.md).

Headers do not expose RPM/TPD balances. Together with the preceding separately
recorded smoke, this client observed five successful calls and 3,527 reported
tokens on this date, well below the recorded daily caps. That local total is
not an organization-wide billing audit. Do not infer cumulative daily request
usage by subtracting the latest remaining header from 1,000: header balances
and reset observations changed between runs. Console usage can lag by up to
15 minutes. Raw header observations are retained rather than reconciled by guess.

### Next proposed bounded step — not executed

Run only `cold_pressure` in **DE and RU**, once per language with both modes,
at 6 RPM and at most **10 new attempts** (five reserved per pair), checking quota
headroom first. Combined with this EN observation, it would give initial
three-language compatibility evidence for one simple source/condition. Review
translated claims, retention of the cold-tire condition, unchanged English
quotes, refusals, usage and paced latency; stop on the first error and preserve
checkpoints. Do not rerun this completed EN pair or automatically extend to other
cases/profiles. Harder EN cases and real-corpus applicability remain subsequent,
separately authorized work. Phase 3 quality acceptance and independent human
review remain open. No merge, tag or Release was performed.
