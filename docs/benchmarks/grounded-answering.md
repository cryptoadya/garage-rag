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

### Compatibility and checks

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
