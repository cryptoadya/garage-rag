# Phase 3 answer evaluation

Phase 3 software acceptance and model-quality acceptance are separate. The
implementation includes numeric-gate and context-expansion repairs; general answer-quality
improvement remains unproven. No independent human or vehicle validation exists.

## Reproduce without API calls

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= GROQ_API_KEY= python -m pytest
ruff check .
python -m build
# Optional scripted comparison; not needed for documentation-only changes.
python -m garage_rag.answer_evaluation --trials 3
```

The evaluator defaults to `output/benchmarks/answers.json`; existing results
are never overwritten. Choose another `--output output/benchmarks/<run>.json`
for subsequent runs. All output stays ignored locally. New raw JSON in this
directory is ignored too; publish only a reviewed final Markdown summary and
necessary synthetic fixtures. Frozen Phase 2 datasets and baselines are unchanged.

The frozen `benchmarks/aurora-grounded-answering.v1.json` has 14 fictional needs
in EN/DE/RU. Three trials exercise 252 real workflow executions with scripted
model replies and zero API calls. Scripts test software contracts; repetitions
are not independent needs and their latency excludes model inference.

Live runs require `--live` and an explicit attempt budget covering five calls
per pair. Both modes independently use the same generator and first context;
seeded order shuffling limits order bias. No scripted reply is sent live, no
retry, output repair, paid fallback or silent model switch occurs. Stop on the
first provider error, save a partial checkpoint, and report incomplete matrices.
Real-corpus runs additionally require the unchanged Phase 2 dataset, matching
index/corpus/chunk digests and locally cached embedding weights. See
[Groq setup](../providers.md) and [workflow architecture](../architecture.md).
Offline replay and the later live regression are reported separately below.

## Metrics and review

Legacy `summary` and `paired` definitions are unchanged:

- Claim support requires entailment, applicability, essential conditions and a
  safe value-to-quantity mapping; completeness covers every requested part.
- Citation validity checks IDs, exact normalized quotes and local page metadata.
  It does not establish claim support. Exact-reference matches are only lexical
  diagnostics for live answers.
- Correct refusals and false refusals use dataset answerability. An unsupported
  positive answer is counted separately. Errors are excluded from these quality
  denominators, but their attempts and latency remain in resource totals.
- Paired summaries include only pairs where both modes completed without error.
  Unmatched successes remain visible in the complete report.
- Report all attempted calls, local searches, reported tokens and workflow
  latency. Separate synchronous HTTP time from RPM and proactive quota waits;
  paced/order-dependent workflow totals are not inference-speed estimates.

Live support/completeness stays unscored until explicit hash-bound review via
`--report-input`, `--review-file` and a new local `--output`. Reviews bind dataset,
case/language/mode/trial, answer SHA and claim count; invalid submissions reject
atomically. Same-model audit verdicts are never evaluation ground truth.

Additive `--source-review-file` labels every distinct source set actually sent
for draft generation as sufficient or insufficient for the entire question.
Bind every ordered source-set hash and answer hash; omit unknown judgments.
The final sent context defines `sufficient_context_runs`,
`insufficient_context_runs` and `unreviewed_runs`. Report:
`false_refusals_with_sufficient_context` (denominator: sufficient contexts),
`generator_false_refusals` (last raw draft explicitly abstains),
`other_sufficient_context_refusals`, `justified_context_refusals` and
`answers_with_insufficient_context`. The last metric alone does not imply every
partial claim is false. Paired source metrics require both reviewed modes.
Retrieved-but-unsent pages never establish source sufficiency. Legacy global
false refusals can include retrieval/context misses; neither metric replaces the other.
Reviewer identity and independence remain per row, including mixed reviews.

Private captures retain exact requests, replies, contexts and validated drafts.
Public summaries contain no private text. Generate an unfilled independent
review packet with:

```bash
python -m garage_rag.review_packet --report output/<checkpoint>.json \
  --captures output/private/<captures>.json --output-dir output/private/<review>
```

Packets verify request/source/reply hashes, escape HTML and omit existing quality
judgments. Claim and source reviews retain per-row provenance; legacy labels
without row provenance never become independent through a partial new review.
Never publish packets. The [archive inventory](phase3-archive.md) records the
existing private handoff; no independent human labels have been supplied.

## Preserved historical results

All historical metrics below predate the repair and are not retroactively
rescored. Agent labels are correlated with implementation, not independent
human judgments. Raw rows, hashes, parameters, timing and reports are in the
[verified private archive](phase3-archive.md).

| Experiment | Evidence and outcome |
| --- | --- |
| Scripted first pass, 42 runs/mode | Unsupported answers direct/grounded 27/3; complete supported 9/9; false refusals 6/12; calls 42/78. Three passes total 252 rows, zero APIs. Deliberately false auditor acceptance remains possible. |
| Initial OpenRouter attempts | 10 completed pairs, 32 attempts, 5 error runs across incomplete matrices. No quality improvement established; calls on completed pairs 10/16. Provider errors are not wrong factual answers. |
| October 26 OpenRouter recheck | Diagnostic and partial smoke retained separately in archive; incomplete matrices and token-limit/provider failures are not a completed quality comparison. |
| Groq cold-pressure EN/DE/RU | 3 complete supported pairs; calls 3/6; tokens 2,051/4,543. Extra audits accepted already correct drafts. One simple need cannot establish robustness. |
| Groq four harder EN needs | 4 pairs, 11/20 attempts, no errors. One controlled missing-evidence recovery; three source-relative ties. Calls 4/7; tokens 2,328/4,726. No incorrect audit-eligible draft occurred. |
| Real BMW E39 E5 Small/BGE-M3 | 14 pairs, 33/60 attempts, no errors. Complete safely supported answers 9/4; sufficient-context false refusals 0/5. Calls 14/19; tokens 36,081/48,288. |

The real BMW matrix covered seat height (EN/RU), chain speed (EN/DE), engine
cylinders (DE/RU) and supplemental hood release (DE), with both profiles.
Four engine-table answers matched the original PDF numerically but lacked safe
text grounding; grounded blocked them locally. They are not four factually
incorrect numbers or evidence of semantic-audit benefit. Only four supported
seat drafts reached the auditor. Chain drafts and BGE hood location supplied
the five local-gate false positives. E5 retrieved the hood page at rank eight
but excluded it at the context cap. Agent source review separates that justified
missing-context refusal from generator abstention. No real-corpus recovery or
overall quality gain occurred in that historical run.

## October 26 offline repair acceptance

Synthetic regressions cover wrapped dual-unit prose, unrelated numbers in a
nonnumeric claim's source, unsupported numbers, damaged tables and essential
conditions. Compact expansion keeps every prior passage and all applicability
metadata; it removes display URL/redundant page labels and JSON whitespace only.
The original first context/direct baseline, ranking, 8-chunk/12,000-character cap,
prompts, datasets and metric methodology are unchanged.

Private saved-draft replay checked all 13 grounded candidates: nine safe prose/
nonnumeric drafts pass, including all five former false positives; four numeric
table drafts remain blocked. Cached E5 retrieval reproduces the original first
context hash; expansion now sends the relevant page at 11,676 characters,
retaining every prior passage. These checks do not simulate new audits or prove
new live answer quality. The replay script and results are archived privately.

## October 26 bounded live repair regression

Fresh generations at `644538405478fa0f8faf91c6a512e590d9c046ab`, unchanged real
BMW E39 held-out corpus/index/chunk digests, first requests and seed `20261026`.
Five predeclared pairs used Groq Free Plan / `openai/gpt-oss-120b`, temperature
zero, 2,048 completion tokens, pinned offline tokenizer accounting and 6 RPM pacing.
**15/25 attempts**, zero errors/retries/fallbacks/model changes. Quota state,
reservations and exact timing remain in the private archive.

Before values below are the unchanged historical selection, not a rerun or
rescore. Answer/source judgments are hash-bound **implementation-agent review,
not independent human review**; the same-model audit is not evaluation truth.
`D/G` denotes direct/grounded; tokens are reported input plus output.

| Pair | Direct before → after | Grounded before → after | Calls D/G before → after | Tokens D/G before → after |
| --- | --- | --- | --- | --- |
| chains_speed, E5 Small/DE | Supported → supported | Numeric gate refusal → supported | 1/1 → 1/2 | 2,088/2,088 → 2,088/4,448 |
| chains_speed, BGE-M3/EN | Supported → supported | Numeric gate refusal → supported | 1/1 → 1/2 | 2,903/2,607 → 2,858/5,746 |
| hood_release, BGE-M3/DE | Supported → supported | Numeric gate refusal → supported | 1/1 → 1/2 | 2,789/2,840 → 2,748/5,730 |
| hood_release, E5 Small/DE | Missing-context refusal → same | Missing-context refusal → supported after expansion | 1/2 → 1/3 | 3,720/6,085 → 2,440/10,021 |
| engine_cylinders, BGE-M3/RU | Unsafe table grounding → same | Numeric gate refusal → same | 1/1 → 1/1 | 2,468/2,310 → 2,340/2,310 |

Complete safely supported answers D/G: **3/0 → 3/4**. Final sufficient sent
contexts: 3/3 → 3/4; false refusals within those contexts: **0/3 → 0/0**
(denominators 3/3 → 3/4), with zero raw-generator false refusals given sufficient
context. The direct E5 hood refusal remains justified. All eight returned
positive answers have valid normalized quotations, source IDs and page mappings;
the two direct cylinder answers remain unsafe text grounding despite the correct
value in the rendered PDF. The new grounded table candidate is blocked before
audit, including its spelled-out Russian number. No table protection was relaxed.

E5 recovery is observed end to end: first raw draft abstains with pages
55/4/53/147/155; expansion actually sends page 137 as S8, and the new draft
answers correctly before audit acceptance. Assembly uses 8 sources / 11,676
characters, retains every previous passage and applicability field, and stays
within the unchanged caps. Previously page 137 was retrieved but unsent. All
ten first request hashes match their historical counterparts and paired modes.

Calls D/G: 5/6 → 5/10; local searches: 5/6 → 5/6; total reported tokens:
13,968/15,930 → 12,474/28,255 (new input 10,745/24,770; output 1,729/3,485).
New HTTP time D/G is 5.14/12.32 s, RPM wait 8.75/14.48 s and proactive quota
wait 204.81/238.57 s. These paced, ordered single trials do not measure inference
speed or establish general model-quality improvement.

Both repairs worked on this selected regression. Reports, captures, agent reviews
and unfilled independent-review packets remain in the
[private live-regression archive](phase3-archive.md#live-repair-regression).
Historical results were not rescored or overwritten.

Before claiming broader quality improvement: independent answer/source/applicability review; genuine
erroneous audit-eligible drafts with independent labels and supported controls;
wider balanced EN/DE/RU and both-profile sampling. Report harmful acceptance,
supported-draft rejection and all added costs. This selected repair regression
contains no genuine erroneous audit-eligible draft, so semantic-audit prevention
remains unmeasured. No independent human or vehicle validation is established.
