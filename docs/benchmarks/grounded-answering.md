# Phase 3 answer evaluation

Phase 3 software acceptance and model-quality acceptance are separate. The
current implementation fixes two confirmed local defects; general answer-quality
improvement remains unproven. No independent human or vehicle validation exists.

## Reproduce without API calls

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= GROQ_API_KEY= python -m pytest
ruff check .
python -m build
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
[Groq setup](../groq.md) and [workflow architecture](../grounded-answering.md).
No new live requests were made during the October 26 repair pass.

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

Packets verify request/source/reply hashes and escape HTML. Never publish them.

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

Before closing Phase 3: independent review of original answers/source sets and
applicability; a separately authorized, predeclared live check of the repaired
workflow; genuine erroneous audit-eligible drafts with independent labels and
supported controls; wider balanced EN/DE/RU and both-profile sampling. Report
harmful acceptance, supported-draft rejection and all added costs. A zero count
of genuine eligible errors means audit prevention remains unmeasured. Do not
replace historical live answers with replayed or scripted outcomes.
