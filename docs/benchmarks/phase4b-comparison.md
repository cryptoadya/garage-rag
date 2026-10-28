# Phase 4B: Direct / Grounded / experimental Agentic

**The bounded comparison is complete, with no demonstrated Agentic advantage.**
The final six-workflow run produced Direct **2/2**, Grounded **2/2** and Agentic
**1/2** complete supported answers under separate, unblinded AI source review.
Agentic's other row is a safe `context_budget_exhausted` stop. Subsequent Agentic
changes and two unsuccessful single-case trials are reported separately below;
they do not replace or augment those historical scores.

Direct and Grounded remain the primary modes. Agentic is opt-in and experimental.
These results support closure of the comparison stage, not production readiness,
vehicle validation, universal answer success or a general quality ranking.

## Scope and evidence

Phase 4B extends the existing answer evaluator, captures and review packets.
It adds a fixed fictional three-mode suite, matched/pairwise summaries, citation
and warning review, and shared attempt/token/pacing guards. Later changes bound
Agentic evidence selection, model context and terminal writer/auditor capacity.
Direct/Grounded behavior and historical Phase 2/3 datasets remain unchanged.

This is the consolidated report as of 2026-10-28. The complete **1,868-line
pre-cleanup report** is preserved at immutable commit
[`de6a965`](https://github.com/cryptoadya/garage-rag/blob/de6a96577322feffcb2ea59c649070402f3ac1f1/docs/benchmarks/phase4b-comparison.md).
It retains all detailed experiments, per-call tables, diagnostics, proposals,
validation counts and private artifact locations. Use it for historical evidence;
intermediate implementation descriptions there are not the current specification.
The source-commit map at the end makes each stage independently inspectable.
No historical commit or private file is removed by consolidation.

## Method and review rules

- Each mode executes its real workflow independently. Direct/Grounded receive the
  same initial context; Grounded can expand its fixed ranking once. Agentic starts
  empty and chooses bounded local dense/BM25 searches, then a writer and audit.
- The fictional suite uses controlled rankings and model replies. It exercises
  contracts, including targeted evidence outside fixed-depth expansion and an
  intentionally unreliable auditor. It does not measure natural retrieval or
  model decision quality. Repetitions are not independent samples.
- Live runs used Groq `openai/gpt-oss-120b`, temperature zero and 1,024 completion
  tokens **including reasoning**, with pinned local tokenizer accounting. Actual
  HTTP starts were spaced at least 61 seconds apart by private execution controls.
  These controls do not imply automatic CLI pacing or retry.
- The [frozen real-corpus protocol](phase4b-real-live-protocol.md) binds the original
  186-page E39 PDF, 241-chunk baseline E5/Qdrant/BM25 index and two existing EN
  questions: `hood_release` and `toolkit_access`. The final order was amended
  before generation to hood then toolkit, Direct → Grounded → Agentic per question.
  Seed `20261027` is retained metadata, not a randomized-order claim.
- Correctness, whole-question completeness, citation entailment, applicability and
  relevant warning preservation are distinct judgments. A real quotation alone
  proves neither completeness nor that the cited text supports the entire claim.
- Same-model audit verdicts are workflow observations, never evaluation labels.
  Live quality stays unscored until explicit answer/context-hash-bound review.
  Review packets bind requests, replies, schemas/tools and actual sent source sets;
  retrieved-but-unsent evidence cannot establish model-context sufficiency.
- Separate AI reviewers inspected original PDF text and rendered relevant pages.
  The final comparison was unblinded AI review, **not independent human review**.
  Uncertain/missing/error rows are not promoted to successful answers. Refusal
  reasons distinguish provider failure, local validation and model abstention.
- Held session reservations never refund failed calls or missing usage. Reported
  provider usage is separate; unknown failed-call usage is not zero. Provider quota
  history can reconcile successful usage while retaining unknown-usage holds.

## Fictional offline comparison

`benchmarks/aurora-comparative.v1.json` has seven fictional English-source needs,
each asked in EN/DE/RU: simple pressure guidance, compound jack preparation,
omitted warning, missing vehicle qualifier, absent repair specification,
conflicting applicable documents and reference following. Translations/annotations
are implementation-authored. Dataset SHA256:
`1e7814cf8186ba26397d0febb15b0dd9e9ea4a345ae95c3e74462d1d024ce4f5`.

The recorded run has **189 workflows**, 63 per mode / matched triples, but only
seven distinct needs: four answerable and three requiring refusal.

| Scripted observation | Direct | Grounded | Agentic |
| --- | ---: | ---: | ---: |
| Complete supported / answerable runs | 9/36 | 18/36 | 27/36 |
| Unsupported positive / positive answers | 27/45 | 9/27 | 9/36 |
| Corpus-relative false refusals / answerable runs | 9/36 | 9/36 | 0/36 |
| Correct refusals / unanswerable runs | 9/27 | 27/27 | 27/27 |
| Sufficient cited evidence / reviewed positive claims | 18/45 | 18/27 | 27/36 |
| Missed warnings / relevant positive answers | 18/18 | 9/18 | 9/18 |
| Locally valid citation answers | 45 | 27 | 36 |
| Scripted model invocations; actual API calls = 0 | 63 | 135 | 261 |
| Local fixture searches | 63 | 90 | 81 |
| Local workflow p50 / p95, ms | 0.29 / 0.40 | 3.06 / 7.62 | 4.35 / 9.97 |

Agentic's nine extra complete answers over Grounded repeat one controlled
reference-following need. Both audited modes miss the deliberately accepted
warning omission. Token usage is unavailable; these latencies measure local
orchestration, not inference. No multilingual parity, statistical significance,
production cost or real-model quality advantage follows from this fixture.

Reproduction below is offline and is not a required new benchmark for cleanup.
Use fresh ignored output paths; existing reports must not be overwritten.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= GROQ_API_KEY= \
  python -m garage_rag.answer_evaluation \
  --dataset benchmarks/aurora-comparative.v1.json \
  --modes direct grounded agentic --trials 3 --seed 20261026 \
  --output output/benchmarks/phase4b.json \
  --private-captures output/private/phase4b.json
```

The comparative suite defaults to three modes; legacy datasets retain two.
For packet creation and offline rescoring, use the existing
[answer-review workflow](grounded-answering.md#metrics-and-review).

## Final real-corpus comparison — 2026-10-27

Production commit [`b41dac9`](https://github.com/cryptoadya/garage-rag/commit/b41dac95d9e527a12bde621d6f7e0dd1fc30704d),
reported in [`8f18901`](https://github.com/cryptoadya/garage-rag/commit/8f18901950d55d30b65447a0f5681a9247aae055).
One authorized run, 19:52:57–20:04:10 Europe/Berlin, six planned rows, caps
**22 API attempts / 128,000 held tokens**, no retry/fallback/resume or tuning.
All four strict structured planner decisions were valid; all 12 HTTP calls
succeeded. Bindings verified code/schema/tokenizer, PDF/index/dataset and latest
rolling quota history, including all earlier failed reservations.

| Question / reviewed outcome | Direct | Grounded | Agentic |
| --- | --- | --- | --- |
| Hood lever location | Complete, p137 | Same complete answer, p137 | Same complete answer, p137 |
| Toolkit location and access | Complete, p23 | Complete, p23 | Incomplete; context guard stop |
| Complete supported answers | **2/2** | **2/2** | **1/2** |
| Supported claims / positive claims | 2/2 | 2/2 | 1/1 |
| Supported citations / positive citations | 2/2 | 2/2 | 1/1 |

Review checked 20 unique actually sent source objects and original PDF pages
137/23. No relevant warning/applicability omission was found in the five positive
answers. Three same-model audits agreed after labels were fixed. Quote matching
allows whitespace normalization; Grounded's toolkit quote is not byte-identical
multiline text. Page 23 is a valid alternative to frozen gold page 150: p150-only
match flags remain false but do not invalidate the source-supported answers.

Direct/Grounded initial HTTP bodies were identical per question; neither Grounded
run expanded. Hood initial pages were 137,73,66,37,68. Agentic chose dense search,
received 137,73,66,116,37, then finalized, wrote and audited the same answer.
For toolkit, Direct/Grounded already had sufficient p23 in 23,180,85,31,100.
Agentic's first dense search returned 180,182,181,8,31: pointers/general notes.
Its second dense search executed, but combined evidence exceeded the then-lossless
context guard before delivery or writer/audit. **Rejected raw second-search hits
were not captured**; their contents/ranks remain unknown. No BM25 was chosen and
no recovery beyond Grounded was observed. The corpus-relative false-refusal count
is one, caused by the local guard, not demonstrated generator over-abstention.

| Case / mode | API / searches | Held tokens | Actual input + completion = total | Paced wall / wire HTTP, s |
| --- | ---: | ---: | --- | ---: |
| Hood / Direct | 1 / 1 | 3,879 | 2,627 + 222 = 2,849 | 1.108 / 1.014 |
| Hood / Grounded | 2 / 1 | 7,914 | 5,410 + 491 = 5,901 | 122.622 / 2.496 |
| Hood / Agentic | 4 / 1 | 14,905 | 7,919 + 664 = 8,583 | 243.673 / 2.984 |
| Toolkit / Direct | 1 / 1 | 3,225 | 1,973 + 422 = 2,395 | 62.778 / 2.534 |
| Toolkit / Grounded | 2 / 1 | 6,636 | 4,132 + 608 = 4,740 | 120.677 / 2.013 |
| Toolkit / Agentic | 2 / 2 | 7,052 | 2,786 + 189 = 2,975 | 121.606 / 1.057 |

Totals: **12 calls / 7 workflow searches / 43,611 held / 27,443 actual tokens**
(24,847 input + 2,596 completion, including 1,854 reasoning); usage is complete.
One local warmup is excluded. HTTP-start intervals: 61.075–61.095 s.
Whole execution: 676.203 s, including 3.580 initialization, 659.448 pacing and
12.098 wire HTTP; provider completion generation: 5.445 s. Serial pacing dominates
wall times; wire time includes transport/queue work. USD billing was not measured.

There are two matched workflow triples and two comparisons per pair, including
the bounded refusal. Direct/Grounded tie on both needs; Agentic uses more aggregate
calls/tokens without improving reviewed quality. Its cheaper toolkit stop is not
an efficiency win. Two preselected needs / one trial do not establish broad ranking.

## Later Agentic changes and observations — separate from comparison scores

The current implementation is specified in
[architecture](../architecture.md#agentic-context-management) and
[provider transport](../providers.md#agentic-planner-transport).
The following measurements describe development stages, not additional benchmark
rows or an authorization for future live tests.

| Change | Confirmed offline effect | Boundary |
| --- | --- | --- |
| Evidence Store / bounded selection (`1790e58`) | Full raw hits and stable IDs survive exclusions; compact history | Initial two-search replay still overflowed |
| Scoped procedure/navigation selection (`641b2b9`) | Same two batches: 13,999 protected chars → 5,779; selected context 7,798 | No procedure in those hits; three-search success used fixed replies and lowered search ceiling |
| Compact wire (`e7a533e`) | Three-batch context 11,462 → 9,660 chars with identical text/membership | Default route needed 25,196/24,000 tokens; audit blocked |
| Token-aware terminal handoff (`587f6c4`) | Default fixed-reply route: skip extra planner; writer + mandatory audit within 19,898 held tokens | No new model decision or semantic-quality result |
| Planner-only navigation hints (`3a79fa5`) | Same fixed routes/sources; writer/auditor inputs unchanged; +450 planner reservation tokens | Delivery does not establish useful model choices |

In the handoff replay, three planners spend 10,844; terminal pair quote is 11,166
(writer 4,681 + audit base 4,301 + growth/frame 2,184). Another planner costs 5,298
and would exceed 24,000. Handoff yields 19,898 actually held tokens, including the
4,373 actual audit quote. With hints, the same fixture holds 20,348. These are
conservative reservations with fixture replies, not provider usage or live success.

Two later, separately authorized toolkit trials used ordinary limits (four searches,
six calls, 24,000 held tokens, twelve steps, top-k ≤5), unchanged corpus/question,
Groq strict planner and 1,024 completion tokens. Fresh authenticated Console checks
confirmed Free Plan and sufficient historical headroom before each execution.

| Production / report commit | Observed route and delivered outcome | Calls / searches | Held / actual tokens | Paced duration |
| --- | --- | ---: | ---: | ---: |
| `587f6c4` / `fce4ed1` | Two dense searches → explicit `insufficient_evidence`; no writer/auditor | 3 / 2 | 11,171 / 5,049 | 126.853 s |
| `3a79fa5` / `de6a965` | Dense → finalize → writer → `unsafe_numeric_evidence`; auditor not invoked | 3 / 1 | 9,790 / 5,214 | 128.508 s |

First trial: ten hits / nine unique, five selected, 8,269 context characters;
no retrieved passage answered location/access. Planner received the p150 pointer
and complete query/ID history. The third turn fit the projected 21,448/24,000
budget and autonomously refused. Separate PDF AI review found a safe but incomplete,
corpus-answerable refusal. No useful procedure was lost by selection.

The subsequent offline diagnosis reproduced that route exactly with HTTP blocked.
Ten analyst-selected local dense/BM25 probes found p150 at BM25 rank one for either
actual Planner query; the original question's dense results already contained p23.
These counterfactuals establish reachability, not autonomous recovery. Blind printed
label → physical page lookup is unsafe: E39 lacks explicit PDF PageLabels, and a
synthetic offset example resolves printed target 2 to physical page 4, not page 2.

Second trial: five hits, S1:p23 / S2:p180 / S4:p39 selected, 4,111 characters.
Planner received the real S2 heading hint with opaque label 150 but chose no BM25
or reference-following search. Its first query also changed, preventing causal
attribution to hints. Audit had 14,210 tokens of protected capacity; budget was not
the stop. The unchanged numeric gate rejected claim 2's spelled-out **two**.
Separate PDF AI review found that claim materially wrong: it conflated sport-wagon
emergency tailgate release with normal toolkit access. Claims 1/3 were supported;
all five quotations existed after whitespace normalization, which did not entail
claim 2. The wrong draft was withheld; the hypothetical audit verdict is unknown.
Neither trial succeeded, exhausted context/token/quota budgets or validated the
budget-forced handoff branch live. They are not pooled with the six-row comparison.

## Limits and retained debt

- Agentic remains experimental; no live toolkit recovery or advantage is established.
  Direct/Grounded remain primary. No independent human or vehicle acceptance exists.
- Same-model audits can share generator errors. The earlier Aurora live answer was
  incomplete despite `accept / complete=true`; valid quotes did not supply location.
- Scope/navigation heuristics can miss paraphrases, unmarked cross-page warnings
  and continuations; conservative unknown-scope protection may also over-refuse.
  A selector cannot create a procedure absent from all searches. Verified procedure/
  warning links or reference resolution would be separate feature work.
- Context characters, call slots and token capacity are independent bounds. Exact
  audit preflight remains necessary even after conservative terminal allocation.
  Numeric heuristics can reject valid prose; the last stop also hid a semantic error.
- Earlier native planner continuations failed with HTTP 400. Structured transport
  and offline replay do not establish general provider inference reliability.
- Failed-call usage remains unknown where reported; no billed USD cost or intrinsic
  latency ranking was measured. Quota snapshots are not organization-wide guarantees.
- Private captures, PDF/renders, reasoning, account evidence and quota ledgers remain
  ignored under `output/private/`; they are not distributed audit artifacts. The
  public historical reports retain diagnostics and hashes, not all underlying data.

## Verifiable history

Every link below points to a full immutable Git commit. Its report contains the
stage's detailed measurements and validation record; the complete pre-cleanup
report above also preserves all sections together. Historical failures are retained,
not relabeled by later implementations. Keep these commits available in Git.

| Stage | Implementation / report source |
| --- | --- |
| Initial offline suite | [f9bbd87](https://github.com/cryptoadya/garage-rag/commit/f9bbd87423f3c089f0635b75e7d6fa74d808b49d) |
| Quota-blocked preflight / partial Aurora live | [30f6307](https://github.com/cryptoadya/garage-rag/commit/30f630768598325ff7eb33e6b5217f4d7b371d09) / [e4fe37b](https://github.com/cryptoadya/garage-rag/commit/e4fe37bdc6195affeb953bf266168b3f3eaa6e09) |
| Error classification / partial real-corpus live | [a6fbe15](https://github.com/cryptoadya/garage-rag/commit/a6fbe151a9cf1fb7dbaecf0d44b2493fade8213d) / [d06102a](https://github.com/cryptoadya/garage-rag/commit/d06102afbfd1e7beccc5810e6a7ba998d34bb0bc) |
| Native terminal actions / successful two-call smoke | [e7626c8](https://github.com/cryptoadya/garage-rag/commit/e7626c845efcd5e694763745aa76c8de793c07e0) / [65da46f](https://github.com/cryptoadya/garage-rag/commit/65da46f2dc1742e4814c72d5c2dd0a5ddd5fb51d) |
| Further native full-workflow failure | [60ab5f0](https://github.com/cryptoadya/garage-rag/commit/60ab5f03cba8dd6d043017d2bec3b83d9be670d1) |
| Structured planner / final six-row comparison | [b41dac9](https://github.com/cryptoadya/garage-rag/commit/b41dac95d9e527a12bde621d6f7e0dd1fc30704d) / [8f18901](https://github.com/cryptoadya/garage-rag/commit/8f18901950d55d30b65447a0f5681a9247aae055) |
| Evidence/context separation | [1790e58](https://github.com/cryptoadya/garage-rag/commit/1790e5800c017f610c0a6a67aecea8406a21d905) |
| Scoped context / compact wire | [641b2b9](https://github.com/cryptoadya/garage-rag/commit/641b2b9614f6db86372b8bdf1749ffff8a16ffad) / [e7a533e](https://github.com/cryptoadya/garage-rag/commit/e7a533ed52fd10eab5eb68bbf54b97a42183d1f8) |
| Token-aware handoff / explicit live refusal | [587f6c4](https://github.com/cryptoadya/garage-rag/commit/587f6c430f51b8fe9c89e31dd94343392d5da36b) / [fce4ed1](https://github.com/cryptoadya/garage-rag/commit/fce4ed1f02b5feddb0caefb7ec6945fb350ed687) |
| Offline navigation diagnosis | [e7209e1](https://github.com/cryptoadya/garage-rag/commit/e7209e1ebaeceb6f6fa4cb46db448278f9bf4c7f) |
| Navigation hints / latest blocked draft | [3a79fa5](https://github.com/cryptoadya/garage-rag/commit/3a79fa5369fed45d8425ef9ed1f5d9f77c290784) / [de6a965](https://github.com/cryptoadya/garage-rag/commit/de6a96577322feffcb2ea59c649070402f3ac1f1) |
