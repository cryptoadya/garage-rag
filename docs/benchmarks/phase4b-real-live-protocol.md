# Phase 4B: preregistered real-corpus comparison

Prepared offline on 2026-10-27. **Not authorized to run yet. Phase 4B remains
open.** This replaces the exact-query Aurora retrieval fixture as the proposed
comparison. The earlier incomplete Grounded answer and false auditor acceptance
remain measured results; neither is relabelled as a success or erased.

## Fixed selection

Use the existing `benchmarks/bmw_e39_2002.phase2b-heldout.v1.json`, selecting only
the following two existing questions, EN, one trial each:

| ID | Exact question | Source-based completeness criteria |
| --- | --- | --- |
| `hood_release` | Where is the lever that unlocks the hood? | Identify the lever under the left-hand side of the instrument panel; naming only the hood or an instruction section is incomplete. |
| `toolkit_access` | Where is the onboard tool kit and how do I access it? | State that it is attached to the underside of the luggage compartment lid/tailgate and that access requires unscrewing the wingnut. Both requested parts must be present. |

Physical PDF pages **137 and 150** were checked by text extraction and rendered
page inspection before any new live output. The location-only hood question does
not request servicing or closing instructions. Do not require unrelated service
or adjacent windshield-wiper warnings; any added unsafe instructions, invented
specifications or lost applicability conditions still fail source review.

These cases were selected before this preparation's retrieval inspection. They
are previously used benchmark needs, not a new blind held-out sample. Two needs
cannot support significance, general superiority or vehicle-safety claims. The
second question has two requested facts; it is not guaranteed to require multiple
searches or give Agentic an advantage.

## Fixed corpus and genuine retrieval

- Original local PDF: `data/private/bmw_e39_2002_manual.pdf`, 186 pages,
  SHA256 `9faf9c33f9958f037c77032e514295378ddb490217c1d37015d6ca0fb48b4574`.
- Dataset canonical SHA256:
  `a3f69c2c075e3fad4adaf6eae27b85f774e24090c6c7b3934214096d24a72077`.
- Existing index: `data/qdrant`, 241 active chunks, baseline multilingual E5
  Small; fingerprint
  `abcaba77b00d2a4440d2f2db11de669a2b12f27d96cf42532484084638da473f`.
- Frozen logical chunk digest:
  `808cfb5e7ef6cf11d343c863f79a4ac562bbe83627a12d52c85c74b81b450a24`.

Use the evaluator's existing corpus branch and `LocalSearch`: real E5/Qdrant
dense search and BM25 over the same active corpus. No query equality switch,
query dictionary, scripted ranks, seeded search results, injected gold passages,
new index, ingestion, profile tuning or production prompt changes. Direct and
Grounded use identical initial top-5 dense results; Grounded retains its existing
unchanged-query top-20 expansion and context limits. Agentic starts empty and
chooses queries/method/top-k through existing serial native local tools (cap 5).

Offline inspection found the hood evidence at dense rank 1. The toolkit evidence
is absent from dense top-5 and at raw rank 14 in top-20; the unchanged context
builder sends only the first eight fitting chunks, excluding that passage.
Natural BM25 on the original toolkit question finds page 150 at rank 1. This is
a real retrieval opportunity, not a promised live route or model-quality result.
Save all actual sent contexts: a retrieved-but-unsent page does not establish
Grounded evidence availability. Do not force Agentic to use the observed BM25
route or change either question after approval.

## Model, schedule and hard budget proposed for approval

Same Groq `openai/gpt-oss-120b` model, temperature zero,
`max_completion_tokens=1024` **including reasoning**, existing reasoning settings,
one trial of Direct/Grounded/Agentic for each question: **six planned outputs**.
Seed `20261027` fixes the schedule. Default per-question ceilings are unchanged:

| Mode | API attempts per question | Local searches per question | Maximum reserved input + output |
| --- | ---: | ---: | ---: |
| Direct | 1 | 1 | 8,000 |
| Grounded | 4 | 2 | 32,000 |
| Agentic | 6 | 4 | 24,000 |
| Both questions, all modes | **22** | **14** | **128,000** |

Completion allowance is at most **22,528 tokens**, already included in 128,000.
The session guard holds reservations without refunds, including failures.
Keep actual HTTP starts at least **61 seconds apart**; no concurrency, retries,
fallback, model switch or automatic resume. Stop the entire experiment on the
first provider/session-budget error; a missing mode remains missing. Workflow
refusals/bounded stops are recorded as such, never rerun to complete a matrix.

Use the existing opt-in **pinned offline tokenizer** accounting, revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. It was loaded locally with no download.
Initial top-5 draft reservations are approximately 3,879 / 3,225 tokens for
hood/toolkit, versus 11,346 / 8,278 with byte accounting. Byte mode would exceed
the 8,000 TPM guard before inference. This is an accounting choice already
supported by the provider adapter; it does not reduce context or output budgets.
Audit/history-dependent reservations remain dynamic. TPM, per-question and
session guards may stop earlier; six successful outputs are not guaranteed.

Immediately before an authorized run, recheck PDF/index/dataset/code/tokenizer
bindings, the same verified Free Plan/key evidence and any known cooldown.
Restore the **latest cumulative** quota checkpoint, including the failed 4,040
reservation from the partial Aurora run and any newer usage. Age entries across
the rolling 86,400-second window; do not reset at midnight or add overlapping
snapshots. Require the full 128,000-token and 22-request local headroom before
sending anything. Stale provider headers do not verify organization-wide balance.

The existing CLI does not import another run's quota history automatically and
`--requests-per-minute 1` gives 60.01, not 61 seconds. Consequently run the
existing evaluator **only through the same execution-only quota-restoration,
one-shot marker and HTTP-spacing controls used for the preserved Aurora run**,
adapted to this frozen selection and current commit after authorization. Do not
invoke a bare CLI command as though it enforces those extra conditions. This is
not a new evaluator; no launch or automatic task is installed by this protocol.

## Independent source assessment, fixed before generation

Use the existing private captures, `review_packet` and hash-bound claim/source
review JSON with the existing offline rescoring path. A reviewer separate from
the workflow generator/auditor and the implementation agent must inspect the
original PDF and exact answers. Independent human review is labelled true only
after an actual human review; otherwise state the reviewer identity and limits.
The current packet exposes mode names and audit details, so do not call it
blinded. Fix source-based judgments before consulting auditor verdicts or costs.

For each answer, judge correctness/applicability, every requested part, each
entire claim against its cited source, exact quote/page mapping and relevant
conditions/warnings. The legacy corpus schema supports claim/completeness and
sent-source review; preserve per-claim citation/warning judgments in answer-hash-
bound private review notes rather than inventing automatic labels. Missing or
uncertain judgments remain unreviewed. A source pointer does not answer a location
question. Authentic quotes alone never establish completeness.

Distinguish a corpus-relative false refusal from missing sent context and from a
provider/validation failure. Count Agentic recovery only if it obtains necessary
evidence absent from **all actually sent Grounded contexts**, then produces a
complete, source-supported answer. If Grounded already had that evidence, any
quality difference is generation/audit behavior, not retrieval recovery. Ties,
no benefit, failures and incomplete answers are valid outcomes.

Report each mode's calls/searches, input and completion tokens (reasoning
included), missing-usage flags, held reservations, workflow wall time and HTTP
time including failures. Separate pacing/initialization from inference. Include
the one existing local warmup separately from metered searches. Retain partial
attempt resources; do not treat missing usage as zero. Only successful matched
rows form quality pairs/triples; use the existing `paired_summary` for each
pair as needed. No audit acceptance becomes an independent evaluation label.

## Offline readiness and stopping boundary

Local PDF/index/chunk/tokenizer checks, genuine dense/BM25 inspection and an
offline contract run reuse the existing evaluator and packet generator. Contract
model replies are controlled refusals and provide **no live-quality estimate**.
Private preparation evidence is under ignored
`output/private/phase4b-real-preparation-20261027/`.

The saved HTTP 400's backend cause remains uncertain (see the diagnostic section
in the comparison report); offline transport replay cannot certify a Groq fix.
The Grounded false completeness acceptance remains a measured auditor weakness.
**Phase 4B stays open pending an explicitly authorized real-corpus comparison and
separate source review.** No new live request, merge, tag or Release is authorized
or performed by this preparation.
