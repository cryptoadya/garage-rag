# Phase 3 engineering release audit — 2026-10-26

Scope: PR #3, `d2f4e52..f9d699a`, plus the targeted audit fixes in this PR.
This is implementation-agent engineering review, not independent human answer
review, a penetration test or vehicle validation. No new LLM requests or mass
benchmarks were needed. Historical results remain unchanged.

## Decision and release blockers

**Ready to merge as the engineering v0.3.0 milestone after the audit-fix commit's
CI passes. No known critical implementation blocker remains.** This is not
acceptance of general answer-quality improvement. Independent quality acceptance
remains open. Merge, tag and Release are separate owner actions, not performed
by this audit. At publication, verify the intended main commit/CI, tests, lint,
builds, existing tags/releases and finalize the existing Unreleased section.
The package version is already 0.3.0; Unreleased must remain until publication.

Confirmed defects repaired during this audit:

- Partial claim review previously copied the latest review's independence flag
  over mixed agent/legacy labels. Claim rows now retain reviewer provenance;
  aggregate independence requires every scored row to have human provenance.
  Empty reviewer identities are rejected atomically.
- Private review HTML exposed existing quality judgments despite instructing
  reviewers to decide first. Observations now use an explicit allowlist omitting
  prior labels, provenance, answerability and reference-match diagnostics.
- The observation wrapper lost provider identity on transport failures, so a
  Groq timeout was described as OpenRouter. It now forwards provider attributes.
- Architecture/privacy/budget documentation now includes audit drafts and Groq's
  distinct token defaults and account-dependent billing boundary.

## Engineering coverage

| Surface | Finding |
| --- | --- |
| LangGraph | Sequential typed state, no checkpointer/tools/planner; at most four attempted completions and two local searches. Drafts reserve an audit; only insufficiency can expand once, with unchanged query/filters and prior passages retained. Schema, quote, language, number and all-claim audit failures refuse. |
| Privacy and citations | Selected passages, metadata, question and audit draft leave the machine for the selected provider. Local source IDs/pages and exact normalized quotes are checked. Whole-graph LangSmith tracing is disabled; public traces and errors are allowlisted. Private captures/review packets remain ignored. |
| OpenRouter | Fixed endpoint, free-model allowlist, zero prompt/completion/request price caps, SDK/application retries disabled, provider fallbacks disabled; no alternate-model list. Actual SDK wire behavior is covered by mock HTTP tests. |
| Groq | Fixed endpoint/model and on_demand tier, no redirects/retries/fallback or upgrades, Free Plan attestation, conservative quota guards/checkpoint reservations. No API-enforced zero-price cap; an external plan change can make requests billable. |
| Tests and reproducibility | Offline synthetic extraction/index/search contracts, real graph routing, wire payloads, budgets, failures, numeric/table regressions, mixed review provenance and tamper/HTML checks. Frozen dataset/index/runtime hashes bind saved live evidence; no general model-quality claim follows from fixture tests. |

Provider contract references: [OpenRouter routing](https://openrouter.ai/docs/guides/routing/provider-selection),
[Groq tiers](https://console.groq.com/docs/service-tiers) and
[Groq billing](https://console.groq.com/docs/billing-faqs). Contracts were checked
against documentation and offline wire tests; no new live compatibility probe
was performed.

## Limitations to document; future quality work

- Same-model drafting/auditing can share errors or follow prompt injection.
  Contained quotations do not prove entailment, applicability or completeness.
  Genuine erroneous audit-eligible drafts with independent labels and supported
  controls are still needed to measure harmful acceptance and rejection.
- Numeric/table heuristics are conservative and incomplete. No OCR/layout
  reconstruction exists; source-mapping safety and numerical agreement with the
  original PDF are different judgments. False refusals remain possible.
- Top-20/8-chunk/12,000-character retrieval cannot establish document-wide
  absence or find every conflict. EN/DE/RU/profile quality sampling is limited.
- No whole-call/workflow wall-clock deadline, transitive cross-platform lockfile,
  MPS validation or provider-availability guarantee. Pinned direct dependencies
  and source/index hashes bound reproducibility, not future live model behavior.
- Provider retention/training policies are external; this is local-first search,
  not offline generation or a zero-retention guarantee. Groq local quota histories
  do not account for other processes; keep account plan/limits under owner control.

These limitations need not block this alpha engineering milestone when disclosed.
Independent human review, broader balanced sampling, auditor efficacy, layout
reconstruction and stronger runtime/account controls can be separate future work.
They must precede stronger quality/safety claims, not a newly enlarged benchmark
in this release audit.

## Existing evidence and external-review handoff

Both private archives were read back and SHA-256 verified: 285 historical archive
members (plus identical local mirror), and 51 latest regression files. Six frozen
Phase 2 dataset/result JSON files are unchanged from main. No new raw Phase 3
reports, private documents, captures, indexes or archives are tracked. Intermediate
historical Git commits still contain old reports; no history rewrite was attempted.
See the [archive inventory](benchmarks/phase3-archive.md).

The ignored local package `output/private/phase3-external-review-20261026/`
contains 13 groups / 38 saved answers: five current repair pairs and fourteen
historical BMW pairs, kept separate. Its entry point is `index.html`; README
defines the review protocol and zero-API rescoring command. It contains the
hash-verified original PDF, questions, exact requests/replies, unscored report
copies, blank claim/source review templates and a file manifest. No human labels
are filled. Mode identities and raw auditor verdicts remain visible, so this is
not a blinded study. It cannot establish auditor prevention without suitable
erroneous drafts. Keep the package private and outside Git.

Validation for the audit fixes: 270 offline pytest tests, Ruff, dependency
consistency and wheel/sdist build with payload inspection. CI on the starting
`f9d699a` passed; the follow-up commit requires its own green CI before merge.
