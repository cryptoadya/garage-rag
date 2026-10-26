# Changelog

Significant user-facing changes are recorded here. Detailed development history
is available in [Git commits](https://github.com/cryptoadya/garage-rag/commits/main).

## [Unreleased]

### Added

- Separate Groq English direct/grounded cold-pressure comparison with hash-bound
  agent source review, complete checkpoints, usage/quota/paced-latency evidence
  and a bounded, unexecuted DE/RU follow-up proposal.

- Optional Groq Cloud Free Plan chat provider (`openai/gpt-oss-120b`) selected by
  CLI/environment, preserving OpenRouter, direct/grounded and EN/DE/RU contracts.
  Includes safe API/quota observations, Retry-After, conservative RPM/TPM/RPD/TPD
  guards and explicit account-plan attestation; no retries, upgrades or fallbacks.
- Atomic live-result checkpoints and explicit resume that skips completed modes,
  retains prior errors and binds results to provider/model, data, code and index.

- Bounded LangGraph answering with a separate semantic claim/completeness audit,
  one conditional local top-20 expansion, explicit request/search budgets,
  conservative numeric-table refusals and privacy-safe local observations.
- Separate EN/DE/RU answer-quality scenarios and a paired direct/grounded
  evaluator: offline scripted regressions, explicitly budgeted free-only live
  runs, hash-bound claim review, refusal/false-refusal metrics, latency and calls.
- Workflow architecture, evaluation methodology and measured limitations in docs.
- Enforced live-evaluation request pacing (up to 15/minute) and runtime attempt
  budgets shared by direct drafts, grounded drafts and audits. Sanitized workflow
  traces retain provider HTTP status/category without response payloads.
- October 26 quota/probe diagnostics and partial live smoke results, with
  separate agent review and explicit incomplete-matrix status.

### Changed

- `ask` defaults to grounded answering; `--workflow direct` retains the original
  single-completion behavior. Existing JSON fields and local citation/language
  validation remain; workflow observations are additive.
- Prepare package version 0.3.0. No Phase 3 release/tag is published before merge.

## [0.2.0] - 2026-10-25

### Added

- Isolated, revision-pinned local embedding profiles via `--profile`, preserving
  the original E5 Small default, CLI and generation validation.
- Phase 2B offline comparison of dense, BM25, hybrid and reranked retrieval,
  frozen development/held-out EN/DE/RU questions, separate chunk-sufficiency
  metrics, latency/resource measurements and reproducibility provenance.

- Offline Phase 2A retrieval benchmark with 18 EN/DE/RU information needs,
  four separate unanswerable cases, original-PDF agent review provenance,
  Hit@1/5/10, MRR@10, warm latency, and a tire-pressure failure analysis.
  Existing retrieval and embedding behavior is unchanged.
- EN/DE/RU answer selection with local language detection, English fallback and
  `--answer-language auto|en|de|ru`.
- Localized refusals and a resolved `language` field in JSON answers, while
  preserving English evidence and local PDF references.
- Offline multilingual regression coverage and thematic project documentation.

### Changed

- Increased the shared reasoning/answer budget to 8,192 tokens, configurable
  with `OPENROUTER_MAX_TOKENS` from 512 to 32,768.
- Reject truncated output explicitly as `output_token_limit`, with actionable
  CLI diagnostics and no automatic retry.
- English-first CLI instructions, project presentation and generation prompt.
- Broader private-data exclusions and an explicit source-distribution allowlist.

## [0.1.0] - 2026-10-25

### Added

- Initial stable CLI for page-aware PDF ingestion, local E5 embeddings,
  persistent Qdrant search and resumable indexing.
- Free-only OpenRouter generation with grounded Russian claims, strict source
  and quote validation, and local PDF page references.
- Synthetic test corpus, automated tests, CI and MIT-licensed project code.

[Unreleased]: https://github.com/cryptoadya/garage-rag/compare/v0.2.0...main
[0.2.0]: https://github.com/cryptoadya/garage-rag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cryptoadya/garage-rag/tree/v0.1.0
