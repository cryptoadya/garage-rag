# Changelog

Significant user-facing changes are recorded here. Detailed development history
is available in [Git commits](https://github.com/cryptoadya/garage-rag/commits/main).

## [Unreleased]

## [0.3.0] - 2026-10-26

### Added

- Bounded LangGraph answering with claim/completeness auditing, one local
  top-20 expansion, explicit budgets and conservative numeric/table checks.
- Optional Groq Free Plan transport, quota accounting and resumable evaluation;
  OpenRouter remains the default with zero-price caps and no retries/fallbacks.
- Separate scripted/live paired evaluation, hash-bound claim/source review and
  private review packets. Compact historical results retain limitations and costs.

### Fixed

- Numeric-gate false positives for wrapped dual-unit prose and nonnumeric claims
  citing sources with unrelated numbers; unsupported values and tables remain blocked.
- Expanded context losing relevant passages to presentation overhead; retain full
  prior text and applicability metadata within the existing context budget.
- Mixed claim reviews incorrectly inheriting the latest review's independence;
  retain per-row provenance and require a named reviewer. Private review packets
  omit existing quality judgments, and observed Groq timeouts retain their provider.
- Source distributions omitting three frozen Phase 2 result artifacts because of
  the raw-report ignore rule; explicitly retain those public baselines.

### Changed

- `ask` defaults to grounded answering; `--workflow direct` preserves the original
  single-completion path.
- Benchmark reports default to ignored local output. Raw Phase 3 results and
  intermediate reports move to a verified private archive; Git keeps final summaries.
- Consolidated architecture/workflow and Phase 2A methodology/results documentation;
  shortened onboarding and evaluation reports, and centralized provider guidance.
- Added package source/documentation/issue links and the build frontend to development
  dependencies so documented release checks work after a development install.

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

[Unreleased]: https://github.com/cryptoadya/garage-rag/compare/v0.3.0...main
[0.3.0]: https://github.com/cryptoadya/garage-rag/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cryptoadya/garage-rag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cryptoadya/garage-rag/tree/v0.1.0
