# Changelog

Significant user-facing changes are recorded here. Detailed development history
is available in [Git commits](https://github.com/cryptoadya/garage-rag/commits/main).

## [Unreleased]

The package version is 0.2.0; these changes are not yet tagged as a release.

### Added

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

[Unreleased]: https://github.com/cryptoadya/garage-rag/compare/v0.1.0...main
[0.1.0]: https://github.com/cryptoadya/garage-rag/tree/v0.1.0
