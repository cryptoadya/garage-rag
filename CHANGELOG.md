# Changelog

Significant user-facing changes are recorded here. Detailed development history
is available in [Git commits](https://github.com/cryptoadya/garage-rag/commits/main).

## [Unreleased]

### Fixed

- Pace Groq calls before sending against RPM/RPD/TPM/TPD and observed reset
  headers, waiting up to 65 seconds per completion for short quota windows.
  Expose long cooldowns as HTTP 429 with `Retry-After`, distinguish oversized
  requests, and record quota wait separately from HTTP time. No HTTP retries.
- Accept intact German/Russian numeric prose for Grounded semantic review while
  retaining damaged-table and value-substitution rejection. Request complete
  procedures and bounded evidence; Grounded can revise an incomplete draft once
  within its existing four-call budget, with a mandatory second audit.
  Direct/Grounded require source IDs and verified source-language quote excerpts;
  source references alone cannot substitute whole chunks for missing quotes.
  No procedure-specific claim rewriting or PDF-font-specific quote repair occurs.
  Verify numbered numeric prose independently of unrelated table rows and recognize
  wrapped US dual units. Spelled-out counts retain conservative rejection without
  object-specific exceptions. Avoid resending
  whole source quotations inside audit drafts when the source is already included.
- Preserve original EN/DE/RU questions and verbatim source-language citations.
  Direct/Grounded add missing source-language queries in at most one Groq call,
  retaining translation opt-out, validation, audit and quota guards. Existing
  indexes need no migration.
- Correct expanded mixed-language retrieval with the pinned local reranker:
  reserve each query's top five within 20 candidates / 60 intact pairs. Model
  loading is lazy; oversized pairs retain dense order. BM25 statistics are built
  only for lexical search. See the [multilingual benchmark](docs/benchmarks/multilingual-c3.md)
  for measured recall, top-one tradeoffs and added local CPU/memory cost.

## [0.5.0] - 2026-10-29

### Changed

- Phase 5E: Groq is the sole LLM provider for CLI, API and all modes. Remove the
  alternate transport, SDK dependencies and current setup instructions; retain
  historical benchmark reports. Existing Groq model, Free Plan attestation,
  conservative quota accounting and no-retry/fallback guards remain unchanged.

- Shorten the README around CLI/API/UI onboarding; keep operational details in
  the usage/testing guides and document retained benchmark artifacts. Mark the
  Phase 4B preparation protocol as historical. Include frontend sources and the
  npm lockfile in source distributions; the Python wheel remains runtime-only.

### Added

- Phase 5E PDF viewer: open a structured citation at its physical PDF page,
  navigate pages and return to the answer. Serve only SHA-256-verified private
  uploads matching active document revisions; unavailable originals (including
  earlier CLI imports) receive a clear message. Existing uploads need no migration.

- Phase 5D PDF management: stream PDF uploads to private local storage, index
  them with the existing ingestion/embeddings/Qdrant pipeline, and list active
  documents. The React UI shows upload progress, indexing, readiness and errors.
  New indexes can start empty; indexing and answers share exclusive access.
- Phase 5C structured HTTP citations: final supported claims expose document,
  physical PDF page and exact quote from existing verified evidence. The UI
  groups citations by claim without parsing answer text; refusals expose no
  rejected draft evidence. Existing answer text, CLI and workflow behavior remain unchanged.
- Phase 5B local React/TypeScript/Vite UI: question submission to the existing
  FastAPI, Direct/Grounded and answer-language selection, source-reference links,
  loading/refusal/error states and responsive layout. Local dev/preview proxy,
  frontend behavior tests and type-checked build; no backend workflow changes.
- Phase 5A FastAPI MVP: `GET /health` and `POST /ask` reuse Direct/Grounded
  workflows and CLI public responses, with strict EN/DE/RU request validation,
  sanitized errors, one in-flight answer and lifespan-owned Qdrant/provider resources.
- Three-mode answer evaluation: fixed fictional EN/DE/RU cases, matched/pairwise
  summaries, completeness, citation/warning review and resource accounting.
- Hash-bound tool/schema captures and external review packets; same-model audits
  remain observations. Shared live call/pacing/token guards cover every binding.
  Comparative live runs require explicit caps and private captures.

### Fixed

- Render supported web answers once from structured claims, with compact PDF-page
  links and grouped, collapsible evidence. Keep full filenames and verbatim quotes
  in source details instead of duplicating the answer and inline citation text.
- Direct/Grounded CLI/API questions in Russian/German first translate the search
  query to English, addressing cross-language retrieval misses in English manuals.
  Preserve the original question, answer language, citation checks and answer
  budgets; account for one additional bounded Groq attempt in workflow telemetry.
  Reject invalid translations without retry/fallback. Local CLI search and
  historical evaluation primitives remain unchanged; translation can be disabled
  with `GARAGE_RAG_TRANSLATE_QUERY=false`.
- PDF uploads distinguish invalid source PDFs (422) from local indexing and
  embedding validation failures (500), retaining existing CLI PDF diagnostics.
- Local web UI text contrast for hints, source metadata, loading and refusal states.
- Groq Agentic uses strict structured planner decisions with local search dispatch;
  the optional native backend uses serial search/finalize/refuse actions.
- Agentic separates private evidence storage from bounded whole-passage context,
  preserves stable source IDs and safety dependencies, distinguishes navigation
  from procedure evidence, and sends compact history/diagnostics. Protected
  overflow and uncertain relevance refuse explicitly.
- Token-aware handoff protects writer/auditor capacity before extra planning.
  Actual audit requests are checked again; failed requests retain reservations.
- Planner-only bounded navigation hints explain dense/BM25 choices without
  guessing PDF pages, forcing a search sequence or changing answer gates.
- Evaluation classifies provider failures from structured observations even when
  the workflow returns a sanitized generic error.

Agentic remains experimental; Direct/Grounded are primary. See the
[consolidated comparison](https://github.com/cryptoadya/garage-rag/blob/v0.5.0/docs/benchmarks/phase4b-comparison.md)
for results and limitations.

## [0.4.0] - 2026-10-26

### Added

- Opt-in Phase 4A agentic LangGraph retrieval on Groq GPT-OSS 120B: model-selected
  local dense/BM25 tools, evidence-driven multilingual query rewrites, cumulative
  source comparison and justified completion/refusal with mandatory semantic audit.
- Native serial Groq tool calls/results, a closed read-only tool allowlist and
  per-question call/search/token/step budgets on top of existing Free Plan guards.
- Offline adaptive-path, safety and protocol acceptance tests. Direct/grounded
  behavior and frozen metrics remain unchanged.
- Real Groq Aurora acceptance of native tool/result delivery, evidence-driven
  RU dense to EN BM25 search, cited writing and mandatory audit. The bounded
  1024-token RU pair completed both modes within the shared guard: seven API
  attempts, 8,081 actual tokens and 29,781/40,000 reserved tokens.

### Fixed

- Separate agentic retrieval/control from structured answer writing after a Groq
  smoke exposed answer JSON instructions alongside native tools. Preserve native
  result IDs and full evidence without advertising tools to the writer; retain
  the three-call one-search path and strict caps; the full cycle is live confirmed.

### Validation and limitations

- Offline tests, Ruff, dependency consistency, wheel/sdist inspection and
  installed-wheel entry-point checks cover the release's software contracts.
- Agentic retrieval is functionally confirmed; this synthetic comparison found
  no quality or overall resource advantage over Grounded and establishes no
  general superiority. Same-model auditing is not independent ground truth.
- Valid quotations and semantic acceptance do not guarantee that the selected
  quotes cover every part of a compound claim or retain every source warning.
  OCR, full table/layout reconstruction and figure understanding remain absent.

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

[Unreleased]: https://github.com/cryptoadya/garage-rag/compare/v0.5.0...main
[0.5.0]: https://github.com/cryptoadya/garage-rag/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cryptoadya/garage-rag/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cryptoadya/garage-rag/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cryptoadya/garage-rag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cryptoadya/garage-rag/tree/v0.1.0
