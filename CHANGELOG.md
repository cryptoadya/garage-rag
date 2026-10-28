# Changelog

Significant user-facing changes are recorded here. Detailed development history
is available in [Git commits](https://github.com/cryptoadya/garage-rag/commits/main).

## [Unreleased]

### Added

- Phase 4B comparison of direct, grounded and agentic using the existing answer
  evaluator: a fixed fictional EN/DE/RU suite, matched and pairwise summaries,
  completeness, claim/citation support, warning omissions and resource accounting.
- Hash-bound native tool captures and external review packets; same-model audit
  decisions remain observations, never quality labels. Live quality stays unscored
  until explicit review; scripted results measure software contracts only.
- Shared live call/pacing/token guards across tool-bound and ordinary requests.
  Comparative live runs require explicit caps and private captures; no live calls
  are part of offline acceptance. Package version prepares the upcoming 0.5.0.

### Fixed

- Agentic explains semantic Dense versus exact-term BM25 search and supplies
  compact, source-bound navigation hints only to Planner. Reference labels remain
  opaque; no page tool, forced search sequence or answer-gate change is introduced.
  Updated planner inputs stay inside the existing token-aware budget checks.
- Agentic checks unspent writer/auditor token headroom before another planning
  turn and hands off substantive evidence when that turn would displace audit.
  Forecasts include complete request/schema/output costs and draft framing/growth;
  they are never charged as requests. Actual normalized audit inputs are checked
  again, failed calls retain their reservations, and no unaudited positive answer
  can escape. Offline BMW replay now completes five calls at the ordinary limits;
  this is orchestration validation, not a live quality result.
- Agentic model inputs omit known index bookkeeping and carry compact diagnostics;
  full provenance and diagnostic maps remain in private evidence/trace captures.
  Exact audit preflight refuses remaining overruns without treating an unaudited
  draft as answer recovery.
- Agentic separates navigation references from substantive evidence, filters generic
  query matches, and attaches conditions through procedure-page or explicit scope
  links. Bare conditional words and adjacent pages no longer protect every hit;
  explicit warnings with unknown scope remain protected. Navigation-only completion
  refuses before writing. Offline captures distinguish corrected packing from a
  controlled three-search workflow; historical live outcomes remain unchanged.
- Agentic keeps a private evidence store separate from its bounded model context.
  Deterministic whole-passage selection can omit low-relevance overflow while
  preserving stable citations, topical evidence, semantic leads, safety cues and
  page companions. Protected overflow and uncertain relevance stop explicitly.
  Sanitized selection/delivery diagnostics and private raw search captures expose
  exclusions; planner history carries one current context instead of stale copies.
  This is offline software validation, separate from historical Phase 4B results.
- Groq Agentic planner uses strict Structured Outputs decisions with local search
  dispatch, without combining native tools and structured output. Schema overhead
  stays inside shared token/quota guards; Direct/Grounded and writer/auditor keep
  their existing behavior. Historical 400s are replayed offline, not retried live.
- Agentic planner uses serial required native search/finalize/refuse actions instead
  of mixed tool calls and plain-text controls. Completion/refusal adds no search;
  existing budgets, evidence gates and mandatory writer/audit remain unchanged.
  Offline regressions cover the second planner turn after a native tool result;
  provider-side reliability still requires a separately authorized live smoke.
- Classify Agentic provider failures from structured HTTP/transport observations
  when the workflow returns a safe generic error; keep failures and their token
  reservations visible without changing generation, auditing or retry behavior.

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

[Unreleased]: https://github.com/cryptoadya/garage-rag/compare/v0.4.0...main
[0.4.0]: https://github.com/cryptoadya/garage-rag/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cryptoadya/garage-rag/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cryptoadya/garage-rag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cryptoadya/garage-rag/tree/v0.1.0
