# Testing and release checks

## Offline acceptance

```bash
python -m pip install -e '.[dev]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= python -m pytest
ruff check .
python -m pip check
python -m build
```

[CI](https://github.com/cryptoadya/garage-rag/actions/workflows/ci.yml) runs pytest
and Ruff on Python 3.12/Linux with model downloads disabled, plus the frontend
tests and build on Node.js 22. The Python build command
creates wheel/sdist artifacts in ignored `dist/`. Local acceptance also uses
Python 3.12/macOS arm64 CPU; MPS is not validated.

Tests generate PDFs from the original fictional [Aurora corpus](../tests/fixtures/corpus.json).
A local tokenizer and deterministic vectors avoid private BMW material, downloaded
weights and live API calls. Qdrant persistence/retrieval and LangGraph execution
are real; model responses and HTTP transports are controlled fixtures.
Production-entry tests cover unchanged EN/DE/RU search queries, reopened legacy
indexes, original question/answer language, call/token telemetry and legacy
translation settings. Expansion tests cover same-language zero-call behavior,
actual-text language hints/cache invalidation, maximum-cosine batch merging and
strict translation validation. Language-combination tests cover EN/DE/RU and mixed source
passages in all three workflows, plus rejection of translated/invented quotes,
wrong sources/pages and unsupported numeric claims. Groq HTTP transport fixtures
also cover German and Russian evidence. These offline checks do not establish
live provider answer quality or general multilingual retrieval recall. See the
[matched real BMW retrieval comparison](benchmarks/multilingual-ru-en.md) for the
measured RU→EN regression and corrected rankings. The
[real Bosch multilingual matrix](benchmarks/multilingual-c3.md) additionally covers
288 question/corpus combinations, independently anchored evidence chunks,
monolingual and mixed PDFs/collections, and zero live provider calls. Unit tests
inject local relevance scores, so ordinary offline CI needs no reranker weights;
they cover source-language pair selection, bounded candidates, unchanged source
objects, invalid score rejection, top-k and long-pair dense preservation.

If the checkout contains private historical graph captures, their replay also
requires the pinned Groq tokenizer in the local cache. Set
`HF_HOME="$PWD/.cache/huggingface"` when using the repository's existing cache.
A fresh clone skips those private-capture tests; do not download model weights
or make provider calls to run them.

| Area | Contracts checked |
| --- | --- |
| Ingestion/index | Page provenance, cleanup, token bounds, overlap, revision reuse/recovery and profile isolation |
| Retrieval evaluation | Hit/MRR arithmetic, labels, frozen/disjoint datasets, corpus guards, active-only BM25, fusion and reranking |
| Generation/workflow | EN/DE/RU, source/quote/schema validation, numeric/table regressions, all-claim audit coverage, expansion retention, budgets and safe failures |
| Agentic retrieval | Evidence-dependent dense/BM25 and multilingual paths, real local search, cumulative source comparison, native call/result IDs, refusal reasons, strict budgets and mandatory audit |
| Provider wire behavior | Fixed Groq endpoint/model, disabled retries/fallbacks, truncation, plan/quota guards, headers, tokenizer accounting and timing |
| Evaluation/review | Checkpoint/resume identity, exact answer/source hashes, mixed reviewer provenance, capture tampering and unfilled private review packets |
| Three-mode comparison | EN/DE/RU matched matrices, pairwise completeness, citation/warning labels, audit separation and shared tool-bound session guards |
| Privacy | Sanitized errors/observations and whole-graph tracing suppression |

These establish software contracts, not embedding quality, remote reasoning
quality, representative latency or vehicle safety. Scripted answer comparisons
are documented in [answer evaluation](benchmarks/grounded-answering.md); they
are separate experiments and are not required to rerun for documentation changes.

## Frontend checks

From the repository root, with Node.js 22.12+ (22.x), 24.x or 26+:

```bash
cd frontend
npm ci
npm test
npm run build
```

Vitest and Testing Library use controlled HTTP responses for request fields,
mode/language selection, duplicate submission, sources, refusals and errors,
plus document listing, PDF transfer progress, indexing/readiness, upload failures
and immediate questions after upload. API tests use real synthetic PDFs and
Qdrant Local to check empty-index startup, ingestion/reuse, safe file storage,
size/type/name validation, failure cleanup and upload/answer exclusivity.
PDF viewer tests cover exact hash-bound bytes, missing/CLI originals, inactive
and superseded revisions, tampering, symlinks and traversal. Frontend tests cover
opening structured citations, page navigation, cleanup and unavailable originals.
They make no provider calls. The build includes TypeScript checking.
For browser smoke testing, check desktop/mobile layouts, submit in both modes,
follow a source reference and verify refusal/error recovery. A live submission
uses the configured backend provider and its quota guards; it requires an
explicitly authorized live scope. See [local UI setup](usage.md#local-web-ui).

## Optional manual checks

Follow the [quick start](../README.md#quick-start) with a permitted EN/DE/RU or mixed-language PDF.
Repeat indexing to check reuse; inspect `stats` and `search --json` before `ask`.
For explicitly authorized live checks, compare EN/DE/RU answers and an answer-language
override with the exact sent evidence and original PDF. Also inspect refusal for
an absent repair specification. Keep captures private. Connectivity and valid
citations do not establish factual accuracy or representative refusal behavior.
Free models can disappear, rate limit, time out or emit invalid JSON; do not
introduce paid fallbacks to obtain a passing result.

## Packaging and publication

Inspect both wheel and sdist contents: the wheel must contain the runtime package,
entry point and MIT license; the sdist additionally contains public docs, tests,
synthetic fixtures, frozen benchmark JSON and frontend sources/configuration with
the npm lockfile. The UI runs from the source checkout or unpacked sdist; it is
not bundled or served by the Python wheel. Neither artifact may include `.env`,
PDFs, indexes, model caches, captures, private archives, `node_modules` or frontend
build output. See the [retained benchmark inventory](benchmarks/README.md).
Install the built wheel in
an isolated environment and check `garage-rag --help` and module entry points.
The sdist allowlist lives in [pyproject.toml](../pyproject.toml).

Check relative Markdown paths/anchors and external links, CLI examples against
`--help`, dependency consistency and the final diff/status. Ignored files already
tracked or present in Git history remain public. New benchmark outputs belong in
ignored `output/`; publish reviewed aggregate summaries, not raw captures.

Before publishing a release, verify the intended `main` commit, its green CI, tests,
lint, package builds and existing tags/releases. Move the existing `Unreleased`
entries into a dated version section, leave an empty `Unreleased`, and update
comparison links. Match package version, `v`-prefixed tag and GitHub Release; use that
CHANGELOG section as release notes. Merge/tag/Release require separate owner
authorization. Preparation alone does not publish a release.
