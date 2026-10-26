# Testing and release checks

## Offline acceptance

```bash
python -m pip install -e '.[dev]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= GROQ_API_KEY= python -m pytest
ruff check .
python -m pip check
python -m build
```

[CI](https://github.com/cryptoadya/garage-rag/actions/workflows/ci.yml) runs pytest
and Ruff on Python 3.12/Linux with model downloads disabled. The build command
creates wheel/sdist artifacts in ignored `dist/`. Local acceptance also uses
Python 3.12/macOS arm64 CPU; MPS is not validated.

Tests generate PDFs from the original fictional [Aurora corpus](../tests/fixtures/corpus.json).
A local tokenizer and deterministic vectors avoid private BMW material, downloaded
weights and live API calls. Qdrant persistence/retrieval and LangGraph execution
are real; model responses and HTTP transports are controlled fixtures.

| Area | Contracts checked |
| --- | --- |
| Ingestion/index | Page provenance, cleanup, token bounds, overlap, revision reuse/recovery and profile isolation |
| Retrieval evaluation | Hit/MRR arithmetic, labels, frozen/disjoint datasets, corpus guards, active-only BM25, fusion and reranking |
| Generation/workflow | EN/DE/RU, source/quote/schema validation, numeric/table regressions, all-claim audit coverage, expansion retention, budgets and safe failures |
| Agentic retrieval | Evidence-dependent dense/BM25 and multilingual paths, real local search, cumulative source comparison, native call/result IDs, refusal reasons, strict budgets and mandatory audit |
| Provider wire behavior | Zero-price routing, disabled retries/fallbacks, truncation, Groq plan/quota guards, headers, tokenizer accounting and timing |
| Evaluation/review | Checkpoint/resume identity, exact answer/source hashes, mixed reviewer provenance, capture tampering and unfilled private review packets |
| Privacy | Sanitized errors/observations and whole-graph tracing suppression |

These establish software contracts, not embedding quality, remote reasoning
quality, representative latency or vehicle safety. Scripted answer comparisons
are documented in [answer evaluation](benchmarks/grounded-answering.md); they
are separate experiments and are not required to rerun for documentation changes.

## Optional manual checks

Follow the [quick start](../README.md#quick-start) with a permitted English PDF.
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
synthetic fixtures and frozen benchmark JSON. Neither may include `.env`, PDFs,
indexes, model caches, captures or private archives. Install the built wheel in
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
