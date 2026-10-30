# Development and verification

## Offline checks

Install development dependencies once in the activated Python environment:

```bash
python -m pip install -e '.[dev]'
```

Run from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= LANGSMITH_TRACING=false python -m pytest
ruff check .
python -m pip check
python -m build
```

[CI](https://github.com/cryptoadya/garage-rag/actions/workflows/ci.yml) runs Python
checks on Python 3.12/Linux and frontend checks on Node.js 22. Local development
also uses macOS arm64 CPU; MPS is not validated. Offline tests use deterministic
vectors, local tokenizers, synthetic PDFs and controlled HTTP/model responses.
Qdrant persistence and LangGraph execution are real. No provider call is needed.

Some optional replays use ignored private historical captures. If those exist,
the pinned Groq tokenizer must already be cached; use
`HF_HOME="$PWD/.cache/huggingface"` for the repository's cache. A fresh clone skips
private-capture replays. Do not download weights or invoke providers to run them.

| Area | Contracts |
| --- | --- |
| Ingestion/index | Page provenance, token/overlap bounds, revisions, interrupted batches and profile isolation |
| Retrieval | Metric arithmetic, frozen datasets, active-only search, multilingual batching and bounded reranking |
| Answers | Language, quotes, schema, numeric gates, audit coverage, recovery, budgets and safe refusals |
| Agentic | Local search protocol, stable evidence IDs, whole-passage selection, planner and terminal budgets |
| Provider | Fixed endpoint/model, quota reservations, pacing, no HTTP retry/fallback, timing and truncation |
| API/UI | Request validation, uploads/listing, exclusive access, structured claims and hash-bound PDF viewing |
| Review/privacy | Capture/hash integrity, review provenance, sanitized errors and tracing suppression |

These checks establish software contracts, not live answer quality or vehicle
safety. Measured experiments are documented in the [evaluation guide](../benchmarks/README.md).

## Frontend

Use Node.js 22.12+ (22.x), 24.x or 26+:

```bash
cd frontend
npm ci
npm test
npm run build
```

Vitest/Testing Library use controlled responses for questions, languages,
loading/refusal/error states, uploads, document lists and PDF-viewer navigation.
The build includes TypeScript checking. For browser changes, also check desktop
and mobile layouts, keyboard interaction, citation navigation and error recovery
with controlled responses. A real answer submission calls Groq and needs an
explicitly agreed live scope; it is not an offline build check.

## Packages and private data

`python -m build` writes ignored wheel/sdist files under `dist/`. Inspect both:

- Wheel: runtime package, entry point and MIT license.
- Sdist: public source, tests/synthetic fixtures, guides, benchmark inputs/results/scripts and frontend source/config/lockfile.
- Neither: `.env`, private PDFs, indexes, model caches, captures, archives, `node_modules` or frontend build output.

The [sdist allowlist](../pyproject.toml) defines package boundaries. Frozen JSON
and reproduction scripts must survive source packaging; UI builds separately and
is not served by the wheel. Install the wheel in an isolated environment and
check `garage-rag --help` and `python -m garage_rag --help` when changing packaging.
Verify local Markdown links/anchors after reorganizing guides. Inspect the final
diff/status and package members; ignored files already tracked remain public.

## Versioning

Keep user-facing changes under `Unreleased`. At release preparation, move those
notes to one dated version, leave an empty `Unreleased`, update comparison links
and synchronize Python/frontend versions. Before publication, verify intended
`main`, tests, lint, builds, CI, existing tags and releases. Package version,
`v`-prefixed tag and release must match; use the CHANGELOG section as release notes.
Merge, tag and GitHub Release require separate authorization; preparation does
not publish a release. See [contributor instructions](../AGENTS.md).
