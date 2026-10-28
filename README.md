# GarageRAG

[![CI](https://github.com/cryptoadya/garage-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/cryptoadya/garage-rag/actions/workflows/ci.yml)

A local-first RAG CLI and HTTP API for automotive documentation. Ask in **English, German or
Russian**, search an English PDF locally, and generate answers with verbatim
evidence and locally verified PDF page references.

GarageRAG demonstrates document provenance, resumable indexing, multilingual
retrieval, bounded LangGraph orchestration and reproducible evaluation. It is an
alpha engineering project; general answer-quality improvement from semantic
auditing remains unproven. It does not replace professional diagnosis.

## Features

- Page-aware PDF extraction and overlapping, tokenizer-bounded chunks.
- Persistent Qdrant Local indexes with atomic revision activation and batch reuse.
- Pinned E5 Small, E5 Base and BGE-M3 embedding profiles with isolated indexes.
- Direct generation or bounded draft/audit answering with one optional local expansion.
- A minimal FastAPI server for Direct/Grounded answers and public workflow metadata.
- Experimental, opt-in agentic retrieval: model-selected dense/BM25 searches
  and multilingual query rewrites.
- OpenRouter free-only routing and an optional Groq transport with Free Plan attestation.
- Offline synthetic tests, frozen retrieval baselines and hash-bound answer/source review.

**Stack:** Python 3.12+, Sentence Transformers, Qdrant Local, pypdf,
LangGraph, LangChain Core / ChatOpenRouter, Pydantic and Lingua.
Indexing and search need no LLM account or Qdrant server.

## Quick start

Developed on macOS Apple Silicon; CI runs Python 3.12 on Linux. CPU is the default.

```bash
git clone https://github.com/cryptoadya/garage-rag.git
cd garage-rag
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
test -f .env || cp .env.example .env
mkdir -p data/private
```

Supply an English text-based PDF you have permission to use. The initial corpus
is the [2002 BMW 5 Series owner's manual](https://www.ge39.com/files/2002Manual-5ser.pdf)
for E39 525i/530i/540i models. Save your own copy at the path below;
**the manual is not distributed here**.

```bash
garage-rag index data/private/bmw_e39_2002_manual.pdf \
  --title 'BMW E39 2002 Owner Manual' --language en \
  --models 'BMW 525i,BMW 530i,BMW 540i' --kind owner_manual \
  --source-url https://www.ge39.com/files/2002Manual-5ser.pdf

garage-rag stats
garage-rag search 'How do I check the engine oil level?' --top-k 5
```

The first embedding operation downloads pinned model weights. After caching,
set `HF_HUB_OFFLINE=1` for offline retrieval. Repeating an unchanged index command
reuses its completed revision.

Set `OPENROUTER_API_KEY` in your local `.env` to generate answers:

```bash
garage-rag ask 'How do I check the engine oil level?'
garage-rag ask 'Wie prüfe ich den Motorölstand?'
garage-rag ask 'Как проверить уровень моторного масла?'
garage-rag ask 'How do I check the engine oil level?' --answer-language de
# Original one-completion baseline:
garage-rag ask 'How do I check the engine oil level?' --workflow direct
```

Default `grounded` mode uses a draft and a same-model audit, with at most four
completion attempts and two local searches. Both modes validate language,
source IDs and evidence quotes locally; valid citations do not prove correct
interpretation. Questions, selected passages and audit drafts go to the chosen
provider. Indexing and retrieval stay local.

OpenRouter accepts only `openrouter/free` or an explicit `provider/model:free`,
with zero price caps and no retries or paid fallback. For Groq, verify your
organization's Free Plan, set `GROQ_API_KEY` and `GROQ_FREE_PLAN_CONFIRMED=true`,
then use `ask --provider groq`. Groq has no per-request zero-price guarantee.
See [provider setup and quotas](docs/providers.md).

### HTTP API

Install the package and prepare an index and provider credentials as above, then run:

```bash
python -m uvicorn garage_rag.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Как проверить уровень моторного масла?","mode":"grounded","language":"ru"}'
```

`GET /health` returns `{"status":"ok"}` after local startup succeeds. It does not
check provider availability or load the embedding weights. `POST /ask` accepts a
nonblank question of at most 4,000 characters, `mode` (`direct` or `grounded`,
default `grounded`) and `language` (`auto`, `en`, `de`, `ru`, default `auto`).
Unknown fields are rejected. The response matches CLI `ask --json`: `supported`,
`text` (including verified PDF references), `reason`, `language` and public
`workflow` observations. Private context, drafts and raw provider errors are omitted.
Retrieval uses the CLI defaults (top-k 5, no score threshold) and the existing
workflow budgets. Valid refusals return HTTP 200 with `supported: false`.
OpenAPI schemas and interactive API documentation are available at `/docs`.

The server reads only the working directory's `.env`, without overriding existing
environment variables. `GARAGE_RAG_INDEX` selects an existing index (default
`data/qdrant`); `GARAGE_RAG_PROFILE` selects `baseline`, `e5-base` or `bge-m3`
(default `baseline`, with the same profile-specific directories as the CLI).
`GARAGE_RAG_DEVICE` selects `cpu` (default) or `mps`. Provider/model settings are
the same as the CLI; credentials are checked at startup without a generation call.
An incompatible, missing or already locked index prevents startup.

One process owns the index and provider client until shutdown and reuses one lazy
embedding adapter. Run **one worker**, without reload, and stop the server before
using the CLI against the same index. One `/ask` runs at a time; overlap returns
503 with `Retry-After: 1`, while health remains responsive. Validation errors return
422, retrieval failures 503, generation failures 502 and unexpected local failures
500, with fixed messages that omit inputs, paths and provider payloads.
There is no authentication; the launch command binds to loopback for local use.

### Agentic retrieval

```bash
# Verify Groq Free Plan and configure its key/attestation first.
garage-rag ask 'Как проверить давление в шинах?' --workflow agentic
# Lower the maximum API exposure explicitly:
GROQ_MAX_TOKENS=512 garage-rag ask 'How do I check tire pressure?' \
  --workflow agentic --max-llm-calls 3 --max-searches 1 --max-total-tokens 16000
```

Agentic defaults to Groq `openai/gpt-oss-120b`, independently of the legacy
`GARAGE_RAG_PROVIDER` setting. It chooses local dense or BM25 tools, rewrites
queries using retrieved evidence, compares accumulated passages and decides when
to stop. Accepted answers still require quote/language/numeric validation and a
separate semantic audit. Defaults: **6 LLM attempts including audit, 4 local
searches, 24,000 reserved input+output/reasoning tokens, 12 graph steps**.
A one-search cap needs three calls (search decision, tools-free JSON writer,
audit); an early stop with the default search budget needs four including its
control turn. Budgets/quotas can stop earlier. `--top-k` caps each search at 1–5 (default 5).

Only predefined local read-only tools are exposed. No web/code/file-writing tool,
API retry, paid fallback or automatic provider switch exists. OpenRouter remains
an explicitly selected compatibility alternative (`--provider openrouter`),
with its existing free-only controls. Direct/Grounded remain the primary modes.
See [current architecture](docs/architecture.md#agentic-retrieval-phase-4a) and the
[comparison results and limits](docs/benchmarks/phase4b-comparison.md).

## Limitations and privacy

- No OCR, table/layout reconstruction or figure understanding. Check numerical
  tables and vehicle/model/market/equipment applicability in the original PDF.
- Retrieval can miss evidence. Similarity scores are not confidence; numeric
  heuristics and the same-model auditor can reject valid claims or share errors.
- A same-model auditor is not independent ground truth. Quote existence and audit
  acceptance do not guarantee that selected quotes cover every part of a compound
  claim or preserve all warnings from its full source passage.
- Free services can be unavailable or rate limited. There is no hard total
  workflow deadline, independent human quality acceptance or vehicle validation.
- Qdrant Local allows one process per index. No Web UI, conversation
  memory, unrestricted agent, MCP or Docker integration is implemented.
- Keep `.env`, PDFs, indexes, caches and captures private, including custom paths.
  Indexes contain source text; `.gitignore` cannot protect already tracked files.

## Development and documentation

```bash
python -m pip install -e '.[dev]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= GROQ_API_KEY= python -m pytest
ruff check .
python -m build
```

| Guide | Contents |
| --- | --- |
| [Usage](docs/usage.md) | CLI, configuration, profiles and troubleshooting |
| [Architecture](docs/architecture.md) | Ingestion, persistence, workflows and trust boundaries |
| [Providers](docs/providers.md) | OpenRouter/Groq setup, billing boundaries and resumable runs |
| [Testing](docs/testing.md) | Offline checks, packaging and release validation |
| [Phase 2A](docs/benchmarks/phase2a-retrieval.md) / [Phase 2B](docs/benchmarks/phase2b-retrieval.md) | Frozen retrieval methodology, results and reproduction |
| [Answer evaluation](docs/benchmarks/grounded-answering.md) | Scripted/live comparisons, costs and unresolved quality limits |
| [Three-mode comparison](docs/benchmarks/phase4b-comparison.md) | Consolidated offline/live results, evidence review and limitations |
| [Changelog](CHANGELOG.md) | Versioned user-facing changes |

## License

[MIT](LICENSE) covers project code and original synthetic materials. BMW manuals
are copyrighted by BMW AG and are not redistributed. Source documents and model
weights retain their own licenses.
