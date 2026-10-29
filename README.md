# GarageRAG

[![CI](https://github.com/cryptoadya/garage-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/cryptoadya/garage-rag/actions/workflows/ci.yml)

A local-first RAG CLI, HTTP API and web UI for automotive documentation. Ask in
**English, German or Russian**, search EN/DE/RU or mixed-language PDFs locally,
and generate answers with verbatim evidence and locally verified PDF page references.

## Features

- Page-aware PDF extraction, resumable indexing and persistent Qdrant Local storage.
- Pinned multilingual E5 Small (default), E5 Base and BGE-M3 embedding profiles.
- Direct answers or bounded Grounded draft/audit answering, with local citation checks.
- FastAPI and a local React UI for Direct/Grounded questions, languages and sources.
- Experimental CLI-only Agentic retrieval with bounded local dense/BM25 searches.
- Groq-only generation with Free Plan attestation and conservative quota guards.
- Open a saved original PDF directly from a citation, with page navigation.

Indexing and search need no LLM account or Qdrant server. Generation sends the
question, selected passages and audit drafts to Groq.

## Quick start

Requires Python 3.12+. Developed on macOS Apple Silicon; CI uses Linux. CPU is the default.

```bash
git clone https://github.com/cryptoadya/garage-rag.git
cd garage-rag
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
test -f .env || cp .env.example .env
mkdir -p data/private
```

Supply an EN/DE/RU or mixed-language text-based PDF you have permission to use.
The initial corpus is the [2002 BMW 5 Series owner's manual](https://www.ge39.com/files/2002Manual-5ser.pdf)
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

### CLI answers

Set `GROQ_API_KEY` in your local `.env`. Verify your organization is on Free Plan
and set `GROQ_FREE_PLAN_CONFIRMED=true` before generating answers:

```bash
garage-rag ask 'How do I check the engine oil level?'
garage-rag ask 'Wie prüfe ich den Motorölstand?'
garage-rag ask 'Как проверить уровень моторного масла?' --answer-language ru
garage-rag ask 'How do I check the engine oil level?' --workflow direct
```

Grounded is the default, with at most four completion attempts and two local
search batches. Direct uses one answer completion. Both retain the original
question. If local source text indicates another language, one Groq call adds
translations into those corpus languages (at most two); same-language questions
need none. Each retrieval round merges at most three dense queries in one local
batch. Expanded mixed-language results also use the existing pinned local MiniLM
reranker over at most 20 candidates; its weights load on the first mixed answer.
This adds local CPU/memory use and may download cached weights, without another
Groq call. Claims use the requested answer language; quotes keep the source language.
Existing indexes need no reindexing. For Groq setup, quotas and experimental
Agentic usage, see [providers](docs/providers.md) and [CLI usage](docs/usage.md).

### HTTP API

With the index and credentials prepared, start one backend process:

```bash
python -m uvicorn garage_rag.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Как проверить уровень моторного масла?","mode":"grounded","language":"ru"}'
```

OpenAPI is available at `/docs`. Run one worker without reload; stop the server
before using the CLI against the same index. See [API configuration and responses](docs/usage.md#http-api).

### Local web UI

Keep the backend running. In a second terminal at the repository root, use
Node.js 22.12+ (22.x), 24.x or 26+ and npm:

```bash
cd frontend
npm ci
npm run dev
```

Open [the local UI](http://127.0.0.1:5173). Choose Direct/Grounded and an answer
language, then submit a question. Keep provider keys in the backend's `.env`.
See [UI behavior and preview builds](docs/usage.md#local-web-ui).

## Limitations and privacy

This is an alpha engineering project. Check vehicle/model/market applicability
and numerical tables in the original PDF; it does not replace professional diagnosis.

- No OCR, table/layout reconstruction or figure understanding.
- Retrieval can miss evidence. Valid quotes and same-model audit acceptance do
  not establish completeness or correctness; general quality improvement is unproven.
- Free providers can be unavailable or rate limited. There is no automatic retry,
  paid fallback, hard total workflow deadline or independent human/vehicle validation.
- CLI and API require exclusive index access. API/UI are local, without authentication;
  conversation history and document deletion are not implemented. PDF upload,
  active-document listing and viewing saved originals are available in the UI.
- Keep `.env`, PDFs, indexes, caches and captures private, including custom paths.
  Indexes contain source text; `.gitignore` cannot protect already tracked files.

## Development and documentation

| Guide | Contents |
| --- | --- |
| [Usage](docs/usage.md) | CLI, API, UI, profiles and troubleshooting |
| [Architecture](docs/architecture.md) | Indexing, workflows and trust boundaries |
| [Providers](docs/providers.md) | Credentials, billing boundaries and quotas |
| [Testing](docs/testing.md) | Python/frontend checks, packaging and release validation |
| [Benchmark inventory](docs/benchmarks/README.md) | Frozen datasets, published results, reproduction and archive policy |
| [Changelog](CHANGELOG.md) | Versioned user-facing changes |

## License

[MIT](LICENSE) covers project code and original synthetic materials. BMW manuals
are copyrighted by BMW AG and are not redistributed. Source documents and model
weights retain their own licenses.
