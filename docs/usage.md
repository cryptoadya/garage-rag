# Usage and configuration

Start with the [README](../README.md#quick-start). Run `garage-rag --help` and
`garage-rag <command> --help` for the complete CLI options.
The [2002 5 Series archive](https://www.bmwsections.com/docs/5series/) is an
alternative source for the initial manual; third-party licensing still applies.

## Environment

The CLI loads `.env` from the current working directory, preserving values
already set in the environment. It does not search parent directories.
Copy missing settings from [`.env.example`](../.env.example) into an existing
`.env` rather than replacing credentials. Never put an API key on the command
line or commit it to Git.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GARAGE_RAG_PROVIDER` | `groq` | Only supported provider for all modes; `ask --provider groq` is optional |
| `GARAGE_RAG_TRANSLATE_QUERY` | `false` | Deprecated compatibility setting; `true` and `false` both keep the original search query |
| `GROQ_API_KEY` | Empty | Required for Groq generation |
| `GROQ_FREE_PLAN_CONFIRMED` | `false` | Required local attestation of Groq Free Plan |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Initial allowlisted Groq model; `--model` overrides it |
| `GROQ_MAX_TOKENS` | `2048` | Reasoning + answer budget, 512–4096 |
| `GROQ_TOKEN_ACCOUNTING` | `bytes` | Conservative reservation; optional pinned offline `tokenizer` mode |
| `GARAGE_RAG_INDEX` | Profile-specific | Private index directory; `--index-dir` overrides it |
| `HF_HOME` | `.cache/huggingface` under the working directory | Local Hugging Face cache |

The first embedding operation downloads the pinned E5 snapshot. After the
initial download, `HF_HUB_OFFLINE=1` prevents subsequent Hugging Face downloads.
Use `--device mps` before the subcommand to try Metal on Apple Silicon; CPU is the
portable default. Indexing uses batch size 16 by default. MPS has not been
validated by this project's acceptance checks.

See [Groq Cloud](providers.md) for account-plan verification, quota behavior and
checkpoint/resume commands. Provider selection does not require re-indexing.

## Index management

Global options go **before** the subcommand. Use the same configuration for all
commands operating on an index:

```bash
garage-rag --index-dir data/qdrant-experiment --chunk-tokens 320 index path/to/manual.pdf
garage-rag --index-dir data/qdrant-experiment --chunk-tokens 320 search 'your question'
```

Changing chunking, embeddings or their fingerprinted dependencies requires a
**new index directory**. The multilingual answer settings do not require
re-embedding an existing Phase 1 index.

Each PDF filename is a document identity and must be unique within an index.
Another `index` command adds another PDF; changed content or metadata replaces
that document's active revision after processing succeeds. Removing a PDF from
disk does not remove it from the index. There is no deletion or compaction
command. Superseded and partial revisions can remain on disk, so protect the
entire directory. Only one process may open a Qdrant Local index at a time.

If indexing is interrupted, rerun the identical command. Complete batches can
be reused; search continues to use the last fully activated revision.

`index` prints JSON with timings, embedded/reused counts and extraction
diagnostics. `stats` reads persisted metadata without loading E5. Review pages
without text, unusually short pages and suspicious chunks before trusting a
new document. These heuristics do not certify extraction completeness.

## Embedding profiles

`--profile baseline` (the default) keeps E5 Small and `data/qdrant`.
`--profile e5-base` and `--profile bge-m3` use pinned alternative embeddings,
with default directories `data/qdrant-e5-base` and `data/qdrant-bge-m3` when
`GARAGE_RAG_INDEX` is unset. An existing `.env` value still takes precedence
over that directory default; use explicit `--index-dir` for experiments.
Profiles never convert an existing index. A wrong profile fails as incompatible.
Use the same profile for `index`, `stats`, `search` and `ask`.

See the [Phase 2B comparison and complete commands](benchmarks/phase2b-retrieval.md).
Every profile retains raw dense cosine search and the original CLI limits.
Do not transfer score thresholds between models without separate calibration.
Direct/Grounded `ask` uses the original question for multilingual dense retrieval,
writing and auditing. EN/DE/RU documents and mixed-language passages need no
language classification or reindexing. Missing, unknown or outdated document
language metadata does not determine the query language. This does not guarantee
retrieval quality; inspect passages for short or ambiguous queries.
`GARAGE_RAG_TRANSLATE_QUERY=true|false` is accepted for existing configurations,
but neither value enables a translation call.
CLI `search` remains dense. BM25 is also available to opt-in agentic retrieval;
fusion and reranking remain offline comparison adapters.

## Search and answers

```bash
garage-rag search 'Wie prüfe ich den Motorölstand?' --top-k 5 --json
garage-rag ask 'How do I check the engine oil level?' --answer-language de --json
```

`search` returns passages, local metadata and raw cosine scores. `--top-k` accepts
1–20; `--min-score` is optional. Inspect results before tuning it: a similarity
score is not an answer probability, and absence from top-k does not establish
absence from the document. Questions must contain 1–4000 characters and fit
E5's 512-token input limit, including its query prefix and special tokens.

`ask --json` retains `supported`, `text`, `reason` and the resolved `language`,
and adds privacy-safe `workflow` counts, limits, timings, token usage and node events.
`index --language en` describes the source document; `ask --answer-language`
controls the answer. CLI diagnostics and the `PDF page N` citation label stay
English. Quotes stay in their original source language regardless of the answer language.

Automatic language selection uses local
[Lingua confidence scores](https://github.com/pemistahl/lingua-py#113-confidence-values) after removing
shared automotive identifiers, units and alphanumeric model names. It requires
a score of at least 0.70 and a margin of 0.25 over the runner-up; otherwise it
falls back to English. These are heuristics relative to EN/DE/RU candidates,
not calibrated probabilities. `Reifendruck?` selects German, while `BMW DDE ABS`
and ambiguous `Motor?` fall back to English. Mixed-language input or unfamiliar
jargon can be misclassified; choose `--answer-language en|de|ru` explicitly.
Language detection selects only the answer language and makes no LLM call.

## Generation failures

Default `ask` uses bounded grounded answering: up to four completion attempts
(including semantic audits) and two local searches. A normal accepted answer
uses two completions. `--max-llm-calls 2 --max-searches 1` disables recovery;
`--workflow direct` selects the one-answer-completion path. These budgets apply
to all question and source languages, with no separate translation allowance.
Each completion
has a 60-second network-operation timeout, not a total deadline. See
[architecture](architecture.md) for routes, budgets and evidence boundaries.
Groq is the only provider, using the allowlisted `openai/gpt-oss-120b` model.
Free Plan attestation is required; quota guards stop without retries or fallback.
See [billing and quota boundaries](providers.md).

| Symptom or reason | Action |
| --- | --- |
| Authentication error | Check the local API key without sharing or printing it |
| HTTP 429 / rate-limit error | Wait and resume manually after checking quota reset |
| Model unavailable | Check Groq availability; no alternate model or paid route is attempted |
| `output_token_limit` | Increase `GROQ_MAX_TOKENS` within 512–4096, subject to quota guards |
| `invalid_model_output` | Output failed format, language or evidence checks; inspect local retrieval before retrying |
| `invalid_audit_output` / `unsupported_claims` | Inspect evidence and applicability; the audit cannot certify truth |
| `ambiguous_question` / `conflicting_sources` | Clarify the vehicle/conditions or resolve conflicting documents |
| `unsafe_numeric_evidence` | Inspect the original PDF layout; do not infer table cells from extracted text |
| `llm_budget_exhausted` / `no_new_context` | No unaudited answer is returned; inspect retrieval before changing budgets |
| `context_budget_exhausted` | Agentic cannot fit all protected evidence; inspect private hits and selection diagnostics |
| `context_relevance_uncertain` | Overflow has no usable relevance anchors; clarify the question or inspect the original sources |
| `no_context` / `insufficient_evidence` | Rephrase the question, inspect search results or supply relevant documentation |

Reasoning tokens share the completion budget with the visible JSON answer.
Increasing the budget can prevent truncation but does not extend the timeout
or relax validation. `finish_reason=length` is rejected even if the response
looks like complete JSON. Responses are also limited to 20,000 characters;
there is no automatic repair request.

Opt-in `--workflow agentic` uses model-selected local dense/BM25 tools, with
defaults of six LLM attempts including writer/audit, four searches, 24,000 reserved
tokens and twelve graph steps. `--max-total-tokens` changes its reservation cap
(1–48,000); `--top-k` caps each tool at 1–5. Direct/grounded retain their existing
answer budgets and ignore this agentic token flag.
With Groq, the per-call output budget
is `GROQ_MAX_TOKENS`. See [agentic architecture](architecture.md#agentic-retrieval-phase-4a)
for stopping rules and [live acceptance](providers.md#phase-4a-live-acceptance)
for what has been confirmed and what remains unproven.

## HTTP API

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
Unknown fields are rejected. The response retains CLI `ask --json` fields: `supported`,
`text` (including verified PDF references), `reason`, `language` and public
`workflow` observations. HTTP also adds `claims`: an ordered array of
`{text, citations: [{source_id, document, pdf_page, quote, pdf_sha256}]}` from the final
supported candidate. `pdf_page` is the physical PDF page counted from 1, not
the printed page label; `quote` preserves the verified quotation verbatim.
`pdf_sha256` identifies the source PDF content for the viewer and is `null` when
the source metadata has no content hash.
Refusals always return `claims: []`, including after an audit rejects a draft.
CLI output is unchanged. Private context, drafts and raw provider errors are omitted.
Retrieval uses the CLI defaults (top-k 5, no score threshold) and the existing
workflow budgets. Valid refusals return HTTP 200 with `supported: false`.
OpenAPI schemas and interactive API documentation are available at `/docs`.

The server reads only the working directory's `.env`, without overriding existing
environment variables. `GARAGE_RAG_INDEX` selects the index (default
`data/qdrant`); `GARAGE_RAG_PROFILE` selects `baseline`, `e5-base` or `bge-m3`
(default `baseline`, with the same profile-specific directories as the CLI).
`GARAGE_RAG_DEVICE` selects `cpu` (default) or `mps`. Provider/model settings are
the same as the CLI; credentials are checked at startup without a generation call.
An absent index is initialized empty for the first UI upload. An incompatible
or already locked index prevents startup.

One process owns the index and provider client until shutdown and reuses one lazy
embedding adapter. Run **one worker**, without reload, and stop the server before
using the CLI against the same index. Indexing and `/ask` share a non-queuing
lock; overlap returns 503 with `Retry-After: 1`. Health and the active-document
list remain responsive. Validation errors return
422, retrieval failures 503, generation failures 502 and unexpected local failures
500, with fixed messages that omit inputs, paths and provider payloads.
There is no authentication; the launch command binds to loopback for local use.

## Local web UI

From a repository checkout, install `.[dev]` in the Python environment and
configure provider credentials as in the quick start. You can start with an
empty index and upload your first PDF through the UI. Use two terminals, both
starting at the repository root:

```bash
# Terminal 1: activate the Python environment, then start the backend.
source .venv/bin/activate
python -m uvicorn garage_rag.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

```bash
# Terminal 2: Node.js 22.12+ (22.x), 24.x or 26+ and npm.
cd frontend
npm ci
npm run dev
```

Open [the local UI](http://127.0.0.1:5173). In **Your PDFs**, choose a PDF and
select **Upload and index PDF**. Wait for **Ready for questions** and the document
to appear in the indexed list. Upload progress covers file transfer; indexing
runs within the same request, without a background job or automatic retry.
The first upload can also load the local tokenizer and embedding weights.
Then enter a question, choose Direct or
Grounded and an answer language (Auto, English, Deutsch or Русский), then submit.
The UI shows loading, supported claim text once with compact PDF-page links,
refusals and request errors. Expand **Sources** for full document names and exact
quotes, grouped by document revision and physical page without repeating claims.
Each claim links to its evidence pages; multiple quotes from one page share a link.
Sources use structured response data; the UI does not parse references from text.
Select a citation page to view the saved original PDF.
Questions are limited to 4,000 characters. A new submission replaces the previous
result; nothing is saved as chat history. Only explicit submissions call `/ask`,
with no automatic retries.

### PDF upload and storage

`GET /documents` returns only active indexed entries:
`{"documents":[{"document":"manual.pdf","pages":12,"chunks":24}]}`.
It includes documents indexed by the CLI; failed and partially indexed revisions
are not listed. **Refresh documents** reloads this list, including after a lost
upload connection where the server may still have completed indexing.

`POST /documents?filename=manual.pdf` accepts a raw `application/pdf` body, up
to 50 MiB, and returns `{"status":"indexed","document":{...}}` after indexing.
Reuploading the same file under the same filename through this API reuses the
existing ingestion path and returns
`"status":"unchanged"` without re-embedding. Changed content with the same
filename activates a new revision using the existing index behavior; other
documents stay indexed. Questions search all active documents as before.

```bash
curl http://127.0.0.1:8000/documents
curl 'http://127.0.0.1:8000/documents?filename=manual.pdf' \
  -H 'Content-Type: application/pdf' --data-binary @/path/to/manual.pdf
```

Accepted PDFs are retained under `~/.garage-rag/pdfs/<index-path-hash>/`, outside
the repository and web directories, in unique folders with directory/file
permissions 0700/0600. The original basename is used for document identity and
citations; paths and control characters in filenames are rejected. Every accepted
upload is retained, including unchanged uploads and older revisions. Temporary
files from failed requests are removed. The viewer endpoint serves only originals
matching an active document and citation hash; no local paths are returned.
PDFs are never committed to Git.

Wrong media types return 415, unsafe filenames 400, oversized uploads 413,
unreadable/damaged/encrypted PDFs or PDFs without text 422, and unexpected local
storage/embedding failures 500 with sanitized messages. Scans need an existing
text layer; this MVP performs no OCR. No document deletion, authentication,
background queue or Agentic changes are included.

The browser posts to `/api/ask`; the local Vite proxy forwards to the existing
`POST http://127.0.0.1:8000/ask`. Backend behavior and provider settings are unchanged.
Keep credentials in the backend's `.env`; the frontend needs no environment file
or API key. Both servers bind to loopback. Stop other CLI/server processes using
the same index before starting FastAPI.

The API preserves document names, physical PDF page numbers and exact quotes in
answer text and structured citations. The viewer uses the structured citations;
the frontend does not verify evidence or run RAG logic.

For a local build check, run `npm run build` and `npm run preview` from `frontend`,
then open [the preview](http://127.0.0.1:4173) with the backend still running.
Preview uses the same local API proxy. This MVP uses Vite's local servers;
the `dist` files alone do not provide an API proxy.

### Original PDF viewer

Select **PDF page** in a structured citation to open the saved original in a
local dialog. Previous/next buttons and a physical page input navigate from page 1;
the browser's PDF preview also supports zoom and search. Use **Open PDF in new tab**
if your browser does not embed PDFs. Close or press Escape to return to the answer.
The local PDF is never sent to Groq by the viewer.

`GET /documents/pdf?document=...&sha256=...` serves only a byte-verified PDF from
the API's private upload store whose name and SHA-256 match an active document.
It returns PDF bytes with `X-PDF-Pages`, `Cache-Control: no-store` and no local paths.
Structured citations expose `pdf_sha256` from existing source metadata; missing
hashes, missing originals and replaced revisions show an unavailable message.
There is no arbitrary file path or remote-URL parameter, and symlinks are rejected.
Earlier CLI imports may have no retained original. Uploading the identical file
with the same filename makes its original available without re-embedding unchanged
content. Custom CLI metadata still follows the existing ingestion revision rules.
Existing API uploads work without migration. Local files outside the private upload
store and superseded revisions are never served. This remains a loopback-only,
single-user API without authentication; do not expose it to a network.

## Privacy

`ask` transmits the question, selected passages and audit draft to the selected
LLM provider (Groq).
Review their retention policies before using sensitive documents. `search`
prints private text to the terminal; exported JSON and generated answers can
also contain copyrighted source text. Do not commit these captures.

The application disables inherited LangSmith tracing. Workflow observations
are local and omit private content; no remote tracing backend is configured. Hugging Face model downloads still
use the network. Keep PDFs, caches, extracted text and indexes private, including
custom paths outside the ignored directories. See the
[licensing notice](../README.md#license) before adding documents.
