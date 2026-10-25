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
| `OPENROUTER_API_KEY` | Empty | Required only for `ask` |
| `OPENROUTER_MODEL` | `openrouter/free` | Free router or explicit `provider/model:free`; `ask --model` overrides it |
| `OPENROUTER_MAX_TOKENS` | `8192` | Combined reasoning and answer budget; integer from 512 to 32768 |
| `GARAGE_RAG_INDEX` | Profile-specific | Private index directory; `--index-dir` overrides it |
| `HF_HOME` | `.cache/huggingface` under the working directory | Local Hugging Face cache |

The first embedding operation downloads the pinned E5 snapshot. After the
initial download, `HF_HUB_OFFLINE=1` prevents subsequent Hugging Face downloads.
Use `--device mps` before the command to try Metal on Apple Silicon; CPU is the
portable default. Indexing uses batch size 16 by default. MPS has not been
validated by this project's acceptance checks.

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
BM25, fusion and reranking are experiment-only adapters, not CLI defaults.

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
English. Quotes stay in their original English even in German or Russian answers.

Automatic language selection uses local
[Lingua confidence scores](https://github.com/pemistahl/lingua-py#113-confidence-values) after removing
shared automotive identifiers, units and alphanumeric model names. It requires
a score of at least 0.70 and a margin of 0.25 over the runner-up; otherwise it
falls back to English. These are heuristics relative to EN/DE/RU candidates,
not calibrated probabilities. `Reifendruck?` selects German, while `BMW DDE ABS`
and ambiguous `Motor?` fall back to English. Mixed-language input or unfamiliar
jargon can be misclassified; choose `--answer-language en|de|ru` explicitly.
No translation or language-detection LLM call is made.

## Generation failures

Default `ask` uses bounded grounded answering: up to four completion attempts
(including semantic audits) and two local searches. A normal accepted answer
uses two completions. `--max-llm-calls 2 --max-searches 1` disables recovery;
`--workflow direct` selects the original one-completion path. Each completion
has a 60-second network-operation timeout, not a total wall-clock deadline. See [workflow design](grounded-answering.md) for
all routes, budgets, numerical restrictions and observation fields. SDK retries and provider fallback are disabled; prompt, completion and
request [price caps](https://openrouter.ai/docs/guides/routing/provider-selection#max-price)
are zero. An explicit free model can be selected with
`--model provider/model:free`; availability still depends on the provider.

| Symptom or reason | Action |
| --- | --- |
| Authentication error | Check the local API key without sharing or printing it |
| HTTP 429 / rate-limit error | Wait and retry manually or choose another explicit free model |
| Free model unavailable | Select another `:free` model; no paid route is attempted |
| `output_token_limit` | Increase `OPENROUTER_MAX_TOKENS` within its bounds or select another free model |
| `invalid_model_output` | Output failed format, language or evidence checks; inspect local retrieval before retrying |
| `invalid_audit_output` / `unsupported_claims` | Inspect evidence and applicability; the audit cannot certify truth |
| `ambiguous_question` / `conflicting_sources` | Clarify the vehicle/conditions or resolve conflicting documents |
| `unsafe_numeric_evidence` | Inspect the original PDF layout; do not infer table cells from extracted text |
| `llm_budget_exhausted` / `no_new_context` | No unaudited answer is returned; inspect retrieval before changing budgets |
| `no_context` / `insufficient_evidence` | Rephrase the question, inspect search results or supply relevant documentation |

[Reasoning tokens share the completion budget](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-tokens-and-max_tokens)
with the visible JSON answer.
Increasing the budget can prevent truncation but does not extend the timeout
or relax validation. `finish_reason=length` is rejected even if the response
looks like complete JSON. Responses are also limited to 20,000 characters;
there is no automatic repair request.

## Privacy

`ask` transmits the question and selected passages to OpenRouter/providers.
Review their retention policies before using sensitive documents. `search`
prints private text to the terminal; exported JSON and generated answers can
also contain copyrighted source text. Do not commit these captures.

The application disables inherited LangSmith tracing and does not configure
content logs, index exports or telemetry. Hugging Face model downloads still
use the network. Keep PDFs, caches, extracted text and indexes private, including
custom paths outside the ignored directories. See the
[licensing notice](../README.md#limits-and-licensing) before adding documents.
