# Architecture and safety boundaries

GarageRAG uses explicit Python stages and a bounded LangGraph answering workflow.
LangChain supplies `Document`, `Embeddings` and message interfaces plus the
ChatOpenRouter integration. Retrieval does not require an LLM or agent runtime. The source is under
[`src/garage_rag`](../src/garage_rag).

## Extraction and chunking

`pypdf` extracts one document per PDF page, retaining blank pages. Metadata keeps
filename, document/content hashes, one-based PDF page position, PDF page label,
title, language, vehicle models, document type and optional source URL. Page
labels are separate from positions: citations always refer to **PDF page N**.
Extraction errors fail the document rather than silently dropping pages.

Cleaning normalizes Unicode and whitespace, removes control characters and
conservative alphabetic line-break hyphenation, and removes repeated long
margin lines. Short headings and numbers are preserved. PDF geometry, images
and tables are not reconstructed; image-only content needs human review or OCR
outside the current application.

E5's fast tokenizer provides character offsets. Chunks stay within a page,
prefer nearby sentence/paragraph boundaries, overlap by up to 48 tokens and fit
384 tokens including `passage:` and special tokens by default. Each chunk keeps
its page, offsets, ordinal and token count. Empty pages produce no chunks;
a document with no text chunks fails indexing. Inputs beyond the 512-token model
limit are rejected rather than silently truncated. Boundaries can still split
tables, procedures and cross-page explanations.

## Embeddings, persistence and retrieval

Sentence Transformers loads a pinned
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
snapshot locally with remote code disabled. English prefixes are required for
all input languages: `passage: ` for documents and `query: ` for questions.
Normalized 384-dimensional vectors are stored in Qdrant Local with cosine
similarity. Model revision, tokenizer, prefixes, chunking and relevant library
versions form the index configuration fingerprint.

SHA-256 fingerprints cover document bytes and supplied metadata. Deterministic
point IDs allow interrupted indexing to reuse persisted batches. An atomic
manifest replacement activates a revision only after every batch is written.
Partial revisions stay invisible to retrieval; a failed update leaves the
previous complete revision active. Old and partial points remain private on
disk until manually managed; there is no automatic compaction.

Retrieval uses `query_points` to search only active revisions. Scores measure
similarity, not relevance probabilities; there is no universal refusal threshold.
Qdrant Local's single-process storage suits a small personal corpus, not a
multi-user production service. CPU and MPS results can differ numerically.

## Generation and trust boundaries

1. Local language detection or an explicit override selects EN, DE or RU. The
   original question is passed unchanged to retrieval and generation.
2. Context construction assigns source IDs to at most eight retrieved chunks
   within a 12,000-character JSON budget. Passages are never truncated to fit;
   each ID maps to exactly the text actually sent.
3. Only the question and selected context cross the OpenRouter boundary. Source
   text and the question are untrusted data: the system instruction forbids
   following embedded instructions or changing the requested answer language.
   Model, market, year, equipment and body-style qualifications must be respected.
4. The model returns JSON claims with English evidence quotes and source IDs.
   Pydantic enforces a strict schema, at most six claims and evidence for each
   claim. A truncated response is rejected before validation.
5. Every claim must pass local language checks. Every source ID must exist in
   the sent context, and its quote must occur in that passage after whitespace
   normalization. Translated, altered or fabricated evidence is rejected.
6. PDF references are rendered from local metadata, not model-supplied page
   fields. Direct mode finishes here. Grounded mode applies a conservative numeric
   gate and a separate semantic audit of entailment, applicability, conflicts and
   whole-question completeness. Only insufficient context may trigger one local
   top-20 expansion and another draft/audit, within four completion attempts.
   There is no LLM repair request. See the [workflow design](grounded-answering.md).

Exact source and quote checks prevent fabricated evidence references; they do
**not** prove that a claim follows logically from its quote, that all conditions
were included, or that retrieval found the best passage. Language checks can
also reject valid short or jargon-heavy claims. Prompt-injection instructions
are treated as untrusted, but this is not a proof of immunity to model errors.
Always inspect the evidence before using an answer for a real vehicle.

## Scope

The initial BMW corpus is an owner's manual for the 2002 E39 525i, 530i and
540i, not a workshop repair manual or evidence of applicability to every E39.
Repair parameters must not be inferred from unrelated owner guidance.

Pinned alternative embedding profiles change the model, dimension and required
prefixes in the fingerprint, while retaining the same Index.search contract.
Baseline remains the default; profiles use separate directories. Phase 2B's
offline experiment adapter also implements active-only in-memory BM25,
deterministic rank fusion and optional top-20 reranking. It delegates scheduling
and page metrics to the unmodified Phase 2A evaluator. These adapters do not
change the production dense retrieval or generation path.

There is no OCR, conversation memory, autonomous agent/MCP integration,
Web UI, Docker integration or production concurrency.
Direct dependencies are pinned; there is no complete cross-platform transitive
lockfile. See [usage](usage.md) for operational settings and
[testing](testing.md) for validation scope.

## Implementation references

- [LangChain ChatOpenRouter integration](https://docs.langchain.com/oss/python/integrations/chat/openrouter)
- [Qdrant Python client and local mode](https://github.com/qdrant/qdrant-client)

## Additional LLM provider

`make_llm(provider=...)` keeps ChatOpenRouter unchanged and optionally constructs
an `httpx` Groq chat adapter with the same `invoke(messages)` result contract.
LangGraph, direct generation, prompts, schemas, local evidence validation and
retrieval remain shared. Only the Groq transport requests JSON object mode.
Fixed endpoint/model/tier, no retries/redirects/tools/fallbacks, account-plan
attestation and conservative quota reservations are described in [Groq](groq.md).
The evaluator atomically checkpoints every completed mode and Groq reservation;
resumption is bound to provider/model, dataset, selection, code and index identity.
