# Offline retrieval evaluation

Phase 2A measures the existing multilingual E5 search against a fixed owner's
manual. It does not change retrieval, extraction, chunking, embeddings or
answer generation. The [baseline report](phase2a-report.md) interprets the
measurements; the [public dataset](../../benchmarks/bmw_e39_2002.v1.json)
contains original questions and page references, never BMW passages.

## Corpus and reproduction

Use your own permitted copy of the 186-page US English manual linked in the
[README](../../README.md#quick-start). The benchmark accepts only SHA-256
`9faf9c33f9958f037c77032e514295378ddb490217c1d37015d6ca0fb48b4574`.
A differently scanned, revised or repaginated manual requires a new reviewed
dataset, not bypassing the hash check.

Install the project and create/cache the baseline index using the README's
unchanged index command. This preparation can require network access for the
pinned E5 weights; the evaluation itself requires the complete local cache.
Do not run another Qdrant Local process against the same directory.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= \
python -m garage_rag.evaluation \
  --dataset benchmarks/bmw_e39_2002.v1.json \
  --pdf data/private/bmw_e39_2002_manual.pdf \
  --index-dir data/qdrant \
  --output output/evaluation/e39-baseline.json \
  --repeats 3 --seed 20261025 --device cpu
```

The output must be a new file. A missing, incompatible, mixed-corpus or
incomplete active index fails before model inference. PDF identity and page
count are checked independently of the index manifest. Evaluation never
calls `ingest` or `embed_documents`; it checks that the manifest is unchanged.
The CLI forces Hugging Face offline mode, disables tracing and telemetry,
does not load `.env`, and does not initialize OpenRouter. No new runtime
dependency or model is required. `python -m garage_rag.evaluation --help`
lists the options. CPU is the measured baseline; MPS is available but untested.

The JSON includes all trials' ordered page/chunk IDs and scores, elapsed
milliseconds, shuffled query order, corpus and dataset hashes, active index
generation, model revision, index configuration/fingerprint, relevant package
versions, platform, Torch thread counts and hashes of the evaluation and
pipeline source files. It deliberately omits passages, payloads, vectors,
local paths, machine names, credentials and environment dumps. The CLI's stdout
is the language summary, not retrieved text. Inspect artifacts before publishing,
especially when replacing the authored questions with your own data.

## Dataset and review provenance

There are 18 answerable information needs with equivalent EN/DE/RU questions
(54 queries), four separate unanswerable needs (12 queries), and two unscored
tire-pressure probes (six queries). Topics cover controls, security, warnings,
driving, maintenance, equipment and roadside procedures. They include a
numeric table, location questions, instructions, a symptom and a minimum-speed
condition. This is a small, deliberately stratified convenience sample, not
an estimate of real user traffic. It is not uniform over all 186 pages.

Questions and gold pages were frozen before the first Phase 2A ranking run. Labels
were established from the original PDF using independent Poppler extraction
and visual inspection of rendered pages, rather than from retrieved chunks.
For the cabin-filter symptom both independently supporting pages are accepted.
Contents pages, index entries, topic mentions and cross-references without the
requested information are not relevant. Equipment, market and body-style
qualifications matter: the adjacent 540i pressure table does not satisfy a
530i question.

**No human reviewed these labels or translations.** `agent_verified_pdf` means
Codex inspected the original page text and image; it is not a human-verification
claim or automatically inferred relevance. Every case records the method,
date, scope and `human_verified: false`. `human_verified_pdf` is reserved for
future actual human review. `unverified` labels are accepted by the format but
excluded from accuracy, with their count reported. Changing labels/questions
requires a new dataset version; keep v1 stable for future comparisons.

The negative cases cover a repair-only engine fastener specification, ECU
coding instructions, an absent infotainment technology and an out-of-corpus
diesel procedure. `agent_scope_checked` records whole-PDF lexical checks and
inspection of the closest source sections. These are provisional absence
judgments, not exhaustive proof that every conceivable answer is absent.
They are never folded into answerable Hit/MRR denominators. No unverified
positive labels are included in v1; all negative labels and translations
still need independent human review.

## Metrics and timing protocol

Each query calls the existing `Index.search(question, embeddings, top_k=10,
min_score=None)`. No translation, filtering, reranking, deduplication, alternate
model, batching of queries or index rebuilding is introduced.

For gold-page set G and the ordered top ten **chunks**, let r be the first
chunk rank whose physical page is in G. Hit@k is 1 if r <= k and 0 otherwise.
MRR@10 is 1/r, or 0 if no relevant chunk occurs in ten. Means are reported
separately for EN, DE and RU over the 18 verified answerable needs. This is
truncated MRR, not full-corpus MRR. Repeated chunks from one page consume ranks,
matching the existing user-facing search. Page matching is intentionally more
permissive than checking whether the returned chunk contains the full answer;
it does not establish chunk relevance, table interpretation or answer quality.

A single warm-up search is excluded. Each of three passes independently
shuffles all 72 queries with one seeded Python RNG. Queries run sequentially,
with no query-embedding cache added. Accuracy uses each query's first measured
trial, so repetitions do not inflate the sample size. Ordered page/chunk
identity consistency is reported across repetitions; review instability before
comparing model quality. The recorded query order supports auditing this rule.

Latency surrounds only `Index.search`: query tokenization and embedding,
vector validation, local Qdrant scoring/filtering and materialization. It
excludes PDF verification, imports, index opening, cached-model loading,
warm-up, report construction and file I/O. Startup components are recorded
separately. Per-language p50/p95 pool all 24 questions x three passes, including
negatives and diagnostics; quantiles use linear interpolation. These are warm,
single-process descriptive timings with filesystem/model caches available,
not cold-start, throughput, concurrency or hardware-independent claims.

For unanswerable queries report the nonempty-result rate and top-score
range/distribution. The unchanged retriever has no abstention decision and
returns nearest neighbors for any question when the corpus is nonempty.
Nonempty results are not measured generation hallucinations or a calibrated
false-positive rate. Similarity is not probability. No threshold is selected
or calibrated on this evaluation set.

## Failure analysis and boundaries

The baseline report lists every need's first relevant rank and all top-10
misses. The JSON preserves enough information to inspect near misses, duplicate
pages and score overlap without distributing copyrighted text. To inspect the
actual passages, use `garage-rag search ... --json` locally and compare with
your original PDF; keep those captures private. Review a suspected missing gold
page independently before changing labels. Preserve the old results if a
later dataset revision corrects a label.

For the tire source audit, the following read-only local inspection reproduces
chunk counts, sizes and token counts and compares nine stored vectors with
fresh embeddings, without printing source text or modifying the index. Run
with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=.cache/huggingface`
and the cached model:

```python
from pathlib import Path
import numpy as np
from garage_rag.embeddings import E5Embeddings
from garage_rag.index import COLLECTION, Index

with Index(Path("data/qdrant")) as index:
    points, cursor = index.client.scroll(COLLECTION, limit=1000, with_vectors=True)
    assert cursor is None  # this baseline has 241 points
    chosen = sorted(
        [p for p in points if p.payload["metadata"]["pdf_page"]
         in (1, 2, 26, 27, 28, 29, 133, 165)],
        key=lambda p: (p.payload["metadata"]["pdf_page"],
                       p.payload["metadata"]["chunk_index"]),
    )
    fresh = E5Embeddings(index.config).embed_documents([p.payload["text"] for p in chosen])
    for point, vector in zip(chosen, fresh):
        meta = point.payload["metadata"]
        difference = float(np.max(np.abs(np.asarray(point.vector) - vector)))
        print(meta["pdf_page"], meta["chunk_index"], len(point.payload["text"]),
              meta["tokens"], difference)
```

Unit/integration checks in `tests/test_evaluation.py` use synthetic ranks and
a synthetic PDF with real Qdrant. They cover cutoff/rank arithmetic, malformed
labels, exclusion of unverified/negative labels, reproducible ordering,
source-identity guards, read-only evaluation and text-free output. They do not
measure E5 quality; the real benchmark remains an explicit local operation.

Limits include one English US-market petrol manual, one phrasing per need per
language, agent-authored translations, no independent assessors, incomplete
relevance judgments, correlated translations and a small sample. Do not treat
54 translated queries or 216 timed searches as 54 or 216 independent needs.
No confidence interval or significance claim is made. The existing extraction
cannot reconstruct tables or image-only symbols; page-level hits can overstate
usable evidence. Generation, refusals, repair accuracy, other manuals, devices
and language variants remain outside this phase.
