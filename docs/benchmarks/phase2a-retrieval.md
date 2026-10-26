# Phase 2A: offline retrieval baseline

Measured on 2026-10-25 against one English US-market E39 525i/530i/540i owner's
manual: 186 physical pages and 241 chunks. This frozen baseline measures the
original E5 Small retriever, without generation or retrieval tuning. The
[dataset](../../benchmarks/bmw_e39_2002.v1.json), [numeric results](phase2a-baseline.v1.json)
and [vector audit](phase2a-tire-audit.v1.json) remain unchanged.

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

## Measured quality and latency

Each language has 18 answerable needs; accuracy is not averaged over repeated
trials or negative cases. Hit@k uses physical-page relevance at the original
chunk rank, without deduplication. MRR is truncated at ten. Latency is warm
`Index.search`, in milliseconds, over 24 questions x three passes per language
(including four negatives and two unscored tire probes).

| Language | Hit@1 | Hit@5 | Hit@10 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EN | 77.8% | 100.0% | 100.0% | 0.8796 | 12.83 | 14.78 |
| DE | 33.3% | 72.2% | 77.8% | 0.5065 | 13.01 | 16.69 |
| RU | 33.3% | 61.1% | 66.7% | 0.4398 | 12.85 | 15.29 |

CPU, Darwin 27.0.0 arm64, Python 3.12.0; Torch 2.14.1, NumPy 2.5.4,
Qdrant client 1.20.0, Sentence Transformers 6.2.0, Transformers 5.20.2.
E5 revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, normalized 384-dimensional
vectors, chunk budget 384 and overlap 48, `query: ` / `passage: ` prefixes.
Torch used four intra-op and eight inter-op threads. Exact Apple chip model,
RAM, thermal state and competing system load were not measured; timings are
not portable hardware claims. Imports/PDF checks/file I/O are excluded.

Cached-model loading took 7.09 s, index opening 34.1 ms, and
the excluded first warm-up search 175.4 ms. Neither is a cold-download measurement.

All 72 query rankings were stable across three passes. A second independent
process with the same seed reproduced all 216 ordered top-ten page/chunk lists
and their scores exactly; latency varied. The published JSON is the final
complete run, not a selected best trial. No crashes, empty positive results
or invalid embeddings occurred. This is repeatability on this local runtime,
not a guarantee of bitwise identity on other platforms.

## Per-need ranks

The following are first relevant **chunk** ranks, not page-deduplicated ranks.
`>10` means the accepted page was absent from the evaluated top ten; it does
not establish absence from the rest of the index.

| Information need ID | Accepted PDF pages | EN rank | DE rank | RU rank |
| --- | --- | ---: | ---: | ---: |
| pressure_label | 26 | 1 | >10 | >10 |
| pressure_table | 27 | 2 | 2 | 3 |
| alarm_sensors | 43 | 1 | 5 | 6 |
| rear_window_lock | 45 | 1 | 2 | >10 |
| seat_memory | 54 | 1 | 2 | 1 |
| airbag_lamp | 60 | 1 | 6 | >10 |
| key_identification | 65 | 1 | 1 | 1 |
| cruise_minimum | 78 | 2 | >10 | >10 |
| ice_warning | 80 | 1 | 1 | 2 |
| dsc_startup | 97 | 2 | 2 | 3 |
| rdc_reset | 99 | 1 | 1 | 1 |
| cabin_filter | 106, 112 | 1 | 4 | 3 |
| flashlight | 116 | 1 | 2 | 4 |
| oil_level | 140 | 1 | 1 | 1 |
| coolant_float | 142 | 1 | 1 | 1 |
| wheel_torque | 159 | 1 | >10 | >10 |
| glovebox_fuses | 160 | 3 | >10 | >10 |
| jump_positive | 162 | 1 | 1 | 1 |

Questions, qualifications and review notes for each ID are in the public
JSON dataset. No source passages, PDF images or index artifacts accompany it.

## Failure analysis and limits

German misses at ten: pressure-label location, cruise minimum speed, wheel torque
and glovebox fuses. Russian additionally misses rear-window locking and the airbag
lamp criterion. Retrieved nearby topics, contents/index entries and related features
do not satisfy the requested detail. Equipment/model qualifications matter: the
adjacent 540i pressure page is not valid evidence for a 530i question. These are
observations, not isolated causes of multilingual ranking errors.

The pressure-label page 26 and 530i table page 27 were fully indexed as individual
chunks (267 and 304 tokens). Cover/navigation fragments sometimes outranked them
for short DE/RU queries. Specific model/tire queries found page 27 at ranks 2/2/3,
while broad queries remained weak; no query rewriting was introduced. Visual
inspection found lost load-pictogram/column relationships in the flattened table.
Better ranking does not establish safe numerical interpretation.

The read-only vector audit re-embedded nine chunks from pages 1, 2, 26–29, 133
and 165: maximum absolute difference was 0.0. This checks those chunks only,
without modifying the index; it does not establish a single cause for the misses.

All four negative needs in all languages returned ten nearest chunks. Negative
top-one scores overlap positive scores: EN 0.8197–0.8511, DE 0.8007–0.8197,
RU 0.7918–0.8224. Nonempty retrieval is expected and is not a hallucination rate
or calibrated refusal rule. No threshold was selected.

Only 18 correlated needs from one manual were scored. Agent-authored labels and
translations lack independent human/native-speaker review and complete relevance
pooling. Known project concerns influenced selection; this is not a blind traffic
sample or significance study. Repeats and translations do not increase independent
sample size. Page hits can overstate chunk sufficiency and cannot reconstruct tables.
Generation, other manuals, MPS and vehicle validation were outside this phase.
Keep v1 frozen; independently review labels and add new held-out needs before
making broader claims. See [Phase 2B](phase2b-retrieval.md) for the subsequent
profile comparison and stricter chunk-sufficiency metrics.
