# Phase 2B: multilingual retrieval optimization

## Experimental boundary

The Phase 2A dataset, numeric report, evaluator, E5 adapter, ingestion and index
implementation remain unchanged. `--profile baseline` preserves the original
E5 Small configuration and its default index. Alternative profiles require
separate directories; incompatible fingerprints fail before loading a model.
The production CLI continues to use dense cosine retrieval. The experimental
BM25, RRF and cross-encoder adapters are confined to the offline comparison
entry point, with no added runtime dependency.

All candidates use the same 186-page PDF, its original hash guard, the same
241 chunks (384-token budget, 48-token overlap), original chunk ranks, top ten
results, no score threshold, CPU, four Torch intra-op/eight inter-op threads,
one excluded warmup and three seeded sequential shuffled passes. The chunk
text/position digest is checked against the frozen datasets before inference.
Model downloads/index construction precede evaluation and are not query latency.
No translation service, generation model, query dictionary or corpus-specific
boost is involved. Every model revision is pinned and remote code is disabled.

The 12-need [development set](../../benchmarks/bmw_e39_2002.phase2b-dev.v1.json)
and 12-need [held-out set](../../benchmarks/bmw_e39_2002.phase2b-heldout.v1.json)
were committed before the first ranking experiment (`223f4f5`). Each has
EN/DE/RU formulations; no need duplicates a Phase 2A information need. Source
labels were established using independent Poppler text and rendered original
pages, without retrieval candidates. Agent review is not independent human
review. New needs can share topics with old needs; these are not topic-disjoint
or independently sampled corpora.

Only development results select the pipeline. The old 18 needs serve as a
legacy regression check; the new held-out needs are scored after selection,
without retuning. Selection considers per-language Hit@5, then MRR, latency,
resources and implementation burden. This is a bounded model comparison,
not an exhaustive hyperparameter sweep or a pre-registered statistical study.

## Candidate definitions

- E5 Small: original normalized 384-dimensional mean-pooled embeddings with
  `query: ` / `passage: ` prefixes. The reference pipeline.
- BM25: in-process Unicode word tokenization and case folding, k1=1.5, b=0.75,
  positive IDF, no stemming, stopwords or translation. Only active revisions
  enter the corpus; cache refreshes when active generations change.
- Hybrid: E5 Small top 20 and BM25 top 20, equal-weight reciprocal rank fusion
  `1/(60 + one-based rank)`, deterministic chunk-key tie breaking. Cosine,
  BM25 and fusion scores are kept distinct; cosine thresholds are rejected.
- [E5 Base](https://huggingface.co/intfloat/multilingual-e5-base): pinned
  `d128750597153bb5987e10b1c3493a34e5a4502a`, 768 dimensions, same prefixes.
  It increases encoder capacity while holding representation and chunks fixed.
- [mMARCO MiniLM reranker](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1):
  pinned `1427fd652930e4ba29e8149678df786c240d8825`, reranks E5 Small's
  top 20 with batch size 8. Oversized query/passage pairs fail rather than
  silently truncating. It cannot recover candidates outside those twenty.
- [BGE-M3](https://huggingface.co/BAAI/bge-m3): pinned
  `5617a9f61b028005a4858fdac845db406aefb181`, normalized 1024-dimensional
  dense vectors, model-supplied CLS pooling, no E5 prefixes. Only its dense
  Sentence Transformers path is tested; sparse/ColBERT modes and FlagEmbedding
  are not added. Inputs retain the application's 512-token bound despite the
  model's longer context capacity. Its tokenizer yields identical corpus chunks.

MiniLM paraphrase embeddings were screened out before execution: their native
[128-token configuration](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
would add a chunking/aggregation intervention. Larger generative query rewriting
would add latency, another model and less predictable behavior. Neither is a
measured losing candidate. RRF combines ranks rather than incomparable raw
scores ([Qdrant explanation](https://qdrant.tech/documentation/search/text-search/hybrid-search/)).
The small corpus makes an in-memory BM25 implementation sufficient for this
experiment; this is not a large-corpus scalability benchmark.

## Development comparison and frozen decision

Measured on Apple M1, 16 GiB RAM, macOS arm64 CPU. Each cell in the
language columns is Hit@5 / MRR@10 over 12 needs. Latency columns show
EN / DE / RU p50 milliseconds. RSS is process peak, not tensor allocation.

| Candidate | EN | DE | RU | p50 ms EN / DE / RU | Peak RSS MiB |
| --- | ---: | ---: | ---: | --- | ---: |
| baseline-dense | 100.0% / 0.7361 | 58.3% / 0.4216 | 41.7% / 0.3063 | 13.09 / 13.25 / 13.10 | 1467 |
| baseline-bm25 | 100.0% / 1.0000 | 16.7% / 0.1134 | 8.3% / 0.0833 | 0.65 / 0.30 / 0.25 | 278 |
| baseline-hybrid | 100.0% / 0.9167 | 66.7% / 0.4132 | 41.7% / 0.2438 | 14.28 / 13.90 / 13.82 | 1434 |
| e5-base-dense | 100.0% / 0.7389 | 58.3% / 0.4500 | 41.7% / 0.2801 | 32.43 / 32.83 / 33.15 | 1369 |
| baseline-rerank | 100.0% / 0.9444 | 83.3% / 0.7083 | 66.7% / 0.5605 | 743.54 / 718.61 / 739.84 | 1687 |
| bge-m3-dense | 100.0% / 0.9167 | 100.0% / 0.7736 | 100.0% / 0.7389 | 98.16 / 100.26 / 100.24 | 2328 |

**Decision, frozen before held-out evaluation: select BGE-M3 dense retrieval.**
It reaches 12/12 page Hit@5 in each language on development, improving DE
from 7/12 and RU from 5/12, while retaining EN 12/12. Its roughly 100 ms
median is slower than Small's 13 ms but substantially faster than the
roughly 730 ms reranker. One encoder and a single dense index preserve the
existing ingestion/search/generation architecture. No fusion weights,
thresholds, language dictionaries or per-topic boosts are tuned. The default
remains E5 Small for compatibility; BGE-M3 is the explicitly selected opt-in.

E5 Base does not improve DE/RU Hit@5 and is rejected for this use. BM25 alone
fails to bridge languages; hybrid only improves one German need while reducing
RU MRR. Reranking improves ordering but cannot recover missing candidates and
has higher latency. This decision is based on development only; it must not be
revised after seeing held-out scores. Six candidates were compared, so model
selection itself is a source of optimistic development estimates.

## Final held-out and legacy comparison

The decision above was committed as `bc984c0` before any held-out ranking.
The following are the first complete final measurements, not the best of
repeated runs. Accuracy uses one trial per need. Latency pools all three trials;
legacy timing includes its four negatives and two diagnostics per language.
The [consolidated numeric artifact](phase2b-results.v1.json) retains first-trial
rankings/scores, later-trial score differences, every latency, consistency and provenance.
Full per-trial local outputs remain private. Early development runs predate
additional weight-size instrumentation; the artifact inventories pinned weight
files separately rather than inventing missing measurements.

### Heldout (12 needs per language)

| Profile | Language | Hit@1 | Hit@5 | Hit@10 | MRR@10 | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | EN | 83.3% | 83.3% | 91.7% | 0.8452 | 12.99 | 15.28 |
| baseline | DE | 33.3% | 58.3% | 83.3% | 0.4562 | 12.97 | 14.97 |
| baseline | RU | 25.0% | 41.7% | 41.7% | 0.3333 | 13.08 | 15.17 |
| bge-m3 | EN | 75.0% | 91.7% | 91.7% | 0.8125 | 99.40 | 107.70 |
| bge-m3 | DE | 66.7% | 75.0% | 100.0% | 0.7226 | 100.43 | 121.78 |
| bge-m3 | RU | 58.3% | 66.7% | 83.3% | 0.6435 | 99.80 | 125.52 |

### Legacy (18 needs per language)

| Profile | Language | Hit@1 | Hit@5 | Hit@10 | MRR@10 | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | EN | 77.8% | 100.0% | 100.0% | 0.8796 | 13.25 | 15.21 |
| baseline | DE | 33.3% | 72.2% | 77.8% | 0.5065 | 13.11 | 15.98 |
| baseline | RU | 33.3% | 61.1% | 66.7% | 0.4398 | 13.54 | 16.74 |
| bge-m3 | EN | 72.2% | 100.0% | 100.0% | 0.8352 | 99.02 | 115.35 |
| bge-m3 | DE | 61.1% | 83.3% | 88.9% | 0.7148 | 98.57 | 116.26 |
| bge-m3 | RU | 66.7% | 94.4% | 94.4% | 0.7657 | 98.95 | 112.42 |

Held-out gains are smaller than development: BGE-M3 is not a universal fix.
English MRR declines from 0.8452 to 0.8125 on held-out and from 0.8796 to
0.8352 on legacy, despite improved/preserved Hit@5. On legacy, coolant,
wheel-torque and glovebox-fuse passages move down; the model-specific pressure
table improves from rank two to one. These are measured ordering tradeoffs.
Baseline remains the default, and a user prioritizing English MRR or low
latency can keep the original profile. The multilingual gains justify the
explicit BGE-M3 option, not an automatic migration or a claim of dominance.

### Chunk sufficiency (held-out, 11 text-answerable needs)

| Profile | Language | Evidence Hit@5 | Evidence MRR@10 |
| --- | --- | ---: | ---: |
| baseline | EN | 81.8% | 0.8312 |
| baseline | DE | 54.5% | 0.4068 |
| baseline | RU | 36.4% | 0.2727 |
| bge-m3 | EN | 90.9% | 0.7955 |
| bge-m3 | DE | 72.7% | 0.6973 |
| bge-m3 | RU | 63.6% | 0.6566 |

The denominator excludes the engine table for every language/profile. On the
development set, hybrid DE and BM25 DE/RU can hit the PDC page while returning
the chunk about malfunction rather than the distance explanation. Page Hit
therefore must not be substituted for sufficient-evidence Hit. These manual
chunk labels still require independent review.

### Resource cost

| Profile | Held-out peak RSS MiB | Legacy peak RSS MiB | Index MiB | Weights MiB | Cached load s (held-out) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1482 | 1375 | 1.34 | 448.8 | 5.57 |
| bge-m3 | 2269 | 2429 | 2.83 | 2165.9 | 5.06 |

One-time BGE-M3 indexing took 96.1 s (7.1 s extraction/chunking and 89.0 s
embedding/storage), compared with 33.7 s for E5 Base in this session. These
index-build timings are not a controlled baseline-Small build comparison;
the original Small index was deliberately not rebuilt. The two final BGE
processes peaked at roughly 2.2–2.4 GiB RSS on the actual M1/16 GiB machine.
CPU search near 0.1 s is practical for an interactive personal CLI, with a
substantial model-download/disk/startup cost. No new paid service or dependency
is required. Warm RSS and model-file size differ because of paging and runtime
overhead; neither alone is a full system-memory budget or battery measurement.

### Remaining errors and abstention

- Held-out EN still misses `toolkit_access` at ten; DE finds it at seven and
  RU at nine. The correct page 150 has the location sentence, mixed wiper
  instructions and the tool-kit heading at the end of extracted text. Both
  ranking and representation deserve investigation; this does not establish
  that extraction caused the miss. No chunks or labels were changed afterward.
- DE mirror defrost is rank nine, wagon backrest rank six; RU mirror defrost
  is rank nine and the wagon-backrest and hood-release pages are absent at ten.
  Related mirror/climate, seat and lock pages outrank the qualified evidence.
  Multilingual specificity and body-style distinctions remain weak points.
- Legacy DE still misses the cruise minimum-speed page and glovebox-fuse access
  at ten; its flashlight passage drops from rank two to six. RU still misses
  glovebox fuses. BGE-M3 improves many old failures but does not eliminate them.
- The engine table and tire-pressure table require original layout review even
  when their pages rank first. Better retrieval cannot verify column alignment,
  load symbols, model applicability or generated numerical interpretation.
- Both final dense profiles return nonempty results for every legacy negative
  query. BGE's scores use a different distribution and still overlap positives;
  there is no calibrated threshold, measured false-answer rate or improved
  abstention claim. Free-provider answer quality was not benchmarked in Phase 2B.

No held-out-specific reranking, prompt changes, dictionary, label correction
or threshold was introduced. The measured English regressions and remaining
misses are retained in the report instead of tuning against this final set.

### Repeatability and compatibility

All ten experiment rankings were stable across their three passes. The four
final comparisons were repeated in fresh processes: all 648 ordered top-ten
trial lists and every score matched exactly, with identical shuffled query
orders. Within repeats, two queries (development BGE belt-accident EN and
held-out baseline manual-reverse EN) have score-only differences of at most
1.11e-16; their ordering is stable. The artifact preserves those differences.
Latency varied, as expected. The original Phase 2A evaluator was used
for the legacy baseline repeat; all 216 trials also matched the published
Phase 2A artifact exactly. This is local CPU repeatability, not a promise of
bitwise identity across other hardware or library versions.

The original evaluator, Index, E5 adapter, chunking, baseline configuration,
generation and language modules remain unchanged. Synthetic tests exercise
selected-profile EN/DE/RU `ask`, verified citations and the existing free-only
routing contracts. No live generation request or browser UI test is needed
for this retrieval-only CLI change; no claim of improved answer reasoning is
made. Benchmark models, PDFs, extracts and indexes remain private.

## Evidence quality and limitations

Page Hit@1/5/10 and MRR@10 use the unchanged Phase 2A arithmetic and scheduler.
The new datasets additionally label chunks whose text is sufficient for the
specific need. Evidence Hit/MRR uses those chunk identities, separately from
page relevance. These labels are tied to the frozen chunk digest, contain no
source passages and were not inferred from a successful search.

The held-out engine-cylinder question requires the table's column mapping.
It is included in page metrics and explicitly excluded from text-sufficiency
metrics. A retrieved table page does not certify a numerical answer. The old
tire-pressure table has the same outstanding layout limitation. No OCR, table
reconstruction or generation-quality certification is claimed.

There are only 12 new needs per split and 18 legacy needs, all from one English
US-market manual. The same agent authored questions, translations and labels;
there is no independent human ground truth or native-speaker review. Languages
are correlated versions of the same need, and repeats do not increase the
accuracy sample size. Configuration selection may inflate development results;
the held-out set reduces direct tuning leakage but is neither a blind external
study nor proof of generalization. No significance claim is made. Similarity
scores are not calibrated confidence or an abstention rule. Real users,
additional manuals, MPS and end-to-end free-provider answer quality remain
unmeasured. Free-only generation and citation/language validation are unchanged.

## Reproduction

Use the same local PDF, metadata and pinned dependencies as
[Phase 2A](retrieval-evaluation.md). Do not replace `data/qdrant` or modify the
v1 benchmark. Index construction may download a model; comparison is offline
and does not read `.env` or initialize OpenRouter. Set `HF_HOME` consistently.
The default CLI profile remains `baseline` for compatibility. For an alternative,
always name the profile and private directory explicitly, including for `ask`:

```bash
# Prepare an isolated candidate index; substitute e5-base for that experiment.
garage-rag --profile bge-m3 --index-dir data/qdrant-bge-m3 index \
  data/private/bmw_e39_2002_manual.pdf \
  --title 'BMW E39 2002 Owner Manual' --language en \
  --models 'BMW 525i,BMW 530i,BMW 540i' --kind owner_manual \
  --source-url https://www.ge39.com/files/2002Manual-5ser.pdf

garage-rag --profile bge-m3 --index-dir data/qdrant-bge-m3 \
  search 'Где находится диагностический разъем OBD?' --top-k 5
garage-rag --profile bge-m3 --index-dir data/qdrant-bge-m3 \
  ask 'Wie prüfe ich den Motorölstand?'

# Preserve the original route, without converting or overwriting its index.
garage-rag --profile baseline --index-dir data/qdrant search 'tire pressure'

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= \
python -m garage_rag.optimization \
  --dataset benchmarks/bmw_e39_2002.phase2b-dev.v1.json \
  --pdf data/private/bmw_e39_2002_manual.pdf \
  --index-dir data/qdrant-bge-m3 --profile bge-m3 --method dense \
  --output output/evaluation/phase2b-dev-bge.json --repeats 3 --seed 20261025
```

For the complete development comparison, run the same command sequentially
with `(profile, method)` pairs `(baseline, dense)`, `(baseline, bm25)`,
`(baseline, hybrid)`, `(e5-base, dense)`, `(baseline, rerank)` and `(bge-m3, dense)`.
Use the matching index and a new output file each time. Prepare the reranker
cache once with `HF_HOME=.cache/huggingface python -c
'from garage_rag.retrieval import LocalReranker; LocalReranker()'` while online.
Do not run concurrent model inference during timing. There is no query cache.

Freeze the development decision before running
`bmw_e39_2002.phase2b-heldout.v1.json`. Evaluate only the baseline and selected
candidate on that split and the unchanged `bmw_e39_2002.v1.json` legacy set.
Repeat those final comparisons in new processes, preserving parameters; compare
ordered chunk IDs and scores, allowing latency to vary. The original
`python -m garage_rag.evaluation ...` command remains available unchanged.

The experiment runner checks corpus completeness, original PDF hash/page count,
configuration, chunk digest, manifest immutability and finite scores, and uses
the Phase 2A evaluator directly. Output includes all repeats, query order,
source/dataset/code hashes, pinned model/ranking parameters, library versions,
startup timings, whole-process peak RSS and logical index/weight bytes. Peak RSS
includes Python/import/model overhead and reflects OS paging; it is not the
model's tensor size, available RAM or guaranteed maximum memory on every device.
Warm latency includes query encoding and every ranking stage, excluding cached
model load, PDF checks, corpus digest, warmup and result-file I/O. Compare wall
clock startup separately. No cold-download, throughput or battery benchmark is
claimed. Local full outputs stay under ignored `output/`; publish only the
consolidated text-free numeric artifact and this report, not a run diary.
