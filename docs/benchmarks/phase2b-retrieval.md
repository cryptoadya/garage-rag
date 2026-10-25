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
