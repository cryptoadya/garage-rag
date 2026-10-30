# Retrieval evaluation

Measured on 2026-10-25 using the English US-market 2002 BMW E39 owner's manual:
186 physical pages, 241 chunks, SHA-256
`9faf9c33f9958f037c77032e514295378ddb490217c1d37015d6ca0fb48b4574`.
The PDF is not distributed. Obtain a permitted copy from the source in the
[setup guide](../../README.md#quick-start).

## Method

The [baseline dataset](../bmw_e39_2002.v1.json) has 18 answerable needs, four
separate negatives and two diagnostic tire-pressure probes, each asked in EN/DE/RU.
Labels were established by agent inspection of original PDF text and rendered
pages before ranking; `human_verified` is false. Equipment/market qualifiers
matter: the adjacent 540i pressure table is not evidence for a 530i question.

Hit@k checks whether a gold physical page occurs among the first k **chunks**;
MRR@10 is the reciprocal first relevant chunk rank, or zero outside ten. Chunks
from the same page consume separate ranks. A page hit does not prove that the
chunk contains a complete answer or preserves table relationships. Negatives
and probes are excluded from accuracy denominators.

CPU runs used pinned models, normalized cosine vectors, 384-token page-local
chunks with 48-token overlap, no score threshold, one excluded warmup and three
seeded shuffled passes. Accuracy uses the first trial; repetitions do not increase
sample size. Warm latency measures search/encoding/ranking, excluding model load,
index creation, PDF checks and file I/O. Timings are local observations, not
hardware-independent limits. No generation calls were made.

## E5 Small baseline

[Full results](../results/phase2a-baseline.v1.json), 18 answerable needs per language:

| Language | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Warm p50 / p95, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN | 77.8% | 100.0% | 100.0% | 0.8796 | 12.83 / 14.78 |
| DE | 33.3% | 72.2% | 77.8% | 0.5065 | 13.01 / 16.69 |
| RU | 33.3% | 61.1% | 66.7% | 0.4398 | 12.85 / 15.29 |

All 72 rankings were stable across three passes; another process reproduced
all 216 ordered top-ten lists and scores. The [vector audit](../results/phase2a-tire-audit.v1.json)
re-embedded nine selected chunks with maximum absolute difference 0.0. This does
not isolate a cause for multilingual misses. Extracted pressure tables lost
column/pictogram relationships even when the correct page was retrieved.

Every negative query returned ten neighbors. Negative top-one scores overlapped
positive scores; no abstention threshold was calibrated. Similarity is not an
answer probability or a measured hallucination rate.

## Candidate comparison

The development and held-out datasets each contain 12 disjoint needs; the original
18-need set remains a separate legacy check. The same PDF, chunk identities and
metric definitions were retained. Models/revisions and remote-code settings were
pinned. BM25 used Unicode terms, k1=1.5 and b=0.75; hybrid combined top-20 dense
and BM25 ranks with equal-weight reciprocal rank fusion. The reranker scored a
bounded dense candidate pool. These were offline adapters, without translation
or generation. See the [complete artifact](../results/phase2b-results.v1.json).

Development cells show Hit@5 / MRR@10:

| Candidate | EN | DE | RU | Warm p50 EN / DE / RU, ms | Peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| E5 Small dense | 100.0% / 0.7361 | 58.3% / 0.4216 | 41.7% / 0.3063 | 13.09 / 13.25 / 13.10 | 1467 |
| BM25 | 100.0% / 1.0000 | 16.7% / 0.1134 | 8.3% / 0.0833 | 0.65 / 0.30 / 0.25 | 278 |
| Hybrid | 100.0% / 0.9167 | 66.7% / 0.4132 | 41.7% / 0.2438 | 14.28 / 13.90 / 13.82 | 1434 |
| E5 Base dense | 100.0% / 0.7389 | 58.3% / 0.4500 | 41.7% / 0.2801 | 32.43 / 32.83 / 33.15 | 1369 |
| E5 Small + MiniLM | 100.0% / 0.9444 | 83.3% / 0.7083 | 66.7% / 0.5605 | 743.54 / 718.61 / 739.84 | 1687 |
| BGE-M3 dense | 100.0% / 0.9167 | 100.0% / 0.7736 | 100.0% / 0.7389 | 98.16 / 100.26 / 100.24 | 2328 |

BGE-M3 was selected before held-out scoring; the decision is recorded in
[commit bc984c0](https://github.com/cryptoadya/garage-rag/commit/bc984c0).
The default runtime profile remains E5 Small. Final held-out results:

| Profile | Language | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Warm p50 / p95, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E5 Small | EN | 83.3% | 83.3% | 91.7% | 0.8452 | 12.99 / 15.28 |
| E5 Small | DE | 33.3% | 58.3% | 83.3% | 0.4562 | 12.97 / 14.97 |
| E5 Small | RU | 25.0% | 41.7% | 41.7% | 0.3333 | 13.08 / 15.17 |
| BGE-M3 | EN | 75.0% | 91.7% | 91.7% | 0.8125 | 99.40 / 107.70 |
| BGE-M3 | DE | 66.7% | 75.0% | 100.0% | 0.7226 | 100.43 / 121.78 |
| BGE-M3 | RU | 58.3% | 66.7% | 83.3% | 0.6435 | 99.80 / 125.52 |

The stricter evidence check excludes one layout-dependent engine-table need:
on 11 text-answerable held-out needs, evidence Hit@5 for E5 Small / BGE-M3 was
81.8% / 90.9% EN, 54.5% / 72.7% DE and 36.4% / 63.6% RU.
Held-out peak process RSS was 1482 / 2269 MiB; cached weights 448.8 / 2165.9 MiB;
index sizes 1.34 / 2.83 MiB. BGE-M3 improved recall with roughly 7–8× warm latency.
It did not improve every top-one result and still missed relevant pages.

Fresh-process repeats of four final comparisons reproduced all 648 ordered
top-ten trial lists and scores. This is repeatability on one local CPU runtime,
not cross-platform bitwise reproducibility or statistical significance.

## Reproduction

Prepare the baseline index using the README. For alternatives, use an explicit
profile and separate index directory; never convert the existing baseline index.
Cache the pinned weights before running offline. From the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= \
python -m garage_rag.evaluation \
  --dataset benchmarks/bmw_e39_2002.v1.json \
  --pdf data/private/bmw_e39_2002_manual.pdf --index-dir data/qdrant \
  --output output/evaluation/e39-baseline.json --repeats 3 --seed 20261025 --device cpu

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= \
python -m garage_rag.optimization \
  --dataset benchmarks/bmw_e39_2002.phase2b-dev.v1.json \
  --pdf data/private/bmw_e39_2002_manual.pdf \
  --index-dir data/qdrant-bge-m3 --profile bge-m3 --method dense \
  --output output/evaluation/bge-dev.json --repeats 3 --seed 20261025
```

Development profile/method pairs were `(baseline, dense)`, `(baseline, bm25)`,
`(baseline, hybrid)`, `(e5-base, dense)`, `(baseline, rerank)`, `(bge-m3, dense)`.
Prepare each matching index with `garage-rag --profile ... --index-dir ... index`.
For final comparison, substitute the held-out or legacy dataset, evaluate only
baseline and the selected candidate, and use a fresh output file for each run.
Evaluators check PDF/configuration/chunk identity and manifest immutability.

## Interpretation

One manual and small agent-authored sets cannot establish general multilingual
quality. Translations and repeated runs are correlated; no human/native-speaker
review, significance test, MPS validation or answer-quality measurement occurred.
Page matches can overstate evidence sufficiency, especially for tables. Full
rankings, model revisions, environment details and legacy scores remain in the
frozen JSON; [multilingual experiments](multilingual.md) study later query routing.
