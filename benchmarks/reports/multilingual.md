# Multilingual query evaluation

Measured on 2026-10-29. English-only query replacement fails on multilingual
PDFs; original-only search loses recall on an English manual. The implemented
path keeps the original question, adds needed source-language queries, and uses
bounded local reranking for expanded mixed-language corpora.

These experiments use frozen EN/DE/RU equivalents instead of fresh translations.
They measure real local extraction, embeddings, persisted Qdrant and retrieval
routing, with **zero live Groq calls**. They do not establish provider translation
quality or end-to-end answer success. Labels are agent-authored, not human gold.

## English BMW manual

The 186-page, 241-chunk source and hash are documented in the
[retrieval report](retrieval.md). All 42 answerable needs from the baseline,
development and held-out datasets were reused with unchanged page labels.
E5 Small, CPU, chunking, prefixes and index were unchanged. Gold hits use physical
pages at original chunk ranks; negatives are excluded.

| Retrieval path | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| Frozen English equivalent | 30/42 | 40/42 | 41/42 | 0.828798 |
| Original Russian only | 11/42 | 21/42 | 25/42 | 0.371259 |
| Original Russian + English | 30/42 | 40/42 | 41/42 | 0.828798 |

Original-only lost 19 old top-five hits; expansion lost none. Hood release and
inflation-label location moved from outside RU top 20 to rank one. The remaining
top-five misses already occurred in English. One frozen counterpart introduces
Latin `US` for Russian `США` and fails the identifier guard: its ranking is a
retrieval-only measurement. Among 41 accepted equivalents, old English and
expanded search both hit 39/41 at five, versus 20/41 original-only.

[All 42 paired rankings](../results/multilingual-ru-en.v1.json) retain case/page/chunk
IDs and cosine scores without source text. These reused correlated cases are not
new independent held-out evidence.

## Bosch C3 language matrix

Robert Bosch GmbH C3 manual, edition 12.2011, 214 physical pages,
[publisher-authored PDF](https://images-eu.ssl-images-amazon.com/images/I/D1FMkfX7UqS.pdf),
SHA-256 `815fa55ce112b1813dae37c2ae72742660310bfb1b5b37125adf6d2e45aac99e`.
Original pages 3–12 DE, 13–22 EN and 183–192 RU supplied eight private layouts:
three monolingual PDFs, one concatenated mixed PDF, DE/RU bilingual, alternating
languages across pages, three languages within each page, and a collection of
three monolingual PDFs. Source passages were not translated or rewritten.

The [dataset](../bosch_c3_multilingual.v1.json) has 12 question triples with
source-page and evidence anchors checked before retrieval. Gold is an anchored
**chunk**, not a related page. Eight layouts × 12 needs × three question languages
produce 288 correlated cases. Metadata deliberately says English even for DE/RU
sources; language hints use actual text. The script exercises `run_answer`
validation/dispatch with frozen translation replies and a refusing writer.
Repeated vectors/reranker pairs are cached in this accuracy run, so it supplies
no latency measurement. Documents/manifests are checked after reopen.

Cells below show anchored Hit@5 counts out of 12, in **DE / EN / RU** order:

| Corpus | Original query | Previous maximum cosine | Corrected reranking |
| --- | ---: | ---: | ---: |
| DE | 10 / 7 / 4 | 10 / 10 / 10 | 10 / 10 / 10 |
| EN | 6 / 8 / 5 | 8 / 8 / 8 | 8 / 8 / 8 |
| RU | 5 / 4 / 10 | 10 / 10 / 10 | 10 / 10 / 10 |
| Mixed PDF | 10 / 8 / 10 | 10 / 10 / 10 | 12 / 12 / 12 |
| DE/RU bilingual | 10 / 5 / 10 | 10 / 10 / 10 | 12 / 12 / 12 |
| Alternating pages | 5 / 5 / 7 | 9 / 9 / 9 | 11 / 11 / 11 |
| Within-page mixing | 10 / 12 / 10 | 11 / 11 / 11 | 12 / 12 / 12 |
| Mixed collection | 10 / 8 / 10 | 10 / 10 / 10 | 12 / 12 / 12 |

Across all 288 cases, previous → corrected:
Hit@1 **0.260417 → 0.479167**, Hit@5 **0.812500 → 0.906250**,
Hit@10 **0.937500 → 0.979167**, MRR@10 **0.471941 → 0.655088**.
Top-five hits improve **234 → 261**; monolingual results stay unchanged. The
previous merge lost ten original-query and eighteen native-query top-five hits;
corrected search loses none of those or the previous top-five/top-ten hits.

## Ranking tradeoffs

Raw cosine across languages is not calibrated relevance. Expanded mixed corpora
reserve each query's first five dense hits before filling a maximum 20-candidate
pool, then use the existing pinned mMARCO MiniLM cross-encoder. Intact passages
are scored with prepared questions in observed source languages; uncertain
passages use available source-language questions. The bound is 60 pairs, with
no source/question truncation. Oversized pairs retain dense order; invalid scores
or other local ranking errors stop retrieval. Original-only and monolingual
paths do not load the reranker.

Admission protects each branch's top-five candidates before reranking, not their
final ranks. Compared with the initial reranker, the audited pool changed 82
ranking rows and reduced Hit@1 **141 → 138** and MRR **0.665505 → 0.655088**,
while top-five/top-ten recall stayed unchanged. The single
[Bosch artifact](../results/multilingual-c3.v2.json) retains original, previous,
pre-audit and corrected rankings. Pre-audit rows preserve the initial candidate;
they are historical measurements, not current results.

Nine prior top-one hits move to anchored rank two, including a small-battery
mode question with a wrong numeric-mode passage first. Twenty-seven top-five
misses remain; these also miss in native-language search. Labels are conservative
anchors, not exhaustive entailment judgments. This matrix guided the fix and is
development evidence from one manual, without broad independent validation.

## Local CPU and memory cost

Separate uncached measurements used three fresh processes per variant/corpus and
seven warm calls per process, serially with order reversed in the second repeat.
macOS 27.2 arm64, Python 3.12.0, CPU, four numerical-library threads, same pinned
E5 Small/MiniLM weights and persisted indexes. No vector/score caches were used.
The previous source is `522a2af`; corrected source hashes are in the
[18-run performance artifact](../results/multilingual-retrieval-performance.v1.json).

| Corpus / path | Queries / pairs | Cold median, s | Warm median / p95, ms | Peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| Bosch mixed, previous | 1 / 0 | 6.830 | 11.03 / 11.77 | 1480.3 |
| Bosch mixed, corrected | 3 / 20 | 8.659 | 813.08 / 871.51 | 1764.0 |
| Within-page, previous | 1 / 0 | 7.309 | 11.05 / 12.35 | 1480.5 |
| Within-page, corrected | 3 / 30 | 8.948 | 1313.68 / 1342.01 | 1845.4 |
| English BMW, previous | 1 / 0 | 6.817 | 14.41 / 14.99 | 1484.1 |
| English BMW, corrected | 2 / 0 | 6.506 | 29.31 / 29.63 | 1496.8 |

Cold totals include imports, index open, language inspection, cached-model load
and first search. They are process-cold, not machine-cold: OS caches are uncontrolled.
Downloads, indexing and provider requests are excluded. Peak RSS is the maximum
whole-process high-water mark, not incremental/steady-state allocation. Mixed
retrieval is about 74× / 119× the previous warm dense path with observed peak
RSS differences 284 / 365 MiB. BMW does not load the reranker; its lower cold
median is not evidence of a speedup. These small corpora and 20/30 measured pairs
do not establish a worst-case bound for 60 pairs or larger indexes.

## Reproduction

Use permitted source PDFs, installed project, cached pinned models and fresh
private paths. Run from the repository root:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
LANGSMITH_TRACING=false python benchmarks/scripts/compare_multilingual_queries.py \
  --output /private/tmp/bmw-query-comparison.json

HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
LANGSMITH_TRACING=false python benchmarks/scripts/compare_multilingual_c3.py \
  --pdf /path/to/bosch-c3-original.pdf --work-dir /private/tmp/c3-matrix \
  --output /private/tmp/c3-matrix.json
```

The BMW script checks PDF identity, its single-document index and unchanged
manifest. Bosch builds layouts/indexes privately; `--previous-merge` selects the
older maximum-cosine path. Its current `expanded` rows correspond to the retained
artifact's `corrected` rows. Pre-audit reproduction requires source at
`01c3d019c35b936491cd93aa96462624dadc5d2a`, without `--previous-merge`.

For performance, extract the previous `src` with `git archive` at
`522a2afafaee7583725e2bc716eb9e97535960b0`. Create a questions JSON object from
Bosch `overheat` or BMW `pressure_label`, then run one fresh process:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
GROQ_API_KEY= LANGSMITH_TRACING=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
python benchmarks/scripts/measure_multilingual_retrieval.py \
  --variant pr --source-dir "$PWD/src" --index-dir /path/to/index-mixed_pdf \
  --questions /path/to/overheat-questions.json --warm-runs 7 \
  --output /private/tmp/mixed-corrected-1.json
```

For previous code use `--variant main --source-dir /path/to/previous/src`.
Those script labels identify historical comparison sides, not the current Git
branch. Repeat three processes per side/corpus with distinct outputs, alternating
order. No provider translation, writing or auditing is included.
