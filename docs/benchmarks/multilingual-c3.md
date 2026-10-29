# Real EN/DE/RU and mixed-PDF retrieval — 2026-10-29

The prior maximum-cosine query merge restored RU→EN BMW recall but introduced
losses in mixed-language corpora. The fix reuses the project's pinned local
mMARCO MiniLM reranker only for expanded mixed-language answers. It adds local
inference, with no additional Groq call, dependency or index migration.

## Source and protocol

- Real Robert Bosch GmbH C3 manual, edition 12.2011, 214 physical PDF pages,
  [publisher-authored PDF hosted by Amazon](https://images-eu.ssl-images-amazon.com/images/I/D1FMkfX7UqS.pdf).
  SHA-256: `815fa55ce112b1813dae37c2ae72742660310bfb1b5b37125adf6d2e45aac99e`.
- Original physical pages 3–12 (DE), 13–22 (EN), 183–192 (RU). No source passages
  were translated or rewritten. PDF derivatives stay private. Different editions
  have different pagination; the script requires this exact original hash.
- 12 agent-authored question triples, with language-specific original pages and
  evidence anchors checked against the source before retrieval. German and
  Russian operation pages were also rendered and visually inspected. No human
  gold review or live provider translation/answer evaluation is claimed.
- Eight corpus layouts: three separate monolingual PDFs; a concatenated EN/DE/RU
  PDF; a DE/RU bilingual PDF; a partitioned PDF whose successive pages use
  alternating source languages; three-column pages containing original DE/EN/RU
  pages; and one collection of three separate monolingual PDFs.
- **288 cases:** eight corpora × 12 needs × three question languages. Gold is a
  case-specific anchored **chunk**, not merely a page with related text. Labels
  and questions are fixed across every path; no labels were relaxed after scoring.
- Real PDF extraction, E5 Small embeddings, persisted/reopened Qdrant indexes,
  local source-language hints, `run_answer` translation validation and retrieval
  dispatch are exercised. Source metadata is deliberately stale (`language: en`)
  even in DE/RU PDFs. Document hashes, dimensions, prefixes and manifests remain
  unchanged after reopen. Matching uses original source text, not the metadata.
- Frozen parallel questions supply translation responses. A scripted writer
  refuses rather than inventing an answer. **Zero live Groq calls.** This tests
  routing/ranking and bounded preparation calls, not real model answer quality.
- Collect top 20 and report Hit@1/5/10 at original chunk ranks. Dense merging has
  a prefix-consistent top-k; reranking scores a bounded 20-candidate pool before
  applying requested top-k. Repeated exact query vectors and local relevance
  pairs are cached in this offline measurement; no latency claim is inferred.

## Results

Each table cell is the number of anchored top-five hits out of 12, in question
language order **DE / EN / RU**. The previous path is commit `53155b2`.

| Corpus | Original query | Previous maximum cosine | Corrected local reranking |
| --- | ---: | ---: | ---: |
| de | 10 / 7 / 4 | 10 / 10 / 10 | 10 / 10 / 10 |
| en | 6 / 8 / 5 | 8 / 8 / 8 | 8 / 8 / 8 |
| ru | 5 / 4 / 10 | 10 / 10 / 10 | 10 / 10 / 10 |
| mixed_pdf | 10 / 8 / 10 | 10 / 10 / 10 | 12 / 12 / 12 |
| bilingual_de_ru | 10 / 5 / 10 | 10 / 10 / 10 | 12 / 12 / 12 |
| partitioned | 5 / 5 / 7 | 9 / 9 / 9 | 11 / 11 / 11 |
| mixed_within_page | 10 / 12 / 10 | 11 / 11 / 11 | 12 / 12 / 12 |
| mixed_collection | 10 / 8 / 10 | 10 / 10 / 10 | 12 / 12 / 12 |

Across all 288 cases:

| Metric | Previous | Initial reranker (`01c3d01`) | Audited candidate pool |
| --- | ---: | ---: | ---: |
| hit@1 | 0.260417 | 0.489583 | 0.479167 |
| hit@5 | 0.812500 | 0.906250 | 0.906250 |
| hit@10 | 0.937500 | 0.979167 | 0.979167 |
| mrr@10 | 0.471941 | 0.665505 | 0.655088 |

Top-five hits improve **234/288 → 261/288**. The prior merge lost ten hits that
original-query search found and eighteen hits found by a source-native question.
Corrected retrieval loses **zero** of those hits, and zero of the previous
maximum-cosine top-five/top-ten hits. Mono-language results are unchanged.

## Correction and operational cost

Raw cosine scores across languages are not calibrated relevance probabilities.
Maximum-score merging can promote unrelated foreign-language passages over a
correct native-language hit. Plain rank fusion and BM25 fusion also left losses
in this development matrix; no per-topic weights or word dictionaries were added.

For expanded multi-language corpora, keep a maximum of 20 dense candidates and
reuse `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, pinned at
`1427fd652930e4ba29e8149678df786c240d8825`. Before trimming the pool, reserve
the first five results of each query (at most 15 unique chunks), then fill the
remaining slots in maximum-cosine order. Deduplicate by point ID and keep the
maximum raw cosine. This protects branch top-five candidates **before** reranking;
it cannot guarantee their rank after reranking. Score intact passages with their
observed language's prepared question, or all available source-language questions
when uncertain. Mixed passages can use multiple questions; retain the best local
relevance score. No source is discarded by language, source IDs/pages/quotes stay
unchanged, and raw dense scores remain available. Min-score is applied to dense
candidates before reranking; the final order uses local relevance, not cosine.

The existing model is loaded lazily, cached on disk and reused by the adapter.
Its weights may download on first use and it adds local CPU/memory cost. At most
60 pairs are scored per batch. Complete pairs must fit the existing 512-token
limit; otherwise retain dense order, without truncating text or adding provider
requests. Invalid/nonfinite scores and other local ranking failures stop safely.
Single-language corpora and original-only searches retain their prior path and
do not load the extra model. Groq preparation remains at most one request;
writer/auditor quotas, numeric/citation gates and no-provider-retry rules persist.
Corpus-language inspection loads active passages without constructing BM25 token
counts, document frequencies or lengths; those are built lazily for lexical search.

The real-Qdrant adversarial regression test gives all twenty hits in one branch
higher cosine scores than every hit in the other branches. The initial pool loses
all other branches' top five; the reserved pool retains them with two or three
queries, including when only one final answer passage is requested. The reranking
limit stays at 20 candidates / 60 intact pairs and one Qdrant batch.

The same frozen 288-case matrix was rerun after that change. 82 complete ranking
rows changed, including 15 metric rows. Top-five/top-ten hits and zero lost
original/native top-five hits remain unchanged. Top-one hits fall **141 → 138**
and MRR@10 **0.665505 → 0.655088** as newly admitted candidates compete in the
same reranker. The [paired artifact](multilingual-c3.v2.json) is the single
retained Bosch result: it includes `original`, `previous`, `pre_audit` and current
`corrected` rankings with unchanged labels. Its `pre_audit` rows exactly preserve
the initial reranker record from commit `01c3d019c35b936491cd93aa96462624dadc5d2a`;
the duplicate v1 file is available in that commit's Git history. No new ranking
model or fusion algorithm was introduced. Local performance is measured below.

The prior 42-question English BMW check was rerun separately: corrected RU+EN
retrieval remains **40/42 at five**, with MRR@10 **0.828798** and no lost old EN
hits. The [existing BMW report](multilingual-ru-en.md) explains its translation
controls and limits.

## Remaining limits

- Top-one ordering is not uniformly preserved. Aggregate top-one hits improve
  **75 → 138**, but nine old top-one hits (three corpus/need pairs in three
  languages) move to annotated rank **2**: disposal in the partitioned corpus,
  small-battery mode and disposal in the within-page mixed corpus. Small-battery
  mode still places a wrong numeric-mode
  passage first. Correct evidence remains in top five; this is not a claim of
  perfect top-one precision, including for `--top-k 1`.
- The fixed anchors are conservative, not exhaustive entailment labels. For
  example, a disposal pictogram repeats a relevant instruction outside the
  annotated disposal section; the labels were not changed to count it as a hit.
- 27 top-five misses remain: two DE needs, four EN needs, two RU needs, and the
  connection-order need in the partitioned corpus, each queried in all three
  languages. Those also miss at five in matching source-native retrieval.
- This matrix guided the fix and is development evidence from one manual, not
  new independent held-out proof or universal recall. Translations and original
  pages are correlated. Frozen translation responses and scripted refusals say
  nothing about live Groq answer quality. Citation and numeric validation cannot
  recover evidence absent from the final context.

## Reproduce

Keep the original PDF private, use the installed project and cached pinned model
weights, and run from the repository root:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  LANGSMITH_TRACING=false .venv/bin/python docs/benchmarks/compare_multilingual_c3.py \
  --pdf /path/to/bosch-c3-original.pdf --work-dir /private/tmp/c3-matrix \
  --output /private/tmp/c3-matrix.json
```

The [dataset](../../benchmarks/bosch_c3_multilingual.v1.json) records questions,
source pages and small evidence anchors. The [script](compare_multilingual_c3.py)
constructs the corpus layouts without redistributing them and runs the actual
retrieval adapter. The [current paired ranking artifact](multilingual-c3.v2.json)
retains source-free original/previous/pre-audit/corrected rankings, metrics and
gold chunk IDs. Use
`--previous-merge` with a different output filename to reproduce the previous
maximum-cosine path (`53155b2`), or omit it for corrected ranking. All other
inputs and labels are identical. The script emits `original` and `expanded`
rows for the selected path; compare current `expanded` rows with the frozen
artifact's `corrected` rows. Historical `previous` and `pre_audit` rows remain
comparison evidence, not newly measured results. To rerun the pre-audit path,
extract `src` at `01c3d019c35b936491cd93aa96462624dadc5d2a` and set `PYTHONPATH`
to that extracted `src` when running the same script without `--previous-merge`.

## Cold/warm latency and native process memory

Same macOS 27.2 arm64 host, Python 3.12.0, CPU, four numerical-library threads,
same pinned E5 Small and local mMARCO MiniLM revisions, same persisted indexes.
Three fresh processes per variant/corpus, seven warm calls per process (21 warm
samples); serial execution, main/PR order reversed in the second repetition.
No query-vector or reranker-score cache. Frozen RU questions use the existing EN
translation on main and RU plus missing corpus languages on the audited PR.
This compares the actual different local paths, not equal numbers of vectors.

Cold total starts before application imports and includes index open, PR corpus
language inspection, lazy model loading from cached weight files and first search.
It is process-cold, not machine-cold: OS filesystem cache is uncontrolled.
Model downloads, PDF indexing and all provider translation/writer/auditor calls
are excluded. Warm timing includes real query embedding, Qdrant and any local
reranking with already loaded models. Peak RSS uses native `getrusage`, including
PyTorch/native allocations, and is the maximum process high-water mark among the
three runs; it is not an incremental or steady-state memory estimate.

| Corpus / path | Queries / scored pairs | Cold total median, s | Warm median / p95, ms | Peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| Bosch mixed PDF, main | 1 / 0 | 6.830 | 11.03 / 11.77 | 1480.3 |
| Bosch mixed PDF, audited PR | 3 / 20 | 8.659 | 813.08 / 871.51 | 1764.0 |
| Bosch within-page mixing, main | 1 / 0 | 7.309 | 11.05 / 12.35 | 1480.5 |
| Bosch within-page mixing, audited PR | 3 / 30 | 8.948 | 1313.68 / 1342.01 | 1845.4 |
| English BMW, main | 1 / 0 | 6.817 | 14.41 / 14.99 | 1484.1 |
| English BMW, audited PR | 2 / 0 | 6.506 | 29.31 / 29.63 | 1496.8 |

The mixed-PDF index has 53 chunks; within-page mixing has 45; BMW has 241.
For the two mixed cases, the reranker alone loads in a median **1.05–1.06 s**;
warm `predict` takes **754 ms** / **1246 ms** respectively. Native source-language
questions give 20 / 30 pairs for these particular requests; the 60-pair bound is
exercised by the adversarial test, not claimed as a measured worst-case latency.
Language inspection takes median 80.5 / 10.2 / 199.7 ms for mixed PDF / within-page
mixing / BMW. BMW never loads the extra model. Its warm cost approximately doubles
because it embeds RU and EN; its slightly lower cold measurement is not evidence
of a speedup. Cold totals vary with initialization and filesystem cache; complete
ranges, stage timings and all 18 raw runs are in the
[performance artifact](multilingual-retrieval-performance.v1.json).

The local reranker is a material CPU/memory cost despite the bounded pool: warm
mixed retrieval is about **74× / 119×** the main dense path on these small corpora,
with maximum observed peak RSS differences of **284 / 365 MiB**. These figures
do not establish a latency bound for larger corpora, longer queries, different
thread counts or the maximum 60 pairs. Candidate admission remains bounded;
there is no silent performance-equivalence claim or speculative replacement.

### Reproduce performance measurements

Use [measure_multilingual_retrieval.py](measure_multilingual_retrieval.py), the
same private indexes from the Bosch reproduction and the unchanged BMW index.
Create a questions JSON object from `cases[id=overheat].questions` in the Bosch
dataset or `cases[id=pressure_label].questions` in the BMW dataset. For example:

```bash
mkdir -p /private/tmp/pr12-main
git archive 522a2afafaee7583725e2bc716eb9e97535960b0 src | tar -x -C /private/tmp/pr12-main
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  GROQ_API_KEY= LANGSMITH_TRACING=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv/bin/python docs/benchmarks/measure_multilingual_retrieval.py \
  --variant pr --source-dir "$PWD/src" --index-dir /path/to/index-mixed_pdf \
  --questions /path/to/overheat-questions.json --warm-runs 7 \
  --output /private/tmp/mixed-pr-1.json
```

For main, replace `--variant pr --source-dir "$PWD/src"` with
`--variant main --source-dir /private/tmp/pr12-main/src` and change the output
filename. Run three separate processes per variant/corpus, serially, alternating
the variant order. The source hashes in the artifact identify the audited local
implementation without requiring a merge or release.
