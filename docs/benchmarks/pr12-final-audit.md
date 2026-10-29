# PR #12 final audit — 2026-10-29

Audited published head `01c3d019c35b936491cd93aa96462624dadc5d2a` against main
`522a2afafaee7583725e2bc716eb9e97535960b0`, then applied the corrections below
locally on `adya/multilingual-pdf-fix`. No merge, tag, Release or live Groq call.

## Findings and corrections

1. **P1: valid branch top-five candidates could disappear before reranking.**
   `Index.search_many` merged up to 60 results by maximum cosine and immediately
   sliced to 20. An entire lower-cosine language branch could be lost, irrespective
   of its relevance. The new real-Qdrant adversarial test gives one branch twenty
   higher-scoring distractors and checks the other branches' fifth results reach
   ranking. All six combinations (two/three branches; final top-k 1/5/20) failed
   before correction. Mixed expanded searches now reserve each branch's first
   five results, deduplicate point IDs, fill remaining slots by maximum cosine,
   and keep maximum raw cosine and deterministic dense order for the selected
   pool. This is a candidate-admission correction; it does not guarantee final
   top-five ranking. The bound remains 20 candidates, 60 pairs, three dense queries
   in one active-filtered batch. Monolingual and original-only admission is unchanged.
2. **P2: corpus-language inspection unnecessarily built BM25 statistics.**
   Shared active-passage loading created token Counters, document frequencies,
   lengths and average length even when only language hints were requested.
   These now build only on lexical search, with their own generation invalidation.
   The regression test fails on any BM25 tokenization during language inspection,
   then exercises real lexical retrieval afterward. Existing active-generation
   replacement tests also pass. Active passages remain cached for language hints.

No new embedding model, ranking algorithm, dependency, provider request, source
rewrite, index migration or threshold change was introduced by this audit.
The inspected PR also retains original questions for writers/auditors, verbatim
source-language evidence, translation validation, numeric/citation gates, bounded
Groq preparation and the existing complete-pair 512-token guard. Oversized pairs
retain dense order; other ranking failures stop without provider retry/fallback.

## Retrieval evidence and retained artifacts

The real Bosch matrix was rerun over identical source PDF, frozen questions,
evidence anchors and all eight corpus layouts, using separate copies of private
indexes. [v1](multilingual-c3.v1.json) and both BMW benchmark artifacts are retained
unchanged. [v2](multilingual-c3.v2.json) keeps the original, maximum-cosine,
pre-audit reranker and current rankings; no labels were changed.
The separate 42-question BMW comparison was also rerun: all paired chunk
rankings and aggregate metrics remain unchanged, including 40/42 Hit@5 and zero
lost old-EN top-five hits. Its existing files therefore need no result update.

| Bosch, 288 combinations | Published PR | After audit |
| --- | ---: | ---: |
| Hit@1 | 141/288 | 138/288 |
| Hit@5 | 261/288 | 261/288 |
| Hit@10 | 282/288 | 282/288 |
| MRR@10 | 0.665505 | 0.655088 |
| Lost original-query / source-native top-five hits | 0 / 0 | 0 / 0 |

82 complete ranking rows and 15 metric rows change. Newly admitted candidates
can displace other passages; preserving branch candidates is not a universal
precision improvement. The additional top-one loss is disposal in the within-page
mixed layout, in all three question languages. The conservative anchored gold is
unchanged, including duplicate valid statements it may undercount. The existing
small-battery numeric-mode top-one error remains. These are development retrieval
measurements, not independent human gold or live answer-quality evidence.

## Validation

- `HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= .venv/bin/python -m pytest --tb=short`: **763 passed**.
- `.venv/bin/ruff check .`: passed.
- `.venv/bin/python -m build`: wheel and sdist built; `.venv/bin/python -m pip check`: passed.
- `npm test` and `npm run build` in `frontend`: **36 passed**, TypeScript/Vite build passed.
- Existing public-head [CI run 115](https://github.com/cryptoadya/garage-rag/actions/runs/40882309081): successful.
  There were no submitted GitHub reviews or inline review threads at inspection.
  That CI run covers the published head, not these local corrections.

No browser or live provider test was run in this audit; no UI code changed.
No performance acceptance threshold was supplied. The measurements below expose
the cost for review rather than asserting that any deployment budget is satisfied.

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
