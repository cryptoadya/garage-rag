# Evaluation

The experiments separate retrieval quality, answer validation and workflow behavior.
Frozen questions and numeric results are versioned; new runs go under ignored
`output/`. Source PDFs, indexes, model weights and private captures are not distributed.

## Results and methods

| Report | Question | Main observation |
| --- | --- | --- |
| [Retrieval](reports/retrieval.md) | How do dense, lexical, hybrid and reranked search compare? | BGE-M3 improves DE/RU recall on one held-out manual split at higher local cost |
| [Answering](reports/answering.md) | What do auditing, recovery and autonomous search add? | Selected recovery works; no general audit or Agentic advantage is established |
| [Multilingual retrieval](reports/multilingual.md) | How should original and source-language queries be combined? | Expansion restores BMW recall; bounded reranking improves the Bosch mixed-language matrix with substantial CPU cost |

Measurements are historical observations of the identified implementations, not
scores for every later commit. Labels and translations are agent-authored; there
is no independent human or vehicle validation. Scripted tests establish software
contracts, not real-model quality. See each report for denominators and limits.

## Inputs

Frozen inputs live in `datasets/`; evaluator defaults and reproduction commands
use those paths.

| Dataset | Contents |
| --- | --- |
| [BMW baseline](datasets/bmw_e39_2002.v1.json) | 18 answerable needs in EN/DE/RU, separate negatives and diagnostic probes |
| [BMW development](datasets/bmw_e39_2002.phase2b-dev.v1.json) | 12 needs used to select a retrieval candidate |
| [BMW held-out](datasets/bmw_e39_2002.phase2b-heldout.v1.json) | 12 disjoint needs for the selected comparison |
| [Aurora answering](datasets/aurora-grounded-answering.v1.json) | 14 fictional needs for Direct/Grounded evaluation |
| [Aurora comparative](datasets/aurora-comparative.v1.json) | Seven fictional needs for the three-mode comparison |
| [Bosch C3](datasets/bosch_c3_multilingual.v1.json) | 12 EN/DE/RU question triples with source-page and chunk anchors |

The original [Aurora corpus fixture](../tests/fixtures/corpus.json) generates
synthetic test PDFs. Real-manual datasets identify exact source hashes; another
edition requires new reviewed labels rather than bypassing identity checks.

## Frozen results

| Artifact | Evidence |
| --- | --- |
| [BMW query comparison](results/multilingual-ru-en.v1.json) | All 42 paired rankings |
| [Bosch query comparison](results/multilingual-c3.v2.json) | All 288 original, previous, pre-audit and corrected rankings |
| [Retrieval performance](results/multilingual-retrieval-performance.v1.json) | All 18 fresh-process measurements and implementation hashes |

Keep the six inputs and three current result files unchanged; use a new version
for revised labels or published measurements. Earlier baseline, vector-audit and
profile-comparison results remain available at fixed Git commits linked from the
[retrieval report](reports/retrieval.md). Filenames retain their original provenance.

## Reproduction

Run commands from the repository root after installing the project. The
[reports](reports/retrieval.md#reproduction) document required PDFs, cached models
and private indexes. The standalone scripts are:

- [compare_multilingual_queries.py](scripts/compare_multilingual_queries.py): frozen BMW RU/EN comparison.
- [compare_multilingual_c3.py](scripts/compare_multilingual_c3.py): construct private Bosch layouts and compare retrieval.
- [measure_multilingual_retrieval.py](scripts/measure_multilingual_retrieval.py): measure one fresh process without vector/score caches.

Every script accepts `--help`. Use a fresh explicit output path; inspect results
before publishing. No new provider call is needed to inspect retained results or
run [offline tests](../docs/testing.md). Result metadata retains paths recorded
by historical runs; use `datasets/` for current input files. Historical live answers
require private captures for exact review; a fresh clone cannot reproduce them
from public data alone.
