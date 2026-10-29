# Benchmark and historical evidence inventory

`benchmarks/` contains versioned evaluation inputs; `docs/benchmarks/` contains
published summaries and frozen numeric comparison artifacts. Keep retained inputs
and rankings stable: tests and published results depend on them. New runs belong
in ignored `output/`; explicitly allowlist reviewed, source-text-free evidence.

## Retained JSON

| Artifact | Why it stays |
| --- | --- |
| [BMW E39 baseline dataset](../../benchmarks/bmw_e39_2002.v1.json) | Original 18-need retrieval labels, corpus identity and evaluation regression coverage |
| [Phase 2B development dataset](../../benchmarks/bmw_e39_2002.phase2b-dev.v1.json) | Frozen development selection used to choose the retrieval candidate |
| [Phase 2B held-out dataset](../../benchmarks/bmw_e39_2002.phase2b-heldout.v1.json) | Disjoint final evaluation and the subsequent two-need real-corpus comparison |
| [Aurora Grounded dataset](../../benchmarks/aurora-grounded-answering.v1.json) | Fictional EN/DE/RU inputs for answer evaluation, review packets and tests |
| [Aurora comparative dataset](../../benchmarks/aurora-comparative.v1.json) | Fictional three-mode inputs for comparative and Agentic regression tests |
| [Phase 2A baseline results](phase2a-baseline.v1.json) | Published per-query ranks, timings, parameters and corpus/model bindings |
| [Phase 2A tire audit](phase2a-tire-audit.v1.json) | Vector audit method and chunk/model identities supporting the baseline analysis |
| [Phase 2B results](phase2b-results.v1.json) | Published candidate decision, comparisons and independent-process repeatability checks |
| [Bosch C3 dataset](../../benchmarks/bosch_c3_multilingual.v1.json) | Frozen EN/DE/RU questions, source hash/pages and chunk evidence anchors |
| [BMW multilingual results](multilingual-ru-en.v1.json) | All 42 paired rankings for old English, original Russian and expanded queries |
| [Bosch multilingual results](multilingual-c3.v2.json) | All 288 paired original, previous, pre-audit and final rankings; supersedes duplicate v1 |
| [Multilingual performance](multilingual-retrieval-performance.v1.json) | All 18 uncached process measurements and implementation hashes |

These twelve files have distinct contents and roles; none is a redundant run copy.
The separate [Aurora corpus fixture](../../tests/fixtures/corpus.json) generates
synthetic PDFs for offline tests. BMW PDFs, indexes and model weights are not
distributed; reproduction requires the permitted source PDF and pinned models.

## Reports and reproduction

| Guide | Retained purpose |
| --- | --- |
| [Phase 2A retrieval](phase2a-retrieval.md) | Corpus checks, baseline methodology, commands and limitations |
| [Phase 2B retrieval](phase2b-retrieval.md) | Development/held-out split, profile tradeoffs and reproduction commands |
| [Grounded answer evaluation](grounded-answering.md) | Evaluation/review contracts and consolidated Phase 3 observations |
| [Phase 3 archive inventory](phase3-archive.md) | Hashes and restoration instructions for previously removed public artifacts and private captures |
| [Three-mode comparison](phase4b-comparison.md) | Consolidated results, failures, limits and immutable source-commit links |
| [Frozen real-corpus protocol](phase4b-real-live-protocol.md) | Preregistered selection, corpus bindings, budgets and review criteria; historical proposal |
| [BMW multilingual comparison](multilingual-ru-en.md) | RU→EN regression, corrected results and reproduction with existing datasets |
| [Bosch multilingual comparison](multilingual-c3.md) | Final retrieval and performance results, limits and reproduction; consolidates the PR #12 audit |

These reports provide different evidence. The Phase 4B report already replaces
the long intermediate narrative with a summary and immutable Git links. The
protocol preserves what was fixed before generation; the archive inventory
preserves provenance for older removals. Removing either would lose context
needed to interpret the published results. Historical captures remain private;
a fresh clone cannot reproduce their exact answers from public artifacts alone.
This inventory does not revalidate the external private archives.

## Repository and package boundaries

- `src/garage_rag/`: Python CLI/API and retrieval/generation implementation.
- `frontend/`: React UI, its tests, Vite configuration and npm lockfile.
- `tests/`: offline Python contracts and synthetic fixtures.
- `docs/`: usage, architecture, providers, testing and the evidence listed above.
- `.github/`: CI; `AGENTS.md`: contributor release instructions.
- `data/`, `output/`, caches, environments and builds: ignored local material.

The Python wheel contains the runtime package. The sdist also contains the
public docs, tests, all twelve benchmark JSON files, the synthetic corpus and
frontend sources/lockfile. The frontend is built separately. See
[package checks](../testing.md#packaging-and-publication) and the
[sdist allowlist](../../pyproject.toml).

The three multilingual scripts stay separate because they measure different
properties: `compare_multilingual_queries.py` reuses the frozen BMW index,
`compare_multilingual_c3.py` builds the Bosch PDF layouts and evaluates retrieval,
and `measure_multilingual_retrieval.py` measures one uncached fresh process.
Their reports document private inputs and commands; no source PDFs, indexes,
weights or local run directories are distributed.
