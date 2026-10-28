# Benchmark and historical evidence inventory

`benchmarks/` contains versioned evaluation inputs; `docs/benchmarks/` contains
published summaries and three frozen numeric artifacts. Keep their paths and
bytes stable: tests and published results depend on them. New runs belong in
ignored `output/`, with reviewed aggregate results published as Markdown.

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

The eight files have distinct contents and roles; none is a redundant run copy.
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
public docs, tests, all eight benchmark JSON files, the synthetic corpus and
frontend sources/lockfile. The frontend is built separately. See
[package checks](../testing.md#packaging-and-publication) and the
[sdist allowlist](../../pyproject.toml).
