# Testing and result reporting

## Automated checks

Install development dependencies and run the same commands as
[GitHub Actions](https://github.com/cryptoadya/garage-rag/actions/workflows/ci.yml):

```bash
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
```

For an explicitly offline test run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENROUTER_API_KEY= python -m pytest
```

Tests generate PDFs from the original,
fictional [Aurora corpus](../tests/fixtures/corpus.json). A local tokenizer and
deterministic embeddings keep tests independent of BMW material, Hugging Face
downloads and live API calls. Qdrant persistence and retrieval are real; remote
model replies are controlled fixtures, and SDK requests use mock transport.

Coverage includes extraction/page provenance, cleanup, token limits and overlap,
embedding prefixes, indexing idempotency, interrupted-revision recovery,
configuration isolation, bounded context, source/quote validation, EN/DE/RU
selection and overrides, refusals, sanitized errors, zero-price routing, output
budgets and truncation handling. These checks establish software contracts,
not E5 ranking quality or remote-model reasoning quality. CI uses Python 3.12
on Linux; local acceptance has also used Python 3.12 on macOS arm64 CPU. MPS
runtime behavior has not been validated.

## Manual smoke checks

Follow [Quick start](../README.md#quick-start) with a permitted English PDF and
cached E5 weights. Repeat the identical index command to check reuse, inspect
`stats`, then inspect `search --json` results before generating answers.

With a local OpenRouter key, run the README's EN/DE/RU oil-level questions
against the BMW owner's manual, or equivalent supported questions against your
own document. Use `--json` to check `supported`, `language`, `reason` and local
PDF references. Compare each English quote with the sent source passage and
review whether the claim actually follows from it. Also ask an unsupported
repair-specification question, such as a cylinder-head bolt torque absent from
that manual, and check for refusal. An explicit language override should change
the answer language without translating evidence.

Limited live checks have produced citation-validated EN/DE/RU answers, including
from `nvidia/nemotron-3.5-super-vl:free`. This is an observation, not a guarantee
of continuing model availability or quality. Free providers can rate limit,
time out, emit malformed JSON or exhaust the reasoning budget. Keep response
captures private and retry manually; do not enable paid fallback to obtain a
passing result. A smoke check does not establish recall, abstention accuracy,
hallucination rates or representative latency.

## Documentation and benchmarks

Keep the README focused on current user-facing behavior and onboarding. Record
significant changes in [CHANGELOG](../CHANGELOG.md) and Git; put durable technical
instructions in thematic documentation. Do not append test-run diaries, repair
narratives or individual API-call reports to the README.

Future benchmarks should be separate reports under `docs/benchmarks/`, with a
clear question, permitted dataset and provenance, reviewed labels, metrics,
configuration/model revisions, software and hardware versions, exact commands,
cache state, repeated measurements and uncertainty or failure cases. Separate
retrieval evaluation from generation and citation checks, and distinguish
measured results from conclusions. Publish only permitted aggregate results;
exclude copyrighted passages, credentials and private model replies.

Phase 2A provides a reproducible [offline retrieval benchmark](benchmarks/retrieval-evaluation.md)
and a [measured EN/DE/RU baseline](benchmarks/phase2a-report.md).
Do not treat one successful query, one refusal or an isolated timing as a
benchmark. Before publishing artifacts,
inspect `git status` and `git ls-files`; ignored files already in Git history
remain public. Source-distribution packaging has an explicit file allowlist;
include public documentation in that allowlist when adding new document paths.

Phase 2B adds synthetic checks for pinned profile isolation, active-only BM25,
deterministic rank fusion, finite reranker scores, strict chunk evidence labels,
disjoint datasets and EN/DE/RU `ask` compatibility. Model-quality comparisons
remain explicit local experiments; see the [Phase 2B report](benchmarks/phase2b-retrieval.md).
