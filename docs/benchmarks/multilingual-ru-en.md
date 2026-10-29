# Matched RU→EN query comparison — 2026-10-29

English-only query replacement was wrong for multilingual PDFs. Removing its
translation step entirely also degraded retrieval on the existing English BMW
manual. The corrected path retains the original question and adds only the
source-language queries needed by the locally observed corpus.

## Protocol

- Real 2002 BMW E39 owner manual, 186 physical PDF pages and 241 existing chunks;
  SHA-256 `9faf9c33f9958f037c77032e514295378ddb490217c1d37015d6ca0fb48b4574`.
- Unchanged baseline index, pinned multilingual-e5-small, CPU, existing prefixes,
  384/48 token chunks. Actual source-text hints classify this corpus as English,
  despite its old `language: unknown` metadata. No document re-embedding/migration.
- All 42 answerable questions from the existing legacy, development and held-out
  datasets. Same Russian question, existing English counterpart and agent-reviewed
  gold physical pages in every paired comparison. Negative cases are not scored.
- **Old EN:** dense retrieval with the frozen English counterpart, representing
  the old translated-query retrieval stage. **PR original RU:** original-only
  retrieval at `e53c1e7`. **Corrected:** actual `AnswerSearch` / `Index.search_many`
  over both original RU and frozen EN, maximum-cosine merge and chunk deduplication.
- One local batch per corrected retrieval round. Collect top 20 and score first
  relevant chunk at its original rank. Hit@k means at least one gold page appears;
  it is not proof that a complete answer or safe table interpretation is possible.
- **Zero Groq calls.** English queries are existing authored equivalents, not
  fresh samples from the old or new translation prompt. This isolates retrieval
  routing/ranking and does not measure provider translation quality or end-to-end
  answer success. One frozen counterpart (`pressure_table`) introduces Latin `US`
  for Russian `США`; the unchanged identifier guard rejects that literal output.
  Its ranking remains a retrieval-only measurement, not a claim of gate acceptance.
- Page 137's hood-release lever was visually checked against the original PDF;
  page 26's inflation-sticker location was checked with independent Poppler text.

## Results

| Retrieval path | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| Old EN | 30/42 | 40/42 | 41/42 | 0.828798 |
| PR original RU | 11/42 | 21/42 | 25/42 | 0.371259 |
| Corrected original + EN | 30/42 | 40/42 | 41/42 | 0.828798 |

Excluding the one counterpart rejected by the unchanged identifier guard leaves
41 accepted frozen equivalents: old EN and corrected search both hit 39/41 at
five, while original RU hits 20/41.

Original-only lost 19 old top-five hits. The corrected path loses **zero** of
those old hits, with identical aggregate metrics. For both the inflation sticker
(page 26) and hood release (page 137), the relevant page moves from outside RU
top 20 to corrected rank 1, matching the English query.

The two remaining top-five misses also existed in the English path: coolant
red-zone guidance (gold page 82) ranks 7, toolkit access (gold page 150) ranks 14.
Maximum-cosine merging is a pragmatic choice on this measured corpus; scores are
not calibrated relevance probabilities. Finite top-k, context budgets, short
queries, mixed-language distractors and language-hint errors can still miss
relevant passages. These reused, agent-authored cases are not new independent
held-out evidence, and this single manual cannot establish universal EN/DE/RU
recall or superiority of one embedding profile.

## Reproduce

With the permitted original PDF, unchanged baseline index, installed project and
cached pinned model, run from the repository root:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  LANGSMITH_TRACING=false .venv/bin/python docs/benchmarks/compare_multilingual_queries.py \
  --output /private/tmp/multilingual-ru-en.json
```

The [script](compare_multilingual_queries.py) verifies source hash, single-document
index and unchanged manifest. The [complete ranking artifact](multilingual-ru-en.v1.json)
contains dataset/case IDs, physical pages, chunk indices, raw cosine scores and
paired metrics, with no PDF text, API keys or private file paths. The source PDF
and index remain private and are not distributed.
