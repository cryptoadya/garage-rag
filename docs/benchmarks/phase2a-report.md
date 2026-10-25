# Phase 2A: E39 retrieval baseline

The unchanged E5 retriever finds an accepted answer page in its default first
five results for all English needs, but only 13/18 German and 11/18 Russian
needs. Specific multilingual weaknesses are reproducible. This baseline
identifies them without tuning the pipeline or selecting a new model.

Measured on 2026-10-25 using the [v1 dataset](../../benchmarks/bmw_e39_2002.v1.json),
[full numeric results](phase2a-baseline.v1.json) and the
[reproduction/methodology guide](retrieval-evaluation.md). Source: 186 physical
PDF pages, 241 active chunks; English US-market 525i/530i/540i manual, part
01 41 0 156 213. PDF and dataset SHA-256, index generation, source-code hashes,
configuration and software versions are embedded in the result file.

## Measured quality and latency

Each language has 18 answerable needs; accuracy is not averaged over repeated
trials or negative cases. Hit@k uses physical-page relevance at the original
chunk rank, without deduplication. MRR is truncated at ten. Latency is warm
`Index.search`, in milliseconds, over 24 questions x three passes per language
(including four negatives and two unscored tire probes).

| Language | Hit@1 | Hit@5 | Hit@10 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EN | 77.8% | 100.0% | 100.0% | 0.8796 | 12.83 | 14.78 |
| DE | 33.3% | 72.2% | 77.8% | 0.5065 | 13.01 | 16.69 |
| RU | 33.3% | 61.1% | 66.7% | 0.4398 | 12.85 | 15.29 |

CPU, Darwin 27.0.0 arm64, Python 3.12.0; Torch 2.14.1, NumPy 2.5.4,
Qdrant client 1.20.0, Sentence Transformers 6.2.0, Transformers 5.20.2.
E5 revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, normalized 384-dimensional
vectors, chunk budget 384 and overlap 48, `query: ` / `passage: ` prefixes.
Torch used four intra-op and eight inter-op threads. Exact Apple chip model,
RAM, thermal state and competing system load were not measured; timings are
not portable hardware claims. Imports/PDF checks/file I/O are excluded.

Cached-model loading took 7.09 s, index opening 34.1 ms, and
the excluded first warm-up search 175.4 ms. Neither is a cold-download measurement.

All 72 query rankings were stable across three passes. A second independent
process with the same seed reproduced all 216 ordered top-ten page/chunk lists
and their scores exactly; latency varied. The published JSON is the final
complete run, not a selected best trial. No crashes, empty positive results
or invalid embeddings occurred. This is repeatability on this local runtime,
not a guarantee of bitwise identity on other platforms.

## Gold-page provenance and every information need

**These are agent-verified PDF labels, not human-reviewed gold.** Before the
first Phase 2A ranking run, Codex independently read original-PDF text through
Poppler and inspected rendered images of every accepted source page: 26, 27,
43, 45, 54, 60, 65, 78, 80, 97, 99, 106, 112, 116, 140, 142, 159, 160 and 162.
No retrieved result was used to assign an answerable gold page. The two cabin
filter pages support the same symptom for different air-conditioning variants.
The dataset records the scope of each judgment and that no human verified it.
No unverified positive labels enter these metrics. English, German and Russian
questions were authored and equivalence-checked by the same agent; independent
native-speaker review remains outstanding.

The following are first relevant **chunk** ranks, not page-deduplicated ranks.
`>10` means the accepted page was absent from the evaluated top ten; it does
not establish absence from the rest of the index.

| Information need ID | Accepted PDF pages | EN rank | DE rank | RU rank |
| --- | --- | ---: | ---: | ---: |
| pressure_label | 26 | 1 | >10 | >10 |
| pressure_table | 27 | 2 | 2 | 3 |
| alarm_sensors | 43 | 1 | 5 | 6 |
| rear_window_lock | 45 | 1 | 2 | >10 |
| seat_memory | 54 | 1 | 2 | 1 |
| airbag_lamp | 60 | 1 | 6 | >10 |
| key_identification | 65 | 1 | 1 | 1 |
| cruise_minimum | 78 | 2 | >10 | >10 |
| ice_warning | 80 | 1 | 1 | 2 |
| dsc_startup | 97 | 2 | 2 | 3 |
| rdc_reset | 99 | 1 | 1 | 1 |
| cabin_filter | 106, 112 | 1 | 4 | 3 |
| flashlight | 116 | 1 | 2 | 4 |
| oil_level | 140 | 1 | 1 | 1 |
| coolant_float | 142 | 1 | 1 | 1 |
| wheel_torque | 159 | 1 | >10 | >10 |
| glovebox_fuses | 160 | 3 | >10 | >10 |
| jump_positive | 162 | 1 | 1 | 1 |

Questions, qualifications and review notes for each ID are in the public
JSON dataset. No source passages, PDF images or index artifacts accompany it.

## Failure analysis

There are four DE and six RU misses at ten:

| Language | Missed need | Top returned physical page |
| --- | --- | ---: |
| DE | pressure_label | 165 |
| DE | cruise_minimum | 4 |
| DE | wheel_torque | 176 |
| DE | glovebox_fuses | 126 |
| RU | pressure_label | 20 |
| RU | rear_window_lock | 61 |
| RU | airbag_lamp | 96 |
| RU | cruise_minimum | 16 |
| RU | wheel_torque | 158 |
| RU | glovebox_fuses | 34 |

Several error types recur:

- **Adjacent topic without the required detail.** English cruise control ranks
  page 79 ahead of the minimum-speed criterion on 78. ASC/DSC controls on 98
  precede automatic-startup behavior on 97 in all three languages. English
  glove-compartment storage on 116 precedes actual fuse access on 160. German
  wheel-torque retrieval starts with the alphabetical index on 176; Russian
  starts with the preceding wheel-change procedure on 158, missing torque on
  159. Topic similarity alone does not satisfy the need.
- **Related feature substitution.** German rear-window locking starts with
  child-restraint information on 61 before reaching the safety switch on 45.
  Russian never reaches 45 in ten. The Russian airbag warning question starts
  with parking-distance control on 96 and misses the lamp-fault criterion on
  60. German reaches that page only at rank six.
- **Cross-language specificity loss.** DE/RU sticker and fuse queries miss the
  accepted page despite the same English information needs succeeding. Their
  wrong top results sometimes lie far from the topic. This is observed
  ranking behavior; the experiment does not isolate whether vocabulary,
  translation phrasing, model representation or short generic chunks is the
  dominant cause. It does not justify a claim that either language is
  generally unsupported by E5.
- **Equipment/body/market mismatch.** The English specific 530i pressure query
  ranks the adjacent pressure page 28 first, while its accepted table is on
  27. Preserving the 530i qualification prevents an optimistic false hit.

These categories are source-based qualitative interpretation, not additional
annotator-validated labels or evidence of causality. Page-level metrics may
still overcount useful retrieval if the returned chunk on a gold page lacks
the requested sentence. No chunk-level answer-sufficiency review was performed.

## Tire-pressure investigation

The known weakness reproduces for both a full question about the pressure
sticker and a short heading-like query. The broad question is deliberately
unscored: pressure depends on model, body style, market, tire specification
and load, so a single unqualified numeric answer would be misleading.

For scored cases the final column shows accepted page:first-chunk-rank. For
unscored broad/short probes it shows only where table/location pages 26-29
appear; these appearances are not correctness scores.

| Query ID | Language | Top five physical pages | Accepted/diagnostic page:rank |
| --- | --- | --- | --- |
| pressure_label | EN | 26, 28, 29, 99, 99 | 26:1 |
| pressure_label | DE | 165, 59, 123, 137, 147 | none in 10 |
| pressure_label | RU | 20, 165, 98, 147, 100 | none in 10 |
| pressure_table | EN | 28, 27, 29, 134, 136 | 27:2 |
| pressure_table | DE | 2, 27, 28, 29, 165 | 27:2 |
| pressure_table | RU | 2, 28, 27, 168, 29 | 27:3 |
| rdc_reset | EN | 99, 99, 100, 100, 26 | 99:1, 99:2 |
| rdc_reset | DE | 99, 99, 100, 100, 96 | 99:1, 99:2 |
| rdc_reset | RU | 99, 99, 100, 96, 100 | 99:1, 99:2 |
| tire_broad | EN | 26, 99, 100, 133, 133 | 26:1 |
| tire_broad | DE | 99, 84, 133, 136, 25 | none in 10 |
| tire_broad | RU | 20, 133, 99, 130, 100 | none in 10 |
| tire_short | EN | 28, 26, 29, 27, 99 | 28:1, 26:2, 29:3, 27:4 |
| tire_short | DE | 1, 99, 104, 107, 84 | none in 10 |
| tire_short | RU | 165, 1, 182, 136, 147 | none in 10 |

The source and index checks narrow the failure:

1. Original PDF page 26 directly locates the vehicle-specific sticker and
   states the temperature qualification. Pages 27-29 contain distinct
   tables. Page 27 is the correct 530i sedan table; a generic pressure-monitor
   passage on 99 is not its replacement.
2. Read-only Qdrant inspection found page 26 in one 1,158-character/267-token
   chunk and page 27 in one 761-character/304-token chunk. Both fit the
   unchanged 384-token budget and cover the full cleaned page text. Pages
   28 and 29 are also indexed. The failure is not missing page ingestion or
   a split of these particular pages at the chunk boundary.
3. Page 165, returned first for the DE sticker query and RU short query, has
   only 48 characters/19 tokens of extracted navigation material. The DE
   short query retrieves the cover on page 1 (26 characters/10 tokens).
   Very short low-information pages can outrank substantive evidence.
4. Direct visual inspection of page 27 shows load pictograms and a multirow,
   multicolumn table. The text-only representation retains numbers and tire
   names but loses the visual relationship between load symbols and columns.
   Better ranking alone would not validate a generated numerical answer.
5. Adding model/market/tire identifiers in the frozen specific question finds
   page 27 at ranks 2/2/3 (EN/DE/RU), while the underspecified probes do not
   reach any of pages 26-29 in the DE/RU top ten. This is a formulation
   sensitivity observation, not a proposed production query rewrite.

A read-only [vector audit](phase2a-tire-audit.v1.json) re-embedded nine stored
chunks from pages 1, 2, 26-29, 133 and 165 using the unchanged E5 adapter.
Every stored float matched its freshly computed counterpart (maximum absolute
difference 0.0). This checks the pressure pages and prominent distractors, not
every vector in the corpus; it argues against a stale or mismatched embedding
for those specific chunks. No index values were written.

The documented source audit and the saved query rankings allow this diagnosis
to be repeated without publishing BMW text. Cross-language ranking is an
observed weakness; table-layout loss is a separate source-representation
limitation. This phase does not implement a remedy or claim a single root cause.

## Unanswerable cases and score overlap

All four negative needs in all three languages returned ten nearest chunks.
Top-one page / similarity score for each:

| Negative need | EN | DE | RU |
| --- | --- | --- | --- |
| head_bolt_torque | 168 / 0.8285 | 168 / 0.8197 | 138 / 0.8162 |
| ncs_coding | 37 / 0.8389 | 97 / 0.8071 | 96 / 0.8001 |
| carplay | 22 / 0.8197 | 162 / 0.8007 | 34 / 0.7918 |
| diesel_dpf | 24 / 0.8511 | 67 / 0.8142 | 24 / 0.8224 |

Negative top-one scores range from 0.8197-0.8511 (EN), 0.8007-0.8197 (DE),
and 0.7918-0.8224 (RU). They overlap answerable-query top-one scores; some
negative scores exceed many positive scores. No universal acceptance threshold
or calibrated confidence is established. Nonempty retrieval is expected
because the existing search has no abstention rule. This is not a measured
hallucination rate, refusal accuracy or evidence that a generated answer would
have been accepted.

Negative labels remain `agent_scope_checked`: whole-PDF lexical checks and
inspection of the closest sections support these scoped absence judgments,
but no human or exhaustive semantic search confirmed them. They remain
separate from Hit/MRR and require review before using them as a refusal test.

## Acceptance and limits

The existing 108 tests remain intact; the suite now has 130 offline cases,
including metric arithmetic, dataset guards, exclusions, real-Qdrant evaluation,
ordering, immutability and text-free outputs. Pytest and Ruff passed locally;
source and wheel builds passed and were checked for private PDF/data/env
exclusions. CI status belongs to the PR and must be assessed separately.

This is a baseline for this corpus and these frozen needs, not a production
quality certification. There are only 18 correlated information needs, one
English US-market petrol manual and one phrasing per language. Selection was
informed by known project concerns, including tire pressure and prior oil-level
smoke checks; it is not a blind test set. There is no independent human gold,
complete relevance pool, significance test, table-answer verification or
real-user latency study. Generation and its citation/refusal behavior were not
evaluated. Existing retrieval, embedding and ingestion code is unchanged.

Keep this dataset stable for later paired comparisons. Before drawing broader
conclusions, obtain independent page/translation review, add new held-out
needs and distinguish ranking, extraction/layout and answer-generation errors.
No LangGraph, hybrid search, new model or retrieval fix is part of Phase 2A.
