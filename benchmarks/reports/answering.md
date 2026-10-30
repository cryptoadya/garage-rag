# Answering evaluation

Direct, Grounded and experimental Agentic share citation validation but have
different retrieval/control paths. This report separates controlled contracts,
historical model observations and source review. It does not rank current model
quality from tests or same-model audit verdicts.

## Method and review

The [Aurora answering dataset](../aurora-grounded-answering.v1.json) contains
14 fictional needs; the [comparative dataset](../aurora-comparative.v1.json)
contains seven, each asked in EN/DE/RU. Controlled model replies and rankings
exercise missing evidence, omitted conditions/warnings, conflicts, refusals and
reference following. They are authored fixtures, not natural model behavior.

Each mode runs independently. Direct/Grounded start with the same context;
Grounded can expand once. Agentic starts empty and chooses bounded local search
before a separate writer/auditor pair. Citation validity, entailment, whole-question
completeness, vehicle applicability and warning preservation are separate checks.
A substring-valid quotation does not establish the others.

Live quality labels require explicit answer/context-hash-bound source review.
Same-model audits are workflow observations, not ground truth. Only actually sent
sources can establish context sufficiency; retrieved-but-unsent material cannot.
Provider errors are reported separately from refusals and factual judgments.
Failed/unknown-usage requests retain reservations; reported tokens are separate.

## Controlled comparison

The recorded three-mode run used 189 workflows: 63 per mode from seven needs ×
three languages × three trials. Only four needs are answerable; repeats and
translations do not create independent samples. Actual API calls: **zero**.

| Scripted observation | Direct | Grounded | Agentic |
| --- | ---: | ---: | ---: |
| Complete supported / answerable runs | 9/36 | 18/36 | 27/36 |
| Unsupported positive / positive answers | 27/45 | 9/27 | 9/36 |
| Corpus-relative false refusals / answerable runs | 9/36 | 9/36 | 0/36 |
| Correct refusals / unanswerable runs | 9/27 | 27/27 | 27/27 |
| Missed warnings / relevant positive answers | 18/18 | 9/18 | 9/18 |
| Scripted model invocations | 63 | 135 | 261 |
| Local fixture searches | 63 | 90 | 81 |

Agentic's additional complete answers repeat one controlled reference-following
case. Both audited modes still miss a deliberately accepted warning omission.
These counts measure enforced contracts and failure paths, not live retrieval,
semantic reliability, provider latency or a general Agentic advantage.

## Direct/Grounded observations — 2026-10-26

All judgments below were agent source review, without independent human labels.
These experiments predate later query-expansion and grounding changes.

| Historical experiment | Observation |
| --- | --- |
| Early provider attempts | Incomplete matrices and provider/token-limit failures; no completed quality comparison |
| Groq cold-pressure EN/DE/RU | Three supported pairs; Direct/Grounded calls 3/6, tokens 2,051/4,543; one simple need |
| Four harder English needs | Four pairs; one controlled missing-evidence recovery, three source-relative ties; calls 4/7, tokens 2,328/4,726 |
| BMW E39, E5 Small/BGE-M3 | 14 pairs; complete safely supported answers 9/4; sufficient-context false refusals 0/5; calls 14/19, tokens 36,081/48,288 |

The BMW run exposed conservative numeric false positives and a retrieved hood
passage excluded by the context cap. Four engine-table answers matched rendered
numbers but lacked safe text grounding: they were not four factually incorrect
values or evidence of semantic-audit benefit.

A selected five-pair repair regression at
[6445384](https://github.com/cryptoadya/garage-rag/commit/644538405478fa0f8faf91c6a512e590d9c046ab)
used the same first request hashes, corpus/index bindings, Groq model and
2,048-token output cap. It completed 15 attempts without HTTP errors/retries:
three former prose false positives became supported; Grounded recovered the hood
procedure after actually sending page 137; an engine-table candidate remained
blocked. Complete safely supported answers Direct/Grounded changed **3/0 → 3/4**.
The direct hood refusal remained justified by missing sent evidence.

Calls changed 5/6 → 5/10, reported tokens 13,968/15,930 → 12,474/28,255.
This preselected regression contained no genuinely erroneous audit-eligible draft;
it does not measure semantic-audit prevention or general model improvement.
Exact historical results and archive provenance remain in the
[original report](https://github.com/cryptoadya/garage-rag/blob/c29e4a4/docs/benchmarks/grounded-answering.md).
Private captures are required for exact answer review and are not distributed.

## Three-mode real-corpus comparison — 2026-10-27

The final comparison used two existing English BMW needs, one trial each, at
[implementation b41dac9](https://github.com/cryptoadya/garage-rag/commit/b41dac95d9e527a12bde621d6f7e0dd1fc30704d),
with [report 8f18901](https://github.com/cryptoadya/garage-rag/commit/8f18901950d55d30b65447a0f5681a9247aae055).
Groq `openai/gpt-oss-120b`, temperature zero, 1,024 completion tokens including
reasoning, pinned local tokenizer accounting; 22-attempt / 128,000-held-token
session caps. HTTP starts were spaced at least 61 seconds apart. Review was
separate, unblinded AI inspection of original PDF text and rendered pages,
**not independent human or vehicle validation**.

| Question / outcome | Direct | Grounded | Agentic |
| --- | --- | --- | --- |
| Hood lever location | Complete, p137 | Complete, p137 | Complete, p137 |
| Toolkit location and access | Complete, p23 | Complete, p23 | `context_budget_exhausted` |
| Complete supported answers | 2/2 | 2/2 | 1/2 |

Page 23 is valid alternative evidence to the frozen toolkit gold page 150.
No relevant warning/applicability omission was found in the five positive answers.
Direct/Grounded initial requests were identical per question; neither Grounded
run expanded. For toolkit, Agentic's second search overflowed the then-lossless
context guard before delivery. Its rejected second-search raw hits were not
captured, so their content and ranks remain unknown. No BM25 recovery occurred.

Totals: **12 API calls, seven workflow searches, 43,611 held / 27,443 actual tokens**.
Elapsed time was 676.203 seconds, including 659.448 seconds pacing and 12.098
seconds wire HTTP. Pacing dominates these timings; they do not measure intrinsic
inference speed. Two preselected needs cannot establish a general quality ranking.

Later bounded context selection, token-aware handoff and navigation hints improved
controlled replays. Two separate toolkit trials still did not answer successfully:
one explicitly refused after two dense searches; another draft was blocked by the
numeric gate and also contained a semantically wrong emergency-release step.
Neither trial is pooled into the six-row scores or evidence of live recovery.
Detailed failures, prior protocol, source bindings and implementation chronology
remain in the [historical comparison](https://github.com/cryptoadya/garage-rag/blob/c29e4a4/docs/benchmarks/phase4b-comparison.md).

## Reproduction and limits

The default evaluator is scripted and makes no provider calls:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROQ_API_KEY= \
python -m garage_rag.answer_evaluation \
  --dataset benchmarks/aurora-comparative.v1.json \
  --modes direct grounded agentic --trials 3 --seed 20261026 \
  --output output/benchmarks/comparative.json \
  --private-captures output/private/comparative.json
```

For the two-mode suite, use `benchmarks/aurora-grounded-answering.v1.json` and
`--modes direct grounded`. Use fresh output/capture paths. Existing output requires
explicit resume with matching dataset/code/settings; it is not silently replaced.
`python -m garage_rag.answer_evaluation --help` lists selection and rescoring flags.

`python -m garage_rag.review_packet --help` documents private review-packet creation.
Packets bind exact requests/replies, source sets, schemas and hashes; independent
review is not automatically completed. Public reports omit source/answer text.
Historical live experiments need their private captures and cannot be regenerated
exactly from a fresh clone. New live runs require explicit scope and
[provider controls](../../docs/providers.md#separate-resumable-comparisons).

Agentic remains experimental; Direct/Grounded are primary. Same-model audit can
share generator mistakes. Context selection cannot create missing evidence, and
numeric heuristics can over-refuse valid prose or miss subtle corruption. No
independent human review, broad quality gain, latency guarantee or production
readiness follows from these experiments.
