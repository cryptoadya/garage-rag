"""Actual EN/DE/RU Bosch PDF retrieval matrix; frozen queries, zero live Groq calls."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

from garage_rag.answering import run_answer
from garage_rag.embeddings import E5Embeddings
from garage_rag.index import Index
from garage_rag.ingestion import clean_text
from garage_rag.profiles import RERANK_MODEL, RERANK_REVISION
from garage_rag.retrieval import AnswerSearch, LocalReranker, Retriever

LANGS = ('de', 'en', 'ru')


def norm(text):
    return ' '.join(clean_text(text).split()).casefold()


def rank_metrics(hits, gold):
    rank = next((i for i, h in enumerate(hits, 1) if Retriever.key(h) in gold), None)
    return {**{f'hit@{k}': int(rank is not None and rank <= k) for k in (1, 5, 10)},
            'mrr@10': 1 / rank if rank is not None and rank <= 10 else 0}


def make_pdfs(reader, root, sections):
    ranges = {lang: list(range(start, end + 1)) for lang, (start, end) in sections.items()}
    specs = {lang: [[(lang, p)] for p in ranges[lang]] for lang in LANGS}
    specs['mixed_pdf'] = sum((specs[lang] for lang in LANGS), [])
    specs['bilingual_de_ru'] = specs['de'] + specs['ru']
    specs['partitioned'] = [[(LANGS[i % 3], ranges[LANGS[i % 3]][i])] for i in range(10)]
    specs['mixed_within_page'] = [
        [(lang, ranges[lang][i]) for lang in LANGS] for i in range(10)
    ]
    mapping = {}
    for name, pages in specs.items():
        writer = PdfWriter()
        for components in pages:
            if len(components) == 1:
                writer.add_page(reader.pages[components[0][1] - 1])
            else:
                width = float(reader.pages[components[0][1] - 1].mediabox.width)
                height = float(reader.pages[components[0][1] - 1].mediabox.height)
                page = PageObject.create_blank_page(width=width * len(components), height=height)
                for i, (_, physical_page) in enumerate(components):
                    page.merge_transformed_page(reader.pages[physical_page - 1],
                                                Transformation().translate(tx=width * i))
                writer.add_page(page)
        path = root / f'{name}.pdf'
        writer.write(path)
        mapping[name] = {path.name: pages}
    mapping['mixed_collection'] = {
        f'{lang}.pdf': specs[lang] for lang in LANGS
    }
    return mapping


class CachedQueries(E5Embeddings):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def embed_query(self, text):
        if text not in self.cache:
            self.cache[text] = super().embed_query(text)
        return self.cache[text]


class CachedReranker:
    """Cache exact repeated scoring pairs in this offline comparison only."""

    def __init__(self):
        self.model = None
        self.cache = {}

    def can_score(self, pairs):
        if self.model is None:
            self.model = LocalReranker()
        return self.model.can_score(pairs)

    def predict(self, pairs, **kwargs):
        missing = list(dict.fromkeys(pair for pair in pairs if pair not in self.cache))
        if missing:
            scores = self.model.predict(missing, **kwargs)
            self.cache.update(zip(missing, map(float, scores), strict=True))
        return [self.cache[pair] for pair in pairs]


class FrozenModel:
    def __init__(self, questions):
        self.questions = questions
        self.targets = []

    def invoke(self, messages):
        payload = json.loads(messages[1].content)
        if 'languages' in payload:
            self.targets.append(payload['languages'])
            value = {'queries': [{'language': lang, 'query': self.questions[lang]}
                                 for lang in payload['languages']]}
        else:
            # Isolate retrieval; no scripted claim can turn a miss into an answer.
            value = {'supported': False, 'claims': []}
        return SimpleNamespace(content=json.dumps(value, ensure_ascii=False),
                               usage_metadata={'input_tokens': 0, 'output_tokens': 0})


def evaluate(pdf, root, dataset, previous_merge=False):
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != dataset['source']['sha256']:
        raise ValueError('Unexpected source PDF')
    root.mkdir(parents=True, exist_ok=True)
    mapping = make_pdfs(PdfReader(pdf), root, dataset['source']['sections'])
    embeddings = CachedQueries()
    reranker = CachedReranker()
    rows, corpora = [], {}
    for corpus, documents in mapping.items():
        directory = root / f'index-{corpus}'
        with Index(directory) as index:
            for document in documents:
                # Deliberately stale metadata; routing must use actual passage text.
                index.ingest(root / document, embeddings, embeddings.tokenizer,
                             metadata={'language': 'en'})
        with Index(directory) as index:
            manifest = index.manifest_path.read_bytes()
            search = (AnswerSearch(index, embeddings) if previous_merge else
                      AnswerSearch(index, embeddings, reranker=reranker))
            languages = search.source_languages
            golds, native_languages = {}, {}
            for case in dataset['cases']:
                golds[case['id']], native_languages[case['id']] = set(), set()
                for hit in search._hits:
                    metadata = hit.document.metadata
                    components = documents[metadata['document']][metadata['pdf_page'] - 1]
                    for lang, original_page in components:
                        evidence = case['evidence'][lang]
                        if original_page == evidence['original_pdf_page'] and all(
                            norm(anchor) in norm(hit.document.page_content)
                            for anchor in evidence['all_anchors']
                        ):
                            golds[case['id']].add(Retriever.key(hit))
                            native_languages[case['id']].add(lang)
                if not golds[case['id']]:
                    raise ValueError(f'No independently anchored gold chunk: {corpus}/{case["id"]}')
            corpora[corpus] = {'languages': languages, 'documents': len(documents),
                              'pages': index.stats()['pages'], 'chunks': index.stats()['chunks']}
            for case in dataset['cases']:
                gold = golds[case['id']]
                native = {lang: search(case['questions'][lang], 20)
                          for lang in native_languages[case['id']]}
                best_native = max((rank_metrics(h, gold) for h in native.values()),
                                  key=lambda m: (m['hit@5'], m['mrr@10']))
                for question_language in LANGS:
                    question = case['questions'][question_language]
                    original = search(question, 20)
                    delivered = []
                    def capture(queries, k, search=search, delivered=delivered,
                                index=index, embeddings=embeddings):
                        if previous_merge and not isinstance(queries, str):
                            hits = index.search_many(queries, embeddings, k)
                        else:
                            hits = search(queries, k)
                        delivered.extend(hits)
                        return hits
                    model = FrozenModel(case['questions'])
                    result = run_answer(question, capture, model, mode='direct', top_k=20,
                                        source_languages=languages)
                    if result.answer.reason == 'query_translation_error':
                        raise ValueError(
                            f'Frozen translation rejected: {case["id"]}/{question_language}'
                        )
                    if result.error or result.trace.search_calls != 1 or len(model.targets) > 1:
                        raise ValueError('Unexpected retrieval path or extra preparation call')
                    rankings = {'original': original, 'expanded': delivered}
                    rows.append({
                        'corpus': corpus, 'id': case['id'], 'question_language': question_language,
                        'targets': model.targets[0] if model.targets else [],
                        'gold': [[k[1], k[2], k[3]] for k in sorted(gold)],
                        'metrics': {**{name: rank_metrics(hits, gold)
                                       for name, hits in rankings.items()},
                                    'best_native': best_native},
                        'rankings': {name: [[h.document.metadata['document'],
                                            h.document.metadata['pdf_page'],
                                            h.document.metadata['chunk_index'], round(h.score, 8)]
                                           for h in hits] for name, hits in rankings.items()},
                    })
            if manifest != index.manifest_path.read_bytes():
                raise ValueError('Reopened index manifest changed')
        print(corpus, corpora[corpus], flush=True)
    return {'schema_version': 1, 'pdf_sha256': dataset['source']['sha256'],
            'provider_calls': 0,
            'strategy': 'maximum_cosine' if previous_merge else 'mixed_local_relevance',
            'reranker': None if previous_merge else {'model': RERANK_MODEL,
                'revision': RERANK_REVISION, 'scope': 'expanded mixed-language batches only'},
            'cache': 'Repeated exact query vectors and relevance pairs, offline measurement only',
            'gold_unit': 'source-page and case-specific anchored chunk',
            'corpora': corpora, 'n': len(rows), 'rows': rows}


def summarize(result):
    grouped = defaultdict(list)
    for row in result['rows']:
        grouped[(row['corpus'], row['question_language'])].append(row)
    summary = []
    for (corpus, language), rows in grouped.items():
        summary.append({'corpus': corpus, 'question_language': language, 'n': len(rows),
                        'hits_at_5': {name: sum(r['metrics'][name]['hit@5'] for r in rows)
                                      for name in ('original', 'expanded', 'best_native')},
                        'lost_original_at_5': [r['id'] for r in rows if
                            r['metrics']['original']['hit@5']
                            and not r['metrics']['expanded']['hit@5']],
                        'lost_native_at_5': [r['id'] for r in rows if
                            r['metrics']['best_native']['hit@5']
                            and not r['metrics']['expanded']['hit@5']]})
    result['summary'] = summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-merge', action='store_true',
                        help='Reproduce the previous maximum-cosine merge without reranking')
    args = parser.parse_args()
    dataset = json.loads(Path('benchmarks/datasets/bosch_c3_multilingual.v1.json').read_text())
    result = evaluate(args.pdf, args.work_dir, dataset, args.previous_merge)
    summarize(result)
    # Keep complete rankings inspectable without exporting any source passage text.
    rows = result.pop('rows')
    prefix = json.dumps(result, ensure_ascii=False, indent=2)[:-2]
    args.output.write_text(prefix + ',\n  "rows": [\n' + ',\n'.join(
        '    ' + json.dumps(row, ensure_ascii=False, separators=(',', ':')) for row in rows
    ) + '\n  ]\n}\n')


if __name__ == '__main__':
    main()
