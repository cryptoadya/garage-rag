"""Bounded source-aware expansion; no live provider calls or model downloads."""

import json

import pytest
from test_answering import Replies, tire_draft
from test_workflow import audit

from garage_rag.answering import run_answer
from garage_rag.index import Index


@pytest.mark.parametrize('mode', ['direct', 'grounded'])
def test_cross_language_expansion_preserves_original_and_uses_one_call(
    tmp_path, pdf_path, tokenizer, embeddings, config, mode,
):
    from garage_rag.retrieval import AnswerSearch

    question = 'Как проверить шины?'
    model = Replies({'queries': [{'language': 'en', 'query': 'How do I check tires?'}]},
                    tire_draft('Проверяйте давление на холодных шинах.'), audit())
    with Index(tmp_path / 'index', config) as index:
        index.ingest(pdf_path, embeddings, tokenizer, metadata={'language': 'ru'})
        search = AnswerSearch(index, embeddings)
        assert search.source_languages == ('en',)  # actual text, not stale metadata
        result = run_answer(question, search, model, mode=mode,
                            source_languages=search.source_languages)
    assert result.answer.supported
    assert result.trace.llm_calls == (2 if mode == 'direct' else 3)
    assert result.trace.llm_call_limit == (2 if mode == 'direct' else 5)
    assert result.trace.search_calls == 1  # one bounded Qdrant batch
    assert result.trace.input_tokens == 10 * result.trace.llm_calls
    assert json.loads(model.messages[1][1].content)['question'] == question
    assert 'PDF page 1' in result.answer.text


@pytest.mark.parametrize('target', ['en', 'de', 'ru'])
def test_same_language_or_unknown_corpus_needs_no_translation(target):
    from test_multilingual import QUESTIONS
    result = run_answer(QUESTIONS[target], lambda q, k: [], Replies(), mode='direct',
                        source_languages=(target,),
                        answer_language={'en': 'ru', 'de': 'en', 'ru': 'de'}[target])
    assert result.answer.reason == 'no_context'
    assert result.trace.llm_calls == 0


def test_mixed_corpus_translates_only_missing_languages_once():
    question = 'Как проверить шины?'
    model = Replies({'queries': [
        {'language': 'de', 'query': 'Wie prüfe ich die Reifen?'},
        {'language': 'en', 'query': 'How do I check tires?'},
    ]})
    seen = []
    result = run_answer(question, lambda q, k: seen.append(q) or [], model,
                        mode='direct', source_languages=('en', 'de', 'ru'))
    assert seen == [(question, 'Wie prüfe ich die Reifen?', 'How do I check tires?')]
    assert result.trace.llm_calls == 1
    assert len(model.messages) == 1


@pytest.mark.parametrize('value', [
    {'queries': []},
    {'queries': [{'language': 'de', 'query': 'Wie prüfe ich die Reifen?'}]},
    {'queries': [{'language': 'en', 'query': 'Как проверить шины?'}]},
    {'queries': [{'language': 'en', 'query': 'How to check BMW E46 at 20 degrees?'}]},
    {'queries': [{'language': 'en', 'query': 'How to check BMW E39 at 20 degrees?'}]},
    {'queries': [{'language': 'en', 'query': 'How to check BMW E39 ABS at -20 degrees?'}]},
    {'queries': [{'language': 'en', 'query': 'How to check BMW E39 at -20 degrees?',
                  'extra': True}]},
    RuntimeError('private-token'),
])
def test_bad_expansion_stops_without_search_or_retry(value):
    model = Replies(value)
    result = run_answer('Как проверить BMW E39 при -20 градусах?',
                        lambda q, k: pytest.fail('search'), model, mode='direct',
                        source_languages=('en',))
    assert result.answer.reason == 'query_translation_error'
    assert result.trace.llm_calls == 1 and result.trace.search_calls == 0
    assert 'private-token' not in json.dumps(result.public_dict())


def test_batch_merge_retains_translated_only_hit_and_original_hit(
    tmp_path, pdf_path, tokenizer, embeddings, config,
):
    from garage_rag.retrieval import AnswerSearch

    class DifferentQueries:
        def embed_query(self, text):
            return [1., 0., 0.] if text.isascii() else [0., 1., 0.]

    with Index(tmp_path / 'index', config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        search = AnswerSearch(index, DifferentQueries())
        assert search('Как проверить шины?', 1)[0].document.metadata['pdf_page'] == 2
        hits = search(('Как проверить шины?', 'How do I check tires?'), 4)
        assert {h.document.metadata['pdf_page'] for h in hits} >= {1, 2}
        assert len(hits) == len({(h.document.metadata['pdf_page'],
                                 h.document.metadata['chunk_index']) for h in hits})
        assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)
        assert search(('How do I check tires?',), 2) == search('How do I check tires?', 2)


@pytest.mark.parametrize('text, expected', [
    ('Prüfen Sie den Reifendruck bei kalten Reifen.', ('de',)),
    ('Проверяйте давление в шинах, когда шины холодные.', ('ru',)),
    ('Check tire pressure when the tires are cold.\n'
     'Prüfen Sie den Reifendruck bei kalten Reifen.\n'
     'Проверяйте давление в шинах, когда шины холодные.', ('de', 'en', 'ru')),
    ('BMW ABS DSC 530i 2.4 bar', ()),
])
def test_source_languages_come_from_passages(text, expected):
    from garage_rag.retrieval import passage_languages

    assert passage_languages([text]) == expected


def test_short_labels_and_copyright_do_not_add_a_translation_language():
    from garage_rag.retrieval import passage_languages

    text = ('Check tire pressure when the tires are cold.\n'
            'Respect the warnings before driving the vehicle.\n'
            '© 2001 Bayerische Motoren Werke\nAktiengesellschaft\nCockpit\nmode')
    assert passage_languages([text]) == ('en',)


def test_max_cosine_merge_keeps_a_hit_from_either_branch_and_respects_threshold(
    tmp_path, pdf_path, tokenizer, embeddings, config,
):
    from garage_rag.retrieval import AnswerSearch

    class Vectors:
        def embed_query(self, query):
            return [0.8, 0.6, 0.] if query == 'original' else [0., 0.6, 0.8]

    with Index(tmp_path / 'index', config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        search = AnswerSearch(index, Vectors())
        hits = search(('original', 'translated'), 4)
        assert hits[-1].document.metadata['pdf_page'] == 2
        assert hits[-1].score == pytest.approx(0.6)
        assert hits[0].score == pytest.approx(0.8)
        filtered = AnswerSearch(index, Vectors(), min_score=0.7)(('original', 'translated'), 4)
        assert {h.document.metadata['pdf_page'] for h in filtered} == {1, 4}
        with pytest.raises(ValueError, match='one and three'):
            search(('a', 'b', 'c', 'd'), 4)


def test_source_language_cache_refreshes_only_when_active_generations_change(
    tmp_path, pdf_path, tokenizer, embeddings, config, monkeypatch,
):
    from garage_rag.retrieval import AnswerSearch

    with Index(tmp_path / 'index', config) as index:
        search = AnswerSearch(index, embeddings)
        assert search.source_languages == ()
        index.ingest(pdf_path, embeddings, tokenizer, metadata={'language': 'de'})
        calls = []
        scroll = index.client.scroll
        def record(*args, **kwargs):
            calls.append(1)
            return scroll(*args, **kwargs)
        monkeypatch.setattr(index.client, 'scroll', record)
        assert search.source_languages == ('en',)
        assert search.source_languages == ('en',)
        assert len(calls) == 1
        index.ingest(pdf_path, embeddings, tokenizer, metadata={'language': 'ru'})
        assert search.source_languages == ('en',)
        assert len(calls) == 2


def test_truncated_expansion_is_not_retried_or_used_for_search():
    from types import SimpleNamespace

    class Truncated:
        def invoke(self, messages):
            return SimpleNamespace(content=json.dumps({'queries': [
                {'language': 'en', 'query': 'How do I check tires?'},
            ]}), response_metadata={'finish_reason': 'length'})
    result = run_answer('Как проверить шины?', lambda q, k: pytest.fail('search'),
                        Truncated(), mode='direct', source_languages=('en',))
    assert result.answer.reason == 'query_translation_error'
    assert result.trace.llm_calls == 1 and result.trace.search_calls == 0


@pytest.mark.parametrize('question, translation', [
    ('Как проверить BMW E39 при −20 градусах?', 'How do I check bmw e39 at -20 degrees?'),
    ('Как проверить индикатор ABS?', 'How do I check the abs warning light?'),
])
def test_expansion_accepts_equivalent_minus_and_identifier_casing(question, translation):
    model = Replies({'queries': [{'language': 'en', 'query': translation}]})
    seen = []
    result = run_answer(question, lambda q, k: seen.append(q) or [], model, mode='direct',
                        source_languages=('en',))
    assert seen == [(question, translation)]
    assert result.answer.reason == 'no_context'


def test_grounded_budget_without_audit_capacity_needs_no_translation():
    from test_workflow import hit

    from garage_rag.workflow import WorkflowLimits

    queries = []
    result = run_answer(
        'Как проверить шины?', lambda q, k: queries.append(q) or [hit()], Replies(),
        source_languages=('en',), limits=WorkflowLimits(max_llm_calls=1),
    )
    assert result.answer.reason == 'llm_budget_exhausted'
    assert result.trace.llm_calls == 0
    assert queries == ['Как проверить шины?']


def test_mixed_reranking_uses_native_questions_and_preserves_original_sources():
    from langchain_core.documents import Document

    from garage_rag.index import SearchHit
    from garage_rag.retrieval import rerank_multilingual

    wrong = SearchHit(Document(
        page_content='Das Ladegerät ist für verschiedene Batterien und Fahrzeuge vorgesehen.',
        metadata={'document': 'de.pdf', 'pdf_page': 3},
    ), 0.94)
    right = SearchHit(Document(
        page_content=('При перегреве зарядного устройства его выходная мощность '
                      'автоматически уменьшается.'),
        metadata={'document': 'ru.pdf', 'pdf_page': 9},
    ), 0.79)
    questions = ('What happens to the output power if the charger overheats?',
                 'Was passiert mit der Ausgangsleistung, wenn das Ladegerät überhitzt?',
                 'Что происходит с выходной мощностью при перегреве зарядного устройства?')
    class Ranker:
        def predict(self, pairs, **kwargs):
            assert pairs == [(questions[1], wrong.document.page_content),
                             (questions[2], right.document.page_content)]
            return [-2.0, 4.0]
    result = rerank_multilingual([wrong, right], questions, ('de', 'ru'), Ranker())
    assert result == [right, wrong]
    assert result[0] is right
    assert result[0].score == 0.79  # raw cosine remains available; order is semantic relevance


@pytest.mark.parametrize('scores', [[], [float('nan')], [float('inf')], [1.0, 2.0]])
def test_mixed_reranking_rejects_incomplete_or_nonfinite_scores(scores):
    from test_workflow import hit

    from garage_rag.retrieval import rerank_multilingual

    class Ranker:
        def predict(self, pairs, **kwargs):
            return scores
    with pytest.raises(ValueError, match='reranker scores'):
        rerank_multilingual([hit()], ('How do I check tires?', 'Wie prüfe ich die Reifen?'),
                            ('en',), Ranker())


def test_mixed_passage_reranking_uses_all_its_observed_languages():
    from test_workflow import hit

    from garage_rag.retrieval import rerank_multilingual

    mixed = hit('Check tire pressure when the tires are cold.\n'
                'Prüfen Sie den Reifendruck bei kalten Reifen.')
    english = 'How do I check tire pressure?'
    german = 'Wie prüfe ich den Reifendruck?'
    class Ranker:
        def predict(self, pairs, **kwargs):
            assert {q for q, _ in pairs} == {english, german}
            assert len(pairs) == 2
            return [0.0, 1.0]
    assert rerank_multilingual([mixed], (english, german), ('en', 'de'), Ranker()) == [mixed]


def test_monolingual_search_never_loads_the_additional_local_ranker(
    tmp_path, pdf_path, tokenizer, embeddings, config, monkeypatch,
):
    from garage_rag.retrieval import AnswerSearch

    monkeypatch.setattr('garage_rag.retrieval.LocalReranker',
                        lambda: pytest.fail('No reranker for a monolingual corpus'))
    with Index(tmp_path / 'index', config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        hits = AnswerSearch(index, embeddings)(('Как проверить шины?', 'How to check tires?'), 2)
        assert hits[0].document.metadata['pdf_page'] == 1


def test_answer_search_reranks_a_bounded_mixed_pool_before_applying_top_k(monkeypatch):
    from test_workflow import hit

    from garage_rag.retrieval import AnswerSearch

    wrong = hit('Check the power cable before connecting the charger to the battery.', 3)
    right = hit('При перегреве зарядного устройства его выходная мощность '
                'автоматически уменьшается.', 9)
    class DenseIndex:
        def search_many(self, queries, embeddings, top_k, min_score, *, preserve_top_five):
            assert top_k == 20
            assert min_score == 0.7
            assert preserve_top_five
            return [wrong, right]
    class Ranker:
        def predict(self, pairs, **kwargs):
            return [-1.0, 3.0]
    monkeypatch.setattr(AnswerSearch, 'source_languages', property(lambda self: ('en', 'ru')))
    search = AnswerSearch(DenseIndex(), None, min_score=0.7, reranker=Ranker())
    result = search(('What happens if the charger overheats?',
                     'Что происходит при перегреве зарядного устройства?'), 1)
    assert result == [right]


@pytest.mark.parametrize('top_k', [0, 21, True])
def test_mixed_reranking_cannot_bypass_the_public_top_k_limit(top_k, monkeypatch):
    from garage_rag.retrieval import AnswerSearch

    class DenseIndex:
        def search_many(self, *args, **kwargs):
            pytest.fail('Reject invalid top_k before retrieval')
    monkeypatch.setattr(AnswerSearch, 'source_languages', property(lambda self: ('en', 'ru')))
    with pytest.raises(ValueError, match='top_k'):
        AnswerSearch(DenseIndex(), None)(('English question', 'Русский вопрос'), top_k)


def test_local_reranker_failure_stops_without_another_provider_call_or_fallback(monkeypatch):
    from test_workflow import hit

    from garage_rag.retrieval import AnswerSearch

    class DenseIndex:
        def search_many(self, *args, **kwargs):
            return [hit()]
    class Ranker:
        calls = 0
        def predict(self, pairs, **kwargs):
            self.calls += 1
            raise ValueError('Question and passage exceed the reranker 512-token limit')
    monkeypatch.setattr(AnswerSearch, 'source_languages', property(lambda self: ('en', 'ru')))
    ranker = Ranker()
    model = Replies({'queries': [{'language': 'ru', 'query': 'Как проверить шины?'}]})
    result = run_answer('How do I check tires?',
                        AnswerSearch(DenseIndex(), None, reranker=ranker), model, mode='direct',
                        source_languages=('en', 'ru'))
    assert result.answer.reason == 'retrieval_error'
    assert not result.answer.supported
    assert result.trace.llm_calls == 1
    assert len(model.messages) == 1
    assert ranker.calls == 1


def test_oversized_reranker_pairs_keep_dense_order_without_truncation_or_a_new_call():
    from test_workflow import hit

    from garage_rag.retrieval import rerank_multilingual

    hits = [hit(), hit('Another original source passage with its full conditions.', 2)]
    class Ranker:
        def can_score(self, pairs):
            assert len(pairs) == 2
            return False
        def predict(self, *args, **kwargs):
            pytest.fail('Never score a truncated question or passage')
    assert rerank_multilingual(hits, ('How do I check tires?',), ('en',), Ranker()) is hits


@pytest.mark.parametrize('length, expected', [(512, True), (513, False)])
def test_existing_local_ranker_pair_limit_is_preserved(length, expected):
    from types import SimpleNamespace

    from garage_rag.retrieval import LocalReranker

    ranker = LocalReranker.__new__(LocalReranker)
    ranker.model = SimpleNamespace(tokenizer=lambda *args, **kwargs: {'input_ids': [0] * length})
    assert ranker.can_score([('Original full question', 'Original full source')]) is expected


@pytest.mark.parametrize('branches', [2, 3])
@pytest.mark.parametrize('top_k', [1, 5, 20])
def test_mixed_pool_preserves_each_branch_top_five_before_reranking(
    tmp_path, config, monkeypatch, branches, top_k,
):
    import math

    from qdrant_client import models

    from garage_rag.index import COLLECTION
    from garage_rag.retrieval import AnswerSearch

    questions = ('How do I check tire pressure?', 'Wie prüfe ich den Reifendruck?',
                 'Как проверить давление в шинах?')[:branches]
    text = ('Check tire pressure when the tires are cold.\n'
            'Prüfen Sie den Reifendruck bei kalten Reifen.\n'
            'Проверяйте давление в шинах, когда шины холодные.')
    class Vectors:
        def embed_query(self, question):
            return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]][questions.index(question)]
    # All twenty results in branch zero have larger cosines than the entire
    # lower-scoring branches. Their fifth hits must still reach the reranker.
    required = {f'{branch}:{rank}' for branch in range(branches) for rank in range(5)}
    target = f'{branches - 1}:4'
    scored = []
    class Ranker:
        def predict(self, pairs, **kwargs):
            scored.extend(pairs)
            pool = {passage.rsplit(' ', 1)[-1] for _, passage in pairs}
            assert required <= pool
            assert len(pool) == 20
            assert len(pairs) <= 60
            return [10. if passage.endswith(' ' + target) else 0. for _, passage in pairs]
    with Index(tmp_path / 'index', config) as index:
        index.manifest['documents'] = {
            'test.pdf': {'generation': 'active', 'chunks': branches * 20},
        }
        points = []
        for branch in range(branches):
            for rank in range(20):
                cosine = (0.98, 0.78, 0.72)[branch] - rank * 0.001
                vector = [0., 0., 0.]
                vector[branch] = cosine
                vector[1 if branch == 0 else 0] = math.sqrt(1 - cosine ** 2)
                points.append(models.PointStruct(id=branch * 20 + rank, vector=vector, payload={
                    'generation': 'active', 'text': f'{text} {branch}:{rank}',
                    'metadata': {'document': 'test.pdf', 'content_sha256': 'test',
                                 'pdf_page': branch + 1, 'chunk_index': rank},
                }))
        index.client.upsert(COLLECTION, points=points)
        batches = []
        batch_query = index.client.query_batch_points
        def record(*args, **kwargs):
            batches.append(kwargs['requests'])
            return batch_query(*args, **kwargs)
        monkeypatch.setattr(index.client, 'query_batch_points', record)
        search = AnswerSearch(index, Vectors(), min_score=0.7, reranker=Ranker())
        hits = search(questions, top_k)
        assert len(hits) == top_k
        assert hits[0].document.page_content.endswith(' ' + target)
        assert hits[0].score == pytest.approx((0.78 if branches == 2 else 0.72) - 0.004)
        assert len(batches) == 1 and len(batches[0]) == branches
        assert all(request.limit == 20 and request.score_threshold == 0.7
                   for request in batches[0])
        assert len(scored) == (40 if branches == 2 else 60)


def test_language_hints_do_not_build_bm25_state(
    tmp_path, pdf_path, tokenizer, embeddings, config, monkeypatch,
):
    from garage_rag import retrieval

    with Index(tmp_path / 'index', config) as index:
        index.ingest(pdf_path, embeddings, tokenizer)
        search = retrieval.AnswerSearch(index, embeddings)
        with monkeypatch.context() as patch:
            patch.setattr(retrieval, 'tokens', lambda text: pytest.fail('Unneeded BM25 tokens'))
            assert search.source_languages == ('en',)
        # Lexical retrieval must still be available after loading language hints.
        assert search._lexical('coolant')[0].document.metadata['pdf_page'] == 2
