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
