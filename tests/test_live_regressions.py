"""Regressions found by real EN/DE/RU manual UI checks; no private source data."""

import json

import pytest
from langchain_core.messages import HumanMessage
from test_generation import Reply
from test_groq import model, success
from test_workflow import Scripted, audit, draft, hit, run

from garage_rag.generation import answer


@pytest.mark.parametrize('source', [
    'Die Abdeckung aus ca. 30 cm Höhe fallen lassen, sodass sie hörbar einrastet.',
    'Опустите крышку с высоты примерно 30 см, чтобы она зафиксировалась.',
])
def test_intact_de_ru_numeric_prose_reaches_semantic_audit(source):
    result = run(Scripted(draft('Allow the cover to fall from 30 cm.', source), audit()),
                 lambda k: [hit(source)])
    assert result.answer.supported
    assert result.answer.reason == 'audited_evidence'


def test_groq_waits_before_sending_when_minute_tokens_are_used(monkeypatch):
    import garage_rag.groq as groq
    now = [1000.0]
    monkeypatch.setattr(groq.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(groq.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
    monkeypatch.setenv('GROQ_MAX_QUOTA_WAIT_SECONDS', '65')
    sent = []
    llm = model(monkeypatch, lambda r: sent.append(now[0]) or success())
    llm._history = [[1000.0, 7900]]
    with llm.client:
        response = llm.invoke([HumanMessage(content='Return JSON')])
    assert len(sent) == 1 and sent[0] >= 1060
    assert response.response_metadata['quota_wait_ms'] >= 60000


def test_daily_quota_has_structured_cooldown_without_sleep(monkeypatch):
    import garage_rag.groq as groq
    monkeypatch.setattr(groq.time, 'sleep', lambda _: pytest.fail('Must not sleep for a day'))
    llm = model(monkeypatch, lambda _: pytest.fail('Must not send'))
    llm._history = [[groq.time.monotonic(), 200000]]
    with llm.client, pytest.raises(Exception) as caught:
        llm.invoke([HumanMessage(content='Return JSON')])
    assert caught.value.status_code == 429
    assert caught.value.rate_limits['retry_after_seconds'] > 3600


def test_reported_token_reset_is_waited_before_http(monkeypatch):
    import garage_rag.groq as groq
    now = [1000.0]
    monkeypatch.setattr(groq.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(groq.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
    monkeypatch.setenv('GROQ_MAX_QUOTA_WAIT_SECONDS', '65')
    sent = []
    llm = model(monkeypatch, lambda r: sent.append(now[0]) or success())
    llm.rate_limits = {'tpm_remaining': 100, 'tpm_reset_seconds': 12.5}
    llm._headers_at = 1000
    with llm.client:
        llm.invoke([HumanMessage(content='Return JSON')])
    assert len(sent) == 1 and sent[0] >= 1012.5


@pytest.mark.parametrize('source', [
    'Drucktabelle\nVorne Hinten\nA1 2.2 2.8\nA2 2.5 3.0',
    'Таблица давления\nПередняя Задняя\nA1 2.2 2.8\nA2 2.5 3.0',
])
def test_de_ru_flattened_tables_remain_unsafe(source):
    result = run(Scripted(draft('The tire pressure is 2.2 bar.', source), audit()),
                 lambda k: [hit(source)])
    assert result.answer.reason == 'unsafe_numeric_evidence'


def test_quote_recovers_exact_source_span_after_broken_font_hyphen():
    source = 'Прогрейте контур до рабочей темИ\nпературы и отключите насос.'
    value = draft('Warm the circuit to operating temperature and stop the pump.',
                  'Прогрейте контур до рабочей температуры и отключите насос.')
    result = answer('How do I check oil?', [hit(source)], Reply(value))
    assert result.supported
    assert source in result.text


def test_broken_font_recovery_never_repairs_changed_value():
    source = 'Выждите примерно 5 минут после прогрева до рабочей темИ\nпературы.'
    value = draft('Wait 9 minutes.',
                  'Выждите примерно 9 минут после прогрева до рабочей температуры.')
    assert not answer('How do I check oil?', [hit(source)], Reply(value)).supported


def test_incomplete_procedure_is_revised_within_existing_grounded_budget():
    source = 'Turn the latch to open the cover. Keep the opening clear when closing it.'
    llm = Scripted(
        draft('Turn the latch to open the cover.', source),
        audit('unsupported', complete=False),
        draft('Turn the latch to open the cover; keep the opening clear when closing it.', source),
        audit(),
    )
    result = run(llm, lambda k: [hit(source)])
    assert result.answer.supported
    assert 'keep the opening clear' in result.answer.text
    assert result.trace.llm_calls == 4
    assert result.trace.search_calls == 1
    assert 'revision_instruction' in llm.calls[2][1].content


def test_source_reference_copies_original_passage_locally():
    source = 'Прогрейте контур до рабочей темИ\nпературы и отключите насос.'
    value = {'supported': True, 'claims': [{
        'text': 'Warm the circuit to operating temperature and stop the pump.',
        'evidence': [{'source_id': 'S1'}],
    }]}
    result = answer('How do I check oil?', [hit(source)], Reply(value))
    assert result.supported
    assert source in result.text


def test_unknown_source_reference_is_not_resolved():
    value = {'supported': True, 'claims': [{
        'text': 'Warm the circuit to operating temperature and stop the pump.',
        'evidence': [{'source_id': 'S99'}],
    }]}
    assert not answer('How do I check oil?', [hit()], Reply(value)).supported


def test_oil_level_answer_retains_explicit_warm_engine_prerequisite():
    source = 'Motorölstand prüfen\n2. Den betriebswarmen Motor abstellen\n3. Nach 5 min prüfen.'
    value = {'supported': True, 'claims': [{
        'text': 'Заглушите двигатель и подождите 5 минут перед проверкой уровня масла.',
        'evidence': [{'source_id': 'S1'}],
    }]}
    result = answer('Как проверить уровень масла?', [hit(source)], Reply(value))
    assert result.supported
    assert 'до рабочей температуры' in result.text
    assert source in result.text


def test_oil_temperature_prerequisite_is_never_invented():
    source = 'Motorölstand prüfen\n2. Den Motor abstellen\n3. Nach 5 min prüfen.'
    value = {'supported': True, 'claims': [{
        'text': 'Заглушите двигатель и подождите 5 минут перед проверкой уровня масла.',
        'evidence': [{'source_id': 'S1'}],
    }]}
    result = answer('Как проверить уровень масла?', [hit(source)], Reply(value))
    assert result.supported
    assert 'до рабочей температуры' not in result.text


@pytest.mark.parametrize('text', [
    'Stop the engine before checking the oil level.',
    'Заглушите холодный двигатель перед проверкой уровня масла.',
])
def test_prerequisite_retention_never_masks_wrong_language_or_contradiction(text):
    source = 'Motorölstand prüfen\n2. Den betriebswarmen Motor abstellen\n3. Nach 5 min prüfen.'
    value = {'supported': True, 'claims': [{'text': text, 'evidence': [{'source_id': 'S1'}]}]}
    assert not answer('Как проверить уровень масла?', [hit(source)], Reply(value)).supported


def test_grounded_accepts_source_verified_marker_count():
    source = 'Der Stand muss zwischen den beiden Markierungen liegen.'
    result = run(Scripted(draft('The level should be between the two marks.', source), audit()),
                 lambda k: [hit(source)])
    assert result.answer.supported


@pytest.mark.parametrize('claim,supported', [
    ('Wait 5 minutes before checking the level.', True),
    ('The pressure is 2.2 bar.', False),
])
def test_numbered_prose_does_not_inherit_unrelated_table_values(claim, supported):
    source = ('Pressure table\nFront Rear\nA1 2.2 2.8\nA2 2.5 3.0\n'
              '3. Nach 5 Minuten den Messstab vorsichtig herausziehen.')
    result = run(Scripted(draft(claim, source), audit()), lambda k: [hit(source)])
    assert result.answer.supported is supported


def test_wrapped_us_dual_units_are_prose_not_a_flattened_table():
    source = 'The reservoir capacity corresponds to approx.\n1.1 US quarts (1 liter).'
    result = run(Scripted(draft('The reservoir holds approximately 1.1 US quarts (1 liter).',
                               source), audit()), lambda k: [hit(source)])
    assert result.answer.supported


def test_audit_sends_full_source_once_and_keeps_local_quote():
    from garage_rag.generation import build_context, generate_candidate
    from garage_rag.grounding import audit_candidate

    source = 'Keep the opening clear before closing the cover.'
    context = build_context([hit(source)])
    candidate = generate_candidate('How do I close the cover?', context,
                                   Reply(draft('Keep the opening clear.', source)), 'en')
    llm = Scripted(audit())
    audit_candidate('How do I close the cover?', candidate, context, llm)
    payload = json.loads(llm.calls[0][1].content)
    assert payload['sources'][0]['text'] == source
    assert payload['draft']['claims'][0]['evidence'] == [{'source_id': 'S1'}]
    assert candidate.draft.claims[0].evidence[0].quote == source


@pytest.mark.parametrize('value,supported', [('5', True), ('7', False)])
def test_procedure_ordinals_are_not_measurements(value, supported):
    source = '1. Park on a level surface.\n2. Wait 5 minutes before checking the level.'
    text = f'Procedure: 1) Park on a level surface; 2) wait {value} minutes before checking.'
    result = run(Scripted(draft(text, source), audit()), lambda k: [hit(source)])
    assert result.answer.supported is supported


@pytest.mark.parametrize('value,supported', [('3', True), ('7', False)])
def test_english_after_clause_confirms_procedural_wait(value, supported):
    source = '2. After approximately 3 minutes, remove the inspection probe and wipe it clean.'
    llm = Scripted(draft(f'Wait {value} minutes before removing the probe.', source), audit())
    result = run(llm, lambda k: [hit(source)])
    assert result.answer.supported is supported


@pytest.mark.parametrize('source', [
    'Pressure limits maximum front rear 2.2 2.8',
    'Reifendruck nach Beladung Vorderachse Hinterachse 2.2 2.8',
    'Давление примерно передняя ось задняя ось 2.2 2.8',
])
def test_table_labels_are_not_prose_predicates(source):
    result = run(Scripted(draft('The tire pressure is 2.2 bar.', source), audit()),
                 lambda k: [hit(source)])
    assert result.answer.reason == 'unsafe_numeric_evidence'


@pytest.mark.parametrize('failure', ['transport', 'malformed'])
def test_recovered_rate_limit_does_not_make_other_errors_recoverable(monkeypatch, failure):
    import httpx

    import garage_rag.groq as groq
    from garage_rag.generation import GenerationError

    now = [1000.0]
    monkeypatch.setattr(groq.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(groq.time, 'time', lambda: now[0])
    sent = []

    def respond(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(429, headers={'retry-after': '1'})
        if failure == 'transport':
            raise httpx.ConnectError('Unavailable', request=request)
        return httpx.Response(200, json={'choices': []})

    llm = model(monkeypatch, respond)
    messages = [HumanMessage(content='Return JSON')]
    with llm.client:
        with pytest.raises(GenerationError):
            llm.invoke(messages)
        now[0] += 61
        with pytest.raises((GenerationError, httpx.TransportError)):
            llm.invoke(messages)
        now[0] += 61
        with pytest.raises(GenerationError):
            llm.invoke(messages)
    assert len(sent) == 2


@pytest.mark.parametrize('retry_after', [None, '10.9'])
def test_provider_quota_error_retains_longer_effective_reset(monkeypatch, retry_after):
    import httpx

    from garage_rag.generation import GenerationError

    headers = {'x-ratelimit-remaining-requests': '0',
               'x-ratelimit-reset-requests': '18m19.5s'}
    if retry_after is not None:
        headers['retry-after'] = retry_after
    llm = model(monkeypatch, lambda _: httpx.Response(429, headers=headers))
    with llm.client, pytest.raises(GenerationError) as caught:
        llm.invoke([HumanMessage(content='Return JSON')])
    assert caught.value.rate_limits['retry_after_seconds'] == 1100
