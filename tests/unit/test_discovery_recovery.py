"""Archive discovery regressions using synthetic targets and provider streams."""
import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest
import requests

from core.llm_client import LLMClient
from core.router import route_alma_science_archive_query
from core.turn_recovery import partial_tool_answer, serialize_tool_result, tool_call_key
from integrations.alma_tap import ALMA_TAP_MIRRORS, AlmaTapService
from integrations.alminer_client import ALminerClient
from integrations.datalab_sia_client import DatalabSiaClient, DatalabSiaClientError
from services.alma_science_queries import normalize_target_alias, science_cone_query, summarize_mous
from services.datalab_image_service import DatalabImageService
from tests.unit.test_response_api_error_recovery import _make_agent
from tests.unit.test_alma_capability import _ctx, _run, _FakeAlminer, _FakeSearchService
from capabilities.alma import QueryAlmaScienceArchive


class _Result:
    def __init__(self, frame):
        self.frame = frame
    def to_table(self):
        return self
    def to_pandas(self):
        return self.frame.copy()


@pytest.mark.parametrize('failure', [requests.ReadTimeout('read timed out'),
                                   requests.HTTPError(response=NS(status_code=503)),
                                   RuntimeError('JDBC PoolExhaustedException')])
def test_tap_retry_rotates_mirror_and_keeps_maxrec(monkeypatch, failure):
    # Budget hierarchy (2026-09-21): one query may try at most
    # DEFAULT_MAX_ATTEMPTS (2) mirrors at DEFAULT_ATTEMPT_TIMEOUT_S (40 s)
    # each, so its worst case (~81 s) stays strictly below the 150 s guard.
    # The old policy — a 120 s floor x 3 mirrors — was inverted against it.
    monkeypatch.setattr(AlmaTapService, '_cooldown', {})
    monkeypatch.delenv('ALMA_TAP_TIMEOUT_SECONDS', raising=False)
    monkeypatch.delenv('ALMA_TAP_MAX_ATTEMPTS', raising=False)
    from services.host_breaker import HostBreaker
    HostBreaker.reset()
    calls, delays = [], []
    def factory(url, session):
        assert session._default_timeout == AlmaTapService.DEFAULT_ATTEMPT_TIMEOUT_S
        assert session._default_timeout * AlmaTapService.DEFAULT_MAX_ATTEMPTS < 150
        def search(query, maxrec):
            calls.append((url, query, maxrec))
            if len(calls) < 2:
                raise failure
            return _Result(pd.DataFrame([{'value': 42}]))
        return NS(search=search)
    monkeypatch.setattr('pyvo.dal.TAPService', factory)
    monkeypatch.setattr('integrations.alma_tap.time.sleep', delays.append)
    result = AlmaTapService().search('SELECT TOP 1 * FROM ivoa.obscore', maxrec=27)
    assert [c[0] for c in calls] == list(ALMA_TAP_MIRRORS[:2])
    assert all(c[2] == 27 for c in calls)
    assert delays == [1]
    assert result.quasar_tap_url == ALMA_TAP_MIRRORS[1]
    assert result.quasar_attempts == 2


def test_tap_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(AlmaTapService, '_cooldown', {})
    from services.host_breaker import HostBreaker
    HostBreaker.reset()
    calls = []
    def factory(url, session):
        def search(query, maxrec):
            calls.append(url)
            raise requests.ReadTimeout('read timed out')
        return NS(search=search)
    monkeypatch.setattr('pyvo.dal.TAPService', factory)
    monkeypatch.setattr('integrations.alma_tap.time.sleep', lambda s: None)
    with pytest.raises(RuntimeError, match='ALMA TAP query failed'):
        AlmaTapService().search('SELECT TOP 1 * FROM ivoa.obscore')
    assert len(calls) == AlmaTapService.DEFAULT_MAX_ATTEMPTS


def test_tap_does_not_retry_invalid_query(monkeypatch):
    monkeypatch.setattr(AlmaTapService, '_cooldown', {})
    calls = []
    def bad(query, maxrec):
        calls.append(query)
        raise ValueError('unknown column invalid_field')
    monkeypatch.setattr('pyvo.dal.TAPService', lambda *a, **kw: NS(search=bad))
    with pytest.raises(RuntimeError, match='invalid_field'):
        AlmaTapService().search('SELECT invalid_field FROM ivoa.obscore')
    assert len(calls) == 1


def test_custom_adql_uses_pyvo_even_with_alminer_installed(monkeypatch):
    client = ALminerClient()
    calls = []
    def search(query, maxrec):
        calls.append((query, maxrec))
        return _Result(pd.DataFrame([{'target_name': 'Synthetic A'}]))
    monkeypatch.setattr(client, '_get_tap_service', lambda: NS(search=search))
    monkeypatch.setattr('integrations.alminer_client.ALMINER_AVAILABLE', True)
    frame = client.search_by_sql('SELECT TOP 1 * FROM ivoa.obscore')
    assert len(frame) == 1 and not frame.attrs.get('quasar_error')
    assert calls == [('SELECT TOP 1 * FROM ivoa.obscore', 20000)]


def test_keyword_search_uses_supported_alminer_api(monkeypatch):
    calls = []
    monkeypatch.setattr('integrations.alminer_client.alminer.keysearch',
                        lambda values, **kw: calls.append(values) or pd.DataFrame())
    result = ALminerClient().search_by_keywords({'target_name': 'Synthetic A'})
    assert calls == [{'target_name': ['Synthetic A']}]
    assert not result.attrs.get('quasar_error')


def test_tap_failure_and_empty_fallback_stays_error(monkeypatch):
    def bad(query, *, maxrec):
        raise RuntimeError('transport unavailable')
    monkeypatch.setattr(ALminerClient, '_get_tap_service', lambda self: NS(search=bad))
    monkeypatch.setattr('integrations.alminer_client.alminer.conesearch', lambda *a, **k: pd.DataFrame())
    frame = ALminerClient().search_by_position(25, -10)
    assert frame.attrs['quasar_error'] and frame.attrs['partial']


@pytest.mark.parametrize('target', ['HH 212', 'NGC 4321', 'Synthetic 999', 'L1551 IRS 5', 'Class 0 protostars'])
def test_router_extracts_user_constraints_but_never_guesses_the_target(target):
    args = route_alma_science_archive_query(None,
        f'ALMA observations of {target}: public science observations, Band 4 or Band 8, resolution under 2.3 arcseconds.')
    assert 'target' not in args  # the model supplies the identifier; a regex guess truncates multi-token names
    assert args['band'] == [4, 8]
    assert args['max_resolution_arcsec'] == 2.3
    assert args['public_only'] and args['science_only']


def test_no_default_target_or_resolution_is_invented():
    args = route_alma_science_archive_query(None, 'ALMA high resolution data in bands 3, 5 and 9')
    assert args['band'] == [3, 5, 9]
    assert 'target' not in args and 'max_resolution_arcsec' not in args
    assert normalize_target_alias('NGC4321') == 'NGC4321'
    assert normalize_target_alias(' V404  Cyg ') == 'V404 Cyg'


def _observations():
    return pd.DataFrame([
        {'member_ous_uid': 'uid://A/X1/X2', 'proposal_id': 'P-A', 'target_name': 'Synthetic A',
         's_ra': 25., 's_dec': -10., 'band_list': '4', 'spatial_resolution': 0.8, 'qa2_passed': 'T'},
        {'member_ous_uid': 'uid://A/X1/X2', 'proposal_id': 'P-A', 'target_name': 'Synthetic_A',
         's_ra': 25., 's_dec': -10., 'band_list': '8', 'spatial_resolution': 0.4, 'qa2_passed': 'T'},
        {'member_ous_uid': 'uid://A/X1/X3', 'proposal_id': 'P-A', 'target_name': 'Synthetic A',
         's_ra': 25., 's_dec': -10., 'band_list': '8', 'spatial_resolution': 1.8, 'qa2_passed': 'T'},
    ])


def test_mous_grouping_keeps_distinct_datasets_in_one_project():
    result = summarize_mous(_observations())
    assert len(result) == 2
    assert result.iloc[0].band_list == '4 8'
    assert result.iloc[0].spatial_resolution == .4
    assert result.iloc[0].rows == 2


def test_science_cone_applies_filters_before_cap():
    query = science_cone_query(25, -10, 75, band=[4, 8], max_resolution_arcsec=2.3,
                               public_only=True, science_only=True, top=21)
    assert 'INTERSECTS' in query and '0.02083333' in query
    assert "band_list = '4'" in query and "band_list = '8'" in query
    assert "spatial_resolution < 2.3" in query and "data_rights = 'Public'" in query
    assert "science_observation = 'T'" in query
    assert 'qa2_passed' in query and 'SELECT TOP 21' in query
    assert "target_name =" not in query


def test_science_capability_returns_mous_and_position_without_hidden_cap():
    service = _FakeSearchService()
    service.alminer_client = _FakeAlminer(_observations())
    ctx, _ = _ctx(search_service=service, resolve_target=lambda target: {'ra_deg': 25., 'dec_deg': -10.})
    result = _run(QueryAlmaScienceArchive(), ctx, query_type='high_resolution_band_data',
                  target='Synthetic A', band=[4,8], max_resolution_arcsec=2.3, public_only=True,
                  science_only=True, radius_arcsec=75, max_results=3)
    assert result['success'] and result['count'] == 2
    assert result['truncated']  # Raw fetch cap must survive grouping.
    assert result['search_position'] == {'ra_deg':25., 'dec_deg':-10., 'radius_arcsec':75.}
    assert result['results'][0]['qa2_passed'] == 'T'


def test_sia_nsa_is_an_explicit_collection_and_preserves_rows(monkeypatch):
    client = DatalabSiaClient()
    rows = [{'obs_bandpass':'z', 'exptime':47., 'proctype':'Stack', 'prodtype':'image', 'access_url':'https://archive.test/image'}]
    calls=[]
    monkeypatch.setattr(client, '_search_endpoint', lambda ep,*args: calls.append(ep) or rows)
    result = client.search(25, -10, .2, service='nsa')
    assert calls == ['https://datalab.noirlab.edu/sia/nsa']
    assert result['rows'] == rows
    monkeypatch.setattr(client, '_search_endpoint', lambda *a: [])
    assert client.search(25,-10,.2,service='nsa')['coverage_gap']
    def bad(*a):
        raise RuntimeError('502')
    monkeypatch.setattr(client, '_search_endpoint', bad)
    with pytest.raises(DatalabSiaClientError):
        client.search(25,-10,.2,service='nsa')


def test_deepest_selection_uses_exposure_and_stack_image_only():
    rows = [dict(obs_bandpass=band, exptime=exposure, proctype=proc, prodtype=prod, access_url=str(i))
            for i,(band,exposure,proc,prod) in enumerate([
                ('z',37,'Stack','image'), ('z',52,'Stack','image'),
                ('z',999,'Raw','image'), ('z',888,'Stack','weight'), ('y',43,'Stack','image')])]
    selected = DatalabImageService.deepest_archive_images(rows, ['z','y'])
    assert selected == [rows[1],rows[4]]


def _events(round_id, *, reasoning=None, tool=None, text=None):
    events=[NS(type='response.created', response=NS(id=f'resp-{round_id}'))]
    if reasoning:
        events += [NS(type='response.reasoning_summary_text.delta',delta=reasoning),
                   NS(type='response.reasoning_summary_text.done')]
    if tool:
        item=NS(type='function_call',name=tool,arguments='{"x": 1}',call_id=f'call-{round_id}',id=f'item-{round_id}')
    if text:
        events.append(NS(type='response.output_text.delta',delta=text))
    events.append(NS(type='response.completed',response=NS(finish_reason='stop',output=[item] if tool else [])))
    return events


class _Responses:
    def __init__(self, rounds):
        self.rounds, self.calls = rounds, []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.rounds[len(self.calls)-1])
    def clear_history(self, *a, **kw):
        pass


def _tool_agent(responses, tool_name='query_archive', result=None):
    agent = _make_agent(responses)
    executed=[]
    agent.tool_registry = NS(get_tool=lambda name: NS(name=name))
    agent.session_memory = NS(record_tool_calls=lambda n: None)
    agent._execute_tool_with_progress = lambda *a,**kw: executed.append(kw) or (result or {'success':True, 'results':[{'value':42}]})
    agent._record_tool_trace = lambda *a,**kw: None
    agent._auto_link_project_papers_from_result = lambda *a: pytest.fail('Unrequested paper lookup')
    return agent, executed


def test_reasoning_only_stop_retries_then_executes_tool_and_answers(capsys):
    responses = _Responses([_events(1,reasoning='Call the archive tool.'),
                            _events(2,tool='query_archive'),_events(3,text='Found the data.')])
    agent, executed = _tool_agent(responses)
    tokens=[]
    result = agent.stream_response_api('hello there', conversation_id='recovery', on_token=tokens.append)
    assert len(responses.calls) == 3 and len(executed) == 1
    # First retry re-samples the SAME round from the pre-round state: identical
    # input, identical previous_response_id, no empty assistant turn chained.
    assert responses.calls[1]['input'] == responses.calls[0]['input']
    assert responses.calls[1]['previous_response_id'] == responses.calls[0]['previous_response_id']
    assert 'Found the data.' in result and "didn't generate" not in ''.join(tokens)
    assert '[RECOVERY] reasoning-only stop' in capsys.readouterr().out


def test_reasoning_retries_are_bounded_and_escalate_to_an_explicit_instruction():
    responses = _Responses([_events(n,reasoning='Still thinking.') for n in range(6)])
    agent,_ = _tool_agent(responses)
    agent.stream_response_api('hello there',conversation_id='bounded')
    assert len(responses.calls) == 4  # original + 3 bounded re-samples
    assert 'actual tool call' not in responses.calls[1]['input']
    assert all('actual tool call' in c['input'] for c in responses.calls[2:])
    # Every attempt branches from the same pre-round state (round 0 -> no chain).
    assert {c['previous_response_id'] for c in responses.calls} == {None}


def test_repeated_successful_query_is_reused_and_forces_final_answer():
    responses = _Responses([_events(n,tool='query_archive') for n in range(3)] + [_events(4,text='Final data.')])
    agent, executed = _tool_agent(responses)
    result=agent.stream_response_api('hello there',conversation_id='repeat')
    assert len(executed) == 1 and 'Final data.' in result
    assert 'You already ran this' in responses.calls[2]['input'][0]['output']
    assert responses.calls[-1]['tool_choice'] == 'none' and 'tools' in responses.calls[-1]  # schemas kept, new calls forbidden


def test_archive_search_does_not_automatically_look_up_papers():
    responses=_Responses([_events(1,tool='search_by_target'),_events(2,text='Done.')])
    agent, executed=_tool_agent(responses)
    assert 'Done.' in agent.stream_response_api('hello there',conversation_id='no-papers')
    assert len(executed)==1


def test_tacc_structured_commentary_calls_are_not_lost():
    chunks=[NS(choices=[NS(delta=NS(content=None,tool_calls=None,reasoning_content='Use a tool.'),finish_reason=None)]),
            NS(choices=[NS(delta=NS(content=None,tool_calls=None,commentary={'tool_calls':[
                {'index':0,'id':'call-a','function':{'name':'query_archive','arguments':'{"x":1}'}}]}),finish_reason='stop')])]
    client=LLMClient(model='gpt-oss-120b')
    client._get_tacc_client=lambda: NS(chat=NS(completions=NS(create=lambda **kw: iter(chunks))))
    events=list(client.responses.create(model='gpt-oss-120b',input='hello',stream=True))
    completed=next(e.response for e in events if e.type=='response.completed')
    assert completed.output[0].name=='query_archive'
    assert json.loads(completed.output[0].arguments)=={'x':1}
    assert not any(e.type=='response.output_text.delta' for e in events)


def test_structured_context_truncation_and_partial_answer_remain_valid_json():
    result={'success':True,'results':[{'i':i,'text':'a'*400} for i in range(100)]}
    compact=serialize_tool_result(result,max_chars=2000)
    assert json.loads(compact)['context_truncated']
    assert len(result['results']) == 100
    partial=json.loads(partial_tool_answer([{'output':compact}], 'deadline'))
    assert partial['partial'] and partial['tool_results'][0]['results']
    assert tool_call_key('q',{'a':1,'b':2}) == tool_call_key('q',{'b':2,'a':1})


def test_total_evidence_and_hard_partial_are_bounded():
    from core.turn_recovery import compact_tool_outputs
    results=[{'output':json.dumps({'success':True,'results':[{'n':n,'payload':'x'*49000}]})} for n in range(60)]
    compact=compact_tool_outputs(results)
    assert sum(map(len,compact)) <= 80000
    assert json.loads(compact[0])['omitted_older_results'] == 36
    partial=partial_tool_answer(results,'deadline')
    assert len(partial) <= 50000 and json.loads(partial)['partial']


def test_identical_call_after_a_failure_is_not_reexecuted_but_a_changed_one_is():
    # UI benchmark 2026-09-22 (L06): re-issuing the SAME failed call (or one
    # that only re-labels a plot) burns the budget; the runner now answers it
    # from the ledger and tells the model to change the approach. A retry with
    # DIFFERENT arguments (the corrected call the prompt asks for) still runs.
    responses=_Responses([_events(1,tool='query_archive'),_events(2,tool='query_archive'),_events(3,text='Done.')])
    agent,executed=_tool_agent(responses)
    def run(*a,**kw):
        executed.append(kw)
        return {'success':False,'error':'502 temporary'} if len(executed)==1 else {'success':True,'rows':[42]}
    agent._execute_tool_with_progress=run
    agent.stream_response_api('hello there',conversation_id='retry-error')
    assert len(executed)==1
    repeat=json.loads(responses.calls[2]['input'][0]['output'])
    assert repeat['identical_call_failed'] and '502 temporary' in repeat['error'] and 'Change the approach' in repeat['error']
    # A corrected retry (different args) is executed.
    rounds=[_events(1,tool='query_archive'),_events(2,tool='query_archive'),_events(3,text='Done.')]
    rounds[1][-1].response.output[0].arguments='{"x": 2}'
    responses=_Responses(rounds)
    agent,executed=_tool_agent(responses)
    agent._execute_tool_with_progress=run
    executed.clear()
    agent.stream_response_api('hello there',conversation_id='retry-error-2')
    assert len(executed)==2


def test_polling_tools_are_never_cached():
    responses=_Responses([_events(n,tool='query_job_status') for n in range(3)]+[_events(4,text='Done.')])
    agent,executed=_tool_agent(responses)
    agent.stream_response_api('hello there',conversation_id='poll')
    assert len(executed)==3


def test_cosmetic_only_reissue_of_a_failed_plot_call_is_refused():
    """L06 pattern: same CMD query, new title each time, every one a timeout."""
    rounds=[]
    for n,title in enumerate(['White dwarfs','White dwarf CMD (retry)','WD CMD attempt 3'],start=1):
        ev=_events(n,tool='datalab_color_magnitude_diagram')
        ev[-1].response.output[0].arguments=json.dumps({'catalog':'gaia_dr3','radius':2.0,'title':title})
        rounds.append(ev)
    rounds.append(_events(4,text='Done.'))
    responses=_Responses(rounds)
    agent,executed=_tool_agent(responses)
    agent._execute_tool_with_progress=lambda *a,**kw: executed.append(kw) or {'success':False,'error':'Read timed out.'}
    agent.stream_response_api('hello there',conversation_id='cosmetic')
    assert len(executed)==1, 'only the first call may run; re-labelled repeats are refused'
    for call in responses.calls[2:4]:
        assert json.loads(call['input'][0]['output'])['identical_call_failed']


def test_soft_finalization_retains_history_for_followup(monkeypatch):
    import copy
    clock=[0.]
    monkeypatch.setenv('CHAT_HARD_MAX_TIMEOUT_SECONDS','600')
    monkeypatch.setattr('core.runner.time.monotonic',lambda:clock[0])
    calls=[]
    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls)==1:
            delta=NS(content=None,tool_calls=[NS(index=0,id='call-1',function=NS(name='query_archive',arguments='{}'))])
        else:
            delta=NS(content='Final data.' if len(calls)==2 else 'Followup data.',tool_calls=None)
        return iter([NS(choices=[NS(delta=delta,finish_reason='stop')])])
    client=LLMClient(model='gpt-oss-120b')
    client._get_tacc_client=lambda:NS(chat=NS(completions=NS(create=create)))
    runner_calls=[]
    _orig_create=client.responses.create
    client.responses.create=lambda **kw: (runner_calls.append(copy.deepcopy(kw)), _orig_create(**kw))[1]
    agent,executed=_tool_agent(client.responses)
    def execute(*a,**kw):
        executed.append(kw);clock[0]=421.
        return {'success':True,'results':[{'value':42}]}
    agent._execute_tool_with_progress=execute
    assert 'Final data.' in agent.stream_response_api('hello there',conversation_id='history')
    assert runner_calls[1]['tool_choice'] == 'none' and 'tools' in runner_calls[1]  # forced final: schemas kept, new calls forbidden
    assert 'Followup data.' in agent.stream_response_api('tell me more',conversation_id='history')
    messages=calls[-1]['messages']
    assert any(m.get('role')=='tool' and '42' in m['content'] for m in messages)
    assert any(m.get('role')=='assistant' and m.get('content')=='Final data.' for m in messages)
    pending=set()
    for message in messages:
        if message['role']=='assistant':
            assert not pending
            pending.update(t['id'] for t in message.get('tool_calls',[]))
        elif message['role']=='tool':
            pending.remove(message['tool_call_id'])
        elif message['role']=='user':
            assert not pending
    assert not pending


def test_hard_deadline_emits_available_partial_without_more_model_calls(monkeypatch):
    clock=[0.]
    monkeypatch.setenv('CHAT_HARD_MAX_TIMEOUT_SECONDS','600')
    monkeypatch.setattr('core.runner.time.monotonic',lambda:clock[0])
    responses=_Responses([_events(1,tool='query_archive')])
    agent,_=_tool_agent(responses)
    def execute(*a,**kw):
        clock[0]=601.
        return {'success':True,'results':[{'value':42}]}
    agent._execute_tool_with_progress=execute
    result=agent.stream_response_api('hello there',conversation_id='hard')
    partial=json.loads(result)
    assert partial['partial'] and partial['tool_results'][0]['results']==[{'value':42}]
    assert len(responses.calls)==1


def test_sia_missing_requested_bands_is_partial():
    from capabilities.datalab import SiaSearch, SiaSearchInput
    from tests.unit.test_datalab_capability import _img_ctx
    rows=[{'obs_bandpass':'z DECam','exptime':42.,'proctype':'Stack','prodtype':'image','access_url':'https://archive.test/z'}]
    service=NS(search=lambda *a,**kw:{'rows':rows,'coverage_gap':False,'used_endpoint':kw['endpoint']})
    result=SiaSearch().run(SiaSearchInput(ra=25,dec=-10,service='nsa',bands=['z','y'],deepest_per_band=True),_img_ctx(service)).to_native()
    assert result['partial'] and result['missing_bands']==['y']
    assert not result['coverage_gap'] and result['deepest_images']==rows
    result=SiaSearch().run(SiaSearchInput(ra=25,dec=-10,service='nsa',bands=['y'],deepest_per_band=True),_img_ctx(service)).to_native()
    assert result['coverage_gap'] and result['missing_bands']==['y']


@pytest.mark.parametrize('target',['IRAS 16293-2422','2MASS J16281370-2431391'])
def test_router_never_truncates_a_catalog_identifier(target):
    args=route_alma_science_archive_query(None,f'ALMA observations of {target}: Band 4, resolution under 2 arcsec')
    assert 'target' not in args or args['target']==target


@pytest.mark.parametrize('ra,dec,radius',[(float('nan'),0,60),(361,0,60),(1,91,60),(1,1,-2)])
def test_science_cone_rejects_invalid_geometry(ra,dec,radius):
    with pytest.raises(ValueError):
        science_cone_query(ra,dec,radius)


def test_missing_mous_rows_are_disclosed_without_losing_valid_groups():
    frame=_observations();frame.loc[2,'member_ous_uid']=None
    grouped=summarize_mous(frame)
    assert len(grouped)==1 and grouped.attrs['partial'] and grouped.attrs['ungroupable_rows']==1


@pytest.mark.parametrize('shape',['dict','object','list'])
def test_tacc_commentary_call_shapes_and_split_arguments(shape):
    def payload(call):
        if shape=='dict':return {'tool_calls':call}
        if shape=='list':return {'tool_calls':[call]}
        return NS(tool_calls=[call])
    calls=[{'index':0,'id':'call-a','function':{'name':'query_archive','arguments':'{"x":'}},
           {'index':0,'function':{'arguments':'1}'}}]
    chunks=[NS(choices=[NS(delta=NS(content=None,tool_calls=None,commentary=payload(call)),finish_reason='stop')]) for call in calls]
    client=LLMClient(model='gpt-oss-120b')
    client._get_tacc_client=lambda:NS(chat=NS(completions=NS(create=lambda **kw:iter(chunks))))
    completed=next(e.response for e in client.responses.create(model='gpt-oss-120b',input='hello',stream=True) if e.type=='response.completed')
    assert completed.output[0].name=='query_archive' and json.loads(completed.output[0].arguments)=={'x':1}


def test_resolver_prefers_sesame_and_keeps_full_precision(monkeypatch):
    """Resolved coordinates must be exactly what the standard astropy resolver
    returns — no lossy rounding — so derived archive requests reproduce."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from capabilities.calc import resolve_target
    simbad_calls = []
    monkeypatch.setattr(SkyCoord, 'from_name',
                        classmethod(lambda cls, name, *a, **k: SkyCoord(ra=25.12345678*u.deg, dec=-10.87654321*u.deg)))
    monkeypatch.setattr('astroquery.simbad.Simbad.query_object', staticmethod(lambda name: simbad_calls.append(name)))
    out = resolve_target('Synthetic 7')
    assert out['success'] and out['resolver'] == 'CDS Sesame'
    assert out['ra_deg'] == 25.12345678 and out['dec_deg'] == -10.87654321
    assert simbad_calls == []


def test_resolver_falls_back_to_simbad_without_rounding(monkeypatch):
    from astropy.coordinates import SkyCoord
    from astropy.coordinates.name_resolve import NameResolveError
    from astropy.table import Table
    from capabilities.calc import resolve_target
    def _fail(cls, name, *a, **k):
        raise NameResolveError('no sesame')
    monkeypatch.setattr(SkyCoord, 'from_name', classmethod(_fail))
    monkeypatch.setattr('astroquery.simbad.Simbad.query_object',
                        staticmethod(lambda name: Table(rows=[(25.1234567891, -10.9876543219)], names=('ra', 'dec'))))
    out = resolve_target('Synthetic 8')
    assert out['success'] and out['resolver'] == 'SIMBAD'
    assert out['ra_deg'] == 25.1234567891 and out['dec_deg'] == -10.9876543219
    monkeypatch.setattr('astroquery.simbad.Simbad.query_object', staticmethod(lambda name: None))
    missing = resolve_target('Synthetic 9')
    assert not missing['success'] and 'no sesame' in missing['error']


def test_mous_rows_carry_integer_band_list():
    grouped = summarize_mous(_observations())
    assert grouped.iloc[0].bands == [4, 8] and grouped.iloc[1].bands == [8]
    assert grouped.iloc[0].band_list == '4 8'


def test_anthropic_forced_final_round_keeps_tools_and_forbids_new_calls(monkeypatch):
    """The Messages API rejects tool_use/tool_result history without `tools`; a
    forced-final round must keep the schemas and disable new calls instead."""
    from core.llm_client import ResponsesShim
    captured = {}
    def fake_stream(self, client, call_kwargs):
        captured.update(call_kwargs)
        yield NS(type='response.created', response=NS(id='resp-a', usage=None))
        yield NS(type='response.completed', response=NS(id='resp-a', output=[], finish_reason='stop', usage=None))
    monkeypatch.setattr(ResponsesShim, '_anthropic_stream_generator', fake_stream)
    monkeypatch.setattr(LLMClient, '_get_anthropic_client', lambda self: NS())
    client = LLMClient(model='claude-sonnet-5')
    tools = [{'type': 'function', 'name': 'query_archive', 'description': 'd',
              'parameters': {'type': 'object', 'properties': {}}}]
    list(client.responses.create(model='claude-sonnet-5', input='answer now', tools=tools,
                                 tool_choice='none', stream=True))
    assert captured['tools'] and captured['tools'][0]['name'] == 'query_archive'
    assert captured['tool_choice'] == {'type': 'none'}
    captured.clear()
    list(client.responses.create(model='claude-sonnet-5', input='hello', tools=tools, stream=True))
    assert 'tool_choice' not in captured  # only the explicit forced-final flag is translated


def test_tacc_channel_calls_without_index_are_keyed_by_id():
    def chunk(call, finish=None):
        return NS(choices=[NS(delta=NS(content=None, tool_calls=None, commentary={'tool_calls': [call]}), finish_reason=finish)])
    chunks = [chunk({'id': 'c1', 'function': {'name': 'q1', 'arguments': '{"a":'}}),
              chunk({'id': 'c2', 'function': {'name': 'q2', 'arguments': '{"b": 2}'}}),
              chunk({'id': 'c1', 'function': {'arguments': ' 1}'}}, finish='stop')]
    client = LLMClient(model='gpt-oss-120b')
    client._get_tacc_client = lambda: NS(chat=NS(completions=NS(create=lambda **kw: iter(chunks))))
    completed = next(e.response for e in client.responses.create(model='gpt-oss-120b', input='hello', stream=True)
                     if e.type == 'response.completed')
    assert [c.call_id for c in completed.output] == ['c1', 'c2']
    assert {c.name: json.loads(c.arguments) for c in completed.output} == {'q1': {'a': 1}, 'q2': {'b': 2}}


def test_list_catalogs_exposes_archive_wide_image_services():
    import capabilities.datalab as dl
    from tests.unit.test_datalab_catalog_coverage import _ctx as _catalog_ctx
    out = dl.ListCatalogs().run(dl.ListCatalogsInput(), _catalog_ctx()).to_native()
    services = {row['service']: row for row in out['image_services']}
    assert services['nsa']['endpoint'].endswith('/sia/nsa') and 'Stack' in services['nsa']['description']
    assert services['coadd_all']['endpoint'].endswith('/sia/coadd_all')
    assert "service='nsa'" in out['note']


def test_router_is_gated_on_archive_intent_and_understands_ranges_and_negations():
    assert route_alma_science_archive_query(None, 'Explain the Sun over solar cycle 11') is None
    assert route_alma_science_archive_query(None, 'Compare 12CO and 13CO molecular lines in the ISM') is None
    args = route_alma_science_archive_query(None,
        'ALMA observations of Synthetic 4 in Bands 3-5 and Band 9, resolution under 0.5 arcsec, including non-public data '
        'from the ALMA Science Archive')
    assert args['band'] == [3, 4, 5, 9] and args['max_resolution_arcsec'] == 0.5
    assert 'public_only' not in args and 'science_only' not in args
    args = route_alma_science_archive_query(None, 'Public ALMA science observations of Synthetic 5, Band 6 to 7, under 200 mas')
    assert args['band'] == [6, 7] and args['public_only'] and args['science_only'] and args['max_resolution_arcsec'] == 0.2


def test_router_line_branch_uses_species_whitelist():
    args = route_alma_science_archive_query(None,
        'Which ALMA projects observed HCO+ and HCN lines toward 3C 273 and 4U 1630-47 at 230GHz?')
    assert args['query_type'] == 'line_set_projects' and args['lines'] == ['HCO+', 'HCN']
    args = route_alma_science_archive_query(None, 'Which ALMA projects observed 12CO, 13CO and C18O lines in Band 6?')
    assert args['lines'] == ['12CO', '13CO', 'C18O'] and args['band'] == [6]
    assert route_alma_science_archive_query(None, 'ALMA projects with 2MASS and 3C lines') is None


def test_reasoning_retry_budget_resets_after_a_productive_round():
    responses = _Responses([_events(1, reasoning='r'), _events(2, tool='query_archive'),
                            _events(3, reasoning='r'), _events(4, reasoning='r'), _events(5, text='Done.')])
    agent, executed = _tool_agent(responses)
    assert 'Done.' in agent.stream_response_api('hello there', conversation_id='reset')
    assert len(executed) == 1 and len(responses.calls) == 5


def test_sia_failed_preferred_endpoint_with_empty_fallback_is_indeterminate(monkeypatch):
    client = DatalabSiaClient()
    monkeypatch.setattr(client, '_candidate_endpoints', lambda **kw: ['https://sia.test/preferred', 'https://sia.test/fallback'])
    def search(ep, *args):
        if ep.endswith('preferred'):
            raise RuntimeError('502 bad gateway')
        return []
    monkeypatch.setattr(client, '_search_endpoint', search)
    out = client.search(25, -10, .1, catalog='synthetic')
    assert out['coverage_gap'] and out['partial'] and out['coverage_status'] == 'unknown'
    assert out['endpoint_errors'][0]['endpoint'].endswith('preferred')


def test_deepest_images_follow_requested_band_order():
    rows = [dict(obs_bandpass=b, exptime=str(e), proctype='Stack', prodtype='image', access_url=b)
            for b, e in (('y', 10), ('z', 20), ('x', 30))]
    assert [r['obs_bandpass'] for r in DatalabImageService.deepest_archive_images(rows, ['x', 'y', 'z'])] == ['x', 'y', 'z']
    assert [r['obs_bandpass'] for r in DatalabImageService.deepest_archive_images(rows)] == ['x', 'y', 'z']


@pytest.mark.parametrize('exc', [RuntimeError('HTTP Error 503: Service Unavailable'), RuntimeError('429 Too Many Requests'),
                                 RuntimeError('remote end closed connection without response')])
def test_tap_retry_classification_reads_status_text(exc):
    from integrations.alma_tap import _retryable
    assert _retryable(exc)
    assert not _retryable(ValueError('unknown column x'))
    wrapped = RuntimeError('query failed')
    try:
        try:
            raise exc
        except Exception:
            raise wrapped
    except RuntimeError as chained:
        assert _retryable(chained)  # found through __context__


def test_forced_round_escalates_to_strict_required_only_after_a_reasoning_only_stop():
    responses = _Responses([_events(1, reasoning='r'), _events(2, tool='query_archive'), _events(3, text='Done.')])
    agent, executed = _tool_agent(responses)
    result = agent.stream_response_api('Show me an image of Synthetic 4', conversation_id='strict')  # imagery class -> forced
    assert 'Done.' in result and len(executed) == 1
    assert responses.calls[0]['tool_choice'] == 'required' and responses.calls[0]['tool_choice_strict'] is False
    assert responses.calls[1]['tool_choice'] == 'required' and responses.calls[1]['tool_choice_strict'] is True


def test_science_cone_has_no_server_side_sort():
    assert 'ORDER BY' not in science_cone_query(25, -10, 60, band=[4], max_resolution_arcsec=1.0)


def test_router_science_flag_tolerates_markdown():
    args = route_alma_science_archive_query(None, 'ALMA **public, science** observations of Synthetic 4, Band 6, under 1 arcsec')
    assert args['public_only'] and args['science_only']


def test_link_guard_recognises_urls_inside_json_encoded_tool_outputs():
    """Runner evidence is json.dumps() of already-serialised tool outputs, so
    every URL is followed by an escaped quote; the guard must not treat the
    trailing backslash as part of the URL and strip legitimate tool links."""
    from core.agent import QuasarAgent
    url = 'https://archive.test/svc/cutout?col=&siaRef=x_g_a1.fits.fz&extn=7&POS=25.1,-10.2&SIZE=0.1,0.1'
    tool_output = json.dumps({'success': True, 'deepest_images': [{'access_url': url, 'exptime': '270'}]})
    evidence = json.dumps([{'type': 'function_call_output', 'call_id': 'c1', 'output': tool_output}], default=str)
    agent = QuasarAgent.__new__(QuasarAgent)
    text = 'Use this image: ' + url + '\n\n```json\n[{"access_url": "' + url + '"}]\n```'
    assert agent._strip_unverified_urls(text, sources=[evidence]) == text
    fabricated = text + '\nSee also https://example.invalid/made-up'
    cleaned = agent._strip_unverified_urls(fabricated, sources=[evidence])
    body, _, notice = cleaned.partition('> 🔗 Removed')
    # The fabricated link is gone from the body; the notice names EXACTLY what
    # was removed (UI benchmark 2026-09-22 D13/D14: counts that did not match).
    assert 'made-up' not in body and url in body
    assert notice.startswith(' 1 external link') and '`https://example.invalid/made-up`' in notice


def _inventory_rows():
    def row(band, exptime, proctype, prodtype, program, url):
        return dict(obs_bandpass=band, exptime=exptime, proctype=proctype, prodtype=prodtype,
                    obs_collection=program, access_url=url, instrument_name='Cam')
    return [row('g', '30', 'Raw', 'image', 'P-1', 'u1'), row('g', '30', 'Raw', 'image', 'P-1', 'u2'),
            row('g', '60', 'InstCal', 'image', 'P-1', 'u3'), row('g', '270', 'Stack', 'image', 'P-2', 'u4'),
            row('r', '270', 'Stack', 'image', 'P-2', 'u5'), row('r', '900', 'Stack', 'wtmap', 'P-2', 'u6'),
            row('z', '240', 'Stack', 'image', 'P-3', 'u7'), row('z', '300', 'Stack', 'image', 'P-3', 'u8')]


def test_inventory_summary_reports_product_types_and_per_band_stack_depth():
    inv = DatalabImageService.summarize_inventory(_inventory_rows())
    assert inv['rows_total'] == 8
    assert inv['products'][0] == {'proctype': 'Stack', 'prodtype': 'image', 'count': 4}
    assert {'proctype': 'Raw', 'prodtype': 'image', 'count': 2} in inv['products']
    assert inv['bands'] == {'g': 4, 'r': 2, 'z': 2}
    assert inv['stack_image_bands'] == ['g', 'r', 'z']
    assert inv['stack_images_per_band']['z'] == {'count': 2, 'max_exptime': 300.0,
        'deepest': {'exptime': '300', 'access_url': 'u8', 'program': 'P-3', 'instrument': 'Cam'}}
    assert inv['stack_images_per_band']['r']['count'] == 1  # the weight map is not an image product
    assert inv['programs'] == {'P-1': 3, 'P-2': 3, 'P-3': 2}


def test_sia_inventory_result_carries_summary_and_collection():
    from capabilities.datalab import SiaSearch, SiaSearchInput
    from tests.unit.test_datalab_capability import _img_ctx
    service = NS(search=lambda *a, **kw: {'rows': _inventory_rows(), 'coverage_gap': False, 'used_endpoint': kw['endpoint']})
    out = SiaSearch().run(SiaSearchInput(ra=25, dec=-10), _img_ctx(service)).to_native()
    assert out['collection'] == 'nsa' and out['selection'] == 'inventory'
    assert out['inventory']['stack_image_bands'] == ['g', 'r', 'z']
    assert out['inventory']['stack_images_per_band']['g']['max_exptime'] == 270.0
    assert 'stack_images_per_band' in out['note'] and 'different product family' in out['note']
    deepest = SiaSearch().run(SiaSearchInput(ra=25, dec=-10, deepest_per_band=True, bands=['g', 'z']), _img_ctx(service)).to_native()
    assert [r['access_url'] for r in deepest['deepest_images']] == ['u4', 'u8']
    assert deepest['inventory']['rows_total'] == 8  # summary describes the whole inventory, not the selection


def test_failed_mirror_is_tried_last_until_its_cooldown_expires(monkeypatch):
    monkeypatch.setattr(AlmaTapService, '_cooldown', {})
    monkeypatch.setattr('integrations.alma_tap.time.sleep', lambda s: None)
    clock = [1000.0]
    monkeypatch.setattr('integrations.alma_tap.time.monotonic', lambda: clock[0])
    calls = []
    def factory(url, session):
        def search(query, maxrec):
            calls.append(url)
            if url == ALMA_TAP_MIRRORS[0] and clock[0] < 1500:  # primary is broken only during the first window
                raise requests.HTTPError(response=NS(status_code=502))
            return _Result(pd.DataFrame([{'value': 1}]))
        return NS(search=search)
    monkeypatch.setattr('pyvo.dal.TAPService', factory)
    first = AlmaTapService().search('SELECT TOP 1 * FROM ivoa.obscore')
    assert calls == [ALMA_TAP_MIRRORS[0], ALMA_TAP_MIRRORS[1]] and first.quasar_tap_url == ALMA_TAP_MIRRORS[1]
    calls.clear()
    second = AlmaTapService().search('SELECT TOP 1 * FROM ivoa.obscore')   # the failed mirror is skipped, no wasted attempt
    assert calls == [ALMA_TAP_MIRRORS[1]] and second.quasar_attempts == 1
    clock[0] = 1000.0 + AlmaTapService.MIRROR_COOLDOWN_S + 1   # cooldown over: primary mirror is tried first again
    calls.clear()
    AlmaTapService().search('SELECT TOP 1 * FROM ivoa.obscore')
    assert calls[0] == ALMA_TAP_MIRRORS[0]
    # a non-transient error (bad ADQL) never penalises a mirror
    monkeypatch.setattr('pyvo.dal.TAPService', lambda *a, **kw: NS(search=lambda q, maxrec: (_ for _ in ()).throw(ValueError('bad column'))))
    with pytest.raises(RuntimeError):
        AlmaTapService().search('SELECT nope FROM ivoa.obscore')
    assert AlmaTapService._cooldown == {}

