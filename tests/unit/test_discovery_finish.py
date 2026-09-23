"""Follow-up, provider-finalization and archive collection/cap regressions."""
import copy
import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from capabilities.datalab import SiaSearch, SiaSearchInput
from core.llm_client import LLMClient
from integrations.alminer_client import ALminerClient
from integrations.datalab_sia_client import DatalabSiaClient
from tests.unit.test_datalab_capability import _img_ctx
from tests.unit.test_discovery_recovery import _Result, _tool_agent


@pytest.mark.parametrize('selection,catalog,endpoint', [
    ({}, None, 'https://datalab.noirlab.edu/sia/nsa'),
    ({'catalog':'ls_dr9'}, 'ls_dr9', None),
    ({'service':'coadd_all'}, None, 'https://datalab.noirlab.edu/sia/coadd_all'),
    ({'endpoint':'https://archive.test/sia'}, None, 'https://archive.test/sia'),
])
def test_inventory_collection_defaults_are_generic_and_explicit_choices_win(selection,catalog,endpoint):
    calls=[]
    def search(*args,**kwargs):
        calls.append(kwargs)
        return {'rows':[], 'coverage_gap':True, 'used_endpoint':kwargs.get('endpoint')}
    result=SiaSearch().run(SiaSearchInput(ra=25,dec=-10,**selection),_img_ctx(NS(search=search))).to_native()
    assert result['success']
    assert calls==[{'catalog':catalog,'endpoint':endpoint}]


def test_deadline_clears_unresolved_multicall_head_before_followup(monkeypatch):
    clock=[0.]
    monkeypatch.setenv('CHAT_HARD_MAX_TIMEOUT_SECONDS','600')
    monkeypatch.setattr('core.runner.time.monotonic',lambda:clock[0])
    calls=[]
    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls)==1:
            delta=NS(content=None,tool_calls=[NS(index=i,id=f'call-{i}',function=NS(name='query_archive',arguments=json.dumps({'x':i}))) for i in range(2)])
        else:
            delta=NS(content='Follow-up complete.',tool_calls=None)
        return iter([NS(choices=[NS(delta=delta,finish_reason='stop')])])
    client=LLMClient(model='gpt-oss-120b')
    client._get_tacc_client=lambda:NS(chat=NS(completions=NS(create=create)))
    agent,executed=_tool_agent(client.responses)
    def execute(*a,**kw):
        executed.append(kw)
        clock[0]=601.
        return {'success':True,'results':[{'value':42}]}
    agent._execute_tool_with_progress=execute
    result=json.loads(agent.stream_response_api('hello there',conversation_id='deadline-followup'))
    assert result['partial'] and len(executed)==1
    assert agent._get_response_id('deadline-followup','gpt-oss-120b') is None
    assert not client.responses._history_cache
    assert 'Follow-up complete.' in agent.stream_response_api('continue please',conversation_id='deadline-followup')
    assert all(m['role'] != 'tool' and not m.get('tool_calls') for m in calls[-1]['messages'])


@pytest.mark.parametrize('stream',[False,True])
def test_google_forced_final_disables_function_calls(stream):
    captured=[]
    def generate(**kwargs):
        captured.append(kwargs)
        result=NS(text='Finished.',candidates=[],usage_metadata=None)
        return iter([result]) if stream else result
    client=LLMClient(model='gemini-2.5-flash')
    client._get_google_client=lambda:NS(models=NS(generate_content=generate,generate_content_stream=generate))
    kwargs=dict(model='gemini-2.5-flash',input='Answer now',tools=[{'type':'function','name':'query_archive',
        'description':'Query','parameters':{'type':'object','properties':{}}}],tool_choice='none')
    if stream:list(client.responses._stream_google(kwargs))
    else:client.responses._call_google(kwargs)
    from google.genai.types import GenerateContentConfig
    config=GenerateContentConfig(**captured[0]['config'])
    assert config.tool_config.function_calling_config.mode.value == 'NONE'
    assert config.tools


@pytest.mark.parametrize('stream',[False,True])
def test_local_forced_final_passes_none_to_chat_api(stream):
    captured=[]
    def create(**kwargs):
        captured.append(kwargs)
        if stream:return iter([NS(choices=[NS(delta=NS(content='Done.',tool_calls=None))])])
        return NS(id='local-response',choices=[NS(message=NS(content='Done.',tool_calls=None),finish_reason='stop')],usage=None)
    client=LLMClient(model='local/example')
    client._get_local_client=lambda:NS(chat=NS(completions=NS(create=create)))
    kwargs=dict(model='local/example',input='Answer now',tools=[{'type':'function','name':'query_archive',
        'description':'Query','parameters':{'type':'object','properties':{}}}],tool_choice='none')
    if stream:list(client.responses._stream_local(kwargs))
    else:client.responses._call_local(kwargs)
    assert captured[0]['tool_choice']=='none' and captured[0]['tools']


@pytest.mark.parametrize('requested,expected',[(30000,20000),(-2,1),(0,1),(None,5000)])
@pytest.mark.parametrize('mode',['cone','frequency'])
def test_archive_query_transport_and_metadata_share_clamped_cap(monkeypatch,requested,expected,mode):
    calls=[]
    def search(query,*,maxrec):
        calls.append((query,maxrec))
        result=_Result(pd.DataFrame([{'target_name':'Synthetic A'}]))
        result.query_status='OVERFLOW'
        result.quasar_tap_url='https://archive.test/tap'
        return result
    client=ALminerClient()
    monkeypatch.setattr(client,'_get_tap_service',lambda:NS(search=search))
    if mode=='cone':frame=client.search_by_position(25,-10,max_results=requested)
    else:frame=client.search_by_frequency(80,110,max_results=requested)
    # A truncated cone also issues ONE COUNT(*) companion (N rows total, showing M;
    # UI benchmark 2026-09-22 D16) -- the row query is still the only row pull.
    row_calls=[c for c in calls if not c[0].startswith('SELECT COUNT(*)')]
    assert len(row_calls)==1 and f'SELECT TOP {expected} ' in row_calls[0][0]
    assert len(calls)-len(row_calls)==(1 if mode=='cone' else 0)
    assert row_calls[0][1]==expected and frame.attrs['row_cap']==expected
    assert frame.attrs['truncated']  # OVERFLOW applies even below the row cap.


def test_custom_adql_clamps_maxrec_and_discloses_overflow(monkeypatch):
    client=ALminerClient()
    def search(query,*,maxrec):
        assert maxrec==20000
        result=_Result(pd.DataFrame([{'target_name':'Synthetic A'}]))
        result.query_status='OVERFLOW'
        return result
    monkeypatch.setattr(client,'_get_tap_service',lambda:NS(search=search))
    frame=client.search_by_sql('SELECT * FROM ivoa.obscore',maxrec=100000)
    assert frame.attrs['truncated'] and frame.attrs['row_cap']==20000


def test_explicit_archive_endpoint_provenance_names_executed_collection(monkeypatch):
    client=DatalabSiaClient(catalog='ls_dr9')
    monkeypatch.setattr(client,'_search_endpoint',lambda *a:[])
    result=client.search(25,-10,.2,endpoint='https://datalab.noirlab.edu/sia/nsa')
    assert result['provenance']['collection']=='nsa'
    assert result['provenance']['catalog'] is None
