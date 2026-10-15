"""Runner-level regressions for the token plan (guard task-5314487-21564 CX-13):
the lack-of-tool re-sample with a saved chain head, and one-shot direct
dispatch. Uses the fake-responses harness of test_response_api_error_recovery."""
import threading
from types import SimpleNamespace

from core.agent import QuasarAgent
from core import tool_packs as tp


def _tool(name):
    return {"type": "function", "name": name, "description": f"{name} tool", "parameters": {"type": "object", "properties": {}}}


class _Scripted:
    """Each create() streams the next scripted (text) and records what it was sent
    plus the agent's stored chain head at that moment."""

    def __init__(self, agent_ref, texts):
        self.texts = list(texts)
        self.calls = []
        self.agent_ref = agent_ref

    def create(self, **kwargs):
        n = len(self.calls) + 1
        agent = self.agent_ref[0]
        self.calls.append({"kwargs": kwargs,
                           "stored_head": agent._get_response_id("conv-tp", kwargs.get("model"))})
        text = self.texts[min(n, len(self.texts)) - 1]
        return iter([
            SimpleNamespace(type="response.created", response=SimpleNamespace(id=f"resp_{n}")),
            SimpleNamespace(type="response.output_text.delta", delta=text),
        ])

    def clear_history(self, response_id=None):
        pass

    def note_untrusted_call(self, call_id):
        pass

    def history_tokens_after(self, rid):
        return None


def _make_agent(texts, tool_names, registry_tools=None, execute=None):
    agent = object.__new__(QuasarAgent)
    ref = [agent]
    fake = _Scripted(ref, texts)
    agent._tls = threading.local()
    agent._conv_response_ids = {}
    agent._conv_run_tokens = {}
    agent._conv_ids_lock = threading.Lock()
    agent.config = SimpleNamespace(model="gpt-oss-120b", temperature=0.2, max_tokens=2048, verbose=False)
    agent.system_prompt = "system prompt"
    agent.ads_client = None
    agent.long_term_memory = None
    agent.client = SimpleNamespace(responses=fake)
    agent._prune_session_if_needed = lambda *a, **k: None
    agent.memory = SimpleNamespace(get_history=lambda: [], max_memory_turns=10)
    agent._build_tools_for_responses_api = lambda: [_tool(n) for n in tool_names]
    reg = {n: SimpleNamespace(name=n, description=f"{n} tool", parameters={}) for n in (registry_tools or tool_names)}
    agent.tool_registry = SimpleNamespace(list_tools=lambda: list(reg.values()), get_tool=lambda n: reg.get(n))
    if execute is not None:
        agent._execute_tool_with_progress = execute
    return agent, fake


def test_lack_of_tool_round_is_resampled_with_every_tool_and_chain_head_restored(monkeypatch):
    monkeypatch.setenv("QUASAR_TOOL_PACKS", "1")
    names = [n for n in tp.CORE if n != tp.FIND_TOOLS] + ["moving_object_check"]
    agent, fake = _make_agent(["I don't have a tool for listing minor bodies.", "Final answer."], names)
    agent._set_response_id("conv-tp", "prev-resp", "gpt-oss-120b")

    out = agent.stream_response_api("hello there", conversation_id="conv-tp", model="gpt-oss-120b")

    assert out.strip() == "Final answer."
    assert len(fake.calls) == 2
    first = {t["name"] for t in fake.calls[0]["kwargs"]["tools"]}
    second = {t["name"] for t in fake.calls[1]["kwargs"]["tools"]}
    assert "moving_object_check" not in first and "moving_object_check" in second
    assert tp.FIND_TOOLS in first
    # the re-sample branches from the pre-round chain head, stored and sent
    assert fake.calls[1]["stored_head"] == "prev-resp"
    assert fake.calls[1]["kwargs"]["previous_response_id"] == "prev-resp"


def test_direct_dispatch_runs_the_tool_and_sends_one_exchange(monkeypatch):
    monkeypatch.setenv("QUASAR_TOOL_PACKS", "1")
    monkeypatch.setenv("QUASAR_ONESHOT_DISPATCH", "1")
    executed = []

    def _execute(tool, args, tool_name=None, step_label=None, on_status=None):
        executed.append((tool_name, dict(args)))
        return {"success": True, "bands_with_public_data": 9}

    names = [n for n in tp.CORE if n != tp.FIND_TOOLS] + ["alma_public_band_status"]
    agent, fake = _make_agent(["Nine bands have public data."], names, execute=_execute)
    q = "How many ALMA bands have public data as of 2026-01-01?"

    out = agent.stream_response_api(q, conversation_id="conv-tp", model="gpt-oss-120b")

    assert executed == [("alma_public_band_status", {"as_of": "2026-01-01"})]
    assert len(fake.calls) == 1, "one model call instead of two"
    inp = fake.calls[0]["kwargs"]["input"]
    assert isinstance(inp, list) and inp[0].get("role") == "user"
    calls = [i for i in inp if i.get("type") == "function_call"]
    outs = [i for i in inp if i.get("type") == "function_call_output"]
    assert [c["name"] for c in calls] == ["alma_public_band_status"]
    assert [o["call_id"] for o in outs] == [c["call_id"] for c in calls]
    assert "ALREADY called" in inp[0]["content"] and "You MUST call `alma_public_band_status`" not in inp[0]["content"]
    assert "Nine bands" in out


def test_runner_marks_the_last_core_tool_for_anthropic_and_openai_only(monkeypatch):
    from core.llm_client import CACHE_BREAKPOINT_KEY

    monkeypatch.setenv("QUASAR_TOOL_PACKS", "1")
    names = [n for n in tp.CORE if n != tp.FIND_TOOLS] + ["query_alma_science_archive"]
    q = "How many ALMA Band 6 projects observed HD 163296?"
    for model, expect in (("claude-haiku-5-5", True), ("gpt-6-luna", True), ("gpt-oss-120b", False),
                          ("deepseek-v4-flash", False)):
        agent, fake = _make_agent(["Done."], names)
        agent.stream_response_api(q, conversation_id="conv-tp", model=model)
        kw = fake.calls[0]["kwargs"]
        assert kw["_instructions_static_len"] == len("system prompt"), "static system prompt length is passed"
        assert kw["instructions"].startswith("system prompt")
        tools = kw["tools"]
        marked = [t["name"] for t in tools if CACHE_BREAKPOINT_KEY in t]
        n_core = tp.core_prefix_len(tools)
        assert "query_alma_science_archive" in [t["name"] for t in tools][n_core:], "a pack follows the core"
        if expect:
            assert marked == [tools[n_core - 1]["name"]] == ["calculate_doppler_shift"], model
        else:
            assert marked == [], f"{model}: no marker outside Anthropic"


def test_anthropic_marker_survives_toolset_growth_and_never_touches_shared_dicts(monkeypatch):
    """CX-04: the full-toolset re-sample appends through the same _tp_offer path
    as a find_tools expansion; the marker must stay on the last core tool, and
    the registry's own dicts must never be marked."""
    from core.llm_client import CACHE_BREAKPOINT_KEY

    monkeypatch.setenv("QUASAR_TOOL_PACKS", "1")
    names = [n for n in tp.CORE if n != tp.FIND_TOOLS] + ["query_alma_science_archive", "moving_object_check"]
    originals = [_tool(n) for n in names]
    agent, fake = _make_agent(["I don't have a tool for listing minor bodies.", "Final answer."], names)
    agent._build_tools_for_responses_api = lambda: list(originals)  # the same dict objects every time

    out = agent.stream_response_api("How many ALMA Band 6 projects observed HD 163296?",
                                    conversation_id="conv-tp", model="claude-haiku-5-5")

    assert out.strip() == "Final answer." and len(fake.calls) == 2
    first, second = (c["kwargs"]["tools"] for c in fake.calls)
    assert "moving_object_check" not in [t["name"] for t in first]
    assert "moving_object_check" in [t["name"] for t in second], "re-sample offered more tools"
    for tools in (first, second):
        n_core = tp.core_prefix_len(tools)
        assert [t["name"] for t in tools if CACHE_BREAKPOINT_KEY in t] == ["calculate_doppler_shift"]
        assert tools[n_core - 1]["name"] == "calculate_doppler_shift", "marker stays at the core boundary"
    assert second[: tp.core_prefix_len(first)] == first[: tp.core_prefix_len(first)], "core prefix unchanged"
    assert all(CACHE_BREAKPOINT_KEY not in t for t in originals), "shared tool dicts are never mutated"
    assert all(t is not o for t in first for o in originals if CACHE_BREAKPOINT_KEY in t)
