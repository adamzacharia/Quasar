"""Stream-level runner tests for core/turn_guards.py wiring (guard task-efccf37-126 CX-21).

A scripted fake Responses client drives agent.stream_response_api through real
rounds: a tool call whose name carries gpt-oss harmony residue, then forced
final rounds whose text leaks reasoning. Harness shape follows
tests/unit/test_token_budget_runner.py.
"""
import json
import threading
from types import SimpleNamespace

import pytest

from core.agent import QuasarAgent
from core import turn_guards as tg

LEAK = ("The user asks: which paper reported it? We need to retrieve papers from ADS. "
        "Bibcode: 2020Natur.586..37G (I think). Actually maybe not.")


def _tool(name):
    return {"type": "function", "name": name, "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}


def _call_events(n, name, args):
    item = SimpleNamespace(type="function_call", call_id=f"call_{n}", id=f"item_{n}", name=name,
                           arguments="")
    return [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id=f"resp_{n}")),
        SimpleNamespace(type="response.output_item.added", item=item),
        SimpleNamespace(type="response.function_call_arguments.delta", call_id=f"call_{n}",
                        delta=json.dumps(args)),
        SimpleNamespace(type="response.output_item.done",
                        item=SimpleNamespace(type="function_call", call_id=f"call_{n}", id=f"item_{n}")),
    ]


def _text_events(n, text):
    return [
        SimpleNamespace(type="response.created", response=SimpleNamespace(id=f"resp_{n}")),
        SimpleNamespace(type="response.output_text.delta", delta=text),
    ]


class _Scripted:
    def __init__(self, agent_ref, script):
        self.script = list(script)  # each: ("call", name, args) or ("text", str)
        self.calls = []
        self.agent_ref = agent_ref

    def create(self, **kwargs):
        n = len(self.calls) + 1
        agent = self.agent_ref[0]
        self.calls.append({"kwargs": kwargs, "stored_head": agent._get_response_id("conv-lit", kwargs.get("model"))})
        step = self.script[min(n, len(self.script)) - 1]
        if step[0] == "call":
            return iter(_call_events(n, step[1], step[2]))
        if step[0] == "calls":
            events = [SimpleNamespace(type="response.created", response=SimpleNamespace(id=f"resp_{n}"))]
            for k, (name, args) in enumerate(step[1]):
                events += _call_events(f"{n}_{k}", name, args)[1:]
            return iter(events)
        return iter(_text_events(n, step[1]))

    def clear_history(self, response_id=None):
        pass

    def note_untrusted_call(self, call_id):
        pass

    def history_tokens_after(self, rid):
        return None


def _make_agent(script, tool_names=("search_papers",)):
    agent = object.__new__(QuasarAgent)
    ref = [agent]
    fake = _Scripted(ref, script)
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
    reg = {n: SimpleNamespace(name=n, description=f"{n} tool", parameters={}) for n in tool_names}
    agent.tool_registry = SimpleNamespace(list_tools=lambda: list(reg.values()), get_tool=lambda n: reg.get(n),
                                          names=lambda: list(reg))
    executed = []

    def _execute(tool, args, tool_name=None, step_label=None, on_status=None):
        executed.append((tool_name, dict(args)))
        if tool_name == "find_citing_papers":
            return {"success": True, "bibcode": args.get("bibcode"), "count": 1,
                    "rebuttals": [{"bibcode": "2021NatAs...5..631V", "title": "No evidence"}], "replies": []}
        return {"success": True, "count": 1,
                "papers": [{"bibcode": "2021NatAs...5..655G", "title": "Phosphine gas", "citations": 220}]}

    agent._execute_tool_with_progress = _execute
    return agent, fake, executed


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("QUASAR_TOOL_PACKS", "0")
    monkeypatch.setenv("QUASAR_ONESHOT_DISPATCH", "0")


def test_harmony_mangled_tool_name_is_repaired_and_executed():
    agent, fake, executed = _make_agent([
        ("call", "search_papers<|channel|>commentary", {"query": "phosphine Venus"}),
        ("text", "Greaves et al. reported it: 2021NatAs...5..655G."),
    ])
    steps = []
    out = agent.stream_response_api("tell me about it", conversation_id="conv-lit", model="gpt-oss-120b",
                                    on_status=lambda text, state: steps.append(text))
    assert executed == [("search_papers", {"query": "phosphine Venus"})]
    assert "2021NatAs...5..655G" in out
    assert not any("<|" in s for s in steps), steps


def test_forced_round_leak_is_resampled_from_the_pre_round_head(monkeypatch):
    monkeypatch.setenv("QUASAR_MAX_TOOL_ROUNDS", "1")
    agent, fake, executed = _make_agent([
        ("call", "search_papers", {"query": "phosphine Venus"}),
        ("text", LEAK),
        ("text", "**Detection**: Greaves et al., 2021NatAs...5..655G."),
    ])
    tokens, steps = [], []
    out = agent.stream_response_api("tell me about it", conversation_id="conv-lit", model="gpt-oss-120b",
                                    on_token=tokens.append, on_status=lambda text, state: steps.append(text))
    assert out.strip() == "**Detection**: Greaves et al., 2021NatAs...5..655G."
    assert "The user asks" not in "".join(tokens), "the leaked text never reached the user"
    assert len(fake.calls) == 3
    forced, resample = fake.calls[1]["kwargs"], fake.calls[2]["kwargs"]
    assert forced.get("tool_choice") == "none" and resample.get("tool_choice") == "none"
    assert resample["previous_response_id"] == forced["previous_response_id"]
    assert tg.LEAK_RESAMPLE_NOTE in json.dumps(resample["input"])
    budget_steps = [s for s in steps if s.startswith("Tool budget reached")]
    assert budget_steps and set(budget_steps) == {"Tool budget reached (all 1 tool rounds used), "
                                                  "composing the final answer from collected results"}
    assert len(budget_steps) == 2, "one running + one completed emission, not repeated on the re-sample"


def test_second_leak_is_stripped(monkeypatch):
    monkeypatch.setenv("QUASAR_MAX_TOOL_ROUNDS", "1")
    agent, fake, executed = _make_agent([
        ("call", "search_papers", {"query": "phosphine Venus"}),
        ("text", LEAK),
        ("text", LEAK + "\n\n**Detection**: 2021NatAs...5..655G."),
    ])
    out = agent.stream_response_api("tell me about it", conversation_id="conv-lit", model="gpt-oss-120b")
    assert out.strip() == "**Detection**: 2021NatAs...5..655G."
    assert len(fake.calls) == 3, "one re-sample only"


def test_clean_forced_answer_streams_without_resample(monkeypatch):
    monkeypatch.setenv("QUASAR_MAX_TOOL_ROUNDS", "1")
    answer = "Greaves et al. (2021NatAs...5..655G) reported phosphine. " * 12
    agent, fake, executed = _make_agent([
        ("call", "search_papers", {"query": "phosphine Venus"}),
        ("text", answer),
    ])
    tokens = []
    out = agent.stream_response_api("tell me about it", conversation_id="conv-lit", model="gpt-oss-120b",
                                    on_token=tokens.append)
    assert out.strip() == answer.strip() and len(fake.calls) == 2
    assert "".join(tokens).strip().startswith("Greaves et al.")


PUSHBACK_Q = "Which paper first reported phosphine on Venus, and which papers pushed back on it?"


def test_pushback_question_queues_a_real_find_citing_call(monkeypatch):
    agent, fake, executed = _make_agent([
        ("call", "search_papers", {"query": "phosphine detection Venus"}),
        ("text", "Greaves et al. 2021NatAs...5..655G; rebutted by 2021NatAs...5..631V."),
    ], tool_names=("search_papers", "find_citing_papers"))
    steps = []
    agent.stream_response_api(PUSHBACK_Q, conversation_id="conv-lit", model="gpt-oss-120b",
                              on_status=lambda text, state: steps.append(text))
    # executed through the normal tool path, after the search
    assert [e[0] for e in executed] == ["search_papers", "find_citing_papers"]
    assert executed[1][1] == {"bibcode": "2021NatAs...5..655G", "focus": "rebuttals", "max_results": 12}
    sent = fake.calls[1]["kwargs"]["input"]
    kinds = [(i.get("type"), i.get("name")) for i in sent if isinstance(i, dict) and i.get("type")]
    # the provider sees the synthetic call right before its output
    k = kinds.index(("function_call", "find_citing_papers"))
    assert kinds[k + 1][0] == "function_call_output" and "2021NatAs...5..631V" in json.dumps(sent[k + 1])
    assert any(s.startswith("Finding papers that responded to 2021NatAs...5..655G") for s in steps)


def test_bench_allowlist_blocks_the_synthetic_call(monkeypatch):
    monkeypatch.setenv("QUASAR_BENCH_TOOL_ALLOWLIST", "search_papers")
    agent, fake, executed = _make_agent([
        ("call", "search_papers", {"query": "phosphine detection Venus"}),
        ("text", "Answer."),
    ], tool_names=("search_papers", "find_citing_papers"))
    agent.stream_response_api(PUSHBACK_Q, conversation_id="conv-lit", model="gpt-oss-120b")
    assert [e[0] for e in executed] == ["search_papers"]


def test_no_duplicate_when_the_model_already_called_it(monkeypatch):
    agent, fake, executed = _make_agent([
        ("calls", [("search_papers", {"query": "phosphine detection Venus"}),
                   ("find_citing_papers", {"bibcode": "2021NatAs...5..655G"})]),
        ("text", "Answer."),
    ], tool_names=("search_papers", "find_citing_papers"))
    agent.stream_response_api(PUSHBACK_Q, conversation_id="conv-lit", model="gpt-oss-120b")
    assert [e[0] for e in executed] == ["search_papers", "find_citing_papers"]


def test_auto_citing_kill_switch(monkeypatch):
    import core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "_LIT_AUTO_CITING", False)
    agent, fake, executed = _make_agent([
        ("call", "search_papers", {"query": "phosphine detection Venus"}),
        ("text", "Answer."),
    ], tool_names=("search_papers", "find_citing_papers"))
    agent.stream_response_api(PUSHBACK_Q, conversation_id="conv-lit", model="gpt-oss-120b")
    assert [e[0] for e in executed] == ["search_papers"]


def test_repaired_name_counts_as_already_called():
    agent, fake, executed = _make_agent([
        ("calls", [("search_papers", {"query": "phosphine detection Venus"}),
                   ("find_citing_papers<|channel|>commentary", {"bibcode": "2021NatAs...5..655G"})]),
        ("text", "Answer."),
    ], tool_names=("search_papers", "find_citing_papers"))
    agent.stream_response_api(PUSHBACK_Q, conversation_id="conv-lit", model="gpt-oss-120b")
    assert [e[0] for e in executed] == ["search_papers", "find_citing_papers"]  # CX-13


def test_failed_synthetic_lookup_closes_as_error_and_is_traced():
    agent, fake, executed = _make_agent([
        ("call", "search_papers", {"query": "phosphine detection Venus"}),
        ("text", "Answer."),
    ], tool_names=("search_papers", "find_citing_papers"))
    base = agent._execute_tool_with_progress

    def _execute(tool, args, tool_name=None, step_label=None, on_status=None):
        if tool_name == "find_citing_papers":
            executed.append((tool_name, dict(args)))
            return {"success": False, "error": "ADS API error 503"}
        return base(tool, args, tool_name=tool_name, step_label=step_label, on_status=on_status)

    agent._execute_tool_with_progress = _execute
    traced = []
    agent._record_tool_trace = lambda name, args, result, **kw: traced.append(name)
    states = []
    agent.stream_response_api(PUSHBACK_Q, conversation_id="conv-lit", model="gpt-oss-120b",
                              on_status=lambda text, state: states.append((text, state)))
    closes = [st for t, st in states if t.startswith("Finding papers that responded to") and st != "running"]
    assert closes == ["error"]  # CX-12
    assert "find_citing_papers" in traced  # CX-10: in the tool trace


def test_no_synthetic_call_for_anthropic(monkeypatch):
    agent, fake, executed = _make_agent([
        ("call", "search_papers", {"query": "phosphine detection Venus"}),
        ("text", "Answer."),
    ], tool_names=("search_papers", "find_citing_papers"))
    agent.config.model = "claude-sonnet-5"
    agent.stream_response_api(PUSHBACK_Q, conversation_id="conv-lit", model="claude-sonnet-5")
    assert [e[0] for e in executed] == ["search_papers"]  # CX-18
