"""Cross-question prompt-cache layouts (2026-10-09, tmp/anthropic-cache-2026-10-09/REPORT.md).

OpenAI GPT-5.6+/GPT-6: tools = fixed core, static system prompt as a developer message with an
explicit cache breakpoint, per-turn tail as a second developer message, the pack as an
additional_tools item; chained calls add only what the chain lacks.
Anthropic: system prompt split into static + tail, non-core tools deferred and surfaced by a
mid-conversation tool_addition system message. Unmarked callers keep the old request.
Guard task-614ae75-751 regressions are marked CX-NN."""
import json
from types import SimpleNamespace as NS

import pytest

from core.llm_client import LLMClient, ResponsesShim
from core import call_tokens


def fn(name, desc=None, params=None):
    return {"type": "function", "name": name, "description": desc or f"{name} tool.",
            "parameters": params or {"type": "object", "properties": {}}}


CORE = [fn("find_tools"), fn("catalog_query")]
PACK = [fn("pack_a"), fn("pack_b")]
MCP = {"type": "mcp", "server_label": "alma", "server_url": "https://x", "require_approval": "never"}
LAYOUT = {"core_tool": "catalog_query", "static_len": len("STATIC"), "own_cache_key": True}
BETA = "mid-conversation-tool-changes-2026-07-01"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("QUASAR_OPENAI_CACHE_LAYOUT", "QUASAR_ANTHROPIC_DEFERRED_TOOLS", "QUASAR_ANTHROPIC_PROMPT_CACHE",
                "QUASAR_ANTHROPIC_CORE_BREAKPOINT", "QUASAR_ANTHROPIC_CORE_CACHE_TTL"):
        monkeypatch.delenv(var, raising=False)


# ── OpenAI ──────────────────────────────────────────────────────────────

def _okw(**over):
    kw = {"model": "gpt-6-luna", "instructions": "STATICTAIL", "input": "Q", "tools": CORE + PACK + [MCP],
          "prompt_cache_key": "quasar-pertoolset", "stream": True}
    kw.update(over)
    return kw


def _texts(items):
    return [b["text"] for i in items if i.get("role") == "developer" and isinstance(i.get("content"), list)
            for b in i["content"]]


def test_openai_first_request_layout():
    shim = LLMClient(model="gpt-6-luna").responses
    kw = _okw()
    before = json.dumps(kw, sort_keys=True)
    out, state = shim._openai_cache_layout(kw, LAYOUT)
    assert json.dumps(kw, sort_keys=True) == before, "caller's kwargs not mutated"
    assert "instructions" not in out
    assert out["tools"] == CORE + [MCP], "tools = fixed core (+ non-function tools)"
    dev_static, dev_tail, addl, user = out["input"]
    assert dev_static == {"role": "developer", "content": [
        {"type": "input_text", "text": "STATIC", "prompt_cache_breakpoint": {"mode": "explicit"}}]}
    assert dev_tail == {"role": "developer", "content": [{"type": "input_text", "text": "TAIL"}]}
    assert addl["type"] == "additional_tools" and addl["role"] == "developer"
    assert [t["name"] for t in addl["tools"]] == ["pack_a", "pack_b"]
    assert user == {"role": "user", "content": "Q"}
    assert [n for n, _ in state["tools"]] == ["pack_a", "pack_b"] and state["tail"] == "TAIL"
    # The cache key no longer depends on the pack: another question, another pack, same key.
    other, _ = shim._openai_cache_layout(_okw(tools=CORE + [fn("pack_z")] + [MCP]), LAYOUT)
    assert other["prompt_cache_key"] == out["prompt_cache_key"] != "quasar-pertoolset"


def test_openai_caller_cache_keys_are_kept_even_with_the_quasar_prefix():  # CX-04
    shim = LLMClient(model="gpt-6-luna").responses
    for key in ("tenant-7", "quasar-tenant-7"):
        own, _ = shim._openai_cache_layout(_okw(prompt_cache_key=key), dict(LAYOUT, own_cache_key=False))
        assert own["prompt_cache_key"] == key


def test_openai_chained_requests_add_only_what_the_chain_lacks():
    shim = LLMClient(model="gpt-6-luna").responses
    out, st = shim._openai_cache_layout(_okw(), LAYOUT)
    shim._openai_remember_chain("resp_1", st)
    fco = {"type": "function_call_output", "call_id": "c1", "output": "{}"}
    # Same tools, same tail: only the new input goes out.
    o2, st2 = shim._openai_cache_layout(_okw(previous_response_id="resp_1", input=[fco]), LAYOUT)
    assert o2["input"] == [fco] and o2["tools"] == CORE + [MCP] and "instructions" not in o2
    shim._openai_remember_chain("resp_2", st2)
    # find_tools added pack_c mid-turn: one additional_tools item with only it.
    o3, st3 = shim._openai_cache_layout(_okw(previous_response_id="resp_2", input=[fco],
                                             tools=CORE + PACK + [fn("pack_c")] + [MCP]), LAYOUT)
    assert [i.get("type") for i in o3["input"]] == ["additional_tools", "function_call_output"]
    assert [t["name"] for t in o3["input"][0]["tools"]] == ["pack_c"]
    assert [n for n, _ in st3["tools"]] == ["pack_a", "pack_b", "pack_c"]


def test_openai_changed_tails_supersede_the_earlier_ones():  # CX-06
    shim = LLMClient(model="gpt-6-luna").responses
    _, st = shim._openai_cache_layout(_okw(instructions="STATIC\nUSER MEMORY: preferred_units: K"), LAYOUT)
    shim._openai_remember_chain("r1", st)
    # The preference was deleted but the tail is still non-empty (memory policy line).
    o2, st2 = shim._openai_cache_layout(_okw(previous_response_id="r1", instructions="STATIC\nMEMORY POLICY ON",
                                             input="q2"), LAYOUT)
    msg = _texts(o2["input"])[0]
    assert msg.startswith("These turn-specific instructions replace all turn-specific instructions given earlier")
    assert msg.endswith("\nMEMORY POLICY ON") and "preferred_units" not in msg
    shim._openai_remember_chain("r2", st2)
    # The RAG scaffold disappears while a personalization tail remains: same rule.
    o3, st3 = shim._openai_cache_layout(_okw(previous_response_id="r2", instructions="STATIC\nPERSONAL", input="q3"),
                                        LAYOUT)
    assert "replace all turn-specific instructions" in _texts(o3["input"])[0]
    shim._openai_remember_chain("r3", st3)
    # No tail at all: an explicit retraction.
    o4, _ = shim._openai_cache_layout(_okw(previous_response_id="r3", instructions="STATIC", input="q4"), LAYOUT)
    assert "no longer apply" in _texts(o4["input"])[0]


def test_openai_changed_tool_definitions_are_resent():  # CX-07
    shim = LLMClient(model="gpt-6-luna").responses
    _, st = shim._openai_cache_layout(_okw(), LAYOUT)
    shim._openai_remember_chain("r1", st)
    for changed in (fn("pack_a", params={"type": "object", "properties": {"x": {"type": "string"}}}),
                    fn("pack_a", params={"type": "object", "properties": {}, "required": ["x"]}),
                    fn("pack_a", desc="A different description.")):
        o, _ = shim._openai_cache_layout(_okw(previous_response_id="r1", input="q",
                                              tools=CORE + [changed, PACK[1]] + [MCP]), LAYOUT)
        addl = [i for i in o["input"] if i.get("type") == "additional_tools"]
        assert len(addl) == 1 and addl[0]["tools"] == [changed], "only the changed tool, in its new form"
        assert "use only the new definition: pack_a" in " ".join(_texts(o["input"]))


def test_openai_chain_state_is_lru():  # CX-03
    shim = LLMClient(model="gpt-6-luna").responses
    shim._OPENAI_CHAIN_STATE_MAX = 2
    _, st = shim._openai_cache_layout(_okw(), LAYOUT)
    shim._openai_remember_chain("active", st)
    shim._openai_remember_chain("idle", st)
    shim._openai_cache_layout(_okw(previous_response_id="active", input="x"), LAYOUT)  # a read refreshes recency
    shim._openai_remember_chain("new", st)
    assert list(shim._openai_chain_state) == ["active", "new"]
    shim._openai_remember_chain("active", st)  # an update refreshes recency too
    shim._openai_remember_chain("newer", st)
    assert list(shim._openai_chain_state) == ["active", "newer"]


@pytest.mark.parametrize("case", ["unknown_chain", "old_model", "flag_off", "no_core", "gpt-4o"])
def test_openai_layout_falls_back_to_the_old_request(case, monkeypatch):
    shim = LLMClient(model="gpt-6-luna").responses
    kw = _okw()
    layout = dict(LAYOUT)
    if case == "unknown_chain":
        kw["previous_response_id"] = "resp_from_before_restart"
    elif case == "old_model":
        kw["model"] = "gpt-5.4"
    elif case == "gpt-4o":
        kw["model"] = "gpt-4o"
    elif case == "flag_off":
        monkeypatch.setenv("QUASAR_OPENAI_CACHE_LAYOUT", "0")
    else:
        layout["core_tool"] = "not_offered"
    out, state = shim._openai_cache_layout(kw, layout)
    assert out is kw and state is None


def test_openai_layout_supported_models():
    sup = ResponsesShim._openai_layout_supported
    assert sup("gpt-6-luna") and sup("gpt-6.1-sol") and sup("gpt-5.6") and sup("gpt-6-astra")
    assert not sup("gpt-5.5") and not sup("gpt-5") and not sup("gpt-4o") and not sup("o3") and not sup("claude-x")


class _FakeOpenAI:
    def __init__(self, fail_first=None):
        self.sent = []
        self.responses = self
        self.fail_first = fail_first

    def create(self, **kwargs):
        self.sent.append(json.loads(json.dumps(kwargs)))
        if self.fail_first is not None:
            hook, self.fail_first = self.fail_first, None
            hook()
        rid = f"resp_{len(self.sent)}"
        if kwargs.get("stream"):
            return iter([NS(type="response.created", response=NS(id=rid)),
                         NS(type="response.completed", response=NS(id=rid, usage=None))])
        return NS(id=rid, output=[], usage=None)


@pytest.mark.parametrize("stream", [False, True])
def test_call_openai_remembers_the_chain_and_keeps_old_calls_unchanged(stream, monkeypatch):
    shim = LLMClient(model="gpt-6-luna").responses
    fake = _FakeOpenAI()
    monkeypatch.setattr(shim._llm, "_get_openai_client", lambda: fake)
    r1 = shim._call_openai(_okw(stream=stream), cache_layout=LAYOUT)
    if stream:
        list(r1)
    r2 = shim._call_openai(_okw(stream=stream, previous_response_id="resp_1",
                                input=[{"type": "function_call_output", "call_id": "c", "output": "{}"}]),
                           cache_layout=LAYOUT)
    if stream:
        list(r2)
    assert fake.sent[0]["input"][0]["role"] == "developer"
    assert fake.sent[1]["input"] == [{"type": "function_call_output", "call_id": "c", "output": "{}"}], \
        "the chain from call 1 is known, so nothing is re-sent"
    # No layout argument: the old passthrough, byte for byte.
    shim._call_openai(_okw(stream=False))
    sent = fake.sent[-1]
    assert sent["instructions"] == "STATICTAIL" and sent["input"] == "Q" and sent["tools"] == CORE + PACK + [MCP]


def test_openai_retries_resend_the_identical_request(monkeypatch):  # CX-05
    shim = LLMClient(model="gpt-6-luna").responses
    _, st = shim._openai_cache_layout(_okw(), LAYOUT)
    shim._openai_remember_chain("parent", st)

    class Unavailable(Exception):
        status_code = 503

    def evict_then_fail():
        shim._openai_chain_state.clear()  # the parent is evicted between attempts
        raise Unavailable("503 Service Unavailable")

    fake = _FakeOpenAI(fail_first=evict_then_fail)
    monkeypatch.setattr(shim._llm, "_get_openai_client", lambda: fake)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    shim._call_openai(_okw(stream=False, previous_response_id="parent", input="q"), cache_layout=LAYOUT)
    assert len(fake.sent) == 2 and fake.sent[0] == fake.sent[1]
    assert "instructions" not in fake.sent[1] and fake.sent[1]["tools"] == CORE + [MCP]


def test_openai_cache_writes_reach_the_call_log():
    usage = NS(input_tokens=11357, output_tokens=5,
               input_tokens_details=NS(cached_tokens=8958, cache_write_tokens=2396))
    assert call_tokens._usage_fields(usage) == {"billed_in": 11357, "cached": 8958, "billed_out": 5,
                                                "cache_write": 2396}
    old = NS(input_tokens=10, output_tokens=1, input_tokens_details=NS(cached_tokens=0))
    assert call_tokens._usage_fields(old)["cache_write"] is None


# ── Anthropic ───────────────────────────────────────────────────────────

def _atools(shim, extra=()):
    return shim._translate_tools_for_anthropic(CORE + PACK + list(extra))


def _ackw(shim, **over):
    ck = {"model": "claude-haiku-5-5", "max_tokens": 100, "system": "STATICTAIL", "tools": _atools(shim),
          "messages": [{"role": "user", "content": "Q"}]}
    ck.update(over)
    return ck


def _all_marks(ck):
    n = sum(1 for t in ck.get("tools") or [] if "cache_control" in t)
    n += sum(1 for b in ck.get("system") or [] if isinstance(b, dict) and "cache_control" in b)
    n += sum(1 for m in ck.get("messages") or [] if isinstance(m.get("content"), list)
             for b in m["content"] if isinstance(b, dict) and "cache_control" in b)
    return n


def _valid(ck):
    """Every tool_reference in the history resolves to a deferred tool, and the beta is sent."""
    refs = ResponsesShim._anthropic_added_names(ck.get("messages") or [])
    deferred = {t["name"] for t in ck.get("tools") or [] if t.get("defer_loading")}
    if refs:
        assert set(refs) <= deferred, f"undeclared references: {set(refs) - deferred}"
        assert BETA in (ck.get("extra_headers") or {}).get("anthropic-beta", "")
    assert any(not t.get("defer_loading") for t in ck.get("tools") or [{}])


def test_anthropic_layout_defers_the_pack_and_splits_the_system_prompt():
    shim = LLMClient(model="claude-haiku-5-5").responses
    out, turn = shim._anthropic_cache_layout(_ackw(shim), "catalog_query", len("STATIC"))
    assert out["system"] == [{"type": "text", "text": "STATIC"}, {"type": "text", "text": "TAIL"}]
    assert [t["name"] for t in out["tools"]] == ["find_tools", "catalog_query", "pack_a", "pack_b"]
    assert [bool(t.get("defer_loading")) for t in out["tools"]] == [False, False, True, True]
    assert out["messages"][-1] == {"role": "system", "content": [
        {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "pack_a"}},
        {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "pack_b"}}]}
    assert set(turn) == {"pack_a", "pack_b"}
    _valid(out)
    final = ResponsesShim._apply_anthropic_cache_control(out, core_tool="catalog_query")
    assert [t["name"] for t in final["tools"] if "cache_control" in t] == ["catalog_query"]
    assert not any("cache_control" in t for t in final["tools"] if t.get("defer_loading")), \
        "a deferred tool cannot carry cache_control (400)"
    assert "cache_control" in final["system"][0] and "cache_control" not in final["system"][1]
    assert "cache_control" in final["messages"][-1]["content"][-1], "breakpoint after the tool additions"
    assert _all_marks(final) <= 4


def _round2(shim, out, turn):
    hist = out["messages"] + [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "pack_a", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}]
    return hist


def test_anthropic_layout_on_later_rounds_and_turns():
    shim = LLMClient(model="claude-haiku-5-5").responses
    out, turn = shim._anthropic_cache_layout(_ackw(shim), "catalog_query", len("STATIC"))
    hist = _round2(shim, out, turn)
    o2, turn2 = shim._anthropic_cache_layout(_ackw(shim, messages=list(hist), tools=_atools(shim, [fn("pack_c")])),
                                             "catalog_query", len("STATIC"), turn)
    assert o2["messages"][-1]["content"] == [
        {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "pack_c"}}]
    _valid(o2)
    # A new turn with another pack: the earlier additions are declared from THIS chain's definitions.
    turn2_msgs = o2["messages"] + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "Q2"}]
    o3, _ = shim._anthropic_cache_layout(
        _ackw(shim, messages=turn2_msgs, tools=shim._translate_tools_for_anthropic(CORE + [fn("pack_d")])),
        "catalog_query", len("STATIC"), turn2)
    assert {t["name"] for t in o3["tools"] if t.get("defer_loading")} == {"pack_a", "pack_b", "pack_c", "pack_d"}
    assert o3["messages"][-1]["content"] == [
        {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "pack_d"}}]
    _valid(o3)


@pytest.mark.parametrize("case", ["core_only_unmarked", "flag_off", "unknown_definition", "old_model"])
def test_anthropic_histories_with_additions_always_stay_valid(case, monkeypatch):  # CX-01
    shim = LLMClient(model="claude-haiku-5-5").responses
    out, turn = shim._anthropic_cache_layout(_ackw(shim), "catalog_query", None)
    hist = _round2(shim, out, turn) + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]
    core_only = shim._translate_tools_for_anthropic(CORE)
    core_tool, chain = "catalog_query", turn
    model = "claude-haiku-5-5"
    if case == "core_only_unmarked":
        core_tool = None
    elif case == "flag_off":
        monkeypatch.setenv("QUASAR_ANTHROPIC_DEFERRED_TOOLS", "0")
    elif case == "unknown_definition":
        chain = {}
    else:
        model = "claude-sonnet-5"
    o, _ = shim._anthropic_cache_layout(_ackw(shim, model=model, messages=list(hist), tools=list(core_only)),
                                        core_tool, None, chain)
    _valid(o)
    assert o["messages"] == hist, "history is append-only; nothing new to add here"
    declared = {t["name"]: t for t in o["tools"] if t.get("defer_loading")}
    assert set(declared) == {"pack_a", "pack_b"}
    if case == "unknown_definition":
        assert "no longer available" in declared["pack_a"]["description"], "an inert stub resolves the reference"
    else:
        assert declared["pack_a"]["description"] == "pack_a tool."


def test_anthropic_toolless_continuation_on_a_history_with_additions():  # CX-01 (reopened)
    shim = LLMClient(model="claude-haiku-5-5").responses
    out, turn = shim._anthropic_cache_layout(_ackw(shim), "catalog_query", None)
    hist = out["messages"] + [{"role": "assistant", "content": "plain prose"}, {"role": "user", "content": "more?"}]
    ck = _ackw(shim, messages=list(hist))
    ck.pop("tools")
    o, _ = shim._anthropic_cache_layout(ck, None, None, turn)
    _valid(o)
    eager = [t for t in o["tools"] if not t.get("defer_loading")]
    assert [t["name"] for t in eager] == ["quasar_no_tool"]
    assert o["tool_choice"] == {"type": "none"}, "a request that offered no tools still cannot call one"
    assert o["messages"] == hist
    # The placeholder never takes a name the history already references.
    tr = shim._translate_tools_for_anthropic
    o1, t1 = shim._anthropic_cache_layout(_ackw(shim, tools=tr(CORE + [fn("quasar_no_tool")])), "catalog_query", None)
    h1 = o1["messages"] + [{"role": "assistant", "content": "prose"}, {"role": "user", "content": "go on"}]
    ck1 = _ackw(shim, messages=list(h1))
    ck1.pop("tools")
    o2, _ = shim._anthropic_cache_layout(ck1, None, None, t1)
    _valid(o2)
    assert [t["name"] for t in o2["tools"] if not t.get("defer_loading")] == ["quasar_no_tool_1"]
    assert {t["name"]: t["description"] for t in o2["tools"]}["quasar_no_tool"] == "quasar_no_tool tool."


def test_anthropic_retries_resend_the_identical_request(monkeypatch):  # CX-10
    shim = LLMClient(model="claude-haiku-5-5").responses
    fake = _FakeAnthropic()
    monkeypatch.setattr(shim._llm, "_get_anthropic_client", lambda: fake)
    kw = {"model": "claude-haiku-5-5", "instructions": "STATICTAIL", "input": "Q", "tools": CORE + PACK,
          "max_output_tokens": 50}
    layout = {"cache_breakpoint_tool": "catalog_query", "instructions_static_len": 6}
    r1 = shim._call_anthropic(dict(kw), **layout).id

    class Overloaded(Exception):
        status_code = 529

    real_create, calls = fake.create, []

    def flaky(**k):
        calls.append(1)
        if len(calls) == 1:
            fake.sent.append(json.loads(json.dumps(k, default=vars)))
            shim._anthropic_chain_tools.clear()  # state changes between attempts
            raise Overloaded("529 overloaded")
        return real_create(**k)

    monkeypatch.setattr(fake, "create", flaky)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    shim._call_anthropic(dict(kw, input="next turn", previous_response_id=r1, tools=CORE), **layout)
    assert fake.sent[-1] == fake.sent[-2], "both attempts carry the same request"
    assert {t["name"]: t["description"] for t in fake.sent[-1]["tools"] if t.get("defer_loading")}["pack_a"] \
        == "pack_a tool.", "real definitions, not stubs, on the retried attempt"


def test_anthropic_definitions_never_cross_conversations():  # CX-02
    shim = LLMClient(model="claude-haiku-5-5").responses
    schema_a = fn("pack_a", desc="conversation A's pack_a")
    schema_b = fn("pack_a", desc="conversation B's pack_a (another user's MCP tool)")
    tr = shim._translate_tools_for_anthropic
    oa, turn_a = shim._anthropic_cache_layout(_ackw(shim, tools=tr(CORE + [schema_a])), "catalog_query", None)
    ob, turn_b = shim._anthropic_cache_layout(_ackw(shim, tools=tr(CORE + [schema_b])), "catalog_query", None)
    # Conversation A moves on to another pack; pack_a must be declared with A's schema.
    hist_a = oa["messages"] + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "Q2"}]
    o, _ = shim._anthropic_cache_layout(_ackw(shim, messages=hist_a, tools=tr(CORE + [fn("pack_z")])),
                                        "catalog_query", None, turn_a)
    declared = {t["name"]: t for t in o["tools"] if t.get("defer_loading")}
    assert declared["pack_a"]["description"] == "conversation A's pack_a"
    assert turn_b["pack_a"]["description"].startswith("conversation B")


def test_anthropic_layout_fallbacks():
    shim = LLMClient(model="claude-haiku-5-5").responses
    ck = _ackw(shim)
    assert shim._anthropic_cache_layout(ck, None, 6)[0] is ck, "unmarked callers without additions: untouched"
    o, _ = shim._anthropic_cache_layout(_ackw(shim, messages=[{"role": "user", "content": "Q"},
                                                              {"role": "assistant", "content": "x"}]),
                                        "catalog_query", None)
    assert not any(t.get("defer_loading") for t in o["tools"]), "a system message cannot follow an assistant turn"
    o, _ = shim._anthropic_cache_layout(_ackw(shim, model="claude-sonnet-5"), "catalog_query", len("STATIC"))
    assert isinstance(o["system"], list) and not any(t.get("defer_loading") for t in o["tools"])
    assert "extra_headers" not in o


def test_anthropic_layout_merges_the_files_beta_header(monkeypatch):
    shim = LLMClient(model="claude-haiku-5-5").responses
    o, _ = shim._anthropic_cache_layout(_ackw(shim, extra_headers={"anthropic-beta": "files-api-2025-04-14"}),
                                        "catalog_query", None)
    assert o["extra_headers"]["anthropic-beta"] == f"files-api-2025-04-14,{BETA}"
    monkeypatch.setenv("QUASAR_ANTHROPIC_DEFERRED_TOOLS", "0")
    o, _ = shim._anthropic_cache_layout(_ackw(shim), "catalog_query", None)
    assert not any(t.get("defer_loading") for t in o["tools"])


class _FakeAnthropic:
    def __init__(self):
        self.sent = []
        self.messages = self

    def _final(self):
        return NS(id=f"msg_{len(self.sent)}",
                  content=[NS(type="thinking", thinking="", signature="sig-1"),
                           NS(type="tool_use", id=f"tu{len(self.sent)}", name="pack_a", input={})],
                  usage=NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0),
                  stop_reason="tool_use", model="claude-haiku-5-5")

    def create(self, **kw):
        self.sent.append(json.loads(json.dumps(kw, default=vars)))  # replayed SDK blocks are objects
        return self._final()

    def stream(self, **kw):
        self.sent.append(json.loads(json.dumps(kw, default=vars)))
        final = self._final()

        class _S:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return False

            def __iter__(s):
                yield NS(type="message_start", message=NS(model="claude-haiku-5-5"))
                yield NS(type="content_block_start", index=1, content_block=NS(type="tool_use", name="pack_a", id="tu"))

            def get_final_message(s):
                return final
        return _S()


def _drive(shim, stream, kw, **layout):
    if stream:
        events = list(shim._stream_anthropic(dict(kw), **layout))
        return next(e.response.id for e in events if e.type == "response.completed")
    return shim._call_anthropic(dict(kw), **layout).id


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_replays_additions_and_thinking_without_repeating_them(stream, monkeypatch):  # CX-09
    shim = LLMClient(model="claude-haiku-5-5").responses
    fake = _FakeAnthropic()
    monkeypatch.setattr(shim._llm, "_get_anthropic_client", lambda: fake)
    kw = {"model": "claude-haiku-5-5", "instructions": "STATICTAIL", "input": "Q", "tools": CORE + PACK,
          "max_output_tokens": 50}
    layout = {"cache_breakpoint_tool": "catalog_query", "instructions_static_len": 6}
    caller_tools = json.dumps(kw["tools"])
    r1 = _drive(shim, stream, kw, **layout)
    out = [{"type": "function_call_output", "call_id": "tu1", "output": "{}"}]
    # Round 2 adds pack_c (find_tools); round 3 is a core-only, UNMARKED request.
    r2 = _drive(shim, stream, dict(kw, input=out, previous_response_id=r1, tools=CORE + PACK + [fn("pack_c")]), **layout)
    _drive(shim, stream, dict(kw, input=[{"type": "function_call_output", "call_id": "tu2", "output": "{}"}],
                              previous_response_id=r2, tools=CORE))
    first, second, third = fake.sent
    assert [m["role"] for m in first["messages"]] == ["user", "system"]
    assert [m["role"] for m in second["messages"]] == ["user", "system", "assistant", "user", "system"]
    assert [b["tool"]["name"] for b in second["messages"][-1]["content"]] == ["pack_c"]
    assert second["messages"][2]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}, \
        "the earlier thinking block is replayed unchanged"
    strip = lambda ms: [{**m, "content": [{k: v for k, v in b.items() if k != "cache_control"} for b in m["content"]]}
                        if isinstance(m.get("content"), list) else m for m in ms]
    assert strip(third["messages"][:5]) == strip(second["messages"]), "history replayed byte-for-byte"
    assert [m["role"] for m in third["messages"]] == ["user", "system", "assistant", "user", "system",
                                                     "assistant", "user"], "nothing added on the unmarked round"
    for s in fake.sent:
        _valid(s)
        assert _all_marks(s) <= 4
    assert json.dumps(kw["tools"]) == caller_tools, "caller's tools never mutated"


# ── Provider boundary with a non-zero static-length hint (CX-08) ────────

def test_static_length_hint_is_stripped_and_routed_for_every_provider(monkeypatch):
    from core.llm_client import CACHE_BREAKPOINT_KEY, LLMResponse

    monkeypatch.setenv("QUASAR_CALL_TOKENS", "0")
    marked = [fn("find_tools"), {**fn("catalog_query"), CACHE_BREAKPOINT_KEY: True}, fn("pack_a")]
    cases = (("gpt-6-luna", "_call_openai", "_call_openai"), ("gpt-oss-120b", "_call_tacc", "_stream_tacc"),
             ("deepseek-v4-flash", "_call_deepseek", "_stream_deepseek"),
             ("gemini-3.8-flash", "_call_google", "_stream_google"),
             ("local/test-model", "_call_local", "_stream_local"),
             ("claude-haiku-5-5", "_call_anthropic", "_stream_anthropic"))
    for model, call_meth, stream_meth in cases:
        for stream in (False, True):
            sent = []
            shim = LLMClient(model=model).responses

            def _fake(kwargs, attachments=None, _stream=stream, **extra):  # noqa: B023 (bound per loop)
                sent.append((dict(kwargs), extra))
                return iter([]) if _stream else LLMResponse(output_text="ok")

            monkeypatch.setattr(shim, stream_meth if stream else call_meth, _fake)
            res = shim.create(model=model, input="hi", instructions="STATICTAIL", tools=list(marked), stream=stream,
                              _instructions_static_len=6, _call_meta={})
            if stream and not isinstance(res, LLMResponse):
                list(res)
            (got, extra), = sent
            assert "_instructions_static_len" not in got, f"{model}: the hint must never reach a provider"
            if model.startswith("claude"):
                assert extra["instructions_static_len"] == 6
            elif model == "gpt-6-luna":
                assert extra["cache_layout"]["static_len"] == 6
            else:
                assert extra == {}
