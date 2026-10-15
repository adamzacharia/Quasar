"""Token plan (tmp/token-budget-2026-10-01/PLAN.md): tool packs, schema
compaction, per-call accounting, history compaction, Anthropic cache
breakpoints and one-shot direct dispatch. Pure-function tests; the registry
lint lives in test_tool_budget_lint.py."""
import json

from core import call_tokens, history_compact, tool_packs as tp
from core.tool_schema_compact import compact_parameters

REG = sorted({n for spec in tp.PACKS.values() for n in spec["tools"]} | set(tp.CORE))


# ── tool packs ─────────────────────────────────────────────────────────

def test_core_is_always_offered_first_and_order_is_stable():
    sel = tp.select("What is the weather like?", registered=REG)
    core = [n for n in tp.CORE if n in REG]
    assert sel["names"][: len(core)] == core
    assert sel["packs"] == []
    again = tp.select("What is the weather like?", registered=list(reversed(REG)))
    assert again["names"] == sel["names"], "order must not depend on registry order (prefix caching)"


def test_alma_question_gets_alma_pack_and_not_datalab():
    sel = tp.select("How many ALMA Band 6 projects observed HD 163296?", registered=REG)
    assert "alma" in sel["packs"] and "query_alma_science_archive" in sel["names"]
    assert "datalab" not in sel["packs"]


def test_forced_and_directive_tools_are_offered_with_their_pack():
    sel = tp.select("hello", registered=REG, forced_tools=["datalab_list_catalogs"],
                    directive_text="MANDATORY: call `mmdc_sed` now")
    assert "datalab_list_catalogs" in sel["names"] and "mmdc_sed" in sel["names"]
    assert "datalab" in sel["packs"] and "blazar_sed" in sel["packs"]


def test_router_packs_and_web_mode():
    sel = tp.select("hello", registered=REG, extra_packs=["literature"], web_mode="always")
    assert "literature" in sel["packs"] and "web_advanced" in sel["packs"]


def test_sticky_packs_follow_the_conversation():
    conv = "conv-sticky-test"
    tp.remember_called(conv, ["datalab_color_magnitude_diagram"])
    sel = tp.select("now make the radius twice as big", registered=REG, conversation_id=conv)
    assert "datalab_color_magnitude_diagram" in sel["names"] and "datalab_science" in sel["packs"]
    tp.remember_called(conv, [])
    tp.remember_called(conv, [])
    assert "datalab_color_magnitude_diagram" not in tp.recently_called(conv)


def test_short_follow_up_inherits_previous_question_packs():
    sel = tp.select("and in band 7?", registered=REG,
                    prior_user_text="List the Data Lab catalogs that cover the SMC")
    assert "datalab" in sel["packs"]
    hist = [{"role": "user", "content": "first q"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "and in band 7?"}]
    assert tp.prior_user_message(hist, "and in band 7?") == "first q"


def test_find_tools_adds_matching_tools():
    class T:
        def __init__(self, name, description):
            self.name, self.description = name, description

    reg = [T(n, f"tool {n.replace('_', ' ')}") for n in REG]
    out = tp.resolve_find_tools({"need": "plot a Kepler light curve and run a period search"}, reg, tp.CORE)
    assert "period_search" in out["add"] and "time_domain" in out["packs"]
    assert tp.FIND_TOOLS not in out["add"]
    assert "Loaded" in out["text"]
    named = tp.resolve_find_tools({"need": "x", "packs": ["cubes"]}, reg, tp.CORE)
    assert "compute_moment_map" in named["add"]
    schema = tp.find_tools_schema()
    assert schema["name"] == tp.FIND_TOOLS and set(schema["parameters"]["properties"]["packs"]["items"]["enum"]) == set(tp.PACKS)


def test_says_lacks_tool():
    yes = [
        "I don't have a tool for querying the Chandra archive.",
        "I do not have access to any tools that can retrieve light curves.",
        "None of the available tools can do that.",
        "That is not available in my current toolset.",
    ]
    no = [
        "The ALMA archive has 12 projects on this source.",
        "I don't have the redshift in the result, so I queried NED.",
        "No observations were found.",
    ]
    assert all(tp.says_lacks_tool(t) for t in yes)
    assert not any(tp.says_lacks_tool(t) for t in no)


# ── schema compaction ──────────────────────────────────────────────────

def test_compaction_is_lossless_for_parameters():
    params = {
        "type": "object", "title": "Args",
        "properties": {
            "ra": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None, "title": "Ra",
                   "description": "The ra"},
            "title": {"type": "string", "title": "Title", "description": "Plot title", "default": "x"},
            "band": {"type": "string", "enum": ["g", "r"], "examples": ["g"]},
        },
        "required": ["band"],
    }
    before = json.dumps(params, sort_keys=True)
    out = compact_parameters(params)
    assert json.dumps(params, sort_keys=True) == before, "must not mutate the registry schema"
    assert set(out["properties"]) == {"ra", "title", "band"}, "a parameter named 'title' must survive"
    assert out["properties"]["ra"] == {"type": ["number", "null"]}
    assert out["properties"]["title"] == {"type": "string", "description": "Plot title", "default": "x"}
    assert out["properties"]["band"] == {"type": "string", "enum": ["g", "r"]}
    assert out["required"] == ["band"] and "title" not in out


def test_description_cut_is_off_by_default(monkeypatch):
    long = "Topic list: " + "; ".join(f"t{i} = thing {i}" for i in range(200))
    out = compact_parameters({"type": "object", "properties": {"topic": {"type": "string", "description": long}}})
    assert out["properties"]["topic"]["description"] == long
    monkeypatch.setenv("QUASAR_TOOL_PARAM_DESC_CHARS", "200")
    cut = compact_parameters({"type": "object", "properties": {"topic": {"type": "string", "description": long}}})
    assert len(cut["properties"]["topic"]["description"]) <= 200


# ── per-call accounting ────────────────────────────────────────────────

def test_breakdown_splits_the_request(capsys, tmp_path, monkeypatch):
    kwargs = {
        "instructions": "You are Quasar. " * 20,
        "tools": [{"type": "function", "name": "a", "description": "d", "parameters": {"type": "object"}}],
        "input": [{"type": "function_call_output", "call_id": "c1", "output": "rows " * 50},
                  {"role": "user", "content": "guidance"}],
    }
    hist = [{"role": "system", "content": "ignored"}, {"role": "user", "content": "q " * 10},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c0", "function": {"name": "a", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "out " * 30}]
    bd = call_tokens.breakdown(kwargs, hist)
    assert bd["sys"] > 0 and bd["tools"] > 0 and bd["n_tools"] == 1
    assert bd["hist"] > bd["hist_tool_out"] > 0 and bd["new_tool_out"] > 0 and bd["new_msg"] > 0
    assert call_tokens.breakdown(kwargs, None)["hist"] is None  # server-side chain
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("QUASAR_CALL_TOKENS_LOG", str(log))

    class U:
        input_tokens, output_tokens, cache_hit_tokens = 1234, 56, 1000

    call_tokens.log_billed("tacc", "gpt-oss-120b", bd, U(), {"round": 0})
    rec = json.loads(log.read_text().strip())
    assert rec["billed_in"] == 1234 and rec["cached"] == 1000 and rec["round"] == 0 and rec["sys"] == bd["sys"]
    assert "[CALL BILLED]" in capsys.readouterr().out


# ── history compaction ─────────────────────────────────────────────────

def _is_new(t):
    return not t.startswith("[SYSTEM CONTINUATION]")


def _turn(i, tool_out_chars=3000):
    cid = f"c{i}"
    return [
        {"role": "user", "content": f"DOCS CONTEXT {'x' * 2000}\n\nUser: question {i}"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": cid, "type": "function",
                                                                "function": {"name": f"tool_{i}", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": cid, "content": "R" * tool_out_chars},
        {"role": "assistant", "content": f"answer {i} " + "a" * 4000},
    ]


def test_history_compaction_stubs_old_tool_outputs_and_keeps_recent():
    msgs = [{"role": "system", "content": "sys"}] + [m for i in range(5) for m in _turn(i)]
    out = history_compact.compact_history(msgs, _is_new, keep_turns=3, keep_tool_turns=2, cap_tokens=100_000)
    assert out[0]["content"] == "sys"
    tools = [m for m in out if m["role"] == "tool"]
    assert len(tools) == 5, "pairing kept: every tool message survives (as a stub when old)"
    assert tools[0]["content"].startswith(history_compact.STUB_PREFIX) and "tool_0" in tools[0]["content"]
    assert tools[-1]["content"] == "R" * 3000 and tools[-2]["content"] == "R" * 3000
    users = [m["content"] for m in out if m["role"] == "user"]
    assert users[0] == "User: question 0", "old user turns lose the context blocks"
    assert users[-1].startswith("DOCS CONTEXT"), "recent turns stay verbatim"
    assert msgs[3]["content"] == "R" * 3000, "never mutates the cached dicts"
    # idempotent: compacting the compacted history changes nothing
    assert history_compact.compact_history(out, _is_new, keep_turns=3, keep_tool_turns=2, cap_tokens=100_000) == out


def test_history_compaction_drops_oldest_past_cap_with_note():
    msgs = [{"role": "system", "content": "sys"}] + [m for i in range(8) for m in _turn(i)]
    out = history_compact.compact_history(msgs, _is_new, keep_turns=3, keep_tool_turns=2, cap_tokens=1500)
    users = [m["content"] for m in out if m["role"] == "user"]
    assert users[0].startswith(history_compact.NOTE_PREFIX) and '"question 0"' in users[0]
    assert sum(1 for m in out if m["role"] == "user") < 8
    # tool_call / tool pairs stay together
    ids_called = {tc["id"] for m in out if m.get("tool_calls") for tc in m["tool_calls"]}
    ids_answered = {m["tool_call_id"] for m in out if m["role"] == "tool"}
    assert ids_called == ids_answered
    # a later compaction keeps the earlier dropped questions in the merged note
    more = out + _turn(8) + _turn(9)
    out2 = history_compact.compact_history(more, _is_new, keep_turns=3, keep_tool_turns=2, cap_tokens=1500)
    first = next(m["content"] for m in out2 if m["role"] == "user")
    assert first.startswith(history_compact.NOTE_PREFIX) and '"question 0"' in first
    assert first.count(history_compact.NOTE_PREFIX) == 1


def test_short_chats_are_untouched():
    msgs = [{"role": "system", "content": "sys"}] + _turn(0) + _turn(1)
    assert history_compact.compact_history(msgs, _is_new, keep_turns=3, keep_tool_turns=2) is msgs


# ── Anthropic cache breakpoints ────────────────────────────────────────

def test_anthropic_cache_control_breakpoints_move_and_never_exceed_four():
    from core.llm_client import ResponsesShim

    tools = [{"name": "a"}, {"name": "b"}]
    msgs = [{"role": "user", "content": [{"type": "text", "text": "old", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "new"}]
    ck = ResponsesShim._apply_anthropic_cache_control({"tools": tools, "system": "sys", "messages": msgs})
    assert ck["tools"][-1]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in tools[-1]
    assert ck["system"][0]["cache_control"] == {"type": "ephemeral"}
    marks = sum(1 for m in ck["messages"] if isinstance(m["content"], list)
                for b in m["content"] if "cache_control" in b)
    assert marks == 1 and ck["messages"][-1]["content"][-1]["cache_control"]
    assert "cache_control" in msgs[0]["content"][0], "caller's list not mutated"


def _marks(ck):
    tools = [t["name"] for t in ck.get("tools") or [] if "cache_control" in t]
    system = sum(1 for b in ck.get("system") or [] if isinstance(b, dict) and "cache_control" in b)
    msgs = sum(1 for m in ck.get("messages") or [] if isinstance(m.get("content"), list)
               for b in m["content"] if isinstance(b, dict) and "cache_control" in b)
    return tools, system, msgs


def test_anthropic_core_breakpoint_placement(monkeypatch):
    from core.llm_client import ResponsesShim

    for var in ("QUASAR_ANTHROPIC_CORE_BREAKPOINT", "QUASAR_ANTHROPIC_CORE_CACHE_TTL", "QUASAR_ANTHROPIC_PROMPT_CACHE"):
        monkeypatch.delenv(var, raising=False)  # CX-02: defaults are checked from a clean env
    tools = [{"name": n} for n in ("find_tools", "web_search", "catalog_query", "pack_a", "pack_b")]
    msgs = [{"role": "user", "content": [{"type": "text", "text": "old", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "new"}]
    ck = ResponsesShim._apply_anthropic_cache_control(
        {"tools": tools, "system": "sys", "messages": msgs}, core_tool="catalog_query")
    t, s, m = _marks(ck)
    assert t == ["catalog_query", "pack_b"], "last core tool + whole tool block"
    assert s == 1 and m == 1 and ck["messages"][-1]["content"][-1]["cache_control"]
    assert len(t) + s + m <= 4, "Anthropic allows at most 4 breakpoints"
    assert all("cache_control" not in x for x in tools), "caller's tools not mutated"

    # Unmarked callers keep exactly the old placement.
    old = ResponsesShim._apply_anthropic_cache_control({"tools": tools, "system": "sys", "messages": msgs})
    assert _marks(old) == (["pack_b"], 1, 1)
    # A core-only turn: the core tool IS the last tool, one marker.
    only = ResponsesShim._apply_anthropic_cache_control({"tools": tools[:3], "system": "s", "messages": msgs},
                                                        core_tool="catalog_query")
    assert _marks(only)[0] == ["catalog_query"]
    # An unknown name (tool not offered) changes nothing.
    gone = ResponsesShim._apply_anthropic_cache_control({"tools": tools, "system": "s", "messages": msgs},
                                                        core_tool="missing")
    assert _marks(gone)[0] == ["pack_b"]
    # Opt-in 1-hour TTL on the core entry only (it is the first breakpoint).
    assert ck["tools"][2]["cache_control"] == {"type": "ephemeral"}, "default TTL unchanged"
    monkeypatch.setenv("QUASAR_ANTHROPIC_CORE_CACHE_TTL", "1h")
    hour = ResponsesShim._apply_anthropic_cache_control({"tools": tools, "system": "s", "messages": msgs},
                                                        core_tool="catalog_query")
    assert hour["tools"][2]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert hour["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert hour["system"][0]["cache_control"] == {"type": "ephemeral"}
    monkeypatch.delenv("QUASAR_ANTHROPIC_CORE_CACHE_TTL")
    # The A/B switch turns off only the core breakpoint; the global switch everything.
    monkeypatch.setenv("QUASAR_ANTHROPIC_CORE_BREAKPOINT", "0")
    off = ResponsesShim._apply_anthropic_cache_control({"tools": tools, "system": "s", "messages": msgs},
                                                       core_tool="catalog_query")
    assert _marks(off) == (["pack_b"], 1, 1)
    monkeypatch.delenv("QUASAR_ANTHROPIC_CORE_BREAKPOINT")
    monkeypatch.setenv("QUASAR_ANTHROPIC_PROMPT_CACHE", "0")
    none = ResponsesShim._apply_anthropic_cache_control({"tools": tools, "system": "s", "messages": msgs},
                                                        core_tool="catalog_query")
    assert none["tools"] is tools and none["system"] == "s"


def test_cache_breakpoint_marker_is_stripped_and_only_anthropic_gets_the_name(monkeypatch):
    from core.llm_client import CACHE_BREAKPOINT_KEY, LLMClient, LLMResponse, _pop_tool_cache_breakpoint

    marked = [{"type": "function", "name": "find_tools"},
              {"type": "function", "name": "catalog_query", CACHE_BREAKPOINT_KEY: True},
              {"type": "function", "name": "pack_a"}]
    kw = {"tools": marked}
    assert _pop_tool_cache_breakpoint(kw) == "catalog_query"
    assert all(CACHE_BREAKPOINT_KEY not in t for t in kw["tools"])
    assert CACHE_BREAKPOINT_KEY in marked[1], "caller's tool dicts not mutated"
    plain = [{"type": "function", "name": "a"}]
    kw = {"tools": plain}
    assert _pop_tool_cache_breakpoint(kw) is None and kw["tools"] is plain, "unmarked lists untouched"

    import copy

    monkeypatch.setenv("QUASAR_CALL_TOKENS", "0")
    unmarked = [{k: v for k, v in t.items() if k != CACHE_BREAKPOINT_KEY} for t in marked]
    # OpenAI streams through _call_openai; the others have a separate stream method.
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
                sent.append((copy.deepcopy(kwargs), extra))
                return iter([]) if _stream else LLMResponse(output_text="ok")

            monkeypatch.setattr(shim, stream_meth if stream else call_meth, _fake)
            for tools in (list(marked), list(unmarked)):  # marked request, then an unmarked control
                res = shim.create(model=model, input="hi", tools=tools, stream=stream, _call_meta={})
                if stream and not isinstance(res, LLMResponse):
                    list(res)  # drain the usage wrapper
            (got, extra), (control, control_extra) = sent
            label = f"{model} stream={stream}"
            assert got == control, f"{label}: the provider must receive the same request as an unmarked call"
            assert all(CACHE_BREAKPOINT_KEY not in t for t in got["tools"]), f"{label}: marker reached a provider"
            if model == "claude-haiku-5-5":
                assert extra == {"cache_breakpoint_tool": "catalog_query", "instructions_static_len": None}, label
                assert control_extra == {"cache_breakpoint_tool": None, "instructions_static_len": None}, \
                    f"{label}: unmarked callers get no name"
            elif model == "gpt-6-luna":
                assert extra == {"cache_layout": {"core_tool": "catalog_query", "static_len": None,
                                                  "own_cache_key": True}}, label
                assert control_extra == {}, f"{label}: unmarked OpenAI calls are exactly the old call"
            else:
                assert extra == control_extra == {}, f"{label}: no new argument outside Anthropic/OpenAI"


def test_core_tool_serialization_is_byte_stable_across_questions():
    from core.llm_client import ResponsesShim

    schemas = {n: {"type": "function", "name": n, "description": f"{n} tool.",
                   "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}
               for n in REG}

    def offered(question, registry):
        sel = tp.select(question, registered=registry)
        return [tp.find_tools_schema()] + [schemas[n] for n in sel["names"] if n != tp.FIND_TOOLS]

    a = offered("How many ALMA Band 6 projects observed HD 163296?", REG)
    b = offered("Plot the Gaia DR3 colour-magnitude diagram of M67", list(reversed(REG)))
    na, nb = tp.core_prefix_len(a), tp.core_prefix_len(b)
    assert na == nb == len([n for n in tp.CORE if n in REG]), "the core block is the same size"
    assert len(a) > na and len(b) > nb, "both questions add pack tools after the core"
    ser = lambda tools: json.dumps(ResponsesShim._translate_tools_for_anthropic(None, tools), ensure_ascii=False)
    assert ser(a[:na]) == ser(b[:nb]), "core prefix must be byte-identical across questions"
    assert ser(a) != ser(b)
    assert tp.core_prefix_len([{"name": "pack_a"}] + a) == 0


def test_cache_writes_flow_from_anthropic_usage_to_the_call_log(tmp_path, monkeypatch):
    from core.llm_client import LLMUsage, _anthropic_usage

    class U:
        input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens = 100, 7, 9000, 500

    u = _anthropic_usage(U())
    assert u.cache_write_tokens == 500 and u.input_tokens == 9600, "input_tokens unchanged (quota, cost)"

    class Zero:  # CX-01: a reported zero is a known zero, not "unknown"
        input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens = 40, 3, 0, 0

    class ReadsOnly:  # ... and a missing write field is unknown, not zero
        input_tokens, output_tokens, cache_read_input_tokens = 40, 3, 9000

    assert _anthropic_usage(Zero()).cache_write_tokens == 0
    assert _anthropic_usage(ReadsOnly()).cache_write_tokens is None
    assert _anthropic_usage(ReadsOnly()).input_tokens == 9040
    assert call_tokens._usage_fields(u) == {"billed_in": 9600, "cached": 9000, "billed_out": 7, "cache_write": 500}
    assert call_tokens._usage_fields(LLMUsage(input_tokens=5, output_tokens=1))["cache_write"] is None
    assert call_tokens._usage_fields(None)["cache_write"] is None

    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("QUASAR_CALL_TOKENS_LOG", str(log))
    call_tokens.log_billed("anthropic", "claude-haiku-5-5", {"est": 9000, "n_tools": 3}, u, {"round": 0})
    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert (rec["billed_in"], rec["cached"], rec["cache_write"], rec["round"]) == (9600, 9000, 500, 0)


# ── one-shot direct dispatch ───────────────────────────────────────────

def test_dispatch_marks_only_complete_intents_and_rewrites_directive():
    from core.oneshot_routing import detect_oneshot_intent, dispatched_directive

    i = detect_oneshot_intent("How many ALMA bands have public data as of 2026-01-01?")
    assert i["tool"] == "alma_public_band_status" and i["dispatch"] is True
    d = dispatched_directive(i["directive"])
    assert "You MUST call" not in d and "ALREADY called" in d and "Report" in d
    link = detect_oneshot_intent("Give me a URL into the ALMA archive search for my query")
    assert link is None or not link.get("dispatch"), "no target -> the model must fill it, no dispatch"


# ── guard task-5314487-21564 regressions ───────────────────────────────

def test_dispatch_exchange_builds_valid_chat_messages_and_deepseek_reasoning():
    """CX-13: the round-1 input of a direct dispatch (question + function_call
    items + outputs) becomes paired chat messages; DeepSeek gets "" reasoning."""
    from core.llm_client import _append_chat_items, _deepseek_reasoning_fields, _is_new_user_turn

    items = [{"role": "user", "content": "\n\nUser: which catalogs cover the LMC?"},
             {"type": "function_call", "call_id": "call_qd1", "name": "datalab_list_catalogs", "arguments": "{}"},
             {"type": "function_call_output", "call_id": "call_qd1", "output": "{\"ok\": true}"}]
    assert _is_new_user_turn(items)
    msgs = [{"role": "system", "content": "sys"}]
    _append_chat_items(msgs, items)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["id"] == msgs[3]["tool_call_id"] == "call_qd1"
    ds = _deepseek_reasoning_fields(msgs)
    assert ds[2]["reasoning_content"] == "" and "reasoning_content" not in msgs[2]
    kept = {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}], "reasoning_content": "thought"}
    assert _deepseek_reasoning_fields([kept])[0]["reasoning_content"] == "thought"


def test_find_tools_only_loads_allowed_tools_and_tolerates_non_object_args():
    class T:
        def __init__(self, name):
            self.name, self.description = name, name.replace("_", " ")

    allowed = [T(n) for n in REG if not n.startswith("web_")]
    out = tp.resolve_find_tools({"need": "crawl a website", "packs": ["web_advanced"]}, allowed, tp.CORE)
    assert not any(n.startswith("web_") for n in out["add"]), "CX-03: web-stripped tools are never offered"
    for bad in (["light curve"], "light curve", 7, None):
        res = tp.resolve_find_tools(bad, allowed, tp.CORE)  # CX-04: no AttributeError
        assert isinstance(res["text"], str)
    assert "period_search" in tp.resolve_find_tools({"need": "x", "packs": "time_domain"}, allowed, tp.CORE)["add"]


def test_compaction_keeps_literal_values_and_required_nullables():
    params = {
        "type": "object", "required": ["mode"],
        "properties": {
            "mode": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Mode"},
            "opt": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "preset": {"type": "object", "default": {"title": "standard"}, "enum": [{"title": "standard"}]},
        },
    }
    out = compact_parameters(params)
    assert out["properties"]["mode"]["type"] == ["string", "null"], "CX-06: explicit null stays valid"
    assert out["properties"]["opt"] == {"type": ["string", "null"]}
    keep = compact_parameters({"type": "object", "properties": {"u": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}}})
    assert "anyOf" in keep["properties"]["u"], "non-null unions are left alone"
    enum_null = compact_parameters({"type": "object", "properties": {"e": {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "null"}]}}})
    assert "anyOf" in enum_null["properties"]["e"], "CX-06: a constrained branch keeps anyOf (null stays valid)"
    null_c = compact_parameters({"type": "object", "properties": {"n": {"anyOf": [{"type": "string"}, {"type": "null", "enum": []}]}}})
    assert "anyOf" in null_c["properties"]["n"], "CX-06: a constrained null branch keeps anyOf"
    assert out["properties"]["preset"]["default"] == {"title": "standard"}, "CX-05: literals untouched"
    assert out["properties"]["preset"]["enum"] == [{"title": "standard"}]


def test_anthropic_usage_counts_cache_reads_and_writes():
    from core.llm_client import _anthropic_usage

    class U:
        input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens = 100, 7, 9000, 500

    u = _anthropic_usage(U())
    assert u.input_tokens == 9600 and u.cache_hit_tokens == 9000 and u.output_tokens == 7
    assert not hasattr(u, "priced_input_tokens"), "CX-15: quota records actual tokens"

    class Plain:
        input_tokens, output_tokens = 50, 5

    assert _anthropic_usage(Plain()).input_tokens == 50 and _anthropic_usage(Plain()).cache_hit_tokens is None


def test_old_user_turn_keeps_question_that_quotes_user_lines():
    q = "\n\nUser: analyze this transcript:\n\nUser: first quote\nAssistant: reply"
    assert history_compact._user_question(q).startswith("User: analyze this transcript")  # CX-07
    assert history_compact._user_question("DOCS\n\nUser: real question") == "User: real question"
    # a documentation excerpt that itself contains "User:" lines, then the runner's notes and marker
    rag = ("DOCS chunk\n\nUser: how do I log in?\nAgent: ..." + "\n\nIMPORTANT CITATION & STRUCTURE RULES:\n1. x"
           + "\n\nUser: what is the ALMA Cycle 13 deadline?")
    assert history_compact._user_question(rag) == "User: what is the ALMA Cycle 13 deadline?"
    long_q = "User: summarize this transcript" + "\n\nUser: quoted line" * 200 + " END"
    clipped = history_compact._clip_middle(long_q, 1500)
    assert clipped.startswith("User: summarize this transcript") and clipped.endswith("END") and len(clipped) < 1520


def test_tool_free_turns_age_the_sticky_window():
    conv = "conv-age-test"
    tp.remember_called(conv, ["period_search"])
    tp.remember_called(conv, set())
    assert "period_search" in tp.recently_called(conv)
    tp.remember_called(conv, set())
    assert "period_search" not in tp.recently_called(conv)  # CX-10




def _detector(create):
    from types import SimpleNamespace
    from core.complexity import ComplexityDetector

    return ComplexityDetector(client=SimpleNamespace(responses=SimpleNamespace(create=create)))


def _wait_probes_done(cx, seconds=5.0):
    """Let stray probe workers of a test finish before the next test resets the
    shared counters (guard CX-09)."""
    import threading
    import time

    deadline = time.monotonic() + seconds
    for t in [t for t in threading.enumerate() if t.name == "complexity-probe"]:
        t.join(max(0.0, deadline - time.monotonic()))
    assert cx._probe_inflight == 0 and cx._probe_stuck == 0


def _reset_probe_state(monkeypatch):
    import threading
    import core.complexity as cx

    _wait_probes_done(cx)

    monkeypatch.setattr(cx, "_probe_timed_out_at", 0.0)
    monkeypatch.setattr(cx, "_probe_stuck", 0)
    monkeypatch.setattr(cx, "_probe_inflight", 0)
    monkeypatch.setattr(cx, "_stuck_since", {})
    monkeypatch.setattr(cx, "_probe_lock", threading.Lock())
    return cx


def test_complexity_probe_is_bounded_and_stuck_probes_are_capped(monkeypatch):
    """A provider that never answers (keep-alives only) must not hang the turn;
    at most _PROBE_MAX_STUCK timed-out probes linger, then probes are skipped."""
    import threading
    import time

    cx = _reset_probe_state(monkeypatch)
    release = threading.Event()
    calls = []

    def _hang(**kw):
        calls.append(1)
        release.wait(10)
        raise RuntimeError("late")

    monkeypatch.setenv("QUASAR_COMPLEXITY_TIMEOUT_S", "0.2")
    det = _detector(_hang)
    t0 = time.monotonic()
    res = det._llm_assess("q", 0.5)
    assert time.monotonic() - t0 < 1.5
    assert res.reasoning == "LLM assessment failed; using heuristic" and res.score == 0.5
    for _ in range(cx._PROBE_MAX_STUCK + 3):
        det._llm_assess("q", 0.5)
    assert len(calls) == cx._PROBE_MAX_STUCK, "no thread pile-up during an outage"
    stuck = [t for t in threading.enumerate() if t.name == "complexity-probe"]
    assert stuck and all(t.daemon for t in stuck), "stray probes never block interpreter exit"
    release.set()
    deadline = time.monotonic() + 3
    while cx._probe_stuck and time.monotonic() < deadline:
        time.sleep(0.02)
    assert cx._probe_stuck == 0, "returning probes free their stuck slot"
    _wait_probes_done(cx)


def test_healthy_concurrent_probes_are_never_skipped(monkeypatch):
    """CX-07: many in-time probes at once all reach the provider."""
    import threading
    import time
    from types import SimpleNamespace

    _reset_probe_state(monkeypatch)
    n = 10
    started = []
    gate = threading.Barrier(n, timeout=5)

    def _slow_ok(**kw):
        started.append(1)
        time.sleep(0.1)
        return SimpleNamespace(output_text='{"score": 0.9, "reasoning": "ok"}')

    det = _detector(_slow_ok)
    results = []

    def _one():
        gate.wait()
        results.append(det._llm_assess("q", 0.5).score)

    ts = [threading.Thread(target=_one) for _ in range(n)]
    [t.start() for t in ts]
    [t.join(5) for t in ts]
    assert len(started) == n and results == [0.9] * n


def test_no_cooldown_by_default(monkeypatch):
    """CX-05: after a timeout, the next probe still runs (cooldown is opt-in)."""
    import time
    from types import SimpleNamespace

    _reset_probe_state(monkeypatch)
    monkeypatch.setenv("QUASAR_COMPLEXITY_TIMEOUT_S", "0.2")
    state = {"n": 0}

    def _first_hangs(**kw):
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(1.0)
            raise RuntimeError("late")
        return SimpleNamespace(output_text='{"score": 0.95, "reasoning": "recovered"}')

    det = _detector(_first_hangs)
    assert det._llm_assess("q", 0.5).score == 0.5
    assert det._llm_assess("q", 0.5).score == 0.95, "a recovered provider is used immediately"
    import core.complexity as cx
    _wait_probes_done(cx)


def test_complexity_probe_in_time_answer_is_unchanged(monkeypatch):
    from types import SimpleNamespace

    _reset_probe_state(monkeypatch)
    seen = {}

    def _ok(**kw):
        seen.update(kw)
        return SimpleNamespace(output_text='{"is_complex": true, "score": 0.9, "reasoning": "r", "subtasks": ["a"]}')

    res = _detector(_ok)._llm_assess("multi step q", 0.5)
    assert res.is_complex and res.score == 0.9 and res.suggested_subtasks == ["a"]
    assert seen["temperature"] == 0 and seen["max_output_tokens"] == 300 and seen["text"] == {"format": {"type": "json_object"}}
    assert seen["model"] == "deepseek-v4-flash" and 'User query: "multi step q"' in seen["input"]
    assert res.reasoning == "r"


def test_complexity_probe_immediate_error_falls_back(monkeypatch):
    _reset_probe_state(monkeypatch)

    def _boom(**kw):
        raise RuntimeError("400 bad request")

    res = _detector(_boom)._llm_assess("q", 0.35)
    assert res.score == 0.35 and res.reasoning == "LLM assessment failed; using heuristic"


def test_concurrent_outage_burst_is_bounded(monkeypatch):
    """CX-02: a burst that starts before any probe has timed out cannot exceed the in-flight ceiling."""
    import threading

    cx = _reset_probe_state(monkeypatch)
    monkeypatch.setattr(cx, "_PROBE_MAX_INFLIGHT", 5)
    monkeypatch.setenv("QUASAR_COMPLEXITY_TIMEOUT_S", "0.5")
    release = threading.Event()
    calls = []
    lock = threading.Lock()

    def _hang(**kw):
        with lock:
            calls.append(1)
        release.wait(10)
        raise RuntimeError("late")

    det = _detector(_hang)
    gate = threading.Barrier(20, timeout=5)

    def _one():
        gate.wait()
        det._llm_assess("q", 0.5)

    ts = [threading.Thread(target=_one) for _ in range(20)]
    [t.start() for t in ts]
    [t.join(5) for t in ts]
    assert len(calls) <= 5
    release.set()
    _wait_probes_done(cx)


def test_inflight_ceiling_is_above_the_chat_worker_pool():
    """CX-07: probes run inside turns, which run on the chat executor; the ceiling
    is far above that pool, so healthy concurrency cannot reach it."""
    import os
    import core.complexity as cx

    pool = max(2, int(os.getenv("CHAT_WORKER_THREADS", "4")))
    assert cx._PROBE_MAX_INFLIGHT >= 4 * pool and cx._PROBE_MAX_INFLIGHT >= 64


def test_stuck_probes_stop_blocking_after_ttl(monkeypatch):
    """CX-10: four probes that never return block new probes only for the TTL."""
    import threading
    import time
    from types import SimpleNamespace

    cx = _reset_probe_state(monkeypatch)
    monkeypatch.setenv("QUASAR_COMPLEXITY_TIMEOUT_S", "0.1")
    monkeypatch.setenv("QUASAR_COMPLEXITY_STUCK_TTL_S", "0.4")
    release = threading.Event()
    healthy = {"on": False}

    def _create(**kw):
        if healthy["on"]:
            return SimpleNamespace(output_text='{"score": 0.9, "reasoning": "back"}')
        release.wait(10)
        raise RuntimeError("late")

    det = _detector(_create)
    for _ in range(cx._PROBE_MAX_STUCK):
        det._llm_assess("q", 0.5)
    assert cx._probe_stuck == cx._PROBE_MAX_STUCK
    healthy["on"] = True
    assert det._llm_assess("q", 0.5).score == 0.5, "blocked while the stuck probes are recent"
    time.sleep(0.5)
    assert det._llm_assess("q", 0.5).score == 0.9, "after the TTL a recovered provider is used"
    release.set()
    _wait_probes_done(cx)
