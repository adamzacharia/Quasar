"""System prompt v2 (duel task-dd87861-15924, 2026-09-25).

Semantic tests at each rule's new home: the compact core, the per-tool
description amendments, the playbooks and the per-turn context. The legacy
bundle stays byte-identical and default; these tests exercise the v2 bundle
explicitly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.prompts import playbooks as pb  # noqa: E402
from core.prompts import system_core as sc  # noqa: E402
from core.prompts.tool_description_overrides import OVERRIDES, effective_description  # noqa: E402


# ---------------------------------------------------------------------------
# Core budget and hygiene
# ---------------------------------------------------------------------------

def test_core_body_under_budget_measured_tokens():
    body = sc.core_body()
    n = sc.count_tokens(body)
    assert n < sc.CORE_TOKEN_BUDGET, f"core body is {n} tokens (cap {sc.CORE_TOKEN_BUDGET})"


def test_core_has_no_harmony_control_tokens_and_no_date():
    body = sc.core_body()
    assert "<|" not in body and "|>" not in body
    assert "Date:" not in body
    full = sc.build_core_prompt("")
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", full), "no date may be baked into the static v2 body"


def test_core_keeps_generated_blocks_and_marker_strings():
    grounding = "\nARCHIVE SCHEMA GROUNDING (canonical, not exhaustive):\n- datalab: call browse_schema('datalab') before writing a query.\n"
    full = sc.build_core_prompt(grounding)
    assert "ARCHIVE SCHEMA GROUNDING" in full and "browse_schema('datalab')" in full
    assert "ALMA ARCHIVE GUARDRAILS" in full  # ALMA kernel appended unchanged
    assert "[GROUNDED_SUMMARY_MODE]" in full  # exact marker preserved
    assert "📚" in full and "🌐" in full  # UI icon contract exemption is explicit


def test_core_carries_cross_cutting_safeguards():
    body = sc.core_body()
    # honesty / evidence
    assert "the card" in body and "never \"above\" or \"below\"" in body
    assert "UNKNOWN" in body and "Never turn a failed call into" in body
    assert "hit its row cap" in body and "no significant period" in body
    assert "Never substitute another survey's imagery" in body
    assert "OpenAlex" in body and "snake_case" in body
    # execution contract (C1 without tool enumeration; native calls; one retry)
    assert "needs a tool call in this turn" in body
    assert "search_by_target" not in body.split("ROUTING PRECEDENCE")[0]
    assert "Call tools natively with schema-valid JSON arguments" in body
    assert "retry once" in body and "stop_polling" in body
    # query correctness (NaN guard short form, bounds, aggregates)
    assert "< 'Infinity'" in body and "GROUP BY aggregate" in body
    assert "datalab_cutout_grid" in body
    # routing rows name the tools the relocated tests used to pin
    for name in ("hips_cutout", "sparcl_plot_spectrum", "survey_covers_position", "search_pulsars",
                 "pulsar_lookup", "galactic_extinction", "search_mmu_hats_catalog", "vo_adql_query"):
        assert name in body, name
    # C2 resolved in one place; C4 stale sentence gone
    assert "do not repeat the paper list" in body
    assert "schema below" not in body


def test_bundle_hash_is_stable_and_short():
    assert sc.bundle_hash() == sc.bundle_hash()
    assert len(sc.bundle_hash()) == 16


# ---------------------------------------------------------------------------
# Tool description amendments
# ---------------------------------------------------------------------------

def test_effective_description_appends_without_mutating_base():
    base = "Search the ALMA archive by target name."
    eff = effective_description("search_by_target", base)
    assert eff.startswith(base)
    assert "resolve_target" in eff and "band='6,7'" in eff
    assert effective_description("no_such_tool", base) == base


def test_override_semantics_relocated_from_legacy_prompt():
    o = OVERRIDES
    # radio SED v1 caveats and flags (test_radio_sed pin moves here)
    assert "flags" in o["radio_sed"] and "without resolution matching, flux-scale corrections, or image-plane photometry" in o["radio_sed"]
    # FAP with any period
    assert "FAP" in o["period_search"]
    # NaN guard long form + bounds + stop_polling
    assert "< 'Infinity'" in o["datalab_sql_query"] and "stop_polling" in o["datalab_sql_query"]
    # SED sample_n, never per-row loop
    assert "sample_n" in o["datalab_sed_plot"] and "never loop per row" in o["datalab_sed_plot"]
    # color image: DR9 has no i band; no silent substitution; source_service
    assert "no i" in o["datalab_color_image"] and "NEVER silently substitute another survey's imagery" in o["datalab_color_image"]
    assert "DR10" in o["hips_cutout"]
    # paper cards: no text listing; ID variant; never web_search
    assert "cards" in o["search_papers"] and "search_papers_by_observation_id" in o["search_papers"]
    assert "web_search" in o["search_papers"]
    # MOC failure continuation exception
    assert "MOCServer fails" in o["survey_covers_position"]
    # multi-target / multi-band in one call
    assert "one call per band" in o["search_by_target"]
    # named transition vs ladder
    assert "check_co_lines" in o["find_alma_line_coverage"] and "find_alma_line_coverage" in o["check_co_lines"]


def test_agent_serializer_applies_overrides_only_in_v2(monkeypatch):
    from core.agent import QuasarAgent
    from core import bench_toolset

    tool = SimpleNamespace(name="radio_sed", description="Compile a radio SED.", parameters={"type": "object", "properties": {}})
    fake = SimpleNamespace(
        tool_registry=SimpleNamespace(list_tools=lambda: [tool]),
        config=SimpleNamespace(enable_mcp=False, mcp_server_url=""),
    )
    monkeypatch.setattr(bench_toolset, "active", lambda: False)
    monkeypatch.setattr(bench_toolset, "allowed", lambda name: True)

    fake.prompt_bundle = "legacy"
    legacy = QuasarAgent._build_tools_for_responses_api(fake)
    assert legacy[0]["description"] == "Compile a radio SED."

    fake.prompt_bundle = "v2"
    v2 = QuasarAgent._build_tools_for_responses_api(fake)
    assert v2[0]["description"].startswith("Compile a radio SED.")
    assert "flags" in v2[0]["description"]
    assert tool.description == "Compile a radio SED.", "registry must never be mutated"

    # neutral benchmark arm: no amendments even in v2
    monkeypatch.setattr(bench_toolset, "active", lambda: True)
    neutral = QuasarAgent._build_tools_for_responses_api(fake)
    assert neutral[0]["description"] == "Compile a radio SED."


def test_build_system_prompt_dispatches_on_bundle(monkeypatch):
    from core.agent import QuasarAgent

    fake = SimpleNamespace()
    fake.prompt_bundle = "v2"
    v2 = QuasarAgent._build_system_prompt(fake)
    assert v2.startswith("You are Quasar, a research assistant for professional astronomers.")
    assert "Date:" not in v2

    fake.prompt_bundle = "legacy"
    legacy = QuasarAgent._build_system_prompt(fake)
    assert legacy.startswith("You are Quasar, an expert AI research assistant for astronomy")
    assert "Date:" in legacy  # legacy control keeps its original date behaviour


# ---------------------------------------------------------------------------
# Playbooks: selection, budget, follow-ups, rendering
# ---------------------------------------------------------------------------

def _select(query, oneshot_tool=None, oneshot_args=None, **intent):
    st = pb.TurnState()
    sel = pb.select_initial_playbooks(query, {"oneshot_tool": oneshot_tool, "oneshot_args": oneshot_args or {}, **intent}, state=st)
    return [p.id for p in sel], st


@pytest.mark.parametrize("query,oneshot,args,expected", [
    ("Show me a color image of the center of M31 from the DECam Legacy Surveys.", "datalab_color_image", {}, {"survey_imagery"}),
    ("Find high-proper-motion white-dwarf candidates in Gaia DR3 around RA = 60, Dec = -50.", "datalab_selection_diagram", {}, {"wd_selection"}),
    ("Combine NSC DR2 photometry with Gaia DR3 proper motions around Palomar 5 to trace its tidal tails.", "datalab_stream_selection", {"cluster_name": "Palomar 5"}, {"stream_membership"}),
    ("Make a stellar density map of a ~20 x 20 degree region from NSC DR2, bin by HEALPix.", "datalab_healpix_density_map", {"preset": "south_gradient"}, {"density_maps", "open_regions"}),
    ("Help me look for a stellar overdensity in SMASH DR1 field 169.", "datalab_satellite_search", {"smash_field": 169}, {"density_maps"}),
    ("Search NSC DR2 for the densest stellar clump within 1 deg of the Hydra II dwarf region.", "datalab_satellite_search", {"preset": "hydra2", "radius_deg": 1.0}, {"density_maps"}),
    ("Select galaxies from SDSS/BOSS in a thin redshift slice and make a wedge plot, pick a region like the SDSS Great Wall.", "datalab_lss_wedge", {}, {"open_regions"}),
    ("Discover new Milky Way satellite dwarf-galaxy candidates using Data Lab's deep imaging catalogs.", "datalab_satellite_search", {"preset": "delve_south"}, {"density_maps", "open_regions"}),
])
def test_initial_selection_on_bench_shaped_questions(query, oneshot, args, expected):
    ids, _ = _select(query, oneshot, args)
    assert set(ids) == expected, ids


@pytest.mark.parametrize("query", [
    "What is the ALMA proprietary period?",
    "How do I calibrate ALMA Band 6 data?",
    "Find recent papers on protoplanetary disks.",
    "How many Gaia DR3 sources lie within 10 arcminutes of Palomar 5?",
])
def test_conceptual_and_plain_questions_select_nothing(query):
    ids, st = _select(query)
    assert ids == [] and st.spent_tokens == 0


def test_intent_keys_select_profile_and_products_dedupe():
    assert _select("Who is Crystal Brogan?", researcher=True)[0] == ["researcher_profile"]
    # ALMA product triage: the runner's product directive already carries the
    # picker rules, so no playbook duplicates it (duel DX-14); the band and
    # no-download guidance sits in the tool description amendment.
    assert _select("Fetch the ALMA FITS products for M87", product_triage=True)[0] == []
    assert "band preference" in OVERRIDES["triage_alma_data_products"]
    assert "Never auto-download" in OVERRIDES["triage_alma_data_products"]


def test_budget_is_cumulative_and_never_truncates():
    st = pb.TurnState()
    st.spent_tokens = pb.TOTAL_BUDGET - 50  # route directives already used almost everything
    sel = pb.select_initial_playbooks("color image of M31 from the DECam Legacy Surveys", {"oneshot_tool": "datalab_color_image"}, state=st)
    assert sel == [] and "survey_imagery" in st.omitted
    for p in pb.PLAYBOOKS:
        assert p.token_cost <= pb.INITIAL_BUDGET, p.id


def test_followup_activation_and_dedupe():
    st = pb.TurnState()
    first = pb.select_tool_followups(["datalab_color_image"], set(), st)
    assert [p.id for p in first] == ["survey_imagery"]
    again = pb.select_tool_followups(["datalab_color_image"], {"coverage_gap"}, st)
    assert again == []  # already sent this turn
    dens = pb.select_tool_followups(["datalab_density_aggregate"], set(), st)
    assert [p.id for p in dens] == ["density_maps"]
    assert st.spent_tokens == first[0].token_cost + dens[0].token_cost


def test_result_flags_only_activate_with_a_relevant_tool():
    st = pb.TurnState()
    # coverage_gap from a satellite-search cutout grid is NOT a color-image workflow
    assert pb.select_tool_followups(["datalab_satellite_search"], {"coverage_gap"}, st) == []
    # row_cap noted by a CMD one-shot is NOT a density-map workflow
    assert pb.select_tool_followups(["datalab_color_magnitude_diagram"], {"row_cap"}, st) == []
    # but a row-capped select_rows pull is
    assert [p.id for p in pb.select_tool_followups(["datalab_select_catalog_rows"], {"row_cap"}, st)] == ["density_maps"]
    # and a coverage_gap from the color-image tool itself is
    assert [p.id for p in pb.select_tool_followups(["datalab_image_cutout"], {"coverage_gap"}, st)] == ["survey_imagery"]


def test_result_flags_extraction():
    outs = [
        {"type": "function_call_output", "call_id": "1", "output": '{"success": false, "coverage_gap": true}'},
        {"type": "function_call_output", "call_id": "2", "output": '{"warnings": ["hit its row cap (500)"]}'},
        {"role": "user", "content": "coverage_gap should be ignored here"},
    ]
    assert pb.result_flags_from_outputs(outs) == {"coverage_gap", "row_cap"}


def test_render_turn_context_and_followup_item():
    st = pb.TurnState()
    sel = pb.select_initial_playbooks("white dwarf candidates", {"oneshot_tool": "datalab_selection_diagram"}, state=st)
    ctx = pb.render_turn_context("2026-09-25 (CDT)", "", sel)
    assert "Turn context [qv2]" in ctx and "Current date: 2026-09-25 (CDT)" in ctx
    assert "[PLAYBOOK wd_selection]" in ctx and "datalab_selection_diagram" in ctx
    item = pb.render_followup_item(sel)
    assert item["role"] == "user" and "applies to this turn only" in item["content"]
    assert pb.render_followup_item([]) is None


def test_playbook_texts_carry_relocated_pins():
    t = {p.id: p.text for p in pb.PLAYBOOKS}
    # test_reb_science_correctness strings, now in the survey_imagery playbook
    assert "NEVER silently substitute another survey's imagery" in t["survey_imagery"]
    assert "explicitly labeled with that survey's name" in t["survey_imagery"]
    assert "aliases serve the Legacy Surveys DR10 HiPS" in t["survey_imagery"]
    # WD: routed one-shot primary, CMD recipe fallback, tool-computed counts
    assert t["wd_selection"].index("datalab_selection_diagram") < t["wd_selection"].index("datalab_color_magnitude_diagram")
    assert "n_wd_candidates" in t["wd_selection"]
    # stream: selected sample over full cone, center within plot
    assert "row-capped crossmatch" in t["stream_membership"] and "cluster center" in t["stream_membership"]
    # density: fieldid bound, matched_filter, no retry of timed-out vetting
    assert "fieldid = N" in t["density_maps"] and "never retry the timed-out vetting call" in t["density_maps"]
    # open regions: presets, never Galactic Centre, Great Wall adopted window
    assert "Galactic Centre" in t["open_regions"] and "RA 150 to 220" in t["open_regions"]
    # researcher template
    assert "## Profile:" in t["researcher_profile"] and "h-index" in t["researcher_profile"]
    assert "alma_products" not in t


# ---------------------------------------------------------------------------
# Recovery request wording and turn context
# ---------------------------------------------------------------------------

def test_recovery_prompt_uses_card_wording_and_turn_context_in_v2(monkeypatch):
    from core.agent import QuasarAgent

    captured = {}

    class _Resp:
        output_text = "final"

    class _Client:
        class responses:
            @staticmethod
            def create(**kw):
                captured.update(kw)
                return _Resp()

    fake = SimpleNamespace(
        client=_Client(), system_prompt="SYS", prompt_bundle="v2",
        config=SimpleNamespace(temperature=0.1, max_tokens=500),
    )
    results = [{"type": "function_call_output", "call_id": "c1", "output": '{"success": true, "rows": 3}'}]
    out = QuasarAgent._compose_final_answer_from_tools(
        fake, "how many rows?", results, "gpt-oss-120b", turn_context="\n\nTurn context\n- Current date: 2026-09-25 (CDT)",
    )
    assert out == "final"
    prompt = captured["input"]
    assert "Refer to them as the card or the figure card" in prompt
    assert "displayed above your reply" not in prompt
    assert "Current date: 2026-09-25 (CDT)" in prompt
    assert "Do not call tools" in prompt

    fake.prompt_bundle = "legacy"
    QuasarAgent._compose_final_answer_from_tools(fake, "q", results, "gpt-oss-120b", turn_context="IGNORED")
    assert "displayed above your reply" in captured["input"] and "IGNORED" not in captured["input"]


def test_v2_flag_default_off(monkeypatch):
    monkeypatch.delenv(sc.ENV_FLAG, raising=False)
    assert sc.v2_enabled() is False
    monkeypatch.setenv(sc.ENV_FLAG, "1")
    assert sc.v2_enabled() is True


# ---------------------------------------------------------------------------
# Runner integration: the turn context reaches the provider request
# ---------------------------------------------------------------------------

def _runner_agent(bundle: str):
    import threading
    from tests.unit.test_response_api_error_recovery import _FakeResponses, _make_agent

    fake = _FakeResponses(failures=0, answer="ok")
    agent = _make_agent(fake)
    agent.prompt_bundle = bundle
    agent.system_prompt = "SYS-" + bundle
    return agent, fake


@pytest.mark.parametrize("bundle", ["v2", "legacy"])
def test_runner_round0_input_carries_turn_context_only_in_v2(bundle, monkeypatch):
    agent, fake = _runner_agent(bundle)
    monkeypatch.delenv("QUASAR_BENCH_TOOLSET", raising=False)
    agent.stream_response_api(
        "Show me a color image of the center of M31 from the DECam Legacy Surveys.",
        conversation_id="conv-v2-test",
    )
    assert fake.create_calls, "no provider request was made"
    first = fake.create_calls[0]
    body = first["input"] if isinstance(first["input"], str) else str(first["input"])
    if bundle == "v2":
        assert "Turn context" in body and "Current date:" in body
        assert "[PLAYBOOK survey_imagery]" in body
        assert first["instructions"].startswith("SYS-v2")
    else:
        assert "Turn context" not in body and "[PLAYBOOK" not in body


# ---------------------------------------------------------------------------
# Shim history hygiene: earlier turns' context is pruned on a new user turn
# ---------------------------------------------------------------------------

def test_prune_turn_context_history_cuts_old_blocks_and_keeps_evidence():
    from core.llm_client import prune_turn_context_history

    ctx = pb.render_turn_context("2026-09-25 (CDT)", "", [pb.get_playbook("wd_selection")])
    hist = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "User: find white dwarfs" + ctx},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"rows": 3}'},
        {"role": "user", "content": pb.render_followup_item([pb.get_playbook("density_maps")])["content"]},
        {"role": "assistant", "content": "answer"},
    ]
    pruned = prune_turn_context_history(hist)
    assert [m["role"] for m in pruned] == ["system", "user", "assistant", "tool", "assistant"]
    assert pruned[1]["content"] == "User: find white dwarfs"
    assert hist[1]["content"].endswith(pb.get_playbook("wd_selection").text), "cache dicts must not be mutated"
    assert pruned[3] is hist[3]  # tool evidence untouched


def test_shim_prunes_context_only_on_new_user_turn(monkeypatch):
    from core.llm_client import ResponsesShim
    import threading

    fake = object.__new__(ResponsesShim)
    fake._history_lock = threading.Lock()
    fake._prune_turn_context = True
    ctx = pb.render_turn_context("2026-09-25 (CDT)", "", [pb.get_playbook("open_regions")])
    fake._history_cache = {"r1": [{"role": "system", "content": "S"}, {"role": "user", "content": "q1" + ctx}, {"role": "assistant", "content": "a1"}]}
    fake._build_chat_messages = lambda *a, **k: []
    # new user turn: pruned
    msgs = fake._chat_messages_for_input("r1", "IGNORED", "q2" + ctx)
    assert msgs[1]["content"] == "q1" and msgs[-1]["content"].startswith("q2")
    # tool-results round within a turn: not pruned (current turn's context must stay)
    msgs2 = fake._chat_messages_for_input("r1", "IGNORED", [{"type": "function_call_output", "call_id": "c", "output": "{}"}])
    assert msgs2[1]["content"] == "q1" + ctx and msgs2[-1]["role"] == "tool"
    # legacy bundle (gate off): nothing is pruned
    fake._prune_turn_context = False
    msgs3 = fake._chat_messages_for_input("r1", "IGNORED", "q3")
    assert msgs3[1]["content"] == "q1" + ctx


def test_prune_ignores_user_text_that_merely_quotes_the_words():
    from core.llm_client import prune_turn_context_history

    hist = [{"role": "user", "content": "In the docs it says: Turn context\n- Current date: is shown per turn. Turn context update (applies to this turn only): also."}]
    assert prune_turn_context_history(hist) == hist


def test_prune_handles_anthropic_content_blocks():
    from core.llm_client import prune_turn_context_history

    ctx = pb.render_turn_context("2026-09-25 (CDT)", "", [])
    hist = [{"role": "user", "content": [{"type": "text", "text": "q1" + ctx}, {"type": "image", "source": {}}]},
            {"role": "user", "content": [{"type": "text", "text": pb.render_followup_item([pb.get_playbook("wd_selection")])["content"]}]}]
    pruned = prune_turn_context_history(hist)
    assert len(pruned) == 1 and pruned[0]["content"][0]["text"] == "q1" and pruned[0]["content"][1]["type"] == "image"
