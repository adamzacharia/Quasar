"""Regression tests for the fixes from the 2026-09-26 DeepSeek-v4-flash vs
gpt-oss-120b comparison (Benchmark/modelcompare/RESULTS-deepseek.md).

F1 DeepSeek reasoning effort, F2 RAG framing / live archive facts, F3 pre-tool
narration and requested identifiers, F4 LS DR9 observed magnitudes, F5 SMASH
release fall-through, F6 Gaia x AllWISE best-neighbour count, F7 Data Lab
Python template, F8 result field names in prose. ArchiveBench DEV questions
only (the held-out half is never used in tests).
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from core import rag_routing as rr
from core.answer_verifier import requested_items_missing
from core.llm_client import LLMClient
from core.oneshot_routing import detect_oneshot_intent
from core.prose_hygiene import humanize_prose
from tests.unit.test_archive_catalogs import FakeClient, _cols
from tests.unit.test_discovery_recovery import _Responses, _events, _tool_agent
from services.archive_catalogs import ArchiveCatalogService

# ArchiveBench dev prompts (Benchmark/archivebench/dev/questions.jsonl)
AB_D_41 = ("How many confirmed planets orbit host stars within 10 parsecs, according to the NASA Exoplanet "
           "Archive as of 24 September 2026?")
AB_D_48 = ("Who is the PI of ALMA project 2016.1.00484.L, what is the project called, and how many member "
           "observing units does the archive hold for it?")
AB_D_49 = "Is there public ALMA Band 3 data on the Bullet Cluster, and what angular resolution does it reach?"
AB_D_51 = ("Which public ALMA projects cover the CO(3-2) line for the starburst galaxy NGC 253, taking its "
           "systemic velocity as 243 km/s?")
AB_D_55 = ("Build a g against g-i colour-magnitude diagram from the SMASH survey for the SMC bar, using everything "
           "within 0.3 degrees of RA 13.19, Dec -72.83. How many stars go into it?")
AB_D_57 = ("Give me Python that pulls the DESI DR1 redshifts within 1 degree of Abell 2029 from the Data Lab and "
           "histograms them. How many good redshifts are there?")


# ── F1: DeepSeek reasoning effort ───────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    (None, "high"), ("", "high"), ("max", "max"), (" MAX ", "max"), ("medium", "medium"),
    ("xhigh", "xhigh"), ("bogus", "high"), ("none", "high"),  # "none" would switch thinking off
])
def test_deepseek_reasoning_effort_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DEEPSEEK_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", value)
    shim = LLMClient(model="deepseek-v4-flash").responses
    assert shim._deepseek_reasoning_effort() == expected


def _deepseek_client(captured):
    chunks = [NS(usage=None, choices=[NS(finish_reason="stop",
                                         delta=NS(content="ok", tool_calls=None, reasoning_content=None))])]

    def create(**kw):
        captured.append(kw)
        return iter(chunks) if kw.get("stream") else NS(
            id="r1", choices=[NS(finish_reason="stop", message=NS(content="ok", tool_calls=None, reasoning_content=None))],
            usage=NS(prompt_tokens=1, completion_tokens=1, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=1))
    return NS(chat=NS(completions=NS(create=create)))


def test_deepseek_stream_sends_configured_effort_and_logs_it(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "high")
    captured = []
    client = LLMClient(model="deepseek-v4-flash")
    client._get_deepseek_client = lambda: _deepseek_client(captured)
    list(client.responses._stream_deepseek({"model": "deepseek-v4-flash", "input": "hi", "max_output_tokens": 100}))
    assert captured[0]["reasoning_effort"] == "high"
    assert captured[0]["extra_body"] == {"thinking": {"type": "enabled"}}  # thinking stays on
    assert "reasoning_effort=high" in capsys.readouterr().out


def test_deepseek_non_stream_sends_configured_effort(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_REASONING_EFFORT", "medium")
    captured = []
    client = LLMClient(model="deepseek-v4-flash")
    client._get_deepseek_client = lambda: _deepseek_client(captured)
    client.responses._call_deepseek({"model": "deepseek-v4-flash", "input": "hi", "max_output_tokens": 100})
    assert captured[0]["reasoning_effort"] == "medium"
    assert "reasoning_effort=medium" in capsys.readouterr().out


# ── F2: live archive facts vs documentation questions ──────────────────────

@pytest.mark.parametrize("q", [AB_D_41, AB_D_48, AB_D_49, AB_D_51, "What is the title of 2019.1.00261.L?"])
def test_live_archive_fact_requests_are_detected(q):
    assert rr.live_archive_fact_request(q)
    assert not rr.documentation_rag_override(q)
    assert not rr.knowledge_question(q)
    assert rr.rag_directive(q) == rr.BACKGROUND_DIRECTIVE


@pytest.mark.parametrize("q", [
    "I downloaded the ALMA archive package for project 2019.1.00001.S and want to make a continuum image in "
    "CASA. Give me the steps and a script.",
    "How do I find out who the PI of an ALMA project is?",
    "What is the proprietary period for ALMA Cycle 13 data?",
    "What are the different ALMA configurations and their corresponding angular resolutions?",
])
def test_documentation_questions_are_not_live_facts(q):
    assert not rr.live_archive_fact_request(q)


def test_howto_question_keeps_the_documentation_override_and_knowledge_directive():
    q = "How do I find out which CASA version was used to process the data from a given ALMA project?"
    assert rr.documentation_rag_override(q)
    assert rr.rag_directive(q) == rr.KNOWLEDGE_DIRECTIVE
    assert rr.rag_directive("What is the proprietary period for ALMA Cycle 13 data?") == rr.KNOWLEDGE_DIRECTIVE


def test_background_directive_sends_facts_to_tools_not_to_the_excerpts():
    d = rr.BACKGROUND_DIRECTIVE
    assert "call the archive tools" in d and "background only" in d
    assert "Answer using the DOCUMENTATION CONTEXT" not in d
    assert "live archive fact" in rr.KNOWLEDGE_DIRECTIVE  # even how-to answers query for facts


def test_live_fact_question_forces_a_round0_tool_call():
    responses = _Responses([_events(1, tool="query_archive"), _events(2, text="The PI is A. Person.")])
    agent, executed = _tool_agent(responses)
    agent.stream_response_api(AB_D_48, conversation_id="live-fact")
    assert responses.calls[0]["tool_choice"] == "required" and len(executed) == 1


# ── F3: pre-tool narration and requested identifiers ───────────────────────

def test_short_pre_tool_narration_becomes_a_thinking_step_not_answer_text():
    responses = _Responses([_events(1, tool="query_archive", text="Let me retry with broader phrasing."),
                            _events(2, text="There are 42 sources.")])
    agent, _ = _tool_agent(responses)
    statuses = []
    result = agent.stream_response_api("hello there", conversation_id="narr",
                                       on_status=lambda label, state: statuses.append(label))
    assert "Let me retry" not in result and "There are 42 sources." in result
    assert any(s.startswith("💭 Let me retry with broader phrasing.") for s in statuses)


def test_long_text_before_a_tool_call_stays_in_the_answer():
    long_text = "Part one of the answer. " * 40  # > 500 characters: answer prose, not narration
    responses = _Responses([_events(1, tool="query_archive", text=long_text), _events(2, text="Part two.")])
    agent, _ = _tool_agent(responses)
    result = agent.stream_response_api("hello there", conversation_id="long")
    assert "Part one of the answer." in result and "Part two." in result


def test_final_round_text_without_tool_calls_is_not_held_back():
    responses = _Responses([_events(1, text="Short final answer.")])
    agent, _ = _tool_agent(responses)
    tokens = []
    result = agent.stream_response_api("hello there", conversation_id="final", on_token=tokens.append)
    assert "Short final answer." in result and "Short final answer." in "".join(tokens)


_BIB = "2021NatAs...5..655G"


def test_requested_items_missing_kinds():
    ev = [{"output": json.dumps({"papers": [{"bibcode": _BIB, "doi": "10.1038/s41550-020-1174-4"}]})}]
    q = "Which paper first reported phosphine on Venus? Give me bibcodes."
    assert requested_items_missing(q, "It is rendered in the paper cards.", ev) == ["bibcodes"]
    assert requested_items_missing(q, f"Greaves et al. (2021), {_BIB}.", ev) == []
    assert requested_items_missing(q, f"```\n{_BIB}\n```", ev) == ["bibcodes"]  # code is not the answer
    assert requested_items_missing("Give me the DOI.", "See the card.", ev) == ["DOIs"]
    assert requested_items_missing(q, "Nothing found.", [{"output": "{}"}]) == []  # tools had none
    counted = [{"output": json.dumps({"success": True, "reported_count": 42})}]
    assert requested_items_missing("How many stars are there?", "The count is in the card.", counted) == ["a count"]
    assert requested_items_missing("How many stars are there?", "None were found.", counted) == []
    assert requested_items_missing("How many stars are there?", "There are 42 stars.", counted) == []
    assert requested_items_missing("Which ALMA project codes?", "2016.1.00484.L", [{"output": "2016.1.00484.L"}]) == []


def test_count_check_needs_a_tool_count_and_ignores_digits_inside_names():
    # guard review CX: a failed tool returned no count -> no re-ask ...
    failed = [{"output": json.dumps({"success": False, "error": "timeout"})}]
    assert requested_items_missing("How many sources?", "The query timed out.", failed) == []
    # ... and "Gaia DR3" / "2MASS" / "W1" digits are not an answer to "how many"
    counted = [{"output": json.dumps({"success": True, "summary": {"n_left_with_match": 721}})}]
    assert requested_items_missing("How many Gaia sources match?", "Gaia DR3 x 2MASS W1 results are in the cards.",
                                   counted) == ["a count"]
    assert requested_items_missing("How many Gaia sources match?", "1,394 Gaia DR3 sources; 721 match.", counted) == []


def test_alma_uids_count_as_observation_ids():
    uid = "uid://A001/X1284/X265"
    ev = [{"output": json.dumps({"success": True, "mous": [uid]})}]
    assert requested_items_missing("Give me the MOUS uids.", "See the cards.", ev) == ["observation ids"]
    assert requested_items_missing("Which observation ids?", f"The MOUS is {uid}.", ev) == []


def test_missing_bibcodes_trigger_one_synthesis_reask():
    tool_result = {"success": True, "papers": [{"bibcode": _BIB, "title": "Phosphine gas"}]}
    responses = _Responses([_events(1, tool="query_archive"),
                            _events(2, text="The full set is rendered in the paper cards.")])
    agent, _ = _tool_agent(responses, result=tool_result)
    asks = []
    agent._compose_final_answer_from_tools = lambda *a, **kw: asks.append(kw) or f"Greaves et al. 2021 ({_BIB})."
    result = agent.stream_response_api("Who first reported phosphine on Venus? Give me bibcodes.",
                                       conversation_id="reask")
    assert len(asks) == 1 and "bibcodes" in asks[0]["extra_instruction"]
    assert _BIB in result and "rendered in the paper cards" not in result


def test_reask_that_does_not_help_keeps_the_original_answer():
    tool_result = {"success": True, "papers": [{"bibcode": _BIB}]}
    responses = _Responses([_events(1, tool="query_archive"), _events(2, text="See the paper cards.")])
    agent, _ = _tool_agent(responses, result=tool_result)
    agent._compose_final_answer_from_tools = lambda *a, **kw: "Still no identifiers."
    result = agent.stream_response_api("Who first reported phosphine on Venus? Give me bibcodes.",
                                       conversation_id="reask2")
    assert "See the paper cards." in result


def test_paper_prompts_allow_naming_requested_bibcodes():
    from core.prompts import system_core, tool_description_overrides as tdo
    import core.agent as agent_mod
    src = open(agent_mod.__file__, encoding="utf-8").read()
    assert "output NOTHING" not in src and "name each one in the text" in src
    assert "bibcodes" in tdo.OVERRIDES["search_papers"] and "write no text" not in tdo.OVERRIDES["search_papers"]
    assert "bibcodes" in open(system_core.__file__, encoding="utf-8").read()


# ── F4: LS DR9 observed magnitudes ─────────────────────────────────────────

def test_ls_dr9_exposes_observed_magnitudes():
    from services import datalab_registry as reg
    cols = reg.DATALAB_CATALOGS["ls_dr9"]["tables"]["tractor"]["columns"]
    assert {"mag_g", "mag_r", "mag_z", "dered_mag_r"} <= set(cols)
    assert cols.index("mag_r") < cols.index("dered_mag_r")
    assert "mag_r" in reg.describe_table("ls_dr9", "tractor")["columns"]
    profile = reg.TABLE_PROFILE_INFO["ls_dr9.tractor"]["columns"]
    assert "brighter than r = 20" in profile["mag_r"][3] and "only when" in profile["dered_mag_r"][3]
    pit = next(p for p in reg._PROFILE_PITFALLS if p["id"] == "per_band_columns")
    assert "plain 'r < 20' cut is mag_r" in pit["detail"]


# ── F5: SMASH DR1 -> DR2 ────────────────────────────────────────────────────

def test_cmd_route_takes_g_against_g_minus_i_on_smash_dr2():
    intent = detect_oneshot_intent(AB_D_55)
    assert intent["tool"] == "datalab_color_magnitude_diagram"
    assert intent["args"]["catalog"] == "smash_dr2" and intent["args"]["red_band"] == "i"
    assert "g vs g-i" in intent["directive"]


def test_empty_smash_dr1_cmd_falls_through_to_dr2(monkeypatch):
    from capabilities import datalab as dl
    from capabilities.base import CallContext
    calls = []

    def fake_cmd(catalog, table, *a, **kw):
        calls.append((catalog, table))
        return {"success": True, "rowcount": 0 if catalog == "smash_dr1" else 1234}

    monkeypatch.setattr(dl.datalab_orchestration, "color_magnitude_diagram", fake_cmd)
    monkeypatch.setattr(dl.ColorMagnitudeDiagram, "_population_features", staticmethod(lambda *a, **k: {}))
    ctx = CallContext(services={"resolve_coordinates": lambda target_name=None, ra=None, dec=None, **k: (ra, dec, "SMC bar")},
                      result_store=None)
    out = dl.ColorMagnitudeDiagram().run(
        dl.ColorMagnitudeDiagramInput(catalog="smash_dr1", ra=13.19, dec=-72.83, radius_deg=0.3,
                                      blue_band="g", red_band="i"), ctx).to_native()
    assert calls == [("smash_dr1", "object"), ("smash_dr2", "object")]
    assert out["rowcount"] == 1234 and out["release_fallback"]["used"] == "smash_dr2"
    assert any("newer release smash_dr2" in w for w in out["warnings"])


@pytest.mark.parametrize("dr2", ["raise", "fail"])
def test_unavailable_dr2_keeps_the_dr1_result_with_a_warning(monkeypatch, dr2):
    from capabilities import datalab as dl
    from capabilities.base import CallContext

    def fake_cmd(catalog, table, *a, **kw):
        if catalog == "smash_dr1":
            return {"success": True, "rowcount": 0}
        if dr2 == "raise":
            raise RuntimeError("Data Lab timeout")
        return {"success": False, "error": "HTTP 503"}

    monkeypatch.setattr(dl.datalab_orchestration, "color_magnitude_diagram", fake_cmd)
    monkeypatch.setattr(dl.ColorMagnitudeDiagram, "_population_features", staticmethod(lambda *a, **k: {}))
    ctx = CallContext(services={"resolve_coordinates": lambda target_name=None, ra=None, dec=None, **k: (ra, dec, "x")},
                      result_store=None)
    res = dl.ColorMagnitudeDiagram().run(dl.ColorMagnitudeDiagramInput(catalog="smash_dr1", ra=13.19, dec=-72.83,
                                                                        radius_deg=0.3), ctx)
    out = res.to_native()
    assert res.success and out["rowcount"] == 0 and out["release_fallback"]["used"] == "smash_dr1"
    assert any("could not be checked" in w and "not evidence of no coverage" in w for w in out["warnings"])


def test_population_verdict_does_not_call_an_empty_query_a_coverage_gap():
    import inspect
    from services import cmd_population
    assert "coverage gap or empty query" not in inspect.getsource(cmd_population)


# ── F6: Gaia DR3 x AllWISE best neighbour ──────────────────────────────────

def _gaia_allwise_client(bn_rows):
    cols = {("gaia", "gaiadr3.gaia_source"): _cols(ra="ra", dec="dec"), ("irsa", "allwise_p3as_psd"): _cols(ra="ra", dec="dec")}
    rules = [(r"allwise_best_neighbour", bn_rows),
             (r"COUNT.*gaia_source", [{"n": 2}]), (r"SELECT TOP \d+ ra, dec FROM gaiadr3", [{"ra": 1.0, "dec": 1.0}, {"ra": 1.1, "dec": 1.0}]),
             (r"COUNT.*allwise_p3as_psd", [{"n": 1}]), (r"SELECT TOP \d+ ra, dec FROM allwise", [{"ra": 1.0, "dec": 1.0}])]
    return FakeClient(rules, cols)


def test_gaia_allwise_crossmatch_adds_the_best_neighbour_count():
    client = _gaia_allwise_client([{"n": 1, "n_allwise": 1}])
    out = ArchiveCatalogService(client).catalog_crossmatch(
        {"service": "gaia", "table": "gaiadr3.gaia_source"}, {"service": "irsa", "table": "allwise_p3as_psd"},
        ra=1.0, dec=1.0, radius_arcsec=600, match_radius_arcsec=1.0)
    bn = out["summary"]["gaia_best_neighbour"]
    assert bn["n_gaia_with_counterpart"] == 1 and bn["preferred"] and "result" not in bn
    assert out["summary"]["n_left_with_match"] == 1  # the positional count is still reported
    q = next(q for q in client.queries if "allwise_best_neighbour" in q)
    assert "x.angular_distance < 1.0" in q and "CIRCLE('ICRS', 1.0, 1.0, 0.16666666666666666)" in q


def test_best_neighbour_failure_is_a_warning_not_a_failed_crossmatch():
    from integrations.archive_tap import TapQueryError
    client = _gaia_allwise_client(TapQueryError("Gaia archive down", query="Q"))
    out = ArchiveCatalogService(client).catalog_crossmatch(
        {"service": "gaia", "table": "gaiadr3.gaia_source"}, {"service": "irsa", "table": "allwise_p3as_psd"},
        ra=1.0, dec=1.0, radius_arcsec=600, match_radius_arcsec=1.0)
    assert out["success"] and out["summary"]["gaia_best_neighbour"]["n_gaia_with_counterpart"] is None
    assert any("best-neighbour count failed" in w for w in out["warnings"])


def test_best_neighbour_is_skipped_for_other_pairs_and_with_cuts():
    from services.archive_catalogs import _survey_kind
    assert _survey_kind({"service": "vizier", "table": "I/355/gaiadr3"}) == "gaia_dr3"
    assert _survey_kind({"service": "vizier", "table": "II/328/allwise"}) == "allwise"
    assert _survey_kind({"service": "irsa", "table": "fp_psc"}) is None
    client = _gaia_allwise_client([{"n": 1}])
    client.rules = [(r"phot_g_mean_mag", [{"n": 2}])] + client.rules  # count query with the cut
    out = ArchiveCatalogService(client).catalog_crossmatch(
        {"service": "gaia", "table": "gaiadr3.gaia_source", "cuts": ["phot_g_mean_mag < 18"]},
        {"service": "irsa", "table": "allwise_p3as_psd"}, ra=1.0, dec=1.0, radius_arcsec=600, match_radius_arcsec=1.0)
    assert "gaia_best_neighbour" not in out.get("summary", {})


# ── F7: Data Lab Python recipe ─────────────────────────────────────────────

def test_desi_python_count_question_routes_to_the_count_first():
    intent = detect_oneshot_intent(AB_D_57)
    assert intent["tool"] == "datalab_cone_count"
    assert intent["args"]["catalog"] == "desi_dr1" and intent["args"]["table"] == "zpix"
    assert intent["args"]["radius_deg"] == 1.0
    assert intent["args"]["value_cuts"] == [{"column": "zwarn", "op": "=", "value": 0}]
    assert "python_code" in intent["directive"] and "astroquery.datalab" in intent["directive"]


@pytest.mark.parametrize("q", [
    "Give me Python that counts all DESI DR1 redshifts within 1 degree of Abell 2029 from Data Lab, including zwarn != 0.",
    "Write a Python script: how many DESI DR1 redshifts are there within 1 degree of Abell 2029 in Data Lab?",
])
def test_desi_python_route_keeps_the_users_own_selection(q):
    # guard review CX: zwarn = 0 only when the user asks for good redshifts
    intent = detect_oneshot_intent(q)
    assert intent["tool"] == "datalab_cone_count" and "value_cuts" not in intent["args"]
    assert "apply only the selection the user asked for" in intent["directive"]


def test_cone_count_returns_a_verified_client_script():
    from capabilities import datalab as dl
    from tests.unit.test_datalab_capability import _ctx
    ctx, client, _ = _ctx(pd.DataFrame({"row_count": [8032]}))
    out = dl.ConeCount().run(dl.ConeCountInput(
        catalog="desi_dr1", table="zpix", ra=227.7257, dec=5.7396, radius_deg=1.0,
        value_cuts=[{"column": "zwarn", "op": "=", "value": 0}]), ctx).to_native()
    code = out["python_code"]
    assert "from dl import queryClient as qc" in code and "astroquery" not in code
    assert client.last_sql.strip() in code  # the exact executed SQL
    assert "8032" in code and "plt.hist(df['z']" in code
    assert "SELECT targetid, mean_fiber_ra, mean_fiber_dec, z, zwarn, spectype" in code
    compile(code, "<template>", "exec")


# ── F8: result field names in prose ────────────────────────────────────────

def test_result_field_names_are_humanised_outside_code():
    text = ("| Stretch | Lupton et al. (2004) asinh (make_lupton_rgb) |\n"
            "| Coverage gap | None, coverage_gap: false |\n"
            "The `fov_rationale` says 15 arcmin and rgb_mapping is z/r/g.\n"
            "```python\nx = out['coverage_gap']\n```\n"
            "Selection: ls_dr9.tractor with dered_mag_r < 20.")
    out = humanize_prose(text, ["datalab_color_image"])
    assert "make_lupton_rgb" not in out and "no coverage gap" in out
    assert "field-of-view choice" in out and "colour mapping" in out
    assert "x = out['coverage_gap']" in out  # code untouched
    assert "ls_dr9.tractor" in out and "dered_mag_r" in out  # catalogue names stay
    assert humanize_prose(out, ["datalab_color_image"]) == out  # idempotent


# ── guard review task-dd87861-1630 (CX-03, 08, 10, 11, 12) ──────────────────

def test_cx03_project_pipeline_fact_is_a_live_archive_request():
    q = "Which ALMA pipeline version processed project 2016.1.00484.L?"
    assert rr.live_archive_fact_request(q) and not rr.documentation_rag_override(q)


def test_cx08_tilde_fenced_code_is_not_answer_prose():
    ev = [{"output": json.dumps({"papers": [{"bibcode": _BIB}]})}]
    assert requested_items_missing("Give me bibcodes.", f"~~~\n{_BIB}\n~~~", ev) == ["bibcodes"]


def test_cx10_capped_positional_side_is_not_called_the_looser_count():
    client = _gaia_allwise_client([{"n": 1, "n_allwise": 1}])
    client.rules = [(r"COUNT.*gaia_source AS g", [{"n": 1, "n_allwise": 1}]),
                    (r"COUNT.*gaia_source", [{"n": 50}])] + client.rules  # 50 in the cone, 2 pulled
    out = ArchiveCatalogService(client).catalog_crossmatch(
        {"service": "gaia", "table": "gaiadr3.gaia_source"}, {"service": "irsa", "table": "allwise_p3as_psd"},
        ra=1.0, dec=1.0, radius_arcsec=600, match_radius_arcsec=1.0)
    note = out["summary"]["gaia_best_neighbour"]["note"]
    assert out["summary"]["complete"] is False and "not comparable" in note and "looser" not in note


def test_cx11_large_cone_template_is_row_bounded():
    from capabilities.datalab import datalab_python_template
    sql = "SELECT COUNT(*) AS row_count\nFROM desi_dr1.zpix\nWHERE q3c_radial_query(mean_fiber_ra, mean_fiber_dec, 1, 2, 3)"
    big = datalab_python_template(sql, 2_500_000, "desi_dr1", "zpix")
    assert "LIMIT 100000" in big and "async" in big
    assert "LIMIT" not in datalab_python_template(sql, 8032, "desi_dr1", "zpix")


def test_cx12_inline_code_field_with_boolean_is_humanised():
    out = humanize_prose("`coverage_gap`: false and `coverage_gap`: true.\n```\ncoverage_gap = False\n```", [])
    assert out.startswith("no coverage gap and a coverage gap.") and "coverage_gap = False" in out


# ── guard verify round 1 (CX-06, 08 reopened; CX-26..29 new) ────────────────

def test_cx06_a_bare_year_is_not_a_count_but_a_tool_count_is():
    counted = [{"output": json.dumps({"success": True, "reported_count": 42})}]
    assert requested_items_missing("How many sources?", "The 2026 results are in the cards.", counted) == ["a count"]
    year_count = [{"output": json.dumps({"success": True, "reported_count": 2016})}]
    assert requested_items_missing("How many sources?", "There are 2016 of them.", year_count) == []


def test_cx08_unclosed_fence_is_not_prose():
    ev = [{"output": json.dumps({"papers": [{"bibcode": _BIB}]})}]
    assert requested_items_missing("Give me bibcodes.", f"Here:\n~~~\n{_BIB}\n", ev) == ["bibcodes"]
    assert requested_items_missing("Give me bibcodes.", f"Here:\n```\n{_BIB}\n", ev) == ["bibcodes"]


def test_cx26_project_specific_howto_stays_documentation():
    assert not rr.live_archive_fact_request("Give me CASA commands to calibrate ALMA project 2016.1.00484.L")
    assert rr.live_archive_fact_request("Which calibrator was used for ALMA project 2016.1.00484.L?")


def test_cx27_numeric_string_counts_are_tool_counts():
    ev = [{"output": json.dumps({"success": True, "count": "42"})}]
    assert requested_items_missing("How many?", "The results are in the cards.", ev) == ["a count"]


def test_cx28_unclosed_fence_code_is_not_rewritten():
    out = humanize_prose("Result below.\n```python\nflags = {'x': 1}\ncoverage_gap: false\n", [])
    assert "coverage_gap: false" in out


def test_cx29_capped_histogram_title_says_it_is_a_slice():
    from capabilities.datalab import datalab_python_template
    sql = "SELECT COUNT(*) AS row_count\nFROM desi_dr1.zpix\nWHERE q3c_radial_query(mean_fiber_ra, mean_fiber_dec, 1, 2, 3)"
    big = datalab_python_template(sql, 2_500_000, "desi_dr1", "zpix")
    assert "first 100000 of 2500000 redshifts" in big and "'desi_dr1.zpix: 2500000 redshifts'" not in big


# ── guard verify round 2 (CX-06, 08, 26, 28) ─────────────────────────────────

@pytest.mark.parametrize("prose", ["The ALMA Band 6 result is in the card.", "The field is at RA 150; see the card.",
                                   "Cycle 10 data are in the card.", "Within 5 arcmin, see the card."])
def test_cx06_label_numbers_are_not_counts(prose):
    counted = [{"output": json.dumps({"success": True, "reported_count": 42})}]
    assert requested_items_missing("How many sources?", prose, counted) == ["a count"]


def test_cx06_counts_next_to_labels_still_count():
    counted = [{"output": json.dumps({"success": True, "reported_count": 431})}]
    assert requested_items_missing("How many sources?", "There are 431 sources within 5 arcmin.", counted) == []
    assert requested_items_missing("How many sources?", "About 430 sources lie within 5 arcmin.", counted) == []


def test_cx08_cx28_four_backtick_fences_contain_three_backtick_lines():
    from core.prose_hygiene import split_fences
    text = f"Intro.\n````markdown\n```\n{_BIB}\ncoverage_gap: false\n```\n````\nAfter."
    parts = split_fences(text)
    assert [f for f, _ in parts] == [False, True, False]
    ev = [{"output": json.dumps({"papers": [{"bibcode": _BIB}]})}]
    assert requested_items_missing("Give me bibcodes.", text, ev) == ["bibcodes"]
    assert "coverage_gap: false" in humanize_prose(text, [])


@pytest.mark.parametrize("q, live", [
    ("Which CASA command was used to calibrate ALMA project 2016.1.00484.L?", True),
    ("Give me CASA commands to calibrate ALMA project 2016.1.00484.L", False),
    ("Give me the title and PI of ALMA project 2016.1.00484.L.", True),
    ("I downloaded the ALMA archive package for project 2019.1.00001.S and want to make a continuum image in CASA. "
     "Give me the steps and a script.", False),
    ("Which ALMA pipeline version processed project 2016.1.00484.L?", True),
])
def test_cx26_processing_needs_a_request_and_a_task(q, live):
    assert rr.live_archive_fact_request(q) is live


# ── guard verify round 3 (CX-06, CX-26) ─────────────────────────────────────

@pytest.mark.parametrize("prose, n", [("The ALMA Band 6 result is in the card.", 6),
                                      ("The field is at RA 150; see the card.", 150)])
def test_cx06_label_equal_to_the_count_is_still_not_a_count(prose, n):
    ev = [{"output": json.dumps({"success": True, "reported_count": n})}]
    assert requested_items_missing("How many sources?", prose, ev) == ["a count"]


def test_cx26_mixed_fact_and_processing_request_keeps_the_archive_call():
    q = "Give me the PI and title of ALMA project 2016.1.00484.L, and explain how to calibrate it in CASA"
    assert rr.live_archive_fact_request(q) and not rr.documentation_rag_override(q)


# ── guard verify round 4 (CX-06, CX-26) ─────────────────────────────────────

@pytest.mark.parametrize("prose, missing", [
    ("The cut is r < 20; see the card.", True),
    ("The match radius is 1 arcsec; see the card.", True),
    ("About 430 sources pass the cut.", False),
    ("The count is 42.", False),
    ("12 TESS sectors have 2-minute data.", False),
])
def test_cx06_parameters_are_not_counts_but_counted_nouns_are(prose, missing):
    ev = [{"output": json.dumps({"success": True, "reported_count": 42})}]
    assert (requested_items_missing("How many?", prose, ev) == ["a count"]) is missing


def test_cx26_public_in_a_processing_request_is_not_a_fact_ask():
    assert not rr.live_archive_fact_request("Give me CASA calibration steps for public ALMA project 2016.1.00484.L")
    assert rr.live_archive_fact_request("Give me CASA steps for ALMA project 2016.1.00484.L and tell me its PI")
