"""RE-B1..RE-B4 regression tests (Robert Nikutta NOIRLab beta-eval fixes, 2026-08).

Covers the science-correctness batch:
  B1 — no unrequested LIMIT: platform-applied caps are flagged (SQL comment +
       platform_row_cap metadata + warning), never silently baked in.
  B2 — sentinel-magnitude hygiene: one named constant, self-describing SQL,
       ranked prompt pitfall, truthful tool descriptions.
  B3 — star/galaxy threshold guardrails: morphology deviation warning,
       DES table pitfall, tightened DLB-05 rubric regex.
  B4 — "color image" honesty: routing, on-survey LS/DECam HiPS aliases,
       CDS-vs-Data Lab source_service provenance labels.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services import datalab_orchestration as orch
from services import datalab_query_builders as B
from services import datalab_registry as reg
from services import datalab_sql_policy as P
from services.datalab_result_store import DatalabResultStore

BENCH_DIR = str(Path(__file__).resolve().parents[2] / "Benchmark" / "datalabbench")
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

PLATFORM_COMMENT = "/* platform row cap, not a science cut */"
SENTINEL_COMMENT = "/* sentinel-removal guard, not a science cut */"


# ─────────────────────────────────────────────────────────────────────────────
# RE-B1 — no unrequested LIMIT; platform caps always disclosed
# ─────────────────────────────────────────────────────────────────────────────
def test_cone_select_flags_platform_default_limit():
    sql, meta = B.build_cone_select("gaia_dr3", "gaia_source", ra=10, dec=0, radius_deg=0.1)
    assert sql.endswith(f"LIMIT 500 {PLATFORM_COMMENT}")
    assert meta["platform_row_cap"] == 500
    assert any("not a science cut" in w and "not user-requested" in w for w in meta["warnings"])


def test_cone_select_explicit_limit_is_users_choice_not_flagged():
    sql, meta = B.build_cone_select("gaia_dr3", "gaia_source", ra=10, dec=0, radius_deg=0.1, limit=1000)
    assert sql.endswith("LIMIT 1000")
    assert PLATFORM_COMMENT not in sql
    assert "platform_row_cap" not in meta
    assert not any("Row cap" in w for w in (meta.get("warnings") or []))


def test_cone_select_clamped_limit_is_flagged_not_silent():
    sql, meta = B.build_cone_select("gaia_dr3", "gaia_source", ra=10, dec=0, radius_deg=0.1, limit=999999)
    assert sql.endswith(f"LIMIT 5000 {PLATFORM_COMMENT}")
    assert meta["platform_row_cap"] == 5000
    assert any("exceeds the platform ceiling" in w for w in meta["warnings"])


def test_platform_capped_builder_sql_passes_the_governor_and_keeps_flags():
    sql, meta = B.build_cone_select("gaia_dr3", "gaia_source", ra=10, dec=0, radius_deg=0.1)
    validated = P.validate(sql, source="builder", meta=meta)
    assert validated.sql == sql  # the self-describing SQL is untouched
    assert validated.meta.get("platform_row_cap") == 500
    assert any("not a science cut" in w for w in validated.warnings)


def test_governor_injected_cap_on_expert_sql_is_self_describing():
    out = P.validate(
        "SELECT ra, dec FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 10, 10, 0.1)",
        source="expert",
    )
    assert out.sql.endswith(f"LIMIT {P.DEFAULT_ROW_CAP} {PLATFORM_COMMENT}")
    assert out.meta.get("platform_row_cap") == P.DEFAULT_ROW_CAP


def test_other_builders_flag_platform_defaults_too():
    for sql, meta in [
        B.build_variable_star_select(source_id="abc123"),
        B.build_variability_rank("smash_dr1", ra=15.0, dec=-72.0, radius_deg=0.3),
        B.build_density_aggregate("nsc_dr2", "object", mode="grid", ra=10, dec=0, radius_deg=1.0),
        B.build_zhistogram("desi_dr1", "zpix"),
        B.build_q3c_crossmatch(ra=229.022, dec=-0.112, radius_deg=1.0),
    ]:
        assert PLATFORM_COMMENT in sql, sql
        assert meta.get("platform_row_cap"), sql
        # And every flagged query still validates cleanly through the governor.
        assert P.validate(sql, source="builder", meta=meta).sql == sql


def test_golden_example_has_no_limit():
    # RE-B1: the des_per_band_cmd golden example seeded literal LIMIT 5000
    # into model SQL, contradicting the row_caps_are_governance pitfall.
    example = next(g for g in reg._PROFILE_GOLDEN_EXAMPLES if g["id"] == "des_per_band_cmd")
    assert "LIMIT" not in example["invocation"]["arguments"]["sql"].upper()
    assert "never add a LIMIT" in example["note"]


class _FakeNSCClient:
    """NSC-style diagram client recording every executed SQL."""

    def __init__(self):
        self.sql = []

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        from integrations.datalab_client import DatalabResult
        self.sql.append(sql or "")
        rng = np.random.default_rng(3)
        n = 150
        df = pd.DataFrame({
            "ra": rng.uniform(259.7, 260.4, n), "dec": rng.uniform(57.6, 58.2, n),
            "gmag": rng.uniform(17, 24, n), "rmag": rng.uniform(16.5, 23.5, n),
        })
        return DatalabResult.from_dataframe(df, {"catalog": "nsc_dr2", "table": "object", "query": sql})


class _FakeDESClient(_FakeNSCClient):
    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        from integrations.datalab_client import DatalabResult
        self.sql.append(sql or "")
        rng = np.random.default_rng(4)
        n = 150
        df = pd.DataFrame({
            "ra": rng.uniform(29.5, 30.5, n), "dec": rng.uniform(-50.5, -49.5, n),
            "mag_auto_g": rng.uniform(18, 23, n), "mag_auto_r": rng.uniform(17, 22, n),
            "mag_auto_i": rng.uniform(16.5, 21.5, n),
            "spread_model_r": rng.normal(0.0, 0.001, n),
        })
        return DatalabResult.from_dataframe(df, {"catalog": "des_dr1", "table": "main", "query": sql})


def test_cmd_default_limit_is_flagged_platform_budget():
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeNSCClient()
    out = orch.color_magnitude_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4, blue_band="g", red_band="r",
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    # The CMD plotting budget (5000) is a platform cap: self-describing SQL,
    # machine-readable provenance, disclosed in warnings.
    assert f"LIMIT 5000 {PLATFORM_COMMENT}" in client.sql[0]
    assert out["provenance"]["platform_row_cap"] == 5000
    assert any("not a science cut" in w and "Row cap" in w for w in out["warnings"])


def test_cmd_explicit_limit_stays_unflagged():
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeNSCClient()
    out = orch.color_magnitude_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4, blue_band="g", red_band="r", limit=1234,
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    assert "LIMIT 1234" in client.sql[0] and PLATFORM_COMMENT not in client.sql[0]
    assert "platform_row_cap" not in out["provenance"]


def test_ccd_default_limit_uses_ccd_budget():
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeDESClient()
    out = orch.color_color_diagram(
        "des_dr1", "main", 30.0, -50.0, 0.5,
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    assert f"LIMIT 3000 {PLATFORM_COMMENT}" in client.sql[0]
    assert out["provenance"]["platform_row_cap"] == 3000


# ─────────────────────────────────────────────────────────────────────────────
# RE-B2 — sentinel-magnitude hygiene
# ─────────────────────────────────────────────────────────────────────────────
def test_sentinel_constant_is_unified():
    assert reg.SENTINEL_MAG_RANGE == (-5.0, 50.0)
    assert B.SENTINEL_MAG_RANGE is reg.SENTINEL_MAG_RANGE
    assert orch._VALID_MAG_RANGE is reg.SENTINEL_MAG_RANGE


def test_epoch_builders_use_unified_annotated_sentinel_guard():
    var_sql, _ = B.build_variable_star_select(source_id="abc123")
    assert "cmag < 99" not in var_sql  # the old unexplained threshold is gone
    assert f"cmag > -5 AND cmag < 50 {SENTINEL_COMMENT}" in var_sql
    rank_sql, _ = B.build_variability_rank("smash_dr1", ra=15.0, dec=-72.0, radius_deg=0.3, limit=50)
    assert f"cmag > -5 AND cmag < 50 {SENTINEL_COMMENT}" in rank_sql


def test_diagram_validity_guard_sql_is_self_describing():
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeNSCClient()
    out = orch.color_magnitude_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4, blue_band="g", red_band="r",
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    sql = client.sql[0]
    # The injected -5/50 guard carries its own explanation in the SQL text.
    assert SENTINEL_COMMENT in sql
    assert "gmag < 50" in sql and "rmag < 50" in sql


def test_period_fold_excludes_negative_sentinels_via_unified_range():
    from services import datalab_analysis
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    store = DatalabResultStore(enable_disk_cache=False)
    period = 0.6
    t = np.linspace(0, 18, 160)
    mag = 15.0 + 0.25 * np.sin(2 * np.pi * t / period)
    mag[::5] = 99.99   # positive sentinel padding
    mag[1::7] = -99.0  # negative sentinel padding
    lc_id = store.put(pd.DataFrame({"mjd": t, "cmag": mag, "cerr": np.full_like(t, 0.03)}),
                      {"catalog": "synthetic", "table": "lc"})
    folded = datalab_analysis.period_fold(lc_id, min_frequency=1.0, max_frequency=4.0,
                                          result_store=store, plotting_service=_MemoryPlottingService())
    lo, hi = reg.SENTINEL_MAG_RANGE
    assert folded["success"]
    assert folded["points"] == int(np.sum((mag > lo) & (mag < hi)))


def test_magnitude_cut_hygiene_pitfall_is_prompt_ranked():
    from services.archive_profiles import get_profile, prompt_index_lines
    profile = get_profile("datalab")
    pitfall = next(p for p in profile.pitfalls if p.id == "magnitude_cut_hygiene")
    assert pitfall.prompt_rank is not None  # unranked pitfalls never reach the prompt
    datalab_line = next(line for line in prompt_index_lines() if line.startswith("- datalab:"))
    assert "sentinel removal" in datalab_line
    assert len(datalab_line) < 600  # CX-07 bound (3 ranked pitfalls x 160 chars)


def test_tool_descriptions_state_the_real_sentinel_guard():
    # RE-B2: descriptions claimed "sentinel magnitudes (99.99) are excluded"
    # while the real guard is -5 < mag < 50 — they must be truthful now.
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    agent._register_tools()
    for name in ("datalab_color_color_diagram", "datalab_color_magnitude_diagram"):
        desc = agent.tool_registry.get_tool(name).description
        assert "-5 < mag < 50" in desc
        assert "not a science cut" in desc
        assert "99.99" not in desc
        # The advertised limit default is gone; the cap is platform-applied.
        limit_schema = agent.tool_registry.get_tool(name).parameters["properties"]["limit"]
        assert "default" not in limit_schema


# ─────────────────────────────────────────────────────────────────────────────
# RE-B3 — star/galaxy threshold guardrails
# ─────────────────────────────────────────────────────────────────────────────
def test_morphology_deviation_warns_on_class_star_scale_applied_to_spread_model():
    warning = B.morphology_deviation_warning(
        "des_dr1", "main", {"column": "spread_model_r", "op": "<", "value": 0.5})
    assert warning is not None
    assert "deviates" in warning and "NOT clamped" in warning
    # Per-band family match: spread_model_g compares against the _r convention.
    assert B.morphology_deviation_warning(
        "des_dr1", "main", {"column": "spread_model_g", "op": ">", "value": 0.5}) is not None


def test_morphology_deviation_stays_quiet_for_sane_cuts():
    # Same-scale variations of the registered conventions never warn.
    assert B.morphology_deviation_warning(
        "des_dr1", "main", {"column": "spread_model_r", "op": ">", "value": 0.005}) is None
    assert B.morphology_deviation_warning(
        "nsc_dr2", "object", {"column": "class_star", "op": ">", "value": 0.5}) is None
    assert B.morphology_deviation_warning(
        "nsc_dr2", "object", {"column": "class_star", "op": ">=", "value": 0.9}) is None
    # Different column family than the registered cut: nothing to compare.
    assert B.morphology_deviation_warning(
        "des_dr1", "main", {"column": "class_star_r", "op": ">", "value": 0.9}) is None
    # No registered convention (gaia) / unknown table: quiet, never raises.
    assert B.morphology_deviation_warning(
        "gaia_dr3", "gaia_source", {"column": "ruwe", "op": "<", "value": 1.4}) is None
    assert B.morphology_deviation_warning("nope", "missing", {"column": "x", "op": ">", "value": 1}) is None
    assert B.morphology_deviation_warning("des_dr1", "main", None) is None


def test_morphology_deviation_warns_on_reverse_conflation():
    # spread_model-scale value on a class_star column (the mirror mistake).
    warning = B.morphology_deviation_warning(
        "nsc_dr2", "object", {"column": "class_star", "op": ">", "value": 0.003})
    assert warning is not None and "deviates" in warning


def test_cmd_surfaces_morphology_deviation_warning_in_result():
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeDESClient()
    out = orch.color_magnitude_diagram(
        "des_dr1", "main", 30.0, -50.0, 0.4, blue_band="g", red_band="r",
        morphology={"column": "spread_model_r", "op": "<", "value": 0.5},
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    # The cut executes as given (never clamped) AND the warning is loud.
    assert "spread_model_r < 0.5" in client.sql[0]
    assert any("MORPHOLOGY THRESHOLD CHECK" in w for w in out["warnings"])


def test_des_morphology_pitfall_registered_table_scoped():
    from services.archive_profiles import get_profile
    profile = get_profile("datalab")
    pitfall = next(p for p in profile.pitfalls if p.id == "des_morphology_threshold")
    assert any(ref.kind == "table" and ref.ref == "des_dr1.main" for ref in pitfall.applies_to)
    assert "0.003" in pitfall.summary and "class_star-ONLY" in pitfall.summary


def _dlb05_c3_earned(ev):
    from dlb_dataset_v1 import get_question
    from run_datalabbench import score_auto_checkpoints
    q = get_question("DLB-05")
    scores = {c.id: c.earned for c in score_auto_checkpoints(q, ev)}
    return scores["C3"]


def test_dlb05_regex_positive_sql_and_structured_args():
    from run_datalabbench import Evidence, ToolCall
    # SQL form: threshold adjacent to spread_model — full C3 credit.
    sql = ("SELECT mag_auto_g - mag_auto_r AS gr FROM des_dr1.main "
           "WHERE q3c_radial_query(ra, dec, 30.0, -50.0, 0.5) "
           "AND spread_model_r > 0.003")
    ev = Evidence()
    ev.calls = [ToolCall(name="datalab_sql_query", arguments={"sql": sql},
                         output='{"success": true}', ok=True, sql=sql)]
    ev.sql_texts = [sql]
    ev.trace_source = "tool_trace"
    assert _dlb05_c3_earned(ev) == 20.0

    # Structured-args form (morphology cut object) — full credit too.
    ev2 = Evidence()
    ev2.calls = [ToolCall(
        name="datalab_color_color_diagram",
        arguments={"catalog": "des_dr1", "ra": 30.0, "dec": -50.0,
                   "morphology": {"column": "spread_model_r", "op": ">", "value": 0.003}},
        output='{"success": true}', ok=True)]
    ev2.trace_source = "tool_trace"
    assert _dlb05_c3_earned(ev2) == 20.0


def test_dlb05_regex_rejects_unrelated_threshold_literal():
    from run_datalabbench import Evidence, ToolCall
    # v1.3 tightening: a WRONG spread_model threshold (0.5) plus an unrelated
    # 0.00x literal elsewhere (step_deg) must NOT earn the threshold credit —
    # under v1.2 any 0.00[2-9] anywhere in the trace passed.
    sql = ("SELECT gr FROM des_dr1.main "
           "WHERE q3c_radial_query(ra, dec, 30.0, -50.0, 0.5) AND spread_model_r > 0.5")
    ev = Evidence()
    ev.calls = [
        ToolCall(name="datalab_sql_query", arguments={"sql": sql},
                 output='{"success": true}', ok=True, sql=sql),
        ToolCall(name="datalab_density_aggregate",
                 arguments={"catalog": "des_dr1", "table": "main", "step_deg": 0.005,
                            "ra": 30.0, "dec": -50.0, "radius_deg": 0.5},
                 output='{"success": true}', ok=True),
    ]
    ev.sql_texts = [sql]
    ev.trace_source = "tool_trace"
    # spread_model still matches (1 of 2 checks) but the threshold check fails.
    assert _dlb05_c3_earned(ev) == 10.0


def test_dlb14_c3_accepts_both_sentinel_guard_spellings():
    # v1.3: the structured lightcurve builder now emits `cmag > -5 AND cmag < 50`
    # instead of `cmag < 99` — correct tool behavior must keep the C3 credit,
    # and the PDF's raw-SQL spelling must keep it too.
    from dlb_dataset_v1 import get_question
    from run_datalabbench import Evidence, ToolCall, score_auto_checkpoints
    q = get_question("DLB-14")

    def _c3(sql):
        ev = Evidence()
        ev.calls = [ToolCall(name="datalab_sql_query", arguments={"sql": sql},
                             output='{"success": true}', ok=True, sql=sql)]
        ev.sql_texts = [sql]
        ev.trace_source = "tool_trace"
        return {c.id: c.earned for c in score_auto_checkpoints(q, ev)}["C3"]

    old_style = ("SELECT mjd, cmag FROM smash_dr1.source "
                 "WHERE q3c_radial_query(ra, dec, 185.4311, -31.9953, 0.000278) "
                 "AND filter = 'g' AND cmag < 99 ORDER BY mjd")
    new_sql, _meta = B.build_variable_star_select(source_id="169.429960")
    new_sql += "\n  AND filter = 'g'"  # band leg of the check
    assert _c3(old_style) == 10.0
    assert _c3(new_sql) == 10.0


def test_dlb05_reference_sql_still_earns_full_c3():
    from dlb_dataset_v1 import get_question
    from run_datalabbench import Evidence, ToolCall
    ref_sql = get_question("DLB-05")["reference_sql"]
    ev = Evidence()
    ev.calls = [ToolCall(name="datalab_sql_query", arguments={"sql": ref_sql},
                         output='{"success": true}', ok=True, sql=ref_sql)]
    ev.sql_texts = [ref_sql]
    ev.trace_source = "tool_trace"
    assert _dlb05_c3_earned(ev) == 20.0


# ─────────────────────────────────────────────────────────────────────────────
# RE-B4 — "color image" honesty
# ─────────────────────────────────────────────────────────────────────────────
def test_ls_decam_hips_aliases_resolve_on_survey():
    from services.hips_images import COLOR_HIPS_ALIASES, resolve_survey
    for alias in ("ls", "legacy_surveys", "decals", "decam"):
        assert resolve_survey(alias) == "CDS/P/DESI-Legacy-Surveys/DR10/color"
        assert alias in COLOR_HIPS_ALIASES  # rejected as an RGB channel input
    for band in "griz":
        assert resolve_survey(f"ls_{band}") == f"CDS/P/DESI-Legacy-Surveys/DR10/{band}"
        assert resolve_survey(f"decam_{band}") == f"CDS/P/DESI-Legacy-Surveys/DR10/{band}"


def test_rgb_composite_rejects_color_alias_and_points_at_ls_bands():
    from services.hips_images import HipsImageService
    out = HipsImageService().rgb_composite(10.0, 41.0, ["decam", "ls_r", "ls_g"])
    assert out["success"] is False
    assert "ls_g/ls_r/ls_i/ls_z" in out["error"]


def test_hips_cutout_result_carries_source_service(monkeypatch, tmp_path):
    from services import plotting
    from services.hips_images import HipsImageService
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", str(tmp_path))
    service = HipsImageService()
    monkeypatch.setattr(service, "_fetch_png", lambda *a, **k: b"\x89PNG\r\n\x1a\nxxxx")
    out = service.cutout(10.68, 41.27, survey="optical")
    assert out["success"] is True
    assert out["source_service"] == "CDS hips2fits"
    assert out["provenance"]["service"] == "CDS hips2fits"


def test_datalab_sia_image_result_labeled_datalab():
    from services.datalab_image_service import DatalabImageService
    out = DatalabImageService._image_result(
        {"base64_png": "abc", "web_url": "/plots/x.png"},
        {"used_endpoint": "https://datalab/sia", "provenance": {"endpoint": "sia"}},
        bands=["g"],
    )
    assert out["source_service"] == "NOIRLab Astro Data Lab (SIA)"


def test_same_survey_color_completion_labeled_cds():
    from services.datalab_image_service import DatalabImageService

    class _FakeHips:
        def cutout(self, ra, dec, fov_deg=0.25, survey=None, width=512, title=None, detect_blank=False):
            return {"success": True, "image_base64": "abc", "path": "/plots/x.png",
                    "png_path": None, "blank": False, "provenance": {"params": {"hips": survey}}}

    service = DatalabImageService.__new__(DatalabImageService)
    service._hips_service = _FakeHips()
    out = service._same_survey_color_completion(
        {"success": False, "coverage_gap": True}, {"provenance": {}},
        ra=10.68, dec=41.27, fov_deg=0.1, catalog="ls_dr9", healthy=["z"],
    )
    assert out["success"] is True
    assert out["source_service"] == "CDS hips2fits"
    assert out["provenance"]["service"] == "CDS hips2fits"


def test_system_prompt_no_longer_instructs_cross_survey_substitution():
    from tests.integration.test_agent_archive_tools import _load_agent_module
    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    prompt = agent._build_system_prompt()
    # The old rule literally instructed the forbidden substitution.
    assert "hips_multiband_panel` for an optical color view" not in prompt
    assert "NEVER silently substitute another survey's imagery" in prompt
    # Honest-gap choices are offered instead, with explicit labeling.
    assert "explicitly labeled with that survey's name" in prompt
    # datalab_color_image joined the imagery routing lists.
    assert "hips_cutout / hips_multiband_panel / vlass_cutout / stamps / datalab_color_image" in prompt
    # The ls_i/decam_i single-band aliases resolve to the DR10 HiPS (which DOES
    # carry i); the prompt must not read as claiming they are DR9 products
    # (guard round task-3c99b37-89, reviewer question 3).
    assert "aliases serve the Legacy Surveys DR10 HiPS" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 2026-08 guard round (task-3c99b37-89) — CX-01..CX-13 regressions
# ─────────────────────────────────────────────────────────────────────────────
def test_q3c_small_side_cap_is_flagged_platform_budget():
    # CX-01: the inner pre-join CTE LIMIT is a platform budget exactly like the
    # outer cap — self-describing SQL + named platform_row_caps metadata.
    sql, meta = B.build_q3c_crossmatch(ra=229.022, dec=-0.112, radius_deg=1.0)
    inner = next(line for line in sql.splitlines() if line.strip().startswith("LIMIT 10000"))
    assert PLATFORM_COMMENT in inner
    assert meta["platform_row_caps"] == {"small_side": 10000}
    assert any("small-side" in w and "not a science cut" in w for w in meta["warnings"])
    # The flagged SQL still validates cleanly through the governor.
    validated = P.validate(sql, source="builder", meta=meta)
    assert validated.sql == sql
    assert validated.meta.get("platform_row_caps") == {"small_side": 10000}


def test_q3c_explicit_small_limit_is_users_choice_not_flagged():
    sql, meta = B.build_q3c_crossmatch(
        ra=229.022, dec=-0.112, radius_deg=1.0, small_limit=2000, limit=100
    )
    assert "LIMIT 2000" in sql and PLATFORM_COMMENT not in sql
    assert "platform_row_caps" not in meta and "platform_row_cap" not in meta
    assert meta["small_limit"] == 2000


def test_variable_tool_limits_are_flagged_defaults_not_schema_baked():
    # CX-02: the capability inputs default to None; the builders apply the
    # documented budgets (100 / 500) and FLAG them like the diagram budgets.
    from capabilities import datalab as dl
    assert dl.VariableCandidatesInput().limit is None
    assert dl.StarLightcurveInput().limit is None
    sql, meta = B.build_variability_rank("smash_dr1", ra=15.0, dec=-72.0, radius_deg=0.3, limit=None)
    assert f"LIMIT 100 {PLATFORM_COMMENT}" in sql and meta["platform_row_cap"] == 100
    sql2, meta2 = B.build_variable_star_select(source_id="abc123", limit=None)
    assert f"LIMIT 500 {PLATFORM_COMMENT}" in sql2 and meta2["platform_row_cap"] == 500
    # User-passed values stay theirs — no flag.
    sql3, meta3 = B.build_variability_rank("smash_dr1", ra=15.0, dec=-72.0, radius_deg=0.3, limit=75)
    assert "LIMIT 75" in sql3 and PLATFORM_COMMENT not in sql3 and "platform_row_cap" not in meta3
    # The tool schemas no longer advertise the defaults (a model echoing a
    # schema default would turn the platform budget into a "user" choice).
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    agent._register_tools()
    for name in ("datalab_variable_candidates", "datalab_star_lightcurve", "datalab_q3c_crossmatch"):
        props = agent.tool_registry.get_tool(name).parameters["properties"]
        assert "default" not in props["limit"], name
    q3c_props = agent.tool_registry.get_tool("datalab_q3c_crossmatch").parameters["properties"]
    assert "default" not in q3c_props["small_limit"]
    ccd_props = agent.tool_registry.get_tool("datalab_color_color_diagram").parameters["properties"]
    assert "default" not in ccd_props["split_threshold"]  # CX-03: 0.003 is not a universal default


def test_split_threshold_derives_from_column_family_not_des_constant():
    # CX-03: class_star family → 0.5 (its registered convention), NEVER 0.003.
    thr, warn = orch._resolve_split_threshold("nsc_dr2", "object", "class_star", None)
    assert thr == 0.5 and warn is None
    # spread_model family → the registered _POINT_SOURCE_CUTS scale (DES 0.003).
    thr2, warn2 = orch._resolve_split_threshold("des_dr1", "main", "spread_model_r", None)
    assert thr2 == pytest.approx(0.003) and warn2 is None
    # Explicit thresholds execute as given (never clamped) and run through the
    # morphology deviation check — the RE-B3 warning fires on the split path too.
    thr3, warn3 = orch._resolve_split_threshold("nsc_dr2", "object", "class_star", 0.003)
    assert thr3 == pytest.approx(0.003)
    assert warn3 is not None and "MORPHOLOGY THRESHOLD CHECK" in warn3
    thr4, warn4 = orch._resolve_split_threshold("des_dr1", "main", "spread_model_r", 0.5)
    assert thr4 == 0.5 and warn4 is not None and "MORPHOLOGY THRESHOLD CHECK" in warn4


class _FakeNSCClassStarClient(_FakeNSCClient):
    """NSC cone with a class_star column: 120 stars (0.98) + 80 galaxies (0.02)."""

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        from integrations.datalab_client import DatalabResult
        self.sql.append(sql or "")
        rng = np.random.default_rng(7)
        n = 200
        df = pd.DataFrame({
            "ra": rng.uniform(259.7, 260.4, n), "dec": rng.uniform(57.6, 58.2, n),
            "gmag": rng.uniform(17, 24, n), "rmag": rng.uniform(16.5, 23.5, n),
            "class_star": np.concatenate([np.full(120, 0.98), np.full(80, 0.02)]),
        })
        return DatalabResult.from_dataframe(df, {"catalog": "nsc_dr2", "table": "object", "query": sql})


def test_ccd_class_star_split_defaults_to_half_with_correct_direction():
    # CX-03: an explicit class_star split with NO threshold must split at 0.5
    # with stars = HIGH class_star — not at the DES 0.003 (which misclassified
    # nearly every source under the old s <= 0.003 rule).
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeNSCClassStarClient()
    out = orch.color_color_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4,
        x_bands=("g", "r"), y_bands=("g", "r"), split_col="class_star",
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    assert out["split_threshold"] == 0.5
    pops = {p["population"]: p["n"] for p in out["populations"]}
    assert pops == {"stars": 120, "galaxies": 80}
    # The derived convention is correct → no deviation warning.
    assert not any("MORPHOLOGY THRESHOLD CHECK" in w for w in out["warnings"])


def test_ccd_explicit_wrong_split_threshold_warns_loudly():
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeNSCClassStarClient()
    out = orch.color_color_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4,
        x_bands=("g", "r"), y_bands=("g", "r"), split_col="class_star", split_threshold=0.003,
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    assert out["split_threshold"] == pytest.approx(0.003)  # executed as given, never clamped
    assert any("MORPHOLOGY THRESHOLD CHECK" in w for w in out["warnings"])


def test_split_groups_spread_model_is_two_sided():
    # CX-04: the registered DES convention is |spread_model_r| <= 0.003 —
    # negative outliers below -0.003 are NOT stars.
    s = pd.Series([-0.01, -0.001, 0.0, 0.002, 0.01])
    finite = pd.Series([True] * 5)
    (label_s, stars), (label_g, gals) = orch._split_groups("spread_model_r", s, finite, 0.003)[0]
    assert label_s == "stars" and list(stars) == [False, True, True, True, False]
    assert label_g == "galaxies" and list(gals) == [True, False, False, False, True]


def test_split_groups_class_star_high_is_star():
    s = pd.Series([0.98, 0.6, 0.4, 0.02])
    finite = pd.Series([True] * 4)
    groups, note = orch._split_groups("class_star", s, finite, 0.5)
    (_, stars), (_, gals) = groups
    assert list(stars) == [True, True, False, False]
    assert list(gals) == [False, False, True, True]
    assert note is not None and "class_star" in note


class _RecordingStore:
    """Minimal result-store stand-in that records every stored meta."""

    def __init__(self):
        self.metas = []

    def put(self, dataframe, meta=None):
        self.metas.append(dict(meta or {}))
        return f"dlr_test{len(self.metas)}"


class _GridCellClient:
    """Returns `rows` density grid cells for every query; records SQL."""

    def __init__(self, rows):
        self.rows = rows
        self.sql = []

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        from integrations.datalab_client import DatalabResult
        self.sql.append(sql or "")
        df = pd.DataFrame([
            {"ra_bin": 10.0 + (i % 50) * 0.05, "dec_bin": (i // 50) * 0.05, "source_count": 5000 - i}
            for i in range(self.rows)
        ])
        return DatalabResult.from_dataframe(df, {"catalog": "nsc_dr2", "table": "object", "query": sql})


def test_tiled_density_aggregate_flags_platform_default_cap_always():
    # CX-07: with NO caller limit the per-tile cap (5000) is a platform choice —
    # SQL comment per tile, a disclosure warning, and platform_row_cap in the
    # final provenance even when NO tile truncates.
    client = _GridCellClient(10)
    store = _RecordingStore()
    out = orch.tiled_density_aggregate(
        "nsc_dr2", "object", mode="grid", step_deg=0.1, ra=10.0, dec=0.0,
        radius_deg=1.0, client=client, result_store=store,
    )
    assert out["success"] is True
    assert all(f"LIMIT 5000 {PLATFORM_COMMENT}" in sql for sql in client.sql)
    assert any("not a science cut" in w for w in out["warnings"])
    prov = store.metas[-1]["provenance"]
    assert prov["platform_row_cap"] == 5000
    assert "limit_truncated" not in prov  # nothing truncated — the cap is still disclosed

    # User-passed limit: their choice, no platform stamp (and no cap warning).
    client2 = _GridCellClient(10)
    store2 = _RecordingStore()
    out2 = orch.tiled_density_aggregate(
        "nsc_dr2", "object", mode="grid", step_deg=0.1, ra=10.0, dec=0.0,
        radius_deg=1.0, limit=25, client=client2, result_store=store2,
    )
    assert out2["success"] is True
    assert not any("row cap" in w for w in out2["warnings"])
    assert "platform_row_cap" not in store2.metas[-1]["provenance"]


class _OnePeakAnalysis:
    @staticmethod
    def _matched_filter_peaks(hist, xe, ye, sigma_small, sigma_large, threshold, max_peaks):
        return [{"ra": float(xe[0]), "dec": float(ye[0]), "significance": 9.0}]


def test_tiled_sky_scan_surfaces_per_tile_row_cap():
    # CX-08: a tile whose density query exactly filled its row cap (5000)
    # dropped its sparsest cells BEFORE peak-finding — the scan must say so in
    # its notes AND on the affected candidates.
    fp = {"ra_min": 10.0, "ra_max": 10.4, "dec_min": 0.0, "dec_max": 0.4}
    client = _GridCellClient(5000)  # exactly the per-tile platform cap
    out = orch.tiled_sky_scan(
        "nsc_dr2", "object", fp, tile_radius_deg=1.0, step_deg=0.05,
        client=client, result_store=_RecordingStore(), analysis=_OnePeakAnalysis(),
        max_tiles=4, candidate_budget=10, confirm=True,
    )
    assert out["success"] is True
    assert out["tiles_truncated"] == out["tiles_scanned"] >= 1
    assert any("per-tile row cap" in n and "LIMIT 5000" in n for n in out["notes"])
    assert out["candidates"], "the planted peak must survive ranking"
    for cand in out["candidates"]:
        assert any("row cap" in w for w in cand["warnings"])


def test_tiled_sky_scan_uncapped_tiles_carry_no_truncation_flags():
    fp = {"ra_min": 10.0, "ra_max": 10.4, "dec_min": 0.0, "dec_max": 0.4}
    out = orch.tiled_sky_scan(
        "nsc_dr2", "object", fp, tile_radius_deg=1.0, step_deg=0.05,
        client=_GridCellClient(40), result_store=_RecordingStore(),
        analysis=_OnePeakAnalysis(), max_tiles=4, candidate_budget=10, confirm=True,
    )
    assert out["tiles_truncated"] == 0
    assert not any("row cap" in n for n in out["notes"])
    assert all("warnings" not in c for c in out["candidates"])


class _NoopGridImageService:
    def __init__(self, grid=None):
        self.calls = []
        self.grid = grid if grid is not None else {"success": True, "panels": []}

    def cutout_grid(self, peaks, fov_deg, *, band="g", catalog=None, **kwargs):
        self.calls.append((list(peaks), fov_deg, band, catalog))
        return dict(self.grid)


def test_density_then_cutouts_candidate_budget_is_flagged():
    # CX-09: the derived budget max(top_n*4, 50) is platform-chosen — SQL
    # comment + platform_row_cap(+named candidate_budget) + a result note.
    client = _GridCellClient(60)
    store = _RecordingStore()
    img = _NoopGridImageService()
    out = orch.density_then_cutouts(
        "nsc_dr2", "object", 185.41, -31.98, 0.5, step_deg=0.05, top_n=3,
        client=client, result_store=store, image_service=img,
    )
    assert out["success"] is True
    assert f"LIMIT 50 {PLATFORM_COMMENT}" in client.sql[0]
    assert any("Candidate budget" in n and "not a science cut" in n for n in out["notes"])
    prov = store.metas[0]["provenance"]
    assert prov["platform_row_cap"] == 50
    assert prov["platform_row_caps"]["candidate_budget"] == 50


def test_density_vetting_nested_grid_timeout_is_loud_but_peaks_keep_success():
    # CX-11 (partial-contest): density peaks ARE real progress, so a fully
    # timed-out nested grid keeps success=True — but the timeout must be loud
    # (note) and machine-readable (cutout_grid_timeout).
    grid = {"success": False, "timeout": True, "panels": [{"timeout": True}], "error": "all panels timed out"}
    out = orch.density_then_cutouts(
        "nsc_dr2", "object", 185.41, -31.98, 0.5, step_deg=0.05, top_n=2,
        client=_GridCellClient(60), result_store=_RecordingStore(),
        image_service=_NoopGridImageService(grid=grid),
    )
    assert out["n_peaks"] >= 1
    assert out["success"] is True          # peaks are real progress
    assert out["cutout_grid_timeout"] is True
    assert "timeout" not in out            # NOT a full-tool timeout while peaks exist
    assert any("CUTOUT GRID TIMEOUT" in n for n in out["notes"])


def test_cutout_grid_mixed_timeouts_carry_panel_timeouts_and_warning():
    # CX-12: some-panels-timed-out grids stay success=True but must say the
    # grid is partial, machine-readably (panel_timeouts) and loudly (warning).
    from services.datalab_image_service import DatalabImageService
    from tests.unit.test_datalab_p1 import _MemoryPlottingService

    svc = DatalabImageService.__new__(DatalabImageService)
    svc.plotting_service = _MemoryPlottingService()
    calls = {"n": 0}

    def fake_search(ra, dec, fov, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("simulated FITS wall-clock timeout")
        return {"coverage_gap": True, "rows": [], "used_endpoint": "sia"}

    svc.search = fake_search
    out = svc.cutout_grid([{"ra": 10.0, "dec": 0.0}, {"ra": 10.2, "dec": 0.2}], 0.05, band="g")
    assert out["success"] is True
    assert out["panel_timeouts"] == 1
    assert any("timed out" in w and "PARTIAL" in w for w in out["warnings"])
    assert out["source_service"] == "NOIRLab Astro Data Lab (SIA)"  # CX-13
    assert out["panels"][0].get("timeout") is True
    assert "timeout" not in out["panels"][1]


def test_no_image_and_error_results_carry_source_service():
    # CX-13: gaps, download failures, and grids are user-visible image results
    # too — they must carry the same source_service label as successes.
    from services.datalab_image_service import DatalabImageService

    gap = DatalabImageService._gap_result(
        {"used_endpoint": "sia", "provenance": {}}, bands=["g"]
    )
    assert gap["source_service"] == "NOIRLab Astro Data Lab (SIA)"

    svc = DatalabImageService.__new__(DatalabImageService)
    svc.search = lambda *a, **k: {
        "coverage_gap": False, "rows": [{"obs_bandpass": "g"}],
        "used_endpoint": "sia", "provenance": {},
    }
    svc.candidates_by_band = lambda rows, bands: {"g": [{"row": rows[0]}]}
    svc._load_first_working = lambda cands, **k: (None, None, ["tile 500"])
    svc._bands_with_healthy_refs = lambda rows, exclude=None: []
    out = svc.cutout(10.0, 0.0, 0.05, band="g")
    assert out["success"] is False
    assert out["source_service"] == "NOIRLab Astro Data Lab (SIA)"


def test_tiled_extra_warnings_reach_the_persisted_store_meta():
    """CX-06 verify: pre-tiling warnings (quality note, morphology deviation)
    must land in the PERSISTED result's meta, not only the returned dict —
    chained consumers read the stored result_id."""
    client = _GridCellClient(10)
    store = _RecordingStore()
    out = orch.tiled_density_aggregate(
        "nsc_dr2", "object", mode="grid", step_deg=0.1, ra=10.0, dec=0.0,
        radius_deg=1.0, client=client, result_store=store,
        extra_warnings=["MORPHOLOGY WARNING: test-sentinel", None, ""],
    )
    assert any("test-sentinel" in w for w in out["warnings"])
    stored = store.metas[-1]["warnings"]
    assert any("test-sentinel" in w for w in stored), (
        "extra_warnings must be persisted with the stored result"
    )


def test_candidate_budget_disclosure_matches_emitted_limit():
    """CX-09 verify regression: for huge top_n the builder clamps the SQL to
    MAX_ROW_LIMIT — the disclosed platform_row_cap and note must say the SAME
    clamped number, never the unclamped top_n*4."""
    from services.datalab_query_builders import MAX_ROW_LIMIT

    class _StubGridService:
        def cutout_grid(self, peaks, fov_deg, **kwargs):
            return {"success": True, "panels": []}

    client = _GridCellClient(10)
    store = _RecordingStore()
    out = orch.density_then_cutouts(
        "nsc_dr2", "object", ra=10.0, dec=0.0, radius_deg=0.5, step_deg=0.05,
        top_n=2000, fov_deg=0.05, client=client, result_store=store,
        image_service=_StubGridService(),
    )
    sql = client.sql[-1]
    assert f"LIMIT {MAX_ROW_LIMIT}" in sql
    assert f"LIMIT {MAX_ROW_LIMIT * 2}" not in sql
    prov_meta = store.metas[-1]
    cap = prov_meta.get("provenance", {}).get("platform_row_cap") or prov_meta.get("platform_row_cap")
    assert int(cap) == MAX_ROW_LIMIT, f"disclosed cap {cap} != emitted LIMIT {MAX_ROW_LIMIT}"
    assert any(f"LIMIT {MAX_ROW_LIMIT}" in n for n in out.get("notes", []) + out.get("warnings", []))
