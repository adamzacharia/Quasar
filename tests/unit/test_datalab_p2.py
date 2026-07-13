from __future__ import annotations

import re
import threading
import time

import numpy as np
import pandas as pd
import pytest

from integrations.datalab_client import DatalabResult


# ── fakes ──────────────────────────────────────────────────────────────────────
pytestmark = pytest.mark.slow


class _FakeTiledClient:
    """Returns a low-density grid per tile, with a planted dense cell in the hot tile."""

    def __init__(self, hot=(150.0, 2.0), hot_radius=1.1):
        self.hot = hot
        self.hot_radius = hot_radius
        self.calls = 0

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        self.calls += 1
        m = re.search(r"q3c_radial_query\([^,]+,\s*[^,]+,\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)", sql or "")
        ra_c, dec_c = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
        rows = []
        for r in np.round(np.linspace(ra_c - 0.4, ra_c + 0.4, 5), 3):
            for d in np.round(np.linspace(dec_c - 0.4, dec_c + 0.4, 5), 3):
                rows.append({"ra_bin": float(r), "dec_bin": float(d), "source_count": 2})
        if abs(ra_c - self.hot[0]) <= self.hot_radius and abs(dec_c - self.hot[1]) <= self.hot_radius:
            rows.append({"ra_bin": float(self.hot[0]), "dec_bin": float(self.hot[1]), "source_count": 500})
        return DatalabResult.from_dataframe(pd.DataFrame(rows), {"catalog": "x", "table": "y", "query": sql})


class _FakeAnalysis:
    """Stand-in for datalab_analysis: returns a peak only for a clearly dense tile."""

    @staticmethod
    def _matched_filter_peaks(hist, xe, ye, ss, sl, threshold, max_peaks):
        if hist.size == 0 or float(np.nanmax(hist)) < 50:
            return []
        i, j = np.unravel_index(int(np.nanargmax(hist)), hist.shape)
        return [{"ra": float((xe[i] + xe[i + 1]) / 2.0), "dec": float((ye[j] + ye[j + 1]) / 2.0),
                 "significance": float(np.nanmax(hist))}]


class _FakeDensityClient:
    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        rows = [{"ra_bin": 185.0 + i * 0.05, "dec_bin": -32.0, "source_count": 100 - i} for i in range(10)]
        return DatalabResult.from_dataframe(pd.DataFrame(rows), {"catalog": "nsc_dr2", "table": "object", "query": sql})


class _FakeImageService:
    def __init__(self):
        self.calls = []

    def cutout_grid(self, peaks, fov_deg, *, band="g", catalog=None, **kwargs):
        peaks = list(peaks)
        self.calls.append((peaks, fov_deg, band))
        return {"success": True, "panels": [{"ra": p["ra"], "dec": p["dec"]} for p in peaks]}


# ── tests ──────────────────────────────────────────────────────────────────────
def test_build_catalog_predicates_and_density_with_cuts():
    from services import datalab_query_builders as B
    preds = B.build_catalog_predicates(
        "nsc_dr2", "object",
        color_cut={"bands": ["gmag", "rmag"], "min": -0.5, "max": 0.5},
        value_cuts=[{"column": "gmag", "op": ">", "value": 19.5}],
        morphology={"column": "class_star", "op": ">", "value": 0.5},
    )
    assert any("gmag - rmag" in p for p in preds)
    sql, meta = B.build_density_aggregate("nsc_dr2", "object", mode="grid", ra=10, dec=0, radius_deg=1.0, predicates=preds)
    assert "q3c_radial_query" in sql and "class_star" in sql and "GROUP BY ra_bin, dec_bin" in sql
    # Unknown-but-safe column identifiers are PERMITTED now (the seed registry is not exhaustive;
    # Data Lab validates them server-side). This is the brittleness fix.
    preds2 = B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "some_unlisted_col", "op": "<", "value": 1}])
    assert any("some_unlisted_col" in p for p in preds2)
    with pytest.raises(ValueError):  # unsafe identifier (injection) still rejected
        B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "x) OR 1=1; --", "op": "<", "value": 1}])
    with pytest.raises(ValueError):  # bad operator still rejected
        B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "gmag", "op": "DROP", "value": 1}])


def test_value_cut_operator_aliases_and_string_literals():
    from services import datalab_query_builders as B
    # '==' (Python/JS equality) coerces to SQL '='; '<>' coerces to '!='.
    preds = B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "gmag", "op": "==", "value": 20}])
    assert preds == ["gmag = 20"]
    preds = B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "gmag", "op": "<>", "value": 20}])
    assert preds == ["(gmag != 20 AND gmag < 'Infinity'::float8)"]
    # String value on '=' becomes a quoted, escaped SQL literal (the P13 class='GALAXY' case).
    preds = B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "type", "op": "=", "value": "GALAXY"}])
    assert preds == ["type = 'GALAXY'"]
    # A value the model already quoted is not double-quoted.
    preds = B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "type", "op": "=", "value": "'STAR'"}])
    assert preds == ["type = 'STAR'"]
    # Numeric strings stay numeric (no quoting).
    preds = B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "fieldid", "op": "=", "value": "169"}])
    assert preds == ["fieldid = 169"]
    # Single quotes in a string value are escaped, not passed through raw (injection guard).
    preds = B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "type", "op": "=", "value": "O'Brien"}])
    assert preds == ["type = 'O''Brien'"]
    # A non-numeric string is rejected for inequality operators (only =/!= take strings).
    with pytest.raises(ValueError):
        B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "gmag", "op": "<", "value": "GALAXY"}])


def test_nan_guard_on_lower_bound_cuts():
    """Postgres orders NaN above every real, and Data Lab stores missing floats
    as NaN — bare >/>=/!= cuts admit every missing-value row (live P6: 1,960
    rows vs ESA's 8 for identical predicates). Builders must emit the guard."""
    from services import datalab_query_builders as B
    preds = B.build_catalog_predicates(
        "gaia_dr3", "gaia_source",
        value_cuts=[
            {"column": "parallax_over_error", "op": ">", "value": 5},
            {"column": "pmra", "op": ">=", "value": 150},
            {"column": "ruwe", "op": "!=", "value": 1},
            {"column": "phot_g_mean_mag", "op": "<", "value": 21},
            {"column": "pmdec", "op": "=", "value": 0},
        ],
    )
    assert preds[0] == "(parallax_over_error > 5 AND parallax_over_error < 'Infinity'::float8)"
    assert preds[1] == "(pmra >= 150 AND pmra < 'Infinity'::float8)"
    assert preds[2] == "(ruwe != 1 AND ruwe < 'Infinity'::float8)"
    # <, = are NaN-safe on their own: no guard.
    assert preds[3] == "phot_g_mean_mag < 21"
    assert preds[4] == "pmdec = 0"
    # String cuts never take the guard (NaN can't leak through a text compare).
    preds = B.build_catalog_predicates("nsc_dr2", "object", value_cuts=[{"column": "type", "op": "!=", "value": "GALAXY"}])
    assert preds == ["type != 'GALAXY'"]
    # A min-only color cut grows a finiteness leg; a two-sided one is already safe.
    preds = B.build_catalog_predicates("nsc_dr2", "object", color_cut={"bands": ["gmag", "rmag"], "min": 0.7})
    assert preds == ["((gmag - rmag) >= 0.7 AND (gmag - rmag) < 'Infinity'::float8)"]
    preds = B.build_catalog_predicates("nsc_dr2", "object", color_cut={"bands": ["gmag", "rmag"], "min": -0.5, "max": 0.5})
    assert preds == ["(gmag - rmag) >= -0.5", "(gmag - rmag) <= 0.5"]
    # Morphology op cuts take the guard too (class_star > 0.5 on NaN rows).
    preds = B.build_catalog_predicates("nsc_dr2", "object", morphology={"column": "class_star", "op": ">", "value": 0.5})
    assert preds == ["(class_star > 0.5 AND class_star < 'Infinity'::float8)"]


def test_client_wall_clock_cuts_slowly_streaming_response(monkeypatch):
    """A scalar requests timeout is per-socket-read: a server trickling bytes
    every <60s held the '60s sync window' open ~8 minutes (live P8). The client
    must cut the stream at a hard wall-clock deadline with a 'timed out' error
    (that substring triggers the async fallback and the tiling path)."""
    import time as _t
    from integrations import datalab_client as dc

    class _TricklingResponse:
        status_code = 200
        encoding = "utf-8"

        def iter_content(self, chunk_size=65536):
            while True:
                _t.sleep(0.2)
                yield b"x" * 10

        def close(self):
            pass

    monkeypatch.setenv("DATALAB_SYNC_WALL_SECONDS", "1")
    monkeypatch.setattr(dc.requests, "get", lambda *a, **k: _TricklingResponse())
    client = dc.DatalabClient(token="anonymous.0.0.anon_access")
    t0 = _t.monotonic()
    with pytest.raises(dc.DatalabClientError, match="timed out"):
        client._get("query", {"sql": "SELECT 1"})
    assert _t.monotonic() - t0 < 5.0  # cut at ~1s wall, not per-read forever


def test_client_status_treats_error_state_as_state(monkeypatch):
    """/status returns the literal state string; 'ERROR' is a valid job STATE,
    which the transport-level ERROR-prefix check misread as a failure (live
    P15/P14: 'Data Lab /status error: ERROR' with no reason)."""
    from integrations import datalab_client as dc
    client = dc.DatalabClient(token="tok")
    monkeypatch.setattr(
        dc.DatalabClient, "_get",
        lambda self, endpoint, params, **kw: (_ for _ in ()).throw(
            dc.DatalabClientError("Data Lab /status error: ERROR")
        ),
    )
    assert client.status("job123") == "ERROR"
    # error() extracts the reason text from the /error body.
    monkeypatch.setattr(
        dc.DatalabClient, "_get",
        lambda self, endpoint, params, **kw: (_ for _ in ()).throw(
            dc.DatalabClientError("Data Lab /error error: ERROR relation too large")
        ),
    )
    assert client.error("job123") == "ERROR relation too large"


def test_plot_sky_map_numeric_color_by_uses_colorbar(monkeypatch):
    """Live P9: numeric color_by (big_gmag over 5,000 rows) went through the
    categorical branch — one legend entry per unique value — and produced a
    1585x236,838 px PNG. Numeric columns must render as c=values + colorbar;
    strings keep the legend, capped so it cannot explode either."""
    from services.plotting import PlottingService

    svc = PlottingService()
    captured = {}

    def _capture(self, fig, filename):
        import matplotlib.pyplot as plt
        captured["fig"] = fig
        plt.close(fig)
        return {"success": True, "filename": filename}

    monkeypatch.setattr(PlottingService, "_save_and_encode", _capture)

    # Numeric: 400 unique mags -> colorbar (a second axes), no legend.
    records = [
        {"ra": 229.0 + i * 1e-3, "dec": -0.1 + i * 1e-3, "big_gmag": 15 + (i % 400) / 40.0}
        for i in range(500)
    ]
    out = svc.plot_sky_map(records, ra_col="ra", dec_col="dec", color_by="big_gmag")
    assert out["success"], out
    fig = captured["fig"]
    assert len(fig.axes) == 2  # main + colorbar
    assert fig.axes[0].get_legend() is None

    # Few string categories: classic legend, single axes.
    records = [
        {"ra": 1.0 + i * 0.01, "dec": 2.0, "band": "gri"[i % 3]} for i in range(30)
    ]
    out = svc.plot_sky_map(records, ra_col="ra", dec_col="dec", color_by="band")
    assert out["success"], out
    fig = captured["fig"]
    assert len(fig.axes) == 1
    legend = fig.axes[0].get_legend()
    assert legend is not None and len(legend.get_texts()) == 3

    # Many string categories: legend capped at 20 + one "other" bucket.
    records = [
        {"ra": 1.0 + i * 0.01, "dec": 2.0, "field": f"F{i % 50:02d}"} for i in range(200)
    ]
    out = svc.plot_sky_map(records, ra_col="ra", dec_col="dec", color_by="field")
    assert out["success"], out
    legend = captured["fig"].axes[0].get_legend()
    assert legend is not None and len(legend.get_texts()) <= 21


def test_async_submit_strips_materialized_sync_keeps_it(monkeypatch):
    """The async query manager's JSQLParser rejects `AS MATERIALIZED`
    ("Encountered MATERIALIZED", live P9 2026-07-13) while the sync endpoint
    accepts it — so async submissions must drop the keyword and sync queries
    must keep it (it is the planner fence that makes crossmatch CTEs fast)."""
    from integrations import datalab_client as dc

    sql = (
        "WITH g AS MATERIALIZED (SELECT ra, dec FROM gaia_dr3.gaia_source "
        "WHERE label = 'AS MATERIALIZED (') SELECT * FROM g LIMIT 5"
    )
    seen = {}

    def _capture(self, endpoint, params, **kw):
        seen["params"] = dict(params)
        return "jobid123" if params.get("async") == "True" else "ra,dec\n1.0,2.0\n"

    monkeypatch.setattr(dc.DatalabClient, "_get", _capture)
    client = dc.DatalabClient(token="tok")

    assert client.submit(sql=sql) == "jobid123"
    sent = seen["params"]["sql"]
    assert "MATERIALIZED (SELECT" not in sent and sent.startswith("WITH g AS (SELECT")
    # The quoted literal survives verbatim.
    assert "label = 'AS MATERIALIZED ('" in sent

    client.query(sql=sql)
    assert seen["params"]["sql"] == sql  # sync path untouched

    # NOT MATERIALIZED and lowercase forms are rejected by the same parser.
    strip = dc.DatalabClient.strip_materialized_for_async
    assert strip("WITH g AS NOT MATERIALIZED (SELECT 1) SELECT * FROM g") == (
        "WITH g AS (SELECT 1) SELECT * FROM g"
    )
    assert strip("with g as materialized (select 1) select * from g") == (
        "with g AS (select 1) select * from g"
    )
    assert strip("SELECT 1") == "SELECT 1"


def test_default_quality_cuts_merge_and_override():
    """Live P11: DESI LRG counts without zwarn/survey/main_primary were 11-27%
    inflated per z-bin. Defaults apply unless the caller cuts the same column."""
    from services import datalab_registry as reg
    merged, note = reg.merge_default_quality_cuts("desi_dr1", "zpix", None)
    cols = [c["column"] for c in merged]
    assert cols == ["zwarn", "survey", "main_primary"]
    assert "zwarn" in note and "Override" in note
    # A caller cut on zwarn suppresses THAT default only.
    merged2, note2 = reg.merge_default_quality_cuts(
        "desi_dr1", "zpix", [{"column": "zwarn", "op": "<", "value": 5}]
    )
    cols2 = [c["column"] for c in merged2]
    assert cols2 == ["zwarn", "survey", "main_primary"] and merged2[0]["value"] == 5
    assert "zwarn" not in note2
    # Catalogs with no registered defaults pass through untouched.
    merged3, note3 = reg.merge_default_quality_cuts("gaia_dr3", "gaia_source", None)
    assert merged3 == [] and note3 is None
    # The DESI defaults render into valid SQL (boolean via string literal).
    from services import datalab_query_builders as B
    preds = B.build_catalog_predicates("desi_dr1", "zpix", value_cuts=merged)
    assert "zwarn = 0" in preds[0]
    assert preds[1] == "survey = 'main'"
    assert preds[2] == "main_primary = 'true'"


def test_policy_injects_default_quality_cuts_into_expert_sql():
    """Live P11: the expert-SQL DESI z-histogram applied zwarn=0 but omitted
    survey='main' AND main_primary, inflating every low-z bin 11-27% with
    sv-survey duplicates. Registry defaults now inject at the policy layer for
    quote-free single-table expert SQL (mirrors the NaN auto-rewrite)."""
    from services import datalab_sql_policy as policy

    # P11 histogram shape: zwarn cut present, survey/main_primary missing.
    sql = (
        "SELECT width_bucket(z, 0.0, 1.6, 32) AS z_bin, COUNT(*) AS n "
        "FROM desi_dr1.zpix WHERE zwarn = 0 GROUP BY z_bin ORDER BY z_bin"
    )
    v = policy.validate(sql, source="expert")
    assert "survey = 'main'" in v.sql and "main_primary = 'true'" in v.sql
    assert "AND (zwarn = 0)" in v.sql  # original condition parenthesized
    assert v.sql.count("zwarn") == 1   # mentioned column is never overridden
    assert any("Auto-applied desi_dr1.zpix" in w for w in v.warnings)

    # No WHERE at all -> one is inserted before GROUP BY with all three cuts.
    v2 = policy.validate(
        "SELECT spectype, COUNT(*) AS n FROM desi_dr1.zpix GROUP BY spectype",
        source="expert",
    )
    assert "WHERE zwarn = 0 AND survey = 'main' AND main_primary = 'true'" in v2.sql
    assert v2.sql.index("WHERE") < v2.sql.index("GROUP BY")

    # Plain string literals are safe (the scrub is offset-preserving): a model
    # following the class-cut rule writes spectype = 'GALAXY' and must still
    # get survey/main_primary injected (live P11: the canonical selection).
    v3 = policy.validate(
        "SELECT spectype, COUNT(*) AS n FROM desi_dr1.zpix "
        "WHERE spectype = 'GALAXY' GROUP BY spectype",
        source="expert",
    )
    assert "survey = 'main'" in v3.sql and "AND (spectype = 'GALAXY')" in v3.sql

    # Escaped quotes and comments make offsets unreliable -> no injection.
    v3b = policy.validate(
        "SELECT spectype, COUNT(*) AS n FROM desi_dr1.zpix "
        "WHERE spectype = 'GAL''AXY' GROUP BY spectype",
        source="expert",
    )
    assert "survey" not in v3b.sql
    v3c = policy.validate(
        "SELECT spectype, COUNT(*) AS n FROM desi_dr1.zpix -- note\n"
        "GROUP BY spectype",
        source="expert",
    )
    assert "survey" not in v3c.sql

    # Builder SQL merges defaults upstream — the policy leaves it alone.
    v4 = policy.validate(
        "SELECT spectype, COUNT(*) AS n FROM desi_dr1.zpix GROUP BY spectype",
        source="builder",
    )
    assert "survey" not in v4.sql

    # NaN rewrite + quality injection COMPOSE: the 'Infinity' literals the NaN
    # guard introduces must not disqualify the injection that follows.
    v5 = policy.validate(
        "SELECT width_bucket(z, 0.4, 0.8, 4) AS z_bin, COUNT(*) AS n "
        "FROM desi_dr1.zpix WHERE (desi_target & 1) > 0 AND z >= 0.4 AND zwarn = 0 "
        "GROUP BY z_bin ORDER BY z_bin",
        source="expert",
    )
    assert "z >= 0.4 AND z < 'Infinity'::float8" in v5.sql
    assert "survey = 'main'" in v5.sql and "main_primary = 'true'" in v5.sql


def test_known_mw_object_veto():
    from services import datalab_registry as reg
    # Draco: live P15's "discovered" candidate #1 sat 0.06 deg from its center.
    hit = reg.match_known_mw_object(260.05, 57.92)
    assert hit and hit["name"] == "Draco dSph" and hit["separation_deg"] < 0.1
    # Hydra II and Pal 5 (the campaign's other recurring fields).
    assert reg.match_known_mw_object(185.4311, -31.9953)["name"] == "Hydra II"
    assert reg.match_known_mw_object(229.022, -0.112)["name"] == "Palomar 5"
    # Empty sky is empty.
    assert reg.match_known_mw_object(150.0, -60.0) is None


def test_rank_candidates_min_separation():
    from services import datalab_orchestration as orch
    cands = [
        {"ra": 260.05, "dec": 57.92, "significance": 10.0},
        {"ra": 260.07, "dec": 57.93, "significance": 9.0},   # neighbor of #1
        {"ra": 261.50, "dec": 57.00, "significance": 8.0},
        {"ra": 260.06, "dec": 57.91, "significance": 7.0},   # another neighbor
    ]
    kept = orch.rank_candidates(cands, limit=5, min_separation_deg=0.15)
    assert [c["significance"] for c in kept] == [10.0, 8.0]
    # Without a separation, plain top-N behavior is unchanged.
    assert len(orch.rank_candidates(cands, limit=3)) == 3


def test_confirm_cone_area_gate():
    from services import datalab_orchestration as orch
    small = orch.confirm_cone_area(80.0, -60.0, 2.0)
    assert small["needs_confirmation"] is False
    wide = orch.confirm_cone_area(80.0, -60.0, 30.0)
    assert wide["needs_confirmation"] is True
    assert wide["area_deg2"] == pytest.approx(2827, rel=0.01)
    assert "confirm=true" in wide["message"]


def test_gaia_mag_template_registered():
    """No gaia entry in _MAG_TEMPLATES meant the '{band}mag' fallback emitted
    'gmag' — guaranteed HTTP 400 on every one-shot Gaia diagram (live P6)."""
    from services import datalab_registry as reg
    assert reg.mag_column("gaia_dr3", "gaia_source", "g") == "phot_g_mean_mag"
    assert reg.mag_column("gaia_dr3", "gaia_source", "bp") == "phot_bp_mean_mag"
    assert reg.mag_column("gaia_dr3", "gaia_source", "rp") == "phot_rp_mean_mag"


class _FakeExprClient:
    def __init__(self):
        self.sql = []

    def query(self, *, sql=None, adql=None, fmt="pandas", **kw):
        import numpy as np
        import pandas as pd
        from integrations.datalab_client import DatalabResult
        self.sql.append(sql or "")
        rng = np.random.default_rng(7)
        n = 60
        df = pd.DataFrame({
            "ra": rng.uniform(59, 61, n), "dec": rng.uniform(-51, -49, n),
            "bp_rp": rng.uniform(-0.3, 1.0, n),
            "phot_g_mean_mag": rng.uniform(15, 20, n),
            "parallax": rng.uniform(5, 60, n),
        })
        return DatalabResult.from_dataframe(df, {"catalog": "gaia_dr3", "table": "gaia_source", "query": sql})


def test_cmd_expr_mode_builds_derived_axes(tmp_path):
    """x_expr/y_expr used to be silently IGNORED by the one-shot diagram tools
    (extra='ignore'), falling back to nonexistent gmag columns (live P6)."""
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    client = _FakeExprClient()
    store = DatalabResultStore(enable_disk_cache=False)
    out = orch.color_magnitude_diagram(
        "gaia_dr3", "gaia_source", 60.0, -50.0, 1.0,
        x_expr="bp_rp", y_expr="phot_g_mean_mag + 5*log10(parallax) - 10",
        value_cuts=[{"column": "parallax_over_error", "op": ">", "value": 5}],
        client=client, result_store=store,
    )
    assert out["success"] is True
    sql = client.sql[0]
    # SELECT list is derived from the expressions' identifiers, not band mags.
    assert "bp_rp" in sql and "phot_g_mean_mag" in sql and "parallax" in sql
    assert "gmag" not in sql
    # Referenced columns get server-side finiteness guards so NaN rows can't
    # eat the LIMIT budget; the value_cut is NaN-guarded by the builder.
    assert "bp_rp < 'Infinity'::float8" in sql
    assert "(parallax_over_error > 5 AND parallax_over_error < 'Infinity'::float8)" in sql
    assert out["x"] == "bp_rp" and out["points"] == 60
    spec = out["plotly_spec"]
    assert spec["layout"]["yaxis"]["autorange"] == "reversed"  # CMD invert_y
    # Half-specified exprs are a loud error, not a silent band fallback.
    with pytest.raises(ValueError, match="BOTH x_expr and y_expr"):
        orch.color_magnitude_diagram(
            "gaia_dr3", "gaia_source", 60.0, -50.0, 1.0,
            x_expr="bp_rp", client=client, result_store=store,
        )


def test_infinity_value_cuts_render_as_quoted_literal():
    """Models add explicit finiteness guards after the NaN prompt rule; bare
    float formatting rendered them as `inf` (live P9: Data Lab read it as a
    column name — "Unknown column(s) 'inf'")."""
    from services import datalab_query_builders as B
    preds = B.build_catalog_predicates(
        "gaia_dr3", "gaia_source",
        value_cuts=[
            {"column": "parallax", "op": "<", "value": "Infinity"},
            {"column": "pmra", "op": ">", "value": "-Infinity"},
        ],
    )
    assert preds[0] == "parallax < 'Infinity'::float8"
    # A > -Infinity cut is NaN-unsafe, so it also grows the guard leg.
    assert preds[1] == "(pmra > '-Infinity'::float8 AND pmra < 'Infinity'::float8)"
    with pytest.raises(ValueError, match="NaN"):
        B.build_catalog_predicates(
            "gaia_dr3", "gaia_source",
            value_cuts=[{"column": "parallax", "op": "<", "value": float("nan")}],
        )


def test_crossmatch_cte_orders_by_distance():
    """The CTE row cap must keep the cone CENTER: a storage-order LIMIT inside
    the MATERIALIZED subquery biased the join input before any outer ORDER BY
    could help (live P9 post-fix: map still showed the far-west corner)."""
    from services.datalab_query_builders import build_q3c_crossmatch
    sql, _meta = build_q3c_crossmatch(ra=229.022, dec=-0.112, radius_deg=5.0)
    # The indented ORDER BY ... LIMIT pair sits inside the CTE body.
    assert "    ORDER BY q3c_dist(ra, dec, 229.022, -0.112)\n    LIMIT" in sql
    # The outer join output is ALSO nearest-first before its own LIMIT.
    assert "\nORDER BY q3c_dist(g.ra, g.dec, 229.022, -0.112)\nLIMIT" in sql


def test_limit_truncation_warning_helper():
    from services import datalab_sql_policy as P
    assert P.limit_truncation_warning(500, 500) is not None
    assert P.limit_truncation_warning(5000, 500) is not None  # >= cap counts too
    assert P.limit_truncation_warning(499, 500) is None
    assert P.limit_truncation_warning(500, None) is None
    assert P.limit_truncation_warning(500, "not-a-number") is None
    text = P.limit_truncation_warning(500, 500)
    assert "row cap" in text and "density_aggregate" in text


def test_policy_records_row_limit_for_expert_sql():
    from services import datalab_sql_policy as P
    # Explicit LIMIT is recorded.
    out = P.validate(
        "SELECT ra, dec FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 10, 10, 0.1) LIMIT 250",
        source="expert",
    )
    assert out.meta.get("row_limit") == 250
    # An injected cap is recorded too.
    out2 = P.validate(
        "SELECT ra, dec FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 10, 10, 0.1)",
        source="expert",
    )
    assert out2.meta.get("row_limit") == P.DEFAULT_ROW_CAP
    assert f"LIMIT {P.DEFAULT_ROW_CAP}" in out2.sql
    # Builder meta's own row_limit is preserved (setdefault semantics).
    out3 = P.validate(
        "SELECT ra, dec FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 10, 10, 0.1) LIMIT 100",
        source="builder",
        meta={"row_limit": 100},
    )
    assert out3.meta.get("row_limit") == 100


def test_density_aggregate_accepts_field_bound_predicates():
    """SMASH `fieldid = 169` is the canonical bound for a whole-field density
    map (live P7: a guessed cone center cut Hydra II out of the map)."""
    from services import datalab_query_builders as B
    from services import datalab_sql_policy as P
    preds = B.build_catalog_predicates(
        "smash_dr1", "object",
        color_cut={"bands": ["gmag", "rmag"], "min": -0.5, "max": 0.5},
        value_cuts=[{"column": "fieldid", "op": "=", "value": 169}],
    )
    sql, meta = B.build_density_aggregate(
        "smash_dr1", "object", mode="grid", step_deg=0.01,
        predicates=preds, field_bound=True,
    )
    assert "q3c_radial_query" not in sql and "fieldid = 169" in sql
    assert "GROUP BY ra_bin, dec_bin" in sql
    assert any("indexed-equality" in w for w in meta["warnings"])
    # The governor accepts it (aggregate + indexed equality bound).
    out = P.validate(sql, source="builder", meta=meta)
    assert out.sql == sql
    # Without field_bound (and no cone / all_sky) the builder still refuses.
    with pytest.raises(ValueError, match="density aggregate requires"):
        B.build_density_aggregate("smash_dr1", "object", mode="grid", predicates=preds)


def test_analysis_truncation_warnings_from_provenance():
    from services.datalab_analysis import _truncation_warnings
    assert _truncation_warnings({"limit_truncated": True, "row_limit": 500})
    assert _truncation_warnings({"rowcount": 500}) == []
    assert _truncation_warnings(None) == []


def test_registry_string_id_columns():
    from services import datalab_registry as reg
    assert reg.string_id_columns("smash_dr1", "source") == ["id"]
    assert reg.string_id_columns("smash_dr1", "object") == ["id"]
    assert reg.string_id_columns("gaia_dr3", "gaia_source") == []  # true int64 ids
    union = reg.string_id_columns(None, None)
    assert "id" in union and "designation" in union and "unwise_objid" in union


def test_string_id_columns_preserved_in_csv_parsing():
    """SMASH ids like '169.429960' are strings with a significant trailing zero;
    pandas float inference corrupted them and the P14 light-curve analysis ran
    on a different, 24th-mag star. dtype=str must be forced at parse time."""
    from integrations.datalab_client import DatalabClient
    csv = "id,mjd,cmag\n169.429960,56371.1,21.5\n169.429960,56372.2,21.4\n"
    query = "SELECT id, mjd, cmag FROM smash_dr1.source WHERE fieldid = 169"
    frame = DatalabClient._csv_to_dataframe(csv, DatalabClient._string_dtypes(query))
    assert list(frame["id"]) == ["169.429960", "169.429960"]
    assert str(frame["mjd"].dtype).startswith("float")
    # Unknown-table context (async job results): global id-name fallback guards.
    frame2 = DatalabClient._csv_to_dataframe(csv, DatalabClient._string_dtypes(None))
    assert list(frame2["id"])[0] == "169.429960"
    # Known table with no string ids: no overrides at all.
    q3 = "SELECT source_id FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 1, 1, 0.1)"
    assert DatalabClient._string_dtypes(q3) == {}
    # The model-facing preview keeps the string exactly.
    from capabilities.datalab import _fit_rows
    rows, _ = _fit_rows(frame, 5)
    assert rows[0]["id"] == "169.429960"


def test_policy_autorewrites_nan_unsafe_expert_cuts():
    """Quote/comment-free expert SQL gets NaN guards injected automatically
    (advisory warnings were live-proven insufficient: gpt-oss ignored them and
    shipped a 97%-NaN selection on the P6 re-test)."""
    from services import datalab_sql_policy as P
    sql = (
        "SELECT source_id FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 60, -50, 1.0) "
        "AND parallax_over_error > 5 AND pmra > 150 LIMIT 100"
    )
    out = P.validate(sql, source="expert")
    assert "(parallax_over_error > 5 AND parallax_over_error < 'Infinity'::float8)" in out.sql
    assert "(pmra > 150 AND pmra < 'Infinity'::float8)" in out.sql
    assert any("Auto-added NaN" in w for w in out.warnings)
    # q3c args and LIMIT are untouched.
    assert "q3c_radial_query(ra, dec, 60, -50, 1.0)" in out.sql and "LIMIT 100" in out.sql
    # Alias-qualified columns keep their qualifier.
    sql_alias = (
        "SELECT g.source_id FROM gaia_dr3.gaia_source g "
        "WHERE q3c_radial_query(g.ra, g.dec, 60, -50, 1.0) AND g.pmra > 150 LIMIT 100"
    )
    out_alias = P.validate(sql_alias, source="expert")
    assert "(g.pmra > 150 AND g.pmra < 'Infinity'::float8)" in out_alias.sql
    # Guarded / two-sided cuts are left alone (and don't warn).
    sql2 = (
        "SELECT source_id FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 60, -50, 1.0) "
        "AND parallax_over_error > 5 AND parallax_over_error < 'Infinity' "
        "AND pmra BETWEEN 150 AND 10000 LIMIT 100"
    )
    out2 = P.validate(sql2, source="expert")
    assert not any("NaN" in w for w in out2.warnings)
    assert "::float8" not in out2.sql
    # SQL containing string literals is NOT rewritten (regex can't safely touch
    # quoted text) — it falls back to the advisory warning.
    sql3 = (
        "SELECT targetid FROM desi_dr1.zpix "
        "WHERE q3c_radial_query(mean_fiber_ra, mean_fiber_dec, 180, 30, 1.0) "
        "AND survey = 'main' AND z > 0.4 LIMIT 100"
    )
    out3 = P.validate(sql3, source="expert")
    assert "::float8" not in out3.sql
    assert any("NaN-unsafe" in w and "z" in w for w in out3.warnings)
    # Builder SQL is never rewritten (builders guard themselves).
    sql4 = (
        "SELECT ra, dec FROM gaia_dr3.gaia_source "
        "WHERE q3c_radial_query(ra, dec, 60, -50, 1.0) AND pmra > 150 LIMIT 100"
    )
    out4 = P.validate(sql4, source="builder")
    assert "::float8" not in out4.sql


def test_confirm_sky_area_gates_wide_scans():
    from services import datalab_orchestration as orch
    small = orch.confirm_sky_area({"ra_min": 10, "ra_max": 11, "dec_min": 0, "dec_max": 1}, 1.0, max_tiles=64)
    assert small["needs_confirmation"] is False
    wide = orch.confirm_sky_area({"ra_min": 0, "ra_max": 40, "dec_min": -20, "dec_max": 20}, 1.0, max_tiles=64)
    assert wide["needs_confirmation"] is True and wide["tiles"] > 64 and wide["area_deg2"] > 0


def test_tiled_sky_scan_finds_and_ranks_planted_overdensity():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    fp = {"ra_min": 149.0, "ra_max": 152.0, "dec_min": 1.0, "dec_max": 3.0}
    client = _FakeTiledClient(hot=(150.0, 2.0))
    out = orch.tiled_sky_scan(
        "nsc_dr2", "object", fp, tile_radius_deg=1.0, step_deg=0.2,
        morphology={"column": "class_star", "op": ">", "value": 0.5},
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        analysis=_FakeAnalysis(), max_tiles=64, candidate_budget=10, confirm=True,
    )
    assert out["success"] is True and out["candidates"]
    assert client.calls == out["tiles_scanned"] > 1
    top = out["candidates"][0]
    assert abs(top["ra"] - 150.0) < 0.6 and abs(top["dec"] - 2.0) < 0.6


def test_tiled_sky_scan_requires_confirmation_when_wide():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    fp = {"ra_min": 0, "ra_max": 40, "dec_min": -20, "dec_max": 20}
    out = orch.tiled_sky_scan(
        "nsc_dr2", "object", fp, tile_radius_deg=1.0,
        client=_FakeTiledClient(), result_store=DatalabResultStore(enable_disk_cache=False),
        analysis=_FakeAnalysis(), max_tiles=64, confirm=False,
    )
    assert out["success"] is False and out["needs_confirmation"] is True


def test_tiled_sky_scan_respects_max_tiles_and_reports_dropped():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    fp = {"ra_min": 0, "ra_max": 40, "dec_min": -20, "dec_max": 20}
    out = orch.tiled_sky_scan(
        "nsc_dr2", "object", fp, tile_radius_deg=1.0,
        client=_FakeTiledClient(), result_store=DatalabResultStore(enable_disk_cache=False),
        analysis=_FakeAnalysis(), max_tiles=8, confirm=True,
    )
    assert out["success"] is True and out["dropped_tiles"] > 0
    assert out["tiles_scanned"] <= 8 and any("Capped at 8 tiles" in n for n in out["notes"])


def test_density_then_cutouts_chains_top_n():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    img = _FakeImageService()
    out = orch.density_then_cutouts(
        "nsc_dr2", "object", 185.41, -31.98, 1.0, step_deg=0.05,
        morphology={"column": "class_star", "op": ">", "value": 0.5}, top_n=3, fov_deg=0.05, band="g",
        client=_FakeDensityClient(), result_store=DatalabResultStore(enable_disk_cache=False), image_service=img,
    )
    assert out["success"] is True and out["n_peaks"] == 3
    assert len(img.calls) == 1 and len(img.calls[0][0]) == 3
    assert out["cutout_grid"]["success"] is True


def test_job_service_succeeds_and_cancels():
    from services.datalab_job_service import DatalabJobService
    svc = DatalabJobService(max_workers=2)

    jid = svc.start("unit", lambda cancel_check: {"ok": True})
    for _ in range(100):
        if svc.status(jid)["status"] in ("succeeded", "failed", "canceled"):
            break
        time.sleep(0.02)
    assert svc.status(jid)["status"] == "succeeded"
    assert svc.results(jid)["result"] == {"ok": True}

    started = threading.Event()

    def _long(cancel_check):
        started.set()
        for _ in range(2000):
            if cancel_check():
                return {"canceled": True}
            time.sleep(0.005)
        return {"done": True}

    jid2 = svc.start("unit", _long)
    assert started.wait(2.0)
    svc.cancel(jid2)
    for _ in range(200):
        if svc.status(jid2)["status"] == "canceled":
            break
        time.sleep(0.02)
    assert svc.status(jid2)["status"] == "canceled"


def test_agent_registers_p2_tools():
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    agent._register_tools()
    for name in [
        "datalab_density_vetting",
        "datalab_tiled_search",
        "datalab_confirm_sky_area",
        "datalab_job_status",
        "datalab_job_results",
        "datalab_job_cancel",
        "datalab_export_notebook",
    ]:
        assert agent.tool_registry.get_tool(name) is not None


# ── Polish items: provenance + reproducible-notebook recipe ────────────────────
def test_provenance_ledger_renders_datalab_catalog_source():
    from services.provenance import ProvenanceLedger
    ledger = ProvenanceLedger()
    ledger.extract_from_results({"catalog": "nsc_dr2", "table": "object", "rowcount": 817, "query": "SELECT 1"})
    md = ledger.render_markdown()
    assert "Data Lab catalogs" in md and "nsc_dr2.object" in md and "817 rows" in md


def test_provenance_ignores_non_datalab_catalog():
    from services.provenance import ProvenanceLedger
    ledger = ProvenanceLedger()
    ledger.extract_from_results({"catalog": "some_alma_collection", "table": "obscore"})
    assert ledger.is_empty()


def test_datalab_notebook_steps_recipe():
    from services.notebook_gen import datalab_notebook_steps
    steps = datalab_notebook_steps(
        sql="SELECT COUNT(*) FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra,dec,1,2,0.1)",
        catalog="gaia_dr3", table="gaia_source",
        sia={"ra": 10.68, "dec": 41.27, "fov_deg": 0.15},
        svo_filters=["CTIO/DECam.g", "WISE/WISE.W1"],
        citation={"text": "Gaia DR3", "url": "https://example/dr3"},
    )
    assert all(s.get("type") in ("markdown", "code") and "content" in s for s in steps)
    blob = "\n".join(s["content"] for s in steps)
    assert "qc.query(sql=q" in blob and "sia.SIAService" in blob and "SvoFps" in blob and "Gaia DR3" in blob


# ── One-shot diagram tools (the reliable P3/P5 path: query + plot in a single call) ────
class _FakeDiagramClient:
    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame({
            "ra": rng.uniform(29.5, 30.5, n), "dec": rng.uniform(-50.5, -49.5, n),
            "mag_auto_g": rng.uniform(18, 23, n), "mag_auto_r": rng.uniform(17, 22, n),
            "mag_auto_i": rng.uniform(16.5, 21.5, n),
            # bimodal spread_model_r → stars (~0) and galaxies (>0.005)
            "spread_model_r": np.concatenate([rng.normal(0.0, 0.001, n // 2), rng.normal(0.02, 0.004, n - n // 2)]),
        })
        return DatalabResult.from_dataframe(df, {"catalog": "des_dr1", "table": "main", "query": sql})


def test_color_color_diagram_one_shot_auto_splits():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    out = orch.color_color_diagram(
        "des_dr1", "main", 30.0, -50.0, 0.5,
        client=_FakeDiagramClient(), result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True and out["image_base64"]
    assert out["split_col"] == "spread_model_r"  # auto-detected from the registry
    assert {p["population"] for p in out["populations"]} == {"stars", "galaxies"}


def test_color_magnitude_diagram_one_shot():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    out = orch.color_magnitude_diagram(
        "des_dr1", "main", 30.0, -50.0, 0.4, blue_band="g", red_band="r",
        client=_FakeDiagramClient(), result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True and out["image_base64"] and out["points"] > 0


def test_agent_registers_diagram_tools():
    from tests.unit.test_datalab_p0 import _make_agent
    agent = _make_agent()
    agent._register_tools()
    for name in ("datalab_color_color_diagram", "datalab_color_magnitude_diagram"):
        tool = agent.tool_registry.get_tool(name)
        assert tool is not None
        # DLB-03 gap fix: the diagram tools must expose point-source selection
        # plus an explicit morphology-cut override.
        assert "point_sources" in tool.parameters["properties"]
        assert "morphology" in tool.parameters["properties"]


# ── Point-source / morphology cuts in the one-shot diagrams (DLB-03 gap) ───────
class _RecordingDiagramClient(_FakeDiagramClient):
    """DES-style diagram client that records every executed SQL."""

    def __init__(self):
        self.sql = []

    def query(self, *, sql=None, **kwargs):
        self.sql.append(sql or "")
        return super().query(sql=sql, **kwargs)


class _FakeNSCDiagramClient:
    """NSC-style columns (gmag/rmag/imag); records every executed SQL."""

    def __init__(self):
        self.sql = []

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        self.sql.append(sql or "")
        rng = np.random.default_rng(2)
        n = 200
        df = pd.DataFrame({
            "ra": rng.uniform(259.7, 260.4, n), "dec": rng.uniform(57.6, 58.2, n),
            "gmag": rng.uniform(17, 24, n), "rmag": rng.uniform(16.5, 23.5, n),
            "imag": rng.uniform(16, 23, n),
        })
        return DatalabResult.from_dataframe(df, {"catalog": "nsc_dr2", "table": "object", "query": sql})


def test_build_cone_select_accepts_predicates():
    from services import datalab_query_builders as B
    preds = B.build_catalog_predicates("nsc_dr2", "object", morphology={"column": "class_star", "op": ">", "value": 0.5})
    sql, meta = B.build_cone_select(
        "nsc_dr2", "object", ra=260.06, dec=57.92, radius_deg=0.4,
        columns=["ra", "dec", "gmag", "rmag"], limit=1000, predicates=preds,
    )
    assert "q3c_radial_query" in sql and "class_star > 0.5 AND class_star < 'Infinity'::float8" in sql and "LIMIT 1000" in sql
    assert meta["spatial_bound"] is True
    # No predicates -> the WHERE stays a single spatial clause (unchanged behavior).
    sql2, _ = B.build_cone_select("nsc_dr2", "object", ra=260.06, dec=57.92, radius_deg=0.4)
    assert "AND" not in sql2


def test_point_source_cut_registry_defaults():
    from services import datalab_registry as reg
    assert reg.point_source_cut("nsc_dr2", "object") == {"column": "class_star", "op": ">", "value": 0.5}
    assert reg.point_source_cut("des_dr1", "main") == {"column": "spread_model_r", "between": [-0.005, 0.005]}
    assert reg.point_source_cut("gaia_dr3", "gaia_source") is None  # no star/galaxy separator
    # Returned cut is a copy: mutating it must not corrupt the registry default.
    cut = reg.point_source_cut("des_dr1", "main")
    cut["between"].append(99)
    assert reg.point_source_cut("des_dr1", "main")["between"] == [-0.005, 0.005]


def test_cmd_point_sources_applies_registry_morphology_cut():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeNSCDiagramClient()
    out = orch.color_magnitude_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4, blue_band="g", red_band="r", point_sources=True,
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True and out["image_base64"] and out["points"] > 0
    assert "class_star > 0.5" in client.sql[0]  # the cut is in the executed SQL, not post-hoc
    assert out["point_sources"] is True


def test_cmd_explicit_value_cut_lands_in_sql():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeNSCDiagramClient()
    out = orch.color_magnitude_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4,
        value_cuts=[{"column": "class_star", "op": ">=", "value": 0.9}],
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True and "class_star >= 0.9" in client.sql[0]


def test_cmd_explicit_morphology_overrides_point_sources():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeNSCDiagramClient()
    cut = {"column": "class_star", "op": ">=", "value": 0.9}
    out = orch.color_magnitude_diagram(
        "nsc_dr2", "object", 260.06, 57.92, 0.4, point_sources=True, morphology=cut,
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True
    # The explicit cut lands in the executed SQL; the registry star cut does NOT.
    assert "class_star >= 0.9" in client.sql[0]
    assert "class_star > 0.5" not in client.sql[0]
    # The resolved cut is surfaced verbatim; point_sources reports the registry
    # cut was NOT the one applied.
    assert out["morphology"] == cut
    assert out["point_sources"] is False


def test_ccd_explicit_morphology_skips_star_galaxy_split():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _RecordingDiagramClient()
    out = orch.color_color_diagram(
        "des_dr1", "main", 30.0, -50.0, 0.5,
        morphology={"column": "spread_model_r", "op": ">", "value": 0.005},
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True and "spread_model_r > 0.005" in client.sql[0]
    # A morphology-selected sample is one population — no stars/galaxies auto-split.
    assert out["split_col"] is None
    assert out["point_sources"] is False
    assert out["morphology"] == {"column": "spread_model_r", "op": ">", "value": 0.005}
    assert [p["population"] for p in out["populations"]] == ["selected"]


class _FakeGaiaDiagramClient:
    """Gaia-style columns (phot_*_mean_mag); records every executed SQL."""

    def __init__(self):
        self.sql = []

    def query(self, *, sql=None, adql=None, fmt="pandas", **kwargs):
        from integrations.datalab_client import DatalabResult
        self.sql.append(sql or "")
        rng = np.random.default_rng(2)
        n = 200
        df = pd.DataFrame({
            "ra": rng.uniform(9.7, 10.4, n), "dec": rng.uniform(-0.4, 0.4, n),
            "phot_g_mean_mag": rng.uniform(15, 21, n),
            "phot_bp_mean_mag": rng.uniform(15, 21, n),
            "phot_rp_mean_mag": rng.uniform(14.5, 20.5, n),
        })
        return DatalabResult.from_dataframe(df, {"catalog": "gaia_dr3", "table": "gaia_source", "query": sql})


def test_cmd_point_sources_without_registered_cut_notes_and_runs():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _FakeGaiaDiagramClient()
    out = orch.color_magnitude_diagram(
        "gaia_dr3", "gaia_source", 10.0, 0.0, 0.4, point_sources=True,
        blue_band="bp", red_band="rp", mag_band="g",
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True and out["point_sources"] is False
    assert any("no registered star/galaxy separator" in w for w in out["warnings"])
    assert "class_star" not in client.sql[0]
    # The Gaia mag template resolves real columns now (not gmag/rmag).
    assert "phot_bp_mean_mag" in client.sql[0]
    # Invalid Gaia bands fail loudly with the valid set, not at the server.
    with pytest.raises(ValueError, match=r"valid bands: bp\|g\|rp"):
        orch.color_magnitude_diagram(
            "gaia_dr3", "gaia_source", 10.0, 0.0, 0.4,
            client=client, result_store=DatalabResultStore(enable_disk_cache=False),
            plotting_service=_MemoryPlottingService(),
        )


def test_ccd_point_sources_applies_cut_and_skips_star_galaxy_split():
    from services import datalab_orchestration as orch
    from services.datalab_result_store import DatalabResultStore
    from tests.unit.test_datalab_p1 import _MemoryPlottingService
    client = _RecordingDiagramClient()
    out = orch.color_color_diagram(
        "des_dr1", "main", 30.0, -50.0, 0.5, point_sources=True,
        client=client, result_store=DatalabResultStore(enable_disk_cache=False),
        plotting_service=_MemoryPlottingService(),
    )
    assert out["success"] is True and "spread_model_r BETWEEN -0.005 AND 0.005" in client.sql[0]
    # The sample is already stars-only, so there is no stars/galaxies auto-split.
    assert out["split_col"] is None
    assert out["point_sources"] is True
    assert [p["population"] for p in out["populations"]] == ["point sources"]


def test_result_store_memory_cap_evicts_lru_but_diskcache_resolves(tmp_path):
    """F-13: the in-memory dict is byte-capped with LRU eviction on every put
    (retention used to be unbounded for never-re-read results — ~1 GB/busy hour
    on a 2 GB Render instance). Evicted ids still resolve via the DiskCache."""
    from services.datalab_result_store import DatalabResultStore

    store = DatalabResultStore(cache_dir=tmp_path / "dl", ttl_seconds=3600, enable_disk_cache=True)
    if store._cache is None:
        pytest.skip("diskcache unavailable")
    store.memory_cap_bytes = 300_000  # ~3 frames of the size below

    frame = pd.DataFrame({"x": np.arange(12_000, dtype=np.float64)})  # ~96 KB
    ids = [store.put(frame, {"catalog": "gaia_dr3", "table": "gaia_source"}) for _ in range(8)]
    in_mem = [rid for rid in ids if rid in store._memory]
    assert len(in_mem) <= 3
    # The newest ids survive; the oldest were evicted.
    assert ids[-1] in store._memory and ids[0] not in store._memory
    # Evicted ids still resolve (from disk) — this is the "30-min-old result_id
    # still works in datalab_get_result" acceptance path.
    recovered = store.get(ids[0])
    pd.testing.assert_frame_equal(recovered.dataframe, frame)

    # Soak: 100 sequential puts leave the in-memory footprint under the cap.
    for _ in range(100):
        store.put(frame, {"catalog": "gaia_dr3", "table": "gaia_source"})
    total = sum(int(p.get("nbytes") or 0) for k, p in store._memory.items() if k.startswith("dlr_"))
    assert total <= store.memory_cap_bytes


def test_result_store_sweeps_expired_on_put(tmp_path):
    from services.datalab_result_store import DatalabResultStore

    store = DatalabResultStore(cache_dir=tmp_path / "dl2", ttl_seconds=3600, enable_disk_cache=False)
    stale = store.put(pd.DataFrame({"x": [1.0]}), {})
    # Backdate it past the TTL; the NEXT put must sweep it without any get().
    store._memory[stale]["created_at"] = time.time() - 7200
    fresh = store.put(pd.DataFrame({"x": [2.0]}), {})
    assert stale not in store._memory and fresh in store._memory
