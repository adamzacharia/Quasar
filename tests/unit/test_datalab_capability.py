"""
Behaviour/parity tests for capabilities/datalab.py (P1 reference migration).

Runs each migrated Data Lab capability OFFLINE with a fake Data Lab client and a
real (disk-cache-disabled) result store, asserting the ``to_native()`` output has
the exact legacy shape the agent tool loop + DataLabBench consume — and that the
capability is transport-pure (no agent, no thread-locals, no network).
"""

import pandas as pd
import pytest

from integrations.datalab_client import DatalabResult
from services.datalab_result_store import DatalabResultStore
from services import datalab_sql_policy
from capabilities.base import CallContext
from capabilities import datalab as dl


class _FakeClient:
    """Stands in for integrations.datalab_client — records the SQL, returns a
    fixed DataFrame, never hits the network."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.last_sql = None

    def query(self, sql: str, fmt: str = "pandas") -> DatalabResult:
        self.last_sql = sql
        return DatalabResult.from_dataframe(
            self.df.copy(),
            {"catalog": "gaia_dr3", "table": "gaia_source", "endpoint": "https://datalab.noirlab.edu"},
        )


def _ctx(df: pd.DataFrame):
    store = DatalabResultStore(enable_disk_cache=False)
    client = _FakeClient(df)
    ctx = CallContext(services={"datalab_client": client}, result_store=store)
    return ctx, client, store


def test_cone_count_returns_legacy_shape():
    df = pd.DataFrame({"row_count": [4242]})
    ctx, client, _ = _ctx(df)
    out = dl.ConeCount().run(
        dl.ConeCountInput(catalog="gaia_dr3", table="gaia_source", ra=229.0, dec=-0.1, radius_deg=0.05),
        ctx,
    ).to_native()

    assert out["success"] is True
    assert out["tool_name"] == "datalab_cone_count"
    assert out["result_id"].startswith("dlr_")
    assert out["rowcount"] == 1
    assert out["reported_count"] == 4242  # from the row_count column
    assert "query_summary" in out and "preview" in out and "warnings" in out
    assert "q3c_radial_query" in client.last_sql  # server-side cone executed


def test_select_catalog_rows_applies_value_cuts_serverside():
    df = pd.DataFrame({"ra": [1.0, 2.0], "dec": [3.0, 4.0], "phot_g_mean_mag": [18.0, 19.0]})
    ctx, client, store = _ctx(df)
    out = dl.SelectCatalogRows().run(
        dl.SelectCatalogRowsInput(
            catalog="gaia_dr3", table="gaia_source", ra=229.0, dec=-0.1, radius_deg=0.05,
            limit=300, value_cuts=[{"column": "phot_g_mean_mag", "op": "<", "value": 20}],
        ),
        ctx,
    ).to_native()
    assert out["success"] is True
    assert out["result_id"].startswith("dlr_")
    assert out["rowcount"] == 2
    # the cut is in the EXECUTED sql, not applied client-side
    assert "phot_g_mean_mag" in client.last_sql


def test_get_result_round_trip():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ctx, client, store = _ctx(df)
    # produce a result_id via cone_count
    res = dl.ConeCount().run(
        dl.ConeCountInput(catalog="gaia_dr3", table="gaia_source", ra=229.0, dec=-0.1, radius_deg=0.05),
        ctx,
    )
    rid = res.result_id
    got = dl.GetResult().run(dl.GetResultInput(result_id=rid, max_rows=200), ctx).to_native()
    assert got["success"] is True
    assert got["result_id"] == rid
    assert got["returned_rows"] == len(got["rows"]) >= 1
    assert "columns" in got


def test_sql_query_requires_expert_ack():
    ctx, _, _ = _ctx(pd.DataFrame({"x": [1]}))
    out = dl.SqlQuery().run(dl.SqlQueryInput(sql="SELECT 1", expert_ack=False), ctx).to_native()
    assert out["success"] is False
    assert "expert_ack" in out["error"]


def test_sql_query_expert_ack_is_strict_identity_no_coercion():
    # Parity: legacy uses `expert_ack is not True`, so a stringified/int truthy
    # value must NOT open the restricted raw-SQL gate. (P1 review A)
    ctx, _, _ = _ctx(pd.DataFrame({"row_count": [1]}))
    for truthy in ["true", "True", "yes", "on", 1]:
        out = dl.SqlQuery().run(
            dl.SqlQueryInput(sql="SELECT 1", expert_ack=truthy, reason="debug"), ctx
        ).to_native()
        assert out["success"] is False, f"{truthy!r} must not open the expert gate"
        assert "expert_ack" in out["error"]
    # A real boolean True opens the gate (proceeds past the ack check).
    out = dl.SqlQuery().run(
        dl.SqlQueryInput(sql="SELECT count(*) FROM gaia_dr3.gaia_source", expert_ack=True, reason="debug"),
        ctx,
    ).to_native()
    assert "requires expert_ack" not in (out.get("error") or "")


def test_list_and_describe_are_pure_registry_calls():
    ctx, _, _ = _ctx(pd.DataFrame())
    lc = dl.ListCatalogs().run(dl.ListCatalogsInput(), ctx).to_native()
    assert lc["success"] is True and lc["count"] == len(lc["catalogs"]) and lc["count"] > 0
    # The default scope must SAY it is the curated governed subset, not the
    # complete Data Lab inventory (NOIRLab beta eval: list flagged incomplete).
    assert lc["scope"] == "registered"
    assert "curated" in lc["note"].lower()


class _FakeSchemasClient:
    """Serves a canned tap_schema.schemas frame for scope='all' listings."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.last_sql = None

    def query(self, sql: str, fmt: str = "pandas") -> DatalabResult:
        self.last_sql = sql
        assert "tap_schema.schemas" in sql
        return DatalabResult.from_dataframe(self.df.copy(), {})


def test_list_catalogs_scope_all_returns_live_schema_list(tmp_path, monkeypatch):
    monkeypatch.setenv("DATALAB_TAP_SCHEMA_CACHE_DIR", str(tmp_path / "tap-schema"))
    schemas_df = pd.DataFrame({
        "schema_name": ["gaia_dr3", "buzzard_dr1"],
        "description": ["Gaia DR3", "Buzzard simulation"],
    })
    ctx = CallContext(
        services={"datalab_client": _FakeSchemasClient(schemas_df)},
        result_store=DatalabResultStore(enable_disk_cache=False),
    )
    out = dl.ListCatalogs().run(dl.ListCatalogsInput(scope="all"), ctx).to_native()
    assert out["success"] is True and out["scope"] == "all"
    by_name = {row["schema"]: row for row in out["schemas"]}
    assert by_name["gaia_dr3"]["registry_governed"] is True
    assert by_name["buzzard_dr1"]["registry_governed"] is False
    # The curated list still rides along for the structured builders.
    assert out["count"] == len(out["catalogs"]) > 0


def test_list_catalogs_scope_all_degrades_to_curated_with_note(tmp_path, monkeypatch):
    monkeypatch.setenv("DATALAB_TAP_SCHEMA_CACHE_DIR", str(tmp_path / "tap-schema"))

    class _DeadClient:
        def query(self, *args, **kwargs):
            raise RuntimeError("service unreachable")

    ctx = CallContext(
        services={"datalab_client": _DeadClient()},
        result_store=DatalabResultStore(enable_disk_cache=False),
    )
    out = dl.ListCatalogs().run(dl.ListCatalogsInput(scope="all"), ctx).to_native()
    # Network failure degrades to the curated list WITH an explicit note —
    # never a silent partial answer.
    assert out["success"] is True
    assert out["count"] == len(out["catalogs"]) > 0
    assert "unavailable" in out["note"]


def test_datalab_error_maps_policy_error_to_fix_hint():
    err = datalab_sql_policy.DatalabPolicyError("bad query", fix_hint="do X instead")
    tr = dl.datalab_error(err)
    native = tr.to_native()
    assert tr.success is False
    assert native["success"] is False
    assert native["error"] == "bad query"
    assert native["fix_hint"] == "do X instead"


def test_plot_capability_returns_raw_result(monkeypatch):
    # Plot capabilities are transport-pure: they return the raw datalab_analysis
    # result (with image_base64); the agent's image wrapper does the UI transport.
    import services.datalab_analysis as da
    canned = {"success": True, "image_base64": "ABC", "title": "t", "path": "/plots/x.png"}
    monkeypatch.setattr(da, "catalog_scatter", lambda *a, **k: canned)
    ctx, _, _ = _ctx(pd.DataFrame({"g": [1.0], "r": [0.5]}))
    tr = dl.CatalogScatter().run(
        dl.CatalogScatterInput(result_id="rid", x_expr="g - r", y_expr="g"), ctx
    )
    assert tr.success is True
    assert tr.to_native() == canned  # verbatim pass-through


def test_plot_capability_maps_exception_to_datalab_error(monkeypatch):
    import services.datalab_analysis as da

    def boom(*a, **k):
        raise ValueError("bad column")

    monkeypatch.setattr(da, "sky_density_map", boom)
    ctx, _, _ = _ctx(pd.DataFrame({"ra": [1.0]}))
    out = dl.SkyDensityMap().run(dl.SkyDensityMapInput(result_id="rid"), ctx).to_native()
    assert out["success"] is False and "bad column" in out["error"]


def test_sed_plot_injects_svo_client(monkeypatch):
    import services.datalab_analysis as da
    seen = {}

    def fake_sed(*a, **k):
        seen["svo"] = k.get("svo_client")
        return {"success": True, "image_base64": "z"}

    monkeypatch.setattr(da, "sed_plot", fake_sed)
    store = DatalabResultStore(enable_disk_cache=False)
    ctx = CallContext(services={"svo_fps_client": "FAKE_SVO"}, result_store=store)
    dl.SedPlot().run(dl.SedPlotInput(result_id="rid"), ctx)
    assert seen["svo"] == "FAKE_SVO"  # SVO client injected via CallContext


# ── SIA image capabilities ───────────────────────────────────────────────────
class _FakeImageService:
    def __init__(self):
        self.calls = []

    def cutout(self, ra, dec, fov, band=None, **k):
        self.calls.append(("cutout", band))
        return {"success": True, "image_base64": "IMG", "path": "/plots/c.png"}

    def color_image(self, ra, dec, fov, **k):
        self.calls.append(("color", k.get("bands")))
        return {"success": True, "image_base64": "COL"}

    def cutout_grid(self, peaks, fov, **k):
        self.calls.append(("grid", len(peaks)))
        return {"success": True, "image_base64": "GRID"}


def _img_ctx(image_service):
    store = DatalabResultStore(enable_disk_cache=False)

    def resolver(target_name=None, ra=None, dec=None):
        if ra is not None and dec is not None:
            return float(ra), float(dec), f"RA={ra}, Dec={dec}"
        return 10.0, 20.0, str(target_name)

    return CallContext(
        services={"datalab_image_service": image_service, "resolve_coordinates": resolver},
        result_store=store,
    )


def test_image_cutout_resolves_and_returns_image():
    svc = _FakeImageService()
    out = dl.ImageCutout().run(
        dl.ImageCutoutInput(fov_deg=0.1, target_name="M31", band="g"), _img_ctx(svc)
    ).to_native()
    assert out["success"] is True and out["image_base64"] == "IMG"
    assert out["_caption"] == "Data Lab g-band cutout: M31"  # computed from resolved label
    assert svc.calls == [("cutout", "g")]


def test_image_cutout_band_substitution():
    class _SvcNoG:
        def __init__(self):
            self.calls = []

        def cutout(self, ra, dec, fov, band=None, **k):
            self.calls.append(band)
            if band == "g":
                return {"success": True, "suggested_bands": ["z"]}  # no usable g tiles
            return {"success": True, "image_base64": "ZIMG", "path": "/plots/z.png"}

    svc = _SvcNoG()
    out = dl.ImageCutout().run(
        dl.ImageCutoutInput(fov_deg=0.1, ra=10.0, dec=20.0, band="g"), _img_ctx(svc)
    ).to_native()
    assert out["image_base64"] == "ZIMG"
    assert out["band_substituted"] == {"requested": "g", "used": "z"}
    assert svc.calls == ["g", "z"]


def test_color_image_and_cutout_grid_capabilities():
    svc = _FakeImageService()
    out = dl.ColorImage().run(dl.ColorImageInput(fov_deg=0.15, ra=10.0, dec=20.0), _img_ctx(svc)).to_native()
    assert out["success"] is True and out["image_base64"] == "COL"
    out2 = dl.CutoutGrid().run(dl.CutoutGridInput(peaks=[{"ra": 1, "dec": 2}], fov_deg=0.05), _img_ctx(svc)).to_native()
    assert out2["success"] is True and out2["image_base64"] == "GRID"


def test_sia_search_inventory_stores_rows_and_filters_band():
    class _SvcInventory:
        def __init__(self):
            self.calls = []

        def search(self, ra, dec, fov, catalog=None, endpoint=None):
            self.calls.append((ra, dec, fov, catalog, endpoint))
            return {
                "success": True,
                "coverage_gap": False,
                "used_endpoint": "https://datalab.noirlab.edu/sia/ls_dr9",
                "provenance": {"service": "NOIRLab Astro Data Lab SIA"},
                "rows": [
                    {"obs_bandpass": "g DECam", "exptime": 900.0, "proctype": "Stack",
                     "prodtype": "image", "access_url": "https://x/1"},
                    {"obs_bandpass": "r DECam", "exptime": 500.0, "proctype": "Stack",
                     "prodtype": "image", "access_url": "https://x/2"},
                ],
            }

    svc = _SvcInventory()
    ctx = _img_ctx(svc)
    out = dl.SiaSearch().run(
        dl.SiaSearchInput(target_name="M31", fov_deg=0.2, band="g", catalog="ls_dr9"),
        ctx,
    ).to_native()

    assert out["success"] is True
    assert out["rowcount"] == 1  # r-band row filtered out
    assert out["result_id"].startswith("dlr_")
    assert out["coverage_gap"] is False
    assert out["position"]["label"] == "M31"
    # preview shows display columns only; access_url stays in the stored rows
    assert all("access_url" not in row for row in out["preview"])
    stored = ctx.result_store.get(out["result_id"]).dataframe
    assert list(stored["access_url"]) == ["https://x/1"]
    assert svc.calls[0][3] == "ls_dr9"


def test_sia_search_requires_result_store():
    class _SvcBoom:
        def search(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("search must not run without a result store")

    ctx = CallContext(
        services={"datalab_image_service": _SvcBoom(),
                  "resolve_coordinates": lambda **k: (1.0, 2.0, "X")},
        result_store=None,
    )
    out = dl.SiaSearch().run(dl.SiaSearchInput(ra=1.0, dec=2.0), ctx).to_native()
    assert out["success"] is False
    assert "result store" in out["error"]


# ── Orchestration / job capabilities ─────────────────────────────────────────
def test_confirm_sky_area_capability(monkeypatch):
    import services.datalab_orchestration as orch
    monkeypatch.setattr(orch, "confirm_sky_area", lambda fp, r, **k: {"tiles": 9, "needs_confirmation": False})
    ctx, _, _ = _ctx(pd.DataFrame())
    out = dl.ConfirmSkyArea().run(
        dl.ConfirmSkyAreaInput(ra_min=10, ra_max=12, dec_min=-1, dec_max=1), ctx
    ).to_native()
    assert out["success"] is True and out["tiles"] == 9


def test_color_color_diagram_capability(monkeypatch):
    import services.datalab_orchestration as orch
    monkeypatch.setattr(orch, "color_color_diagram", lambda *a, **k: {"success": True, "image_base64": "CC"})
    ctx = _img_ctx(_FakeImageService())  # provides the coordinate resolver
    out = dl.ColorColorDiagram().run(
        dl.ColorColorDiagramInput(catalog="nsc_dr2", table="object", ra=10.0, dec=20.0), ctx
    ).to_native()
    assert out["success"] is True and out["image_base64"] == "CC"
    assert out["_caption"] == "Color-color diagram: RA=10.0, Dec=20.0"


def test_color_magnitude_diagram_capability(monkeypatch):
    import services.datalab_orchestration as orch
    monkeypatch.setattr(orch, "color_magnitude_diagram", lambda *a, **k: {"success": True, "image_base64": "CMD"})
    ctx = _img_ctx(_FakeImageService())
    out = dl.ColorMagnitudeDiagram().run(
        dl.ColorMagnitudeDiagramInput(catalog="nsc_dr2", table="object", ra=10.0, dec=20.0), ctx
    ).to_native()
    assert out["success"] is True and out["image_base64"] == "CMD"
    assert out["_caption"] == "Color-magnitude diagram: RA=10.0, Dec=20.0"


def test_job_results_and_cancel_capabilities():
    class _JobSvc:
        def results(self, jid):
            return {"rows": [1, 2], "job_id": jid}

        def cancel(self, jid):
            return {"cancelled": True, "job_id": jid}

    store = DatalabResultStore(enable_disk_cache=False)
    ctx = CallContext(services={"datalab_job_service": _JobSvc()}, result_store=store)
    out = dl.JobResults().run(dl.JobIdInput(job_id="dlj_j1"), ctx).to_native()
    assert out["success"] is True and out["rows"] == [1, 2]
    out2 = dl.JobCancel().run(dl.JobIdInput(job_id="dlj_j1"), ctx).to_native()
    assert out2["success"] is True and out2["cancelled"] is True


def test_capability_never_touches_network_or_agent():
    # The capability reaches its data ONLY through the client injected on the
    # CallContext: the fake records the SQL it was handed, and there is no other
    # client anywhere in scope (no agent, no module global, no thread-local).
    df = pd.DataFrame({"row_count": [7]})
    ctx, client, _ = _ctx(df)
    tr = dl.ConeCount().run(
        dl.ConeCountInput(catalog="gaia_dr3", table="gaia_source", ra=10.0, dec=10.0, radius_deg=0.05),
        ctx,
    )
    assert tr.success is True
    assert client.last_sql and "q3c_radial_query" in client.last_sql  # the INJECTED client ran the cone
    assert tr.to_native()["reported_count"] == 7  # …and its rows are what came back


# ── DensityAggregate: per-turn timeout-table state injected via CallContext ───
_AGG_DF = pd.DataFrame({"ra_bin": [1.0, 2.0], "dec_bin": [3.0, 4.0], "source_count": [10, 20]})


class _TimeoutClient:
    """Raises the sync-window timeout the live Data Lab TAP endpoint raises."""

    def __init__(self):
        self.calls = 0

    def query(self, sql: str, fmt: str = "pandas") -> DatalabResult:
        self.calls += 1
        raise RuntimeError("Query timed out after 60 seconds")


def _agg_ctx(client, timeout_tables=None):
    """CallContext carrying the per-turn timeout-table set the agent owns."""
    tables = set() if timeout_tables is None else timeout_tables
    ctx = CallContext(
        services={"datalab_client": client, "datalab_agg_timeout_tables": tables},
        result_store=DatalabResultStore(enable_disk_cache=False),
    )
    return ctx, tables


def _patch_tiling(monkeypatch):
    """Record every tiled_density_aggregate call instead of hitting the network."""
    calls = []

    def fake_tiled(catalog, table, **kw):
        calls.append({"catalog": catalog, "table": table, **kw})
        return {"success": True, "result_id": "dlr_tiled", "tiles": 4, "merged": True}

    monkeypatch.setattr(dl.datalab_orchestration, "tiled_density_aggregate", fake_tiled)
    return calls


def _agg_input(**over):
    # radius 3.0 > the 2.5-degree unconfirmed cap (F-7 HITL gate), so the tiling
    # tests model a user-confirmed wide scan.
    base = dict(catalog="gaia_dr3", table="gaia_source", ra=266.4, dec=-29.0, radius_deg=3.0, confirm=True)
    base.update(over)
    return dl.DensityAggregateInput(**base)


def test_density_aggregate_happy_path_leaves_timeout_state_clean():
    client = _FakeClient(_AGG_DF)
    ctx, tables = _agg_ctx(client)
    out = dl.DensityAggregate().run(_agg_input(radius_deg=0.2), ctx).to_native()

    assert out["success"] is True
    assert out["tool_name"] == "datalab_density_aggregate"
    assert out["result_id"].startswith("dlr_")
    assert out["rowcount"] == 2
    assert tables == set()  # nothing timed out → no table marked


def test_density_aggregate_first_timeout_tiles_with_full_budget(monkeypatch):
    calls = _patch_tiling(monkeypatch)
    client = _TimeoutClient()
    ctx, tables = _agg_ctx(client)

    out = dl.DensityAggregate().run(_agg_input(radius_deg=3.0), ctx).to_native()

    assert out["success"] is True and out["tiles"] == 4
    assert client.calls == 1  # the sync attempt WAS made the first time
    assert len(calls) == 1
    # FIRST timeout on a table gets the full tiling budget (None), not 120s.
    assert calls[0]["max_seconds"] is None
    assert tables == {"gaia_dr3.gaia_source"}  # table_key is lower-cased


def test_density_aggregate_repeat_timeout_gets_reduced_budget(monkeypatch):
    calls = _patch_tiling(monkeypatch)
    ctx1, tables = _agg_ctx(_TimeoutClient())
    dl.DensityAggregate().run(_agg_input(), ctx1)

    # Second call in the SAME turn reuses the same injected set.
    ctx2, _ = _agg_ctx(_TimeoutClient(), timeout_tables=tables)
    dl.DensityAggregate().run(_agg_input(), ctx2)

    assert [c["max_seconds"] for c in calls] == [None, 120.0]


def test_density_aggregate_skip_sync_short_circuits_the_query(monkeypatch):
    calls = _patch_tiling(monkeypatch)
    client = _TimeoutClient()
    # Pre-seed the set as if an earlier aggregate this turn had timed out.
    ctx, tables = _agg_ctx(client, timeout_tables={"gaia_dr3.gaia_source"})

    out = dl.DensityAggregate().run(_agg_input(radius_deg=2.5), ctx).to_native()

    assert client.calls == 0  # the doomed 60s sync attempt was skipped entirely
    assert out["success"] is True and len(calls) == 1
    assert calls[0]["max_seconds"] == 120.0


def test_density_aggregate_radius_exactly_two_degrees_tiles(monkeypatch):
    calls = _patch_tiling(monkeypatch)
    ctx, _ = _agg_ctx(_TimeoutClient())
    dl.DensityAggregate().run(_agg_input(radius_deg=2.0), ctx)
    assert len(calls) == 1  # the gate is `>= 2.0`, inclusive


def test_density_aggregate_narrow_cone_timeout_does_not_tile(monkeypatch):
    calls = _patch_tiling(monkeypatch)
    ctx, tables = _agg_ctx(_TimeoutClient())

    out = dl.DensityAggregate().run(_agg_input(radius_deg=1.0), ctx).to_native()

    assert calls == []  # narrow cones surface the error instead of tiling
    assert out["success"] is False and "timed out" in out["error"].lower()
    assert tables == set()


def test_density_aggregate_all_sky_without_cone_never_tiles(monkeypatch):
    calls = _patch_tiling(monkeypatch)
    ctx, _ = _agg_ctx(_TimeoutClient())
    out = dl.DensityAggregate().run(
        dl.DensityAggregateInput(catalog="gaia_dr3", table="gaia_source", all_sky=True), ctx
    ).to_native()
    assert calls == []  # has_cone is False → no tiling regardless of the error
    assert out["success"] is False


def test_density_aggregate_non_tiling_path_returns_the_same_toolresult(monkeypatch):
    # The success path must hand back the ToolResult execute_datalab_sql built,
    # untouched, so result_id / provenance / warnings survive for non-native
    # consumers. Rebuilding it would silently drop them.
    sentinel = dl.ToolResult(success=True, result_id="dlr_sentinel",
                             native={"success": True, "result_id": "dlr_sentinel"})
    monkeypatch.setattr(dl, "execute_datalab_sql", lambda *a, **k: sentinel)
    ctx, _ = _agg_ctx(_FakeClient(_AGG_DF))
    assert dl.DensityAggregate().run(_agg_input(radius_deg=0.2), ctx) is sentinel


def test_density_aggregate_state_is_shared_across_calls_in_a_turn(monkeypatch):
    _patch_tiling(monkeypatch)
    tables = set()
    ctx_a, _ = _agg_ctx(_TimeoutClient(), timeout_tables=tables)
    dl.DensityAggregate().run(_agg_input(), ctx_a)
    # A FRESH CallContext built around the same agent-owned set sees the mutation.
    ctx_b, _ = _agg_ctx(_TimeoutClient(), timeout_tables=tables)
    assert ctx_b.service("datalab_agg_timeout_tables") is tables
    assert "gaia_dr3.gaia_source" in tables


def test_density_aggregate_missing_state_service_is_a_typed_error():
    # The adapter must inject the set; a missing service becomes a typed error
    # (via datalab_error), never an uncaught KeyError crash.
    ctx = CallContext(services={"datalab_client": _FakeClient(_AGG_DF)},
                      result_store=DatalabResultStore(enable_disk_cache=False))
    out = dl.DensityAggregate().run(_agg_input(radius_deg=0.2), ctx).to_native()
    assert out["success"] is False
    assert "datalab_agg_timeout_tables" in out["error"]


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", "maybe", 1, "true"])
def test_density_aggregate_stringy_all_sky_is_rejected(raw):
    # C18 (S45): the builder gate is now the strict identity `elif all_sky is
    # True:`. The capability field stays `Any` (PV-3, no pydantic coercion), so
    # a stringy "false" — which under the legacy bare-truthiness gate silently
    # triggered an unbounded whole-catalog scan — now falls through to the
    # "requires a cone" guard instead. Only the real JSON boolean true may
    # select the all-sky branch.
    client = _FakeClient(_AGG_DF)
    ctx, _ = _agg_ctx(client)
    out = dl.DensityAggregate().run(
        dl.DensityAggregateInput(catalog="gaia_dr3", table="gaia_source", all_sky=raw), ctx
    ).to_native()
    assert bool(raw), "test only covers truthy-but-not-True JSON values"
    assert out["success"] is False
    assert "requires a cone" in out["error"]


def test_density_aggregate_bool_true_all_sky_takes_all_sky_branch():
    # The strict gate must still honor a genuine JSON true: unbounded aggregate,
    # no cone predicate in the SQL.
    client = _FakeClient(_AGG_DF)
    ctx, _ = _agg_ctx(client)
    out = dl.DensityAggregate().run(
        dl.DensityAggregateInput(catalog="gaia_dr3", table="gaia_source", all_sky=True), ctx
    ).to_native()
    assert out["success"] is True
    assert "q3c_radial_query" not in client.last_sql


@pytest.mark.parametrize("falsy", [False, None, 0, ""])
def test_density_aggregate_falsy_all_sky_without_cone_is_rejected(falsy):
    # The other half of the same gate: a falsy value with no cone must still hit
    # the builder's "requires a cone (…) or an explicit all_sky=True" guard.
    ctx, _ = _agg_ctx(_FakeClient(_AGG_DF))
    out = dl.DensityAggregate().run(
        dl.DensityAggregateInput(catalog="gaia_dr3", table="gaia_source", all_sky=falsy), ctx
    ).to_native()
    assert out["success"] is False
    assert "requires a cone" in out["error"]


def test_density_aggregate_forwards_explicit_null_mode_to_the_builder():
    # Models emit explicit JSON null. Legacy passed it straight to the builder
    # (which does `str(mode or "grid")`); the InputModel must not coerce, and
    # must not raise a ValidationError either.
    client = _FakeClient(_AGG_DF)
    ctx, _ = _agg_ctx(client)
    out = dl.DensityAggregate().run(_agg_input(radius_deg=0.2, mode=None), ctx).to_native()
    assert out["success"] is True
    assert "GROUP BY" in client.last_sql  # builder resolved None → "grid"


# ── JobStatus: per-turn poll counter injected via CallContext ────────────────
class _JobStatusSvc:
    def __init__(self, status="running"):
        self.status_value = status
        self.calls = 0

    def status(self, job_id):
        self.calls += 1
        return {"job_id": job_id, "status": self.status_value}


def _job_ctx(svc, counts=None):
    counts = {} if counts is None else counts
    return CallContext(services={"datalab_job_service": svc, "datalab_job_poll_counts": counts}), counts


def test_job_status_first_two_polls_do_not_stop_polling():
    ctx, counts = _job_ctx(_JobStatusSvc("running"))
    for expected in (1, 2):
        out = dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j1"), ctx).to_native()
        assert out["success"] is True and out["status"] == "running"
        assert "stop_polling" not in out
        assert counts == {"dlj_j1": expected}


def test_job_status_third_poll_tells_the_model_to_stop():
    ctx, counts = _job_ctx(_JobStatusSvc("queued"), counts={"dlj_j1": 2})
    out = dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j1"), ctx).to_native()
    assert out["stop_polling"] is True
    assert counts == {"dlj_j1": 3}
    assert "dlj_j1" in out["instruction"]
    assert "after 3 polls" in out["instruction"]
    assert "Do NOT call datalab_job_status again this turn." in out["instruction"]


def test_job_status_counts_each_job_id_separately():
    ctx, counts = _job_ctx(_JobStatusSvc("running"))
    dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j1"), ctx)
    dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j2"), ctx)
    dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j1"), ctx)
    assert counts == {"dlj_j1": 2, "dlj_j2": 1}
    # neither reached the 3-poll nudge
    assert dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j2"), ctx).to_native().get("stop_polling") is None


def test_job_status_mixed_case_running_status_still_counts():
    ctx, counts = _job_ctx(_JobStatusSvc("RUNNING"))
    dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j1"), ctx)
    assert counts == {"dlj_j1": 1}


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "canceled"])
def test_job_status_terminal_status_never_counts_or_stops(terminal):
    # These are the three terminal statuses services/datalab_job_service._TERMINAL
    # actually emits. The counter service is fetched ONLY on the non-terminal
    # path, so a terminal poll works even with no counter injected at all.
    svc = _JobStatusSvc(terminal)
    ctx = CallContext(services={"datalab_job_service": svc})
    out = dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j1"), ctx).to_native()
    assert out["success"] is True and out["status"] == terminal
    assert "stop_polling" not in out and "instruction" not in out


def test_job_status_service_failure_is_a_typed_error():
    class _Boom:
        def status(self, job_id):
            raise RuntimeError("job store unreachable")

    ctx, _ = _job_ctx(_Boom())
    out = dl.JobStatus().run(dl.JobIdInput(job_id="dlj_j1"), ctx).to_native()
    assert out["success"] is False and "job store unreachable" in out["error"]


# ── Agent wiring: per-turn state flows through _datalab_ctx_provider ─────────
def _wiring_agent():
    """A QuasarAgent shell (no __init__) that can build a Data Lab CallContext."""
    import threading
    from core.tools import ToolRegistry
    from tests.integration.test_agent_archive_tools import _load_agent_module

    module = _load_agent_module()
    agent = module.QuasarAgent.__new__(module.QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = ToolRegistry()
    agent.last_search_results = None
    agent.last_run_result = None
    agent.ads_client = None
    agent.openalex_client = None
    agent._datalab_client_instance = _FakeClient(_AGG_DF)
    agent._datalab_result_store_instance = DatalabResultStore(enable_disk_cache=False)
    return agent


def test_ctx_provider_injects_the_agents_own_per_turn_state_objects():
    agent = _wiring_agent()
    agent._tls.current_user_id = "alice"
    ctx1 = agent._datalab_ctx_provider()
    ctx2 = agent._datalab_ctx_provider()

    # Same object identity across calls → mutations from tool A are visible to
    # tool B within the same turn (that is the whole point of the injection).
    assert ctx1.service("datalab_agg_timeout_tables") is ctx2.service("datalab_agg_timeout_tables")
    assert ctx1.service("datalab_job_poll_counts") is ctx2.service("datalab_job_poll_counts")
    # …and they ARE the agent's attributes, lazily created.
    assert ctx1.service("datalab_agg_timeout_tables") is agent._datalab_agg_timeout_tables
    assert ctx1.service("datalab_job_poll_counts") is agent._job_poll_counts
    assert ctx1.user_id == "alice"
    agent._sandbox_tool_bridge("missing", {}, user_id="bob")
    assert agent._tls.current_user_id == "bob"


def test_ctx_provider_picks_up_the_per_turn_reset():
    # stream_response_api REBINDS these two attributes at the top of every turn.
    # The provider must re-read them per call, never cache them — otherwise the
    # poll counter and timeout tables would leak across turns.
    agent = _wiring_agent()
    agent._datalab_ctx_provider().service("datalab_job_poll_counts")["j1"] = 9

    agent._job_poll_counts = {}                      # ← the per-turn reset
    agent._datalab_agg_timeout_tables = set()

    ctx = agent._datalab_ctx_provider()
    assert ctx.service("datalab_job_poll_counts") == {}
    assert ctx.service("datalab_job_poll_counts") is agent._job_poll_counts
    assert ctx.service("datalab_agg_timeout_tables") is agent._datalab_agg_timeout_tables


def test_migrated_tools_keep_their_legacy_registration_surface():
    # The migration swapped only `function=`; description + parameters schema
    # (the LLM-facing surface) must be byte-identical to the inline versions.
    agent = _wiring_agent()
    agent._register_tools()

    names = {c.name for c in dl.CAPABILITIES}
    assert {"datalab_density_aggregate", "datalab_job_status"} <= names
    assert len(names) == len(dl.CAPABILITIES)  # no duplicate capability names

    agg = agent.tool_registry.get_tool("datalab_density_aggregate")
    assert agg is not None
    assert agg.description.startswith("Aggregate Data Lab source density by RA/Dec grid")
    assert "automatically tiled into sub-cones" in agg.description
    assert agg.parameters["required"] == ["catalog", "table"]
    assert "healpix_column" in agg.parameters["properties"]

    job = agent.tool_registry.get_tool("datalab_job_status")
    assert job is not None
    assert job.description == ("Poll the status of a Data Lab background job — local 'dlj_' ids (tiled "
                               "search, local async queries) and real server-side job ids from async_submit.")
    assert job.parameters["required"] == ["job_id"]


def test_density_aggregate_dispatches_through_the_registry():
    import json as _json
    agent = _wiring_agent()
    agent._register_tools()
    payload = _json.loads(
        agent._dispatch_tool_call(
            "datalab_density_aggregate",
            _json.dumps({"catalog": "gaia_dr3", "table": "gaia_source",
                         "ra": 10.0, "dec": 0.0, "radius_deg": 0.2}),
        )
    )
    assert payload["success"] is True
    assert payload["tool_name"] == "datalab_density_aggregate"
    assert payload["result_id"].startswith("dlr_")


# ── DensityVetting: nested image marked for the adapter-boundary attach ──────
def _vetting_ctx(image_service=None):
    store = DatalabResultStore(enable_disk_cache=False)

    def resolver(target_name=None, ra=None, dec=None):
        if ra is not None and dec is not None:
            return float(ra), float(dec), f"RA={ra}, Dec={dec}"
        return 10.0, 20.0, str(target_name)

    return CallContext(
        services={
            "datalab_client": _FakeClient(_AGG_DF),
            "datalab_image_service": image_service or _FakeImageService(),
            "resolve_coordinates": resolver,
        },
        result_store=store,
    )


def _patch_vetting(monkeypatch, grid):
    """Record the density_then_cutouts call and return a canned P12 result."""
    calls = []

    def fake(catalog, table, ra, dec, radius_deg, **kw):
        calls.append({"catalog": catalog, "table": table, "ra": ra, "dec": dec,
                      "radius_deg": radius_deg, **kw})
        return {"success": True, "result_id": "dlr_vet", "n_peaks": 1,
                "peaks": [{"ra": 1.0, "dec": 2.0, "label": "n=10"}], "cutout_grid": grid}

    monkeypatch.setattr(dl.datalab_orchestration, "density_then_cutouts", fake)
    return calls


def test_density_vetting_marks_nested_grid_and_injects_services(monkeypatch):
    grid = {"image_base64": "GRID64", "path": "/plots/grid.png"}
    calls = _patch_vetting(monkeypatch, grid)
    ctx = _vetting_ctx()
    out = dl.DensityVetting().run(
        dl.DensityVettingInput(catalog="nsc_dr2", table="object", radius_deg=0.3, ra=10.0, dec=20.0),
        ctx,
    ).to_native()

    assert out["success"] is True
    assert out["target"] == "RA=10.0, Dec=20.0"
    # The nested grid is a COPY marked with the private caption under exactly
    # the legacy attach condition (has image_base64/path) + success setdefault.
    marked = out["cutout_grid"]
    assert marked is not grid
    assert marked["success"] is True
    assert marked["_caption"] == "Density-peak cutout grid: RA=10.0, Dec=20.0 (top 5)"
    assert marked["image_base64"] == "GRID64"  # capability does NOT strip — the wrapper does
    # The orchestration got the ctx-injected client/store/image service.
    assert calls[0]["client"] is ctx.service("datalab_client")
    assert calls[0]["result_store"] is ctx.result_store
    assert calls[0]["image_service"] is ctx.service("datalab_image_service")


def test_density_vetting_imageless_grid_left_untouched(monkeypatch):
    grid = {"panels": []}  # no image_base64 / path → legacy never attached
    _patch_vetting(monkeypatch, grid)
    out = dl.DensityVetting().run(
        dl.DensityVettingInput(catalog="nsc_dr2", table="object", radius_deg=0.3, ra=10.0, dec=20.0),
        _vetting_ctx(),
    ).to_native()
    assert out["cutout_grid"] is grid          # same object, no copy
    assert "_caption" not in out["cutout_grid"]
    assert "success" not in out["cutout_grid"]  # no setdefault outside the attach branch


def test_density_vetting_null_coerces_optionals_to_legacy_defaults(monkeypatch):
    calls = _patch_vetting(monkeypatch, {"panels": []})
    out = dl.DensityVetting().run(
        dl.DensityVettingInput(catalog="nsc_dr2", table="object", radius_deg=None,
                               ra=10.0, dec=20.0, step_deg=None, top_n=None,
                               fov_deg=None, band=None),
        _vetting_ctx(),
    ).to_native()
    assert out["success"] is True
    c = calls[0]
    assert (c["radius_deg"], c["step_deg"], c["top_n"], c["fov_deg"], c["band"]) == (0.5, 0.05, 5, 0.05, "g")


def test_density_vetting_radius_key_is_required_but_nullable():
    # Legacy took radius_deg as a required positional and null-coerced an
    # explicit null; the InputModel mirrors that: missing key → validation
    # error, present-but-null → accepted (coerced in run()).
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        dl.DensityVettingInput(catalog="nsc_dr2", table="object")
    inp = dl.DensityVettingInput(catalog="nsc_dr2", table="object", radius_deg=None)
    assert inp.radius_deg is None


def test_density_vetting_resolver_failure_is_typed(monkeypatch):
    _patch_vetting(monkeypatch, {"panels": []})

    def bad_resolver(target_name=None, ra=None, dec=None):
        raise ValueError("Could not resolve target 'Nope'")

    ctx = CallContext(
        services={"datalab_client": _FakeClient(_AGG_DF),
                  "datalab_image_service": _FakeImageService(),
                  "resolve_coordinates": bad_resolver},
        result_store=DatalabResultStore(enable_disk_cache=False),
    )
    out = dl.DensityVetting().run(
        dl.DensityVettingInput(catalog="nsc_dr2", table="object", radius_deg=0.3, target_name="Nope"),
        ctx,
    ).to_native()
    assert out["success"] is False and "Could not resolve" in out["error"]


# ── TiledSearch: HITL gate + background job start ────────────────────────────
class _FakeJobService:
    def __init__(self):
        self.started = []

    def start(self, name, fn, params=None):
        self.started.append({"name": name, "fn": fn, "params": params})
        return f"job_{len(self.started)}"


def _tiled_ctx(job_service=None):
    svc = job_service or _FakeJobService()
    return CallContext(services={"datalab_job_service": svc}), svc


def _tiled_input(**over):
    base = dict(catalog="nsc_dr2", table="object", ra_min=10.0, ra_max=12.0,
                dec_min=-1.0, dec_max=1.0)
    base.update(over)
    return dl.TiledSearchInput(**base)


def test_tiled_search_needs_confirmation_gate(monkeypatch):
    monkeypatch.setattr(dl.datalab_orchestration, "confirm_sky_area",
                        lambda fp, r, **k: {"needs_confirmation": True, "tiles": 200,
                                            "area_deg2": 4000.0, "message": "confirm first"})
    ctx, svc = _tiled_ctx()
    out = dl.TiledSearch().run(_tiled_input(), ctx).to_native()
    assert out["success"] is False and out["needs_confirmation"] is True
    assert out["tiles"] == 200
    assert "confirm=true" in out["hint"]
    assert svc.started == []  # the scan was NOT started


def test_tiled_search_starts_job_when_confirmed(monkeypatch):
    seen = {}

    def fake_confirm(fp, r, max_tiles=None):
        seen["confirm_args"] = (fp, r, max_tiles)
        return {"needs_confirmation": True, "tiles": 200, "area_deg2": 4000.0, "message": "m"}

    monkeypatch.setattr(dl.datalab_orchestration, "confirm_sky_area", fake_confirm)
    ctx, svc = _tiled_ctx()
    out = dl.TiledSearch().run(_tiled_input(confirm=True, tile_radius_deg=1.5, max_tiles=32), ctx).to_native()

    assert out["success"] is True and out["status"] == "queued"
    assert out["job_id"] == "job_1"
    assert "datalab_job_status" in out["note"]
    assert out["tiles"] == 200  # the decision dict is merged into the reply
    assert seen["confirm_args"] == ({"ra_min": 10.0, "ra_max": 12.0, "dec_min": -1.0, "dec_max": 1.0}, 1.5, 32)
    assert svc.started[0]["name"] == "tiled_sky_scan"
    assert svc.started[0]["params"] == {"catalog": "nsc_dr2", "table": "object",
                                        "ra_min": 10.0, "ra_max": 12.0, "dec_min": -1.0, "dec_max": 1.0}


def test_tiled_search_small_area_runs_without_confirm(monkeypatch):
    monkeypatch.setattr(dl.datalab_orchestration, "confirm_sky_area",
                        lambda fp, r, **k: {"needs_confirmation": False, "tiles": 4})
    ctx, svc = _tiled_ctx()
    out = dl.TiledSearch().run(_tiled_input(), ctx).to_native()
    assert out["success"] is True and len(svc.started) == 1


def test_tiled_search_confirm_keeps_legacy_truthiness(monkeypatch):
    # The legacy gate is `... and not confirm` on the RAW JSON value, so a
    # stringified "false"/"0" counted as confirmed and ran the scan. A `bool`
    # field would coerce those to False and bounce the scan back. (Same
    # hazard/fix as expert_ack and all_sky.)
    monkeypatch.setattr(dl.datalab_orchestration, "confirm_sky_area",
                        lambda fp, r, **k: {"needs_confirmation": True, "tiles": 200})
    for raw in ("false", "0", "no"):
        ctx, svc = _tiled_ctx()
        out = dl.TiledSearch().run(_tiled_input(confirm=raw), ctx).to_native()
        assert out["success"] is True, f"{raw!r} must count as confirmed (legacy truthiness)"
        assert len(svc.started) == 1


def test_tiled_search_job_closure_runs_the_scan_confirmed(monkeypatch):
    monkeypatch.setattr(dl.datalab_orchestration, "confirm_sky_area",
                        lambda fp, r, **k: {"needs_confirmation": False, "tiles": 4})
    scan_calls = []

    def fake_scan(catalog, table, fp, **kw):
        scan_calls.append({"catalog": catalog, "table": table, "fp": fp, **kw})
        return {"success": True, "candidates": []}

    monkeypatch.setattr(dl.datalab_orchestration, "tiled_sky_scan", fake_scan)
    ctx, svc = _tiled_ctx()
    dl.TiledSearch().run(_tiled_input(step_deg=0.1, peak_threshold=2.5, candidate_budget=10), ctx)

    # The deferred job body runs the scan with confirm FORCED True and the
    # cancel_check handed through by the job service.
    sentinel = object()
    result = svc.started[0]["fn"](sentinel)
    assert result == {"success": True, "candidates": []}
    call = scan_calls[0]
    assert call["confirm"] is True
    assert call["cancel_check"] is sentinel
    assert (call["tile_radius_deg"], call["step_deg"]) == (2.0, 0.1)
    assert (call["peak_threshold"], call["max_tiles"], call["candidate_budget"]) == (2.5, 64, 10)


def test_tiled_search_error_is_typed(monkeypatch):
    def boom(fp, r, **k):
        raise RuntimeError("registry offline")

    monkeypatch.setattr(dl.datalab_orchestration, "confirm_sky_area", boom)
    ctx, _ = _tiled_ctx()
    out = dl.TiledSearch().run(_tiled_input(), ctx).to_native()
    assert out["success"] is False and "registry offline" in out["error"]


# ── ExportNotebook: injected generator owns the card transport ───────────────
def _notebook_ctx(generator=None, record=None):
    record = record if record is not None else []

    def default_generator(title, steps):
        record.append({"title": title, "steps": steps})
        return {"success": True, "message": "Notebook generated successfully. Let the user know it is ready to download."}

    return CallContext(services={"generate_notebook": generator or default_generator}), record


def test_export_notebook_builds_steps_and_calls_the_injected_generator():
    ctx, record = _notebook_ctx()
    out = dl.ExportNotebook().run(
        dl.ExportNotebookInput(sql="SELECT 1", catalog="gaia_dr3", table="gaia_source"), ctx
    ).to_native()
    assert out["success"] is True and "Notebook generated" in out["message"]
    assert record[0]["title"] == "NOIRLab Data Lab analysis"  # legacy default title
    assert isinstance(record[0]["steps"], list) and record[0]["steps"]


def test_export_notebook_sia_recipe_needs_both_coordinates():
    ctx, record = _notebook_ctx()
    dl.ExportNotebook().run(dl.ExportNotebookInput(sql="SELECT 1", sia_ra=10.0), ctx)
    steps_without = record[-1]["steps"]
    dl.ExportNotebook().run(
        dl.ExportNotebookInput(sql="SELECT 1", sia_ra=10.0, sia_dec=20.0, sia_fov_deg=0.2), ctx
    )
    steps_with = record[-1]["steps"]
    # Adding the dec completes the SIA recipe → strictly more notebook steps.
    assert len(steps_with) > len(steps_without)


def test_export_notebook_unknown_catalog_citation_is_swallowed():
    # datalab_registry.citation raises for unknown catalogs; legacy swallowed it
    # and still produced the notebook.
    ctx, record = _notebook_ctx()
    out = dl.ExportNotebook().run(
        dl.ExportNotebookInput(sql="SELECT 1", catalog="not_a_real_catalog"), ctx
    ).to_native()
    assert out["success"] is True and record


def test_export_notebook_generator_failure_is_a_typed_error():
    def failing_generator(title, steps):
        return {"success": False, "error": "nbformat missing"}

    ctx, _ = _notebook_ctx(generator=failing_generator)
    tr = dl.ExportNotebook().run(dl.ExportNotebookInput(sql="SELECT 1"), ctx)
    assert tr.success is False and tr.error == "nbformat missing"
    assert tr.to_native() == {"success": False, "error": "nbformat missing"}  # verbatim


# ── Agent wiring: nested-image attach + the clear_card opt-out ───────────────
def test_density_vetting_wrapper_attaches_nested_grid_card(monkeypatch):
    grid = {"image_base64": "GRID64", "path": "/plots/grid.png"}
    _patch_vetting(monkeypatch, grid)
    agent = _wiring_agent()
    fn = agent._datalab_image_tool_fn("datalab_density_vetting", nested_key="cutout_grid")

    out = fn(catalog="nsc_dr2", table="object", radius_deg=0.3, ra=10.0, dec=20.0)

    # The UI card was set from the NESTED grid with the computed caption…
    card = agent.last_run_result
    assert card["type"] == "image"
    assert card["image_url"] == "/plots/grid.png"
    assert card["caption"] == "Density-peak cutout grid: RA=10.00000, Dec=20.00000 (top 5)"
    # …the heavy base64 was stripped from the LLM-facing nested dict, the
    # private mark never leaked, and the top level was left untouched.
    assert out["cutout_grid"]["image_attached"] is True
    assert "image_base64" not in out["cutout_grid"]
    assert "_caption" not in out["cutout_grid"]
    assert "image_attached" not in out
    assert out["success"] is True and out["result_id"] == "dlr_vet"


def test_export_notebook_wrapper_does_not_clear_the_card():
    # datalab_export_notebook registers with clear_card=False: the injected
    # generator sets the card, so the ctx provider must NOT wipe a prior one
    # (legacy deliberately never cleared last_run_result here).
    agent = _wiring_agent()
    sentinel = {"type": "data", "data": "keep-me"}
    agent.last_run_result = sentinel
    agent._generate_notebook = lambda title, steps: {"success": True, "message": "ok"}

    fn = agent._datalab_tool_fn("datalab_export_notebook", clear_card=False)
    out = fn(sql="SELECT 1")

    assert out["success"] is True
    assert agent.last_run_result is sentinel  # untouched by the ctx provider


def test_plain_tool_fn_still_clears_the_card():
    # Contrast: every other datalab tool keeps the provider's clear.
    agent = _wiring_agent()
    agent.last_run_result = {"type": "data", "data": "stale"}
    fn = agent._datalab_tool_fn("datalab_confirm_sky_area")
    out = fn(ra_min=10.0, ra_max=10.5, dec_min=0.0, dec_max=0.5)
    # Assert the call actually SUCCEEDED (CX-12): a context/adapter failure would
    # also clear the card and make a bare `is None` assertion pass vacuously.
    assert out["success"] is True
    assert agent.last_run_result is None


def test_export_notebook_end_to_end_sets_the_notebook_card():
    # Through the agent's _generate_notebook: the notebook card is the
    # deliverable. (The importlib harness stubs generate_analysis_notebook to
    # return {} inside the loaded agent module, so assert the card SHAPE here;
    # the real cell content is covered by the direct-capability test below.)
    agent = _wiring_agent()
    fn = agent._datalab_tool_fn("datalab_export_notebook", clear_card=False)
    out = fn(sql="SELECT TOP 5 * FROM gaia_dr3.gaia_source", catalog="gaia_dr3", table="gaia_source")
    assert out["success"] is True
    card = agent.last_run_result
    assert card["type"] == "notebook"
    assert card["title"] == "NOIRLab Data Lab analysis"
    assert isinstance(card["notebook_data"], dict)


def test_export_notebook_real_generator_produces_cells():
    # Same wiring as the agent's _generate_notebook, with the REAL (unstubbed)
    # services.notebook_gen: the produced notebook has the TAP-recipe cells.
    from services.notebook_gen import generate_analysis_notebook
    cards = {}

    def gen(title, steps):
        nb = generate_analysis_notebook(title, steps)
        cards["card"] = {"type": "notebook", "notebook_data": nb, "title": title}
        return {"success": True, "message": "ok"}

    ctx = CallContext(services={"generate_notebook": gen})
    out = dl.ExportNotebook().run(
        dl.ExportNotebookInput(sql="SELECT TOP 5 * FROM gaia_dr3.gaia_source",
                               catalog="gaia_dr3", table="gaia_source"),
        ctx,
    ).to_native()
    assert out["success"] is True
    cells = cards["card"]["notebook_data"]["cells"]
    assert cells
    blob = "".join("".join(c.get("source", [])) for c in cells)
    assert "qc.query(sql=q, fmt='pandas')" in blob  # the governed TAP recipe cell


def test_new_tools_keep_their_legacy_registration_surface():
    # The migration swapped only `function=`; description + parameters schema
    # (the LLM-facing surface) must be byte-identical to the inline versions.
    agent = _wiring_agent()
    agent._register_tools()

    vet = agent.tool_registry.get_tool("datalab_density_vetting")
    assert vet is not None
    assert vet.description.startswith("P12: find the densest catalog cells")
    assert vet.parameters["required"] == ["catalog", "table", "radius_deg"]

    tiled = agent.tool_registry.get_tool("datalab_tiled_search")
    assert tiled is not None
    assert tiled.description.startswith("P15: tiled region-bounded overdensity search")
    assert tiled.parameters["required"] == ["catalog", "table", "ra_min", "ra_max", "dec_min", "dec_max"]
    assert tiled.parameters["properties"]["confirm"]["default"] is False

    nb = agent.tool_registry.get_tool("datalab_export_notebook")
    assert nb is not None
    assert nb.description.startswith("Export a reproducible Jupyter notebook")
    assert nb.parameters["required"] == []


# ── Registry-dispatch parity for the 3 new tools (CX-11) ─────────────────────
# The tests above build the wrappers by hand; these dispatch through the ACTUAL
# registered function via _dispatch_tool_call, so a wrong registered function,
# a missing nested_key, or a lost clear_card=False would be caught end-to-end.
def test_density_vetting_dispatches_through_the_registry(monkeypatch):
    import json as _json
    grid = {"image_base64": "GRID64", "path": "/plots/grid.png"}
    _patch_vetting(monkeypatch, grid)
    agent = _wiring_agent()
    # density_then_cutouts is mocked, so the real image service is never needed.
    agent._register_tools()
    payload = _json.loads(
        agent._dispatch_tool_call(
            "datalab_density_vetting",
            _json.dumps({"catalog": "nsc_dr2", "table": "object",
                         "radius_deg": 0.3, "ra": 10.0, "dec": 20.0}),
        )
    )
    assert payload["success"] is True and payload["result_id"] == "dlr_vet"
    # nested_key wiring: the grid card attached, base64 stripped, top level clean.
    assert payload["cutout_grid"]["image_attached"] is True
    assert "image_base64" not in payload["cutout_grid"]
    assert "_caption" not in payload["cutout_grid"]
    card = agent.last_run_result
    assert card["type"] == "image" and card["image_url"] == "/plots/grid.png"


def test_tiled_search_dispatches_through_the_registry(monkeypatch):
    import json as _json
    monkeypatch.setattr(dl.datalab_orchestration, "confirm_sky_area",
                        lambda fp, r, **k: {"needs_confirmation": True, "tiles": 200,
                                            "area_deg2": 4000.0, "message": "confirm first"})
    agent = _wiring_agent()
    agent._register_tools()
    payload = _json.loads(
        agent._dispatch_tool_call(
            "datalab_tiled_search",
            _json.dumps({"catalog": "nsc_dr2", "table": "object",
                         "ra_min": 10.0, "ra_max": 40.0, "dec_min": -10.0, "dec_max": 10.0}),
        )
    )
    # HITL gate reached end-to-end (no confirm → no job started).
    assert payload["success"] is False and payload["needs_confirmation"] is True
    assert "confirm=true" in payload["hint"]


def test_export_notebook_dispatches_through_the_registry():
    import json as _json
    agent = _wiring_agent()
    agent._register_tools()
    # Pre-seed a card; clear_card=False must leave it for the generator to set,
    # not wipe it via the provider (end-to-end proof the flag is wired at
    # registration, not just in a hand-built wrapper).
    agent.last_run_result = {"type": "data", "data": "prior"}
    payload = _json.loads(
        agent._dispatch_tool_call(
            "datalab_export_notebook",
            _json.dumps({"sql": "SELECT 1", "catalog": "gaia_dr3", "table": "gaia_source"}),
        )
    )
    assert payload["success"] is True
    card = agent.last_run_result
    assert card["type"] == "notebook"  # the generator's card, not the prior data card


# ── Guard-verify fixes: fail-before-query + export service independence ───────
def test_execute_datalab_sql_none_store_fails_before_query():
    # Guard-verify NEW-REGRESSION fix: the agent provider wraps service getters
    # in `_lazy` (a failing unrelated constructor → None). A None result_store
    # must fail BEFORE the remote query fires, never issue client.query() and
    # then crash at store.put(). Beta built the store eagerly at context build,
    # so no query ran when the store was unavailable.
    client = _FakeClient(_AGG_DF)
    ctx = CallContext(services={"datalab_client": client}, result_store=None)
    out = dl.execute_datalab_sql(
        "SELECT count(*) FROM gaia_dr3.gaia_source", {"catalog": "gaia_dr3", "table": "gaia_source"},
        tool_name="datalab_cone_count", ctx=ctx,
    ).to_native()
    assert out["success"] is False
    assert "result store is unavailable" in out["error"].lower()
    assert client.last_sql is None  # the remote query was NEVER issued


def test_cone_count_none_store_does_not_query(monkeypatch):
    # Same guarantee through a real capability (not just the helper).
    client = _FakeClient(pd.DataFrame({"row_count": [1]}))
    ctx = CallContext(services={"datalab_client": client}, result_store=None)
    out = dl.ConeCount().run(
        dl.ConeCountInput(catalog="gaia_dr3", table="gaia_source", ra=10.0, dec=0.0, radius_deg=0.05),
        ctx,
    ).to_native()
    assert out["success"] is False and client.last_sql is None


def test_export_notebook_provider_builds_no_datalab_services():
    # Guard-verify CX-02 fix: export_notebook routes to the MINIMAL notebook
    # provider, which constructs no Data Lab client / image service. Prove it by
    # making those getters RAISE — export must still succeed because it never
    # touches them (beta parity: a pure notebook export is independent of Data
    # Lab service init).
    agent = _wiring_agent()

    def _boom(*a, **k):
        raise RuntimeError("image service init failed (unwritable plots dir)")

    agent._get_datalab_image_service = _boom
    agent._get_datalab_client = _boom
    agent._get_svo_fps_client = _boom
    agent._register_tools()
    import json as _json
    payload = _json.loads(
        agent._dispatch_tool_call(
            "datalab_export_notebook",
            _json.dumps({"sql": "SELECT 1", "catalog": "gaia_dr3", "table": "gaia_source"}),
        )
    )
    assert payload["success"] is True  # export ran with zero Data Lab services
    assert agent.last_run_result["type"] == "notebook"


def test_export_notebook_minimal_provider_has_only_generator():
    # The minimal provider injects generate_notebook and nothing Data Lab-y.
    agent = _wiring_agent()
    ctx = agent._datalab_notebook_ctx_provider()
    assert set(ctx.services) == {"generate_notebook"}
    assert ctx.result_store is None


def test_sia_search_stamps_authenticated_owner_on_stored_rows():
    """f2-CX-01 / f2-CX-19: the SIA producer stamps ctx.user_id as owner_id so
    the export route can enforce ownership; an anonymous ctx stays ownerless."""
    class _SvcInv:
        def search(self, ra, dec, fov, catalog=None, endpoint=None):
            return {
                "success": True, "coverage_gap": False,
                "used_endpoint": "https://datalab.noirlab.edu/sia/ls_dr9",
                "provenance": {"service": "NOIRLab Astro Data Lab SIA"},
                "rows": [{"obs_bandpass": "g DECam", "access_url": "https://x/1"}],
            }

    def _ctx_for(user_id):
        store = DatalabResultStore(enable_disk_cache=False)
        ctx = CallContext(
            services={
                "datalab_image_service": _SvcInv(),
                "resolve_coordinates": lambda target_name=None, ra=None, dec=None, **k: (10.0, 20.0, "X"),
            },
            result_store=store,
            user_id=user_id,
        )
        return ctx, store

    ctx, store = _ctx_for("user-42")
    out = dl.SiaSearch().run(dl.SiaSearchInput(ra=10.0, dec=20.0, fov_deg=0.2), ctx).to_native()
    assert out["success"] is True
    _frame, meta, status = store.lookup(out["result_id"])
    assert status == "ok" and meta["owner_id"] == "user-42"

    ctx2, store2 = _ctx_for(None)
    out2 = dl.SiaSearch().run(dl.SiaSearchInput(ra=10.0, dec=20.0, fov_deg=0.2), ctx2).to_native()
    assert out2["success"] is True
    _frame2, meta2, _status2 = store2.lookup(out2["result_id"])
    assert "owner_id" not in meta2


def test_sia_search_records_upstream_truncation_structurally():
    """f2-CX-21: when SIA caps remote rows, the cap is recorded structurally —
    store meta + tool summary carry upstream_truncated/upstream_total, and the
    stored result's provenance surfaces both to every consumer (the
    datalab_get_result path that later builds a data card)."""
    class _SvcMany:
        def search(self, ra, dec, fov, catalog=None, endpoint=None):
            return {
                "success": True, "coverage_gap": False, "used_endpoint": "e",
                "provenance": {"service": "NOIRLab Astro Data Lab SIA"},
                "rows": [{"obs_bandpass": "g", "access_url": f"u{i}"} for i in range(5)],
            }

    store = DatalabResultStore(enable_disk_cache=False)
    ctx = CallContext(
        services={
            "datalab_image_service": _SvcMany(),
            "resolve_coordinates": lambda target_name=None, ra=None, dec=None, **k: (1.0, 2.0, "X"),
        },
        result_store=store,
        user_id="u1",
    )
    out = dl.SiaSearch().run(dl.SiaSearchInput(ra=1.0, dec=2.0, fov_deg=0.1, limit=2), ctx).to_native()

    assert out["success"] is True
    assert out["upstream_truncated"] is True and out["upstream_total"] == 5
    _frame, meta, _status = store.lookup(out["result_id"])
    assert meta["upstream_truncated"] is True and meta["upstream_total"] == 5
    res = store.get(out["result_id"])
    assert res.provenance["upstream_truncated"] is True
    assert res.provenance["upstream_total"] == 5


def test_sia_search_uncapped_result_carries_no_upstream_flags():
    """f2-CX-21: an uncapped SIA result must NOT carry the flags — over-flagging
    would be dishonest in the other direction."""
    class _SvcFew:
        def search(self, ra, dec, fov, catalog=None, endpoint=None):
            return {
                "success": True, "coverage_gap": False, "used_endpoint": "e",
                "provenance": {"service": "NOIRLab Astro Data Lab SIA"},
                "rows": [{"obs_bandpass": "g", "access_url": "u0"}],
            }

    store = DatalabResultStore(enable_disk_cache=False)
    ctx = CallContext(
        services={
            "datalab_image_service": _SvcFew(),
            "resolve_coordinates": lambda target_name=None, ra=None, dec=None, **k: (1.0, 2.0, "X"),
        },
        result_store=store,
        user_id="u1",
    )
    out = dl.SiaSearch().run(dl.SiaSearchInput(ra=1.0, dec=2.0, fov_deg=0.1), ctx).to_native()

    assert out["success"] is True
    assert "upstream_truncated" not in out
    _frame, meta, _status = store.lookup(out["result_id"])
    assert "upstream_truncated" not in meta
