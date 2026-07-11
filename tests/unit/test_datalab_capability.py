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
    out = dl.JobResults().run(dl.JobIdInput(job_id="j1"), ctx).to_native()
    assert out["success"] is True and out["rows"] == [1, 2]
    out2 = dl.JobCancel().run(dl.JobIdInput(job_id="j1"), ctx).to_native()
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
    base = dict(catalog="gaia_dr3", table="gaia_source", ra=266.4, dec=-29.0, radius_deg=3.0)
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
def test_density_aggregate_all_sky_keeps_legacy_truthiness(raw):
    # The builder's gate is a bare `elif all_sky:` truthiness check, and the
    # legacy method forwarded the raw JSON value — so ANY non-empty string
    # (including "false") selected the all-sky branch. Typing the field `bool`
    # would let pydantic coerce "false"/"0"/"no"/"off" to False and refuse the
    # scan the legacy would have run. `Any` preserves parity exactly.
    # (Same hazard as SqlQueryInput.expert_ack.)
    client = _FakeClient(_AGG_DF)
    ctx, _ = _agg_ctx(client)
    out = dl.DensityAggregate().run(
        dl.DensityAggregateInput(catalog="gaia_dr3", table="gaia_source", all_sky=raw), ctx
    ).to_native()
    assert bool(raw), "test only covers truthy JSON values"
    assert out["success"] is True                 # took the all-sky branch, as legacy did
    assert "q3c_radial_query" not in client.last_sql   # unbounded: no cone predicate


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
        out = dl.JobStatus().run(dl.JobIdInput(job_id="j1"), ctx).to_native()
        assert out["success"] is True and out["status"] == "running"
        assert "stop_polling" not in out
        assert counts == {"j1": expected}


def test_job_status_third_poll_tells_the_model_to_stop():
    ctx, counts = _job_ctx(_JobStatusSvc("queued"), counts={"j1": 2})
    out = dl.JobStatus().run(dl.JobIdInput(job_id="j1"), ctx).to_native()
    assert out["stop_polling"] is True
    assert counts == {"j1": 3}
    assert "j1" in out["instruction"]
    assert "after 3 polls" in out["instruction"]
    assert "Do NOT call datalab_job_status again this turn." in out["instruction"]


def test_job_status_counts_each_job_id_separately():
    ctx, counts = _job_ctx(_JobStatusSvc("running"))
    dl.JobStatus().run(dl.JobIdInput(job_id="j1"), ctx)
    dl.JobStatus().run(dl.JobIdInput(job_id="j2"), ctx)
    dl.JobStatus().run(dl.JobIdInput(job_id="j1"), ctx)
    assert counts == {"j1": 2, "j2": 1}
    # neither reached the 3-poll nudge
    assert dl.JobStatus().run(dl.JobIdInput(job_id="j2"), ctx).to_native().get("stop_polling") is None


def test_job_status_mixed_case_running_status_still_counts():
    ctx, counts = _job_ctx(_JobStatusSvc("RUNNING"))
    dl.JobStatus().run(dl.JobIdInput(job_id="j1"), ctx)
    assert counts == {"j1": 1}


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "canceled"])
def test_job_status_terminal_status_never_counts_or_stops(terminal):
    # These are the three terminal statuses services/datalab_job_service._TERMINAL
    # actually emits. The counter service is fetched ONLY on the non-terminal
    # path, so a terminal poll works even with no counter injected at all.
    svc = _JobStatusSvc(terminal)
    ctx = CallContext(services={"datalab_job_service": svc})
    out = dl.JobStatus().run(dl.JobIdInput(job_id="j1"), ctx).to_native()
    assert out["success"] is True and out["status"] == terminal
    assert "stop_polling" not in out and "instruction" not in out


def test_job_status_service_failure_is_a_typed_error():
    class _Boom:
        def status(self, job_id):
            raise RuntimeError("job store unreachable")

    ctx, _ = _job_ctx(_Boom())
    out = dl.JobStatus().run(dl.JobIdInput(job_id="j1"), ctx).to_native()
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
    ctx1 = agent._datalab_ctx_provider()
    ctx2 = agent._datalab_ctx_provider()

    # Same object identity across calls → mutations from tool A are visible to
    # tool B within the same turn (that is the whole point of the injection).
    assert ctx1.service("datalab_agg_timeout_tables") is ctx2.service("datalab_agg_timeout_tables")
    assert ctx1.service("datalab_job_poll_counts") is ctx2.service("datalab_job_poll_counts")
    # …and they ARE the agent's attributes, lazily created.
    assert ctx1.service("datalab_agg_timeout_tables") is agent._datalab_agg_timeout_tables
    assert ctx1.service("datalab_job_poll_counts") is agent._job_poll_counts


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
    assert job.description == "Poll the status of a Data Lab background job (e.g. a tiled search)."
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
