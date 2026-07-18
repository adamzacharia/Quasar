"""Unit tests for capabilities/alma.py (P1 family migration #2).

The capabilities must be byte-parity relocations of the legacy inline
``core/agent.py`` methods: same output dicts, same truthiness gates, same
state writes (through the injected accessors), same preserved legacy quirks
(the DEAD positional fallback, the ``is False`` identity check, the stringy
boolean hazards). Services are faked; no network.
"""

import threading

import pandas as pd
import pytest

import capabilities.alma as alma
from capabilities.alma import (
    CAPABILITIES,
    AdvancedSearch,
    CheckCoLines,
    CheckLineCoverage,
    DownloadAlmaData,
    DownloadData,
    FilterResults,
    FindAlmaLineCoverage,
    GetObservationDetails,
    MatchCrossArchiveSources,
    MatchPerseusProtostarsAlmaJwst,
    PlotAlmaResults,
    QueryAlmaScienceArchive,
    SearchAlmaCoInRedshiftRange,
    SearchAlmaWithKeywords,
    SearchByFrequency,
    SearchByPosition,
    SearchByTarget,
    SearchCatalog,
    SearchCadc,
    filter_by_scan_intent,
)
from capabilities.base import CallContext


# ─────────────────────────────────────────────────────────────────────────────
# Harness: a CallContext with recording fakes
# ─────────────────────────────────────────────────────────────────────────────
class _State:
    """Stands in for the agent's thread-local last-results state."""

    def __init__(self):
        self.last_search_results = None
        self.last_run_result = None
        self.console = []


def _ctx(state=None, **service_overrides):
    state = state or _State()
    services = {
        "console_log": state.console.append,
        "get_last_search_results": lambda: state.last_search_results,
        "get_last_run_result": lambda: state.last_run_result,
        "alma_tap_provenance": {"query": None, "url": None},
    }

    def _set_lsr(v):
        state.last_search_results = v

    def _set_lrr(v):
        state.last_run_result = v

    services["set_last_search_results"] = _set_lsr
    services["set_last_run_result"] = _set_lrr
    services.update(service_overrides)
    return CallContext(services=services), state


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


class _FakeSearchService:
    def __init__(self, df=None, by_target=None):
        self.df = df if df is not None else pd.DataFrame()
        self.by_target = by_target or {}
        self.calls = []

    def cone_search(self, ra, dec, radius, facility, max_results):
        self.calls.append(("cone_search", ra, dec, radius, facility, max_results))
        return self.df.copy()

    def search_by_target(self, name, facility=None, date_range=None, max_results=100, **kw):
        self.calls.append(("search_by_target", name))
        if name in self.by_target:
            out = self.by_target[name]
            if isinstance(out, Exception):
                raise out
            return out.copy()
        return self.df.copy()

    def search_by_frequency(self, lo, hi, facility, max_results):
        self.calls.append(("search_by_frequency", lo, hi))
        return self.df.copy()

    def search_alma_with_keywords(self, keywords):
        self.calls.append(("keywords", keywords))
        return self.df.copy()

    def advanced_search(self, query):
        self.calls.append(("advanced_search", query))
        return self.df.copy()

    def get_observation_details(self, obs_id):
        return {"obs_id": obs_id, "ok": True}

    def download_alma_data(self, df, dry_run=False):
        self.calls.append(("download", len(df), dry_run))
        return f"downloaded {len(df)} rows (dry_run={dry_run})"

    def check_line_coverage_on_last(self, df, freq, z, name):
        return df.head(1).copy()

    def check_co_lines_on_last(self, df, z):
        return df.head(1).copy()

    def search_catalog(self, catalog):
        self.calls.append(("search_catalog", catalog))
        return pd.DataFrame([{"n": len(v)} for v in catalog.values()][:1])

    def plot_alma_results(self, df, plot_type):
        self.calls.append(("plot", plot_type))
        return b"PNGBYTES"


_ALMA_DF = pd.DataFrame([
    {"target_name": "M87", "proposal_id": "2019.1.00001.S", "band_list": "6",
     "member_ous_uid": "uid://A/X1/X1", "access_url": "http://a/1",
     "scan_intent": "TARGET", "spatial_resolution": 0.5, "frequency": 230.0,
     "t_exptime": 100.0},
    {"target_name": "M87", "proposal_id": "2021.1.00002.S", "band_list": "7",
     "member_ous_uid": "uid://A/X2/X1", "access_url": "http://a/2",
     "scan_intent": "BANDPASS", "spatial_resolution": 0.05, "frequency": 345.0,
     "t_exptime": 900.0},
])


# ─────────────────────────────────────────────────────────────────────────────
# filter_by_scan_intent (relocated staticmethod)
# ─────────────────────────────────────────────────────────────────────────────
def test_filter_by_scan_intent_relocated_verbatim():
    filtered, label = filter_by_scan_intent(_ALMA_DF, "TARGET only")
    assert label == "Scan Intent TARGET"
    assert filtered["proposal_id"].tolist() == ["2019.1.00001.S"]

    same, no_label = filter_by_scan_intent(_ALMA_DF, None)
    assert no_label == "" and len(same) == 2


# ─────────────────────────────────────────────────────────────────────────────
# search_by_position
# ─────────────────────────────────────────────────────────────────────────────
def test_search_by_position_success_sets_state_and_summary():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, state = _ctx(search_service=svc)
    out = _run(SearchByPosition(), ctx, ra=187.7, dec=12.39)

    assert out["success"] is True
    assert out["total_results"] == 2
    assert out["top_mous_uids"] == ["uid://A/X1/X1", "uid://A/X2/X1"]
    assert out["top_project_codes"] == ["2019.1.00001.S", "2021.1.00002.S"]
    assert state.last_run_result["tool_name"] == "search_by_position"
    assert state.last_run_result["source"] == "ALMA"
    assert state.last_search_results is not None


def test_search_by_position_c3_archive_error_is_typed():
    df = pd.DataFrame()
    df.attrs["quasar_error"] = "TAP 503"
    ctx, _ = _ctx(search_service=_FakeSearchService(df))
    out = _run(SearchByPosition(), ctx, ra=1.0, dec=2.0, radius=0.3)

    assert out["success"] is False
    assert out["error"] == "TAP 503"
    assert out["ra"] == 1.0 and out["dec"] == 2.0 and out["radius_deg"] == 0.3
    assert "archive/service error" in out["note"]


def test_search_by_position_band_string_filter():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, state = _ctx(search_service=svc)
    out = _run(SearchByPosition(), ctx, ra=0.0, dec=0.0, band="7")

    assert out["success"] is True
    assert out["total_results"] == 1
    assert state.console and "[FILTER] Band 7" in state.console[0]


def test_search_by_position_facility_normalization():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, state = _ctx(search_service=svc)
    _run(SearchByPosition(), ctx, ra=0.0, dec=0.0, facility="EVLA")
    assert state.last_run_result["source"] == "VLA"


# ─────────────────────────────────────────────────────────────────────────────
# search_by_target
# ─────────────────────────────────────────────────────────────────────────────
def test_search_by_target_zero_min_values_mean_no_filter():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, _ = _ctx(search_service=svc)
    out = _run(SearchByTarget(), ctx, target_name="M87",
               min_resolution=0, min_freq_ghz=0, min_exp_s=0,
               max_resolution=500, max_freq_ghz=9999)

    assert out["success"] is True
    assert out["total_results"] == 2          # nothing filtered
    assert out["filters_applied"] == []


def test_search_by_target_resolution_filter_applies():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, state = _ctx(search_service=svc)
    out = _run(SearchByTarget(), ctx, target_name="M87", max_resolution=0.1)

    assert out["total_results"] == 1
    assert out["filters_applied"] == ['res ≤ 0.1"']
    assert "[FILTER] max_resolution 0.1" in state.console[0]
    assert state.last_run_result["filter_label"] == 'ALMA › M87 [res ≤ 0.1"]'


def test_search_by_target_multi_target_concat_and_c3_suberrors():
    ok = _ALMA_DF.head(1)
    svc = _FakeSearchService(by_target={"M87": ok, "Sz65": RuntimeError("archive down")})
    ctx, state = _ctx(search_service=svc)
    out = _run(SearchByTarget(), ctx, target_name="M87 and Sz65")

    assert out["success"] is True
    assert out["target"] == "M87"              # only the succeeding target remains in the label
    assert out["total_results"] == 1
    assert any("[MULTI] 'Sz65' failed" in line for line in state.console)


def test_search_by_target_all_targets_failing_is_a_typed_error():
    svc = _FakeSearchService(by_target={
        "M87": RuntimeError("archive down"),
        "Sz65": RuntimeError("archive down"),
    })
    # The empty-result fallback path calls resolve_target: return the legacy
    # ra_deg/dec_deg shape (which keeps the fallback DEAD, as in the monolith).
    ctx, _ = _ctx(search_service=svc,
                  resolve_target=lambda name: {"success": True, "ra_deg": 1.0, "dec_deg": 2.0})
    out = _run(SearchByTarget(), ctx, target_name="M87, Sz65")

    assert out["success"] is False
    assert "archive down" in out["error"]
    assert "archive/service error" in out["note"]


def test_search_by_target_per_target_band_mode():
    m87 = pd.DataFrame([{"target_name": "M87", "band_list": "6"},
                        {"target_name": "M87", "band_list": "7"}])
    sz = pd.DataFrame([{"target_name": "Sz65", "band_list": "7"}])
    svc = _FakeSearchService(by_target={"M87": m87, "Sz65": sz})
    ctx, state = _ctx(search_service=svc)
    out = _run(SearchByTarget(), ctx, target_name="M87 in band 6, Sz65 in band 7")

    assert out["success"] is True
    assert out["total_results"] == 2           # one band-6 M87 row + one band-7 Sz65 row
    assert out["target"] == "M87 + Sz65"
    assert any("[PER-TARGET] 'M87' band=6" in line for line in state.console)


def test_search_by_target_positional_fallback_stays_dead():
    # Preserved-verbatim legacy quirk: the gate reads resolved.get("ra") but the
    # resolver returns ra_deg/dec_deg → the wider cone search must NOT run.
    svc = _FakeSearchService(pd.DataFrame())
    resolver_calls = []

    def _resolver(name):
        resolver_calls.append(name)
        return {"success": True, "ra_deg": 187.7, "dec_deg": 12.39}

    ctx, _ = _ctx(search_service=svc, resolve_target=_resolver)
    out = _run(SearchByTarget(), ctx, target_name="M87")

    assert resolver_calls == ["M87"]
    assert not any(c[0] == "cone_search" for c in svc.calls)
    assert out == {"success": True, "total_results": 0, "target": "M87", "note": "No results found."}


# ─────────────────────────────────────────────────────────────────────────────
# search_by_frequency / search_cadc_archive
# ─────────────────────────────────────────────────────────────────────────────
def test_search_by_frequency_sets_card_and_summary():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, state = _ctx(search_service=svc)
    out = _run(SearchByFrequency(), ctx, min_freq_ghz=200.0, max_freq_ghz=300.0)

    assert out["success"] is True and out["total_results"] == 2
    assert state.last_run_result["filter_label"] == "ALMA › 200.0–300.0 GHz"
    assert state.last_run_result["tool_name"] == "search_by_frequency"


def test_search_cadc_requires_target_or_coordinates():
    ctx, _ = _ctx()
    out = _run(SearchCadc(), ctx)
    assert out == {"success": False, "error": "Provide target_name or (ra, dec) coordinates."}


# ─────────────────────────────────────────────────────────────────────────────
# keywords / advanced_search
# ─────────────────────────────────────────────────────────────────────────────
def test_search_alma_with_keywords_accepts_json_string():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, state = _ctx(search_service=svc)
    out = _run(SearchAlmaWithKeywords(), ctx, keywords='{"pi_name": "Smith"}')

    assert out["success"] is True and out["count"] == 2
    assert svc.calls[0] == ("keywords", {"pi_name": "Smith"})
    assert state.last_run_result["source"] == "Keywords: {'pi_name': 'Smith'}"


def test_advanced_search_rejects_datalab_schemas():
    ctx, _ = _ctx()  # must not need the search service at all
    out = _run(AdvancedSearch(), ctx, query="SELECT * FROM gaia_dr3.gaia_source")
    assert out["success"] is False
    assert "gaia_dr3" in out["error"]
    assert "datalab_sql_query" in out["hint"]


def test_advanced_search_requires_obscore():
    ctx, _ = _ctx()
    out = _run(AdvancedSearch(), ctx, query="SELECT * FROM somewhere_else")
    assert out["success"] is False
    assert "ivoa.obscore" in out["error"]


def test_advanced_search_happy_path():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, state = _ctx(search_service=svc)
    q = "SELECT TOP 5 * FROM ivoa.obscore"
    out = _run(AdvancedSearch(), ctx, query=q)
    assert out["success"] is True and out["count"] == 2
    assert state.last_run_result["source"] == f"SQL: {q}"


# ─────────────────────────────────────────────────────────────────────────────
# query_alma_science_archive
# ─────────────────────────────────────────────────────────────────────────────
def test_query_alma_unknown_type_and_missing_cycle():
    ctx, _ = _ctx(search_service=_FakeSearchService())
    assert _run(QueryAlmaScienceArchive(), ctx, query_type="nope") == \
        {"success": False, "error": "Unknown query_type: nope"}
    assert _run(QueryAlmaScienceArchive(), ctx, query_type="cycle_solar_projects") == \
        {"success": False, "error": "cycle is required"}


def test_query_alma_redshifted_lines_provenance_and_state(monkeypatch):
    tap_df = pd.DataFrame([{
        "proposal_id": "2023.1.00010.S", "target_name": "z galaxy",
        "frequency_support": "172.8..173.1GHz", "band_list": "5",
    }])
    captured = {}

    def _fake_tap(where, *, max_results=5000, order_by="proposal_id", ctx):
        captured["where"] = where
        prov = ctx.service("alma_tap_provenance")
        prov["query"] = f"SELECT TOP {max_results} ... WHERE {where}"
        prov["url"] = "https://almascience.nrao.edu/tap"
        return tap_df.copy()

    monkeypatch.setattr(alma, "_tap_obscore_dataframe", _fake_tap)
    ctx, state = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="redshifted_line_projects",
               redshift_min=1, redshift_max=2, rest_species="CO")

    assert out["success"] is True
    assert out["mode"] == "redshifted_line_projects"
    assert out["unique_projects"] == 1
    assert out["provenance"]["adql"].startswith("SELECT TOP")
    assert out["provenance"]["tap_url"] == "https://almascience.nrao.edu/tap"
    assert state.last_run_result["tool_name"] == "query_alma_science_archive"
    assert "frequency" in captured["where"]


def test_query_alma_include_adql_false_drops_the_query(monkeypatch):
    monkeypatch.setattr(alma, "_tap_obscore_dataframe",
                        lambda where, *, max_results=5000, order_by="proposal_id", ctx:
                        pd.DataFrame([{"proposal_id": "P1", "frequency_support": "1..2GHz"}]))
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="redshifted_line_projects",
               include_adql=False)
    assert out["provenance"]["adql"] is None


def test_query_alma_require_same_project_identity_gate(monkeypatch):
    # Legacy: `require_same_project is False` — a real False appends the
    # API-compat warning; the stringy "false" must NOT (no pydantic coercion).
    monkeypatch.setattr(alma, "_tap_obscore_dataframe",
                        lambda where, *, max_results=5000, order_by="proposal_id", ctx:
                        pd.DataFrame([{"proposal_id": "P1", "frequency_support": "1..2GHz"}]))
    ctx, _ = _ctx(search_service=_FakeSearchService())

    real_false = _run(QueryAlmaScienceArchive(), ctx, query_type="redshifted_line_projects",
                      require_same_project=False)
    stringy = _run(QueryAlmaScienceArchive(), ctx, query_type="redshifted_line_projects",
                   require_same_project="false")

    assert any("require_same_project=False" in w for w in real_false["warnings"])
    assert not any("require_same_project=False" in w for w in stringy["warnings"])


# ─────────────────────────────────────────────────────────────────────────────
# cross-archive matching
# ─────────────────────────────────────────────────────────────────────────────
class _FakeTapResult:
    def __init__(self, df):
        self._df = df

    def to_table(self):
        return self

    def to_pandas(self):
        return self._df.copy()


class _FakeAlminer:
    def __init__(self, df):
        self._df = df
        self.queries = []

    def _get_tap_service(self):
        outer = self

        class _Svc:
            def search(self, q):
                outer.queries.append(q)
                return _FakeTapResult(outer._df)

        return _Svc()

    def _standardize_columns(self, df):
        return df


class _FakeMast:
    def __init__(self, df):
        self._df = df
        self.calls = []

    def search_by_position(self, ra, dec, radius_arcmin=1.0, mission=None, max_results=500):
        self.calls.append((ra, dec, mission))
        return self._df.copy()


def _cross_match_services(alma_df, mast_df):
    svc = _FakeSearchService()
    svc.alminer_client = _FakeAlminer(alma_df)
    return {"search_service": svc, "mast_client": _FakeMast(mast_df)}


def test_match_cross_archive_inline_sources():
    source = {"source_name": "Custom Source", "ra": 150.1234, "dec": 2.3456}
    alma_df = pd.DataFrame([{"target_name": source["source_name"], "proposal_id": "2024.1.00099.S",
                             "s_ra": source["ra"], "s_dec": source["dec"]}])
    mast_df = pd.DataFrame([{"target_name": source["source_name"], "telescope": "HST",
                             "instrument_name": "WFC3", "project_code": "9999"}])
    ctx, state = _ctx(**_cross_match_services(alma_df, mast_df))
    out = _run(MatchCrossArchiveSources(), ctx,
               catalog_name="inline", sources=[source], archives=["ALMA", "HST"])

    assert out["success"] is True
    assert out["catalog_name"] == "inline_sources"
    assert out["matched_sources"] == 1
    assert state.last_run_result["tool_name"] == "match_cross_archive_sources"
    assert state.last_run_result["partial"] is False


def test_match_cross_archive_partial_when_alma_fails():
    source = {"source_name": "Custom Source", "ra": 150.1234, "dec": 2.3456}
    mast_df = pd.DataFrame([{"target_name": source["source_name"], "telescope": "HST",
                             "instrument_name": "WFC3", "project_code": "9999"}])
    services = _cross_match_services(pd.DataFrame(), mast_df)

    class _Boom:
        def _get_tap_service(self):
            raise RuntimeError("temporary TAP outage")

    services["search_service"].alminer_client = _Boom()
    ctx, state = _ctx(**services)
    out = _run(MatchCrossArchiveSources(), ctx,
               catalog_name="inline", sources=[source], archives=["ALMA", "HST"])

    assert out["success"] is True
    assert out["matched_sources"] == 1
    assert "ALMA TAP failed" in out["archive_errors"][0]
    assert state.last_run_result["partial"] is True


def test_match_cross_archive_bad_catalog_is_typed():
    ctx, _ = _ctx(**_cross_match_services(pd.DataFrame(), pd.DataFrame()))
    out = _run(MatchCrossArchiveSources(), ctx, catalog_name="no_such_catalog")
    assert out["success"] is False and out["error"]


def test_match_perseus_rewrites_the_same_card_dict():
    from services.cross_archive_matcher import PERSEUS_PROTOSTARS
    source = PERSEUS_PROTOSTARS[0]
    alma_df = pd.DataFrame([{"target_name": source["source_name"], "proposal_id": "2022.1.00001.S",
                             "s_ra": source["ra"], "s_dec": source["dec"]}])
    mast_df = pd.DataFrame([{"target_name": source["source_name"], "telescope": "JWST",
                             "instrument_name": "NIRCAM", "project_code": "1234"}])
    ctx, state = _ctx(**_cross_match_services(alma_df, mast_df))
    out = _run(MatchPerseusProtostarsAlmaJwst(), ctx, max_sources=1)

    assert out["success"] is True
    assert out["mode"] == "perseus_alma_jwst_cross_match"
    # The rewrite happens on the SAME dict object the agent state holds.
    assert state.last_run_result["tool_name"] == "match_perseus_protostars_alma_jwst"


# ─────────────────────────────────────────────────────────────────────────────
# plot_alma_results / download_alma_data
# ─────────────────────────────────────────────────────────────────────────────
def test_plot_alma_results_needs_prior_results():
    ctx, _ = _ctx()
    out = _run(PlotAlmaResults(), ctx, plot_type="sky")
    assert out == {"success": False, "error": "No results available to plot. Please run a search first."}


def test_plot_alma_results_overview_mode_sets_image_card():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, state = _ctx(search_service=svc)
    state.last_search_results = _ALMA_DF
    out = _run(PlotAlmaResults(), ctx, plot_type="sky")

    assert out == {"success": True, "message": "Generated sky plot successfully"}
    assert state.last_run_result == {"type": "image", "image_bytes": b"PNGBYTES",
                                     "caption": "ALMA Sky Plot"}


def test_plot_alma_results_publication_mode_delegates():
    class _Plotting:
        def __init__(self):
            self.kw = None

        def plot_alma_results(self, data_records=None, **kw):
            self.kw = (len(data_records), kw)
            return {"success": True, "plotted": True}

    plotting = _Plotting()
    ctx, state = _ctx(plotting_service=plotting)
    state.last_search_results = _ALMA_DF
    out = _run(PlotAlmaResults(), ctx, x_column="frequency", y_column="t_exptime", dark_mode=True)

    assert out == {"success": True, "plotted": True}
    assert plotting.kw == (2, {"x_column": "frequency", "y_column": "t_exptime", "dark_mode": True})


def test_download_alma_data_requires_results_then_downloads():
    svc = _FakeSearchService()
    ctx, state = _ctx(search_service=svc)
    assert _run(DownloadAlmaData(), ctx)["error"] == "No results available to download."

    state.last_search_results = _ALMA_DF
    out = _run(DownloadAlmaData(), ctx, dry_run=True)
    assert out == {"success": True, "message": "downloaded 2 rows (dry_run=True)"}


# ─────────────────────────────────────────────────────────────────────────────
# line coverage / catalogs / details / stub
# ─────────────────────────────────────────────────────────────────────────────
def test_check_line_coverage_and_co_lines_read_and_overwrite_state():
    svc = _FakeSearchService()
    ctx, state = _ctx(search_service=svc)
    assert "No previous search results" in _run(CheckLineCoverage(), ctx, line_freq_ghz=230.5)["error"]

    state.last_search_results = _ALMA_DF
    out = _run(CheckLineCoverage(), ctx, line_freq_ghz=230.5, line_name="CO(2-1)")
    assert out["success"] is True and out["count"] == 1
    assert state.last_run_result["source"] == "Line Check: CO(2-1) @ 230.5GHz"
    assert len(state.last_search_results) == 1     # overwritten, so it can be plotted

    state.last_search_results = _ALMA_DF
    out = _run(CheckCoLines(), ctx, z=0.5)
    assert out["success"] is True
    assert state.last_run_result["source"] == "CO Lines Check"


def test_find_alma_line_coverage_maps_projects(monkeypatch):
    import services.spectral_line_explorer as sle

    def _fake(**kwargs):
        return {
            "success": True,
            "projects": [{"proposal_id": "P1", "target_name": "M87", "covered_line_count": 1,
                          "all_lines_full": True, "minimum_edge_margin_mhz": 10.0,
                          "angular_separation_arcsec": 0.2, "best_angular_resolution_arcsec": 0.1,
                          "total_exposure_seconds": 100.0, "archive_url": "http://x"}],
            "selected_line": {"frequency_ghz": 230.538, "observed_frequency_ghz": 230.0},
            "target": {"redshift": 0.004, "redshift_source": "NED", "ra_deg": 187.7,
                       "dec_deg": 12.39, "coordinate_source": "SIMBAD"},
            "project_count": 1, "backend": "tap", "degraded": False,
            "warnings": [], "deep_link": "http://explorer",
        }

    monkeypatch.setattr(sle, "find_alma_line_coverage", _fake)
    ctx, state = _ctx()
    out = _run(FindAlmaLineCoverage(), ctx, target_name="M87", species="CO", transition="2-1")

    assert out["success"] is True
    assert out["rest_frequency_ghz"] == 230.538
    assert out["projects"][0]["proposal_id"] == "P1"
    assert out["line_explorer_url"] == "http://explorer"
    assert state.last_run_result["source"] == "Exact ALMA coverage: M87 CO(2-1)"


def test_search_catalog_pivots_objects():
    svc = _FakeSearchService()
    ctx, state = _ctx(search_service=svc)
    out = _run(SearchCatalog(), ctx, objects=[{"Name": "A", "RAJ2000": 1.0},
                                              {"Name": "B", "RAJ2000": 2.0}])
    assert out["success"] is True
    assert svc.calls[0] == ("search_catalog", {"Name": ["A", "B"], "RAJ2000": [1.0, 2.0]})
    assert state.last_run_result["source"] == "Catalog Search"


def test_get_observation_details_and_download_stub():
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = _run(GetObservationDetails(), ctx, obs_id="uid://A/B/C")
    assert out == {"success": True, "details": {"obs_id": "uid://A/B/C", "ok": True}}

    out = _run(DownloadData(), ctx, obs_id="X1")
    assert out["success"] is False and out["obs_id"] == "X1"
    assert "download_alma_data" in out["error"]


# ─────────────────────────────────────────────────────────────────────────────
# filter_results
# ─────────────────────────────────────────────────────────────────────────────
def test_filter_results_needs_prior_results_with_datalab_hint():
    ctx, _ = _ctx()
    out = _run(FilterResults(), ctx, column="resolution", operator="<", value=0.1)
    assert out["success"] is False
    assert "datalab_select_catalog_rows" in out["hint"]


def test_filter_results_alias_resolution_and_state_update():
    # Legacy alias list for 'resolution' is ['resolution', 's_resolution',
    # 'angular_resolution'] — 'spatial_resolution' is deliberately NOT an alias
    # (only search_by_target's res_col list knows it). Pin that verbatim.
    ctx, state = _ctx()
    df = _ALMA_DF.rename(columns={"spatial_resolution": "s_resolution"})
    state.last_search_results = df
    out = _run(FilterResults(), ctx, column="resolution", operator="<", value=0.1)

    assert out["success"] is True
    assert out["filter_applied"] == "s_resolution < 0.1"
    assert out["original_count"] == 2 and out["filtered_count"] == 1
    assert len(state.last_search_results) == 1
    assert state.last_run_result["source"] == "Filtered: s_resolution < 0.1"

    # …and the un-aliased spatial_resolution column is NOT matched (legacy).
    state.last_search_results = _ALMA_DF
    out = _run(FilterResults(), ctx, column="resolution", operator="<", value=0.1)
    assert out["success"] is False and "not found" in out["error"]


def test_filter_results_unknown_operator_and_missing_column():
    ctx, state = _ctx()
    state.last_search_results = _ALMA_DF
    assert _run(FilterResults(), ctx, column="frequency", operator="~=", value=1)["error"] == \
        "Unknown operator: ~="
    out = _run(FilterResults(), ctx, column="no_such", operator="<", value=1)
    assert out["success"] is False and "not found" in out["error"]


# ─────────────────────────────────────────────────────────────────────────────
# analyze_uv_coverage (C1 reflection)
# ─────────────────────────────────────────────────────────────────────────────
def test_analyze_uv_coverage_reflects_service_failure():
    from capabilities.alma import AnalyzeUvCoverage

    class _Analysis:
        def analyze_uv_coverage(self, path):
            return {"success": False, "message": "CASA unavailable"}

    ctx, state = _ctx(analysis_service=_Analysis())
    out = _run(AnalyzeUvCoverage(), ctx, ms_path="/x.ms")
    assert out["success"] is False and out["error"] == "CASA unavailable"
    assert state.last_run_result["type"] == "analysis"


# ─────────────────────────────────────────────────────────────────────────────
# agent wiring (real module surface)
# ─────────────────────────────────────────────────────────────────────────────
def _wiring_agent():
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
    return agent


def test_alma_registrations_keep_their_legacy_surface():
    # The migration swapped only `function=`; description + parameters schema
    # (the LLM-facing surface) must be identical to the inline versions.
    agent = _wiring_agent()
    agent._register_tools()

    pos = agent.tool_registry.get_tool("search_by_position")
    assert pos is not None
    assert pos.parameters["required"] == ["ra", "dec"]
    assert pos.parameters["properties"]["facility"]["enum"] == ["VLA", "VLBA", "ALMA", "GBT"]

    tgt = agent.tool_registry.get_tool("search_by_target")
    assert tgt.parameters["required"] == ["target_name"]
    assert "CRITICAL: ONLY pass optional filter parameters" in tgt.description

    q = agent.tool_registry.get_tool("query_alma_science_archive")
    assert q.category == "archive"
    assert "cycle_solar_projects" in q.parameters["properties"]["query_type"]["enum"]

    flt = agent.tool_registry.get_tool("filter_results")
    assert flt.parameters["properties"]["operator"]["enum"] == ["<", ">", "<=", ">=", "==", "!="]


def test_alma_ctx_provider_wires_the_agents_tls_state():
    agent = _wiring_agent()
    agent.search_service = _FakeSearchService(_ALMA_DF)

    ctx = agent._alma_ctx_provider()
    ctx.service("set_last_search_results")(_ALMA_DF)
    ctx.service("set_last_run_result")({"type": "data", "data": _ALMA_DF})

    # The setters write the agent's thread-local properties…
    assert agent.last_search_results is _ALMA_DF
    assert agent.last_run_result["type"] == "data"
    # …and the getters read them back.
    assert ctx.service("get_last_search_results")() is _ALMA_DF
    # The tap-provenance dict is agent-owned and shared across calls.
    assert ctx.service("alma_tap_provenance") is agent._alma_ctx_provider().service("alma_tap_provenance")


def test_alma_tool_fn_end_to_end_through_the_agent():
    agent = _wiring_agent()
    agent.search_service = _FakeSearchService(_ALMA_DF)

    fn = agent._alma_tool_fn("search_by_position", log_name="_search_by_position")
    out = fn(ra=187.7, dec=12.39)

    assert out["success"] is True and out["total_results"] == 2
    assert agent.last_run_result["tool_name"] == "search_by_position"
    assert agent.last_search_results is not None


def test_every_family_capability_is_registered():
    agent = _wiring_agent()
    agent._register_tools()
    for cap in CAPABILITIES:
        assert agent.tool_registry.get_tool(cap.name) is not None, cap.name


def test_unknown_capability_name_raises():
    agent = _wiring_agent()
    with pytest.raises(KeyError):
        agent._alma_tool_fn("not_a_tool")


# ─────────────────────────────────────────────────────────────────────────────
# Guard-review coverage (CX-01/CX-02/CX-04 + missing execution paths)
# ─────────────────────────────────────────────────────────────────────────────
def test_null_query_reaches_the_legacy_obscore_error():
    # CX-01: an explicit JSON null must NOT die in pydantic — the legacy body
    # null-coerced (str(query or "")) and returned its SPECIFIC error + hint.
    # cap.execute() runs the same InputModel validation the adapter does.
    ctx, _ = _ctx()
    out = AdvancedSearch().execute({"query": None}, ctx).to_native()
    assert out["success"] is False
    assert "ivoa.obscore" in out["error"]
    assert "datalab_sql_query" in out["hint"]


def test_null_query_type_reaches_the_legacy_unknown_error():
    ctx, _ = _ctx()
    out = QueryAlmaScienceArchive().execute({"query_type": None}, ctx).to_native()
    assert out == {"success": False, "error": "Unknown query_type: "}


def test_null_filter_column_keeps_the_legacy_control_flow():
    # With no prior results the no-results guard fires FIRST even for nulls…
    ctx, state = _ctx()
    out = FilterResults().execute({"column": None, "operator": None, "value": None}, ctx).to_native()
    assert out["success"] is False
    assert "No ALMA/archive search results" in out["error"]

    # …and WITH results, a null column hits .lower() inside the legacy try.
    state.last_search_results = _ALMA_DF
    out = FilterResults().execute({"column": None, "operator": "<", "value": 1}, ctx).to_native()
    assert out["success"] is False
    assert out["error"].startswith("Filter failed:")


def test_null_obs_id_flows_to_the_service():
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = GetObservationDetails().execute({"obs_id": None}, ctx).to_native()
    # The fake echoes it back — the point is that validation did NOT reject it.
    assert out == {"success": True, "details": {"obs_id": None, "ok": True}}


def test_null_ra_dec_flow_into_the_legacy_try():
    svc = _FakeSearchService(_ALMA_DF)
    ctx, _ = _ctx(search_service=svc)
    out = SearchByPosition().execute({"ra": None, "dec": None}, ctx).to_native()
    # The fake cone_search accepts None (a real archive error would surface as
    # the legacy {"success": False, "error": ...} dict) — no pydantic rejection.
    assert out["success"] is True
    assert svc.calls[0][:3] == ("cone_search", None, None)


def test_query_alma_band_stays_raw_string_zero(monkeypatch):
    # CX-02: band="0" is truthy in legacy — `band or 6` must NOT flip to 6.
    captured = {}
    monkeypatch.setattr(alma, "_tap_obscore_dataframe",
                        lambda where, *, max_results=5000, order_by="proposal_id", ctx:
                        (captured.__setitem__("where", where),
                         pd.DataFrame([{"proposal_id": "P1", "frequency_support": "1..2GHz",
                                        "band_list": "0"}]))[1])
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="line_set_projects", band="0")

    # CAP-06: exact-token match on the space-delimited band_list — the old
    # substring LIKE let band=1 also match Band 10. band="0" must still stay
    # the raw string (not flip to 6).
    assert "band_list = '0'" in captured["where"]
    assert "band_list LIKE '% 0 %'" in captured["where"]
    assert out["source"].startswith("ALMA Band 0 projects")


def test_query_alma_early_returns_need_no_search_service():
    # CX-04: unknown query_type / missing cycle must return their typed errors
    # WITHOUT the search service being injected at all (legacy touched
    # self.search_service only inside the branches that queried).
    ctx, _ = _ctx()  # deliberately NO search_service
    assert _run(QueryAlmaScienceArchive(), ctx, query_type="nope")["error"] == "Unknown query_type: nope"
    assert _run(QueryAlmaScienceArchive(), ctx, query_type="cycle_solar_projects")["error"] == "cycle is required"


def _fake_pyvo(df, monkeypatch):
    import sys
    import types

    class _Res:
        def to_table(self):
            return self

        def to_pandas(self):
            return df.copy()

    class _Tap:
        def __init__(self, url):
            self.url = url

        def search(self, q):
            _fake_pyvo.last_query = q
            return _Res()

    fake = types.ModuleType("pyvo")
    fake.dal = types.SimpleNamespace(TAPService=_Tap)
    monkeypatch.setitem(sys.modules, "pyvo", fake)


def test_co_redshift_range_empty_archive_reports_freq_windows(monkeypatch):
    _fake_pyvo(pd.DataFrame(), monkeypatch)
    ctx, _ = _ctx()
    out = _run(SearchAlmaCoInRedshiftRange(), ctx, z_min=1.0, z_max=2.0)

    assert out["success"] is True and out["count"] == 0
    assert out["co_obs_freq_ranges_ghz"]["CO(2-1)"] == {
        "nu_min_ghz": round(230.538 / 3.0, 2), "nu_max_ghz": round(230.538 / 2.0, 2)}


def test_co_redshift_range_annotates_transitions_and_state(monkeypatch):
    df = pd.DataFrame([{
        "target_name": "zgal", "proposal_id": "2023.1.00001.S",
        "frequency": 100.0, "bandwidth": 8.0e9,     # covers CO(2-1) at z=1-2 (76.8–115.3)
        "scientific_category": "Galaxy evolution", "science_keyword": "",
        "s_ra": 1.0, "s_dec": 2.0, "t_exptime": 10.0, "s_resolution": 0.5,
    }])
    _fake_pyvo(df, monkeypatch)
    ctx, state = _ctx()
    out = _run(SearchAlmaCoInRedshiftRange(), ctx, z_min=1.0, z_max=2.0)

    assert out["success"] is True and out["count"] == 1
    assert "CO(2-1)" in out["results"][0]["CO_transitions_covered"]
    assert state.last_run_result["source"] == "CO z=1.0-2.0"
    assert state.last_search_results is not None


def test_datalab_provider_survives_a_failing_unrelated_service():
    # CX-03: a broken SVO/image-service constructor must not block tools that
    # never touch it (tiled_search/export_notebook) — the provider injects
    # None and CallContext.service() raises only if a tool actually asks.
    agent = _wiring_agent()

    def _boom():
        raise RuntimeError("malformed SVO TTL")

    agent._get_svo_fps_client = _boom
    agent._get_datalab_image_service = _boom
    ctx = agent._datalab_ctx_provider()

    assert ctx.services["svo_fps_client"] is None
    assert ctx.services["datalab_image_service"] is None
    # …and the services that DID construct are still live.
    assert ctx.services["datalab_job_service"] is not None
    with pytest.raises(KeyError):
        ctx.service("svo_fps_client")


# ─────────────────────────────────────────────────────────────────────────────
# R2 — sensitivity_search + data_publications templates
# ─────────────────────────────────────────────────────────────────────────────
def test_query_alma_sensitivity_search_requires_threshold():
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="sensitivity_search")
    assert out["success"] is False and "sensitivity_mjy" in out["error"]


def test_query_alma_sensitivity_search_happy_path(monkeypatch):
    tap_df = pd.DataFrame([
        {"proposal_id": "2019.1.00001.S", "target_name": "deep field",
         "band_list": "6", "sensitivity_10kms": 0.05},
        {"proposal_id": "2019.1.00002.S", "target_name": "other",
         "band_list": "6", "sensitivity_10kms": 0.09},
    ])
    captured = {}

    def _fake_tap(where, *, max_results=5000, order_by="proposal_id",
                  extra_columns=(), ctx):
        captured["where"] = where
        captured["order_by"] = order_by
        captured["extra_columns"] = extra_columns
        prov = ctx.service("alma_tap_provenance")
        prov["query"] = f"SELECT ... WHERE {where}"
        prov["url"] = "https://almascience.nrao.edu/tap"
        return tap_df.copy()

    monkeypatch.setattr(alma, "_tap_obscore_dataframe", _fake_tap)
    ctx, state = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="sensitivity_search",
               sensitivity_mjy=0.1, band=6)

    assert out["success"] is True and out["mode"] == "sensitivity_search"
    assert "sensitivity_10kms <= 0.1" in captured["where"]
    assert "band_list = '6'" in captured["where"]
    assert captured["order_by"] == "sensitivity_10kms"
    assert "sensitivity_10kms" in captured["extra_columns"]
    # Best (smallest) sensitivity first.
    assert out["results"][0]["proposal_id"] == "2019.1.00001.S"
    assert out["results"][0]["best_sensitivity_10kms_mjy_beam"] == 0.05
    assert any("estimated achieved rms" in w for w in out["warnings"])
    assert state.last_run_result["tool_name"] == "query_alma_science_archive"


def test_query_alma_sensitivity_search_continuum_column(monkeypatch):
    captured = {}

    def _fake_tap(where, *, max_results=5000, order_by="proposal_id",
                  extra_columns=(), ctx):
        captured["where"] = where
        return pd.DataFrame()

    monkeypatch.setattr(alma, "_tap_obscore_dataframe", _fake_tap)
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="sensitivity_search",
               sensitivity_mjy=1.0, continuum=True)
    assert out["success"] is True
    assert "cont_sensitivity_bandwidth <= 1" in captured["where"]


def test_query_alma_data_publications_requires_identifier():
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="data_publications")
    assert out["success"] is False and "identifier is required" in out["error"]


def test_query_alma_data_publications_unrecognized_identifier():
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="data_publications",
               identifier="HL Tau")
    assert out["success"] is False and "Unrecognized identifier" in out["error"]


def test_query_alma_data_publications_forward_join(monkeypatch):
    tap_df = pd.DataFrame([
        {"proposal_id": "2019.1.01528.S", "target_name": "COSMOS",
         "band_list": "6", "member_ous_uid": "uid://A001/X1465/X9c6",
         "bib_reference": "2024A&A...685A...1A 2024A&A...688A..55M",
         "pub_title": "blob", "publication_year": 2024, "first_author": "A"},
    ])
    captured = {}

    def _fake_tap(where, *, max_results=5000, order_by="proposal_id",
                  extra_columns=(), ctx):
        captured["where"] = where
        captured["extra_columns"] = extra_columns
        return tap_df.copy()

    monkeypatch.setattr(alma, "_tap_obscore_dataframe", _fake_tap)
    ctx, state = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="data_publications",
               identifier="2019.1.01528.S")

    assert out["success"] is True and out["mode"] == "data_publications"
    assert captured["where"] == "proposal_id = '2019.1.01528.S'"
    assert "bib_reference" in captured["extra_columns"]
    assert out["n_publications"] == 2
    assert [p["bibcode"] for p in out["publications"]] == \
        ["2024A&A...685A...1A", "2024A&A...688A..55M"]
    assert out["results"][0]["n_publications"] == 2
    assert state.last_run_result["type"] == "data"


def test_query_alma_data_publications_reverse_bibcode(monkeypatch):
    captured = {}

    def _fake_tap(where, *, max_results=5000, order_by="proposal_id",
                  extra_columns=(), ctx):
        captured["where"] = where
        return pd.DataFrame([{
            "proposal_id": "2017.1.00001.S", "target_name": "HL Tau",
            "band_list": "6", "member_ous_uid": "uid://A/X/Y",
            "bib_reference": "2018ApJ...869L..41A",
        }])

    monkeypatch.setattr(alma, "_tap_obscore_dataframe", _fake_tap)
    ctx, _ = _ctx(search_service=_FakeSearchService())
    out = _run(QueryAlmaScienceArchive(), ctx, query_type="data_publications",
               identifier="2018ApJ...869L..41A")
    assert out["success"] is True
    assert captured["where"] == "bib_reference LIKE '%2018ApJ...869L..41A%'"
    assert "archived data used by publication" in out["source"]


# ─────────────────────────────────────────────────────────────────────────────
# R1 — resolver rule: flag-gated positional fallback (default OFF)
# ─────────────────────────────────────────────────────────────────────────────
def test_positional_fallback_enabled_by_flag_runs_cone(monkeypatch):
    monkeypatch.setenv("QUASAR_ALMA_POSITIONAL_FALLBACK", "1")
    cone_df = pd.DataFrame([{"target_name": "M87", "band_list": "6"}])
    svc = _FakeSearchService(pd.DataFrame())
    svc.by_target = {"M87": pd.DataFrame()}

    def _cone(ra, dec, radius, facility, max_results):
        svc.calls.append(("cone_search", ra, dec, radius))
        return cone_df.copy()

    svc.cone_search = _cone
    ctx, state = _ctx(search_service=svc,
                      resolve_target=lambda name: {"success": True, "ra_deg": 187.7, "dec_deg": 12.39})
    out = _run(SearchByTarget(), ctx, target_name="M87")

    cone_calls = [c for c in svc.calls if c[0] == "cone_search"]
    assert cone_calls == [("cone_search", 187.7, 12.39, 0.14)]
    assert out["success"] is True and out["total_results"] == 1


def test_positional_fallback_flag_off_stays_dead(monkeypatch):
    monkeypatch.setenv("QUASAR_ALMA_POSITIONAL_FALLBACK", "0")
    svc = _FakeSearchService(pd.DataFrame())
    ctx, _ = _ctx(search_service=svc,
                  resolve_target=lambda name: {"success": True, "ra_deg": 187.7, "dec_deg": 12.39})
    out = _run(SearchByTarget(), ctx, target_name="M87")
    assert not any(c[0] == "cone_search" for c in svc.calls)
    assert out["total_results"] == 0


def test_positional_fallback_flag_on_legacy_ra_keys_also_work(monkeypatch):
    # A resolver returning the plain ra/dec shape must also feed the cone.
    monkeypatch.setenv("QUASAR_ALMA_POSITIONAL_FALLBACK", "true")
    svc = _FakeSearchService(pd.DataFrame())
    captured = []

    def _cone(ra, dec, radius, facility, max_results):
        captured.append((ra, dec))
        return pd.DataFrame()

    svc.cone_search = _cone
    ctx, _ = _ctx(search_service=svc,
                  resolve_target=lambda name: {"success": True, "ra": 10.0, "dec": -5.0})
    _run(SearchByTarget(), ctx, target_name="NGC 253")
    assert captured == [(10.0, -5.0)]


def test_cx01_fallback_skipped_when_date_range_present(monkeypatch):
    # cone_search cannot honor a date_range — the enabled fallback must skip
    # rather than silently widen into out-of-period observations.
    monkeypatch.setenv("QUASAR_ALMA_POSITIONAL_FALLBACK", "1")
    svc = _FakeSearchService(pd.DataFrame())
    ctx, state = _ctx(search_service=svc,
                      resolve_target=lambda name: {"success": True, "ra_deg": 187.7, "dec_deg": 12.39})
    out = _run(SearchByTarget(), ctx, target_name="M87", date_range="2019-01-01,2019-12-31")
    assert not any(c[0] == "cone_search" for c in svc.calls)
    assert out["total_results"] == 0
    assert any("date_range cannot be applied" in line for line in state.console)


def test_cx02_missing_dec_does_not_crash_fallback(monkeypatch):
    monkeypatch.setenv("QUASAR_ALMA_POSITIONAL_FALLBACK", "1")
    svc = _FakeSearchService(pd.DataFrame())
    ctx, _ = _ctx(search_service=svc,
                  resolve_target=lambda name: {"success": True, "ra_deg": 187.7})
    out = _run(SearchByTarget(), ctx, target_name="M87")
    # No cone attempt, no TypeError — clean empty result.
    assert not any(c[0] == "cone_search" for c in svc.calls)
    assert out == {"success": True, "total_results": 0, "target": "M87", "note": "No results found."}
