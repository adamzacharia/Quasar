"""Unit tests for capabilities/archives.py (P1 family migration #4 — MAST/ESO/IRSA)."""

import threading

import pandas as pd
import pytest

from capabilities.archives import (
    CAPABILITIES,
    GetMastProducts,
    SearchEso,
    SearchIrsa,
    SearchMast,
    SearchMastByCriteria,
)
from capabilities.base import CallContext


class _State:
    def __init__(self):
        self.last_search_results = None
        self.last_run_result = None


def _ctx(state=None, **service_overrides):
    state = state or _State()

    def _set_lsr(v):
        state.last_search_results = v

    def _set_lrr(v):
        state.last_run_result = v

    services = {
        "get_last_search_results": lambda: state.last_search_results,
        "get_last_run_result": lambda: state.last_run_result,
        "set_last_search_results": _set_lsr,
        "set_last_run_result": _set_lrr,
    }
    services.update(service_overrides)
    return CallContext(services=services), state


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


_MAST_DF = pd.DataFrame([
    {"obsid": 101, "target_name": "Carina Nebula", "telescope": "JWST",
     "instrument_name": "NIRCAM", "project_code": "1345", "filters": "F200W"},
    {"obsid": 102, "target_name": "Carina Nebula", "telescope": "HST",
     "instrument_name": "WFC3", "project_code": "999", "filters": "F160W"},
])


class _Radius:
    def to(self, unit):
        assert unit == "arcmin"

        class _V:
            value = 0.5
        return _V()


class _FakeMast:
    def __init__(self, df=None, product_df=None):
        self.df = df if df is not None else _MAST_DF.copy()
        self.product_df = product_df if product_df is not None else pd.DataFrame()
        self.calls = []

    def search_by_target(self, target, mission, instrument, radius):
        self.calls.append(("target", target, mission, instrument, radius))
        return self.df.copy()

    def search_by_position(self, ra, dec, radius_arcmin, mission, instrument):
        self.calls.append(("position", ra, dec, radius_arcmin, mission, instrument))
        return self.df.copy()

    def _parse_radius(self, radius):
        self.calls.append(("parse_radius", radius))
        return _Radius()

    def search_by_criteria(self, **kw):
        self.calls.append(("criteria", kw))
        return self.df.copy()

    def get_product_list(self, df, productType, extension):
        self.calls.append(("products", len(df), productType, extension))
        return self.product_df.copy()


class _FakeEso:
    def __init__(self, df):
        self.df = df
        self.calls = []

    def search_by_target(self, target, instrument, radius_arcmin):
        self.calls.append(("target", target, instrument, radius_arcmin))
        return self.df.copy()

    def search_by_position(self, ra, dec, radius_arcmin, instrument):
        self.calls.append(("position", ra, dec, radius_arcmin, instrument))
        return self.df.copy()


class _FakeIrsa:
    def __init__(self, df):
        self.df = df
        self.calls = []

    def search_by_target(self, target, catalog, radius_arcsec):
        self.calls.append(("target", target, catalog, radius_arcsec))
        return self.df.copy()

    def search_by_position(self, ra, dec, radius_arcsec, catalog):
        self.calls.append(("position", ra, dec, radius_arcsec, catalog))
        return self.df.copy()


# ─────────────────────────────────────────────────────────────────────────────
# search_mast
# ─────────────────────────────────────────────────────────────────────────────
def test_search_mast_target_path_sets_state_and_summaries():
    mast = _FakeMast()
    ctx, state = _ctx(mast_client=mast)
    out = _run(SearchMast(), ctx, target_name="Carina Nebula", mission="JWST")

    assert out["success"] is True and out["total_results"] == 2
    assert out["missions"] == {"JWST": 1, "HST": 1}
    assert out["unique_targets"] == 1 and out["unique_programs"] == 2
    assert state.last_run_result["tool_name"] == "search_mast"
    assert state.last_run_result["filter_label"] == "MAST › Carina Nebula [JWST]"
    assert state.last_search_results is not None
    assert mast.calls[0] == ("target", "Carina Nebula", "JWST", None, "30s")


def test_search_mast_positional_path_parses_radius_via_client():
    mast = _FakeMast()
    ctx, _ = _ctx(mast_client=mast)
    out = _run(SearchMast(), ctx, ra=10.0, dec=-5.0, radius="1m")
    assert out["success"] is True
    assert ("parse_radius", "1m") in mast.calls
    assert mast.calls[-1] == ("position", 10.0, -5.0, 0.5, None, None)


def test_search_mast_requires_target_or_coordinates():
    ctx, _ = _ctx(mast_client=_FakeMast())
    out = _run(SearchMast(), ctx)
    assert out == {"success": False, "error": "Provide target_name or (ra, dec) coordinates."}


def test_search_mast_empty_df_note_keeps_card():
    mast = _FakeMast(df=pd.DataFrame())
    ctx, state = _ctx(mast_client=mast)
    out = _run(SearchMast(), ctx, target_name="X", mission="JWST", instrument="MIRI")
    assert out == {"success": True, "total_results": 0,
                   "note": "No MAST observations found for 'X' [JWST] [MIRI]"}
    assert state.last_run_result["tool_name"] == "search_mast"
    assert state.last_search_results is None  # empty result does NOT cache


def test_search_mast_error_is_typed_with_prefix():
    class _Boom(_FakeMast):
        def search_by_target(self, *a, **k):
            raise RuntimeError("portal down")

    ctx, _ = _ctx(mast_client=_Boom())
    out = _run(SearchMast(), ctx, target_name="X")
    assert out == {"success": False, "error": "MAST search failed: portal down"}


def test_search_mast_missing_client_is_caught_like_legacy():
    ctx, _ = _ctx(mast_client=None)
    out = _run(SearchMast(), ctx, target_name="X")
    assert out["success"] is False and out["error"].startswith("MAST search failed:")


# ─────────────────────────────────────────────────────────────────────────────
# search_mast_by_criteria
# ─────────────────────────────────────────────────────────────────────────────
def test_criteria_assembles_date_range_and_labels():
    mast = _FakeMast()
    ctx, state = _ctx(mast_client=mast)
    out = _run(SearchMastByCriteria(), ctx, mission="JWST", proposal_id="1345",
               target_name="Carina", start_date="2022-07-01", end_date="2023-07-01")

    assert out["success"] is True
    kw = mast.calls[0][1]
    assert kw["date_range"] == ("2022-07-01", "2023-07-01")
    assert state.last_run_result["filter_label"] == "MAST Criteria [JWST] Program 1345 › Carina"
    assert out["filters_used"] == {"F200W": 1, "F160W": 1}


def test_criteria_start_date_alone_means_no_range():
    mast = _FakeMast()
    ctx, _ = _ctx(mast_client=mast)
    _run(SearchMastByCriteria(), ctx, start_date="2022-07-01")
    assert mast.calls[0][1]["date_range"] is None


def test_criteria_empty_note_lists_criteria():
    mast = _FakeMast(df=pd.DataFrame())
    ctx, _ = _ctx(mast_client=mast)
    out = _run(SearchMastByCriteria(), ctx, mission="JWST", filters="F200W")
    assert out["note"] == "No MAST observations found for criteria: mission=JWST, filter=F200W"

    out2 = _run(SearchMastByCriteria(), _ctx(mast_client=_FakeMast(df=pd.DataFrame()))[0])
    assert out2["note"] == "No MAST observations found for criteria: unspecified"


# ─────────────────────────────────────────────────────────────────────────────
# get_mast_products
# ─────────────────────────────────────────────────────────────────────────────
def test_products_requires_prior_results_and_needs_no_client():
    ctx, _ = _ctx()  # NO mast_client injected: early returns are service-free
    out = _run(GetMastProducts(), ctx)
    assert out["success"] is False and "No MAST search results" in out["error"]

    state = _State()
    state.last_search_results = pd.DataFrame([{"target_name": "M87"}])
    ctx2, _ = _ctx(state)
    out2 = _run(GetMastProducts(), ctx2)
    assert out2["success"] is False and "not MAST observation results" in out2["error"]


def test_products_happy_path_caches_product_table():
    product_df = pd.DataFrame([{"productFilename": "a.fits", "productType": "SCIENCE"}])
    state = _State()
    state.last_search_results = _MAST_DF.copy()
    mast = _FakeMast(product_df=product_df)
    ctx, _ = _ctx(state, mast_client=mast)
    out = _run(GetMastProducts(), ctx, product_type="SCIENCE")

    assert out["success"] is True and out["total_products"] == 1
    assert out["product_types"] == {"SCIENCE": 1}
    assert state.last_run_result["tool_name"] == "get_mast_products"
    pd.testing.assert_frame_equal(state.last_search_results, product_df)


def test_products_empty_result_is_a_note():
    state = _State()
    state.last_search_results = _MAST_DF.copy()
    ctx, _ = _ctx(state, mast_client=_FakeMast(product_df=pd.DataFrame()))
    out = _run(GetMastProducts(), ctx)
    assert out == {"success": True, "total_products": 0, "note": "No data products found."}


# ─────────────────────────────────────────────────────────────────────────────
# search_eso_archive / search_irsa
# ─────────────────────────────────────────────────────────────────────────────
_ESO_DF = pd.DataFrame([
    {"target_name": "NGC 1068", "instrument_name": "MUSE", "dataproduct_type": "cube"},
])


def test_search_eso_target_path():
    eso = _FakeEso(_ESO_DF)
    ctx, state = _ctx(eso_client=eso)
    out = _run(SearchEso(), ctx, target_name="NGC 1068", instrument="MUSE")

    assert out["success"] is True and out["total_results"] == 1
    assert out["instruments"] == {"MUSE": 1} and out["data_types"] == {"cube": 1}
    assert state.last_run_result["tool_name"] == "search_eso_archive"
    assert state.last_run_result["filter_label"] == "ESO › NGC 1068 [MUSE]"
    assert eso.calls[0] == ("target", "NGC 1068", "MUSE", 1.0)


def test_search_eso_positional_and_no_args():
    eso = _FakeEso(_ESO_DF)
    ctx, _ = _ctx(eso_client=eso)
    out = _run(SearchEso(), ctx, ra=40.7, dec=-0.01, radius_arcmin=2.0)
    assert out["success"] is True
    assert eso.calls[0] == ("position", 40.7, -0.01, 2.0, None)

    out2 = _run(SearchEso(), _ctx(eso_client=eso)[0])
    assert out2 == {"success": False, "error": "Provide target_name or (ra, dec) coordinates."}


def test_search_eso_error_prefix():
    class _Boom(_FakeEso):
        def search_by_target(self, *a, **k):
            raise RuntimeError("TAP 500")

    ctx, _ = _ctx(eso_client=_Boom(_ESO_DF))
    out = _run(SearchEso(), ctx, target_name="X")
    assert out == {"success": False, "error": "ESO archive search failed: TAP 500"}


_IRSA_DF = pd.DataFrame([
    {"target_name": "M31", "w1mpro": 10.2},
])


def test_search_irsa_defaults_to_allwise_and_uppercases_label():
    irsa = _FakeIrsa(_IRSA_DF)
    ctx, state = _ctx(irsa_client=irsa)
    out = _run(SearchIrsa(), ctx, target_name="M31")

    assert out["success"] is True and out["catalog"] == "allwise"
    assert out["columns"] == ["target_name", "w1mpro"]
    assert state.last_run_result["filter_label"] == "IRSA › ALLWISE › M31"
    assert irsa.calls[0] == ("target", "M31", "allwise", 30.0)


def test_search_irsa_empty_note_and_no_args():
    irsa = _FakeIrsa(pd.DataFrame())
    ctx, state = _ctx(irsa_client=irsa)
    out = _run(SearchIrsa(), ctx, target_name="M31", catalog="2mass")
    assert out == {"success": True, "total_results": 0,
                   "note": "No IRSA sources found for 'M31' in catalog '2mass'"}
    assert state.last_run_result["tool_name"] == "search_irsa"

    out2 = _run(SearchIrsa(), _ctx(irsa_client=irsa)[0])
    assert out2 == {"success": False, "error": "Provide target_name or (ra, dec) coordinates."}


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


def test_archives_registrations_keep_their_legacy_surface():
    agent = _wiring_agent()
    agent._register_tools()

    sm = agent.tool_registry.get_tool("search_mast")
    assert sm is not None
    assert sm.parameters["required"] == []
    assert "JWST, HST, TESS, Kepler" in sm.description

    crit = agent.tool_registry.get_tool("search_mast_by_criteria")
    assert "F200W" in crit.parameters["properties"]["filters"]["description"]

    eso = agent.tool_registry.get_tool("search_eso_archive")
    assert "TAP/ADQL" in eso.description

    irsa = agent.tool_registry.get_tool("search_irsa")
    assert "'allwise' (default)" in irsa.parameters["properties"]["catalog"]["description"]


def test_archives_tool_fn_end_to_end_through_the_agent():
    agent = _wiring_agent()
    agent.mast_client = _FakeMast()

    out = agent._archives_tool_fn("search_mast")(target_name="Carina Nebula")
    assert out["success"] is True and out["total_results"] == 2
    assert agent.last_run_result["tool_name"] == "search_mast"
    assert agent.last_search_results is not None

    # get_mast_products consumes the cached observations through the getter:
    agent.mast_client = _FakeMast(product_df=pd.DataFrame([{"productType": "SCIENCE"}]))
    out2 = agent._archives_tool_fn("get_mast_products")()
    assert out2["success"] is True and out2["total_products"] == 1


def test_every_family_capability_is_registered():
    agent = _wiring_agent()
    agent._register_tools()
    for cap in CAPABILITIES:
        assert agent.tool_registry.get_tool(cap.name) is not None, cap.name


def test_unknown_capability_name_raises():
    agent = _wiring_agent()
    with pytest.raises(KeyError):
        agent._archives_tool_fn("not_a_tool")
