"""Unit tests for capabilities/viz.py (P1 family migration #6 — viz/FITS)."""

import threading

import pandas as pd
import pytest

from capabilities.viz import (
    CAPABILITIES,
    ComputeMomentMap,
    ExtractSpectrum,
    FitSpectralLine,
    GenerateFindingChart,
    GetSkyImage,
    OverlayArchiveImages,
    OverlayFitsImages,
    RenderFitsImage,
    mast_product_access_url,
    overlay_region_coordinates,
    pick_mast_fits_product,
)
from capabilities.base import CallContext


class _State:
    def __init__(self):
        self.last_run_result = None


def _ctx(state=None, **service_overrides):
    state = state or _State()

    def _set_lrr(v):
        state.last_run_result = v

    services = {"set_last_run_result": _set_lrr}
    services.update(service_overrides)
    return CallContext(services=services), state


def _run(cap, ctx, **kwargs):
    return cap.run(cap.InputModel(**kwargs), ctx).to_native()


# ─────────────────────────────────────────────────────────────────────────────
# self-free helpers
# ─────────────────────────────────────────────────────────────────────────────
def test_overlay_region_coordinates_explicit_and_parsed():
    assert overlay_region_coordinates("Custom", ra_deg=150.1, dec_deg=2.3) == (150.1, 2.3, "Custom")
    assert overlay_region_coordinates("150.1, 2.3") == (150.1, 2.3, "150.1, 2.3")


def test_overlay_region_coordinates_hudf_alias_and_out_of_range():
    assert overlay_region_coordinates("HUDF") == (53.1625, -27.7914, "HUDF")
    # RA >= 360 fails the explicit gate → returns None (no parse fallback here).
    assert overlay_region_coordinates("x", ra_deg=999.0, dec_deg=2.0) is None
    assert overlay_region_coordinates("") is None


def test_mast_product_access_url_prefers_http_then_datauri():
    assert mast_product_access_url({"access_url": "http://a/1.fits"}) == "http://a/1.fits"
    out = mast_product_access_url({"dataURI": "mast:JWST/x.fits"})
    assert out.startswith("https://mast.stsci.edu/api/v0.1/Download/file?uri=")
    assert mast_product_access_url({}) == ""


def test_pick_mast_fits_product_filters_and_sizes():
    df = pd.DataFrame([
        {"productFilename": "a.jpg", "productType": "PREVIEW", "size_mb": 1, "access_url": "http://a/a.jpg"},
        {"productFilename": "b.fits", "productType": "SCIENCE", "size_mb": 50, "access_url": "http://a/b.fits"},
        {"productFilename": "c.fits", "productType": "SCIENCE", "size_mb": 9999, "access_url": "http://a/c.fits"},
    ])
    picked = pick_mast_fits_product(df, max_product_mb=150.0)
    assert picked["productFilename"] == "b.fits"
    assert pick_mast_fits_product(pd.DataFrame(), max_product_mb=150.0) is None


# ─────────────────────────────────────────────────────────────────────────────
# get_sky_image
# ─────────────────────────────────────────────────────────────────────────────
class _FakeSkyview:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_image(self, **kw):
        self.calls.append(("get_image", kw))
        return dict(self.result)

    def generate_finding_chart(self, **kw):
        self.calls.append(("finding_chart", kw))
        return dict(self.result)


def test_get_sky_image_success_without_preview_sets_no_card():
    sky = _FakeSkyview({"success": True, "survey": "DSS2", "fits_path": "/f.fits",
                        "image_shape": [10, 10]})
    ctx, state = _ctx(skyview_client=sky)
    out = _run(GetSkyImage(), ctx, target_name="M87")
    assert out["success"] is True and out["survey"] == "DSS2"
    assert out["target"] == "M87"
    assert state.last_run_result is None  # no preview_path → no image card
    assert sky.calls[0][1]["save"] is True


def test_get_sky_image_failure_passes_through():
    ctx, state = _ctx(skyview_client=_FakeSkyview({"success": False, "error": "no survey"}))
    out = _run(GetSkyImage(), ctx, target_name="M87")
    assert out == {"success": False, "error": "no survey"}
    assert state.last_run_result is None


def test_get_sky_image_error_prefix_on_missing_client():
    ctx, _ = _ctx(skyview_client=None)
    out = _run(GetSkyImage(), ctx, target_name="M87")
    assert out["success"] is False and out["error"].startswith("Sky image fetch failed:")


def test_get_sky_image_preview_sets_card(tmp_path, monkeypatch):
    png = tmp_path / "preview.png"
    png.write_bytes(b"\x89PNG\r\n")
    sky = _FakeSkyview({"success": True, "survey": "2MASS", "preview_path": str(png),
                        "fits_path": "/f.fits", "image_shape": [10, 10]})
    # Force the data-URI fallback by making the /plots copy path raise.
    import services.plotting as plotting
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", "/nonexistent\0/plots", raising=False)
    ctx, state = _ctx(skyview_client=sky)
    out = _run(GetSkyImage(), ctx, target_name="M87")
    assert out["success"] is True
    assert state.last_run_result is not None
    assert state.last_run_result["tool_name"] == "get_sky_image"
    assert state.last_run_result["image_url"].startswith("data:image/png;base64,")


# ─────────────────────────────────────────────────────────────────────────────
# render / overlay / moment / spectrum / fit — all delegate to fits_service
# ─────────────────────────────────────────────────────────────────────────────
def _patch_fits(monkeypatch, name, result):
    import services.fits_service as fs
    calls = []

    def _fake(*a, **k):
        calls.append((a, k))
        return dict(result)

    monkeypatch.setattr(fs, name, _fake)
    return calls


def test_render_fits_image_sets_card_on_success(monkeypatch):
    _patch_fits(monkeypatch, "render_fits_image",
                {"success": True, "image_path": "/plots/x.png", "caption": "cap"})
    ctx, state = _ctx()
    out = _run(RenderFitsImage(), ctx, url="http://a/x.fits", title="t")
    assert out["success"] is True and out["image_path"] == "/plots/x.png"
    assert state.last_run_result == {"type": "image", "image_url": "/plots/x.png", "caption": "cap"}


def test_render_fits_image_failure_leaves_card_none(monkeypatch):
    _patch_fits(monkeypatch, "render_fits_image", {"success": False, "error": "bad fits"})
    ctx, state = _ctx()
    out = _run(RenderFitsImage(), ctx, url="http://a/x.fits")
    assert out == {"success": False, "error": "bad fits"}
    assert state.last_run_result is None


def test_overlay_fits_images_success(monkeypatch):
    _patch_fits(monkeypatch, "overlay_fits_images",
                {"success": True, "image_path": "/plots/o.png", "caption": "ov"})
    ctx, state = _ctx()
    out = _run(OverlayFitsImages(), ctx, base_url="http://a/b.fits", contour_url="http://a/c.fits")
    assert out["success"] is True
    assert state.last_run_result["image_url"] == "/plots/o.png"


def test_compute_moment_map_and_extract_spectrum_and_fit(monkeypatch):
    for cap_cls, fn_name in [
        (ComputeMomentMap, "compute_moment_map"),
        (ExtractSpectrum, "extract_spectrum"),
        (FitSpectralLine, "fit_spectral_line"),
    ]:
        _patch_fits(monkeypatch, fn_name,
                    {"success": True, "image_path": f"/plots/{fn_name}.png", "caption": "c"})
        ctx, state = _ctx()
        out = _run(cap_cls(), ctx, url="http://a/cube.fits")
        assert out["success"] is True, fn_name
        assert state.last_run_result["image_url"] == f"/plots/{fn_name}.png"


def test_generate_finding_chart_success(monkeypatch):
    sky = _FakeSkyview({"success": True, "image_path": "/plots/fc.png", "caption": "chart"})
    ctx, state = _ctx(skyview_client=sky)
    out = _run(GenerateFindingChart(), ctx, target="M87")
    assert out["success"] is True
    assert state.last_run_result["image_url"] == "/plots/fc.png"


# ─────────────────────────────────────────────────────────────────────────────
# overlay_archive_images end-to-end (region resolution + product picking)
# ─────────────────────────────────────────────────────────────────────────────
def test_overlay_archive_images_unresolvable_region():
    ctx, _ = _ctx(mast_client=object())
    out = _run(OverlayArchiveImages(), ctx, region="")
    assert out["success"] is False and "Could not resolve overlay region" in out["error"]


def test_overlay_archive_images_rejects_non_mast_alma():
    ctx, _ = _ctx(mast_client=object())
    out = _run(OverlayArchiveImages(), ctx, region="HUDF", base_archive="ESO")
    assert out["success"] is False
    assert "currently supports MAST/JWST" in out["error"]


def test_overlay_archive_images_no_mast_observations():
    class _M:
        def search_by_position(self, *a, **k):
            return pd.DataFrame()

    ctx, _ = _ctx(mast_client=_M())
    out = _run(OverlayArchiveImages(), ctx, region="HUDF")
    assert out["success"] is False and "No JWST MAST observations found" in out["error"]


# ─────────────────────────────────────────────────────────────────────────────
# agent wiring
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


def test_viz_registrations_keep_their_legacy_surface():
    agent = _wiring_agent()
    agent._register_tools()

    gi = agent.tool_registry.get_tool("get_sky_image")
    assert gi is not None and gi.parameters["required"] == []

    rf = agent.tool_registry.get_tool("render_fits_image")
    assert rf.parameters["required"] == ["url"]
    assert rf.parameters["properties"]["stretch"]["enum"] == ["sqrt", "log", "linear", "asinh"]
    assert rf.category == "analysis"

    oa = agent.tool_registry.get_tool("overlay_archive_images")
    assert oa.parameters["required"] == ["region"]


def test_viz_tool_fn_end_to_end_through_the_agent(monkeypatch):
    agent = _wiring_agent()

    import services.fits_service as fs
    monkeypatch.setattr(fs, "render_fits_image",
                        lambda *a, **k: {"success": True, "image_path": "/plots/x.png", "caption": "c"})
    out = agent._viz_tool_fn("render_fits_image")(url="http://a/x.fits")
    assert out["success"] is True
    assert agent.last_run_result["image_url"] == "/plots/x.png"


def test_viz_unknown_capability_name_raises():
    agent = _wiring_agent()
    with pytest.raises(KeyError):
        agent._viz_tool_fn("not_a_tool")


def test_every_viz_capability_is_registered():
    agent = _wiring_agent()
    agent._register_tools()
    for cap in CAPABILITIES:
        assert agent.tool_registry.get_tool(cap.name) is not None, cap.name


# ─────────────────────────────────────────────────────────────────────────────
# S21 gap-fill (masterplan): ctx-provider injection, remaining required-fields,
# typed service exceptions
# ─────────────────────────────────────────────────────────────────────────────
def test_viz_ctx_provider_injects_services_and_keeps_card():
    # _viz_ctx_provider must inject the 5 documented services and must NOT
    # clear last_run_result (the legacy inline tools never did).
    agent = _wiring_agent()
    agent.last_run_result = {"type": "image", "image_url": "/plots/keep.png"}
    agent.skyview_client = object()
    agent.mast_client = object()
    agent.search_service = object()
    agent.datalink_client = object()

    ctx = agent._viz_ctx_provider()
    for key in ("skyview_client", "mast_client", "search_service",
                "datalink_client", "set_last_run_result"):
        assert key in ctx.services, key
    assert ctx.services["skyview_client"] is agent.skyview_client
    # building the context must not touch the existing card
    assert agent.last_run_result == {"type": "image", "image_url": "/plots/keep.png"}
    # and the injected setter writes through to the agent
    ctx.services["set_last_run_result"]({"type": "image", "image_url": "/plots/new.png"})
    assert agent.last_run_result["image_url"] == "/plots/new.png"


def test_overlay_fits_images_required_fields():
    agent = _wiring_agent()
    agent._register_tools()
    of = agent.tool_registry.get_tool("overlay_fits_images")
    assert of.parameters["required"] == ["base_url", "contour_url"]


def test_ctx_service_typed_exceptions():
    # ctx.service("missing") raises KeyError; ctx.services.get returns None —
    # the two access styles the capabilities rely on for required vs optional
    # services.
    ctx, _ = _ctx()
    with pytest.raises(KeyError):
        ctx.service("definitely_not_a_service")
    assert ctx.services.get("definitely_not_a_service") is None
    # GetSkyImage calls ctx.service("set_last_run_result") FIRST — a context
    # missing it propagates the KeyError (typed contract, not a silent skip).
    from capabilities.base import CallContext
    bare = CallContext(services={})
    with pytest.raises(KeyError):
        GetSkyImage().run(GetSkyImage.InputModel(target_name="M87"), bare)
