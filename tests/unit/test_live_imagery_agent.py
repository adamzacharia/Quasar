import pandas as pd

from tests.integration.test_agent_archive_tools import _load_agent_module
from tests.unit.test_datalab_p0 import _make_agent
from tests.unit.test_mmu_hats_agent import load_api_main_module


class FakeHipsService:
    def cutout(self, ra, dec, fov_deg=0.25, survey="optical", width=512):
        return {"success": True, "image_base64": "abc123", "path": "/plots/hips_test.png", "survey_id": "CDS/P/DSS2/color"}

    def multiband_panel(self, ra, dec, fov_deg=0.25, surveys=None, title=None):
        return {"success": True, "image_base64": "abc123", "path": "/plots/panel_test.png"}

    def vlass_cutout(self, ra, dec, fov_deg=0.1):
        return {"success": True, "image_base64": "abc123", "path": "/plots/vlass_test.png"}


class FakeAlerceService:
    def cone_objects(self, ra, dec, radius_arcsec=120, max_rows=25):
        return {
            "success": True,
            "rows": [{"oid": "ZTF1", "ndet": 2, "meanra": ra, "meandec": dec, "firstmjd": 1, "lastmjd": 2}],
            "count": 1,
            "warnings": [],
            "provenance": {"radius_arcsec": radius_arcsec},
        }

    def plot_light_curve(self, oid):
        return {"success": True, "image_base64": "abc123", "path": "/plots/lc.png"}

    def stamp_triplet(self, oid, candid=None):
        return {"success": True, "image_base64": "abc123", "path": "/plots/stamps.png"}


class FakeSparclService:
    def find_spectra(self, ra, dec, radius_arcsec=60, data_release=None, limit=20):
        return {
            "success": True,
            "rows": [{"sparcl_id": "sid1", "ra": ra, "dec": dec, "distance_arcsec": 0.1, "redshift": 0.2, "spectype": "GALAXY", "data_release": "DESI-EDR"}],
            "count": 1,
            "warnings": [],
            "provenance": {"radius_arcsec": radius_arcsec},
        }

    def plot_spectrum(self, sparcl_id, mark_lines=True, smooth=0):
        return {"success": True, "image_base64": "abc123", "path": "/plots/spec.png"}


def test_live_imagery_tool_registration_prompt_and_status_labels():
    agent = _make_agent()
    agent._register_tools()

    expected = {
        "hips_cutout": "archive",
        "hips_multiband_panel": "archive",
        "vlass_cutout": "archive",
        "search_ztf_alerts": "archive",
        "ztf_light_curve": "analysis",
        "ztf_stamps": "analysis",
        "ned_sed_plot": "analysis",
        "sparcl_find_spectra": "archive",
        "sparcl_plot_spectrum": "analysis",
    }
    for name, category in expected.items():
        tool = agent.tool_registry.get_tool(name)
        assert tool is not None
        assert tool.category == category
        assert name in agent.tool_registry.categories[category]

    prompt = agent._build_system_prompt()
    assert "LIVE IMAGERY RULE" in prompt
    assert "hips_cutout" in prompt and "sparcl_plot_spectrum" in prompt
    assert "Fetching wise cutout" in agent._tool_status_label("hips_cutout", {"survey": "wise"})
    assert "Plotting ZTF light curve ZTF1" in agent._tool_status_label("ztf_light_curve", {"oid": "ZTF1"})


def test_hips_handler_attaches_image_and_strips_base64():
    _load_agent_module()
    agent = _make_agent()
    agent._hips_image_service_instance = FakeHipsService()

    out = agent._hips_cutout(ra=10.0, dec=-2.0, survey="optical")

    assert out["success"] is True
    assert "image_base64" not in out
    assert out["image_attached"] is True
    assert agent.last_run_result == {
        "type": "image",
        "image_url": "/plots/hips_test.png",
        "caption": "HiPS optical cutout: RA=10.00000, Dec=-2.00000",
        "meta": {"kind": "hips", "ra": 10.0, "dec": -2.0, "fov_deg": 0.25, "survey": "CDS/P/DSS2/color"},
    }


def test_sia_cutout_card_exposes_fits_download_url():
    """A single-band SIA cutout retains the tile's Data Lab access URL in
    provenance.source_url; the image card surfaces it as meta.fits_url so the
    frontend can offer a raw FITS download (T7.1)."""
    _load_agent_module()
    agent = _make_agent()
    src = "https://datalab.noirlab.edu/svc/cutout?id=abc&format=fits"
    result = {
        "success": True,
        "image_base64": "abc123",
        "path": "/plots/sia_test.png",
        "coverage_gap": False,
        "provenance": {"source_url": src, "ra": 10.0, "dec": -2.0},
    }

    out = agent._datalab_attach_image_result(result, "Data Lab g-band cutout")

    assert agent.last_run_result["type"] == "image"
    assert agent.last_run_result["meta"]["fits_url"] == src
    # The heavy base64 is stripped from the LLM-facing dict as before.
    assert "image_base64" not in out and out["image_attached"] is True


def test_card_fits_url_ignores_local_paths_and_plots():
    """_datalab_card_fits_url only returns http(s) source_urls; a coverage-gap or
    plot result (no provenance.source_url) yields no download link."""
    agent = _make_agent()
    assert agent._datalab_card_fits_url({"provenance": {"source_url": "/tmp/local.fits"}}) is None
    assert agent._datalab_card_fits_url({"provenance": {}}) is None
    assert agent._datalab_card_fits_url({}) is None

    # A plot result (no source_url) attaches a card with no fits_url key.
    plot = {"success": True, "image_base64": "x", "path": "/plots/scatter.png", "provenance": {"fov_deg": 1.0}}
    agent._datalab_attach_image_result(plot, "Catalog scatter")
    assert "fits_url" not in (agent.last_run_result.get("meta") or {})


def test_fits_url_merges_with_supplied_meta_without_clobbering():
    """When a caller passes its own meta (e.g. HiPS ra/dec/survey), the derived
    SIA fits_url merges in; an explicit fits_url already in meta is preserved."""
    agent = _make_agent()
    result = {
        "success": True,
        "image_base64": "abc",
        "path": "/plots/x.png",
        "provenance": {"source_url": "https://datalab.noirlab.edu/svc/cutout?id=1"},
    }
    agent._datalab_attach_image_result(result, "cut", meta={"kind": "hips", "ra": 1.0, "dec": 2.0})
    meta = agent.last_run_result["meta"]
    assert meta["kind"] == "hips" and meta["ra"] == 1.0
    assert meta["fits_url"] == "https://datalab.noirlab.edu/svc/cutout?id=1"

    agent2 = _make_agent()
    agent2._datalab_attach_image_result(
        result, "cut", meta={"fits_url": "https://datalab.noirlab.edu/preferred"}
    )
    assert agent2.last_run_result["meta"]["fits_url"] == "https://datalab.noirlab.edu/preferred"


def test_run_result_identity_helper_rejects_stale_result_objects():
    module = _load_agent_module()
    prior = {"type": "image", "image_url": "/plots/a.png"}

    assert module._run_result_is_new(None, prior) is True
    assert module._run_result_is_new(prior, prior) is False
    assert module._run_result_is_new(prior, None) is False
    prior["caption"] = "mutated in place"
    assert module._run_result_is_new(prior, prior) is False
    assert module._run_result_is_new(prior, {"type": "image", "image_url": "/plots/a.png"}) is True


def test_search_ztf_alerts_sets_external_catalog_table():
    _load_agent_module()
    agent = _make_agent()
    agent._alerce_client_instance = FakeAlerceService()

    out = agent._search_ztf_alerts(ra=10.0, dec=-2.0)

    assert out["success"] is True
    assert out["results_preview"][0]["oid"] == "ZTF1"
    assert agent.last_run_result["type"] == "data"
    assert agent.last_run_result["table_kind"] == "external_catalog"
    assert agent.last_run_result["tool_name"] == "search_ztf_alerts"


def test_sparcl_find_spectra_sets_external_catalog_table():
    _load_agent_module()
    agent = _make_agent()
    agent._sparcl_spectra_service_instance = FakeSparclService()

    out = agent._sparcl_find_spectra(ra=10.0, dec=-2.0)

    assert out["success"] is True
    assert out["results_preview"][0]["sparcl_id"] == "sid1"
    assert agent.last_run_result["table_kind"] == "external_catalog"
    assert agent.last_run_result["source"] == "NOIRLab SparCL spectra"


def test_external_catalog_data_card_columns_and_no_fits_estimate():
    api_main = load_api_main_module()
    df = pd.DataFrame([{"sparcl_id": "sid1", "ra": 10.0, "dec": -2.0, "redshift": 0.1, "spectype": "GALAXY"}])

    _, rich = api_main._build_data_card_event(
        {
            "type": "data",
            "data": df,
            "source": "NOIRLab SparCL spectra",
            "tool_name": "sparcl_find_spectra",
            "table_kind": "external_catalog",
        }
    )

    assert rich["tableKind"] == "external_catalog"
    assert "SparCL ID" in rich["columns"]
    assert rich["fitsEstimate"] is None
