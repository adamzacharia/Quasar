"""Tool results must say when a call's scope or shape does not match its intent.

Two live losses on the AstroDataBench data-discovery tasks (2026-09-17):

* NGC1068 "deepest g/r/i stacks in the NOIRLab Science Archive": on 3 of 4
  completed trials the model volunteered a survey scope (catalog='ls_dr9' or
  service='coadd_all') and reported DES tiles instead of the archive-wide
  Stack images — a different product family, silently.
* HH212 (latent, fix7 t2): an enforced re-sample put every constraint into the
  free-text ``topic_filter`` of ``high_resolution_band_data``, which ignores it,
  and hand-filtered the unfiltered MOUS list to 11 instead of 14.

Nothing here keys on a target or a benchmark: the SIA tool discloses what the
archive-wide collection holds whenever a deepest-per-band request is narrowed
to a survey scope, and the ALMA tool says when ``topic_filter`` is inert.
"""
from types import SimpleNamespace as NS

from capabilities.alma import QueryAlmaScienceArchive
from capabilities.datalab import SiaSearch, SiaSearchInput
from tests.unit.test_alma_capability import _ctx as _alma_ctx, _run, _FakeAlminer, _FakeSearchService
from tests.unit.test_datalab_capability import _img_ctx
from tests.unit.test_discovery_recovery import _inventory_rows, _observations

NSA = "https://datalab.noirlab.edu/sia/nsa"
COADD = "https://datalab.noirlab.edu/sia/coadd_all"


def _survey_rows():
    def row(band, exptime, url):
        return {"obs_bandpass": band, "exptime": exptime, "proctype": "Stack", "prodtype": "image",
                "access_url": url, "obs_collection": "survey_dr1", "instrument": "Cam"}
    return [row("g", "450", "survey-g"), row("r", "450", "survey-r"), row("z", "450", "survey-z")]


class _TwoCollectionService:
    """Archive-wide endpoint holds the per-program stacks; any other scope returns survey tiles."""

    def __init__(self, fail_wide=False):
        self.calls = []
        self.fail_wide = fail_wide

    def search(self, ra, dec, fov, *, catalog=None, endpoint=None):
        self.calls.append({"catalog": catalog, "endpoint": endpoint})
        if endpoint and endpoint.rstrip("/") == NSA:
            if self.fail_wide:
                raise RuntimeError("nsa endpoint 502")
            return {"rows": _inventory_rows(), "coverage_gap": False, "used_endpoint": NSA}
        return {"rows": _survey_rows(), "coverage_gap": False, "used_endpoint": endpoint or "survey-endpoint"}


def test_deepest_request_narrowed_to_a_survey_catalog_discloses_the_archive_wide_stacks():
    service = _TwoCollectionService()
    out = SiaSearch().run(SiaSearchInput(ra=25, dec=-10, catalog="survey_dr1", deepest_per_band=True,
                                         bands=["g", "r"]), _img_ctx(service)).to_native()
    assert [r["access_url"] for r in out["deepest_images"]] == ["survey-g", "survey-r"]  # the requested scope is still answered
    assert [c["endpoint"] for c in service.calls[-1:]] == [NSA]  # exactly one advisory archive-wide query
    cmp = out["archive_wide_comparison"]
    assert cmp["collection"] == "nsa" and cmp["rows_total"] == 8
    assert [r["access_url"] for r in cmp["deepest_images"]] == ["u4", "u5"]
    assert set(cmp["stack_images_per_band"]) == {"g", "r"}  # limited to the requested bands
    warning = "\n".join(out["warnings"])
    assert "catalog='survey_dr1' narrowed this search to survey coadd/tile products" in warning
    assert "g: 270 s, r: 270 s" in warning and "archive_wide_comparison" in warning


def test_deepest_request_on_the_coadd_collection_is_compared_too():
    service = _TwoCollectionService()
    out = SiaSearch().run(SiaSearchInput(ra=25, dec=-10, service="coadd_all", deepest_per_band=True,
                                         bands=["z"]), _img_ctx(service)).to_native()
    assert out["collection"] == "coadd_all"
    assert [r["access_url"] for r in out["archive_wide_comparison"]["deepest_images"]] == ["u8"]
    assert any("service='coadd_all' narrowed" in w for w in out["warnings"])


def test_archive_wide_default_and_inventory_calls_do_not_pay_for_a_comparison():
    service = _TwoCollectionService()
    SiaSearch().run(SiaSearchInput(ra=25, dec=-10, deepest_per_band=True, bands=["g"]), _img_ctx(service)).to_native()
    assert len(service.calls) == 1  # the default IS the archive-wide collection: nothing to compare
    service = _TwoCollectionService()
    out = SiaSearch().run(SiaSearchInput(ra=25, dec=-10, catalog="survey_dr1"), _img_ctx(service)).to_native()
    assert len(service.calls) == 1 and "archive_wide_comparison" not in out  # plain inventory browse: no comparison
    service = _TwoCollectionService()
    out = SiaSearch().run(SiaSearchInput(ra=25, dec=-10, endpoint="https://example.test/sia", deepest_per_band=True,
                                         bands=["g"]), _img_ctx(service)).to_native()
    assert len(service.calls) == 1 and "archive_wide_comparison" not in out  # explicit endpoint = the user's choice


def test_failed_archive_wide_comparison_is_disclosed_not_fatal():
    service = _TwoCollectionService(fail_wide=True)
    out = SiaSearch().run(SiaSearchInput(ra=25, dec=-10, catalog="survey_dr1", deepest_per_band=True,
                                         bands=["g"]), _img_ctx(service)).to_native()
    assert out["success"] and out["archive_wide_comparison"]["error"].startswith("nsa endpoint 502")
    assert any("could not be retrieved" in w for w in out["warnings"])


def test_service_takes_precedence_over_catalog_with_a_warning():
    service = _TwoCollectionService()
    out = SiaSearch().run(SiaSearchInput(ra=25, dec=-10, service="nsa", catalog="survey_dr1"), _img_ctx(service)).to_native()
    assert out["collection"] == "nsa" and service.calls[0]["catalog"] is None
    assert any("service='nsa' takes precedence; catalog='survey_dr1' was ignored" in w for w in out["warnings"])


def test_sia_search_tool_schema_states_scope_precedence():
    import core.tool_registrations as reg
    import inspect
    src = inspect.getsource(reg)
    assert "Takes precedence over catalog" in src
    assert "never add it to an archive-wide request" in src


# ── ALMA: inert topic_filter is disclosed ────────────────────────────────────

def _alma_result(**kwargs):
    service = _FakeSearchService()
    service.alminer_client = _FakeAlminer(_observations())
    ctx, _ = _alma_ctx(search_service=service, resolve_target=lambda target: {"ra_deg": 25.0, "dec_deg": -10.0})
    return _run(QueryAlmaScienceArchive(), ctx, query_type="high_resolution_band_data", target="Synthetic A", **kwargs)


def test_topic_filter_without_structured_filters_is_reported_as_unfiltered():
    result = _alma_result(topic_filter="Band 6 or 7, resolution under 1 arcsec, public science data")
    assert result["success"]
    warning = "\n".join(result["warnings"])
    assert "UNFILTERED" in warning and "max_resolution_arcsec" in warning and "band=[...]" in warning


def test_topic_filter_alongside_structured_filters_is_reported_as_ignored():
    result = _alma_result(topic_filter="protostellar disks", band=[4, 8], max_resolution_arcsec=2.3, public_only=True)
    warning = "\n".join(result["warnings"])
    assert "topic_filter is ignored" in warning and "band, max_resolution_arcsec, public_only" in warning
    assert "UNFILTERED" not in warning


def test_structured_call_without_topic_filter_gets_no_scope_warning():
    result = _alma_result(band=[4, 8], max_resolution_arcsec=2.3)
    assert not any("topic_filter" in w for w in result["warnings"])
