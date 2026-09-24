"""Offline tests for vo_image_search: SIA 2.0 with SIA 1.0 fallback, named
wavebands, DataLink flagging, archive -> endpoint resolution, and the SSRF
guard. pyvo is replaced by injected fakes."""

from __future__ import annotations

import pytest

from capabilities.base import CallContext
from capabilities.vo import VoImageSearch
from services.vo_registry import WAVEBAND_METERS, VoRegistryService

SIA = "https://sia.example.org/sia2"


class _Table:
    def __init__(self, colnames, rows):
        self.colnames = colnames
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


ROWS = [
    {"obs_collection": "ALMA", "dataproduct_type": "cube", "calib_level": 2,
     "access_url": "https://sia.example.org/datalink?ID=1",
     "access_format": "application/x-votable+xml;content=datalink"},
    {"obs_collection": "ALMA", "dataproduct_type": "image", "calib_level": 2,
     "access_url": "https://sia.example.org/datalink?ID=2",
     "access_format": "application/x-votable+xml;content=datalink"},
]


class FakeSIA2:
    def __init__(self, rows=ROWS):
        self.calls = []
        self.rows = rows

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return _Table(list(self.rows[0]) if self.rows else [], list(self.rows))


class FakeSIA1:
    def __init__(self):
        self.calls = []

    def search(self, pos, size=1.0, **kw):
        self.calls.append({"pos": pos, "size": size})
        return _Table(["title", "access_url"], [{"title": "img", "access_url": "https://x/1.fits"}])


def _svc(sia2=None, sia1=None, probe_fails=False):
    def sia2_factory(url):
        if probe_fails:
            raise RuntimeError("Unable to access the capabilities endpoint")
        return sia2

    return VoRegistryService(sia2_factory=sia2_factory, sia1_factory=lambda url: sia1)


def test_sia2_search_maps_filters():
    sia2 = FakeSIA2()
    out = _svc(sia2).image_search(SIA, 187.7, 12.39, radius_deg=0.1, waveband="mm",
                                  calib_level=2, dataproduct_type="cube",
                                  collection="ALMA", max_rows=50)
    assert out["success"] and out["protocol"] == "SIA2" and out["count"] == 2
    call = sia2.calls[0]
    assert call["pos"] == (187.7, 12.39, 0.1) and call["maxrec"] == 50
    assert call["band"] == WAVEBAND_METERS["mm"]
    assert call["calib_level"] == 2 and call["data_type"] == "cube"
    assert call["collection"] == "ALMA"
    assert any("DataLink" in w for w in out["warnings"])
    assert out["provenance"]["endpoint"] == SIA


def test_auto_falls_back_to_sia1_and_discloses_dropped_filters():
    sia1 = FakeSIA1()
    out = _svc(sia1=sia1, probe_fails=True).image_search(
        SIA, 10.0, -5.0, radius_deg=0.2, waveband="optical")
    assert out["success"] and out["protocol"] == "SIA1"
    assert sia1.calls == [{"pos": (10.0, -5.0), "size": 0.4}]
    assert any("SIA 1.0" in w for w in out["warnings"])
    assert any("waveband" in w for w in out["warnings"])


def test_version_2_does_not_fall_back():
    out = _svc(sia1=FakeSIA1(), probe_fails=True).image_search(SIA, 10.0, 0.0, version="2")
    assert out["success"] is False and "capabilities" in out["error"]


@pytest.mark.parametrize("kwargs,fragment", [
    ({"ra": 400.0, "dec": 0.0}, "ra must be"),
    ({"ra": 10.0, "dec": 0.0, "waveband": "gamma-ish"}, "Unknown waveband"),
    ({"ra": 10.0, "dec": 0.0, "version": "3"}, "version must be"),
])
def test_input_validation(kwargs, fragment):
    out = _svc(FakeSIA2()).image_search(SIA, **kwargs)
    assert out["success"] is False and fragment in out["error"]


def test_radius_and_rows_clamped():
    sia2 = FakeSIA2()
    out = _svc(sia2).image_search(SIA, 10.0, 0.0, radius_deg=45, max_rows=10**6)
    assert sia2.calls[0]["pos"][2] == 2.0 and sia2.calls[0]["maxrec"] == 500
    assert any("clamped" in w for w in out["warnings"])


def test_exact_cap_warns_about_more_images():
    """CX-12: a result exactly at max_rows is flagged as possibly partial."""
    out = _svc(FakeSIA2()).image_search(SIA, 10.0, 0.0, max_rows=2)
    assert out["count"] == 2 and any("reached max_rows" in w for w in out["warnings"])


def test_private_endpoint_rejected_before_probe():
    built = []
    svc = VoRegistryService(sia2_factory=lambda url: built.append(url))
    out = svc.image_search("http://127.0.0.1:8000/sia", 10.0, 0.0)
    assert out["success"] is False and "public network address" in out["error"]
    assert built == []


# ── capability ──────────────────────────────────────────────────────────────
class _Recorder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, rows, **kwargs):
        self.kwargs = dict(kwargs, rows=rows)
        return {"success": True, "total_results": len(rows), "tool_name": kwargs["tool_name"]}


def _ctx(service):
    rec = _Recorder()
    ctx = CallContext(services={
        "get_vo_registry_service": lambda: service,
        "external_catalog_table_result": rec,
        "live_imagery_coordinates": lambda **kw: (187.7, 12.39, "M87"),
    })
    return ctx, rec


def _run(ctx, **kw):
    cap = VoImageSearch()
    return cap.run(cap.InputModel(**kw), ctx).to_native()


def test_capability_resolves_archive_to_profile_sia_endpoint():
    seen = []

    class Svc(VoRegistryService):
        def image_search(self, access_url, *a, **k):
            seen.append(access_url)
            return {"success": True, "rows": ROWS, "columns": list(ROWS[0]),
                    "protocol": "SIA2", "provenance": {"radius_deg": 0.05}}

    ctx, rec = _ctx(Svc())
    out = _run(ctx, archive="alma", target_name="M87")
    assert out["success"] and seen == ["https://almascience.nrao.edu/sia2"]
    assert rec.kwargs["tool_name"] == "vo_image_search"
    assert rec.kwargs["columns"][0] == "obs_collection"
    assert rec.kwargs["filter_label"] == "images at M87, r=0.05 deg"


def test_capability_unknown_archive_lists_known_ones():
    ctx, _ = _ctx(VoRegistryService())
    out = _run(ctx, archive="nowhere", ra=1.0, dec=2.0)
    assert out["success"] is False and "alma" in out["error"]


def test_capability_requires_archive_or_url():
    ctx, _ = _ctx(VoRegistryService())
    out = _run(ctx, ra=1.0, dec=2.0)
    assert out["success"] is False and "access_url" in out["error"]
