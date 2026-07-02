import base64
import os

from services import plotting
from services.alerce_client import AlerceClient


def _plot_dir(name):
    path = os.path.join("test_results", "live_imagery", name)
    os.makedirs(path, exist_ok=True)
    return path


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeResponse:
    def __init__(self, payload=None, content=PNG_1X1, status_code=200, text=""):
        self._payload = payload if payload is not None else {}
        self.content = content
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


def test_cone_objects_params_and_row_normalization(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {}), timeout))
        return FakeResponse(
            {
                "items": [
                    {
                        "oid": "ZTF1",
                        "ndet": 3,
                        "meanra": 10.1,
                        "meandec": -2.2,
                        "firstmjd": 59000,
                        "lastmjd": 59010,
                        "classalerce": "AGN",
                        "probability": 0.8,
                    }
                ]
            }
        )

    from services import alerce_client

    monkeypatch.setattr(alerce_client.requests, "get", fake_get)
    out = AlerceClient(base_url="https://alerce.test/ztf/v1", timeout=7).cone_objects(10, -2, radius_arcsec=7200, max_rows=5)

    assert out["success"] is True
    assert out["count"] == 1
    assert out["rows"][0]["oid"] == "ZTF1"
    assert out["rows"][0]["classalerce"] == "AGN"
    assert calls[0][0] == "https://alerce.test/ztf/v1/objects"
    assert calls[0][1]["radius"] == 3600.0
    assert calls[0][1]["page_size"] == 5
    assert calls[0][2] == 7
    assert out["warnings"]


def test_light_curve_plot_from_canned_payload(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("alerce"))

    def fake_get(url, params=None, timeout=None):
        assert url.endswith("/objects/ZTF1/lightcurve")
        return FakeResponse(
            {
                "detections": [
                    {"mjd": 59000.1, "magpsf": 18.2, "sigmapsf": 0.1, "fid": 1, "candid": 1, "has_stamp": True},
                    {"mjd": 59001.1, "magpsf": 18.5, "sigmapsf": 0.2, "fid": 2, "candid": 2, "has_stamp": False},
                ],
                "non_detections": [{"mjd": 58999.5, "fid": 1, "diffmaglim": 20.5}],
            }
        )

    from services import alerce_client

    monkeypatch.setattr(alerce_client.requests, "get", fake_get)
    out = AlerceClient().plot_light_curve("ZTF1")

    assert out["success"] is True
    assert out["path"].endswith(".png")
    assert out["n_detections"] == 2
    assert out["n_non_detections"] == 1


def test_stamp_triplet_selects_newest_has_stamp(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("alerce"))
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if "lightcurve" in url:
            return FakeResponse(
                {
                    "detections": [
                        {"mjd": 59000.0, "fid": 1, "candid": 10, "has_stamp": True, "magpsf": 18.0},
                        {"mjd": 59005.0, "fid": 1, "candid": 20, "has_stamp": True, "magpsf": 18.0},
                        {"mjd": 59006.0, "fid": 1, "candid": 30, "has_stamp": False, "magpsf": 18.0},
                    ],
                    "non_detections": [],
                }
            )
        return FakeResponse(content=PNG_1X1)

    from services import alerce_client

    monkeypatch.setattr(alerce_client.requests, "get", fake_get)
    out = AlerceClient().stamp_triplet("ZTF1")

    assert out["success"] is True
    assert out["candid"] == "20"
    stamp_calls = [params for _, params in calls if params.get("format") == "png"]
    assert [params["type"] for params in stamp_calls] == ["science", "template", "difference"]
    assert {params["candid"] for params in stamp_calls} == {"20"}


def test_stamp_404_returns_success_false(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if "lightcurve" in url:
            return FakeResponse({"detections": [{"mjd": 1, "candid": 1, "has_stamp": True}], "non_detections": []})
        return FakeResponse(status_code=404, text="missing")

    from services import alerce_client

    monkeypatch.setattr(alerce_client.requests, "get", fake_get)
    out = AlerceClient().stamp_triplet("ZTF1")
    assert out["success"] is False
    assert "HTTP 404" in out["error"]
