import base64
import os

import pytest

from services import plotting
from services.hips_images import HipsImageError, HipsImageService, resolve_survey


def _plot_dir(name):
    path = os.path.join("test_results", "live_imagery", name)
    os.makedirs(path, exist_ok=True)
    return path


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeResponse:
    def __init__(self, content=PNG_1X1, status_code=200, text=""):
        self.content = content
        self.status_code = status_code
        self.text = text


def test_resolve_survey_alias_raw_passthrough_and_unknown():
    assert resolve_survey("optical") == "CDS/P/DSS2/color"
    assert resolve_survey("2MASS") == "CDS/P/2MASS/color"
    assert resolve_survey("CDS/P/custom") == "CDS/P/custom"
    with pytest.raises(HipsImageError):
        resolve_survey("not_a_survey")


def test_cutout_builds_hips2fits_params_and_writes_png(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("hips_cutout"))
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = dict(params or {})
        captured["timeout"] = timeout
        return FakeResponse()

    from services import hips_images

    monkeypatch.setattr(hips_images.requests, "get", fake_get)
    svc = HipsImageService(base_url="https://example.test/hips2fits", timeout=12)
    out = svc.cutout(10.0, -2.0, fov_deg=0.2, survey="sdss", width=128)

    assert out["success"] is True
    assert out["path"].startswith("/plots/hips_") and out["path"].endswith(".png")
    assert out["image_base64"] == base64.b64encode(PNG_1X1).decode("ascii")
    assert captured["url"] == "https://example.test/hips2fits"
    assert captured["timeout"] == 12
    assert captured["params"] == {
        "hips": "CDS/P/SDSS9/color",
        "ra": 10.0,
        "dec": -2.0,
        "fov": 0.2,
        "width": 128,
        "height": 128,
        "projection": "TAN",
        "format": "png",
    }


def test_cutout_rejects_non_png(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("hips_cutout"))
    from services import hips_images

    monkeypatch.setattr(hips_images.requests, "get", lambda *a, **k: FakeResponse(content=b"not png"))
    out = HipsImageService().cutout(10.0, 0.0)
    assert out["success"] is False
    assert "PNG" in out["error"]


def test_vlass_dec_guard_blocks_all_vlass_aliases(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("hips_cutout"))
    from services import hips_images

    def fail_get(*args, **kwargs):
        raise AssertionError("network should not be called for Dec below VLASS coverage")

    monkeypatch.setattr(hips_images.requests, "get", fail_get)
    svc = HipsImageService()
    for survey in ("vlass", "radio", "NRAO/P/VLASS-Quicklook-MedianStack"):
        out = svc.cutout(10.0, -45.0, survey=survey)
        assert out["success"] is False
        assert "VLASS covers Dec > -40" in out["error"]


def test_cutout_clamps_fov_and_width(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("hips_cutout"))
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["params"] = dict(params or {})
        return FakeResponse()

    from services import hips_images

    monkeypatch.setattr(hips_images.requests, "get", fake_get)
    out = HipsImageService().cutout(10.0, 0.0, fov_deg=50, width=9999)
    assert out["success"] is True
    assert out["fov_deg"] == 10.0
    assert out["width"] == 2048
    assert captured["params"]["fov"] == 10.0
    assert captured["params"]["width"] == 2048
    assert out["warnings"]
