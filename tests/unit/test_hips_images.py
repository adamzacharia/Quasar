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


# ── Same-survey color HiPS support (no product substitution on color requests) ─

def _png_bytes(uniform):
    import io

    import numpy as np
    from PIL import Image

    if uniform:
        arr = np.full((8, 8, 3), 7, dtype=np.uint8)
    else:
        arr = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_color_hips_for_catalog_mapping():
    from services.hips_images import CATALOG_COLOR_HIPS, color_hips_for_catalog

    # Both Legacy Surveys releases resolve to the DR10 color HiPS (no DR9 color
    # HiPS exists at CDS; verified live 2026-07-28).
    assert color_hips_for_catalog("ls_dr9")["hips_id"] == "CDS/P/DESI-Legacy-Surveys/DR10/color"
    assert color_hips_for_catalog("LS_DR10")["hips_id"] == "CDS/P/DESI-Legacy-Surveys/DR10/color"
    assert color_hips_for_catalog("des_dr2")["hips_id"] == "CDS/P/DES-DR2/ColorIRG"
    # Surveys that publish no color HiPS yield None — the caller must report the
    # product impossible, never switch surveys.
    for missing in ("delve_dr2", "nsc_dr2", "smash_dr2", "", None):
        assert color_hips_for_catalog(missing) is None
    for descriptor in CATALOG_COLOR_HIPS.values():
        assert descriptor["hips_id"] and descriptor["survey_label"] and descriptor["bands"]


def test_cutout_detect_blank_flags_uniform_png(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("hips_cutout"))
    from services import hips_images

    # Uniform PNG = hips2fits' out-of-footprint response (the DR10 color HiPS
    # at M31) → blank=True so the caller can refuse instead of shipping it.
    monkeypatch.setattr(hips_images.requests, "get", lambda *a, **k: FakeResponse(content=_png_bytes(uniform=True)))
    svc = HipsImageService()
    out = svc.cutout(10.6847, 41.2687, survey="CDS/P/DESI-Legacy-Surveys/DR10/color", detect_blank=True)
    assert out["success"] is True and out["blank"] is True

    monkeypatch.setattr(hips_images.requests, "get", lambda *a, **k: FakeResponse(content=_png_bytes(uniform=False)))
    out = svc.cutout(150.0, 2.2, survey="CDS/P/DESI-Legacy-Surveys/DR10/color", detect_blank=True)
    assert out["success"] is True and out["blank"] is False
    # Default path is unchanged: no blank key, v1 coverage note.
    out = svc.cutout(150.0, 2.2)
    assert "blank" not in out


class _PanelPlotter:
    """In-memory stand-in so multiband_panel needs no plot directory."""

    def _apply_style(self, dark=False):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt

    def _save_and_encode(self, fig, filename):
        import matplotlib.pyplot as plt

        plt.close(fig)
        return {"success": True, "base64_png": "abc", "web_url": f"/plots/{filename}.png",
                "png_path": None, "pdf_path": None}


def test_multiband_panel_note_says_comparison_not_color(monkeypatch):
    from services import hips_images

    monkeypatch.setattr(hips_images.requests, "get", lambda *a, **k: FakeResponse(content=_png_bytes(uniform=False)))
    out = HipsImageService(plotting_service=_PanelPlotter()).multiband_panel(
        10.0, -2.0, surveys=["optical", "2mass"], width=64
    )
    assert out["success"] is True
    # Beta eval 2026-07: the panel was passed off as a 'color image'. The result
    # itself now says what it is and where color requests belong.
    assert "NOT a color" in out["note"]
    assert "datalab_color_image" in out["note"]
