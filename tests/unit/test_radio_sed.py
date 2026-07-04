import os

import pytest

from services import plotting
from services.radio_sed import RadioSedService


def _plot_dir(name):
    path = os.path.join("test_results", "radio_sed", name)
    os.makedirs(path, exist_ok=True)
    return path


def _votable(fields, rows):
    field_xml = []
    for field in fields:
        name = field[0]
        datatype = field[1]
        attrs = [f'name="{name}"', f'datatype="{datatype}"']
        if len(field) > 2 and field[2]:
            attrs.append(f'ucd="{field[2]}"')
        if len(field) > 3 and field[3]:
            attrs.append(f'unit="{field[3]}"')
        field_xml.append("<FIELD " + " ".join(attrs) + "/>")
    row_xml = []
    for row in rows:
        cells = "".join(f"<TD>{value}</TD>" for value in row)
        row_xml.append(f"<TR>{cells}</TR>")
    xml = """<?xml version="1.0"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
  <RESOURCE>
    <TABLE>
      {fields}
      <DATA><TABLEDATA>{rows}</TABLEDATA></DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
""".format(fields="\n      ".join(field_xml), rows="".join(row_xml))
    return xml.encode("utf-8")


def _survey(survey_id, path, freq, flux_cols, scale=1.0, resolution=30.0, epoch=2000, err_cols=None):
    return (survey_id, path, float(freq), list(flux_cols), float(scale), float(resolution), int(epoch), list(err_cols or []))


def test_compile_params_scaling_nearest_warnings_and_status(monkeypatch):
    from services import radio_sed

    registry = [
        _survey("TGSS", "TGSS/path", 150, ["Stotal"], 1.0, 25, 2016, ["e_Stotal"]),
        _survey("GLEAM", "GLEAM/path", 200, ["Fintwide"], 1000.0, 120, 2014, ["e_Fintwide"]),
        _survey("NVSS", "NVSS/path", 1400, ["S1.4"], 1.0, 45, 1995, ["e_S1.4"]),
        _survey("FIRST", "FIRST/path", 1400, ["Fint"], 1.0, 5, 2000, []),
        _survey("BROKEN", "BAD/path", 5000, ["Flux"], 1.0, 10, 2020, []),
        _survey("NOFLUX", "NO/flux", 5000, ["Missing"], 1.0, 10, 2020, []),
    ]
    monkeypatch.setattr(radio_sed, "SURVEY_REGISTRY", registry)
    calls = []

    def fetcher(url, params, timeout):
        calls.append((url, dict(params), timeout))
        if "BAD/path" in url:
            raise RuntimeError("HTTP 404 missing")
        if "TGSS/path" in url:
            return _votable(
                [("_r", "double", "", "arcmin"), ("Stotal", "double"), ("e_Stotal", "double")],
                [(0.5, 999, 9), (0.1, 2000, 50)],
            )
        if "GLEAM/path" in url:
            return _votable([("_r", "double"), ("Fintwide", "double"), ("e_Fintwide", "double")], [(0.2, 2.0, 0.1)])
        if "NVSS/path" in url:
            return _votable([("_r", "double"), ("S1.4", "double"), ("e_S1.4", "double")], [(0.05, 1000, 30)])
        if "FIRST/path" in url:
            return _votable([("_r", "double"), ("Fint", "double")], [(0.03, 700)])
        return _votable([("_r", "double"), ("Other", "double")], [(0.1, 1.0)])

    out = RadioSedService(base_url="https://viz.test/base", timeout=7, fetcher=fetcher).compile_sed(10, -2, radius_arcsec=30)

    assert out["success"] is True
    assert out["count"] == 4
    assert calls[0][0] == "https://viz.test/base/TGSS/path"
    assert calls[0][1] == {"RA": 10.0, "DEC": -2.0, "SR": pytest.approx(30 / 3600), "VERB": 2}
    assert calls[0][2] == 7
    by_survey = {point["survey"]: point for point in out["points"]}
    assert by_survey["TGSS"]["flux_mjy"] == pytest.approx(2000)
    assert by_survey["TGSS"]["flux_err_mjy"] == pytest.approx(50)
    assert by_survey["TGSS"]["sep_arcsec"] == pytest.approx(6.0)
    assert by_survey["GLEAM"]["flux_mjy"] == pytest.approx(2000)
    assert by_survey["GLEAM"]["flux_err_mjy"] == pytest.approx(100)
    assert by_survey["FIRST"]["flux_err_mjy"] is None
    assert any("HTTP 404" in warning for warning in out["warnings"])
    assert any("no positive unmasked flux" in warning for warning in out["warnings"])
    assert any("returned 2 candidates" in warning for warning in out["warnings"])
    statuses = {row["survey"]: row["status"] for row in out["provenance"]["survey_status"]}
    assert statuses["BROKEN"] == "failed"
    assert statuses["NOFLUX"] == "skipped_missing_flux"


def test_compile_radius_clamp_ucd_fallback_empty_and_validation(monkeypatch):
    from services import radio_sed

    monkeypatch.setattr(radio_sed, "SURVEY_REGISTRY", [_survey("UCD", "ucd/path", 1000, ["Flux"])])
    calls = []

    def fetcher(url, params, timeout):
        calls.append(dict(params))
        return _votable(
            [("RAJ2000", "double", "pos.eq.ra;meta.main"), ("DEJ2000", "double", "pos.eq.dec;meta.main"), ("Flux", "double")],
            [(10.1, -2.0, 3.0), (10.001, -2.0, 5.0)],
        )

    out = RadioSedService(fetcher=fetcher).compile_sed(10, -2, radius_arcsec=999)
    assert out["success"] is True
    assert out["count"] == 1
    assert calls[0]["SR"] == pytest.approx(120 / 3600)
    assert out["provenance"]["radius_arcsec"] == 120.0
    assert out["points"][0]["flux_mjy"] == pytest.approx(5.0)
    assert out["points"][0]["sep_arcsec"] == pytest.approx(3.6, rel=0.02)
    assert any("computed separation" in warning for warning in out["warnings"])
    assert any("clamped to 120" in warning for warning in out["warnings"])

    monkeypatch.setattr(radio_sed, "SURVEY_REGISTRY", [_survey("EMPTY", "empty/path", 1000, ["Flux"])])
    empty = RadioSedService(fetcher=lambda *_args: _votable([("_r", "double"), ("Flux", "double")], [])).compile_sed(10, -2)
    assert empty["success"] is True
    assert empty["count"] == 0
    assert empty["provenance"]["survey_status"][0]["status"] == "empty"

    bad = RadioSedService(fetcher=fetcher).compile_sed(400, 0)
    assert bad["success"] is False
    assert "ra must" in bad["error"]


def _power_law_points(alpha=-0.7, with_errors=True):
    freqs = [150.0, 200.0, 843.0, 1400.0]
    points = []
    for idx, freq in enumerate(freqs):
        flux = 1000.0 * (freq / 1400.0) ** alpha
        points.append(
            {
                "survey": f"S{idx}",
                "freq_mhz": freq,
                "flux_mjy": flux,
                "flux_err_mjy": 0.02 * flux if with_errors else None,
                "resolution_arcsec": 30.0,
                "epoch": 2010,
                "candidate_count": 1,
            }
        )
    return points


def test_fit_weighted_power_law_and_unweighted_warning():
    svc = RadioSedService(fetcher=lambda *_args: b"")
    out = svc.fit_spectral_index(_power_law_points())
    assert out["success"] is True
    assert out["alpha"] == pytest.approx(-0.7, abs=0.05)
    assert out["alpha_err"] > 0
    assert out["chi2_red"] is not None
    assert out["s_1400_mjy_predicted"] == pytest.approx(1000, rel=0.01)

    unweighted = svc.fit_spectral_index(_power_law_points(with_errors=False))
    assert unweighted["success"] is True
    assert any("unweighted" in warning for warning in unweighted["warnings"])


def test_fit_flags_and_edge_cases():
    svc = RadioSedService(fetcher=lambda *_args: b"")
    flagged = [
        {"survey": "GLEAM", "freq_mhz": 200, "flux_mjy": 3000, "flux_err_mjy": 60, "resolution_arcsec": 120, "epoch": 2014, "candidate_count": 2},
        {"survey": "TGSS", "freq_mhz": 150, "flux_mjy": 4000, "flux_err_mjy": 80, "resolution_arcsec": 25, "epoch": 2016, "candidate_count": 1},
        {"survey": "NVSS", "freq_mhz": 1400, "flux_mjy": 1000, "flux_err_mjy": 20, "resolution_arcsec": 45, "epoch": 1995, "candidate_count": 1},
        {"survey": "FIRST", "freq_mhz": 1400, "flux_mjy": 500, "flux_err_mjy": 10, "resolution_arcsec": 5, "epoch": 2000, "candidate_count": 1},
    ]
    out = svc.fit_spectral_index(flagged)
    assert out["success"] is True
    assert any("Beam sizes span" in flag for flag in out["flags"])
    assert any("Survey epochs span" in flag for flag in out["flags"])
    assert any("NVSS/FIRST" in flag for flag in out["flags"])
    assert any("nearest match may be confused" in flag for flag in out["flags"])

    one = svc.fit_spectral_index(flagged[:1])
    assert one["success"] is False
    assert "at least 2" in one["error"]

    duplicate_only = svc.fit_spectral_index(flagged[2:4])
    assert duplicate_only["success"] is False
    assert "distinct frequencies" in duplicate_only["error"]

    dropped = svc.fit_spectral_index(flagged[:2] + [{"survey": "BAD", "freq_mhz": 1, "flux_mjy": 0}])
    assert dropped["success"] is True
    assert any("Skipped 1" in warning for warning in dropped["warnings"])


def test_plot_writes_png_with_and_without_fit(monkeypatch):
    monkeypatch.setattr(plotting, "PLOT_OUTPUT_DIR", _plot_dir("plot"))
    svc = RadioSedService(fetcher=lambda *_args: b"")
    points = _power_law_points()
    fit = svc.fit_spectral_index(points)

    with_fit = svc.plot_sed(points, fit=fit, title="Radio SED test")
    assert with_fit["success"] is True
    assert with_fit["path"].endswith(".png")
    assert os.path.exists(with_fit["png_path"])

    no_fit = svc.plot_sed(points, title="Radio SED no fit")
    assert no_fit["success"] is True
    assert no_fit["path"].endswith(".png")

    empty = svc.plot_sed([{"survey": "BAD", "freq_mhz": 1, "flux_mjy": 0}])
    assert empty["success"] is False
    assert "No positive" in empty["error"]


def test_agent_registration_prompt_status_and_radio_sed_wrapper():
    from tests.integration.test_agent_archive_tools import _load_agent_module
    from tests.unit.test_datalab_p0 import _make_agent

    _load_agent_module()
    agent = _make_agent()
    agent._register_tools()
    tool = agent.tool_registry.get_tool("radio_sed")
    assert tool is not None
    assert tool.category == "analysis"
    assert "GLEAM" in tool.description
    assert "VLASS" not in tool.description
    assert "LoTSS" not in tool.description
    assert agent._tool_status_label("radio_sed", {}) == "Compiling radio SED + spectral index"
    prompt = agent._build_system_prompt()
    assert "radio_sed" in prompt
    assert "flags" in prompt

    class FakeRadioSedService:
        def compile_sed(self, ra, dec, radius_arcsec=30.0):
            self.compile_args = (ra, dec, radius_arcsec)
            return {
                "success": True,
                "points": _power_law_points(),
                "count": 4,
                "warnings": ["compile warning"],
                "provenance": {"survey_status": []},
            }

        def fit_spectral_index(self, points):
            return {
                "success": True,
                "alpha": -0.7,
                "alpha_err": 0.1,
                "chi2_red": 1.2,
                "n_points": len(points),
                "s_1400_mjy_predicted": 1000.0,
                "flags": ["flag sentence."],
                "warnings": ["fit warning"],
            }

        def plot_sed(self, points, fit=None, title="Radio SED"):
            return {"success": True, "path": "/plots/radio.png", "png_path": "radio.png", "warnings": ["plot warning"]}

    fake = FakeRadioSedService()
    agent._radio_sed_service_instance = fake
    out = agent._radio_sed(ra=10.0, dec=-2.0, radius_arcsec=30.0)
    assert out["success"] is True
    assert out["image_attached"] is not True if "image_attached" in out else True
    assert out["points"]
    assert out["alpha"] == -0.7
    assert out["alpha_err"] == 0.1
    assert out["flags"] == ["flag sentence."]
    assert any("compile warning" in warning for warning in out["warnings"])
    assert any("fit warning" in warning for warning in out["warnings"])
    assert agent.last_run_result["caption"].startswith("Radio SED")
    assert fake.compile_args == (10.0, -2.0, 30.0)

    class EmptyRadioSedService:
        def compile_sed(self, ra, dec, radius_arcsec=30.0):
            return {"success": True, "points": [], "count": 0, "warnings": [], "provenance": {"survey_status": [{"status": "empty"}]}}

    agent._radio_sed_service_instance = EmptyRadioSedService()
    empty = agent._radio_sed(ra=10.0, dec=-2.0)
    assert empty["success"] is True
    assert "No radio catalog detections" in empty["note"]
