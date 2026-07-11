import math

from services.dust_extinction import EXTINCTION_COEFF, DustExtinctionService
from tests.integration.test_agent_archive_tools import _load_agent_module
from tests.unit.test_datalab_p0 import _make_agent


XML_BOTH = """<?xml version="1.0"?>
<results xmlns="urn:test">
  <result>
    <description>E(B-V) Reddening</description>
    <statistics>
      <refPixelValueSFD>E(B-V)=0.0200 (mag)</refPixelValueSFD>
      <meanValueSFD>0.0220 (mag)</meanValueSFD>
      <refPixelValueSandF>0.0172 (mag)</refPixelValueSandF>
      <meanValueSandF>0.0189 (mag)</meanValueSandF>
    </statistics>
  </result>
</results>
"""

XML_SFD_ONLY = """<?xml version="1.0"?>
<results>
  <result>
    <statistics>
      <refPixelValueSFD>1.0e-1 (mag)</refPixelValueSFD>
      <meanValueSFD>0.12 (mag)</meanValueSFD>
    </statistics>
  </result>
</results>
"""

XML_NO_STATISTICS = """<?xml version="1.0"?>
<results>
  <refPixelValueSFD>0.030 (mag)</refPixelValueSFD>
  <refPixelValueSandF>0.0258 (mag)</refPixelValueSandF>
</results>
"""


class FakeResponse:
    def __init__(self, text=XML_BOTH, status_code=200, content=b""):
        self.text = text
        self.status_code = status_code
        self.content = content


class FakeDustService:
    def __init__(self):
        self.calls = []

    def extinction_table(self, ra, dec, bands=None):
        self.calls.append({"ra": ra, "dec": dec, "bands": bands})
        return {
            "success": True,
            "rows": [{"band": "sdss_r", "A_lambda": 0.039302, "coeff": 2.285}],
            "count": 1,
            "ebv_sfd": 0.0200,
            "ebv_sf11": 0.0172,
            "ebv_sfd_mean": 0.0220,
            "ebv_sf11_mean": 0.0189,
            "warnings": ["unit-test warning"],
            "provenance": {"service": "IRSA DUST extinction table", "ebv_used": "ebv_sf11"},
        }


def test_ebv_params_timeout_and_parses_sfd_sandf_stats():
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {}), timeout))
        return FakeResponse(XML_BOTH)

    out = DustExtinctionService(base_url="https://irsa.test/dust", timeout=7, http_get=fake_get).ebv(187.2779, 2.0524)

    assert out["success"] is True
    assert out["ebv_sfd"] == 0.0200
    assert out["ebv_sf11"] == 0.0172
    assert out["ebv_sfd_mean"] == 0.0220
    assert out["ebv_sf11_mean"] == 0.0189
    assert calls == [("https://irsa.test/dust", {"locstr": "187.2779 2.0524 equ j2000", "regSize": "2.0"}, 7)]
    assert out["provenance"]["params"]["locstr"] == "187.2779 2.0524 equ j2000"


def test_missing_sandf_computes_sf11_and_warns():
    out = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(XML_SFD_ONLY)).ebv(10, 0)

    assert out["success"] is True
    assert math.isclose(out["ebv_sfd"], 0.1)
    assert math.isclose(out["ebv_sf11"], 0.086)
    assert math.isclose(out["ebv_sf11_mean"], 0.1032)
    assert any("computed SF11 E(B-V)" in warning for warning in out["warnings"])


def test_low_latitude_position_warns():
    out = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(XML_BOTH)).ebv(266.4, -29.0)

    assert out["success"] is True
    assert any("low Galactic latitude" in warning for warning in out["warnings"])
    assert abs(out["provenance"]["galactic_b_deg"]) < 5.0


def test_http_500_returns_success_false():
    out = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(status_code=500, text="boom")).ebv(10, 0)

    assert out["success"] is False
    assert "HTTP 500" in out["error"]


def test_invalid_xml_and_missing_stats_are_informative():
    invalid = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse("<html>")).ebv(10, 0)
    assert invalid["success"] is False
    assert "invalid XML" in invalid["error"]

    missing = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse("<results><statistics /></results>")).ebv(10, 0)
    assert missing["success"] is False
    assert "did not include" in missing["error"]

    fallback = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(XML_NO_STATISTICS)).ebv(10, 0)
    assert fallback["success"] is True
    assert fallback["ebv_sfd"] == 0.030
    assert any("searched all leaf tags" in warning for warning in fallback["warnings"])


def test_validation_failure_is_success_false():
    out = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(XML_BOTH)).ebv(400, 0)

    assert out["success"] is False
    assert "ra must" in out["error"]


def test_extinction_table_math_aliases_unknown_band_and_all_coefficients():
    svc = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(XML_SFD_ONLY))
    out = svc.extinction_table(10, 0, bands=["V", "SDSS r band", "foo", "r"])

    assert out["success"] is True
    assert [row["band"] for row in out["rows"]] == ["V", "sdss_r"]
    assert math.isclose(out["rows"][0]["A_lambda"], 2.742 * 0.086)
    assert math.isclose(out["rows"][1]["A_lambda"], 2.285 * 0.086)
    assert any("Unknown extinction band 'foo'" in warning for warning in out["warnings"])
    assert any("Band 'r' is ambiguous" in warning for warning in out["warnings"])
    assert out["provenance"]["ebv_used"] == "ebv_sf11"

    all_bands = svc.extinction_table(10, 0)
    assert all_bands["success"] is True
    assert all_bands["count"] == len(EXTINCTION_COEFF)
    assert all_bands["rows"][0]["band"] == "U"


def test_natural_band_names_resolve_like_bare_letters():
    # "V band", "Johnson V", "Cousins R", "Bessell V" must resolve to the same
    # SF11 Landolt keys as bare "V"/"B" instead of an empty table.
    svc = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(XML_SFD_ONLY))
    out = svc.extinction_table(
        10,
        0,
        bands=[
            "V band",
            "B band",
            "Johnson V",
            "Johnson B",
            "johnson u",
            "Cousins R",
            "Cousins I",
            "Bessell V",
            "Johnson-Cousins I",
        ],
    )

    assert out["success"] is True
    assert [row["band"] for row in out["rows"]] == ["V", "B", "U", "R", "I"]
    assert math.isclose(out["rows"][0]["A_lambda"], 2.742 * 0.086)
    assert not any("Unknown" in warning or "ambiguous" in warning for warning in out["warnings"])


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("IRSA_DUST_BASE_URL", "https://env.test/dust")
    monkeypatch.setenv("IRSA_DUST_TIMEOUT", "11")

    svc = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(XML_BOTH))

    assert svc.base_url == "https://env.test/dust"
    assert svc.timeout == 11.0


def test_agent_registration_prompt_status_and_wrapper_schema():
    _load_agent_module()
    agent = _make_agent()
    agent._register_tools()

    tool = agent.tool_registry.get_tool("galactic_extinction")
    assert tool is not None
    assert tool.category == "archive"
    assert "galactic_extinction" in agent.tool_registry.categories["archive"]
    assert tool.parameters["properties"]["bands"]["type"] == "array"

    prompt = agent._build_system_prompt()
    assert "galactic_extinction" in prompt
    assert agent._tool_status_label("galactic_extinction", {}) == "Querying IRSA dust maps"


def test_agent_galactic_extinction_sets_table_and_preserves_ebv_values():
    _load_agent_module()
    agent = _make_agent()
    fake = FakeDustService()
    agent._dust_extinction_service_instance = fake

    out = agent._galactic_extinction(ra=10.0, dec=-2.0, bands=["SDSS r band"])

    assert out["success"] is True
    assert out["ebv_sfd"] == 0.0200
    assert out["ebv_sf11"] == 0.0172
    assert out["results_preview"][0]["band"] == "sdss_r"
    assert out["provenance"]["ebv_used"] == "ebv_sf11"
    assert agent.last_run_result["table_kind"] == "external_catalog"
    assert agent.last_run_result["tool_name"] == "galactic_extinction"
    assert agent.last_run_result["source"] == "IRSA DUST (SFD98/SF11)"
    assert "SFD=0.0200" in agent.last_run_result["filter_label"]
    assert fake.calls == [{"ra": 10.0, "dec": -2.0, "bands": ["SDSS r band"]}]


def test_band_suffix_normalization_regression():
    # Codex review P2 (f09-dust): "V band" / "J band" / "W1 band" must resolve
    # to their unique keys; only multi-system letters stay ambiguous.
    svc = DustExtinctionService(http_get=lambda *args, **kwargs: FakeResponse(XML_SFD_ONLY))
    out = svc.extinction_table(10, 0, bands=["V band", "J band", "W1 band", "y", "r band"])

    assert out["success"] is True
    assert [row["band"] for row in out["rows"]] == ["V", "J", "W1", "ps1_y"]
    assert any("Band 'r band' is ambiguous" in warning for warning in out["warnings"])
    assert not any("'V band'" in warning for warning in out["warnings"])
