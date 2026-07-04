from services.moc_coverage import MocCoverageService
from tests.integration.test_agent_archive_tools import _load_agent_module
from tests.unit.test_datalab_p0 import _make_agent


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text="", json_exc=None):
        self._payload = [] if payload is None else payload
        self.status_code = status_code
        self.text = text
        self._json_exc = json_exc

    def json(self):
        if self._json_exc:
            raise self._json_exc
        return self._payload


class FakeMocService:
    def __init__(self):
        self.coverage_calls = []
        self.covers_calls = []

    def coverage_at(self, ra, dec, radius_deg=0.0, dataproduct_type=None, keyword=None, max_rows=50, regime=None):
        self.coverage_calls.append(
            {
                "ra": ra,
                "dec": dec,
                "radius_deg": radius_deg,
                "dataproduct_type": dataproduct_type,
                "keyword": keyword,
                "max_rows": max_rows,
                "regime": regime,
            }
        )
        return {
            "success": True,
            "rows": [
                {
                    "id": "CDS/P/VLASS",
                    "title": "VLASS radio survey",
                    "dataproduct_type": "image",
                    "regime": "radio",
                    "moc_sky_fraction": 0.2,
                }
            ],
            "count": 1,
            "total_matches": 1,
            "warnings": [],
            "provenance": {"radius_deg": 0.1, "service": "CDS MOCServer"},
        }

    def survey_covers(self, survey_keyword, ra, dec):
        self.covers_calls.append((survey_keyword, ra, dec))
        return {
            "success": True,
            "covered": True,
            "matches": ["CDS/P/VLASS"],
            "matched_count": 1,
            "warnings": [],
            "provenance": {"service": "CDS MOCServer"},
        }


def test_coverage_at_params_rows_regime_and_sort(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {}), timeout))
        return FakeResponse(
            [
                {
                    "ID": "CDS/P/RADIO",
                    "obs_title": "Radio survey",
                    "dataproduct_type": "image",
                    "em_min": 0.02,
                    "em_max": 0.04,
                    "moc_sky_fraction": "0.20",
                },
                {
                    "ID": "CDS/P/OPTICAL",
                    "obs_title": "Optical survey",
                    "dataproduct_type": "catalog",
                    "em_min": 4e-7,
                    "em_max": 8e-7,
                    "moc_sky_fraction": "0.01",
                },
            ]
        )

    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", fake_get)
    out = MocCoverageService(base_url="https://moc.test/query", timeout=7).coverage_at(10, -2, radius_deg=0.1)

    assert out["success"] is True
    assert out["count"] == 2
    assert out["total_matches"] == 2
    assert [row["id"] for row in out["rows"]] == ["CDS/P/OPTICAL", "CDS/P/RADIO"]
    assert [row["regime"] for row in out["rows"]] == ["optical", "radio"]
    assert calls[0][0] == "https://moc.test/query"
    assert calls[0][2] == 7
    assert calls[0][1]["RA"] == 10.0
    assert calls[0][1]["DEC"] == -2.0
    assert calls[0][1]["SR"] == 0.1
    assert calls[0][1]["intersect"] == "overlaps"
    assert calls[0][1]["get"] == "record"
    assert calls[0][1]["fmt"] == "json"
    assert "ID,obs_title,dataproduct_type" in calls[0][1]["fields"]


def test_radius_and_max_rows_clamp(monkeypatch):
    calls = []
    payload = [
        {"ID": f"CDS/P/{idx}", "obs_title": "Survey", "dataproduct_type": "image", "moc_sky_fraction": idx / 1000}
        for idx in range(250)
    ]

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        return FakeResponse(payload)

    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", fake_get)
    out = MocCoverageService().coverage_at(10, 0, radius_deg=90, max_rows=500)

    assert out["success"] is True
    assert calls[0]["SR"] == 30.0
    assert out["count"] == 200
    assert out["total_matches"] == 250
    assert any("clamped to 30" in warning for warning in out["warnings"])
    assert any("clamped to 200" in warning for warning in out["warnings"])
    assert any("showing first 200" in warning for warning in out["warnings"])


def test_keyword_dataproduct_and_regime_filters_are_client_side(monkeypatch):
    calls = []
    payload = [
        {"ID": "CDS/P/VLASS", "obs_title": "VLASS radio image", "dataproduct_type": "image", "em_min": 0.02, "em_max": 0.04, "moc_sky_fraction": 0.4},
        {"ID": "CDS/P/VLASS-CAT", "obs_title": "VLASS radio catalog", "dataproduct_type": "catalog", "em_min": 0.02, "em_max": 0.04, "moc_sky_fraction": 0.1},
        {"ID": "CDS/P/SDSS", "obs_title": "SDSS optical image", "dataproduct_type": "image", "em_min": 4e-7, "em_max": 8e-7, "moc_sky_fraction": 0.2},
        {"ID": "CDS/P/ALMA", "obs_title": "ALMA cube", "dataproduct_type": "cube", "em_min": 1e-3, "em_max": 2e-3, "moc_sky_fraction": 0.3},
    ]

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        return FakeResponse(payload)

    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", fake_get)
    out = MocCoverageService().coverage_at(10, 0, dataproduct_type="image", keyword="vlass", regime="radio")

    assert out["success"] is True
    assert calls[0]["expr"] == "dataproduct_type=image"
    assert out["total_matches"] == 1
    assert out["rows"][0]["id"] == "CDS/P/VLASS"


def test_truncation_warning(monkeypatch):
    payload = [
        {"ID": f"CDS/P/{idx:02d}", "obs_title": "Survey", "dataproduct_type": "image", "moc_sky_fraction": idx}
        for idx in range(60)
    ]

    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", lambda *args, **kwargs: FakeResponse(payload))
    out = MocCoverageService().coverage_at(10, 0, max_rows=50)

    assert out["success"] is True
    assert out["count"] == 50
    assert out["total_matches"] == 60
    assert any("showing first 50" in warning for warning in out["warnings"])


def test_http_500_returns_success_false(monkeypatch):
    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", lambda *args, **kwargs: FakeResponse(status_code=500, text="boom"))
    out = MocCoverageService().coverage_at(10, 0)

    assert out["success"] is False
    assert "HTTP 500" in out["error"]


def test_survey_covers_true_false_and_filters_before_cap(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        params = dict(params or {})
        calls.append(params)
        if params["DEC"] < -70:
            return FakeResponse([])
        payload = [
            {"ID": f"CDS/P/OTHER-{idx}", "obs_title": "Other", "dataproduct_type": "image", "moc_sky_fraction": idx}
            for idx in range(60)
        ]
        payload.append({"ID": "CDS/P/VLASS", "obs_title": "VLASS", "dataproduct_type": "image", "moc_sky_fraction": 61})
        return FakeResponse(payload)

    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", fake_get)
    svc = MocCoverageService()

    yes = svc.survey_covers("VLASS", 187.2779, 2.0524)
    no = svc.survey_covers("VLASS", 10.0, -75.0)

    assert yes["success"] is True and yes["covered"] is True
    assert yes["matches"] == ["CDS/P/VLASS"]
    assert yes["matched_count"] == 1
    assert no["success"] is True and no["covered"] is False
    assert calls[0]["SR"] == 0.0


def test_expr_error_retries_without_expr(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        params = dict(params or {})
        calls.append(params)
        if "expr" in params:
            return FakeResponse(status_code=500, text="bad expr")
        return FakeResponse([{"ID": "CDS/P/VLASS", "obs_title": "VLASS", "dataproduct_type": "image", "moc_sky_fraction": 0.1}])

    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", fake_get)
    out = MocCoverageService().coverage_at(10, 0, dataproduct_type="image")

    assert out["success"] is True
    assert len(calls) == 2
    assert calls[0]["expr"] == "dataproduct_type=image"
    assert "expr" not in calls[1]
    assert "expr" not in out["provenance"]["params"]
    assert any("retried without expr" in warning for warning in out["warnings"])


def test_empty_parse_and_validation_failures(monkeypatch):
    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", lambda *args, **kwargs: FakeResponse([]))
    empty = MocCoverageService().coverage_at(10, 0)
    assert empty["success"] is True
    assert empty["count"] == 0

    monkeypatch.setattr(moc_coverage.requests, "get", lambda *args, **kwargs: FakeResponse(json_exc=ValueError("bad json")))
    bad_json = MocCoverageService().coverage_at(10, 0)
    assert bad_json["success"] is False
    assert "invalid JSON" in bad_json["error"]

    monkeypatch.setattr(moc_coverage.requests, "get", lambda *args, **kwargs: FakeResponse({"items": []}))
    non_list = MocCoverageService().coverage_at(10, 0)
    assert non_list["success"] is False
    assert "unexpected JSON" in non_list["error"]

    invalid_coords = MocCoverageService().coverage_at(400, 0)
    assert invalid_coords["success"] is False
    assert "ra must" in invalid_coords["error"]

    invalid_type = MocCoverageService().coverage_at(10, 0, dataproduct_type="spectrum")
    assert invalid_type["success"] is False
    assert "dataproduct_type" in invalid_type["error"]


def test_point_and_nonpositive_normalization(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        return FakeResponse([])

    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", fake_get)
    out = MocCoverageService().coverage_at(10, 0, radius_deg=-1, max_rows=0)

    assert out["success"] is True
    assert calls[0]["SR"] == 0.0
    assert any("negative" in warning for warning in out["warnings"])
    assert any("max_rows must be positive" in warning for warning in out["warnings"])


def test_tiny_nonzero_radius_clamped_up(monkeypatch):
    # MocServer 500s for 0 < SR < ~1e-4; the service must clamp up, never send 1e-6.
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        return FakeResponse([])

    from services import moc_coverage

    monkeypatch.setattr(moc_coverage.requests, "get", fake_get)
    out = MocCoverageService().coverage_at(10, 0, radius_deg=1e-6)

    assert out["success"] is True
    assert calls[0]["SR"] == moc_coverage.MIN_NONZERO_RADIUS_DEG


def test_agent_registration_prompt_status_and_wrappers():
    _load_agent_module()
    agent = _make_agent()
    agent._register_tools()

    coverage_tool = agent.tool_registry.get_tool("survey_coverage")
    covers_tool = agent.tool_registry.get_tool("survey_covers_position")
    assert coverage_tool is not None
    assert covers_tool is not None
    assert coverage_tool.category == "archive"
    assert covers_tool.category == "archive"
    assert "survey_coverage" in agent.tool_registry.categories["archive"]
    assert "regime" in coverage_tool.parameters["properties"]

    prompt = agent._build_system_prompt()
    assert "MOC COVERAGE RULE" in prompt
    assert "survey_covers_position" in prompt
    assert agent._tool_status_label("survey_coverage", {}) == "Checking sky coverage (MOCServer)"
    assert agent._tool_status_label("survey_covers_position", {}) == "Checking survey footprint"


def test_agent_survey_coverage_sets_external_catalog_table():
    _load_agent_module()
    agent = _make_agent()
    fake = FakeMocService()
    agent._moc_coverage_service_instance = fake

    out = agent._survey_coverage(ra=10.0, dec=-2.0, radius_deg=0.1, dataproduct_type="image", keyword="VLASS", regime="radio")

    assert out["success"] is True
    assert out["results_preview"][0]["id"] == "CDS/P/VLASS"
    assert agent.last_run_result["table_kind"] == "external_catalog"
    assert agent.last_run_result["tool_name"] == "survey_coverage"
    assert agent.last_run_result["source"] == "CDS MOCServer"
    assert fake.coverage_calls[0]["regime"] == "radio"


def test_agent_survey_covers_position_returns_boolean_result_without_table():
    _load_agent_module()
    agent = _make_agent()
    fake = FakeMocService()
    agent._moc_coverage_service_instance = fake

    out = agent._survey_covers_position("VLASS", ra=10.0, dec=-2.0)

    assert out["success"] is True
    assert out["covered"] is True
    assert out["target"] == "RA=10.00000, Dec=-2.00000"
    assert fake.covers_calls == [("VLASS", 10.0, -2.0)]
    assert agent.last_run_result is None