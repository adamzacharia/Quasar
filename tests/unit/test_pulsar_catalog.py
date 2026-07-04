import pandas as pd

from services.pulsar_catalog import FULL_COLUMNS, PulsarCatalogService
from tests.integration.test_agent_archive_tools import _load_agent_module
from tests.unit.test_datalab_p0 import _make_agent


def _sample_table():
    return pd.DataFrame(
        [
            {
                "JNAME": "J0332+5434",
                "NAME": "B0329+54",
                "BNAME": "B0329+54",
                "RAJD": 53.2475,
                "DECJD": 54.5787,
                "P0": 0.714520,
                "P1": 2.05e-15,
                "DM": 26.7641,
                "DIST": 1.0,
                "AGE": 5.5e6,
                "BSURF": 1.2e12,
                "EDOT": 1.4e31,
                "S1400": 203.5,
                "BINARY": None,
                "ASSOC": "",
                "TYPE": "HE",
            },
            {
                "JNAME": "J0534+2200",
                "NAME": "B0531+21",
                "BNAME": "B0531+21",
                "RAJD": 83.63322,
                "DECJD": 22.01446,
                "P0": 0.033392,
                "P1": 4.21e-13,
                "DM": 56.77,
                "DIST": 2.0,
                "AGE": 1250.0,
                "BSURF": 3.8e12,
                "EDOT": 4.6e38,
                "S1400": 14.0,
                "BINARY": None,
                "ASSOC": "SNR:Crab",
                "TYPE": "HE",
            },
            {
                "JNAME": "J0437-4715",
                "NAME": "J0437-4715",
                "BNAME": None,
                "RAJD": 69.3162,
                "DECJD": -47.2525,
                "P0": 0.00575745,
                "P1": 5.73e-20,
                "DM": 2.64,
                "DIST": 0.156,
                "AGE": 1.6e9,
                "BSURF": 5.8e8,
                "EDOT": 1.2e34,
                "S1400": 150.0,
                "BINARY": "ELL1",
                "ASSOC": None,
                "TYPE": "MSP",
            },
            {
                "JNAME": "J9999+0000",
                "NAME": "J9999+0000",
                "BNAME": None,
                "RAJD": float("nan"),
                "DECJD": float("nan"),
                "P0": float("nan"),
                "P1": None,
                "DM": pd.NA,
                "DIST": None,
                "AGE": None,
                "BSURF": None,
                "EDOT": None,
                "S1400": None,
                "BINARY": None,
                "ASSOC": None,
                "TYPE": None,
            },
            {
                "JNAME": "J0108-1431",
                "NAME": "B0105-14",
                "BNAME": "B0105-14",
                "RAJD": 17.145,
                "DECJD": -14.516,
                "P0": 0.807565,
                "P1": 7.7e-17,
                "DM": 2.38,
                "DIST": 0.24,
                "AGE": 1.7e8,
                "BSURF": 2.5e11,
                "EDOT": 5.8e30,
                "S1400": None,
                "BINARY": None,
                "ASSOC": None,
                "TYPE": "NRAD",
            },
        ]
    )


def test_cone_search_finds_nearest_and_drops_nan_position():
    svc = PulsarCatalogService(table_loader=_sample_table)
    out = svc.search_pulsars(53.2475, 54.5787, radius_deg=1.0, max_rows=25)

    assert out["success"] is True
    assert out["count"] >= 1
    assert out["rows"][0]["jname"] == "J0332+5434"
    assert out["rows"][0]["p0_s"] == 0.71452
    assert out["rows"][0]["dm_pc_cm3"] == 26.76
    assert out["rows"][0]["sep_arcmin"] == 0.0
    assert all(row["jname"] != "J9999+0000" for row in out["rows"])
    assert any("without RAJD/DECJD" in warning for warning in out["warnings"])


def test_radius_and_max_rows_clamps_warn():
    svc = PulsarCatalogService(table_loader=_sample_table)
    out = svc.search_pulsars(53.2475, 54.5787, radius_deg=90, max_rows=999)

    assert out["success"] is True
    assert out["provenance"]["radius_deg"] == 30.0
    assert out["provenance"]["max_rows"] == 200
    assert any("clamped to 30" in warning for warning in out["warnings"])
    assert any("clamped to 200" in warning for warning in out["warnings"])


def test_lookup_matches_bname_psr_prefix_jname_and_dedupes():
    svc = PulsarCatalogService(table_loader=_sample_table)

    bname = svc.pulsar_lookup("B0329+54")
    prefixed = svc.pulsar_lookup("psr b0329+54")
    jname = svc.pulsar_lookup("J0332+5434")
    no_prefix = svc.pulsar_lookup("0329+54")
    unknown = svc.pulsar_lookup("J0000+0000")

    for out in (bname, prefixed, jname, no_prefix):
        assert out["success"] is True
        assert out["count"] == 1
        assert out["rows"][0]["jname"] == "J0332+5434"
        assert out["rows"][0]["bname"] == "B0329+54"
    assert unknown["success"] is True
    assert unknown["count"] == 0
    assert unknown["rows"] == []


def test_loader_failure_is_not_cached_and_retries():
    calls = []

    def loader():
        calls.append("call")
        raise OSError("network down")

    svc = PulsarCatalogService(table_loader=loader)
    first = svc.search_pulsars(10, 0)
    second = svc.pulsar_lookup("B0329+54")

    assert first["success"] is False
    assert second["success"] is False
    assert "ATNF catalogue download failed" in first["error"]
    assert "network down" in second["error"]
    assert calls == ["call", "call"]


def test_nan_values_normalize_to_none_in_rows():
    svc = PulsarCatalogService(table_loader=_sample_table)
    out = svc.pulsar_lookup("J9999+0000")

    assert out["success"] is True
    assert out["count"] == 1
    row = out["rows"][0]
    assert row["ra_deg"] is None
    assert row["dec_deg"] is None
    assert row["p0_s"] is None
    assert row["dm_pc_cm3"] is None
    assert row["binary"] is None


def test_successful_load_is_cached_once():
    calls = []

    def loader():
        calls.append("call")
        return _sample_table()

    svc = PulsarCatalogService(table_loader=loader)
    assert svc.search_pulsars(53.2475, 54.5787)["success"] is True
    assert svc.pulsar_lookup("B0329+54")["success"] is True
    assert calls == ["call"]


def test_empty_cone_and_invalid_coords_do_not_error_or_load_unnecessarily():
    calls = []

    def loader():
        calls.append("call")
        return _sample_table()

    svc = PulsarCatalogService(table_loader=loader)
    invalid = svc.search_pulsars(400, 0)
    assert invalid["success"] is False
    assert "ra must" in invalid["error"]
    assert calls == []

    empty = svc.search_pulsars(180, 0, radius_deg=0.1)
    assert empty["success"] is True
    assert empty["count"] == 0
    assert empty["rows"] == []
    assert calls == ["call"]


def test_missing_optional_columns_are_filled_with_none():
    table = pd.DataFrame([{"JNAME": "J1111+2222", "RAJD": 1.0, "DECJD": 2.0, "P0": 1.23456789}])
    out = PulsarCatalogService(table_loader=lambda: table).pulsar_lookup("J1111+2222")

    assert out["success"] is True
    assert out["count"] == 1
    row = out["rows"][0]
    assert set(FULL_COLUMNS).issubset(row)
    assert row["name"] is None
    assert row["dm_pc_cm3"] is None
    assert row["p0_s"] == 1.23457


class FakePulsarService:
    def __init__(self):
        self.search_calls = []
        self.lookup_calls = []

    def search_pulsars(self, ra, dec, radius_deg=1.0, max_rows=25):
        self.search_calls.append({"ra": ra, "dec": dec, "radius_deg": radius_deg, "max_rows": max_rows})
        return {
            "success": True,
            "rows": [
                {
                    "jname": "J0332+5434",
                    "p0_s": 0.71452,
                    "dm_pc_cm3": 26.76,
                    "s1400_mjy": 203.5,
                    "dist_kpc": 1.0,
                    "binary": None,
                    "assoc": None,
                    "sep_arcmin": 0.0,
                }
            ],
            "count": 1,
            "warnings": ["service warning"],
            "provenance": {"service": "ATNF Pulsar Catalogue (psrqpy)", "radius_deg": radius_deg},
        }

    def pulsar_lookup(self, name):
        self.lookup_calls.append(name)
        return {
            "success": True,
            "rows": [
                {
                    "jname": "J0332+5434",
                    "name": "B0329+54",
                    "bname": "B0329+54",
                    "ra_deg": 53.2475,
                    "dec_deg": 54.5787,
                    "p0_s": 0.71452,
                    "p1": 2.05e-15,
                    "dm_pc_cm3": 26.76,
                    "dist_kpc": 1.0,
                    "age_yr": 5.5e6,
                    "bsurf_g": 1.2e12,
                    "edot_erg_s": 1.4e31,
                    "s1400_mjy": 203.5,
                    "binary": None,
                    "assoc": None,
                    "type": "HE",
                }
            ],
            "count": 1,
            "warnings": [],
            "provenance": {"service": "ATNF Pulsar Catalogue (psrqpy)", "name": name},
        }


def test_agent_registration_prompt_status_and_schema():
    _load_agent_module()
    agent = _make_agent()
    agent._register_tools()

    search_tool = agent.tool_registry.get_tool("search_pulsars")
    lookup_tool = agent.tool_registry.get_tool("pulsar_lookup")
    assert search_tool is not None
    assert lookup_tool is not None
    assert search_tool.category == "archive"
    assert lookup_tool.category == "archive"
    assert "search_pulsars" in agent.tool_registry.categories["archive"]
    assert "pulsar_lookup" in agent.tool_registry.categories["archive"]
    assert search_tool.parameters["required"] == []
    assert lookup_tool.parameters["required"] == ["name"]

    prompt = agent._build_system_prompt()
    assert "PULSAR CATALOG RULE" in prompt
    assert "search_pulsars" in prompt
    assert "pulsar_lookup" in prompt
    assert agent._tool_status_label("search_pulsars", {}) == "Searching ATNF pulsar catalogue"
    assert agent._tool_status_label("pulsar_lookup", {}) == "Looking up pulsar parameters"


def test_agent_wrappers_set_external_catalog_tables():
    _load_agent_module()
    agent = _make_agent()
    fake = FakePulsarService()
    agent._pulsar_catalog_service_instance = fake

    search = agent._search_pulsars(ra=53.2475, dec=54.5787, radius_deg=1.0, max_rows=5)
    assert search["success"] is True
    assert search["rowcount"] == 1
    assert search["results_preview"][0]["jname"] == "J0332+5434"
    assert agent.last_run_result["table_kind"] == "external_catalog"
    assert agent.last_run_result["tool_name"] == "search_pulsars"
    assert agent.last_run_result["source"] == "ATNF Pulsar Catalogue"
    assert fake.search_calls == [{"ra": 53.2475, "dec": 54.5787, "radius_deg": 1.0, "max_rows": 5}]

    lookup = agent._pulsar_lookup("B0329+54")
    assert lookup["success"] is True
    assert lookup["rowcount"] == 1
    assert lookup["results_preview"][0]["dm_pc_cm3"] == 26.76
    assert agent.last_run_result["tool_name"] == "pulsar_lookup"
    assert fake.lookup_calls == ["B0329+54"]