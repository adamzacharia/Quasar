import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from tests.integration.test_agent_archive_tools import _load_agent_module
from tests.unit.test_datalab_p0 import _make_agent


class FakeService:
    def __init__(self, module, frame=None, available=True, reason="", error=None):
        self.module = module
        self.enabled = True
        self.frame = frame if frame is not None else pd.DataFrame({"source_id": [1], "ra": [10.0], "dec": [-2.0]})
        self.available = available
        self.reason = reason
        self.error = error
        self.calls = []

    def list_catalogs(self):
        return [
            {"key": "gaia", "label": "Gaia DR3 (MMU)", "uri": "hf://datasets/UniverseTBD/mmu_gaia_gaia@main/mmu_gaia_gaia", "description": "Gaia", "preferred_columns": ["source_id", "ra", "dec"]},
            {"key": "desi_edr_sv3", "label": "DESI EDR SV3 (MMU)", "uri": "hf://datasets/UniverseTBD/mmu_desi_edr_sv3@main/mmu_desi_edr_sv3", "description": "DESI", "preferred_columns": ["targetid", "ra", "dec"]},
            {"key": "sdss", "label": "SDSS (MMU)", "uri": "hf://datasets/UniverseTBD/mmu_sdss_sdss@main/mmu_sdss_sdss", "description": "SDSS", "preferred_columns": ["ra", "dec"]},
            {"key": "tess_spoc", "label": "TESS SPOC (MMU)", "uri": "hf://datasets/UniverseTBD/mmu_tess_spoc@main/mmu_tess_spoc", "description": "TESS", "preferred_columns": ["ra", "dec"]},
            {"key": "chandra_spectra", "label": "Chandra Spectra (MMU)", "uri": "hf://datasets/UniverseTBD/mmu_chandra_spectra@main/mmu_chandra_spectra", "description": "Chandra", "preferred_columns": ["ra", "dec"]},
        ]

    def is_available(self):
        return self.available, self.reason

    def cone_search(self, catalog_key, **kwargs):
        self.calls.append((catalog_key, kwargs))
        if self.error:
            raise self.error
        return {
            "dataframe": self.frame,
            "rowcount": len(self.frame),
            "returned_rows": len(self.frame),
            "columns": list(self.frame.columns),
            "warnings": [],
            "provenance": {
                "backend": "lsdb+huggingface",
                "catalog_key": catalog_key,
                "catalog_label": "Gaia DR3 (MMU)",
                "uri": "hf://datasets/UniverseTBD/mmu_gaia_gaia@main/mmu_gaia_gaia",
                "cone": {"ra_deg": float(kwargs.get("ra")), "dec_deg": float(kwargs.get("dec")), "radius_arcsec": 120.0},
                "max_rows": kwargs.get("max_rows") or 500,
                "selected_columns": list(self.frame.columns),
            },
        }

    def crossmatch_catalogs(self, left_catalog_key, right_catalog_key, **kwargs):
        self.calls.append((left_catalog_key, right_catalog_key, kwargs))
        if self.error:
            raise self.error
        frame = pd.DataFrame({"ra": [10.0], "dec": [-2.0], "source_id_gaia": [1], "targetid_desi_edr_sv3": [2]})
        return {
            "dataframe": frame,
            "rowcount": 1,
            "returned_rows": 1,
            "columns": list(frame.columns),
            "warnings": [],
            "provenance": {
                "backend": "lsdb+huggingface",
                "left_catalog_key": left_catalog_key,
                "right_catalog_key": right_catalog_key,
                "left_catalog_label": "Gaia DR3 (MMU)",
                "right_catalog_label": "DESI EDR SV3 (MMU)",
                "cone": {"ra_deg": float(kwargs.get("ra")), "dec_deg": float(kwargs.get("dec")), "radius_arcsec": 120.0},
                "match_radius_arcsec": kwargs.get("match_radius_arcsec") or 1.0,
            },
        }


def load_api_main_module():
    pytest.importorskip("fastapi")

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    stub = types.SimpleNamespace(logger=DummyLogger(), init_rollbar=lambda: None)
    original = sys.modules.get("core.logger")
    sys.modules["core.logger"] = stub
    try:
        path = Path(__file__).resolve().parents[2] / "ui-pro" / "api" / "main.py"
        spec = importlib.util.spec_from_file_location("quasar_api_main_for_mmu_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if original is None:
            sys.modules.pop("core.logger", None)
        else:
            sys.modules["core.logger"] = original


def test_search_mmu_hats_catalog_success_sets_data_card_and_small_payload():
    module = _load_agent_module()
    agent = _make_agent()
    frame = pd.DataFrame({"source_id": range(5000), "ra": [10.0] * 5000, "dec": [-2.0] * 5000, "long_text": ["x" * 200] * 5000})
    agent._mmu_hats_service_instance = FakeService(module, frame=frame)

    out = agent._search_mmu_hats_catalog(catalog_key="gaia", ra=10.0, dec=-2.0)

    assert out["success"] is True
    assert agent.last_run_result["type"] == "data"
    assert agent.last_run_result["table_kind"] == "mmu_hats"
    assert agent.last_run_result["tool_name"] == "search_mmu_hats_catalog"
    assert isinstance(agent.last_run_result["data"], pd.DataFrame)
    assert agent.last_run_result["source"].startswith("Multimodal Universe /")
    assert {"rowcount", "returned_rows", "columns", "warnings", "provenance", "note", "results_preview"} <= set(out)
    assert len(out["results_preview"]) <= 10
    assert "Full table" in out["note"]
    assert len(json.dumps(out, default=str)) < 8000


def test_search_target_name_resolution_path():
    module = _load_agent_module()
    agent = _make_agent()
    fake = FakeService(module)
    agent._mmu_hats_service_instance = fake
    agent._resolve_target = lambda name: {"success": True, "ra_deg": 12.5, "dec_deg": -1.5}

    out = agent._search_mmu_hats_catalog(catalog_key="gaia", target_name="M87")

    assert out["success"] is True
    assert fake.calls[0][1]["ra"] == 12.5
    assert fake.calls[0][1]["dec"] == -1.5


def test_validation_error_leaves_last_run_result_none():
    module = _load_agent_module()
    agent = _make_agent()
    agent._mmu_hats_service_instance = FakeService(module, error=module.MMUHatsError("bad input"))

    out = agent._search_mmu_hats_catalog(catalog_key="gaia", ra=10.0, dec=0.0)

    assert out == {"success": False, "error": "bad input"}
    assert agent.last_run_result is None


def test_unavailable_service_sets_unavailable_flag():
    module = _load_agent_module()
    agent = _make_agent()
    err = module.MMUHatsUnavailableError("install lsdb==0.9.2 and huggingface_hub or set ENABLE_MMU_HATS=true")
    agent._mmu_hats_service_instance = FakeService(module, error=err)

    out = agent._search_mmu_hats_catalog(catalog_key="gaia", ra=10.0, dec=0.0)

    assert out["success"] is False
    assert out["unavailable"] is True
    assert "lsdb" in out["error"] and "ENABLE_MMU_HATS" in out["error"]
    assert agent.last_run_result is None


def test_list_mmu_hats_catalogs_works_when_unavailable_and_clears_card():
    module = _load_agent_module()
    agent = _make_agent()
    agent.last_run_result = {"type": "data", "stale": True}
    agent._mmu_hats_service_instance = FakeService(module, available=False, reason="missing lsdb")

    out = agent._list_mmu_hats_catalogs()

    assert out["success"] is True
    assert out["available"] is False
    assert out["reason"] == "missing lsdb"
    assert out["count"] == 5
    assert agent.last_run_result is None


def test_tool_registration_category_and_prompt_rule():
    agent = _make_agent()
    agent._register_tools()

    for name in ["list_mmu_hats_catalogs", "search_mmu_hats_catalog", "crossmatch_mmu_hats_catalogs"]:
        tool = agent.tool_registry.get_tool(name)
        assert tool is not None
        assert tool.category == "mmu_hats"
        assert name in agent.tool_registry.categories["mmu_hats"]

    desc = agent.tool_registry.get_tool("search_mmu_hats_catalog").description.lower()
    assert "catalog/source properties" in desc
    assert "archive observations" in desc and "do not" in desc
    assert "MMU/HATS CATALOG RULE" in agent._build_system_prompt()


def test_dispatch_tool_call_keeps_json_valid_under_8000_chars():
    module = _load_agent_module()
    agent = _make_agent()
    agent._register_tools()
    frame = pd.DataFrame({"source_id": range(5000), "ra": [1.0] * 5000, "dec": [2.0] * 5000, "blob": ["y" * 500] * 5000})
    agent._mmu_hats_service_instance = FakeService(module, frame=frame)

    raw = agent._dispatch_tool_call("search_mmu_hats_catalog", json.dumps({"catalog_key": "gaia", "ra": 1.0, "dec": 2.0}))
    payload = json.loads(raw)

    assert payload["success"] is True
    assert len(raw) < 8000
    assert "results_preview" in payload


def test_search_with_nested_cells_keeps_payload_small():
    module = _load_agent_module()
    agent = _make_agent()
    frame = pd.DataFrame({
        "source_id": range(50),
        "ra": [1.0] * 50,
        "dec": [2.0] * 50,
        "spectrum": [list(range(20000))] * 50,
    })
    agent._mmu_hats_service_instance = FakeService(module, frame=frame)

    out = agent._search_mmu_hats_catalog(catalog_key="gaia", ra=1.0, dec=2.0, columns=["source_id", "spectrum"])

    assert out["success"] is True
    assert len(json.dumps(out, default=str)) < 8000
    assert all(isinstance(row.get("spectrum"), str) for row in out["results_preview"])


def test_crossmatch_mmu_hats_catalogs_sets_data_card():
    module = _load_agent_module()
    agent = _make_agent()
    agent._mmu_hats_service_instance = FakeService(module)

    out = agent._crossmatch_mmu_hats_catalogs("gaia", "desi_edr_sv3", ra=10.0, dec=-2.0)

    assert out["success"] is True
    assert out["source"].startswith("Multimodal Universe / Gaia")
    assert agent.last_run_result["table_kind"] == "mmu_hats"
    assert agent.last_run_result["tool_name"] == "crossmatch_mmu_hats_catalogs"


def test_mmu_data_card_columns_and_no_fits_estimate():
    api_main = load_api_main_module()
    df = pd.DataFrame([
        {"source_id": 1, "ra": 10.0, "dec": -2.0, "obsid": "123", "exposure": 45.0, "custom_col": "kept"}
    ])

    _, rich = api_main._build_data_card_event({
        "type": "data",
        "data": df,
        "source": "Multimodal Universe / Chandra Spectra (MMU)",
        "tool_name": "search_mmu_hats_catalog",
        "table_kind": "mmu_hats",
    })

    assert "ObsID" in rich["columns"]
    assert "Exposure" in rich["columns"]
    assert "custom_col" in rich["columns"]
    assert rich["tableKind"] == "mmu_hats"
    assert rich["fitsEstimate"] is None
    assert rich["hasPreview"] is True
