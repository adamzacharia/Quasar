import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def load_api_main_module():
    pytest.importorskip("fastapi")
    path = Path(__file__).resolve().parents[2] / "ui-pro" / "api" / "main.py"
    spec = importlib.util.spec_from_file_location("quasar_api_main_for_qa2_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_alma_observation_data_card_includes_qa2_column(monkeypatch):
    api_main = load_api_main_module()
    lookup = SimpleNamespace(
        statuses={"A001_X1_X1": "Pass"},
        requested=1,
        incomplete_count=0,
        capped=False,
        timed_out=False,
        errors={},
    )
    monkeypatch.setattr(api_main, "fetch_qa2_statuses", lambda *args, **kwargs: lookup)

    df = pd.DataFrame([
        {
            "obs_publisher_did": "ADS/JAO.ALMA#2017.1.00001.S",
            "target_name": "Sz65",
            "member_ous_uid": "uid://A001/X1/X1",
            "obs_collection": "ALMA",
            "instrument_name": "ALMA",
            "band_list": "6",
        }
    ])

    _, rich = api_main._build_data_card_event({
        "type": "data",
        "data": df,
        "source": "ALMA Archive",
        "tool_name": "search_alma_archive",
    })

    assert "QA2" in rich["columns"]
    assert rich["rows"][0]["QA2"] == "Pass"
    assert rich["partial"] is False


def test_alma_product_data_card_includes_qa2_column(monkeypatch):
    api_main = load_api_main_module()
    lookup = SimpleNamespace(
        statuses={"A001_X2_X1": "SemiPass"},
        requested=1,
        incomplete_count=0,
        capped=False,
        timed_out=False,
        errors={},
    )
    monkeypatch.setattr(api_main, "fetch_qa2_statuses", lambda *args, **kwargs: lookup)

    df = pd.DataFrame([
        {
            "filename": "science.pbcor.fits",
            "product_kind": "primary-beam-corrected FITS",
            "size_mb": 22,
            "proposal_id": "2017.1.00001.S",
            "target_name": "Sz65",
            "member_ous_uid": "uid://A001/X2/X1",
            "access_url": "https://almascience.nrao.edu/dataPortal/science.pbcor.fits",
        }
    ])

    _, rich = api_main._build_data_card_event({
        "type": "data",
        "data": df,
        "source": "ALMA Data Products",
        "tool_name": "triage_alma_data_products",
        "table_kind": "alma_products",
    })

    assert "QA2" in rich["columns"]
    assert rich["rows"][0]["QA2"] == "SemiPass"
