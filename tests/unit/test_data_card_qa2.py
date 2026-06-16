import importlib.util
from pathlib import Path

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


def test_alma_observation_data_card_includes_qa2_column():
    api_main = load_api_main_module()

    df = pd.DataFrame([
        {
            "obs_publisher_did": "ADS/JAO.ALMA#2017.1.00001.S",
            "target_name": "Sz65",
            "scan_intent": "TARGET",
            "qa2_passed": "T",
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
    assert "Scan Intent" in rich["columns"]
    assert rich["rows"][0]["QA2"] == "Pass"
    assert rich["rows"][0]["Scan Intent"] == "TARGET"
    assert rich["partial"] is False


def test_alma_observation_qa2_false_maps_to_semipass():
    api_main = load_api_main_module()

    df = pd.DataFrame([
        {
            "obs_publisher_did": "ADS/JAO.ALMA#2017.1.00569.S",
            "target_name": "Sz65",
            "scan_intent": "TARGET",
            "qa2_passed": "F",
            "member_ous_uid": "uid://A001/X12a3/X407",
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

    assert rich["rows"][0]["QA2"] == "SemiPass"


def test_alma_product_data_card_includes_qa2_column():
    api_main = load_api_main_module()

    df = pd.DataFrame([
        {
            "filename": "science.pbcor.fits",
            "product_kind": "primary-beam-corrected FITS",
            "size_mb": 22,
            "proposal_id": "2017.1.00001.S",
            "target_name": "Sz65",
            "scan_intent": "TARGET",
            "qa2_status": "SemiPass",
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
    assert "Scan Intent" in rich["columns"]
    assert rich["rows"][0]["QA2"] == "SemiPass"
    assert rich["rows"][0]["Scan Intent"] == "TARGET"
