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


def test_sky_footprints_extracted_from_s_region():
    """obscore s_region STC-S strings become per-observation footprints keyed by
    the same positional index as skyCoords; NaN/garbage regions are skipped (T7.2)."""
    api_main = load_api_main_module()
    df = pd.DataFrame([
        {"s_ra": 10.0, "s_dec": -2.0, "target_name": "A",
         "s_region": "Polygon ICRS 10.0 -2.0 10.1 -2.0 10.1 -1.9 10.0 -1.9"},
        {"s_ra": 10.2, "s_dec": -2.1, "target_name": "B",
         "s_region": "Circle J2000 10.2 -2.1 0.01"},
        {"s_ra": 10.3, "s_dec": -2.2, "target_name": "C", "s_region": None},           # skipped (null)
        {"s_ra": 10.4, "s_dec": -2.3, "target_name": "D", "s_region": "not a region"}, # skipped (not STC-S)
    ])

    demographics, _ = api_main._compute_demographics(df)
    footprints = demographics["skyFootprints"]

    assert [f["i"] for f in footprints] == [0, 1]
    assert footprints[0]["label"] == "A" and footprints[0]["stcs"].startswith("Polygon")
    assert footprints[1]["stcs"].startswith("Circle")
    assert "skyFootprintsTruncated" not in demographics

    # And it rides through to the serialized card payload the frontend reads.
    _, rich = api_main._build_data_card_event({
        "type": "data", "data": df, "source": "ALMA Archive", "tool_name": "search_alma_archive",
    })
    assert len(rich["demographics"]["skyFootprints"]) == 2


def test_sky_footprints_capped_and_flagged_over_limit():
    api_main = load_api_main_module()
    df = pd.DataFrame([
        {"s_ra": 10.0, "s_dec": -2.0, "target_name": f"T{k}",
         "s_region": "Polygon ICRS 10.0 -2.0 10.1 -2.0 10.1 -1.9"}
        for k in range(600)
    ])

    demographics, _ = api_main._compute_demographics(df)

    assert len(demographics["skyFootprints"]) == 512
    assert demographics["skyFootprintsTruncated"] is True


def test_no_s_region_column_yields_no_footprints():
    api_main = load_api_main_module()
    df = pd.DataFrame([{"s_ra": 10.0, "s_dec": -2.0, "target_name": "A"}])

    demographics, _ = api_main._compute_demographics(df)

    assert "skyFootprints" not in demographics


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
