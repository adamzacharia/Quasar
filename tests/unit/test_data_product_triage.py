import threading

import pandas as pd

from core.agent import QuasarAgent
from services.data_product_triage import (
    build_product_row,
    classify_alma_product_request,
    summarize_project_options,
)


def test_classifies_project_code_and_target_band():
    exact = classify_alma_product_request("Fetch ALMA products for project 2016.1.00484.L")
    assert exact["kind"] == "project_code"
    assert exact["identifier"] == "2016.1.00484.L"

    target = classify_alma_product_request("Triage M87 Band 6 FITS products")
    assert target["kind"] == "target"
    assert target["target"] == "M87"
    assert target["band"] == "6"


def test_project_picker_groups_options():
    df = pd.DataFrame([
        {"proposal_id": "2019.1.00001.S", "target_name": "M87", "band_list": "6", "member_ous_uid": "uid://A/X1/X1"},
        {"proposal_id": "2019.1.00001.S", "target_name": "M87", "band_list": "7", "member_ous_uid": "uid://A/X1/X2"},
        {"proposal_id": "2021.1.00002.S", "target_name": "M87", "band_list": "3", "member_ous_uid": "uid://A/X2/X1"},
    ])

    picker = summarize_project_options(df)

    assert list(picker["proposal_id"]) == ["2019.1.00001.S", "2021.1.00002.S"]
    assert picker.iloc[0]["member_ous_count"] == 2
    assert picker.iloc[0]["observations"] == 2


def test_project_picker_shows_all_projects_by_default():
    df = pd.DataFrame([
        {
            "proposal_id": f"2024.1.{index:05d}.S",
            "target_name": "M87",
            "band_list": "6",
            "member_ous_uid": f"uid://A/X{index}/X1",
        }
        for index in range(15)
    ])

    picker = summarize_project_options(df)

    assert len(picker) == 15


def test_product_row_scores_header_checked_fits():
    row = build_product_row(
        {
            "filename": "target.pbcor.fits",
            "size_mb": 12.5,
            "access_url": "https://example.test/target.pbcor.fits",
            "content_type": "application/fits",
        },
        member_ous_uid="uid://A/X1/X1",
        proposal_id="2016.1.00484.L",
        target_name="AS 209",
        metadata={
            "success": True,
            "image_size": "512 x 512",
            "beam_major_arcsec": 0.04,
            "beam_minor_arcsec": 0.03,
            "bunit": "Jy/beam",
            "all_headers": {"CTYPE1": "RA---SIN", "CTYPE2": "DEC--SIN"},
        },
    )

    assert row["product_kind"] == "primary-beam-corrected FITS"
    assert row["triage_status"] == "header checked"
    assert row["readiness_score"] == 100
    assert row["warnings"] == ""


class FakeSearchService:
    def __init__(self, df):
        self.df = df
        self.keyword_calls = []
        self.target_calls = []

    def search_alma_with_keywords(self, keywords):
        self.keyword_calls.append(keywords)
        project_code = keywords.get("project_code") or keywords.get("proposal_id")
        if project_code and "proposal_id" in self.df.columns:
            return self.df[self.df["proposal_id"] == project_code].copy()
        return self.df

    def advanced_search(self, query):
        return self.df

    def search_by_target(self, target_name, facility=None, max_results=100):
        self.target_calls.append((target_name, facility, max_results))
        return self.df


class FakeDataLinkClient:
    def list_files(self, mous_uid, pattern=None):
        return {
            "success": True,
            "mous_uid": mous_uid,
            "files": [
                {
                    "filename": "science.pbcor.fits",
                    "size_mb": 22,
                    "access_url": "https://example.test/science.pbcor.fits",
                    "content_type": "application/fits",
                },
                {
                    "filename": "calibration.tar",
                    "size_mb": 1200,
                    "access_url": "https://example.test/calibration.tar",
                    "content_type": "application/x-tar",
                },
            ],
        }


class FakeFitsService:
    def extract_metadata_from_url(self, access_url):
        return {
            "success": True,
            "object_name": "AS 209",
            "beam_major_arcsec": 0.04,
            "beam_minor_arcsec": 0.03,
            "rest_freq_ghz": 230.538,
            "image_size": "512 x 512",
            "bunit": "Jy/beam",
            "all_headers": {"CTYPE1": "RA---SIN", "CTYPE2": "DEC--SIN", "CTYPE3": "FREQ"},
        }


def make_agent(df):
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent._tls.current_conversation_id = "unit-conversation"
    agent._alma_project_picker_by_conversation = {}
    agent._alma_project_picker_lock = threading.Lock()
    agent.search_service = FakeSearchService(df)
    agent.datalink_client = FakeDataLinkClient()
    agent.fits_service = FakeFitsService()
    return agent


def test_exact_project_code_triages_products_directly():
    df = pd.DataFrame([
        {
            "proposal_id": "2016.1.00484.L",
            "target_name": "AS 209",
            "band_list": "6",
            "member_ous_uid": "uid://A/X1/X1",
        }
    ])
    agent = make_agent(df)

    result = agent._triage_alma_data_products("Fetch products for project 2016.1.00484.L")

    assert result["success"] is True
    assert result["mode"] == "triage"
    assert result["total_products_found"] == 2
    assert result["header_checks"] == 1
    assert agent.search_service.keyword_calls[0] == {"project_code": "2016.1.00484.L"}
    assert agent.last_run_result["table_kind"] == "alma_products"
    assert len(agent.last_run_result["data"]) == 2


def test_generic_target_returns_project_picker():
    df = pd.DataFrame([
        {
            "proposal_id": "2019.1.00001.S",
            "target_name": "M87",
            "band_list": "6",
            "member_ous_uid": "uid://A/X1/X1",
        },
        {
            "proposal_id": "2021.1.00002.S",
            "target_name": "M87",
            "band_list": "7",
            "member_ous_uid": "uid://A/X2/X1",
        },
    ])
    agent = make_agent(df)

    result = agent._triage_alma_data_products("Fetch ALMA data products for M87")

    assert result["success"] is True
    assert result["mode"] == "needs_project_selection"
    assert result["project_count"] == 2
    assert agent.search_service.target_calls[0][0] == "M87"
    assert agent.last_run_result["table_kind"] == "alma_project_picker"
    assert list(agent.last_run_result["data"]["proposal_id"]) == ["2019.1.00001.S", "2021.1.00002.S"]


def test_project_picker_row_selection_triages_selected_project():
    df = pd.DataFrame([
        {
            "proposal_id": "2019.1.00001.S",
            "target_name": "M87",
            "band_list": "6",
            "member_ous_uid": "uid://A/X1/X1",
        },
        {
            "proposal_id": "2021.1.00002.S",
            "target_name": "M87",
            "band_list": "6",
            "member_ous_uid": "uid://A/X2/X1",
        },
    ])
    agent = make_agent(df)

    picker = agent._triage_alma_data_products("Fetch ALMA data products for M87")
    result = agent._triage_alma_data_products("use #2")

    assert picker["mode"] == "needs_project_selection"
    assert result["success"] is True
    assert result["mode"] == "triage"
    assert result["project_code"] == "2021.1.00002.S"
    assert agent.search_service.keyword_calls[-1] == {"project_code": "2021.1.00002.S"}
    assert agent.last_run_result["table_kind"] == "alma_products"
