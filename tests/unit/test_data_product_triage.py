import threading
import sys
import types

import pandas as pd


_original_modules = {}
_stubbed_names = []


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if name not in _original_modules:
        _original_modules[name] = sys.modules.get(name)
    _stubbed_names.append(name)
    sys.modules[name] = module
    return module


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


def _identity_decorator(fn):
    return fn


_dummy_logger = types.SimpleNamespace(
    debug=lambda *args, **kwargs: None,
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
    success=lambda *args, **kwargs: None,
)

_stub_module("core.logger", logger=_dummy_logger, log_tool=_identity_decorator)
_stub_module("core.llm_client", LLMClient=_Dummy, detect_provider=lambda *args, **kwargs: "openai")
_stub_module("core.memory", ConversationMemory=_Dummy)
_stub_module("core.tools", ToolRegistry=_Dummy, Tool=_Dummy)
_stub_module("core.complexity", ComplexityDetector=_Dummy)
_stub_module("core.sandbox", SandboxExecutor=_Dummy)
_stub_module("core.conductor", Conductor=_Dummy)
_stub_module("core.model_router", ModelRouter=_Dummy)
_stub_module("core.recovery", RecoveryEngine=_Dummy)
_stub_module("core.observability", QueryTracer=_Dummy)
_stub_module("core.agent_pool", AgentPool=_Dummy)
_stub_module("core.context_manager", ContextManager=_Dummy)
_stub_module("core.session_memory", SessionMemory=_Dummy)
_stub_module("core.token_budget", TokenBudget=_Dummy, apply_tool_result_budget=lambda result, *args, **kwargs: result)
_stub_module("core.health_monitor", HealthMonitor=_Dummy)
_stub_module("core.prompts.lit_to_code", LIT_TO_CODE_PROMPT="")
_stub_module("services.ads_auto_link", build_exact_project_paper_links=lambda *args, **kwargs: None)
_stub_module("services.evidence_quality", annotate_web_source_evidence=lambda value: value, rank_web_sources=lambda value: value)
_stub_module("services.search", SearchService=_Dummy)
_stub_module("services.analysis", RadioAnalysisService=_Dummy)
_stub_module("services.rag_service", RAGService=_Dummy)
_stub_module("services.memory_service", MemoryService=_Dummy)
_stub_module("services.browser", BrowserService=_Dummy)
_stub_module("services.plotting", PlottingService=_Dummy)
_stub_module("services.splatalogue", SplatalogueTool=_Dummy)
_stub_module("services.multi_archive", MultiArchiveMatcher=_Dummy)
_stub_module("services.casa_generator", CASAScriptGenerator=_Dummy)
_stub_module("services.gcn_monitor", GCNAlertMonitor=_Dummy)
_stub_module("services.notebook_gen", generate_analysis_notebook=lambda *args, **kwargs: {})
_stub_module("services.pdf_processing", PDFProcessingService=_Dummy)
_stub_module("services.fits_processing", FITSProcessingService=_Dummy)
_stub_module(
    "services.alma_science_queries",
    LINE_REST_FREQ_GHZ={},
    bandwidth_switching_candidates=lambda *args, **kwargs: pd.DataFrame(),
    filter_band=lambda df, *args, **kwargs: df,
    filter_resolution=lambda df, *args, **kwargs: df,
    line_names_for_species=lambda *args, **kwargs: [],
    normalize_target_alias=lambda value: value,
    projects_covering_all_lines=lambda *args, **kwargs: pd.DataFrame(),
    projects_with_array_combo=lambda *args, **kwargs: pd.DataFrame(),
    project_prefix_where=lambda *args, **kwargs: "",
    redshifted_line_projects=lambda *args, **kwargs: pd.DataFrame(),
    select_obscore_query=lambda *args, **kwargs: "",
    summarize_projects=lambda *args, **kwargs: pd.DataFrame(),
)
_stub_module(
    "services.cross_archive_matcher",
    PERSEUS_PROTOSTARS=[],
    alma_bulk_cone_adql=lambda *args, **kwargs: "",
    attach_nearest_source=lambda *args, **kwargs: pd.DataFrame(),
    normalize_source_catalog=lambda *args, **kwargs: pd.DataFrame(),
    summarize_cross_archive_matches=lambda *args, **kwargs: {},
)
_stub_module("integrations.datalink", DataLinkClient=_Dummy)
_stub_module("integrations.ads_client", ADSService=_Dummy)
_stub_module("integrations.openalex_client", OpenAlexService=_Dummy)
_stub_module("integrations.mast_client", MASTClient=_Dummy)
_stub_module("integrations.eso_tap_client", ESOTAPClient=_Dummy)
_stub_module("integrations.irsa_client", IRSAClient=_Dummy)
_stub_module("integrations.skyview_client", SkyViewClient=_Dummy)
_stub_module(
    "services.astro_calculators",
    calculate_redshift=lambda *args, **kwargs: {},
    convert_coordinates=lambda *args, **kwargs: {},
    calculate_beam=lambda *args, **kwargs: {},
    calculate_alma_sensitivity=lambda *args, **kwargs: {},
)

from core.agent import QuasarAgent
from services.data_product_triage import (
    build_product_row,
    classify_alma_product_request,
    summarize_project_options,
)

# Clean up stubs from sys.modules so they don't pollute other tests during collection/execution
for name in _stubbed_names:
    orig = _original_modules[name]
    if orig is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = orig


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


def test_scan_intent_filter_keeps_target_rows():
    df = pd.DataFrame([
        {"target_name": "M87", "scan_intent": "TARGET"},
        {"target_name": "J1924-2914", "scan_intent": "BANDPASS FLUX WVR"},
    ])

    filtered, label = QuasarAgent._filter_by_scan_intent(df, "TARGET only")

    assert label == "Scan Intent TARGET"
    assert filtered["target_name"].tolist() == ["M87"]


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
            "scan_intent": "TARGET",
            "qa2_passed": "T",
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
    assert set(agent.last_run_result["data"]["scan_intent"]) == {"TARGET"}
    assert set(agent.last_run_result["data"]["qa2_passed"]) == {"T"}


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
