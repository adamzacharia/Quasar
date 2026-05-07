import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

from core.tools import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_MODULE_NAME = "quasar_agent_test_module"


def _dummy_class(name):
    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

    Dummy.__name__ = name
    return Dummy


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_agent_import_stubs():
    dummy = _dummy_class

    prompts_module = _stub_module(
        "core.prompts",
        INTENT_CLASSIFICATION_PROMPT="",
        ENTITY_EXTRACTION_PROMPT="",
        RESPONSE_GENERATION_PROMPT="",
        ALMA_TAP_SCHEMA="",
    )
    prompts_module.__path__ = []

    stubs = {
        "openai": _stub_module("openai", OpenAI=dummy("OpenAI")),
        "core.llm_client": _stub_module(
            "core.llm_client",
            LLMClient=dummy("LLMClient"),
            detect_provider=lambda model: "local",
        ),
        "core.logger": _stub_module(
            "core.logger",
            logger=_DummyLogger(),
            log_tool=lambda *args, **kwargs: None,
        ),
        "core.memory": _stub_module("core.memory", ConversationMemory=dummy("ConversationMemory")),
        "core.prompts": prompts_module,
        "core.prompts.lit_to_code": _stub_module(
            "core.prompts.lit_to_code",
            LIT_TO_CODE_PROMPT="",
        ),
        "integrations.datalink": _stub_module("integrations.datalink", DataLinkClient=dummy("DataLinkClient")),
        "integrations.ads_client": _stub_module("integrations.ads_client", ADSService=dummy("ADSService")),
        "integrations.mast_client": _stub_module("integrations.mast_client", MASTClient=dummy("MASTClient")),
        "integrations.eso_tap_client": _stub_module("integrations.eso_tap_client", ESOTAPClient=dummy("ESOTAPClient")),
        "integrations.irsa_client": _stub_module("integrations.irsa_client", IRSAClient=dummy("IRSAClient")),
        "integrations.skyview_client": _stub_module("integrations.skyview_client", SkyViewClient=dummy("SkyViewClient")),
        "services.search": _stub_module("services.search", SearchService=dummy("SearchService")),
        "services.analysis": _stub_module("services.analysis", RadioAnalysisService=dummy("RadioAnalysisService")),
        "services.rag_service": _stub_module("services.rag_service", RAGService=dummy("RAGService")),
        "services.memory_service": _stub_module("services.memory_service", MemoryService=dummy("MemoryService")),
        "core.rlm": _stub_module("core.rlm", RecursiveLanguageModel=dummy("RecursiveLanguageModel")),
        "services.browser": _stub_module("services.browser", BrowserService=dummy("BrowserService")),
        "services.plotting": _stub_module("services.plotting", PlottingService=dummy("PlottingService")),
        "services.splatalogue": _stub_module("services.splatalogue", SplatalogueTool=dummy("SplatalogueTool")),
        "services.multi_archive": _stub_module("services.multi_archive", MultiArchiveMatcher=dummy("MultiArchiveMatcher")),
        "services.casa_generator": _stub_module("services.casa_generator", CASAScriptGenerator=dummy("CASAScriptGenerator")),
        "services.gcn_monitor": _stub_module("services.gcn_monitor", GCNAlertMonitor=dummy("GCNAlertMonitor")),
        "services.notebook_gen": _stub_module(
            "services.notebook_gen",
            generate_analysis_notebook=lambda *args, **kwargs: {},
        ),
        "services.pdf_processing": _stub_module("services.pdf_processing", PDFProcessingService=dummy("PDFProcessingService")),
        "services.fits_processing": _stub_module("services.fits_processing", FITSProcessingService=dummy("FITSProcessingService")),
        "core.conductor": _stub_module("core.conductor", Conductor=dummy("Conductor")),
        "core.model_router": _stub_module("core.model_router", ModelRouter=dummy("ModelRouter")),
        "core.recovery": _stub_module("core.recovery", RecoveryEngine=dummy("RecoveryEngine")),
        "core.observability": _stub_module("core.observability", QueryTracer=dummy("QueryTracer")),
        "core.agent_pool": _stub_module("core.agent_pool", AgentPool=dummy("AgentPool")),
        "core.context_manager": _stub_module("core.context_manager", ContextManager=dummy("ContextManager")),
        "core.session_memory": _stub_module("core.session_memory", SessionMemory=dummy("SessionMemory")),
        "core.token_budget": _stub_module(
            "core.token_budget",
            TokenBudget=dummy("TokenBudget"),
            apply_tool_result_budget=lambda results: results,
        ),
        "core.health_monitor": _stub_module("core.health_monitor", HealthMonitor=dummy("HealthMonitor")),
    }

    originals = {}
    for name, module in stubs.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module
    return originals


def _restore_modules(originals):
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _load_agent_module():
    if AGENT_MODULE_NAME in sys.modules:
        return sys.modules[AGENT_MODULE_NAME]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    originals = _install_agent_import_stubs()
    try:
        spec = importlib.util.spec_from_file_location(
            AGENT_MODULE_NAME,
            PROJECT_ROOT / "core" / "agent.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[AGENT_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules(originals)


class _StubMASTClient:
    def __init__(self, search_df=None, criteria_df=None, product_df=None):
        self.search_df = search_df if search_df is not None else pd.DataFrame()
        self.criteria_df = criteria_df if criteria_df is not None else pd.DataFrame()
        self.product_df = product_df if product_df is not None else pd.DataFrame()

    def search_by_target(self, **kwargs):
        return self.search_df.copy()

    def search_by_position(self, **kwargs):
        return self.search_df.copy()

    def search_by_criteria(self, **kwargs):
        return self.criteria_df.copy()

    def get_product_list(self, observations, productType=None, extension=None):
        return self.product_df.copy()


class _StubESOClient:
    def __init__(self, df):
        self.df = df

    def search_by_target(self, **kwargs):
        return self.df.copy()

    def search_by_position(self, **kwargs):
        return self.df.copy()


class _StubIRSAClient:
    def __init__(self, df):
        self.df = df

    def search_by_target(self, **kwargs):
        return self.df.copy()

    def search_by_position(self, **kwargs):
        return self.df.copy()


class AgentArchiveToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_module = _load_agent_module()

    def _make_agent(self):
        agent = self.agent_module.QuasarAgent.__new__(self.agent_module.QuasarAgent)
        agent.tool_registry = ToolRegistry()
        agent.last_search_results = None
        agent.last_run_result = None
        agent.ads_client = None
        return agent

    def test_register_tools_includes_new_archive_tools(self):
        agent = self._make_agent()

        agent._register_tools()

        for tool_name in [
            "search_mast",
            "search_mast_by_criteria",
            "get_mast_products",
            "search_eso_archive",
            "search_irsa",
        ]:
            self.assertIsNotNone(agent.tool_registry.get_tool(tool_name))

    def test_search_mast_sets_last_search_results(self):
        df = pd.DataFrame(
            [
                {
                    "target_name": "Carina Nebula",
                    "telescope": "JWST",
                    "instrument_name": "NIRCAM",
                    "project_code": "1345",
                }
            ]
        )
        agent = self._make_agent()
        agent.mast_client = _StubMASTClient(search_df=df)

        result = agent._search_mast(target_name="Carina Nebula", mission="JWST")

        self.assertTrue(result["success"])
        self.assertEqual(result["total_results"], 1)
        self.assertEqual(agent.last_run_result["tool_name"], "search_mast")
        pd.testing.assert_frame_equal(agent.last_search_results, df)

    def test_search_mast_by_criteria_sets_last_search_results(self):
        df = pd.DataFrame(
            [
                {
                    "target_name": "Carina Nebula",
                    "telescope": "JWST",
                    "instrument_name": "NIRCAM",
                    "filters": "F200W",
                }
            ]
        )
        agent = self._make_agent()
        agent.mast_client = _StubMASTClient(criteria_df=df)

        result = agent._search_mast_by_criteria(
            mission="JWST",
            filters="F200W",
            target_name="Carina Nebula",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["total_results"], 1)
        self.assertEqual(agent.last_run_result["tool_name"], "search_mast_by_criteria")
        pd.testing.assert_frame_equal(agent.last_search_results, df)

    def test_get_mast_products_requires_mast_observation_results(self):
        agent = self._make_agent()
        agent.last_search_results = pd.DataFrame([{"target_name": "M87"}])
        agent.mast_client = _StubMASTClient()

        result = agent._get_mast_products()

        self.assertFalse(result["success"])
        self.assertIn("not MAST observation results", result["error"])

    def test_get_mast_products_caches_product_table(self):
        observation_df = pd.DataFrame(
            [
                {
                    "obsid": 101,
                    "obs_id": "jw12345",
                    "target_name": "M87",
                    "telescope": "JWST",
                }
            ]
        )
        product_df = pd.DataFrame(
            [
                {
                    "productFilename": "jw12345_cal.fits",
                    "productType": "SCIENCE",
                    "obs_collection": "JWST",
                }
            ]
        )
        agent = self._make_agent()
        agent.last_search_results = observation_df
        agent.mast_client = _StubMASTClient(product_df=product_df)

        result = agent._get_mast_products(product_type="SCIENCE")

        self.assertTrue(result["success"])
        self.assertEqual(result["total_products"], 1)
        self.assertEqual(agent.last_run_result["tool_name"], "get_mast_products")
        pd.testing.assert_frame_equal(agent.last_search_results, product_df)

    def test_search_eso_sets_last_search_results(self):
        df = pd.DataFrame(
            [
                {
                    "target_name": "NGC 1068",
                    "instrument_name": "MUSE",
                    "dataproduct_type": "cube",
                }
            ]
        )
        agent = self._make_agent()
        agent.eso_client = _StubESOClient(df)

        result = agent._search_eso(target_name="NGC 1068", instrument="MUSE")

        self.assertTrue(result["success"])
        self.assertEqual(result["total_results"], 1)
        self.assertEqual(agent.last_run_result["tool_name"], "search_eso_archive")
        pd.testing.assert_frame_equal(agent.last_search_results, df)

    def test_search_irsa_sets_last_search_results(self):
        df = pd.DataFrame(
            [
                {
                    "target_name": "M31",
                    "telescope": "IRSA",
                    "instrument_name": "ALLWISE_P3AS_PSD",
                }
            ]
        )
        agent = self._make_agent()
        agent.irsa_client = _StubIRSAClient(df)

        result = agent._search_irsa(target_name="M31", catalog="allwise")

        self.assertTrue(result["success"])
        self.assertEqual(result["total_results"], 1)
        self.assertEqual(agent.last_run_result["tool_name"], "search_irsa")
        pd.testing.assert_frame_equal(agent.last_search_results, df)


if __name__ == "__main__":
    unittest.main()
