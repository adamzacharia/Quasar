"""Offline boot smoke — the app imports, the tool registry is intact (151 tools),
and all 7 capability families import, with no network and no running server.

Mirrors the CI import + tool-count guards (scripts/gen_tool_registry_doc.py,
.github/workflows/ci.yml) but runs in-process and offline.
"""

import os

# ── Offline env guards (mirror scripts/gen_tool_registry_doc._pin_offline_env +
#    tests/conftest defaults) BEFORE any api/core import. ──
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_API_KEY"] = ""
os.environ.setdefault("QUASAR_ENV", "testing")
os.environ.setdefault("QUASAR_FORCE_LOCAL_DB", "1")
os.environ.setdefault("DEFAULT_LLM_MODEL", "gpt-4.1")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("NASA_ADS_API_KEY", "test-dummy")

import importlib
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UI_PRO = os.path.join(PROJECT_ROOT, "ui-pro")
for _p in (PROJECT_ROOT, UI_PRO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The CI-guarded count in docs/TOOLS.md (kept in lockstep by
# scripts/gen_tool_registry_doc.py --check). 140 at S27 planning time; 151
# after the 2026-07 datalab/SX tool additions — see LEDGER S27 amendment.
EXPECTED_TOOL_COUNT = 151
CAPABILITY_FAMILIES = ["datalab", "alma", "papers", "archives", "vo", "viz", "calc"]


def test_app_imports_offline():
    from api.main import app
    assert app.__class__.__name__ == "FastAPI"


def test_tool_registry_count_matches_docs():
    from core.agent import QuasarAgent
    from core.tools import ToolRegistry
    agent = QuasarAgent.__new__(QuasarAgent)
    agent._tls = threading.local()
    agent.tool_registry = ToolRegistry()
    agent.last_search_results = None
    agent.last_run_result = None
    agent.ads_client = None
    agent.openalex_client = None
    agent._register_tools()
    assert len(agent.tool_registry.list_tools()) == EXPECTED_TOOL_COUNT


def test_all_capability_families_import():
    for fam in CAPABILITY_FAMILIES:
        mod = importlib.import_module(f"capabilities.{fam}")
        assert isinstance(getattr(mod, "CAPABILITIES", None), list)
        assert len(mod.CAPABILITIES) > 0
