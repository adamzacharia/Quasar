"""
Memory profiler — measures actual RSS usage of each Quasar component.
Run:  python measure_memory.py
"""
import os, sys, gc

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows fix stubs (same as main.py)
if sys.platform == "win32" and "pwd" not in sys.modules:
    import types
    _pwd_stub = types.ModuleType("pwd")
    _pwd_stub.getpwuid = lambda uid: type("pw", (), {"pw_name": "user"})()
    sys.modules["pwd"] = _pwd_stub

if "cgi" not in sys.modules:
    import email.message, types
    _cgi_stub = types.ModuleType("cgi")
    def _parse_header(line):
        m = email.message.EmailMessage()
        m['content-type'] = line
        return m.get_content_type(), m.get_params() or {}
    _cgi_stub.parse_header = _parse_header
    sys.modules["cgi"] = _cgi_stub

import psutil

proc = psutil.Process(os.getpid())

def mb():
    gc.collect()
    return proc.memory_info().rss / (1024 * 1024)

def measure(label, fn):
    before = mb()
    try:
        result = fn()
        after = mb()
        delta = after - before
        print(f"  {label:45s}  {delta:+8.1f} MB  (total: {after:7.1f} MB)")
        return result
    except Exception as e:
        after = mb()
        delta = after - before
        print(f"  {label:45s}  {delta:+8.1f} MB  (total: {after:7.1f} MB)  [FAILED: {e}]")
        return None

print(f"\n{'='*75}")
print(f"  QUASAR MEMORY PROFILER")
print(f"  Python {sys.version}")
print(f"{'='*75}")
print(f"\n  Baseline (Python + psutil):                          {mb():7.1f} MB\n")

# ── Phase 1: Individual imports ──────────────────────────────────
print("── Phase 1: Library imports ──────────────────────────────")
measure("import pandas", lambda: __import__("pandas"))
measure("import numpy", lambda: __import__("numpy"))
measure("import openai", lambda: __import__("openai"))
measure("import anthropic", lambda: __import__("anthropic"))
measure("import astropy", lambda: __import__("astropy"))
measure("import astroquery", lambda: __import__("astroquery"))
measure("import matplotlib", lambda: __import__("matplotlib"))
measure("import plotly", lambda: __import__("plotly"))
measure("import scipy", lambda: __import__("scipy"))
measure("import langchain", lambda: __import__("langchain"))
measure("import langchain_community", lambda: __import__("langchain_community"))
measure("import langchain_openai", lambda: __import__("langchain_openai"))
measure("import tiktoken", lambda: __import__("tiktoken"))
measure("import qdrant_client", lambda: __import__("qdrant_client"))
measure("import fastapi", lambda: __import__("fastapi"))

try:
    measure("import mem0", lambda: __import__("mem0"))
except:
    print("  import mem0                                    SKIPPED (not installed)")

measure("import google.genai", lambda: __import__("google.genai"))

print(f"\n  After all imports:                                    {mb():7.1f} MB\n")

# ── Phase 2: Service instantiation ───────────────────────────────
print("── Phase 2: Service instantiation ────────────────────────")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

measure("RAGService()", lambda: __import__("services.rag_service", fromlist=["RAGService"]).RAGService())
measure("MemoryService()", lambda: __import__("services.memory_service", fromlist=["MemoryService"]).MemoryService())
measure("SearchService()", lambda: __import__("services.search", fromlist=["SearchService"]).SearchService())
measure("RadioAnalysisService()", lambda: __import__("services.analysis", fromlist=["RadioAnalysisService"]).RadioAnalysisService())
measure("BrowserService()", lambda: __import__("services.browser", fromlist=["BrowserService"]).BrowserService())
measure("PlottingService()", lambda: __import__("services.plotting", fromlist=["PlottingService"]).PlottingService())
measure("SplatalogueTool()", lambda: __import__("services.splatalogue", fromlist=["SplatalogueTool"]).SplatalogueTool())
measure("MultiArchiveMatcher()", lambda: __import__("services.multi_archive", fromlist=["MultiArchiveMatcher"]).MultiArchiveMatcher())
measure("CASAScriptGenerator()", lambda: __import__("services.casa_generator", fromlist=["CASAScriptGenerator"]).CASAScriptGenerator())
measure("GCNAlertMonitor()", lambda: __import__("services.gcn_monitor", fromlist=["GCNAlertMonitor"]).GCNAlertMonitor())
measure("PDFProcessingService()", lambda: __import__("services.pdf_processing", fromlist=["PDFProcessingService"]).PDFProcessingService(os.getenv("OPENAI_API_KEY","")))
measure("DataLinkClient()", lambda: __import__("integrations.datalink", fromlist=["DataLinkClient"]).DataLinkClient())
measure("FITSProcessingService()", lambda: __import__("services.fits_processing", fromlist=["FITSProcessingService"]).FITSProcessingService())
measure("AuthService()", lambda: __import__("services.auth", fromlist=["AuthService"]).AuthService())
measure("ConversationService()", lambda: __import__("services.conversation_service", fromlist=["ConversationService"]).ConversationService())
measure("ADSService()", lambda: __import__("integrations.ads_client", fromlist=["ADSService"]).ADSService())

print(f"\n  After all services:                                   {mb():7.1f} MB\n")

# ── Phase 3: Full agent ──────────────────────────────────────────
print("── Phase 3: Full QuasarAgent init ────────────────────────")
def init_agent():
    from core.agent import QuasarAgent, AgentConfig
    return QuasarAgent(AgentConfig())

measure("QuasarAgent(AgentConfig())", init_agent)

print(f"\n  FINAL TOTAL:                                          {mb():7.1f} MB")
print(f"{'='*75}\n")
