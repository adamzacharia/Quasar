# agents/__init__.py
"""
Specialist sub-agents for the Quasar multi-agent workforce.

Each agent has:
  - A domain-specific system prompt
  - Access to a subset of the 27+ tools
  - A run() method that executes tasks via its allowed tools
"""

from agents.base_agent import BaseSubAgent
from agents.archive_agent import ArchiveAgent
from agents.literature_agent import LiteratureAgent
from agents.analysis_agent import AnalysisAgent
from agents.viz_agent import VizAgent
from agents.web_agent import WebAgent
from agents.synthesis_agent import SynthesisAgent

AGENT_REGISTRY = {
    "archive": ArchiveAgent,
    "literature": LiteratureAgent,
    "analysis": AnalysisAgent,
    "viz": VizAgent,
    "web": WebAgent,
    "synthesis": SynthesisAgent,
}

__all__ = [
    "BaseSubAgent",
    "ArchiveAgent",
    "LiteratureAgent",
    "AnalysisAgent",
    "VizAgent",
    "WebAgent",
    "SynthesisAgent",
    "AGENT_REGISTRY",
]
