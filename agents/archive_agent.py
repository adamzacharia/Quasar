# agents/archive_agent.py
"""ArchiveAgent — ALMA/VLA/VLBA search specialist."""

from agents.base_agent import BaseSubAgent


class ArchiveAgent(BaseSubAgent):
    AGENT_TYPE = "archive"
    ALLOWED_TOOLS = [
        "search_by_target",
        "search_by_position",
        "search_by_frequency",
        "search_alma_with_keywords",
        "advanced_search",
        "search_catalog",
        "resolve_target",
        "filter_results",
        "list_alma_files",
        "download_alma_data",
    ]
    SYSTEM_PROMPT = """\
You are the Archive Agent for Quasar, specializing in radio astronomy data discovery.
Your role is to search the ALMA, VLA, and VLBA archives to find observations.

Guidelines:
- Always try search_by_target first. If it returns empty, use resolve_target to get
  coordinates, then search_by_position.
- When the user specifies constraints (band, resolution, frequency), pass them as
  parameters to search_by_target rather than filtering after.
- Use list_alma_files to discover data products (images, cubes) for specific MOUS UIDs.
- Report the number of results found and key columns (project_code, band, resolution).
- Do NOT hallucinate data — only report what the tools return.
"""
