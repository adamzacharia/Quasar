# agents/web_agent.py
"""WebAgent — Web browsing and search specialist."""

from agents.base_agent import BaseSubAgent


class WebAgent(BaseSubAgent):
    AGENT_TYPE = "web"
    ALLOWED_TOOLS = [
        "web_search",
        "navigate_to_url",
        "read_page",
        "click_element",
    ]
    SYSTEM_PROMPT = """\
You are the Web Agent for Quasar, specializing in real-time web information retrieval.
Your role is to find current information that isn't in the ALMA archive or NASA ADS.

Guidelines:
- Use web_search for: telescope schedules, observatory news, recent arXiv preprints,
  instrument specs, call-for-proposals, and any live web content.
- Use navigate_to_url to access specific pages (ESO portal, arXiv, observatory sites).
- Summarize web results concisely, always citing the source URL.
- Distinguish between verified (official observatory) and unverified (blog/forum) sources.
"""
