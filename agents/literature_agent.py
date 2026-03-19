# agents/literature_agent.py
"""LiteratureAgent — NASA ADS paper search specialist."""

from agents.base_agent import BaseSubAgent


class LiteratureAgent(BaseSubAgent):
    AGENT_TYPE = "literature"
    ALLOWED_TOOLS = [
        "search_papers",
        "get_author_papers",
        "get_paper_metrics",
        "get_author_metrics",
        "get_paper_abstract",
        "export_bibtex",
        "reproduce_paper_methods",
        "list_ads_libraries",
        "get_ads_library_papers",
        "create_ads_library",
        "add_to_ads_library",
    ]
    SYSTEM_PROMPT = """\
You are the Literature Agent for Quasar, specializing in astronomical literature search.
Your role is to find, analyze, and organize papers from NASA ADS.

Guidelines:
- Use proper ADS query syntax: abstract:"ALMA" AND abstract:"topic", author:"Last, First", year:2020-2024.
- When asked about author impact, use get_author_metrics for h-index and citation stats.
- For paper details, use get_paper_abstract to get full metadata.
- When asked to cite papers, use export_bibtex to generate proper BibTeX.
- Report paper titles, authors, years, and citation counts in a structured format.
"""
