"""ADS/OpenAlex literature profile (non-tabular: tables=None).

The literature surface is deliberately NATURAL LANGUAGE — an internal query
builder translates to ADS syntax, and the tool description forbids the model
from constructing field syntax itself. The profile grounds routing (which
tool for which ask), not query syntax.
"""

from __future__ import annotations

from services.archive_profiles.schema import ArchiveProfile

PROFILE = ArchiveProfile.model_validate(
    {
        "archive": "ads_openalex",
        "aliases": ("ads", "openalex", "papers", "literature"),
        "description": (
            "Literature and researcher services: NASA ADS paper search (natural-language in, "
            "AI-built ADS syntax underneath) plus OpenAlex-backed researcher profiles and "
            "publication/funding trends."
        ),
        "endpoints": [
            {
                "id": "ads_api",
                "description": "NASA ADS API (reached through search_papers*, never directly).",
                "url": "https://api.adsabs.harvard.edu/v1/",
                "protocol": "rest",
            },
            {
                "id": "openalex_api",
                "description": "OpenAlex API backing researcher/trend lookups.",
                "url": "https://api.openalex.org/",
                "protocol": "rest",
            },
        ],
        "query_surfaces": [
            {
                "id": "paper_search",
                "purpose": "Any papers/publications/literature request — pass the ask as natural language.",
                "tool": "search_papers",
                "request_kind": "natural_language",
                "query_argument": "query",
                "parameters": [
                    {"name": "sort", "json_type": "string",
                     "allowed_values": ["date desc", "citation_count desc", "score desc"],
                     "description": "'date desc' newest (default), 'citation_count desc' most cited, 'score desc' relevance."},
                    {"name": "max_results", "json_type": "integer", "unit": "1",
                     "description": "Result count (default 15, max 50)."},
                ],
                "endpoint_ids": ["ads_api"],
            },
            {
                "id": "papers_by_identifier",
                "purpose": "Papers connected to a specific archive identifier (ALMA project code, MOUS/ASDM UID, dataset id).",
                "tool": "search_papers_by_observation_id",
                "request_kind": "structured_args",
                "endpoint_ids": ["ads_api"],
            },
            {
                "id": "researcher_profile",
                "purpose": "Who is X / researcher metrics, affiliations, topics (by name or ORCID).",
                "tool": "lookup_researcher",
                "request_kind": "natural_language",
                "query_argument": "query",
                "endpoint_ids": ["openalex_api"],
            },
            {
                "id": "research_trends",
                "purpose": "Papers-per-year and funding landscape for a topic.",
                "tool": "get_research_trends",
                "request_kind": "natural_language",
                "query_argument": "query",
                "endpoint_ids": ["openalex_api"],
            },
        ],
        "tables": None,
        "pitfalls": [
            {
                "id": "natural_language_only",
                "summary": "pass the user's ask to search_papers as NATURAL LANGUAGE — never construct ADS field syntax yourself",
                "applies_to": [{"kind": "surface", "ref": "paper_search"}],
                "prompt_rank": 1,
            },
            {
                "id": "identifiers_route",
                "summary": "archive identifiers (project codes like 2019.1.00123.S, uid://... MOUS/ASDM ids) go to search_papers_by_observation_id, not free-text search",
                "applies_to": [{"kind": "surface", "ref": "papers_by_identifier"}],
                "prompt_rank": 2,
            },
            {
                "id": "no_web_search",
                "summary": "never use web_search for paper/literature requests — search_papers is mandatory for them",
                "applies_to": [{"kind": "archive", "ref": "ads_openalex"}],
            },
            {
                "id": "knowledge_vs_papers",
                "summary": "how-does-X-work / policy / deadline questions are KNOWLEDGE queries (answer from documentation), not paper searches",
                "applies_to": [{"kind": "surface", "ref": "paper_search"}],
            },
        ],
        "golden_examples": [
            {
                "id": "topic_search",
                "intent": "Recent papers on protoplanetary disk gaps observed with ALMA.",
                "invocation": {
                    "tool": "search_papers",
                    "arguments": {"query": "recent ALMA papers on protoplanetary disk gaps", "sort": "date desc"},
                },
                "note": "Natural language in — the internal builder produces the ADS syntax.",
            },
            {
                "id": "papers_for_project_code",
                "intent": "Papers connected to ALMA project 2019.1.00123.S.",
                "invocation": {
                    "tool": "search_papers_by_observation_id",
                    "arguments": {"identifier": "2019.1.00123.S"},
                },
            },
            {
                "id": "researcher",
                "intent": "Who is Crystal Brogan?",
                "invocation": {"tool": "lookup_researcher", "arguments": {"query": "Crystal Brogan"}},
            },
        ],
        "citations": [
            {"id": "cite_ads", "text": "NASA Astrophysics Data System", "url": "https://ui.adsabs.harvard.edu/"},
            {"id": "cite_openalex", "text": "OpenAlex", "url": "https://openalex.org/"},
        ],
    }
)
