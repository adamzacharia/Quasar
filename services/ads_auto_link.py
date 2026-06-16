"""Deterministic ADS paper linking for ALMA archive search results."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional


PROJECT_CODE_RE = re.compile(r"\b\d{4}\.\d\.\d{5}\.[A-Z]\b")


def build_exact_project_paper_links(
    ads_client: Any,
    tool_name: str,
    tool_result: Any,
    max_project_codes: int = 3,
    max_results: int = 20,
) -> Optional[Dict[str, Any]]:
    """Return a paper result payload using exact ALMA project-code ADS searches."""
    enabled = str(os.getenv("QUASAR_AUTO_LINK_PROJECT_PAPERS", "true")).strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    if tool_name not in {"search_by_target", "search_by_position"}:
        return None
    if not ads_client or not getattr(ads_client, "api_key", ""):
        return None
    if not isinstance(tool_result, dict) or not tool_result.get("success"):
        return None

    project_codes = []
    for value in tool_result.get("top_project_codes") or []:
        code = str(value or "").strip().upper()
        if PROJECT_CODE_RE.search(code) and code not in project_codes:
            project_codes.append(code)
        if len(project_codes) >= max_project_codes:
            break
    if not project_codes:
        return None

    merged_papers: Dict[str, Dict[str, Any]] = {}
    ads_queries = []
    errors: Dict[str, str] = {}
    for code in project_codes:
        try:
            result = ads_client.search_by_observation_identifier(
                identifier=code,
                max_results=max_results,
                facility="ALMA",
            )
        except Exception as exc:
            errors[code] = str(exc) or "ADS exact project-code search failed"
            continue

        query = str(result.get("query") or "")
        if query:
            ads_queries.append(query)
        for paper in result.get("papers", []):
            key = paper.get("bibcode") or paper.get("doi") or paper.get("title") or str(id(paper))
            if key not in merged_papers:
                merged_papers[key] = paper
                continue
            current_links = merged_papers[key].setdefault("observation_links", [])
            for link in paper.get("observation_links", []):
                if link not in current_links:
                    current_links.append(link)

    papers = list(merged_papers.values())[:max_results]
    warnings = []
    if errors:
        warnings.append(
            "Exact ADS project-code paper lookup failed for "
            + ", ".join(sorted(errors.keys()))
            + "."
        )
    if not papers:
        warnings.append(
            "No NASA ADS papers were found with exact ALMA project-code links for "
            + ", ".join(project_codes)
            + "."
        )

    return {
        "type": "papers",
        "tool_name": "search_papers_by_observation_id",
        "papers": papers,
        "source": "ADS exact project-code links",
        "warnings": warnings,
        "paper_provenance": {
            "identifiers": project_codes,
            "identifier_type": "project_code",
            "ads_query": " OR ".join(ads_queries),
            "facility": "ALMA",
            "auto_linked": True,
        },
    }
