# core/model_router.py
"""
Model Router — Route each Conductor subtask to the optimal OpenAI model.

Uses a 3-tier OpenAI cost ladder:
  Tier 1 (heavy):  GPT-5.4      -- synthesis, scientific reasoning (expensive, smart)
  Tier 2 (solid):  GPT-4.1      -- archive search, analysis, viz (solid tool-calling)
  Tier 3 (cheap):  GPT-4.1-mini -- simple QA, web search, literature (cheap + fast)

The routing table maps agent_type → model.  Falls back to text classification
when agent_type is unknown.  HealthMonitor integration allows automatic
fallback if a model starts failing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from core.llm_client import detect_provider

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Route subtasks to the optimal LLM based on task type.

    The routing table maps task categories to models with rationale.
    Falls back to the default model if classification fails.
    """

    # -- OpenAI-only cost ladder --------------------------------------------------
    # Tier 1 (heavy): GPT-5.4      -- synthesis, scientific reasoning
    # Tier 2 (solid): GPT-4.1      -- archive search, analysis, viz, code
    # Tier 3 (cheap): GPT-4.1-mini -- simple QA, web search, literature
    #
    # Rationale: GPT-5.4 is ~10-20× more expensive than 4.1-mini.
    # Reserve it for tasks that genuinely need intelligence (combining
    # results, multi-step reasoning).  Archive search is tool-calling —
    # GPT-4.1 handles that perfectly.  Literature search is just calling
    # search_papers with a query — 4.1-mini is fine.
    ROUTING_TABLE: Dict[str, Dict[str, str]] = {
        "archive_search":       {"model": "gpt-4.1",       "reason": "Solid tool-calling for archive queries"},
        "literature_review":    {"model": "gpt-4.1-mini",  "reason": "Simple ADS search — cheap + fast"},
        "scientific_reasoning": {"model": "gpt-5.4",       "reason": "Complex multi-step reasoning needs intelligence"},
        "code_generation":      {"model": "gpt-4.1",       "reason": "Reliable code generation + tool calling"},
        "data_analysis":        {"model": "gpt-4.1",       "reason": "Structured output for spectral analysis"},
        "synthesis":            {"model": "gpt-5.4",       "reason": "Combining results requires deep reasoning"},
        "simple_qa":            {"model": "gpt-4.1-mini",  "reason": "Fast + cheap for trivial questions"},
        "web_search":           {"model": "gpt-4.1-mini",  "reason": "Simple tool calls"},
        "visualization":        {"model": "gpt-4.1",       "reason": "Tool calling for FITS rendering"},
    }

    # Keywords for heuristic classification
    _CATEGORY_PATTERNS = {
        "archive_search": [
            r"search", r"find", r"query", r"alma", r"archive",
            r"observation", r"download", r"mous", r"target",
        ],
        "literature_review": [
            r"paper", r"publication", r"cite", r"author",
            r"bibtex", r"ads", r"literature", r"review",
        ],
        "scientific_reasoning": [
            r"explain", r"why", r"how does", r"physics",
            r"compare.*sensitivity", r"trade-off", r"interpret",
        ],
        "code_generation": [
            r"script", r"code", r"casa", r"python",
            r"calibrat", r"imaging", r"reduce",
        ],
        "data_analysis": [
            r"spectral.*line", r"frequency", r"beam",
            r"fits.*header", r"rms", r"noise", r"cross-match",
        ],
        "synthesis": [
            r"table", r"summary", r"combin", r"aggregat",
            r"comparison", r"synthesiz", r"final",
        ],
        "visualization": [
            r"plot", r"map", r"chart", r"notebook",
            r"visualiz", r"contour", r"spectrum",
        ],
        "web_search": [
            r"web", r"search.*online", r"latest.*news",
            r"schedule", r"announcement",
        ],
    }

    def __init__(self, default_model: str = "gpt-4.1"):
        self.default_model = default_model
        self.health_monitor = None  # Injected by agent.py

    def route(self, task_description: str, agent_type: str = "general") -> str:
        """
        Return the model ID best suited for this task.

        First tries agent_type mapping, then falls back to text classification.
        """
        # Direct mapping from agent_type
        type_to_category = {
            "archive": "archive_search",
            "literature": "literature_review",
            "analysis": "data_analysis",
            "viz": "visualization",
            "web": "web_search",
            "synthesis": "synthesis",
        }

        if agent_type in type_to_category:
            category = type_to_category[agent_type]
            entry = self.ROUTING_TABLE.get(category)
            if entry:
                logger.debug(
                    "Routed '%s' → %s (%s)",
                    agent_type, entry["model"], entry["reason"],
                )
                return self._apply_health_check(entry["model"])

        # Fallback: classify from description
        category = self.classify_task_type(task_description)
        entry = self.ROUTING_TABLE.get(category)
        if entry:
            return self._apply_health_check(entry["model"])

        return self.default_model

    def classify_task_type(self, description: str) -> str:
        """Classify a task description into one of the routing categories."""
        desc_lower = description.lower()
        scores: Dict[str, int] = {}

        for category, patterns in self._CATEGORY_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, desc_lower))
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)
        return "simple_qa"

    def get_routing_info(self, task_description: str, agent_type: str = "general") -> Dict[str, str]:
        """Get full routing info including model and reasoning."""
        model = self.route(task_description, agent_type)
        category = self.classify_task_type(task_description)
        entry = self.ROUTING_TABLE.get(category, {})
        return {
            "model": model,
            "category": category,
            "reason": entry.get("reason", "Default routing"),
        }

    def _apply_health_check(self, model: str) -> str:
        """
        Check if the model's provider is healthy. If not, return a fallback.
        The static ROUTING_TABLE is unchanged — health is a filter on top.
        """
        if self.health_monitor is None:
            return model

        provider = detect_provider(model)
        if self.health_monitor.is_healthy(provider):
            return model

        fallback = self.health_monitor.get_fallback_model(model, provider)
        if fallback:
            logger.warning(
                "Health check: '%s' (%s) unhealthy — routing to fallback '%s'",
                model, provider, fallback,
            )
            return fallback

        return model  # No fallback available, try anyway
