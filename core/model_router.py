# core/model_router.py
"""
Model Router — Route each subtask to the optimal LLM based on task type.

Uses a static routing table (tunable) that maps task categories to the
best-suited model.  The agent already has clients for OpenAI, Anthropic,
and Google Gemini from the multi-model routing work.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Route subtasks to the optimal LLM based on task type.

    The routing table maps task categories to models with rationale.
    Falls back to the default model if classification fails.
    """

    ROUTING_TABLE: Dict[str, Dict[str, str]] = {
        "archive_search":       {"model": "gpt-4o",          "reason": "Best tool-calling accuracy"},
        "literature_review":    {"model": "gemini-2.0-pro",  "reason": "2M context for long papers"},
        "scientific_reasoning": {"model": "claude-sonnet",   "reason": "Strong multi-step reasoning"},
        "code_generation":      {"model": "gpt-4o",          "reason": "Best code generation"},
        "data_analysis":        {"model": "gpt-4o",          "reason": "Reliable structured output"},
        "synthesis":            {"model": "gemini-2.0-pro",  "reason": "Long context for aggregation"},
        "simple_qa":            {"model": "gpt-4o-mini",     "reason": "Fast + cheap"},
        "web_search":           {"model": "gpt-4o-mini",     "reason": "Simple tool calls"},
        "visualization":        {"model": "gpt-4o",          "reason": "Code + tool calling"},
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

    def __init__(self, default_model: str = "gpt-4o"):
        self.default_model = default_model

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
                return entry["model"]

        # Fallback: classify from description
        category = self.classify_task_type(task_description)
        entry = self.ROUTING_TABLE.get(category)
        if entry:
            return entry["model"]

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
