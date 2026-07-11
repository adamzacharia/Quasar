# core/tools.py
"""
Tool registry and management for Quasar.

The registry is transport-agnostic: it holds ``Tool``s whether they are backed
by a legacy in-process implementation or by a V2 capability (via a thin native
adapter — see ``adapters/native``). The orchestration brain reads the LIVE
registry, so both paths are visible to the planner.
"""

import logging
import re
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Canonical tool-name shape. Names outside this are logged (not rejected, to keep
# the ~145 legacy registrations working) so drift is visible.
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")


@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]  # full JSON schema
    category: str = "general"

    def execute(self, **kwargs) -> Any:
        """Execute the tool function"""
        return self.function(**kwargs)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI tool schema format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """Registry for managing available tools.

    Backward-compatible with the legacy ``register(Tool(...))`` call sites, with
    added namespacing, name validation and collision protection.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}

    def register(
        self,
        tool: Tool,
        *,
        namespace: Optional[str] = None,
        replace: bool = True,
    ) -> Tool:
        """Register a tool.

        namespace : optional ``{namespace}__{name}`` prefix (used when mounting an
                    external MCP server's tools so they can't collide with native
                    ones). The tool's ``name`` is rewritten in place.
        replace   : when False, registering an existing name raises instead of
                    overwriting. Default True preserves legacy behavior (later
                    registrations win — e.g. a capability tool replacing an inline
                    one behind a feature flag).
        """
        if namespace:
            tool.name = f"{namespace}__{tool.name}"

        if not _TOOL_NAME_RE.match(tool.name or ""):
            logger.warning("Registering tool with non-canonical name: %r", tool.name)

        existing = self.tools.get(tool.name)
        if existing is not None:
            if not replace:
                raise ValueError(f"Tool '{tool.name}' is already registered")
            if existing.function is not tool.function:
                logger.info("Tool '%s' re-registered (implementation replaced)", tool.name)

        self.tools[tool.name] = tool

        names = self.categories.setdefault(tool.category, [])
        if tool.name not in names:  # avoid duplicate category entries on re-register
            names.append(tool.name)
        return tool

    def register_all(self, tools: List[Tool], **kw: Any) -> None:
        """Register many tools (e.g. a whole capability family via an adapter)."""
        for t in tools:
            self.register(t, **kw)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        tool = self.tools.pop(name, None)
        if tool is None:
            return False
        names = self.categories.get(tool.category)
        if names and name in names:
            names.remove(name)
        return True

    def has(self, name: str) -> bool:
        return name in self.tools

    def names(self, category: Optional[str] = None) -> List[str]:
        if category:
            return list(self.categories.get(category, []))
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        """List all tools or tools in category"""
        if category:
            tool_names = self.categories.get(category, [])
            return [self.tools[name] for name in tool_names if name in self.tools]
        return list(self.tools.values())

    def get_openai_tools(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get tools in OpenAI format. Optionally restrict to ``names`` (for
        intent-based tool subsetting — stops sending all ~142 schemas each round)."""
        tools = (
            [self.tools[n] for n in names if n in self.tools]
            if names is not None
            else list(self.tools.values())
        )
        return [tool.to_openai_schema() for tool in tools]
