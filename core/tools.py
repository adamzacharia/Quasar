# core/tools.py
"""
Tool registry and management for Quasar
"""

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
import inspect
import json

@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] # Changed from Dict[str, str] to support full JSON schema
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
    """Registry for managing available tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}

    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool

        if tool.category not in self.categories:
            self.categories[tool.category] = []
        self.categories[tool.category].append(tool.name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        """List all tools or tools in category"""
        if category:
            tool_names = self.categories.get(category, [])
            return [self.tools[name] for name in tool_names]
        return list(self.tools.values())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI format"""
        return [tool.to_openai_schema() for tool in self.tools.values()]

