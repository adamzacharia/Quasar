"""
Quasar core package.

Keep package-level imports optional so lightweight submodules such as
``core.tools`` can be imported without pulling in the full agent runtime.
"""

from __future__ import annotations

__all__ = []


try:
    from .agent import QuasarAgent, AgentConfig

    __all__ += ["QuasarAgent", "AgentConfig"]
except Exception:
    QuasarAgent = None
    AgentConfig = None

try:
    from .memory import ConversationMemory

    __all__.append("ConversationMemory")
except Exception:
    ConversationMemory = None

try:
    from .tools import Tool, ToolRegistry

    __all__ += ["Tool", "ToolRegistry"]
except Exception:
    Tool = None
    ToolRegistry = None
