"""
Quasar V2 shared core — the ``capabilities`` layer.

A *capability* is the science operation written **once** as a transport-pure
function: it takes a typed input model + a :class:`CallContext` and returns a
:class:`ToolResult`. It knows nothing about MCP, the agent, SSE, or the
streaming loop. Two thin adapters expose it:

  * ``adapters/native`` — registers it as a ``core.tools.Tool`` (in-process).
  * ``adapters/mcp``    — registers it as a FastMCP ``@mcp.tool()`` (out-of-process).

Both adapters are ~10 lines: validate input → ``cap.run(...)`` → serialize.
There is exactly ONE implementation of each capability; duplication between the
native and MCP paths is a bug, not a stage. See ``docs/v2/SHARED_CORE_ARCHITECTURE.md``.
"""

from capabilities.base import (
    Budget,
    Capability,
    BaseCapability,
    CallContext,
    Provenance,
    ResultStore,
    ServiceClient,
    ToolResult,
)

__all__ = [
    "Budget",
    "Capability",
    "BaseCapability",
    "CallContext",
    "Provenance",
    "ResultStore",
    "ServiceClient",
    "ToolResult",
]
