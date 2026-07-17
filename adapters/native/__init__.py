"""
adapters/native — register capabilities as in-process ``core.tools.Tool``s.

This is the PRODUCTION path (quasarassistant.com runs native, in-process). The
adapter is deliberately tiny: build a per-call :class:`CallContext`, validate the
model's arguments against the capability's pydantic ``InputModel``, run the
capability, and return ``ToolResult.to_native()`` — the exact dict shape the
agent's tool loop already consumes. No business logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable

from pydantic import ValidationError

from capabilities.base import PROVENANCE_SIDECAR_KEY, CallContext, ToolResult
from core.tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)

# A context provider builds the per-request CallContext (injected clients/store,
# tokens, emit, timestamp). The agent supplies one; capabilities never build it.
ContextProvider = Callable[[], CallContext]


def build_tool(cap: Any, ctx_provider: ContextProvider) -> Tool:
    """Wrap one capability as a native :class:`Tool`."""

    def _fn(**kwargs: Any) -> Dict[str, Any]:
        # Build per-call context (tokens/clients/store/now) via the provider.
        try:
            ctx = ctx_provider()
        except Exception as exc:  # pragma: no cover - provider misconfig
            logger.error("CallContext provider failed for %s: %s", cap.name, exc)
            return {"success": False, "error": f"internal: could not build call context ({exc})"}

        # Validate arguments against the capability's typed input model. A bad
        # tool call becomes a typed error the model can correct, not a crash.
        try:
            inp = cap.InputModel(**(kwargs or {}))
        except ValidationError as ve:
            return {
                "success": False,
                "error": f"Invalid arguments for {cap.name}: {ve.errors()}",
            }

        result = cap.run(inp, ctx)
        if not isinstance(result, ToolResult):  # defensive: capabilities must return ToolResult
            logger.error("%s returned %s, not ToolResult", cap.name, type(result))
            return {"success": False, "error": "internal: capability returned a non-ToolResult"}
        native = result.to_native()
        # Attach canonical provenance alongside (never inside) the frozen
        # to_native() payload, so the agent can record the EXACT request it made
        # without the model's view of the result changing. The agent pops this
        # key before serializing the result. A copy keeps `native=` passthrough
        # tools' own dicts unmutated.
        sidecar = result.provenance_sidecar()
        if sidecar is not None and isinstance(native, dict):
            native = {**native, PROVENANCE_SIDECAR_KEY: sidecar}
        return native

    return Tool(
        name=cap.name,
        description=cap.description,
        function=_fn,
        parameters=cap.parameters_schema(),
        category=cap.category,
    )


def register_capabilities(
    registry: ToolRegistry,
    capabilities: Iterable[Any],
    ctx_provider: ContextProvider,
    *,
    namespace: str | None = None,
    replace: bool = True,
) -> list[str]:
    """Register a family of capabilities into the live registry. Returns the
    registered tool names."""
    registered: list[str] = []
    for cap in capabilities:
        tool = build_tool(cap, ctx_provider)
        registry.register(tool, namespace=namespace, replace=replace)
        registered.append(tool.name)
    return registered
