"""
adapters/mcp — MCP-transport shims.

Consumer side (this module): normalize results from EXTERNAL MCP servers
(e.g. MANNA) into the native tool-result shape so the rest of Quasar — the
tool trace, provenance harvester, data-card serializer, Conductor soft-failure
detection — sees one shape regardless of transport.

Producer side (``adapters/mcp/server.py``, P2) will expose Quasar's own
capabilities over FastMCP; it does not exist yet.
"""

from .normalize import normalize_mcp_failure, normalize_mcp_result

__all__ = ["normalize_mcp_failure", "normalize_mcp_result"]
