"""
adapters/ — thin transport shims over the ``capabilities`` layer.

``adapters/native`` registers capabilities as in-process ``core.tools.Tool``s;
``adapters/mcp`` (P2) exposes the SAME capabilities over FastMCP. Adapters
contain NO business logic — they validate input, call ``cap.run(...)``, and
serialize the ``ToolResult``. If you find archive/query logic here, it belongs
in ``capabilities/``.
"""
