"""Normalize external MCP tool results to the native tool-result shape.

MCP servers return content blocks (text, image, resource, ...) plus an
optional ``structuredContent`` dict and an ``isError`` flag. Native tools
return the ``ToolResult.to_native()`` dict shape (``success``/``data``/
``error``). This module maps the former onto the latter and attaches the
provenance sidecar (``capabilities.base.PROVENANCE_SIDECAR_KEY``) — on both
success AND failure — so bridged calls surface in the tool trace /
query-provenance UI like any native archive call.

No I/O, no MCP SDK imports — operates on duck-typed result objects so it can
be unit-tested without the SDK installed.
"""

import json
from typing import Any, Dict, List, Optional

from capabilities.base import PROVENANCE_SIDECAR_KEY

# Keys an MCP tool's arguments may use for the executed query / endpoint.
# MANNA's TAP tools take (endpoint, adql); SCS/SIA take (endpoint, ra, dec...).
_QUERY_ARG_KEYS = ("adql", "query", "sql")
_ENDPOINT_ARG_KEYS = ("endpoint", "url", "job_url", "service_url")


def _text_blocks(result: Any) -> List[str]:
    out = []
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", "") == "text":
            text = getattr(block, "text", None)
            if text:
                out.append(text)
    return out


def _non_text_block_types(result: Any) -> List[str]:
    return sorted({
        getattr(block, "type", "?") or "?"
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "type", "") != "text"
    })


def _provenance_sidecar(
    server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    prov: Dict[str, Any] = {"service": server_name, "tool": tool_name}
    args = arguments or {}
    for key in _QUERY_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            prov["query"] = value
            break
    for key in _ENDPOINT_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            prov["endpoint"] = value
            break
    return {"provenance": prov}


def normalize_mcp_failure(
    server_name: str,
    tool_name: str,
    error: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Native-shaped failure for bridge/infrastructure errors (timeout, dead
    loop, transport exception). Carries provenance so a failed archive call is
    still auditable in the tool trace."""
    return {
        "success": False,
        "error": error,
        PROVENANCE_SIDECAR_KEY: _provenance_sidecar(server_name, tool_name, arguments),
    }


def normalize_mcp_result(
    server_name: str,
    tool_name: str,
    result: Any,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map one MCP CallToolResult onto the native tool-result dict shape.

    Precedence: ``isError`` → typed failure; ``structuredContent`` (any dict,
    including empty) → data; single JSON text block → parsed data; one or more
    text blocks → newline-joined text; non-text-only content → a typed note
    (image/resource payloads are not consumed yet). A dict payload that
    already carries a boolean ``success`` — from structuredContent or a JSON
    text block — passes through unwrapped, so a server speaking the native
    contract (including its failures) is preserved. Never raises.
    """
    try:
        texts = _text_blocks(result)

        if getattr(result, "isError", False):
            return normalize_mcp_failure(
                server_name,
                tool_name,
                "\n".join(texts) or f"MCP tool {tool_name} failed",
                arguments,
            )

        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            payload: Any = structured
        elif texts:
            if len(texts) == 1:
                try:
                    payload = json.loads(texts[0])
                except ValueError:
                    payload = texts[0]
            else:
                payload = "\n".join(texts)
        else:
            non_text = _non_text_block_types(result)
            if non_text:
                payload = {
                    "note": (
                        "MCP result contained only non-text content "
                        f"({', '.join(non_text)}), which this bridge does not "
                        "consume yet."
                    ),
                    "content_types": non_text,
                }
            else:
                payload = None

        if isinstance(payload, dict) and isinstance(payload.get("success"), bool):
            out = dict(payload)
        else:
            out = {"success": True, "data": payload}

        out[PROVENANCE_SIDECAR_KEY] = _provenance_sidecar(
            server_name, tool_name, arguments
        )
        return out
    except Exception as e:  # a bridge result must never raise into the loop
        out = {"success": False, "error": f"MCP result normalization failed: {e}"}
        try:
            out[PROVENANCE_SIDECAR_KEY] = _provenance_sidecar(
                server_name, tool_name, arguments
            )
        except Exception:
            pass  # provenance is best-effort on this last-resort path
        return out
