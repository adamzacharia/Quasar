"""core/provenance.py — build the uniform ``request`` object for a tool call.

Feature 1 (raw-query provenance surface): every executed tool contributes a
``request`` describing the EXACT call it made, so a user can see and copy it.

The single output contract is::

    {"kind": "adql"|"http"|"ads"|"params"|"args",
     "text": "<the literal, copyable request>",   # ALWAYS present
     ...kind-specific extras}

``text`` is the copy-button payload for every kind — the UI never has to know
how to render a particular archive's request shape.

Source of truth is ``ToolResult.provenance`` (``query``/``endpoint``) plus
``reproducible_snippet``, delivered here by the native adapter's sidecar (see
``capabilities.base.PROVENANCE_SIDECAR_KEY``). Legacy in-process tools that
never became capabilities have no ``ToolResult``; for those we fall back to the
flattened dict fields and finally to the raw tool arguments, so the surface is
uniform across the whole registry.

Everything emitted here is secret-redacted: this object reaches the browser.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from services.secret_redaction import redact_secrets, redact_url

# Cap the copyable text. The tool_trace SSE event has a 128 KB budget with a
# degrade ladder (sse.py) — a runaway ADQL string must not eat it.
REQUEST_TEXT_CAP = 2000

_SQL_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def _looks_like_sql(text: str) -> bool:
    return bool(_SQL_START.match(text or ""))


def _clean(value: Any) -> str:
    """Normalize a provenance string field to a stripped str ("" when absent)."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def _cap(text: str) -> str:
    text = text or ""
    return text[:REQUEST_TEXT_CAP] + "…" if len(text) > REQUEST_TEXT_CAP else text


def _dumps(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str, sort_keys=True)
    except Exception:
        return str(value)


def extract_provenance(result_obj: Any = None, sidecar: Any = None) -> Dict[str, Any]:
    """Best-effort provenance dict for one tool call.

    Prefers the adapter sidecar (the canonical ``ToolResult.provenance``, which
    ``to_native()`` cannot carry); falls back to a ``provenance`` key flattened
    into a legacy tool's result dict.
    """
    if isinstance(sidecar, dict):
        prov = sidecar.get("provenance")
        if isinstance(prov, dict) and prov:
            return prov
    if isinstance(result_obj, dict):
        prov = result_obj.get("provenance")
        if isinstance(prov, dict):
            return prov
    return {}


def build_tool_request(
    tool_name: str,
    args: Any,
    result_obj: Any = None,
    sidecar: Any = None,
) -> Dict[str, Any]:
    """Build the uniform, redacted ``request`` for one executed tool call.

    Never raises — provenance is an audit surface, not a control path; a tool
    call must never fail because its request could not be described.
    """
    try:
        args_dict: Dict[str, Any] = args if isinstance(args, dict) else {}
        prov = extract_provenance(result_obj, sidecar)
        snippet = None
        if isinstance(sidecar, dict):
            snippet = _clean(sidecar.get("reproducible_snippet")) or None

        service = _clean(prov.get("service"))
        endpoint = _clean(prov.get("endpoint"))
        query = _clean(prov.get("query"))

        # Legacy fallbacks: pre-capability tools flatten the executed SQL onto
        # the result dict; the model's own `sql` argument is the last resort.
        if not query and isinstance(result_obj, dict):
            query = _clean(result_obj.get("validated_sql")) or _clean(
                result_obj.get("query_summary")
            )
        if not query and isinstance(args_dict.get("sql"), str):
            query = _clean(args_dict["sql"])

        request: Dict[str, Any]
        if query and _looks_like_sql(query):
            request = {"kind": "adql", "text": redact_secrets(query)}
            if endpoint:
                request["endpoint"] = redact_url(endpoint)
        elif service == "ads":
            q = query or _clean(args_dict.get("query")) or _clean(args_dict.get("q"))
            request = {"kind": "ads", "text": redact_secrets(q), "q": redact_secrets(q)}
            for key in ("fq", "rows", "sort"):
                if args_dict.get(key) is not None:
                    request[key] = args_dict[key]
            # ADS `rows` is the executed result count; the papers tool spells it
            # `max_results` (CX-37) — surface it so the request is exact.
            if "rows" not in request and args_dict.get("max_results") is not None:
                request["rows"] = args_dict["max_results"]
            if endpoint:
                request["endpoint"] = redact_url(endpoint)
        elif endpoint:
            url = redact_url(endpoint)
            request = {
                "kind": "http",
                "method": _clean(prov.get("method")) or "GET",
                "url": url,
                "text": url,
            }
        elif query:
            # Provenance-declared but neither SQL nor a URL: a parameterized
            # service call (Splatalogue / Spectral Line Explorer).
            request = {
                "kind": "params",
                "text": redact_secrets(query),
                "params": _cap_struct(_redact_mapping(args_dict)),
            }
        elif service:
            params = _cap_struct(_redact_mapping(args_dict))
            request = {"kind": "params", "text": _dumps(params), "params": params}
        else:
            # No provenance at all (legacy tool): the arguments ARE the request.
            safe_args = _cap_struct(_redact_mapping(args_dict))
            request = {"kind": "args", "text": _dumps(safe_args), "args": safe_args}

        if service:
            # `service` is normally a bare label ("datalab"), but a capability
            # can emit a URL-shaped value ("TAP: https://…?TOKEN=") — redact the
            # query string too, not just key-shaped tokens (CX-02).
            request["service"] = redact_url(service) if "://" in service else redact_secrets(service)
        if snippet:
            request["snippet"] = _cap(redact_secrets(snippet))
        request["text"] = _cap(request.get("text") or "")
        return request
    except Exception:  # pragma: no cover - provenance must never break a call
        return {"kind": "args", "text": "", "args": {}}


# A URL hides in a plain string arg (e.g. `access_url`, `endpoint`). These reach
# the browser AND get persisted, so a token in the query string must never
# survive. `redact_url` is a no-op on non-URL text, so it is safe to run widely.
_URLISH = re.compile(r"^\s*https?://", re.IGNORECASE)


def _redact_value(val: Any, depth: int = 0) -> Any:
    """Recursively redact secret-shaped values in an arg tree (CX-02).

    Handles nested dicts/lists (a token buried under `params.auth` must not
    leak), and URL-valued strings (a `?TOKEN=` in the value, not the key).
    Depth-bounded so a pathological structure can't loop.
    """
    if depth > 6:
        return "[…]"
    if isinstance(val, dict):
        return {
            k: ("[REDACTED]" if _is_secret_key(str(k)) else _redact_value(v, depth + 1))
            for k, v in val.items()
        }
    if isinstance(val, (list, tuple)):
        return [_redact_value(v, depth + 1) for v in val][:200]
    if isinstance(val, str):
        return redact_url(val) if _URLISH.match(val) else redact_secrets(val)
    return val


def _redact_mapping(value: Dict[str, Any]) -> Dict[str, Any]:
    """Redact secret-shaped values inside an argument mapping (deep)."""
    out: Dict[str, Any] = {}
    for key, val in (value or {}).items():
        if _is_secret_key(str(key)):
            out[key] = "[REDACTED]"
        else:
            out[key] = _redact_value(val)
    return out


_SECRET_KEYS = re.compile(
    r"(?i)^(token|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|auth|authorization|"
    r"password|passwd|secret|client[_-]?secret|session|session[_-]?id|sessionid|sid|"
    r"signature|sig|bearer)$"
)


def _is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEYS.match(key or ""))


# The uniform request rides inside the 128 KB tool_trace SSE event (and is
# persisted). `text` is capped elsewhere; a huge `args`/`params` dict would
# still blow the budget and get whole calls dropped (CX-17), so bound the
# structured copy to a JSON-size ceiling, dropping keys deterministically.
_STRUCT_CAP = 4000


def _cap_struct(mapping: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    if len(_dumps(mapping)) <= _STRUCT_CAP:
        return mapping
    out: Dict[str, Any] = {}
    for key in sorted(mapping.keys(), key=lambda k: len(_dumps({k: mapping[k]}))):
        trial = {**out, key: mapping[key]}
        if len(_dumps(trial)) > _STRUCT_CAP:
            out["_truncated"] = True
            break
        out = trial
    return out


def persistable_trace(trace: Any) -> list:
    """A redacted, reload-safe projection of the tool trace for persistence.

    The live `tool_trace` SSE event (which the benchmark scores) keeps raw
    `arguments`/`output`; those must NOT be written to `messages.metadata`
    verbatim — a raw `access_url`/`endpoint` argument can carry a `?TOKEN=` and
    would then live permanently in the DB (CX-01). The reload UI only reads the
    already-redacted `request`, so persist just what it needs, redacting the
    small residue we keep for debuggability.
    """
    out = []
    for call in trace or []:
        if not isinstance(call, dict):
            continue
        slim: Dict[str, Any] = {
            "name": call.get("name"),
            "ok": call.get("ok", True),
        }
        request = call.get("request")
        if isinstance(request, dict) and request:
            slim["request"] = request  # built redacted by build_tool_request
        if isinstance(call.get("rowcount"), (int, float)):
            slim["rowcount"] = int(call["rowcount"])
        sql = call.get("sql")
        if isinstance(sql, str) and sql:
            slim["sql"] = _cap(redact_secrets(sql))
        out.append(slim)
    return out


def request_for_display(request: Optional[Dict[str, Any]]) -> str:
    """The literal text a user copies. Mirrors the frontend's copy payload."""
    if not isinstance(request, dict):
        return ""
    return str(request.get("text") or "")
