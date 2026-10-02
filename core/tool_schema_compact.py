# core/tool_schema_compact.py
"""
Compact tool parameter schemas at serialization time (token plan rank 3).

CALLED BY: core/agent.py _build_tools_for_responses_api
CALLS:     nothing (pure, never mutates the registry's schemas)

Lossless or near-lossless rewrites only; the registry keeps the original
schema (tests, the benchmark allowlist arm and QUASAR_COMPACT_TOOL_SCHEMAS=0
see it unchanged):

* ``title`` strings on schema nodes are dropped (pydantic noise; property
  names already say it);
* ``anyOf: [{"type": T, ...}, {"type": "null"}]`` becomes
  ``{"type": [T, "null"], ...}`` (same valid values, fewer tokens);
* ``default: null`` and ``examples`` are dropped;
* a parameter description that repeats the parameter name only ("The ra")
  is dropped;
* optionally, very long parameter descriptions are cut at a sentence
  boundary (``QUASAR_TOOL_PARAM_DESC_CHARS``; OFF by default: measured
  2026-10-02 it saved ~350 tokens over all 200 tools and cut value lists
  such as browse_alma_guidance's topic catalogue).

Measured on the full registry (2026-10-02): 39,090 -> 36,866 parameter
tokens (-5.7%). Most of a schema is JSON structure, not prose.
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict

ENV = "QUASAR_COMPACT_TOOL_SCHEMAS"


def enabled() -> bool:
    return os.getenv(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _desc_cap() -> int:
    """0 = never cut (default)."""
    try:
        cap = int(os.getenv("QUASAR_TOOL_PARAM_DESC_CHARS", "0"))
    except ValueError:
        return 0
    return max(80, cap) if cap > 0 else 0


_SENT_END = re.compile(r"(?<=[.!?;])\s+")


def _cut(text: str, cap: int) -> str:
    if cap <= 0:
        return text
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= cap:
        return text
    out = ""
    for part in _SENT_END.split(text):
        nxt = (out + " " + part).strip() if out else part
        if len(nxt) > cap:
            break
        out = nxt
    return out or text[:cap].rstrip()


def _trivial(desc: str, name: str) -> bool:
    d = re.sub(r"[^a-z0-9]", "", desc.lower())
    n = re.sub(r"[^a-z0-9]", "", name.lower())
    return bool(n) and d in (n, "the" + n, n + "parameter", "the" + n + "parameter")


# Keys whose values are literal data, never schema nodes (guard CX-05).
_LITERAL_KEYS = ("default", "enum", "const", "examples", "example")


def _node(node: Any, cap: int, name: str = "", required: bool = False) -> Any:
    if isinstance(node, list):
        return [_node(x, cap) for x in node]
    if not isinstance(node, dict):
        return node
    any_of = node.get("anyOf")
    # anyOf [{type: T}, {type: null}] -> {type: [T, "null"]}: the same set of
    # valid values (explicit null stays valid, guard CX-06), fewer tokens. Only
    # when the non-null branch is a BARE type (annotations allowed): any value
    # constraint (enum, pattern, items, minimum, ...) would also apply to null
    # after merging, so such unions keep their anyOf.
    if isinstance(any_of, list) and len(any_of) == 2:
        non_null = [x for x in any_of if not (isinstance(x, dict) and x.get("type") == "null")]
        nulls = [x for x in any_of if isinstance(x, dict) and x.get("type") == "null"]
        if (len(non_null) == 1 and isinstance(non_null[0], dict) and isinstance(non_null[0].get("type"), str)
                and non_null[0]["type"] != "null"
                and set(non_null[0]) <= {"type", "title", "description"}
                and len(nulls) == 1 and set(nulls[0]) <= {"type", "title", "description"}
                and not (set(node) - {"anyOf", "title", "description", "default"})):
            merged = {k: v for k, v in node.items() if k != "anyOf"}
            for k, v in non_null[0].items():
                merged.setdefault(k, v)
            merged["type"] = [non_null[0]["type"], "null"]
            node = merged
    out: Dict[str, Any] = {}
    for k, v in node.items():
        if k == "title" and isinstance(v, str):
            continue
        if k in ("examples", "example"):
            continue
        if k == "default" and v is None:
            continue
        if k == "properties" and isinstance(v, dict):
            req = set(node.get("required") or []) if isinstance(node.get("required"), list) else set()
            out[k] = {pn: _node(pv, cap, pn, pn in req) for pn, pv in v.items()}
            continue
        if k in _LITERAL_KEYS:
            out[k] = copy.deepcopy(v)
            continue
        if k == "description" and isinstance(v, str):
            if name and _trivial(v, name):
                continue
            out[k] = _cut(v, cap)
            continue
        out[k] = _node(v, cap)
    return out


def compact_parameters(parameters: Any) -> Any:
    """A compacted deep copy of a JSON-schema parameters object."""
    if not isinstance(parameters, dict):
        return parameters
    return _node(copy.deepcopy(parameters), _desc_cap())


__all__ = ["ENV", "enabled", "compact_parameters"]
