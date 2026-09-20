"""Provider-independent safeguards for tool-loop progress and output."""
from __future__ import annotations

import copy
import json


def tool_call_key(name, args):
    return name, json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


def serialize_tool_result(result, max_chars=50000):
    """Keep result JSON valid and disclose any context reduction explicitly."""
    text = json.dumps(result, default=str)
    if len(text) <= max_chars:
        return text
    reduced = copy.deepcopy(result)
    if isinstance(reduced, dict):
        reduced.update(context_truncated=True, partial=True)
        reduced["context_note"] = "Result exceeds the model context allowance; this is an incomplete preview."
        while len(json.dumps(reduced, default=str)) > max_chars:
            lists = [(k, v) for k, v in reduced.items() if isinstance(v, list) and v]
            if not lists:
                break
            key, rows = max(lists, key=lambda kv: len(json.dumps(kv[1], default=str)))
            reduced[key] = rows[:len(rows) // 2]
        text = json.dumps(reduced, default=str)
        if len(text) <= max_chars:
            return text
    return json.dumps({"partial": True, "context_truncated": True,
                       "preview_text": text[:max_chars // 2]})


def compact_tool_outputs(results, max_chars=80000):
    """Budget the entire evidence set, with duplicate query results counted once."""
    unique = []
    seen = set()
    for item in reversed(results):
        raw = item.get("output", "{}")
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            value = {"partial": True, "text": str(raw)}
        if isinstance(value, dict) and value.get("repeated_call"):
            value = value.get("previous_result", value)
        key = json.dumps(value, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    # Bound both the number and size of records. A metadata record discloses
    # older omissions instead of silently dropping evidence.
    omitted = max(0, len(unique) - 24)
    unique = unique[:24]
    per_result = max(256, (max_chars - 1000) // max(1, len(unique)))
    out = [serialize_tool_result(value, per_result) for value in reversed(unique)]
    if omitted:
        out.insert(0, json.dumps({"partial": True, "omitted_older_results": omitted}))
    return out


def partial_tool_answer(results, reason):
    evidence = []
    for raw in compact_tool_outputs(results, max_chars=48000):
        try:
            evidence.append(json.loads(raw))
        except (ValueError, TypeError):
            evidence.append({"partial": True, "text": raw})
    return json.dumps({"partial": True, "reason": reason, "tool_results": evidence}, default=str)
