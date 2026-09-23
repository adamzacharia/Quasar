"""Provider-independent safeguards for tool-loop progress and output."""
from __future__ import annotations

import copy
import json
import re


def tool_call_key(name, args):
    return name, json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


# Arguments that only change how a result is LABELLED, never what is fetched
# or computed. Two calls that differ only in these are the same call (UI
# benchmark 2026-09-22, L06: five datalab_color_magnitude_diagram calls ~70 s
# each, differing in title/labels, every one a ReadTimeout).
# Presentation-only arguments. NOT here (guard CX-13): width / height / dpi /
# figsize (cutout pixel sizes, image resolution) and description (ADS library
# text) -- those change what a tool returns.
COSMETIC_ARG_KEYS = frozenset({
    "title", "label", "labels", "x_label", "y_label", "xlabel", "ylabel", "plot_title",
    "subtitle", "caption", "note", "notes", "legend", "legend_title",
    "color", "colour", "cmap", "colormap", "style",
    "marker", "alpha", "point_size", "markersize", "grid", "theme",
})

_WS_RE = re.compile(r"\s+")


def _canonical_value(value):
    if isinstance(value, str):
        text = _WS_RE.sub(" ", value).strip()
        # SQL / ADQL and identifiers are case-insensitive for our purposes;
        # a trailing semicolon or comment-only change is cosmetic too.
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        text = _WS_RE.sub(" ", text).strip().rstrip(";").strip()
        return text.lower()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {str(k).lower(): _canonical_value(v) for k, v in value.items() if str(k).lower() not in COSMETIC_ARG_KEYS}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(v) for v in value]
    return value


def canonical_tool_call_key(name, args):
    """``(tool, canonical args)``: cosmetic keys dropped, whitespace/case
    normalised, floats rounded -- so a re-issue that only re-labels a plot or
    re-spaces a query is recognised as the SAME call."""
    canon = _canonical_value(args if isinstance(args, dict) else {"_": args})
    return str(name or ""), json.dumps(canon, sort_keys=True, separators=(",", ":"), default=str)


def tool_result_failure_reason(result):
    """A short reason when ``result`` (dict or its JSON text) represents a
    FAILED tool call -- ``success: False``, a top-level ``error`` without an
    explicit success, a timeout / breaker / budget stop -- else None. Used by
    the runner's repeated-call detector: an identical call after a failure is
    not re-executed."""
    parsed = result
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (ValueError, TypeError):
            return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("repeated_call") and isinstance(parsed.get("previous_result"), dict):
        parsed = parsed["previous_result"]
    if parsed.get("timeout") is True:
        return str(parsed.get("error") or "timed out")[:300]
    if parsed.get("infrastructure_failure") or parsed.get("circuit_breaker"):
        return str(parsed.get("error") or "service unreachable (circuit open)")[:300]
    if parsed.get("cancelled") is True:
        return str(parsed.get("error") or "turn cancelled")[:300]
    if parsed.get("success") is False:
        return str(parsed.get("error") or parsed.get("message") or "the tool reported failure")[:300]
    if parsed.get("error") and parsed.get("success") is not True:
        return str(parsed.get("error"))[:300]
    return None


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
