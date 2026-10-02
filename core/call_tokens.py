# core/call_tokens.py
"""
Per-call input-token breakdown (token plan rank 7).

CALLED BY: core/llm_client.ResponsesShim.create (every LLM call)
CALLS:     tiktoken (o200k_base; gpt-oss harmony uses the same BPE family)

Each call logs one line next to the provider lines:

  [CALL TOKENS] tacc gpt-oss-120b sys=4180 tools=9312(n=41) hist=2210(tool_out=1180)
                rag=0 new_tool_out=5400 new_msg=130 est=21232

and, after the provider reports usage, one ``[CALL BILLED]`` line with the
billed input and cached tokens. Counts are local o200k estimates of what
Quasar sends (provider chat templates add a few hundred tokens of framing;
the billed line is the ground truth). ``QUASAR_CALL_TOKENS_LOG=<path>``
also appends one JSON object per call (estimate + billed) for the eval
cost tables. ``QUASAR_CALL_TOKENS=0`` turns the accounting off.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Iterable, Optional

_lock = threading.Lock()
_enc = None
_tool_cache: Dict[Any, int] = {}


def enabled() -> bool:
    return os.getenv("QUASAR_CALL_TOKENS", "1").strip().lower() not in ("0", "false", "no", "off")


def count(text: Any) -> int:
    global _enc
    if text is None:
        return 0
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, default=str)
        except Exception:
            text = str(text)
    if not text:
        return 0
    try:
        if _enc is None:
            import tiktoken

            _enc = tiktoken.get_encoding("o200k_base")
        return len(_enc.encode_ordinary(text))
    except Exception:  # pragma: no cover - tokenizer missing
        return max(1, len(text) // 4)


def tools_tokens(tools: Optional[Iterable[dict]]) -> int:
    tools = list(tools or [])
    if not tools:
        return 0
    key = tuple((t.get("name"), id(t.get("parameters")), len(str(t.get("description", "")))) for t in tools
                if isinstance(t, dict))
    with _lock:
        hit = _tool_cache.get(key)
    if hit is not None:
        return hit
    n = count(tools)
    with _lock:
        if len(_tool_cache) > 512:
            _tool_cache.clear()
        _tool_cache[key] = n
    return n


def _msg_tokens(messages: Iterable[Any]) -> Dict[str, int]:
    """(all, tool outputs) tokens of chat-completions messages, system excluded."""
    total = tool_out = 0
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") == "system":
            continue
        c = m.get("content")
        n = count(c) if c else 0
        if m.get("tool_calls"):
            n += count(m.get("tool_calls"))
        if m.get("role") == "tool" or (isinstance(c, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in c)):
            tool_out += n
        total += n
    return {"all": total, "tool_out": tool_out}


def breakdown(kwargs: Dict[str, Any], history: Optional[list], rag_tokens: int = 0) -> Dict[str, Any]:
    """Estimate of what this call sends. ``history`` = the cached chat for
    previous_response_id (chat-shim providers) or None when the provider
    keeps it server-side (OpenAI previous_response_id)."""
    inp = kwargs.get("input")
    new_tool_out = new_msg = 0
    if isinstance(inp, list):
        for it in inp:
            if not isinstance(it, dict):
                new_msg += count(it)
            elif it.get("type") == "function_call_output":
                new_tool_out += count(it.get("output", ""))
            elif it.get("type") == "function_call":
                new_msg += count(it.get("arguments", "")) + 8
            else:
                new_msg += count(it.get("content", ""))
    else:
        new_msg = count(inp)
    rag = min(new_msg, max(0, int(rag_tokens or 0)))
    tools = kwargs.get("tools") or []
    h = _msg_tokens(history) if history is not None else None
    out = {
        "sys": count(kwargs.get("instructions") or ""),
        "tools": tools_tokens(tools),
        "n_tools": len(tools),
        "hist": None if h is None else h["all"],
        "hist_tool_out": None if h is None else h["tool_out"],
        "rag": rag,
        "new_tool_out": new_tool_out,
        "new_msg": max(0, new_msg - rag),
    }
    out["est"] = sum(v for k, v in out.items() if k in ("sys", "tools", "hist", "new_tool_out", "new_msg", "rag")
                     and isinstance(v, int))
    return out


def log_estimate(provider: str, model: str, bd: Dict[str, Any]) -> None:
    hist = "server" if bd.get("hist") is None else f"{bd['hist']}(tool_out={bd['hist_tool_out']})"
    att = f" attachments={bd['attachments']}(not in est)" if bd.get("attachments") else ""
    print(
        f"[CALL TOKENS] {provider} {model} sys={bd['sys']} tools={bd['tools']}(n={bd['n_tools']}) hist={hist} "
        f"rag={bd['rag']} new_tool_out={bd['new_tool_out']} new_msg={bd['new_msg']} est={bd['est']}{att}"
    )


def _usage_fields(usage: Any) -> Dict[str, Optional[int]]:
    if usage is None:
        return {"billed_in": None, "cached": None, "billed_out": None}
    billed_in = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    billed_out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    cached = getattr(usage, "cache_hit_tokens", None)
    if cached is None:
        det = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
        cached = getattr(det, "cached_tokens", None) if det is not None else None
        if cached is None and isinstance(det, dict):
            cached = det.get("cached_tokens")
    return {"billed_in": billed_in, "cached": cached, "billed_out": billed_out}


def log_billed(provider: str, model: str, bd: Optional[Dict[str, Any]], usage: Any, meta: Optional[dict] = None) -> None:
    f = _usage_fields(usage)
    if f["billed_in"] is not None:
        est = (bd or {}).get("est")
        print(f"[CALL BILLED] {provider} {model} input={f['billed_in']} cached={f['cached']} "
              f"output={f['billed_out']} est={est}")
    path = os.getenv("QUASAR_CALL_TOKENS_LOG", "").strip()
    if not path:
        return
    rec = {"ts": time.time(), "provider": provider, "model": model, **(bd or {}), **f, **(meta or {})}
    try:
        with _lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - accounting must never bite
        print(f"[CALL TOKENS] could not append to {path}: {exc}")


__all__ = ["enabled", "count", "tools_tokens", "breakdown", "log_estimate", "log_billed"]
