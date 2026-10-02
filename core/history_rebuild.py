# core/history_rebuild.py
"""
Rebuild a conversation's earlier turns when the provider chain is gone.

CALLED BY: core/runner.py (round 0, when there is no previous_response_id)
CALLS:     nothing (pure)

The LLM only "remembers" a chat because Quasar resends it. For the stateless
providers that resend lives in an in-process cache keyed by
conversation|provider|model (core/llm_client.ResponsesShim._history_cache,
core/agent._conv_response_ids). It is empty after a backend restart (every
deploy), after switching models mid-chat, and after eviction (>500 chats). The
saved conversation in the DB was never used to refill it, so the model
silently forgot the chat. This module turns the saved turns (already synced
into ``agent.memory`` from the DB each request, sse.py) into a bounded seed:
recent turns verbatim, older turns abridged, all within a token budget.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

RECAP_HEADER = "[Earlier in this conversation (restored from the saved chat; the newest turns are verbatim, older ones abridged)]"
RECAP_FOOTER = "[End of earlier conversation. The user's new message follows.]"


def _budget_tokens() -> int:
    try:
        return max(0, int(os.getenv("QUASAR_HISTORY_REBUILD_TOKENS", "6000")))
    except ValueError:
        return 6000


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _clip(text: str, chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= chars:
        return text
    return text[:chars].rstrip() + " [...]"


def _clean(text: str) -> str:
    # Saved assistant texts can end with long source/disclaimer footers and
    # markdown tables; keep prose, drop the bulk.
    text = re.sub(r"\n{3,}", "\n\n", text or "")
    return text.strip()


def prior_turns(history: List[Dict], current_query: str) -> List[Dict[str, str]]:
    """user/assistant messages before the current one, oldest first."""
    msgs = [
        {"role": str(m.get("role", "")), "content": _clean(str(m.get("content", "")))}
        for m in (history or [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ]
    if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"].strip() == (current_query or "").strip():
        msgs = msgs[:-1]
    while msgs and msgs[0]["role"] != "user":  # a seed must open with the user
        msgs = msgs[1:]
    return msgs


def build_seed(history: List[Dict], current_query: str, budget_tokens: int = None) -> Tuple[List[Dict[str, str]], int]:
    """Bounded seed: (messages, approx_tokens). Newest turns kept verbatim
    (each message clipped at 6,000 chars), older ones abridged to 300 chars,
    oldest dropped once the budget is spent."""
    budget = _budget_tokens() if budget_tokens is None else budget_tokens
    msgs = prior_turns(history, current_query)
    if not msgs or budget <= 0:
        return [], 0
    out: List[Dict[str, str]] = []
    used = 0
    verbatim_left = 4  # last 2 turns (user + assistant each) in full
    for m in reversed(msgs):
        text = _clip(m["content"], 6000) if verbatim_left > 0 else _clip(m["content"], 300)
        cost = _approx_tokens(text)
        if used + cost > budget:
            if verbatim_left > 0:  # try the abridged form before giving up
                text = _clip(m["content"], 300)
                cost = _approx_tokens(text)
            if used + cost > budget:
                break
        out.append({"role": m["role"], "content": text})
        used += cost
        verbatim_left -= 1
    out.reverse()
    while out and out[0]["role"] != "user":
        used -= _approx_tokens(out[0]["content"])
        out = out[1:]
    return out, used


def as_recap_text(seed: List[Dict[str, str]]) -> str:
    """Text form for providers without role-preserving list input."""
    if not seed:
        return ""
    lines = [RECAP_HEADER]
    for m in seed:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    lines.append(RECAP_FOOTER)
    return "\n\n".join(lines) + "\n\n"
