# core/history_compact.py
"""
Bounded chat history for the chat-shim providers (token plan rank 4).

CALLED BY: core/llm_client.ResponsesShim._chat_messages_for_input (on every
           NEW user turn, before the turn's input is appended)
CALLS:     nothing (pure; never mutates the cached message dicts)

The shim cache replays every earlier turn, tool call and tool output on
every call, so a long data chat grew without bound. On each new user turn:

* the last ``QUASAR_HISTORY_KEEP_TURNS`` (3) turns keep their text verbatim;
* tool outputs older than the last ``QUASAR_HISTORY_KEEP_TOOL_TURNS`` (2)
  turns become one-line stubs (tool name + the head of the output; the call
  id is kept so tool_call/tool pairing stays valid). The data cards stay in
  the UI and the model can re-run a query;
* older turns' user messages lose the per-turn context the runner put in
  front of the question (documentation excerpts, web evidence) and are
  clipped; older assistant answers are clipped;
* if the compacted older turns still exceed ``QUASAR_HISTORY_CAP_TOKENS``
  (8000), the oldest turns are dropped whole and their questions are listed
  in a short note on the first kept user message.

The compacted list becomes the stored history after the round, so each
stub is written once and stays byte-identical afterwards (prefix caches).
``QUASAR_HISTORY_COMPACT=0`` turns it off.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, List, Optional

STUB_PREFIX = "[Earlier tool output, shortened to save context:"
NOTE_PREFIX = "[Earlier turns removed to save context. The user had asked:"
_USER_MARKER = "\n\nUser: "


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def enabled() -> bool:
    return os.getenv("QUASAR_HISTORY_COMPACT", "1").strip().lower() not in ("0", "false", "no", "off")


def _approx_tokens(m: dict) -> int:
    n = 0
    c = m.get("content")
    if isinstance(c, str):
        n += len(c)
    elif c is not None:
        n += len(json.dumps(c, default=str))
    if m.get("tool_calls"):
        n += len(json.dumps(m.get("tool_calls"), default=str))
    return n // 4 + 4


def _clip(text: str, chars: int) -> str:
    text = text or ""
    return text if len(text) <= chars else text[:chars].rstrip() + " [...]"


def _one_line(text: str, chars: int) -> str:
    return _clip(re.sub(r"\s+", " ", text or "").strip(), chars)


# The runner's fixed notes that sit between its context blocks (documentation
# excerpts, web evidence) and the "User: " marker (core/runner.py full_input).
_RUNNER_NOTES = ("\n\nIMPORTANT CITATION & STRUCTURE RULES:", "\n\nNOTE: Web search is DISABLED")


def _user_question(content: str) -> str:
    """The question part of a runner-built user message (drops the context
    blocks placed in front of 'User: '). With documentation/web context, the
    runner's marker is the first one AFTER its fixed notes, so a "User:" line
    inside an excerpt or inside the question itself is never mistaken for it
    (guard CX-07)."""
    start = 0
    for note in _RUNNER_NOTES:
        j = content.rfind(note)
        if j >= 0:
            start = max(start, j + len(note))
    i = content.find(_USER_MARKER, start)
    if i < 0 and start:
        i = content.find(_USER_MARKER)
    if i >= 0:
        return content[i + 2:]
    return content


def _clip_middle(text: str, chars: int) -> str:
    """Keep the head (the question's start) and the tail (its end and the
    route directives), cut the middle."""
    if len(text) <= chars:
        return text
    head = chars * 2 // 5
    return text[:head].rstrip() + " [...] " + text[-(chars - head):].lstrip()


def split_turns(body: List[dict], is_new_user_turn: Callable[[str], bool]) -> List[List[dict]]:
    turns: List[List[dict]] = []
    for m in body:
        starts = (isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
                  and is_new_user_turn(m["content"]))
        if starts or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    return turns


def compact_history(
    messages: List[dict],
    is_new_user_turn: Callable[[str], bool],
    *,
    keep_turns: Optional[int] = None,
    keep_tool_turns: Optional[int] = None,
    cap_tokens: Optional[int] = None,
) -> List[dict]:
    """Compacted copy of a Chat Completions history (system message first, if any)."""
    if not messages:
        return messages
    keep_turns = _env_int("QUASAR_HISTORY_KEEP_TURNS", 3, 1) if keep_turns is None else keep_turns
    keep_tool_turns = _env_int("QUASAR_HISTORY_KEEP_TOOL_TURNS", 2, 0) if keep_tool_turns is None else keep_tool_turns
    cap_tokens = _env_int("QUASAR_HISTORY_CAP_TOKENS", 8000, 500) if cap_tokens is None else cap_tokens
    head = [messages[0]] if isinstance(messages[0], dict) and messages[0].get("role") == "system" else []
    body = messages[len(head):]
    turns = split_turns(body, is_new_user_turn)
    if len(turns) <= min(keep_turns, keep_tool_turns) and len(turns) <= keep_turns:
        return messages

    names: Dict[str, str] = {}
    for m in body:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                names[str((tc or {}).get("id", ""))] = str(fn.get("name", "") or "tool")

    n = len(turns)
    out_turns: List[List[dict]] = []
    for idx, turn in enumerate(turns):
        age = n - idx  # 1 = most recent stored turn
        stub_tools = age > keep_tool_turns
        old_text = age > keep_turns
        new_turn = []
        for m in turn:
            if not isinstance(m, dict):
                new_turn.append(m)
                continue
            role = m.get("role")
            c = m.get("content")
            if role == "tool" and stub_tools and isinstance(c, str) and not c.startswith(STUB_PREFIX) \
                    and len(c) > 400:
                name = names.get(str(m.get("tool_call_id", "")), "tool")
                m = {**m, "content": f"{STUB_PREFIX} {name}: {_one_line(c, 300)}]"}
            elif role == "user" and old_text and isinstance(c, str) and not c.startswith(NOTE_PREFIX):
                q = _clip_middle(_user_question(c), 1500)
                if q != c:
                    m = {**m, "content": q}
            elif role == "assistant" and old_text and isinstance(c, str) and len(c) > 3000:
                m = {**m, "content": _clip(c, 3000)}
            new_turn.append(m)
        out_turns.append(new_turn)

    # Hard cap on the turns older than the verbatim window.
    old = out_turns[:-keep_turns] if len(out_turns) > keep_turns else []
    recent = out_turns[len(old):]
    dropped_questions: List[str] = []
    prior_note = ""
    while old and sum(_approx_tokens(m) for t in old for m in t if isinstance(m, dict)) > cap_tokens:
        t = old.pop(0)
        first = t[0] if t else {}
        c = first.get("content") if isinstance(first, dict) else ""
        if isinstance(c, str) and c.startswith(NOTE_PREFIX):
            end = c.find("]\n\n")
            prior_note = c[len(NOTE_PREFIX):end].strip() if end > 0 else ""
            c = c[end + 3:] if end > 0 else ""
        if isinstance(c, str) and c.strip():
            dropped_questions.append(_one_line(re.sub(r"^User:\s*", "", _user_question(c)), 140))
    kept = old + recent
    if dropped_questions or prior_note:
        asked = ([prior_note] if prior_note else []) + [f'"{q}"' for q in dropped_questions]
        note = NOTE_PREFIX + " " + " | ".join(asked)[-1500:] + "]\n\n"
        if kept and kept[0] and isinstance(kept[0][0], dict) and isinstance(kept[0][0].get("content"), str):
            first = kept[0][0]
            fc = first["content"]
            if fc.startswith(NOTE_PREFIX):  # merge with an existing note
                end = fc.find("]\n\n")
                fc = fc[end + 3:] if end > 0 else fc
            kept[0] = [{**first, "content": note + fc}] + kept[0][1:]
    flat = [m for t in kept for m in t]
    return head + flat


__all__ = ["STUB_PREFIX", "NOTE_PREFIX", "enabled", "compact_history", "split_turns"]
