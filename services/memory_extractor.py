# services/memory_extractor.py
"""
Extract memory writes from the user's OWN typed message.

CALLED BY: ui-pro/api/sse.py (in parallel with the turn, off the answer path)
CALLS:     an LLM callable (gpt-oss-120b on TACC at low reasoning effort),
           services/user_memory_service.py

Input hygiene (duel task-d03fcd0-9562 DX-08): the extractor sees exactly the
raw user message (``request.message``, never the enriched turn), with chat
attachment payloads stripped, plus the slot schema and the user's current
values. It never sees tool output, uploaded documents, web pages or assistant
text, so a poisoned document cannot write memory. Each op must:

* be a first-person assertion (not a quote, hypothetical, negation of
  something else, question, or a statement about another person);
* carry an evidence span that occurs verbatim in the raw message;
* pass the slot's type validation (no links, no instructions).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from services.user_memory_service import (
    SLOTS,
    MemoryValidationError,
    UserMemoryService,
    normalize_value,
)

# Cheap gate: most turns are plain questions and never reach the LLM.
_PREFILTER_RE = re.compile(
    r"\b(?:remember|from\s+now\s+on|going\s+forward|in\s+(?:the\s+)?future|for\s+(?:all\s+)?future|"
    r"always|never|prefer|preference|i'?d\s+like|i\s+like|please\s+use|i\s+use|i\s+usually|i\s+no\s+longer|"
    r"i\s+switched|switch(?:ed)?\s+to|instead\s+of|i\s+work\s+(?:on|with)|i'?m\s+working\s+on|i\s+am\s+working\s+on|"
    r"my\s+(?:research|field|focus|project|target|targets|thesis|proposal|group|team)|i\s+study|i'?m\s+an?\b|i\s+am\s+an?\b|"
    r"i'?m\s+new\s+to|forget|stop\s+using|don'?t\s+use|cite|citation|bibcode|units?\b|jy/beam|kelvin|decimal\s+degrees|"
    r"sexagesimal|lsrk|barycentric|undergrad|postdoc|phd)\b",
    re.IGNORECASE,
)
_EXPLICIT_RE = re.compile(
    r"\b(?:remember|from\s+now\s+on|going\s+forward|in\s+(?:the\s+)?future|for\s+(?:all\s+)?future|"
    r"update\s+(?:to\s+)?my\s+preferences?|save\s+(?:this|that)|note\s+(?:this|that)\s+for|"
    r"forget\s+(?:that|my|about\s+my)|stop\s+using)\b",
    re.IGNORECASE,
)

MAX_OPS = 8

# Own small pool: extraction must never take a slot from the chat executor.
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory-extract")


class _Op(BaseModel):
    op: str = Field(pattern=r"^(set|add|remove|clear)$")
    slot: str
    value: Any = None
    evidence: str = Field(min_length=3, max_length=400)
    kind: str


class _Ops(BaseModel):
    ops: List[_Op] = Field(default_factory=list)


def should_extract(message: str) -> bool:
    return bool(message and len(message) <= 4000 and _PREFILTER_RE.search(message))


def is_explicit(message: str) -> bool:
    return bool(_EXPLICIT_RE.search(message or ""))


def _sentence_of(message: str, evidence: str) -> str:
    """The sentence(s) of ``message`` that contain ``evidence`` (normalised),
    so explicitness is judged per op, not for the whole message."""
    ev = _norm(evidence)
    sentences = re.split(r"(?<=[.!?;:\n])\s+", message or "")
    hit = [x for x in sentences if ev and (ev in _norm(x) or _norm(x) in ev)]
    return " ".join(hit) if hit else ""


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = t.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                                   "‐": "-", "‑": "-", "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", t).strip().strip(".!?,;:\"'")


def _schema_text() -> str:
    lines = []
    for key, spec in SLOTS.items():
        if spec["kind"] == "enum":
            lines.append(f"- {key} (one of: {', '.join(spec['choices'])}): {spec['label']}")
        elif spec["kind"] == "list":
            lines.append(f"- {key} (list of short strings): {spec['label']}")
        else:
            lines.append(f"- {key} (short text): {spec['label']}")
    return "\n".join(lines)


INSTRUCTIONS = (
    "You extract durable user preferences and research context for an astronomy assistant's memory. "
    "Return ONLY a JSON object {\"ops\": [...]} and nothing else. Each op: "
    "{\"op\": \"set\"|\"add\"|\"remove\"|\"clear\", \"slot\": <slot key>, \"value\": <value or null>, "
    "\"evidence\": <the exact words from the user message that state it, copied verbatim>, "
    "\"kind\": \"assertion\"|\"quote\"|\"hypothetical\"|\"question\"|\"third_party\"|\"temporary\"}.\n"
    "Rules: only the user's own durable first-person preferences or facts about themselves count as "
    "\"assertion\". Use \"quote\" for text the user cites from a paper or someone else, \"hypothetical\" for "
    "'if I were...'/'suppose', \"question\" for questions, \"third_party\" for someone else's preference "
    "('my advisor prefers'), \"temporary\" for things that apply only to this one request ('for this plot use K'). "
    "\"set\" replaces a value; \"add\"/\"remove\" change one list entry; \"clear\" forgets a slot "
    "('I no longer...', 'forget my...'). If nothing qualifies return {\"ops\": []}. Never invent values.\n"
    "Slots:\n"
)


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.S)
    try:
        return json.loads(t)
    except Exception:
        pass
    # gpt-oss sometimes emits near-JSON (observed live: a stray "]" closing
    # the ops array twice). Recover every well-formed op object on its own;
    # each still goes through full validation below.
    dec = json.JSONDecoder()
    ops = []
    for m in re.finditer(r"\{", t):
        try:
            obj, _end = dec.raw_decode(t, m.start())
        except Exception:
            continue
        if isinstance(obj, dict) and "op" in obj and "slot" in obj:
            ops.append(obj)
    return {"ops": ops} if ops else None


def propose_ops(message: str, current: List[Dict[str, Any]], llm_call: Callable[[str, str], str]) -> List[Dict[str, Any]]:
    """Ask the LLM for candidate ops and keep only those that pass every check."""
    current_txt = "\n".join(f"- {it['slot']}: {json.dumps(it['value'], ensure_ascii=False)}" for it in current) or "(none)"
    instructions = INSTRUCTIONS + _schema_text()
    user_input = (f"Current saved values:\n{current_txt}\n\nUser message (the ONLY source you may extract from):\n"
                  f"<<<\n{message}\n>>>")
    raw = llm_call(instructions, user_input)
    data = _parse_json(raw)
    if data is None:
        print(f"[MEMORY] extractor output not parseable: {str(raw)[:200]!r}")
        return []
    try:
        parsed = _Ops.model_validate(data)
    except ValidationError as e:
        print(f"[MEMORY] extractor output failed schema: {str(e)[:200]}")
        return []
    msg_norm = _norm(message)
    out: List[Dict[str, Any]] = []
    for op in parsed.ops[:MAX_OPS]:
        if op.kind != "assertion" or op.slot not in SLOTS:
            continue
        ev = _norm(op.evidence)
        if len(ev) < 3 or ev not in msg_norm:
            print(f"[MEMORY] dropped op {op.slot}: evidence not verbatim in the user's message")
            continue  # evidence must be the user's own words, verbatim
        out.append(op.model_dump())
    return out


def apply_ops(svc: UserMemoryService, user_id: str, ops: List[Dict[str, Any]], *, explicit: bool,
              conversation_id: Optional[str], observed_at: float,
              message: Optional[str] = None) -> List[Dict[str, Any]]:
    """Write validated ops; returns the saved events (for the UI chip).

    With ``message``, an op is explicit (confirmed) only when the sentence
    holding its evidence, or the message's opening request, asks to remember;
    "I always forget the bibcode" no longer makes every op confirmed.
    """
    current = {it["slot"]: it for it in svc.active_items(user_id)}
    events: List[Dict[str, Any]] = []
    lead = re.split(r"(?<=[.!?;:\n])\s+", message or "")[0] if message else ""
    for op in ops:
        if message is None:
            op_explicit = explicit
        else:
            op_explicit = explicit and (is_explicit(_sentence_of(message, op["evidence"])) or is_explicit(lead))
        source = "chat_explicit" if op_explicit else "chat_inferred"
        slot, kind = op["slot"], SLOTS[op["slot"]]["kind"]
        try:
            if op["op"] == "clear":
                status, ev = svc.forget_slot(user_id, slot, source=source, conversation_id=conversation_id,
                                             observed_at=observed_at)
            elif op["op"] in ("add", "remove") and kind == "list":
                existing = list((current.get(slot) or {}).get("value") or [])
                item = normalize_value(slot, op["value"])  # -> list
                if op["op"] == "add":
                    new = existing + [x for x in item if x.lower() not in {e.lower() for e in existing}]
                else:
                    drop = {x.lower() for x in item}
                    new = [e for e in existing if e.lower() not in drop]
                if not new:
                    status, ev = svc.forget_slot(user_id, slot, source=source, conversation_id=conversation_id,
                                                 observed_at=observed_at)
                else:
                    status, ev = svc.set_slot(user_id, slot, new, source=source, evidence=op["evidence"],
                                              conversation_id=conversation_id, observed_at=observed_at)
            elif op["op"] in ("set", "add"):
                value = op["value"]
                if kind == "list" and not isinstance(value, list):
                    value = [value]
                status, ev = svc.set_slot(user_id, slot, value, source=source, evidence=op["evidence"],
                                          conversation_id=conversation_id, observed_at=observed_at)
            else:
                continue
        except MemoryValidationError as e:
            print(f"[MEMORY] rejected op {op.get('op')} {slot}: {e}")
            continue
        if status == "saved" and ev:
            events.append(ev)
            current = {it["slot"]: it for it in svc.active_items(user_id)}
        else:
            print(f"[MEMORY] op {op.get('op')} {slot} -> {status}")
    return events


def extract_and_save(svc: UserMemoryService, user_id: str, message: str, llm_call: Callable[[str, str], str], *,
                     conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Full pipeline for one user message. Never raises."""
    observed_at = time.time()
    try:
        if not should_extract(message) or svc.is_paused(user_id):
            return []
        ops = propose_ops(message, svc.active_items(user_id), llm_call)
        if not ops:
            return []
        return apply_ops(svc, user_id, ops, explicit=is_explicit(message),
                         conversation_id=conversation_id, observed_at=observed_at, message=message)
    except Exception as e:
        print(f"[MEMORY] extraction failed (non-fatal): {type(e).__name__}: {e}")
        return []
