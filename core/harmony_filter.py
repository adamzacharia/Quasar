"""core/harmony_filter.py — strip leaked gpt-oss "harmony" control markup from
assistant text.

gpt-oss speaks the harmony format internally: every message is
``<|start|>assistant<|channel|>NAME [to=functions.TOOL] [<|constrain|>json]
<|message|>BODY`` closed by ``<|call|>`` (tool call), ``<|end|>`` or
``<|return|>``. vLLM normally parses this and returns clean ``content`` /
``tool_calls`` fields, but live 2026-09-21 23:56 (gpt-oss-120b via TACC,
round after a tool result) the chat-completions stream delivered BOTH a proper
structured tool call AND the raw harmony text of that same call as ``content``::

    <|channel|>commentary to=functions.match_perseus_protostars_alma_jwst
    <|constrain|>json<|message|>{"max_alma_rows":5000,...,"radius_arcsec":5}

The UI rendered that verbatim above the answer. This module removes such
spans from the user-visible text without touching structured tool calls:

* every ``<|name|>`` control token is dropped;
* a ``<|channel|>…<|message|>`` header is dropped;
* the body that follows a header addressed to a tool (``to=…`` or
  ``<|constrain|>``) is dropped — it is a tool-call payload, never prose — and
  recorded in :attr:`HarmonyStreamFilter.dropped_calls` so the caller can log
  it against the structured calls of the same round;
* the body of an ``analysis`` channel is chain-of-thought and is returned as
  reasoning text instead of visible text;
* a ``final`` (or unrecognised) channel body is visible text.

The filter is streaming-safe: a control token split across deltas (``<|``,
``chan``, ``nel|>``) is held back until it can be classified, and a trailing
``<`` that never becomes a token is released on :meth:`flush`.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

__all__ = ["HarmonyStreamFilter", "strip_harmony_markup"]

# A complete control token: <|channel|>, <|message|>, <|constrain|>, <|call|>,
# <|end|>, <|return|>, <|start|>, <|endoftext|>, <|reserved_200012|>, ...
_TOKEN_RE = re.compile(r"<\|([A-Za-z_][A-Za-z0-9_]*)\|>")
# What an incomplete token looks like at the end of a delta ("<|", "<|chan",
# "<|channel|"). Anything longer than this without a closing "|>" is prose.
_PARTIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z0-9_]{0,32}\|?")
_RECIPIENT_RE = re.compile(r"\bto=([A-Za-z0-9_.\-]+)")

_HARMONY_TOKENS = ("channel", "message", "constrain", "call", "end", "return", "start")
_HEADER_OPENERS = frozenset({"channel", "start"})
_TERMINATORS = frozenset({"call", "end", "return"})
_CONSTRAIN_MARK = "<|constrain|>"

_MODE_TEXT = "text"
_MODE_HEADER = "header"
_MODE_BODY = "body"

_BODY_TOOL = "tool"
_BODY_REASONING = "reasoning"
_BODY_TEXT = "text"


class HarmonyStreamFilter:
    """Stateful filter for one assistant message stream.

    Feed each content delta to :meth:`feed`; it returns the user-visible text
    and any chain-of-thought text recovered from a leaked ``analysis`` channel.
    Call :meth:`flush` once the stream ends.
    """

    def __init__(self) -> None:
        self._mode = _MODE_TEXT
        self._pending = ""       # possible partial control token held back
        self._header = ""        # header text between <|channel|> and <|message|>
        self._body_kind = _BODY_TEXT
        self._body_buf = ""      # dropped tool-call payload (for the log)
        self._body_tool: Optional[str] = None
        self._json_depth = 0     # brace depth of a tool payload (ends it without <|call|>)
        self._json_in_string = False
        self._json_escape = False
        self._json_started = False
        self.dropped_calls: List[Dict[str, str]] = []
        self.dropped_tokens = 0
        self.saw_markup = False

    # ------------------------------------------------------------------ public
    def feed(self, delta: str) -> Tuple[str, str]:
        """Consume one content delta → ``(visible_text, reasoning_text)``."""
        if not delta:
            return "", ""
        if self._mode == _MODE_TEXT and not self._pending and "<" not in delta:
            return delta, ""  # fast path: ordinary prose

        buf = self._pending + delta
        self._pending = ""
        visible: List[str] = []
        reasoning: List[str] = []
        pos, n = 0, len(buf)
        while pos < n:
            lt = buf.find("<|", pos)
            if lt == -1:
                if buf.endswith("<"):
                    self._emit(buf[pos:-1], visible, reasoning)
                    self._pending = "<"
                else:
                    self._emit(buf[pos:], visible, reasoning)
                break
            self._emit(buf[pos:lt], visible, reasoning)
            m = _TOKEN_RE.match(buf, lt)
            if m is None:
                tail = buf[lt:]
                if _PARTIAL_TOKEN_RE.fullmatch(tail):
                    self._pending = tail  # token may complete in the next delta
                    break
                self._emit("<|", visible, reasoning)  # not a token: literal text
                pos = lt + 2
                continue
            self._on_token(m.group(1))
            pos = m.end()
        return "".join(visible), "".join(reasoning)

    def flush(self) -> Tuple[str, str]:
        """End of stream: release or discard held-back state."""
        visible: List[str] = []
        reasoning: List[str] = []
        pending, self._pending = self._pending, ""
        if pending:
            if pending == "<" or not any(
                f"<|{tok}|>".startswith(pending) for tok in _HARMONY_TOKENS
            ):
                self._emit(pending, visible, reasoning)  # never was a harmony token
            else:
                self.dropped_tokens += 1  # a truncated harmony token
        if self._mode == _MODE_BODY:
            self._close_body()
        elif self._mode == _MODE_HEADER:
            self._header = ""
        self._mode = _MODE_TEXT
        return "".join(visible), "".join(reasoning)

    # ----------------------------------------------------------------- private
    def _emit(self, text: str, visible: List[str], reasoning: List[str]) -> None:
        if not text:
            return
        if self._mode == _MODE_TEXT:
            visible.append(text)
        elif self._mode == _MODE_HEADER:
            self._header += text
        elif self._body_kind == _BODY_REASONING:
            reasoning.append(text)
        elif self._body_kind == _BODY_TOOL:
            self._consume_tool_body(text, visible, reasoning)
        else:
            visible.append(text)

    def _consume_tool_body(self, text: str, visible: List[str], reasoning: List[str]) -> None:
        """Swallow a tool-call payload. A balanced JSON object ends the body even
        without a ``<|call|>`` token (that token is a stop token and rarely
        survives into ``content``); text after it is ordinary prose again."""
        for i, ch in enumerate(text):
            self._body_buf += ch
            if not self._json_started:
                if ch.isspace():
                    continue
                if ch != "{":
                    # Not JSON: swallow until a control token ends the message.
                    self._body_buf += text[i + 1:]
                    return
                self._json_started = True
                self._json_depth = 1
                continue
            if self._json_in_string:
                if self._json_escape:
                    self._json_escape = False
                elif ch == "\\":
                    self._json_escape = True
                elif ch == '"':
                    self._json_in_string = False
                continue
            if ch == '"':
                self._json_in_string = True
            elif ch == "{":
                self._json_depth += 1
            elif ch == "}":
                self._json_depth -= 1
                if self._json_depth == 0:
                    self._close_body()
                    self._mode = _MODE_TEXT
                    self._emit(text[i + 1:], visible, reasoning)
                    return

    def _on_token(self, name: str) -> None:
        self.saw_markup = True
        self.dropped_tokens += 1
        if name in _HEADER_OPENERS:
            if self._mode == _MODE_BODY:
                self._close_body()
            self._mode = _MODE_HEADER
            self._header = ""
        elif name == "constrain":
            if self._mode == _MODE_HEADER:
                self._header += _CONSTRAIN_MARK
        elif name == "message":
            if self._mode == _MODE_HEADER:
                self._open_body(self._header)
                self._header = ""
                self._mode = _MODE_BODY
        elif name in _TERMINATORS:
            if self._mode == _MODE_BODY:
                self._close_body()
            self._header = ""
            self._mode = _MODE_TEXT
        # any other <|token|> (e.g. <|endoftext|>) is simply dropped

    def _open_body(self, header: str) -> None:
        h = header.strip()
        channel = h.split()[0].lower() if h else ""
        recipient_match = _RECIPIENT_RE.search(h)
        recipient = recipient_match.group(1) if recipient_match else None
        self._body_buf = ""
        self._json_depth = 0
        self._json_in_string = False
        self._json_escape = False
        self._json_started = False
        if recipient or _CONSTRAIN_MARK in h:
            self._body_kind = _BODY_TOOL
            tool = recipient or ""
            if tool.startswith("functions."):
                tool = tool[len("functions."):]
            self._body_tool = tool or None
        elif channel == "analysis":
            self._body_kind = _BODY_REASONING
            self._body_tool = None
        else:
            self._body_kind = _BODY_TEXT
            self._body_tool = None

    def _close_body(self) -> None:
        if self._body_kind == _BODY_TOOL:
            self.dropped_calls.append(
                {"name": self._body_tool or "", "payload": self._body_buf.strip()}
            )
        self._body_kind = _BODY_TEXT
        self._body_tool = None
        self._body_buf = ""


def strip_harmony_markup(text: str) -> str:
    """One-shot scrub of a complete text: control tokens, channel headers and
    tool-call payloads removed; ``analysis`` bodies removed (chain of thought
    is not part of the answer); ``final`` bodies kept."""
    if not text or "<|" not in text:
        return text
    f = HarmonyStreamFilter()
    visible, _ = f.feed(text)
    tail, _ = f.flush()
    return visible + tail
