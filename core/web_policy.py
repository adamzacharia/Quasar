"""Per-turn "is web search allowed?" flag, visible to every thread of the turn.

The chat runner strips ``web_*`` tools from the model when the user turns web
search off, but Conductor sub-agents build their own tool list on executor
threads and always got the web tools (D1). The runner sets the flag for the
turn thread, the Conductor re-installs it on each sub-agent thread, and the
tool guard refuses a web tool while it is off (defence in depth).

Unset (non-chat entry points, bare test agents) means allowed, which is the
previous behaviour.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

WEB_TOOL_NAMES = frozenset({
    "web_search",
    "web_extract_url",
    "web_map_site",
    "web_crawl_site",
    "web_research",
    "web_research_status",
})

_tls = threading.local()


def is_web_tool(name: Any) -> bool:
    n = str(name or "")
    return n in WEB_TOOL_NAMES or n.startswith("web_")


def set_web_allowed(allowed: Optional[bool]) -> None:
    """Set (True/False) or clear (None) the flag for the CURRENT thread."""
    _tls.allowed = None if allowed is None else bool(allowed)


def web_allowed() -> bool:
    value = getattr(_tls, "allowed", None)
    return True if value is None else bool(value)


def current_setting() -> Optional[bool]:
    """The raw value for this thread (None = never set), for re-installing elsewhere."""
    return getattr(_tls, "allowed", None)


@contextmanager
def web_scope(allowed: Optional[bool], event_sink: Any = None) -> Iterator[None]:
    """Install the web switch (and optionally the turn's event sink) for the
    duration of a block on the CURRENT thread; the previous values return."""
    previous = getattr(_tls, "allowed", None)
    previous_sink = getattr(_tls, "event_sink", None)
    set_web_allowed(allowed)
    if event_sink is not None:
        _tls.event_sink = event_sink
    try:
        yield
    finally:
        _tls.allowed = previous
        _tls.event_sink = previous_sink


def set_event_sink(sink: Any) -> None:
    """The TURN's status/event callback for this thread (None clears it).
    Conductor sub-agents use it for source cards instead of the process-wide
    ``agent._last_on_status``, which a concurrent turn can overwrite
    (web search redesign, guard CX-01)."""
    _tls.event_sink = sink


def event_sink() -> Any:
    return getattr(_tls, "event_sink", None)


def strip_web_tools(tools: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tool definitions without the web tools (Responses-API ``{"name": ...}``
    or chat-completions ``{"function": {"name": ...}}`` shapes)."""
    out = []
    for tool in tools or []:
        name = ""
        if isinstance(tool, dict):
            name = tool.get("name") or (tool.get("function") or {}).get("name") or ""
        if is_web_tool(name):
            continue
        out.append(tool)
    return out
