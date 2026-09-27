"""Per-user MCP connections for the web app.

The FastAPI app runs ONE shared QuasarAgent built without a user id, so the
agent's own per-user loader (QuasarAgent._load_mcp_servers) never ran for web
users: servers saved in Settings > MCP Servers were stored but never reached a
chat (verified live 2026-09-27). Mounting them into the shared ToolRegistry
would be worse, since every user would then see (and call, with the saver's
credentials) everyone else's tools.

This pool keeps each user's connections separate:

* ``tools_for(user_id)`` connects (or reuses) that user's saved servers and
  returns their tools as ``Tool`` objects named ``{server}__{tool}``. The
  runner adds them to that request only.
* ``test(user_id, name)`` reconnects one server and reports what happened, so
  "Save & Connect" can say "connected, 3 tools" or show the real error.
* ``forget(user_id, name)`` drops a connection when a server is edited or
  deleted.

All sessions live on one background event loop. Each connection is held open
by a single task (enter transport -> initialize -> list_tools -> wait for
close -> exit), because the MCP SDK's anyio scopes must be entered and exited
in the same task.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

CONNECT_TIMEOUT_SECONDS = 20.0
# A server that failed is not retried on every message (each retry would add
# up to CONNECT_TIMEOUT_SECONDS to the turn); it is retried after this long,
# or at once when the user presses "Test connection" / saves it again.
FAILED_RETRY_SECONDS = 120.0
IDLE_TTL_SECONDS = 30 * 60

_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def tool_function_name(server: str, tool: str) -> str:
    """Provider-safe function name (OpenAI-style: [A-Za-z0-9_-]{1,64})."""
    return _NAME_RE.sub("_", f"{server}__{tool}")[:64]


def _describe_error(exc: BaseException) -> str:
    """The innermost useful message (the SDK wraps failures in task groups)."""
    seen = 0
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions and seen < 10:
        exc = exc.exceptions[0]
        seen += 1
    text = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__}: {text}"[:300]


class _Conn:
    def __init__(self, cfg: dict, fingerprint: str):
        self.cfg = cfg
        self.fingerprint = fingerprint
        self.session = None
        self.tools: list = []
        self.transport_used: Optional[str] = None
        self.error: Optional[str] = None
        self.ready = threading.Event()
        self.close_evt: Optional[asyncio.Event] = None
        self.failed_at: Optional[float] = None
        self.last_used = time.monotonic()

    @property
    def alive(self) -> bool:
        return self.session is not None and self.error is None


class UserMCPPool:
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._conns: Dict[Tuple[str, str], _Conn] = {}

    # ── event loop ──────────────────────────────────────────────────────
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            loop = asyncio.new_event_loop()
            started = threading.Event()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(started.set)
                loop.run_forever()

            thread = threading.Thread(target=_run, name="user-mcp-pool", daemon=True)
            thread.start()
            started.wait(5)
            self._loop, self._thread = loop, thread
            return loop

    # ── one connection ──────────────────────────────────────────────────
    async def _open(self, stack, cfg: dict):
        from mcp.client.session import ClientSession

        transport = (cfg.get("transport") or "stdio").strip().lower()
        url = (cfg.get("url") or "").strip()
        headers = {str(k): str(v) for k, v in (cfg.get("headers") or {}).items() if k} or None
        if transport == "streamable_http":
            from mcp.client.streamable_http import streamablehttp_client

            read, write, _ = await stack.enter_async_context(streamablehttp_client(url, headers=headers))
            used = "streamable_http"
        elif transport == "http":
            # Legacy SSE transport. Stored "http" configs predate the
            # streamable option and the form labelled it "HTTP/SSE", so
            # users pasted modern /mcp URLs into it; the caller retries those
            # as streamable HTTP when SSE fails.
            from mcp.client.sse import sse_client

            read, write = await stack.enter_async_context(sse_client(url, headers=headers))
            used = "sse"
        else:
            from mcp.client.stdio import StdioServerParameters, stdio_client
            import os

            env = os.environ.copy()
            env.update(cfg.get("env") or {})
            params = StdioServerParameters(command=cfg["command"], args=cfg.get("args") or [], env=env)
            read, write = await stack.enter_async_context(stdio_client(params))
            used = "stdio"
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        return session, list(listed.tools), used

    async def _hold(self, conn: _Conn) -> None:
        from contextlib import AsyncExitStack

        conn.close_evt = asyncio.Event()
        cfgs = [conn.cfg]
        if (conn.cfg.get("transport") or "").lower() == "http":
            # Try modern streamable HTTP first: it answers fast, while a
            # legacy SSE attempt against a modern endpoint only fails at the
            # timeout. Genuine SSE servers still connect on the second try.
            cfgs = [{**conn.cfg, "transport": "streamable_http"}, conn.cfg]
        errors: List[str] = []
        for cfg in cfgs:
            try:
                async with AsyncExitStack() as stack:
                    # asyncio.timeout (not wait_for): the transport's anyio
                    # scopes must be entered and exited in THIS task.
                    async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS):
                        session, tools, used = await self._open(stack, cfg)
                    conn.session, conn.tools, conn.transport_used = session, tools, used
                    conn.error = None
                    conn.ready.set()
                    await conn.close_evt.wait()
                    return
            except asyncio.TimeoutError:
                errors.append(f"timed out after {CONNECT_TIMEOUT_SECONDS:.0f}s")
            except asyncio.CancelledError:
                return
            except BaseException as exc:  # noqa: BLE001 - reported, not raised
                errors.append(_describe_error(exc))
            finally:
                conn.session = None
        conn.error = " | ".join(errors) or "connection failed"
        conn.failed_at = time.monotonic()
        conn.ready.set()

    def _close(self, conn: _Conn) -> None:
        loop = self._loop
        if loop is None or not loop.is_running() or conn.close_evt is None:
            return
        loop.call_soon_threadsafe(conn.close_evt.set)

    # ── public API ──────────────────────────────────────────────────────
    def _configs(self, user_id: str) -> List[dict]:
        from services.mcp_server_service import MCPServerService, mcp_stdio_enabled

        out = []
        for cfg in MCPServerService().load_servers(user_id) or []:
            if not isinstance(cfg, dict) or not cfg.get("name"):
                continue
            transport = (cfg.get("transport") or "stdio").strip().lower()
            if transport not in ("http", "streamable_http") and not mcp_stdio_enabled():
                continue  # S2 RCE gate: never spawn a user-supplied command
            out.append(cfg)
        return out

    def _connect(self, user_id: str, cfg: dict, *, force: bool = False) -> _Conn:
        key = (user_id, cfg["name"])
        fingerprint = json.dumps(cfg, sort_keys=True, default=str)
        with self._lock:
            conn = self._conns.get(key)
            reuse = (
                conn is not None and not force and conn.fingerprint == fingerprint
                and (conn.alive or (conn.failed_at is not None
                                    and time.monotonic() - conn.failed_at < FAILED_RETRY_SECONDS)
                     or not conn.ready.is_set())
            )
            if reuse:
                conn.last_used = time.monotonic()
                return conn
            if conn is not None:
                self._close(conn)
            conn = _Conn(cfg, fingerprint)
            self._conns[key] = conn
        loop = self._ensure_loop()
        asyncio.run_coroutine_threadsafe(self._hold(conn), loop)
        return conn

    def _reap_idle(self) -> None:
        now = time.monotonic()
        with self._lock:
            for key, conn in list(self._conns.items()):
                if now - conn.last_used > IDLE_TTL_SECONDS:
                    self._close(conn)
                    del self._conns[key]

    def tools_for(self, user_id: str) -> List[Any]:
        """Tools from every saved server of this user that is reachable."""
        if not user_id or user_id in ("user", "anonymous"):
            return []
        try:
            configs = self._configs(user_id)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[UserMCP] could not read configs for {user_id}: {exc}")
            return []
        if not configs:
            return []
        self._reap_idle()
        conns = [self._connect(user_id, cfg) for cfg in configs]
        deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS * 2 + 2
        from core.tools import Tool

        tools: List[Any] = []
        for conn in conns:
            conn.ready.wait(max(0.0, deadline - time.monotonic()))
            name = conn.cfg["name"]
            if not conn.alive:
                if conn.error:
                    print(f"[UserMCP] {name} unavailable for {user_id}: {conn.error}")
                continue
            conn.last_used = time.monotonic()
            for t in conn.tools:
                tools.append(Tool(
                    name=tool_function_name(name, t.name),
                    description=(t.description or f"Tool {t.name} from the {name} MCP server")[:1024],
                    function=self._tool_fn(user_id, conn.cfg, t.name),
                    parameters=t.inputSchema or {"type": "object", "properties": {}},
                    category="mcp",
                ))
        if tools:
            print(f"[UserMCP] {len(tools)} tool(s) for {user_id}: {', '.join(x.name for x in tools[:8])}")
        return tools

    def _tool_fn(self, user_id: str, cfg: dict, tool_name: str):
        """Sync callable for one MCP tool. The session is resolved at CALL
        time, not captured: a reconnect (edit, "Test connection", idle reap)
        between building the tool list and the model's call must not leave
        the tool pointing at a closed session."""
        from concurrent.futures import TimeoutError as FutureTimeoutError

        from adapters.mcp.normalize import normalize_mcp_failure, normalize_mcp_result
        from core.agent import _mcp_call_timeout_seconds

        server = cfg["name"]

        def call(**kwargs):
            def fail(msg: str):
                return normalize_mcp_failure(server, tool_name, msg or "MCP call failed", kwargs)

            with self._lock:
                conn = self._conns.get((user_id, server))
            if conn is None or not conn.alive:
                conn = self._connect(user_id, cfg)
                conn.ready.wait(CONNECT_TIMEOUT_SECONDS * 2 + 2)
            if not conn.alive or self._loop is None:
                return fail(f"{server} is not connected: {conn.error or 'connection closed'}")
            conn.last_used = time.monotonic()
            session = conn.session

            async def run():
                res = await session.call_tool(tool_name, arguments=kwargs)
                return normalize_mcp_result(server, tool_name, res, kwargs)

            timeout = _mcp_call_timeout_seconds()
            fut = asyncio.run_coroutine_threadsafe(run(), self._loop)
            try:
                return fut.result(timeout=timeout)
            except FutureTimeoutError:
                fut.cancel()
                return fail(f"MCP tool call timed out after {timeout:.0f} seconds")
            except BaseException as exc:  # noqa: BLE001 - reported to the model
                return fail(_describe_error(exc))

        return call

    def test(self, user_id: str, name: str) -> Dict[str, Any]:
        """Reconnect one saved server now and report the outcome."""
        cfg = next((c for c in self._configs(user_id) if c.get("name") == name), None)
        if cfg is None:
            return {"connected": False, "tools": [], "error": "Server not found, or local (stdio) servers are disabled here."}
        conn = self._connect(user_id, cfg, force=True)
        conn.ready.wait(CONNECT_TIMEOUT_SECONDS * 2 + 2)
        if conn.alive:
            return {"connected": True, "transport": conn.transport_used,
                    "tools": [t.name for t in conn.tools], "error": None}
        return {"connected": False, "tools": [], "error": conn.error or "timed out connecting"}

    def status(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Last known state of this user's connections (no new connects)."""
        with self._lock:
            return {
                name: {"connected": c.alive, "tools": [t.name for t in c.tools] if c.alive else [],
                       "error": c.error, "transport": c.transport_used}
                for (uid, name), c in self._conns.items() if uid == user_id and c.ready.is_set()
            }

    def forget(self, user_id: str, name: Optional[str] = None) -> None:
        with self._lock:
            for key in [k for k in self._conns if k[0] == user_id and (name is None or k[1] == name)]:
                self._close(self._conns.pop(key))


_POOL: Optional[UserMCPPool] = None
_POOL_LOCK = threading.Lock()


def get_user_mcp_pool() -> UserMCPPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = UserMCPPool()
        return _POOL
