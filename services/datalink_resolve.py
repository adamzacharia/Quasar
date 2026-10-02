"""Follow an IVOA DataLink document to the file it describes (its ``#this`` row).

CADC's SIA 2.0 and ObsCore ``access_url`` values are DataLink VOTables, not
FITS files. On the 2026-10-01 MANNA-evals run gpt-oss reported a DataLink URL
or a local ``file://`` path as "the direct FITS URL" (T12, MQ09, MQ16); MANNA's
find_observations_of_target followed DataLink and reported the real file.

Requests go through services.vo_registry._TimeoutHTTPSession, so every hop
(including redirects) passes the SSRF guard, the host breaker and the running
tool's time budget.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Dict, Iterable, List, Optional

PER_LINK_TIMEOUT_S = 8.0
OVERALL_TIMEOUT_S = 15.0
MAX_DATALINK_BYTES = 2_000_000  # a DataLink document is a few KB; anything huge is not one


def looks_like_datalink(url: str, access_format: Optional[str] = None) -> bool:
    text = str(url or "").lower()
    fmt = str(access_format or "").lower()
    return "datalink" in fmt or "datalink" in text or "/caom2ops/" in text


def pick_this_row(rows: Iterable[Dict[str, str]]) -> Optional[str]:
    """The #this file URL from parsed DataLink rows (FITS preferred), else None."""
    candidates: List[Dict[str, str]] = []
    for row in rows:
        semantics = str(row.get("semantics") or "").strip().lower()
        url = str(row.get("access_url") or "").strip()
        if semantics.endswith("#this") and url and not str(row.get("error_message") or "").strip():
            candidates.append(row)
    if not candidates:
        return None
    for row in candidates:
        ctype = str(row.get("content_type") or "").lower()
        if "fits" in ctype or str(row.get("access_url")).lower().split("?")[0].endswith((".fits", ".fits.gz", ".fits.fz")):
            return str(row["access_url"]).strip()
    return str(candidates[0]["access_url"]).strip()


def parse_datalink(content: bytes) -> List[Dict[str, str]]:
    """DataLink VOTable bytes -> list of row dicts (lower-cased column names)."""
    from astropy.io.votable import parse_single_table

    table = parse_single_table(io.BytesIO(content)).to_table()
    names = [c.lower() for c in table.colnames]
    rows: List[Dict[str, str]] = []
    for rec in table:
        row = {}
        for name, value in zip(names, rec):
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            row[name] = "" if value is None or str(value) == "--" else str(value)
        rows.append(row)
    return rows


def resolve_this(url: str, *, timeout: float = PER_LINK_TIMEOUT_S, session=None) -> Optional[str]:
    """The #this file URL behind one DataLink URL, or None on any failure."""
    own_session = session is None
    try:
        if session is None:
            from services.vo_registry import _TimeoutHTTPSession

            session = _TimeoutHTTPSession(timeout=timeout)
        # Every read (connect, headers, each chunk) waits at most timeout/4 and
        # the body is abandoned once timeout/2 has passed; the watchdog below
        # enforces the wall clock even for a read that never returns. The
        # guarded session also clamps each read to the running tool's budget.
        read_timeout = max(timeout / 4.0, 0.05)
        abandon_at = time.monotonic() + timeout / 2.0
        # Hard wall clock, armed BEFORE the request (Codex CX-04 rounds 2-3): a
        # stalled body, slow headers or a redirect chain can each block inside
        # one call, so a daemon timer closes whatever is open at the limit (the
        # response once it exists, the session's pooled connections before),
        # which makes the blocked read return or raise.
        held: Dict[str, object] = {}

        def _expire() -> None:
            for obj in (held.get("resp"), session):
                try:
                    getattr(obj, "close", lambda: None)()
                except Exception:  # noqa: BLE001 - best effort
                    pass

        watchdog = threading.Timer(max(0.05, abandon_at + read_timeout - time.monotonic()), _expire)
        watchdog.daemon = True
        watchdog.start()
        if hasattr(session, "max_redirects"):
            session.max_redirects = min(int(getattr(session, "max_redirects", 3) or 3), 3)
        try:
            resp = session.get(url, timeout=read_timeout, stream=True)
        except Exception:
            watchdog.cancel()
            raise
        held["resp"] = resp
        try:
            if int(getattr(resp, "status_code", 0) or 0) != 200:
                return None
            chunks: List[bytes] = []
            size = 0
            iter_content = getattr(resp, "iter_content", None)
            if iter_content is None:
                body = resp.content
            else:
                for chunk in iter_content(chunk_size=16384):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_DATALINK_BYTES or time.monotonic() > abandon_at:
                        return None
                body = b"".join(chunks)
            return pick_this_row(parse_datalink(body))
        finally:
            watchdog.cancel()
            # Always release the streamed connection, including non-200 and
            # abandoned bodies (Codex CX-11).
            getattr(resp, "close", lambda: None)()
    except Exception:  # noqa: BLE001 - an unresolved link stays a DataLink URL
        return None
    finally:
        if own_session and session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass


def resolve_many(urls: Iterable[str], *, timeout: float = PER_LINK_TIMEOUT_S,
                 overall: float = OVERALL_TIMEOUT_S, session=None) -> Dict[str, str]:
    """{datalink_url: file_url} for the DataLink URLs that resolve within ``overall`` s.

    Sequential, in the CALLING thread: the running tool's deadline and
    cancellation are thread-local (services/tool_budgets), so worker threads
    would outlive them (Codex CX-04). Each request's timeout is the smaller of
    ``timeout`` and what is left of ``overall`` (and the session further clamps
    it to the tool budget)."""
    todo = [u for u in dict.fromkeys(str(u) for u in urls if u) if looks_like_datalink(u)]
    out: Dict[str, str] = {}
    start = time.monotonic()
    for url in todo:
        left = overall - (time.monotonic() - start)
        if left < 1.0:
            break
        value = resolve_this(url, timeout=min(timeout, left), session=session)
        if value:
            out[url] = value
    return out


__all__ = ["looks_like_datalink", "parse_datalink", "pick_this_row", "resolve_many", "resolve_this"]
