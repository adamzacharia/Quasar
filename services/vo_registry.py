"""VO Registry autonomous discovery: find, inspect, and query any VO service.

Four capabilities: (1) search the IVOA registry for services by keyword /
type / waveband; (2) list + describe tables of an arbitrary TAP service;
(3) run guarded SELECT-only ADQL against an arbitrary TAP service; (4) cone
search an arbitrary SCS service. All pyvo imports are lazy; every HTTP call
is timeout-bound via an injected session (pyvo issues requests without
timeouts otherwise).
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT_S = 45.0
DEFAULT_MAXREC_CAP = 1000
MAX_REGISTRY_ROWS = 100
MAX_TABLES = 200
MAX_COLUMNS = 200
MAX_CONE_RADIUS_DEG = 5.0
DEFAULT_AUTO_SYNC_TIMEOUT_S = 20.0
DEFAULT_ASYNC_MAXREC_CAP = 5000
MAX_REMEMBERED_JOBS = 5000
# auto mode promotes a timed-out sync query only if at least this much of the
# tool budget is left: submit = POST + job GET + run POST, and a job created
# upstream whose URL never reaches the model is an orphan.
MIN_PROMOTE_BUDGET_S = 15.0
MAX_STATUS_WAIT_S = 20.0
_ADQL_MODES = ("sync", "async", "auto")
MAX_SIA_RADIUS_DEG = 2.0
MAX_SIA_ROWS = 500
# Named regimes -> SIA2 BAND wavelength interval in metres (overlap match).
WAVEBAND_METERS: Dict[str, Tuple[float, float]] = {
    "radio": (1e-2, 1e4),        # longer than 1 cm
    "mm": (3e-4, 1e-2),          # 0.3 mm - 1 cm (ALMA, NOEMA)
    "infrared": (7e-7, 3e-4),
    "optical": (3.2e-7, 7e-7),
    "uv": (1e-8, 3.2e-7),
    "xray": (1e-12, 1e-8),
}

_SERVICE_TYPE_ALIASES = {
    "tap": "tap", "sia": "sia", "ssa": "ssa",
    "scs": "conesearch", "conesearch": "conesearch",
}


class _TimeoutHTTPSession(requests.Session):
    """requests.Session that enforces a default timeout on every request.

    pyvo issues requests without a timeout, which can hang indefinitely on
    slow or filtered networks (same pattern as integrations/tap.py).
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_S):
        super().__init__()
        self._default_timeout = timeout

    def request(self, method, url, **kwargs):
        # Same contract as integrations/tap.py: timeout clamped to the running
        # tool's remaining budget, transport failures open the host breaker.
        from services.host_breaker import HostBreaker, host_of
        from services.tool_budgets import bounded_timeout

        host = host_of(url)
        timeout = kwargs.get("timeout")
        if timeout is None:
            timeout = self._default_timeout
        kwargs["timeout"] = bounded_timeout(float(timeout), label=f"VO {host}")
        HostBreaker.check(url)  # URL: soft failures are scoped to the service path
        from services.http_budget_hook import suppressed

        try:
            with suppressed():
                response = super().request(method, url, **kwargs)
        except requests.RequestException as exc:
            HostBreaker.record_failure(url, exc)
            raise
        status = int(getattr(response, "status_code", 0) or 0)
        if status in (502, 503, 504):
            HostBreaker.record_failure(url, status=status)
        else:
            HostBreaker.record_success(url)
        return response

    def send(self, request, **kwargs):
        # SSRF guard on EVERY hop: requests routes the first request and each
        # redirect it follows through send(), and pyvo reaches server-supplied
        # UWS job URLs through the same session (services/vo_url_guard.py).
        from services.vo_url_guard import ensure_public_url

        ensure_public_url(request.url, param="VO request URL")
        return super().send(request, **kwargs)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


class _DeadlineExceeded(Exception):
    """A VO call exceeded its total wall-clock budget."""


def _run_with_deadline(fn: Callable[[], Any], seconds: float, label: str) -> Any:
    """Run fn() with a TOTAL wall-clock bound.

    The session timeout is per-socket-READ: a server that trickles bytes
    (VizieR TAP_SCHEMA scans, live-verified 2026-07-05 at >9 min) never trips
    it, which stalls an entire chat turn. The worker thread is abandoned on
    expiry (it dies with its socket); a bounded leak beats an unbounded hang.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _FuturesTimeout

    from services.tool_budgets import adopt_deadline, current_deadline, end_tool_deadline

    # The worker is a fresh thread: hand it a CHILD of the caller's tool
    # deadline (guard CX-09) so its HTTP calls keep the budget clamp, the
    # breaker call identity and the turn cancellation; an abandoned worker's
    # child is cancelled so it cannot start another request.
    parent = current_deadline()
    child = parent.child(label=label) if parent is not None else None
    if parent is not None:
        seconds = min(float(seconds), max(0.0, parent.remaining()))
        if seconds <= 0.05:
            raise _DeadlineExceeded(f"{label}: no time left in the tool budget; not started.")

    def _worker():
        adopt_deadline(child)
        try:
            return fn()
        finally:
            end_tool_deadline()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_worker)
        try:
            return future.result(timeout=seconds)
        except _FuturesTimeout:
            future.cancel()
            if child is not None:
                child.cancel(f"{label}: abandoned after {seconds:g}s")
            raise _DeadlineExceeded(
                f"{label} exceeded {seconds:g}s total; the service is responding too "
                f"slowly — try another service or query it directly with vo_query."
            )
    finally:
        executor.shutdown(wait=False)


class VoRegistryService:
    """Discovery + guarded querying across the whole Virtual Observatory."""

    def __init__(
        self,
        *,
        timeout: Optional[float] = None,
        maxrec_cap: Optional[int] = None,
        regsearch_fn: Optional[Callable[..., Any]] = None,
        tap_factory: Optional[Callable[[str], Any]] = None,
        scs_factory: Optional[Callable[[str], Any]] = None,
        job_factory: Optional[Callable[[str], Any]] = None,
        sia2_factory: Optional[Callable[[str], Any]] = None,
        sia1_factory: Optional[Callable[[str], Any]] = None,
    ):
        self.timeout = float(timeout if timeout is not None else _env_float("VO_REGISTRY_TIMEOUT", DEFAULT_TIMEOUT_S))
        cap_env = os.getenv("VO_ADQL_MAXREC_CAP")
        self.maxrec_cap = int(maxrec_cap if maxrec_cap is not None else (cap_env or DEFAULT_MAXREC_CAP))
        self._regsearch_fn = regsearch_fn
        self._tap_factory = tap_factory
        self._scs_factory = scs_factory
        self._job_factory = job_factory
        self._sia2_factory = sia2_factory
        self._sia1_factory = sia1_factory
        self.auto_sync_timeout = _env_float("VO_AUTO_SYNC_TIMEOUT", DEFAULT_AUTO_SYNC_TIMEOUT_S)
        async_env = os.getenv("VO_ASYNC_MAXREC_CAP")
        self.async_maxrec_cap = int(async_env or DEFAULT_ASYNC_MAXREC_CAP)

    # ── factories (lazy pyvo) ───────────────────────────────────────────────
    def _tap_service(self, access_url: str, timeout: Optional[float] = None):
        if self._tap_factory is not None:
            return self._tap_factory(access_url)
        import pyvo  # lazy

        session = _TimeoutHTTPSession(self.timeout if timeout is None else timeout)
        return pyvo.dal.TAPService(access_url, session=session)

    def _scs_service(self, access_url: str):
        if self._scs_factory is not None:
            return self._scs_factory(access_url)
        import pyvo  # lazy

        return pyvo.dal.SCSService(access_url, session=_TimeoutHTTPSession(self.timeout))

    # ── 1. registry search ──────────────────────────────────────────────────
    def registry_search(self, keywords: Any, service_type: Any = None,
                        waveband: Any = None, max_rows: Any = 30) -> Dict[str, Any]:
        try:
            kw_list = _coerce_keywords(keywords)
            if not kw_list:
                return {"success": False, "error": "At least one search keyword is required."}
            st_norm, st_warnings = _normalize_service_type(service_type)
            rows_cap, warnings = _clamp(max_rows, 30, MAX_REGISTRY_ROWS, "max_rows")
            warnings = st_warnings + warnings

            results = self._run_regsearch(kw_list, st_norm, waveband)

            rows: List[Dict[str, Any]] = []
            seen_urls: set = set()
            skipped_no_url = 0
            for rec in results:
                if len(rows) >= rows_cap:
                    warnings.append(
                        f"Registry returned more than {rows_cap} services; showing first {rows_cap}."
                    )
                    break
                url = _resolve_access_url(rec, st_norm)
                if not url:
                    skipped_no_url += 1
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                wb = getattr(rec, "waveband", None)
                rows.append({
                    "ivoid": _str_or_none(getattr(rec, "ivoid", None)),
                    "short_name": _str_or_none(getattr(rec, "short_name", None)),
                    "title": _trunc(_str_or_none(getattr(rec, "res_title", None)), 120),
                    "service_type": _str_or_none(getattr(rec, "res_type", None)),
                    "access_url": url,
                    "waveband": ",".join(wb) if isinstance(wb, (list, tuple)) else _str_or_none(wb),
                    "description": _trunc(_str_or_none(getattr(rec, "res_description", None)), 200),
                })
            if skipped_no_url:
                warnings.append(f"Skipped {skipped_no_url} registry record(s) with no resolvable access URL.")
            return {
                "success": True, "rows": rows, "count": len(rows), "warnings": warnings,
                "provenance": {"service": "IVOA Registry (pyvo regsearch)",
                               "keywords": kw_list, "service_type": st_norm,
                               "waveband": _str_or_none(waveband)},
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _run_regsearch(self, kw_list: List[str], st_norm: Optional[str], waveband: Any):
        if self._regsearch_fn is not None:
            return self._regsearch_fn(keywords=kw_list, servicetype=st_norm, waveband=waveband)
        from pyvo import registry  # lazy

        constraints = [registry.Freetext(*kw_list)]
        if st_norm:
            # VERIFIED 2026-07-04 (pyvo 1.8.1): plain Servicetype('tap')
            # EXCLUDES VizieR-style catalogs whose TAP is an auxiliary
            # capability — include_auxiliary_services() is required.
            constraints.append(registry.Servicetype(st_norm).include_auxiliary_services())
        if waveband:
            constraints.append(registry.Waveband(str(waveband)))
        return registry.search(*constraints)

    # ── 2. table discovery ──────────────────────────────────────────────────
    def list_tables(self, access_url: Any, keyword: Any = None,
                    max_tables: Any = 50) -> Dict[str, Any]:
        try:
            url = _require_url(access_url)
            cap, warnings = _clamp(max_tables, 50, MAX_TABLES, "max_tables")
            kw = str(keyword or "").strip().lower()

            svc = self._tap_service(url)
            # PREFERRED PATH (verified 2026-07-04): query TAP_SCHEMA.tables —
            # tiny and server-side filterable. pyvo's svc.tables downloads the
            # ENTIRE tableset XML, which for VizieR (~60k tables) aborts the
            # connection mid-transfer.
            try:
                rows, ts_warnings = _run_with_deadline(
                    lambda: self._list_tables_via_tap_schema(svc, kw, cap),
                    self.timeout, f"TAP_SCHEMA listing on {url}",
                )
                warnings.extend(ts_warnings)
                return {"success": True, "rows": rows, "count": len(rows),
                        "warnings": warnings,
                        "provenance": {"service": f"TAP_SCHEMA: {url}", "keyword": kw or None}}
            except _DeadlineExceeded as slow_err:
                # The fallback downloads the FULL tableset — strictly slower
                # than the query that just stalled. Fail fast and honestly so
                # an agent turn moves on instead of dying on its watchdog.
                return {"success": False, "error": str(slow_err), "warnings": warnings}
            except Exception as ts_err:
                warnings.append(f"TAP_SCHEMA listing unavailable ({_trunc(str(ts_err), 80)}); "
                                "falling back to the full tableset (slow on large services).")

            def _scan_tableset():
                found: List[Dict[str, Any]] = []
                seen = 0
                for name, table in _iter_tables(svc):
                    seen += 1
                    desc = _str_or_none(getattr(table, "description", None)) or ""
                    if kw and kw not in name.lower() and kw not in desc.lower():
                        continue
                    if len(found) >= cap:
                        continue  # keep counting matches for the warning
                    found.append({
                        "table_name": name,
                        "description": _trunc(desc, 150) or None,
                        "n_columns": _column_count(table),
                    })
                return found, seen

            try:
                rows, total_seen = _run_with_deadline(
                    _scan_tableset, self.timeout, f"Tableset download from {url}")
            except _DeadlineExceeded as slow_err:
                return {"success": False, "error": str(slow_err), "warnings": warnings}
            if total_seen > len(rows) and len(rows) >= cap:
                warnings.append(
                    f"Service exposes many tables ({total_seen} scanned); showing {cap}. "
                    "Pass a keyword to narrow the listing."
                )
            return {"success": True, "rows": rows, "count": len(rows),
                    "warnings": warnings,
                    "provenance": {"service": f"TAP tables: {url}", "keyword": kw or None}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _list_tables_via_tap_schema(self, svc: Any, kw: str, cap: int
                                    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        safe_kw = kw.replace("'", "''")
        where = (f" WHERE LOWER(table_name) LIKE '%{safe_kw}%' "
                 f"OR LOWER(description) LIKE '%{safe_kw}%'") if kw else ""
        adql = (f"SELECT TOP {cap + 1} table_name, description "
                f"FROM TAP_SCHEMA.tables{where}")
        try:
            result = svc.run_sync(adql, maxrec=cap + 1)
        except Exception as lower_err:
            # LOWER() is optional in ADQL 2.0 and TAPVizieR's parser rejects it
            # outright ('Encountered "("...', live 2026-07-18, scan-L7) — which
            # used to dump every VizieR listing onto the tableset-download
            # fallback and die in pyvo's VOSI parse. Retry the same query
            # case-SENSITIVE and say so, rather than losing TAP_SCHEMA at all.
            if not kw or "(" not in str(lower_err):
                raise
            where_cs = (f" WHERE table_name LIKE '%{safe_kw}%' "
                        f"OR description LIKE '%{safe_kw}%'")
            adql = (f"SELECT TOP {cap + 1} table_name, description "
                    f"FROM TAP_SCHEMA.tables{where_cs}")
            result = svc.run_sync(adql, maxrec=cap + 1)
            warnings.append(
                "This service rejected LOWER() in ADQL, so the keyword match is "
                "case-sensitive here — try exact-case variants if a table seems missing."
            )
        raw_rows, _ = _normalize_dal_result(result)
        if not raw_rows and not kw:
            raise ValueError("TAP_SCHEMA.tables returned no rows")
        rows = [{"table_name": _str_or_none(r.get("table_name")),
                 "description": _trunc(_str_or_none(r.get("description")), 150),
                 "n_columns": None}
                for r in raw_rows[:cap] if _str_or_none(r.get("table_name"))]
        if len(raw_rows) > cap:
            warnings.append(f"More than {cap} tables match; showing {cap}. "
                            "Pass a keyword (or a narrower one) to refine.")
        return rows, warnings

    def describe_table(self, access_url: Any, table_name: Any) -> Dict[str, Any]:
        try:
            url = _require_url(access_url)
            wanted = str(table_name or "").strip()
            if not wanted:
                return {"success": False, "error": "table_name is required."}

            svc = self._tap_service(url)
            # PREFERRED PATH: TAP_SCHEMA.columns (same rationale as
            # list_tables — svc.tables downloads everything on VizieR).
            try:
                safe = wanted.replace("'", "''")
                adql = (f"SELECT TOP {MAX_COLUMNS + 1} column_name, datatype, unit, ucd, "
                        f"description FROM TAP_SCHEMA.columns "
                        f"WHERE LOWER(table_name) = LOWER('{safe}')")
                try:
                    result = _run_with_deadline(
                        lambda: svc.run_sync(adql, maxrec=MAX_COLUMNS + 1),
                        self.timeout, f"TAP_SCHEMA columns on {url}",
                    )
                except _DeadlineExceeded:
                    raise
                except Exception as lower_err:
                    # TAPVizieR rejects LOWER() (scan-L7) — retry exact-case
                    # rather than falling into the full-tableset download.
                    if "(" not in str(lower_err):
                        raise
                    adql_cs = (f"SELECT TOP {MAX_COLUMNS + 1} column_name, datatype, unit, "
                               f"ucd, description FROM TAP_SCHEMA.columns "
                               f"WHERE table_name = '{safe}'")
                    result = _run_with_deadline(
                        lambda: svc.run_sync(adql_cs, maxrec=MAX_COLUMNS + 1),
                        self.timeout, f"TAP_SCHEMA columns on {url}",
                    )
                raw_rows, _ = _normalize_dal_result(result)
                if raw_rows:
                    warnings: List[str] = []
                    if len(raw_rows) > MAX_COLUMNS:
                        raw_rows = raw_rows[:MAX_COLUMNS]
                        warnings.append(f"Table has more than {MAX_COLUMNS} columns; list truncated.")
                    rows = [{
                        "name": _str_or_none(r.get("column_name")),
                        "datatype": _str_or_none(r.get("datatype")),
                        "unit": _str_or_none(r.get("unit")),
                        "ucd": _str_or_none(r.get("ucd")),
                        "description": _trunc(_str_or_none(r.get("description")), 80),
                    } for r in raw_rows]
                    return {"success": True, "rows": rows, "count": len(rows),
                            "warnings": warnings,
                            "provenance": {"service": f"TAP_SCHEMA: {url}", "table": wanted}}
            except Exception:
                pass  # fall back to the tableset walk below

            def _find_in_tableset():
                found = None
                for name, table in _iter_tables(svc):
                    if name == wanted:
                        return table
                    if found is None and name.lower() == wanted.lower():
                        found = table
                return found

            try:
                # Same trickle-hazard as list_tables: the tableset download can
                # stall for minutes on big services — bound it identically.
                match = _run_with_deadline(
                    _find_in_tableset, self.timeout, f"Tableset download from {url}")
            except _DeadlineExceeded as slow_err:
                return {"success": False, "error": str(slow_err)}
            if match is None:
                return {"success": False,
                        "error": f"Table {wanted!r} not found on {url} (try vo_list_tables with a keyword)."}
            warnings: List[str] = []
            rows = []
            for col in list(getattr(match, "columns", []) or []):
                if len(rows) >= MAX_COLUMNS:
                    warnings.append(f"Table has more than {MAX_COLUMNS} columns; list truncated.")
                    break
                rows.append({
                    "name": _str_or_none(getattr(col, "name", None)),
                    "datatype": _datatype_str(col),
                    "unit": _str_or_none(getattr(col, "unit", None)),
                    "ucd": _str_or_none(getattr(col, "ucd", None)),
                    "description": _trunc(_str_or_none(getattr(col, "description", None)), 80),
                })
            return {"success": True, "rows": rows, "count": len(rows),
                    "warnings": warnings,
                    "provenance": {"service": f"TAP schema: {url}", "table": wanted}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── 3. guarded ADQL ─────────────────────────────────────────────────────
    def _validate_select(self, query: str) -> Optional[str]:
        """Model-facing error for a non-SELECT / multi-statement query, else None."""
        stripped = _strip_adql_comments(query).strip()
        if not stripped.upper().startswith("SELECT"):
            return "Only SELECT queries are allowed on remote TAP services."
        # guard review P2 (rollout-guard): reject multi-statement payloads
        # ("SELECT 1; DELETE ...") — anything after a top-level semicolon.
        if _has_trailing_statement(stripped):
            return ("Only a single SELECT statement is allowed; remove "
                    "everything after the semicolon.")
        return None

    def run_adql(self, access_url: Any, adql: Any, max_rows: Any = 200,
                 mode: Any = "sync", owner: Optional[str] = None) -> Dict[str, Any]:
        """Guarded ADQL. ``mode``: ``sync`` (default), ``async`` (submit a UWS
        job and return its handle at once) or ``auto`` (submit a UWS job, wait
        a bounded time for it, and return the rows if it finished, else the
        job handle). ``auto`` never runs the query twice: an earlier design
        (sync first, resubmit async on timeout) could not stop the abandoned
        sync read, so both copies could run at once (guard CX-07)."""
        url = str(access_url or "")
        query = str(adql or "")
        mode_norm = str(mode or "sync").strip().lower()
        if mode_norm not in _ADQL_MODES:
            return {"success": False,
                    "error": f"mode must be one of {', '.join(_ADQL_MODES)}; got {mode!r}."}
        if mode_norm == "async":
            return self.submit_async(access_url, adql, max_rows=max_rows, owner=owner)
        if mode_norm == "auto":
            return self._run_auto(access_url, adql, max_rows=max_rows, owner=owner)
        try:
            url = _require_url(access_url)
            problem = self._validate_select(query)
            if problem:
                return {"success": False, "error": problem}
            maxrec, warnings = _clamp(max_rows, 200, self.maxrec_cap, "max_rows")
            svc = self._tap_service(url)
            try:
                result = svc.run_sync(query, maxrec=maxrec)
            except Exception as exc:
                if not _is_timeout(exc):
                    raise
                return {"success": False,
                        "error": (f"{exc} (the query timed out; retry with mode='auto' or "
                                  "mode='async' for slow or heavy queries)"),
                        "provenance": {"service": "vo_tap", "endpoint": url, "query": query}}
            rows, columns = _normalize_dal_result(result)
            truncated = len(rows) >= maxrec
            if truncated:
                warnings.append(f"Result hit the row cap ({maxrec}); it may be truncated.")
            return {"success": True, "rows": rows, "count": len(rows),
                    "columns": columns, "truncated": truncated, "warnings": warnings,
                    # `query` is the FULL executed ADQL (CX-04): the provenance
                    # request surface reads it and renders kind:"adql"; the
                    # truncated `adql` stays for the model-facing summary.
                    "provenance": {"service": "vo_tap", "endpoint": url,
                                   "query": query, "adql": _trunc(query, 500),
                                   "maxrec": maxrec}}
        except Exception as exc:
            # DALQueryError text contains the server's ADQL complaint — that is
            # exactly what the model needs to fix its query. Preserve it.
            return {"success": False, "error": str(exc),
                    "provenance": {"service": "vo_tap", "endpoint": url,
                                   "query": query}}

    def _run_auto(self, access_url: Any, adql: Any, max_rows: Any = 200,
                  owner: Optional[str] = None) -> Dict[str, Any]:
        left = _budget_remaining()
        if left is not None and left < MIN_PROMOTE_BUDGET_S:
            # Too little budget to submit, poll and fetch a job safely: run the
            # query synchronously instead (each request is clamped to what is
            # left), so a fast query still answers (guard CX-18).
            out = self.run_adql(access_url, adql, max_rows=max_rows, mode="sync", owner=owner)
            out.setdefault("warnings", []).insert(0, (
                f"mode='auto' ran synchronously: only {left:.0f}s of the tool budget was left, "
                "too little for an async job. Use mode='async' in a fresh call for a slow query."))
            return out
        cap, _ = _clamp(max_rows, 200, self.maxrec_cap, "max_rows")
        submitted = self.submit_async(access_url, adql, max_rows=cap, owner=owner)
        if not submitted.get("success"):
            return submitted
        job_url = submitted["job_url"]
        waited = 0.0
        wait_budget = self.auto_sync_timeout
        status: Dict[str, Any] = submitted
        while waited < wait_budget:
            remaining = _budget_remaining()
            if remaining is not None and remaining < 10.0:
                break
            step = min(MAX_STATUS_WAIT_S, wait_budget - waited)
            if remaining is not None:
                step = min(step, max(0.0, remaining - 10.0))
            if step <= 0:
                break
            t0 = time.monotonic()
            status = self.job_status(job_url, wait_seconds=step, owner=owner)
            waited += max(time.monotonic() - t0, 0.5)
            if not status.get("success") or status.get("phase") not in ("PENDING", "QUEUED", "EXECUTING"):
                break
        if not status.get("success"):
            return {"success": False, "status": "job_submitted_status_unknown", "job_url": job_url,
                    "error": ("The async job was submitted, but checking its status failed: "
                              f"{status.get('error') or 'unknown error'}. Check it with "
                              "vo_tap_job(job_url, action='status') before relying on it."),
                    "provenance": submitted.get("provenance")}
        phase = status.get("phase")
        if phase == "COMPLETED":
            out = self.job_results(job_url, max_rows=cap, owner=owner)
            if out.get("success"):
                out.setdefault("warnings", []).insert(
                    0, "Ran as an async TAP job (mode='auto') and finished within the wait.")
            return out
        if phase == "ERROR":
            return {"success": False, "job_url": job_url, "phase": phase,
                    "error": "The query failed on the server: " + str(status.get("job_error") or "no error summary"),
                    "provenance": status.get("provenance") or submitted.get("provenance")}
        handle = dict(submitted)
        handle["phase"] = phase or submitted.get("phase")
        handle.setdefault("warnings", []).insert(0, (
            f"The job was still {handle['phase']} after waiting {waited:.0f}s; it keeps running on "
            "the server. Poll it with vo_tap_job(job_url, action='status', wait_seconds=20)."))
        return handle

    # ── 3b. async TAP (UWS jobs) ────────────────────────────────────────────
    # Idea from MANNA's async TAP tools (NSF-Simons CosmicAI, MIT). Quasar's
    # version fetches the finished rows itself (table card + provenance)
    # instead of handing the model a result link.
    def _load_job(self, job_url: str):
        if self._job_factory is not None:
            return self._job_factory(job_url)
        from pyvo.dal import AsyncTAPJob  # lazy

        # delete=False: never let pyvo clean up a job the user may still poll.
        return AsyncTAPJob(job_url, session=_TimeoutHTTPSession(self.timeout), delete=False)

    def submit_async(self, access_url: Any, adql: Any, max_rows: Any = None,
                     owner: Optional[str] = None) -> Dict[str, Any]:
        url = str(access_url or "")
        query = str(adql or "")
        try:
            url = _require_url(access_url)
            problem = self._validate_select(query)
            if problem:
                return {"success": False, "error": problem}
            requested = max_rows if max_rows is not None else self.async_maxrec_cap
            maxrec, warnings = _clamp(requested, self.async_maxrec_cap,
                                      self.async_maxrec_cap, "max_rows")
            svc = self._tap_service(url)
            job = svc.submit_job(query, maxrec=maxrec)
            # pyvo deletes a job on context exit unless told not to.
            if hasattr(job, "_delete_on_exit"):
                job._delete_on_exit = False
            job_url = str(job.url)
            # Record BEFORE run(): a job created upstream whose run request
            # fails must stay addressable for status/abort (guard CX-06).
            _remember_job(job_url, endpoint=url, query=query, maxrec=maxrec, owner=owner)
            try:
                job.run()
            except Exception as run_exc:
                # The phase POST may have reached the server before failing
                # (e.g. a response timeout), so the start state is UNKNOWN (guard CX-17).
                return {"success": False, "status": "job_created_start_unknown", "job_url": job_url,
                        "error": (f"The async job was created, but the request to start it failed "
                                  f"({run_exc}), so it may or may not be running. Check it with "
                                  "vo_tap_job(job_url, action='status') and abort it if unwanted."),
                        "provenance": {"service": "vo_tap_async", "endpoint": url, "query": query,
                                       "job_url": job_url}}
            return {"success": True, "status": "job_submitted", "job_url": job_url,
                    "phase": _job_phase(job),
                    "maxrec": maxrec, "warnings": warnings,
                    "message": ("Async TAP job submitted. Poll it with "
                                "vo_tap_job(job_url, action='status'), then fetch rows "
                                "with action='results' once the phase is COMPLETED."),
                    "provenance": {"service": "vo_tap_async", "endpoint": url,
                                   "query": query, "adql": _trunc(query, 500),
                                   "maxrec": maxrec, "job_url": job_url}}
        except Exception as exc:
            return {"success": False, "error": str(exc),
                    "provenance": {"service": "vo_tap_async", "endpoint": url,
                                   "query": query}}

    def job_status(self, job_url: Any, wait_seconds: Any = 0,
                   owner: Optional[str] = None) -> Dict[str, Any]:
        url = str(job_url or "")
        try:
            url = _require_url(job_url, param="job_url")
            job = self._load_job(url)
            phase = _job_phase(job)
            try:
                wait = min(max(float(wait_seconds or 0), 0.0), MAX_STATUS_WAIT_S)
            except (TypeError, ValueError):
                wait = 0.0
            left = _budget_remaining()
            if left is not None:
                wait = min(wait, max(0.0, left - 5.0))
            if wait > 0 and phase in ("PENDING", "QUEUED", "EXECUTING"):
                # UWS blocking poll (WAIT=n): returns early on a phase change.
                updater = getattr(job, "_update", None)
                if callable(updater):
                    try:
                        updater(wait_for_statechange=True, timeout=wait)
                    except Exception:
                        pass  # a failed long-poll leaves the phase we already have
                    phase = _job_phase(job)
            out: Dict[str, Any] = {"success": True, "job_url": url, "phase": phase,
                                   "provenance": _job_provenance(url, owner=owner)}
            if phase == "ERROR":
                out["job_error"] = (_job_error_message(job)
                                    or "The job failed without an error summary.")
            elif phase == "COMPLETED":
                out["message"] = ("Job finished; fetch rows with "
                                  "vo_tap_job(job_url, action='results').")
            elif phase in ("PENDING", "QUEUED", "EXECUTING"):
                out["message"] = "Job still running; check again shortly."
            return out
        except Exception as exc:
            return {"success": False, "error": _job_gone_or(exc), "job_url": url,
                    "provenance": _job_provenance(url, owner=owner)}

    def job_results(self, job_url: Any, max_rows: Any = 200,
                    owner: Optional[str] = None) -> Dict[str, Any]:
        url = str(job_url or "").strip()
        # Results download the whole VOTable before capping, so only jobs
        # submitted here (whose MAXREC Quasar set) may be fetched.
        refusal = _ownership_refusal(url, owner, "fetch results for")
        if refusal:
            return refusal
        try:
            url = _require_url(job_url, param="job_url")
            job = self._load_job(url)
            phase = _job_phase(job)
            if phase != "COMPLETED":
                out: Dict[str, Any] = {"success": False, "job_url": url, "phase": phase,
                                       "provenance": _job_provenance(url)}
                if phase == "ERROR":
                    out["error"] = ("The async job failed: "
                                    + (_job_error_message(job) or "no error summary given."))
                else:
                    out["error"] = (f"The job is {phase}, not COMPLETED; poll "
                                    "action='status' until it finishes.")
                return out
            cap, warnings = _clamp(max_rows, 200, self.async_maxrec_cap, "max_rows")
            result = job.fetch_result()
            rows, columns = _normalize_dal_result(result)
            total = len(rows)
            truncated = total > cap
            if truncated:
                rows = rows[:cap]
                warnings.append(f"The job returned {total} rows; showing the first {cap}.")
            known = _SUBMITTED_JOBS.get(url) or {}
            if known.get("maxrec") and total >= int(known["maxrec"]):
                truncated = True
                warnings.append(f"The job hit its row cap ({known['maxrec']}); the result "
                                "may be incomplete.")
            return {"success": True, "job_url": url, "phase": phase, "rows": rows,
                    "count": len(rows), "total_rows": total, "columns": columns,
                    "truncated": truncated, "warnings": warnings,
                    "provenance": _job_provenance(url)}
        except Exception as exc:
            return {"success": False, "error": _job_gone_or(exc), "job_url": url,
                    "provenance": _job_provenance(url)}

    def job_abort(self, job_url: Any, owner: Optional[str] = None) -> Dict[str, Any]:
        url = str(job_url or "").strip()
        # Abort POSTs to the job. Only jobs this user submitted through Quasar
        # may be aborted, so an injected instruction cannot aim a write at an
        # arbitrary URL or at another user's job.
        refusal = _ownership_refusal(url, owner, "abort")
        if refusal:
            return refusal
        try:
            url = _require_url(url, param="job_url")
            job = self._load_job(url)
            # UWS PHASE=ABORT (not DELETE): the job record stays inspectable.
            job.abort()
        except Exception as exc:
            # Aborting a job the archive already removed is not an error.
            if not _is_job_gone(exc):
                return {"success": False, "error": str(exc), "job_url": url,
                        "provenance": _job_provenance(url)}
        return {"success": True, "job_url": url, "phase": "ABORTED",
                "message": "Abort requested (or the job was already gone).",
                "provenance": _job_provenance(url)}

    # ── 4. cone search ──────────────────────────────────────────────────────
    def cone_search(self, access_url: Any, ra: Any, dec: Any,
                    radius_deg: Any = 0.1, max_rows: Any = 100) -> Dict[str, Any]:
        try:
            url = _require_url(access_url)
            ra_f = float(ra)
            dec_f = float(dec)
            if not (0.0 <= ra_f < 360.0) or not (-90.0 <= dec_f <= 90.0):
                return {"success": False, "error": f"Position ({ra}, {dec}) out of range."}
            warnings: List[str] = []
            radius = float(radius_deg)
            if not math.isfinite(radius) or radius <= 0:
                radius = 0.1
                warnings.append("radius_deg was non-positive or non-finite; using 0.1 deg.")
            elif radius > MAX_CONE_RADIUS_DEG:
                radius = MAX_CONE_RADIUS_DEG
                warnings.append(f"radius_deg clamped to {MAX_CONE_RADIUS_DEG:g} deg.")
            cap, cap_warn = _clamp(max_rows, 100, self.maxrec_cap, "max_rows")
            warnings.extend(cap_warn)

            svc = self._scs_service(url)
            # guard review P2: push the row cap to the service when supported
            # (SCS has no standard MAXREC, so fall back to local slicing).
            try:
                result = svc.search(pos=(ra_f, dec_f), radius=radius, maxrec=cap)
            except TypeError:
                result = svc.search(pos=(ra_f, dec_f), radius=radius)
            rows, columns = _normalize_dal_result(result)
            if len(rows) > cap:
                rows = rows[:cap]
                warnings.append(f"Cone search returned more rows; showing first {cap}.")
            return {"success": True, "rows": rows, "count": len(rows),
                    "columns": columns, "warnings": warnings,
                    "provenance": {"service": f"SCS: {url}", "ra": ra_f, "dec": dec_f,
                                   "radius_deg": radius}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}


    # ── 5. image search (SIA 2.0, SIA 1.0 fallback) ─────────────────────────
    # Idea from MANNA's search_images_by_position (NSF-Simons CosmicAI, MIT).
    # Differences: every request (capabilities probe included) goes through
    # the guarded, budgeted session; wavebands are named regimes mapped to
    # wavelength intervals; DataLink indirection is flagged in warnings.
    def _sia2_service(self, access_url: str):
        if self._sia2_factory is not None:
            return self._sia2_factory(access_url)
        from pyvo.dal.sia2 import SIA2Service  # lazy

        return SIA2Service(access_url, session=_TimeoutHTTPSession(self.timeout))

    def _sia1_service(self, access_url: str):
        if self._sia1_factory is not None:
            return self._sia1_factory(access_url)
        from pyvo.dal.sia import SIAService  # lazy

        return SIAService(access_url, session=_TimeoutHTTPSession(self.timeout))

    def image_search(self, access_url: Any, ra: Any, dec: Any, radius_deg: Any = 0.05,
                     waveband: Any = None, calib_level: Any = None,
                     dataproduct_type: Any = None, collection: Any = None,
                     max_rows: Any = 100, version: Any = "auto") -> Dict[str, Any]:
        url = str(access_url or "")
        try:
            url = _require_url(access_url)
            ra_f, dec_f = float(ra), float(dec)
            if not (0.0 <= ra_f < 360.0) or not (-90.0 <= dec_f <= 90.0):
                return {"success": False, "error": "ra must be in [0, 360) and dec in [-90, 90]."}
            warnings: List[str] = []
            try:
                radius = float(radius_deg if radius_deg is not None else 0.05)
            except (TypeError, ValueError):
                radius = 0.05
                warnings.append("radius_deg was not numeric; using 0.05.")
            if not math.isfinite(radius) or radius <= 0:
                radius = 0.05
                warnings.append("radius_deg must be positive; using 0.05.")
            if radius > MAX_SIA_RADIUS_DEG:
                warnings.append(f"radius_deg {radius:g} exceeds {MAX_SIA_RADIUS_DEG:g}; clamped.")
                radius = MAX_SIA_RADIUS_DEG
            cap, cap_warn = _clamp(max_rows, 100, MAX_SIA_ROWS, "max_rows")
            warnings.extend(cap_warn)
            band = None
            wb = str(waveband or "").strip().lower()
            if wb:
                if wb not in WAVEBAND_METERS:
                    return {"success": False,
                            "error": (f"Unknown waveband {waveband!r}; use one of "
                                      f"{', '.join(WAVEBAND_METERS)}.")}
                band = WAVEBAND_METERS[wb]
            ver = str(version or "auto").strip().lower()
            if ver not in ("auto", "2", "1"):
                return {"success": False, "error": "version must be 'auto', '2' or '1'."}

            protocol = "SIA2"
            table = None
            if ver in ("auto", "2"):
                try:
                    svc = self._sia2_service(url)
                except Exception as exc:
                    if ver == "2":
                        raise
                    warnings.append("The service did not answer as SIA 2.0; retried as SIA 1.0 "
                                    f"({_trunc(str(exc), 120)}).")
                    svc = None
                if svc is not None:
                    kwargs: Dict[str, Any] = {"pos": (ra_f, dec_f, radius), "maxrec": cap}
                    if band is not None:
                        kwargs["band"] = band
                    if calib_level is not None and str(calib_level).strip() != "":
                        kwargs["calib_level"] = int(calib_level)
                    if dataproduct_type:
                        kwargs["data_type"] = str(dataproduct_type)
                    if collection:
                        kwargs["collection"] = str(collection)
                    table = svc.search(**kwargs)
            if table is None:
                protocol = "SIA1"
                ignored = [n for n, v in (("waveband", band), ("calib_level", calib_level),
                                          ("dataproduct_type", dataproduct_type),
                                          ("collection", collection)) if v not in (None, "")]
                if ignored:
                    warnings.append("SIA 1.0 has no " + ", ".join(ignored)
                                    + " filter; those constraints were not applied.")
                table = self._sia1_service(url).search(pos=(ra_f, dec_f), size=2.0 * radius)

            rows, columns = _normalize_dal_result(table)
            total = len(rows)
            if total > cap:
                rows = rows[:cap]
                warnings.append(f"The service returned {total} images; showing the first {cap}.")
            elif total == cap:
                warnings.append(f"The result reached max_rows ({cap}); more images may exist.")
            if any("datalink" in str(r.get("access_format") or "").lower()
                   or "/datalink" in str(r.get("access_url") or "").lower() for r in rows):
                warnings.append("Some access_url values are DataLink documents (a VOTable "
                                "listing the files), not the image itself: open the DataLink "
                                "and pick the #this row to reach the FITS file.")
            return {"success": True, "rows": rows, "count": len(rows), "columns": columns,
                    "protocol": protocol, "truncated": total > cap or total >= cap,
                    "warnings": warnings,
                    "provenance": {"service": f"{protocol}: {url}", "endpoint": url,
                                   "ra": ra_f, "dec": dec_f, "radius_deg": radius,
                                   "waveband": wb or None, "maxrec": cap}}
        except Exception as exc:
            return {"success": False, "error": str(exc),
                    "provenance": {"service": "SIA", "endpoint": url}}


# ── helpers ─────────────────────────────────────────────────────────────────
def _coerce_keywords(keywords: Any) -> List[str]:
    if keywords is None:
        return []
    if isinstance(keywords, str):
        parts = [p.strip() for p in re.split(r"[,\s]+", keywords)]
        return [p for p in parts if p]
    try:
        return [str(k).strip() for k in keywords if str(k).strip()]
    except TypeError:
        return [str(keywords).strip()]


def _normalize_service_type(service_type: Any) -> Tuple[Optional[str], List[str]]:
    raw = str(service_type or "").strip().lower()
    if not raw:
        return None, []
    if raw in _SERVICE_TYPE_ALIASES:
        return _SERVICE_TYPE_ALIASES[raw], []
    return None, [f"Unknown service_type {service_type!r}; searching all service types."]


def _resolve_access_url(rec: Any, st_norm: Optional[str]) -> Optional[str]:
    # Preferred (verified pyvo 1.8.1): get_service resolves aux capabilities.
    try:
        svc = rec.get_service(st_norm or "tap", lax=True)
        base = getattr(svc, "baseurl", None) or getattr(svc, "access_url", None)
        if base:
            return str(base)
    except Exception:
        pass
    try:
        url = getattr(rec, "access_url", None)
        if url:
            return str(url)
    except Exception:
        pass
    try:
        for iface in rec.list_interfaces() or []:
            url = getattr(iface, "access_url", None)
            if url and "paramhttp" in str(getattr(iface, "type", "")).lower():
                return str(url)
        for iface in rec.list_interfaces() or []:
            url = getattr(iface, "access_url", None)
            if url:
                return str(url)
    except Exception:
        pass
    return None


def _iter_tables(svc: Any):
    tables = getattr(svc, "tables", None)
    if tables is None:
        return
    try:
        items = tables.items()
    except AttributeError:
        items = ((getattr(t, "name", str(i)), t) for i, t in enumerate(tables))
    for name, table in items:
        yield str(name), table


def _column_count(table: Any) -> Optional[int]:
    try:
        return len(list(table.columns))
    except Exception:
        return None  # some services lazy-load column metadata per table


def _datatype_str(col: Any) -> Optional[str]:
    dt = getattr(col, "datatype", None)
    if dt is None:
        return None
    content = getattr(dt, "content", None)
    return _str_or_none(content if content is not None else dt)


def _strip_adql_comments(query: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    no_line = re.sub(r"--[^\n]*", " ", no_block)
    return no_line


def _has_trailing_statement(stripped_query: str) -> bool:
    """True when a semicolon OUTSIDE quotes is followed by non-whitespace."""
    in_single = in_double = False
    for i, ch in enumerate(stripped_query):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ";" and not in_single and not in_double:
            if stripped_query[i + 1:].strip():
                return True
    return False


def _normalize_dal_result(result: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    table = result.to_table() if hasattr(result, "to_table") else result
    columns = [str(c) for c in getattr(table, "colnames", [])]
    rows: List[Dict[str, Any]] = []
    for row in table:
        out: Dict[str, Any] = {}
        for col in columns:
            out[col] = _json_safe(row[col])
        rows.append(out)
    return rows, columns


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and value != value:  # NaN
            return None
        return value
    # numpy masked / masked constants
    if getattr(value, "mask", None) is True or str(value) in ("--", "masked"):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def _clamp(value: Any, default: int, cap: int, label: str) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
        warnings.append(f"{label} was not numeric; using {default}.")
    if out <= 0:
        out = default
        warnings.append(f"{label} must be positive; using {default}.")
    elif out > cap:
        warnings.append(f"{label} {out} exceeds {cap}; clamped.")
        out = cap
    return out, warnings


def _require_url(access_url: Any, param: str = "access_url") -> str:
    url = str(access_url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"{param} must be an http(s) URL, got {access_url!r}.")
    # Cheap early rejection (no DNS) of local names, private IP literals and
    # allowlist misses; the resolving check runs per request in send().
    from services.vo_url_guard import ensure_public_url

    return ensure_public_url(url, param=param, resolve=False)


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _trunc(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    s = re.sub(r"\s+", " ", value).strip()
    if len(s) <= limit:
        return s or None
    return s[: limit - 1] + "…"


__all__ = ["VoRegistryService"]


# ── async job helpers ───────────────────────────────────────────────────────
class _JobBook:
    """Bounded, thread-safe record of async jobs this process submitted.
    Gates ``job_abort`` and remembers each job's row cap and query."""

    def __init__(self, limit: int = MAX_REMEMBERED_JOBS):
        self._limit = limit
        self._jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, job_url: str, info: Dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_url] = info
            self._jobs.move_to_end(job_url)
            while len(self._jobs) > self._limit:
                self._jobs.popitem(last=False)

    def get(self, job_url: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_url)

    def pop(self, job_url: str, default: Any = None) -> Any:
        with self._lock:
            return self._jobs.pop(job_url, default)

    def __contains__(self, job_url: object) -> bool:
        with self._lock:
            return job_url in self._jobs

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()


_SUBMITTED_JOBS = _JobBook()


def _remember_job(job_url: str, *, endpoint: str, query: str, maxrec: int,
                  owner: Optional[str] = None) -> None:
    _SUBMITTED_JOBS.add(job_url, {"endpoint": endpoint, "query": query, "maxrec": maxrec,
                                  "owner": owner or ""})


def _ownership_refusal(job_url: str, owner: Optional[str], verb: str) -> Optional[Dict[str, Any]]:
    """A failure dict unless ``job_url`` was submitted here by ``owner``."""
    known = _SUBMITTED_JOBS.get(job_url)
    if known is None:
        return {"success": False, "job_url": job_url,
                "error": (f"Quasar can only {verb} async jobs it submitted in this server "
                          "session; this job_url was not submitted here (or has aged out of "
                          "the job record). Resubmit the query with vo_adql_query(mode='async').")}
    if (known.get("owner") or "") != (owner or ""):
        return {"success": False, "job_url": job_url,
                "error": f"This async job belongs to another user; Quasar will not {verb} it."}
    return None


def _budget_remaining() -> Optional[float]:
    """Seconds left in the running tool's deadline, or None outside a tool."""
    from services.tool_budgets import current_deadline

    deadline = current_deadline()
    return None if deadline is None else float(deadline.remaining())


def _job_provenance(job_url: str, owner: Optional[str] = "__owner_checked__") -> Dict[str, Any]:
    """Provenance for a job. The stored query (and endpoint) is included only
    when ``owner`` matches the submitter; pass the caller's owner from any
    path that has not already enforced ownership (guard CX-02)."""
    known = _SUBMITTED_JOBS.get(job_url) or {}
    if known and owner != "__owner_checked__" and (known.get("owner") or "") != (owner or ""):
        known = {}
    prov: Dict[str, Any] = {"service": "vo_tap_async", "job_url": job_url,
                            "endpoint": known.get("endpoint") or job_url}
    if known.get("query"):
        prov["query"] = known["query"]
        prov["adql"] = _trunc(known["query"], 500)
    return prov


def _job_phase(job: Any) -> str:
    # pyvo's ``phase`` property re-GETs the job on every read; the parsed
    # UWS tree from the last fetch (``job._job``) already holds it.
    tree = getattr(job, "_job", None)
    phase = getattr(tree, "phase", None) if tree is not None else None
    if not phase:
        phase = getattr(job, "phase", "")
    return str(phase or "UNKNOWN").upper()


def _job_error_message(job: Any) -> Optional[str]:
    """The UWS errorSummary text. pyvo keeps the parsed UWS tree on
    ``job._job``; this is the traversal its own raise_if_error() uses."""
    tree = getattr(job, "_job", None)
    summary = getattr(tree, "errorsummary", None)
    message = getattr(summary, "message", None)
    text = getattr(message, "content", None)
    text = str(text).strip() if text else ""
    return text or None


def _exc_chain(exc: BaseException):
    seen = set()
    stack: List[Any] = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen or not isinstance(cur, BaseException):
            continue
        seen.add(id(cur))
        yield cur
        stack.extend([getattr(cur, "cause", None), cur.__cause__, cur.__context__])


def _is_timeout(exc: BaseException) -> bool:
    """A service/read timeout. An exhausted TOOL budget is not one: promoting
    it would only fail the async submit and hide the real error."""
    from services.tool_budgets import BudgetExhausted

    for cur in _exc_chain(exc):
        if isinstance(cur, BudgetExhausted):
            return False
    for cur in _exc_chain(exc):
        if isinstance(cur, (requests.exceptions.Timeout, TimeoutError, _DeadlineExceeded)):
            return True
        if "timed out" in str(cur).lower():
            return True
    return False


def _is_job_gone(exc: BaseException) -> bool:
    for cur in _exc_chain(exc):
        status = getattr(getattr(cur, "response", None), "status_code", None)
        if status in (404, 410):
            return True
    return False


def _job_gone_or(exc: BaseException) -> str:
    if _is_job_gone(exc):
        return ("The archive no longer has this job (deleted or expired). "
                "Resubmit the query with vo_adql_query(mode='async').")
    return str(exc)
