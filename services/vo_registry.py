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
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT_S = 45.0
DEFAULT_MAXREC_CAP = 1000
MAX_REGISTRY_ROWS = 100
MAX_TABLES = 200
MAX_COLUMNS = 200
MAX_CONE_RADIUS_DEG = 5.0

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
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(method, url, **kwargs)


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

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn)
        try:
            return future.result(timeout=seconds)
        except _FuturesTimeout:
            future.cancel()
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
    ):
        self.timeout = float(timeout if timeout is not None else _env_float("VO_REGISTRY_TIMEOUT", DEFAULT_TIMEOUT_S))
        cap_env = os.getenv("VO_ADQL_MAXREC_CAP")
        self.maxrec_cap = int(maxrec_cap if maxrec_cap is not None else (cap_env or DEFAULT_MAXREC_CAP))
        self._regsearch_fn = regsearch_fn
        self._tap_factory = tap_factory
        self._scs_factory = scs_factory

    # ── factories (lazy pyvo) ───────────────────────────────────────────────
    def _tap_service(self, access_url: str):
        if self._tap_factory is not None:
            return self._tap_factory(access_url)
        import pyvo  # lazy

        return pyvo.dal.TAPService(access_url, session=_TimeoutHTTPSession(self.timeout))

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
    def run_adql(self, access_url: Any, adql: Any, max_rows: Any = 200) -> Dict[str, Any]:
        url = str(access_url or "")
        query = str(adql or "")
        try:
            url = _require_url(access_url)
            stripped = _strip_adql_comments(query).strip()
            if not stripped.upper().startswith("SELECT"):
                return {"success": False,
                        "error": "Only SELECT queries are allowed on remote TAP services."}
            # guard review P2 (rollout-guard): reject multi-statement payloads
            # ("SELECT 1; DELETE ...") — anything after a top-level semicolon.
            if _has_trailing_statement(stripped):
                return {"success": False,
                        "error": "Only a single SELECT statement is allowed; remove "
                                 "everything after the semicolon."}
            maxrec, warnings = _clamp(max_rows, 200, self.maxrec_cap, "max_rows")

            svc = self._tap_service(url)
            result = svc.run_sync(query, maxrec=maxrec)
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


def _require_url(access_url: Any) -> str:
    url = str(access_url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"access_url must be an http(s) URL, got {access_url!r}.")
    return url


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
