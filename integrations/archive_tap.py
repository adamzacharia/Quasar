"""Small synchronous TAP client for a fixed set of well-known archive services.

Why not the generic ``vo_adql_query`` path: the model has to discover the
endpoint, the table names and the column names before it can ask anything,
and the ArchiveBench dev run (2026-09-24) showed gpt-oss spending its whole
tool budget on that discovery. The dedicated archive tools
(``services/archive_catalogs.py``) know their endpoints and build their ADQL,
so they need only a thin, predictable transport:

* one POST to ``<base>/sync`` per query (``REQUEST=doQuery``, ``LANG=ADQL``,
  ``FORMAT=votable``, ``MAXREC``);
* the timeout comes from ``services.tool_budgets.bounded_timeout`` so a query
  never outlives the calling tool's guard (and the process-wide requests hook
  routes the host through the circuit breaker);
* the server's own error text (VOTable ``INFO name="QUERY_STATUS"
  value="ERROR"``) is raised as :class:`TapQueryError` with the ADQL attached,
  never swallowed into an empty result;
* ``OVERFLOW`` (the MAXREC cap was hit) is reported as ``truncated``.

The HTTP function is injectable, so every caller is testable offline.
"""
from __future__ import annotations

import io
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "KNOWN_SERVICES",
    "TapService",
    "TapQueryError",
    "TapResult",
    "ArchiveTapClient",
    "quote_ident",
    "quote_table",
    "radec_columns",
    "sql_literal",
    "is_identifier",
]


@dataclass(frozen=True)
class TapService:
    key: str
    base_url: str
    label: str
    # VizieR stores table and column names quoted and is case sensitive: every
    # identifier sent to it must be double-quoted.
    quote_all: bool = False


KNOWN_SERVICES: Dict[str, TapService] = {
    "heasarc": TapService("heasarc", "https://heasarc.gsfc.nasa.gov/xamin/vo/tap", "NASA HEASARC"),
    "exoplanet": TapService("exoplanet", "https://exoplanetarchive.ipac.caltech.edu/TAP", "NASA Exoplanet Archive"),
    "simbad": TapService("simbad", "https://simbad.cds.unistra.fr/simbad/sim-tap", "CDS SIMBAD"),
    "vizier": TapService("vizier", "https://tapvizier.cds.unistra.fr/TAPVizieR/tap", "CDS VizieR", quote_all=True),
    "irsa": TapService("irsa", "https://irsa.ipac.caltech.edu/TAP", "NASA/IPAC IRSA"),
    "gaia": TapService("gaia", "https://gea.esac.esa.int/tap-server/tap", "ESA Gaia archive"),
}

DEFAULT_TIMEOUT_S = 45.0
DEFAULT_MAXREC = 2000
MAX_MAXREC = 50000

# ADQL reserved words that real archive columns use as names (HEASARC "time",
# "pi", "class"; SIMBAD "class"). Such names must be double-quoted.
_RESERVED = {
    "abs", "all", "and", "as", "asc", "avg", "between", "by", "case", "cast", "class", "count", "date",
    "desc", "distinct", "end", "exists", "first", "from", "group", "having", "in", "is", "join", "last",
    "like", "max", "min", "mod", "not", "null", "offset", "on", "or", "order", "pi", "power", "select",
    "size", "sum", "table", "time", "timestamp", "top", "type", "user", "value", "values", "where",
    "zone", "position", "year", "month", "day", "section", "level", "key", "names", "mode",
}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE_RE = re.compile(r'^"?[A-Za-z0-9_][A-Za-z0-9_./+\-]*"?$')


class TapQueryError(RuntimeError):
    """A TAP query failed; ``message`` is the server's (or transport's) own text."""

    def __init__(self, message: str, *, service: str = "", query: str = "", status: Optional[int] = None):
        self.message = message
        self.service = service
        self.query = query
        self.status = status
        super().__init__(message)


@dataclass
class TapResult:
    rows: List[Dict[str, Any]]
    columns: List[str]
    truncated: bool
    elapsed_s: float
    query: str
    endpoint: str
    service: str
    units: Dict[str, str] = field(default_factory=dict)

    def provenance(self) -> Dict[str, Any]:
        return {"service": self.service, "endpoint": self.endpoint, "query": self.query,
                "rows": len(self.rows), "truncated": self.truncated, "elapsed_s": round(self.elapsed_s, 2)}


def is_identifier(name: str) -> bool:
    return bool(_IDENT_RE.match(str(name or "").strip('"')))


def quote_ident(name: str, *, force: bool = False) -> str:
    """Return a column/alias name safe for ADQL; quote when reserved or forced."""
    raw = str(name or "").strip()
    bare = raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] == '"' else raw
    if not bare or '"' in bare:
        raise ValueError(f"invalid column name {name!r}")
    if force or not _IDENT_RE.match(bare) or bare.lower() in _RESERVED:
        return f'"{bare}"'
    return bare


def quote_table(name: str, *, force: bool = False) -> str:
    raw = str(name or "").strip()
    if not _TABLE_RE.match(raw):
        raise ValueError(f"invalid table name {name!r}")
    bare = raw.strip('"')
    if force or "/" in bare or "+" in bare or "-" in bare:
        return f'"{bare}"'
    return bare


def sql_literal(value: Any) -> str:
    """An ADQL literal for a number or string (strings single-quoted and escaped)."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number in a query value")
        return repr(float(value))
    text = str(value)
    if len(text) > 200:
        raise ValueError("query value too long")
    return "'" + text.replace("'", "''") + "'"


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a hard dependency
        np = None
    if value is None:
        return None
    if np is not None:
        if value is np.ma.masked:
            return None
        if isinstance(value, np.generic):
            value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _default_post(url: str, data: Dict[str, Any], timeout: float) -> Any:
    import requests

    return requests.post(url, data=data, timeout=timeout,
                         headers={"User-Agent": "Quasar-archive-tap/1.0"})


def _bounded(default: float, label: str) -> float:
    try:
        from services.tool_budgets import bounded_timeout

        return bounded_timeout(default, minimum=2.0, label=label)
    except ImportError:  # pragma: no cover - services always importable in the app
        return default


class ArchiveTapClient:
    """Synchronous ADQL against :data:`KNOWN_SERVICES` (or an explicit URL in tests)."""

    def __init__(self, *, http_post: Optional[Callable[..., Any]] = None, timeout: float = DEFAULT_TIMEOUT_S,
                 services: Optional[Dict[str, TapService]] = None):
        self.http_post = http_post or _default_post
        self.timeout = float(timeout)
        self.services = dict(services or KNOWN_SERVICES)
        self._columns_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    # ── core query ────────────────────────────────────────────────────
    def service(self, key: str) -> TapService:
        svc = self.services.get(str(key or "").lower())
        if svc is None:
            raise TapQueryError(f"unknown TAP service {key!r}; known: {', '.join(sorted(self.services))}")
        return svc

    def query(self, service: str, adql: str, *, maxrec: int = DEFAULT_MAXREC,
              timeout: Optional[float] = None) -> TapResult:
        svc = self.service(service)
        maxrec = max(1, min(int(maxrec), MAX_MAXREC))
        endpoint = svc.base_url.rstrip("/") + "/sync"
        data = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": adql, "MAXREC": maxrec}
        started = time.monotonic()
        try:
            wait = _bounded(float(timeout or self.timeout), f"{svc.label} TAP")
            resp = self.http_post(endpoint, data, wait)
        except TapQueryError:
            raise
        except Exception as exc:
            raise TapQueryError(f"{svc.label} TAP request failed: {type(exc).__name__}: {exc}",
                                service=svc.key, query=adql) from exc
        elapsed = time.monotonic() - started
        status = getattr(resp, "status_code", 200)
        content = getattr(resp, "content", b"") or b""
        rows, columns, units, truncated, error = _parse_votable(content)
        if error:
            raise TapQueryError(f"{svc.label} rejected the query: {error}", service=svc.key, query=adql, status=status)
        if status and int(status) >= 400:
            snippet = content[:300].decode("utf-8", "replace") if isinstance(content, bytes) else str(content)[:300]
            raise TapQueryError(f"{svc.label} TAP returned HTTP {status}: {snippet}", service=svc.key,
                                query=adql, status=status)
        if columns is None:
            snippet = content[:300].decode("utf-8", "replace") if isinstance(content, bytes) else str(content)[:300]
            raise TapQueryError(f"{svc.label} TAP returned no result table: {snippet}", service=svc.key, query=adql)
        truncated = truncated or len(rows) >= maxrec
        return TapResult(rows=rows, columns=columns, truncated=truncated, elapsed_s=elapsed, query=adql,
                         endpoint=endpoint, service=svc.key, units=units)

    # ── metadata ──────────────────────────────────────────────────────
    def table_columns(self, service: str, table: str) -> List[Dict[str, Any]]:
        """TAP_SCHEMA columns of one table (name without quotes, ucd, unit, datatype), cached."""
        svc = self.service(service)
        bare = str(table).strip().strip('"')
        key = (svc.key, bare.lower())
        with self._lock:
            cached = self._columns_cache.get(key)
        if cached is not None:
            return cached
        names = [bare]
        if svc.quote_all:
            names.append(f'"{bare}"')
        where = " OR ".join(f"table_name = {sql_literal(n)}" for n in names)
        res = self.query(svc.key, f"SELECT column_name, ucd, unit, datatype FROM TAP_SCHEMA.columns WHERE {where}",
                         maxrec=2000)
        if not res.rows and not svc.quote_all:
            res = self.query(svc.key, "SELECT column_name, ucd, unit, datatype FROM TAP_SCHEMA.columns "
                                      f"WHERE LOWER(table_name) = {sql_literal(bare.lower())}", maxrec=2000)
        cols = []
        for r in res.rows:
            name = str(r.get("column_name") or "").strip().strip('"')
            if name:
                cols.append({"name": name, "ucd": str(r.get("ucd") or ""), "unit": str(r.get("unit") or ""),
                             "datatype": str(r.get("datatype") or "")})
        if cols:
            with self._lock:
                if len(self._columns_cache) > 512:
                    self._columns_cache.clear()
                self._columns_cache[key] = cols
        return cols


def radec_columns(columns: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Pick a table's main RA/Dec columns from UCDs, then from common names."""
    def by_ucd(prefix: str) -> Optional[str]:
        main = [c["name"] for c in columns if c.get("ucd", "").lower().startswith(prefix)
                and "meta.main" in c.get("ucd", "").lower()]
        if main:
            return main[0]
        exact = [c["name"] for c in columns if c.get("ucd", "").lower() == prefix]
        return exact[0] if exact else None

    ra, dec = by_ucd("pos.eq.ra"), by_ucd("pos.eq.dec")
    if ra and dec:
        return ra, dec
    names = {c["name"].lower(): c["name"] for c in columns}
    for r_name, d_name in (("ra", "dec"), ("raj2000", "dej2000"), ("ra_icrs", "de_icrs"), ("ra_deg", "dec_deg"),
                           ("_raj2000", "_dej2000"), ("radeg", "dedeg"), ("ra2000", "dec2000")):
        if r_name in names and d_name in names:
            return names[r_name], names[d_name]
    return ra, dec


def _parse_votable(content: bytes):
    """-> (rows, columns | None, units, truncated, error_text | None)."""
    if not content:
        return [], None, {}, False, None
    try:
        from astropy.io.votable import parse
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vt = parse(io.BytesIO(content), verify="ignore")
    except Exception:
        text = content[:2000].decode("utf-8", "replace") if isinstance(content, bytes) else str(content)[:2000]
        m = re.search(r'QUERY_STATUS"\s+value="ERROR"[^>]*>(.*?)</INFO>', text, re.S)
        if m:
            return [], None, {}, False, re.sub(r"\s+", " ", m.group(1)).strip()[:600]
        return [], None, {}, False, None
    error = None
    truncated = False
    tables = []
    for res in vt.resources:
        for info in res.infos:
            if (info.name or "").upper() == "QUERY_STATUS":
                val = (info.value or "").upper()
                if val == "ERROR":
                    error = re.sub(r"\s+", " ", str(info.content or "query error")).strip()[:600]
                elif val == "OVERFLOW":
                    truncated = True
        tables.extend(res.tables)
    for info in getattr(vt, "infos", []) or []:
        if (info.name or "").upper() == "QUERY_STATUS" and (info.value or "").upper() == "ERROR":
            error = error or re.sub(r"\s+", " ", str(info.content or "query error")).strip()[:600]
    if error:
        return [], None, {}, truncated, error
    if not tables:
        return [], None, {}, truncated, None
    # IRSA gives FIELDs ids like "col_0" and the real column in name=; prefer names.
    try:
        table = tables[0].to_table(use_names_over_ids=True)
    except Exception:
        table = tables[0].to_table()
    columns = list(table.colnames)
    units = {c: str(table[c].unit) for c in columns if table[c].unit is not None}
    rows = []
    for rec in table:
        rows.append({c: _json_safe(rec[c]) for c in columns})
    return rows, columns, units, truncated, None
