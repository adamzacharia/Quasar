"""Dedicated archive query builders over well-known TAP services (2026-09).

Closes the ArchiveBench dev-set gaps where Quasar had no client at all
(HEASARC observation tables, the NASA Exoplanet Archive, SIMBAD as a
database, TNS) or only a name resolver / capped row pull (Gaia, VizieR,
cross-matches). See ``Benchmark/archivebench/GAPS.md`` and
``tools-plan.md``.

Every public method returns a plain dict:

    {"success": True, "summary": {...scalars the model must see...},
     "rows": [...], "columns": [...], "warnings": [...], "provenance": {...}}

or ``{"success": False, "error": "...", "provenance": {...}}``. Failures are
never reported as empty results. Counts are computed by the archive
(``COUNT(*)`` / ``SUM``), not by counting a capped row pull, and every row
pull states whether it was truncated. Each executed ADQL statement is listed
in ``provenance["queries"]`` (the first one also as ``provenance["query"]``,
which the UI "Show query" panel renders).

Nothing here knows about any particular target: targets arrive as positions
(resolved by the caller) or as identifiers the archive itself matches.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import html as _html
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from integrations.archive_tap import (
    ArchiveTapClient,
    KNOWN_SERVICES,
    TapQueryError,
    TapResult,
    quote_ident,
    quote_table,
    radec_columns,
    sql_literal,
)

__all__ = ["ArchiveCatalogService", "HEASARC_MISSIONS", "parse_cuts", "mjd_to_iso", "iso_to_mjd"]

MJD_UNIX_EPOCH = 40587.0

# ── time helpers ───────────────────────────────────────────────────────


def mjd_to_iso(mjd: Any) -> Optional[str]:
    try:
        value = float(mjd)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    seconds = (value - MJD_UNIX_EPOCH) * 86400.0
    try:
        return (_dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(seconds=seconds)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, ValueError):
        return None


def iso_to_mjd(text: Any) -> Optional[float]:
    if text is None or str(text).strip() == "":
        return None
    raw = str(text).strip()
    try:
        return float(raw) if re.fullmatch(r"\d{5}(\.\d+)?", raw) else _parse_iso_mjd(raw)
    except ValueError:
        return None


def _parse_iso_mjd(raw: str) -> float:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            d = _dt.datetime.strptime(raw.replace("Z", ""), fmt).replace(tzinfo=_dt.timezone.utc)
            return (d - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)).total_seconds() / 86400.0 + MJD_UNIX_EPOCH
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {raw!r} (use YYYY-MM-DD)")


def _now_mjd() -> float:
    now = _dt.datetime.now(_dt.timezone.utc)
    return (now - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)).total_seconds() / 86400.0 + MJD_UNIX_EPOCH


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── geometry ───────────────────────────────────────────────────────────


def _sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    s = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(s)))) * 3600.0


def _validate_position(ra: Any, dec: Any) -> Tuple[float, float]:
    try:
        ra_f, dec_f = float(ra), float(dec)
    except (TypeError, ValueError):
        raise ValueError("ra and dec must be numbers in degrees")
    if not (0.0 <= ra_f < 360.0) or not (-90.0 <= dec_f <= 90.0):
        raise ValueError(f"position out of range: ra={ra_f}, dec={dec_f}")
    return ra_f, dec_f


def _cone(ra_col: str, dec_col: str, ra: float, dec: float, radius_deg: float) -> str:
    return (f"1=CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), "
            f"CIRCLE('ICRS', {ra!r}, {dec!r}, {radius_deg!r}))")


def _float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ── cuts ───────────────────────────────────────────────────────────────

_OPS = {
    "=": "=", "==": "=", "eq": "=", "!=": "<>", "<>": "<>", "ne": "<>", "<": "<", "lt": "<", "<=": "<=",
    "le": "<=", ">": ">", "gt": ">", ">=": ">=", "ge": ">=", "like": "LIKE", "not like": "NOT LIKE",
    "between": "BETWEEN", "in": "IN", "is null": "IS NULL", "is not null": "IS NOT NULL",
}
_CUT_STRING_RE = re.compile(
    r"^\s*(?P<col>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>>=|<=|!=|<>|==|=|<|>|\bnot like\b|\blike\b|\bbetween\b|\bis not null\b|\bis null\b|\bin\b)\s*(?P<val>.*)$",
    re.I,
)


def _coerce_value(text: Any) -> Any:
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return text
    raw = str(text).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    try:
        return int(raw) if re.fullmatch(r"[+-]?\d+", raw) else float(raw)
    except ValueError:
        return raw


def parse_cuts(cuts: Any) -> List[Tuple[str, str, Any]]:
    """Normalise cuts given as dicts {column, op, value} or strings "col op value".

    ``between`` takes a two-item list (or "a and b"); ``in`` takes a list.
    """
    if cuts is None or cuts == "":
        return []
    items = cuts if isinstance(cuts, (list, tuple)) else [cuts]
    out: List[Tuple[str, str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            col = str(item.get("column") or item.get("col") or "").strip()
            op = str(item.get("op") or item.get("operator") or "=").strip().lower()
            val = item.get("value", item.get("values"))
        else:
            text = str(item).strip()
            if not text:
                continue
            if " and " in text.lower() and not re.search(r"\bbetween\b", text, re.I):
                for part in re.split(r"\s+and\s+", text, flags=re.I):
                    out.extend(parse_cuts(part))
                continue
            m = _CUT_STRING_RE.match(text)
            if not m:
                raise ValueError(f"cannot parse cut {text!r}; use 'column op value', e.g. 'parallax > 10'")
            col, op, val = m.group("col"), m.group("op").lower(), m.group("val").strip()
            if op == "between":
                parts = re.split(r"\s+and\s+", val, flags=re.I)
                val = [_coerce_value(p) for p in parts]
            elif op == "in":
                val = [_coerce_value(p) for p in val.strip("()[] ").split(",") if p.strip()]
            elif op in ("is null", "is not null"):
                val = None
            else:
                val = _coerce_value(val)
        if op not in _OPS:
            raise ValueError(f"unsupported operator {op!r}; use one of {', '.join(sorted(set(_OPS)))}")
        if not col:
            raise ValueError("a cut needs a column")
        out.append((col, _OPS[op], val))
    return out


def _cut_sql(col_sql: str, op: str, value: Any) -> str:
    if op in ("IS NULL", "IS NOT NULL"):
        return f"{col_sql} {op}"
    if op == "BETWEEN":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("between needs two values")
        return f"{col_sql} BETWEEN {sql_literal(_coerce_value(value[0]))} AND {sql_literal(_coerce_value(value[1]))}"
    if op == "IN":
        vals = value if isinstance(value, (list, tuple)) else [value]
        if not vals or len(vals) > 100:
            raise ValueError("in needs 1 to 100 values")
        return f"{col_sql} IN ({', '.join(sql_literal(_coerce_value(v)) for v in vals)})"
    if isinstance(value, (list, tuple)):
        raise ValueError(f"operator {op} takes one value")
    return f"{col_sql} {op} {sql_literal(_coerce_value(value))}"


class _Columns:
    """Case-insensitive column resolver for one table (from TAP_SCHEMA)."""

    def __init__(self, cols: Sequence[Dict[str, Any]], quote_all: bool):
        self.cols = list(cols)
        self.quote_all = quote_all
        self.by_lower = {c["name"].lower(): c["name"] for c in self.cols}

    def names(self) -> List[str]:
        return [c["name"] for c in self.cols]

    def resolve(self, name: str) -> str:
        key = str(name or "").strip().strip('"').lower()
        if key in self.by_lower:
            return self.by_lower[key]
        close = difflib.get_close_matches(key, list(self.by_lower), n=5, cutoff=0.6)
        hint = f" Close matches: {', '.join(self.by_lower[c] for c in close)}." if close else ""
        raise ValueError(f"column {name!r} is not in this table.{hint}")

    def sql(self, name: str) -> str:
        return quote_ident(self.resolve(name), force=self.quote_all)


def _merge_prov(results: Sequence[TapResult], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    queries = [r.query for r in results]
    prov: Dict[str, Any] = {
        "service": results[0].service if results else "",
        "endpoint": results[0].endpoint if results else "",
        # The UI "Show query" panel renders one string: list every statement run.
        "query": "\n\n".join(queries),
        "queries": queries,
        "retrieved_utc": _utc_now(),
    }
    if extra:
        prov.update(extra)
    return prov


def _fail(message: str, *, queries: Sequence[str] = (), service: str = "") -> Dict[str, Any]:
    prov: Dict[str, Any] = {"service": service, "queries": list(queries)}
    if queries:
        prov["query"] = queries[-1]
    return {"success": False, "error": message, "provenance": prov}


# ── HEASARC mission master tables ──────────────────────────────────────
# Column maps verified against the live HEASARC TAP_SCHEMA on 2026-09-24.
HEASARC_MISSIONS: Dict[str, Dict[str, Any]] = {
    "chandra": {"table": "chanmaster", "label": "Chandra", "obsid": "obsid", "time": "time", "exposure": "exposure",
                "public": "public_date", "status": "status", "name": "name", "pi": "pi", "radius_arcmin": 10.0,
                "extra": ["detector", "grating", "type", "data_mode", "proposal", "cycle", "category"]},
    "xmm": {"table": "xmmmaster", "label": "XMM-Newton", "obsid": "obsid", "time": "time", "exposure": "duration",
            "public": "public_date", "status": "status", "name": "name", "pi": "pi_lname", "radius_arcmin": 15.0,
            "extra": ["pn_time", "mos1_time", "mos2_time", "rgs1_time", "rgs2_time", "om_time", "pi_title"],
            "instruments": {"pn": "pn_time", "epic-pn": "pn_time", "mos": "mos1_time", "mos1": "mos1_time",
                            "mos2": "mos2_time", "rgs": "rgs1_time", "rgs1": "rgs1_time", "rgs2": "rgs2_time",
                            "om": "om_time"}},
    "swift": {"table": "swiftmastr", "label": "Swift", "obsid": "obsid", "time": "start_time",
              "exposure": "xrt_exposure", "public": "archive_date", "status": None, "name": "name", "pi": "pi",
              "radius_arcmin": 12.0, "extra": ["xrt_exposure", "uvot_exposure", "bat_exposure", "target_id"],
              "instruments": {"xrt": "xrt_exposure", "uvot": "uvot_exposure", "bat": "bat_exposure"}},
    "nustar": {"table": "numaster", "label": "NuSTAR", "obsid": "obsid", "time": "time", "exposure": "exposure_a",
               "public": "public_date", "status": "status", "name": "name", "pi": "pi_lname", "radius_arcmin": 6.0,
               "extra": ["observation_mode", "obs_type", "title"]},
    "nicer": {"table": "nicermastr", "label": "NICER", "obsid": "obsid", "time": "time", "exposure": "exposure",
              "public": "public_date", "status": None, "name": "name", "pi": "pi_lname", "radius_arcmin": 3.0,
              "extra": ["obs_type", "title"]},
    "suzaku": {"table": "suzamaster", "label": "Suzaku", "obsid": "obsid", "time": "time", "exposure": "exposure",
               "public": "public_date", "status": None, "name": "name", "pi": "pi_lname", "radius_arcmin": 10.0,
               "extra": ["obs_type", "title"]},
    "rosat": {"table": "rosmaster", "label": "ROSAT", "obsid": "seq_id", "time": "start_time", "exposure": "exposure",
              "public": None, "status": None, "name": "name", "pi": "pi_lname", "radius_arcmin": 30.0,
              "extra": ["instrument", "filter", "title"], "instrument_like": "instrument"},
}
_MISSION_ALIASES = {"cxo": "chandra", "chandra x-ray observatory": "chandra", "xmm-newton": "xmm",
                    "xmmnewton": "xmm", "swift-xrt": "swift", "neil gehrels swift": "swift", "nu-star": "nustar",
                    "pspc": "rosat", "hri": "rosat"}


# ── Exoplanet Archive ──────────────────────────────────────────────────
EXO_PLANET_COLUMNS = [
    "pl_name", "hostname", "discoverymethod", "disc_year", "disc_facility",
    "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_rade", "pl_radeerr1", "pl_radeerr2",
    "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2", "pl_bmassprov", "pl_orbsmax", "pl_orbeccen",
    "pl_eqt", "pl_insol", "st_teff", "st_rad", "st_mass", "st_spectype", "sy_dist", "sy_pnum", "ra", "dec",
]
EXO_DEFAULT_SOLUTION_COLUMNS = [
    "pl_name", "pl_refname", "default_flag", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_rade",
    "pl_radeerr1", "pl_radeerr2", "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2", "pl_bmassprov", "pl_masse",
    "pl_masseerr1", "pl_masseerr2", "pl_msinie", "pl_msinieerr1", "pl_msinieerr2", "rv_flag", "tran_flag",
    "pl_pubdate", "disc_refname",
]
EXO_CENSUS_COLUMNS = ["pl_name", "hostname", "sy_dist", "pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt", "st_teff",
                      "discoverymethod", "disc_facility", "disc_year"]
# typed census filter -> (column, operator)
EXO_TYPED_FILTERS = {
    "max_distance_pc": ("sy_dist", "<="), "min_distance_pc": ("sy_dist", ">="),
    "min_host_teff_k": ("st_teff", ">="), "max_host_teff_k": ("st_teff", "<="),
    "min_eq_temp_k": ("pl_eqt", ">="), "max_eq_temp_k": ("pl_eqt", "<="),
    "min_radius_earth": ("pl_rade", ">="), "max_radius_earth": ("pl_rade", "<="),
    "min_mass_earth": ("pl_bmasse", ">="), "max_mass_earth": ("pl_bmasse", "<="),
    "min_period_days": ("pl_orbper", ">="), "max_period_days": ("pl_orbper", "<="),
    "min_disc_year": ("disc_year", ">="), "max_disc_year": ("disc_year", "<="),
}


# ── Gaia ───────────────────────────────────────────────────────────────
GAIA_SOURCE_TABLE = "gaiadr3.gaia_source"
GAIA_SOURCE_COLUMNS = [
    "source_id", "ra", "dec", "ref_epoch", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec",
    "pmdec_error", "phot_g_mean_mag", "phot_g_mean_flux_over_error", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "bp_rp", "radial_velocity", "radial_velocity_error", "ruwe", "phot_variable_flag",
]
GAIA_VARI_TABLES = {
    "rrlyrae": "gaiadr3.vari_rrlyrae", "rr_lyrae": "gaiadr3.vari_rrlyrae", "vari_rrlyrae": "gaiadr3.vari_rrlyrae",
    "cepheid": "gaiadr3.vari_cepheid", "vari_cepheid": "gaiadr3.vari_cepheid",
    "eclipsing_binary": "gaiadr3.vari_eclipsing_binary", "vari_eclipsing_binary": "gaiadr3.vari_eclipsing_binary",
    "long_period_variable": "gaiadr3.vari_long_period_variable", "lpv": "gaiadr3.vari_long_period_variable",
    "vari_long_period_variable": "gaiadr3.vari_long_period_variable",
    "summary": "gaiadr3.vari_summary", "vari_summary": "gaiadr3.vari_summary",
    "classifier": "gaiadr3.vari_classifier_result", "vari_classifier_result": "gaiadr3.vari_classifier_result",
    "short_timescale": "gaiadr3.vari_short_timescale", "rotation_modulation": "gaiadr3.vari_rotation_modulation",
    "planetary_transit": "gaiadr3.vari_planetary_transit", "agn": "gaiadr3.vari_agn",
}
GAIA_EPOCH = 2016.0


class ArchiveCatalogService:
    """Structured queries for HEASARC, Exoplanet Archive, SIMBAD, Gaia, VizieR/IRSA catalogues, TNS and ALeRCE."""

    def __init__(
        self,
        client: Optional[ArchiveTapClient] = None,
        *,
        vizier_find: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        http_get: Optional[Callable[..., Any]] = None,
        now_mjd: Optional[Callable[[], float]] = None,
    ):
        self.client = client or ArchiveTapClient()
        self._vizier_find = vizier_find
        self._http_get = http_get
        self._now_mjd = now_mjd or _now_mjd

    # ── shared helpers ──────────────────────────────────────────────
    def _columns(self, service: str, table: str) -> _Columns:
        svc = self.client.service(service)
        cols = self.client.table_columns(svc.key, table)
        if not cols:
            raise ValueError(
                f"table {table!r} was not found on {svc.label} (TAP_SCHEMA has no columns for it). "
                "Use catalog_find to look up the exact table name.")
        return _Columns(cols, svc.quote_all)

    def _get(self, url: str, *, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0,
             headers: Optional[Dict[str, str]] = None) -> Any:
        from services.tool_budgets import bounded_timeout

        wait = bounded_timeout(timeout, minimum=2.0, label=url.split("/")[2])
        if self._http_get is not None:
            return self._http_get(url, params=params, timeout=wait, headers=headers)
        import requests

        return requests.get(url, params=params, timeout=wait,
                            headers=headers or {"User-Agent": "Quasar-archive-tools/1.0"})

    # ── HEASARC ────────────────────────────────────────────────────
    def heasarc_observations(
        self, mission: str, ra: Any, dec: Any, *, radius_arcmin: Any = None, instrument: Optional[str] = None,
        public_only: bool = False, start_date: Any = None, end_date: Any = None, limit: Any = 100,
        target_label: str = "",
    ) -> Dict[str, Any]:
        key = str(mission or "").strip().lower()
        key = _MISSION_ALIASES.get(key, key)
        spec = HEASARC_MISSIONS.get(key)
        if spec is None:
            return _fail(f"unknown mission {mission!r}; supported: {', '.join(sorted(HEASARC_MISSIONS))}",
                         service="heasarc")
        try:
            ra_f, dec_f = _validate_position(ra, dec)
        except ValueError as exc:
            return _fail(str(exc), service="heasarc")
        warnings: List[str] = []
        radius = _float(radius_arcmin)
        if radius is None or radius <= 0:
            radius = float(spec["radius_arcmin"])
            warnings.append(f"radius defaulted to {radius:g} arcmin (about half the {spec['label']} field of view); "
                            "pointings within this offset of the position are listed")
        radius = min(radius, 120.0)
        limit_i = max(1, min(int(_float(limit) or 100), 500))
        q = lambda c: quote_ident(c)  # noqa: E731
        exposure_col = spec["exposure"]
        where = [_cone("ra", "dec", ra_f, dec_f, radius / 60.0)]
        instrument_note = None
        if instrument and str(instrument).strip():
            inst = str(instrument).strip()
            clause, exposure_override, instrument_note = self._heasarc_instrument(key, spec, inst)
            if clause is None:
                return _fail(instrument_note or f"instrument {inst!r} is not recognised for {spec['label']}",
                             service="heasarc")
            where.append(clause)
            if exposure_override:
                exposure_col = exposure_override
        now = self._now_mjd()
        if public_only:
            if spec.get("public"):
                where.append(f"{q(spec['public'])} <= {now!r}")
            else:
                warnings.append(f"{spec['label']} has no public-date column; all rows are archival")
        t0, t1 = iso_to_mjd(start_date), iso_to_mjd(end_date)
        if start_date and t0 is None or end_date and t1 is None:
            return _fail("start_date / end_date must be YYYY-MM-DD", service="heasarc")
        if t0 is not None:
            where.append(f"{q(spec['time'])} >= {t0!r}")
        if t1 is not None:
            where.append(f"{q(spec['time'])} <= {t1 + 1.0!r}")
        where_sql = " AND ".join(where)
        table = spec["table"]
        count_sql = (f"SELECT COUNT(*) AS n_obs, SUM({q(exposure_col)}) AS total_exposure_s, "
                     f"MIN({q(spec['time'])}) AS first_mjd, MAX({q(spec['time'])}) AS last_mjd "
                     f"FROM {table} WHERE {where_sql}")
        select_cols = [spec["name"], "ra", "dec", spec["time"], exposure_col]
        for c in [spec.get("public"), spec.get("status"), spec.get("pi")] + list(spec.get("extra", [])):
            if c and c not in select_cols:
                select_cols.append(c)
        # xamin renames a bare obsid column to "DataLinkID" in its output; an alias keeps the name.
        rows_sql = (f"SELECT TOP {limit_i} {q(spec['obsid'])} AS obs_id, {', '.join(q(c) for c in select_cols)} "
                    f"FROM {table} WHERE {where_sql} ORDER BY {q(spec['time'])}")
        results: List[TapResult] = []
        try:
            cres = self.client.query("heasarc", count_sql, maxrec=10)
            results.append(cres)
            rres = self.client.query("heasarc", rows_sql, maxrec=limit_i)
            results.append(rres)
            public_count = None
            if spec.get("public") and not public_only:
                pres = self.client.query(
                    "heasarc", f"SELECT COUNT(*) AS n_public FROM {table} WHERE {where_sql} AND "
                               f"{q(spec['public'])} <= {now!r}", maxrec=10)
                results.append(pres)
                public_count = int((pres.rows[0] or {}).get("n_public") or 0) if pres.rows else 0
        except TapQueryError as exc:
            return _fail(exc.message, queries=[r.query for r in results] + [exc.query], service="heasarc")
        crow = cres.rows[0] if cres.rows else {}
        n_obs = int(crow.get("n_obs") or 0)
        rows = []
        for r in rres.rows:
            obsid = r.get("obs_id", r.get(spec["obsid"], r.get("DataLinkID")))
            row = {"obsid": str(obsid) if obsid is not None else None, "target": r.get(spec["name"]),
                   "offset_arcmin": round(_sep_arcsec(ra_f, dec_f, float(r["ra"]), float(r["dec"])) / 60.0, 3)
                   if r.get("ra") is not None and r.get("dec") is not None else None,
                   "start_utc": mjd_to_iso(r.get(spec["time"])), "exposure_s": r.get(exposure_col)}
            if spec.get("public"):
                pub = r.get(spec["public"])
                row["public_date_utc"] = mjd_to_iso(pub)
                row["is_public"] = bool(pub is not None and _float(pub) is not None and float(pub) <= now)
            if spec.get("status"):
                row["status"] = r.get(spec["status"])
            if spec.get("pi"):
                row["pi"] = r.get(spec["pi"])
            for c in spec.get("extra", []):
                if c != exposure_col:
                    row[c] = r.get(c)
            rows.append(row)
        status_counts: Dict[str, int] = {}
        for row in rows:
            if row.get("status") is not None:
                status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
        summary: Dict[str, Any] = {
            "mission": spec["label"], "table": table, "center": {"ra": ra_f, "dec": dec_f, "label": target_label},
            "radius_arcmin": radius, "n_observations": n_obs,
            "total_exposure_s": _float(crow.get("total_exposure_s")),
            "exposure_column": exposure_col,
            "first_observation_utc": mjd_to_iso(crow.get("first_mjd")),
            "last_observation_utc": mjd_to_iso(crow.get("last_mjd")),
            "obsids": [row["obsid"] for row in rows],
            "listed": len(rows), "list_truncated": n_obs > len(rows),
        }
        if public_count is not None:
            summary["n_public"] = public_count
        if public_only:
            summary["public_only"] = True
        if instrument_note:
            summary["instrument_filter"] = instrument_note
        if status_counts:
            summary["status_counts_in_list"] = status_counts
        if n_obs > len(rows):
            warnings.append(f"{n_obs} observations match; the list shows the first {len(rows)} by date "
                            "(the count and total exposure cover all of them)")
        if n_obs == 0:
            warnings.append(f"no {spec['label']} observation is centred within {radius:g} arcmin of this position "
                            "(the archive answered normally; this is a real zero)")
        columns = list(rows[0].keys()) if rows else ["obsid", "target", "offset_arcmin", "start_utc", "exposure_s"]
        return {"success": True, "summary": summary, "rows": rows, "columns": columns, "warnings": warnings,
                "provenance": _merge_prov(results, {"table": table})}

    @staticmethod
    def _heasarc_instrument(key: str, spec: Dict[str, Any], inst: str) -> Tuple[Optional[str], Optional[str], str]:
        q = quote_ident
        text = inst.strip()
        up = text.upper().replace(" ", "")
        if key == "chandra":
            clauses, notes = [], []
            for part in re.split(r"[/+,&]|\bWITH\b", up):
                part = part.strip()
                if not part:
                    continue
                if part in ("HETG", "HETGS", "LETG", "LETGS", "NONE"):
                    g = part.rstrip("S") if part != "NONE" else "NONE"
                    clauses.append(f"grating = {sql_literal(g)}")
                    notes.append(f"grating={g}")
                elif part.startswith("ACIS") or part.startswith("HRC"):
                    clauses.append(f"detector LIKE {sql_literal(part + '%')}")
                    notes.append(f"detector={part}*")
                else:
                    return None, None, (f"Chandra instrument {text!r} not recognised; use ACIS, ACIS-S, ACIS-I, "
                                        "HRC, HRC-I, HRC-S, HETG, LETG or NONE (no grating)")
            return (" AND ".join(clauses) or None), None, ", ".join(notes)
        inst_map = spec.get("instruments")
        if inst_map:
            col = inst_map.get(text.lower().replace(" ", ""))
            if not col:
                return None, None, (f"{spec['label']} instrument {text!r} not recognised; use one of "
                                    f"{', '.join(sorted(set(inst_map)))}")
            return f"{q(col)} > 0", col, f"{text} exposure > 0 ({col})"
        like_col = spec.get("instrument_like")
        if like_col:
            return f"{q(like_col)} LIKE {sql_literal(up + '%')}", None, f"{like_col}={up}*"
        return None, None, f"{spec['label']} has a single instrument configuration; omit the instrument filter"

    # ── Exoplanet Archive ──────────────────────────────────────────
    def exoplanet_lookup(self, planet_name: Optional[str] = None, hostname: Optional[str] = None) -> Dict[str, Any]:
        name = str(planet_name or "").strip()
        host = str(hostname or "").strip()
        if not name and not host:
            return _fail("give planet_name (e.g. a planet designation) or hostname", service="exoplanet")
        results: List[TapResult] = []
        try:
            cols = self._columns("exoplanet", "pscomppars")
            planet_cols = [c for c in EXO_PLANET_COLUMNS if c.lower() in cols.by_lower]
            ps_cols_obj = self._columns("exoplanet", "ps")
            ps_cols = [c for c in EXO_DEFAULT_SOLUTION_COLUMNS if c.lower() in ps_cols_obj.by_lower]
            if name:
                cond = f"pl_name = {sql_literal(name)}"
            else:
                cond = f"hostname = {sql_literal(host)}"
            res = self.client.query("exoplanet", f"SELECT {', '.join(planet_cols)} FROM pscomppars WHERE {cond}",
                                    maxrec=200)
            results.append(res)
            if not res.rows:
                token = name or host
                loose = re.sub(r"[\s\-_]+", "%", token.strip())
                field = "pl_name" if name else "hostname"
                res = self.client.query(
                    "exoplanet",
                    f"SELECT {', '.join(planet_cols)} FROM pscomppars WHERE LOWER({field}) LIKE "
                    f"{sql_literal('%' + loose.lower() + '%')}", maxrec=50)
                results.append(res)
            if not res.rows:
                return {"success": True, "summary": {"found": False, "query_name": name or host,
                                                     "note": "no planet or host with that name in the composite "
                                                             "planet table (pscomppars) of confirmed planets"},
                        "rows": [], "columns": planet_cols, "warnings": [],
                        "provenance": _merge_prov(results, {"table": "pscomppars"})}
            planets = [r["pl_name"] for r in res.rows]
            sol = self.client.query(
                "exoplanet",
                f"SELECT {', '.join(ps_cols)} FROM ps WHERE pl_name IN ({', '.join(sql_literal(p) for p in planets[:20])}) "
                "AND default_flag = 1", maxrec=50)
            results.append(sol)
            nsol = self.client.query(
                "exoplanet",
                f"SELECT pl_name, COUNT(*) AS n_solutions FROM ps WHERE pl_name IN "
                f"({', '.join(sql_literal(p) for p in planets[:20])}) GROUP BY pl_name", maxrec=50)
            results.append(nsol)
        except (TapQueryError, ValueError) as exc:
            msg = exc.message if isinstance(exc, TapQueryError) else str(exc)
            return _fail(msg, queries=[r.query for r in results], service="exoplanet")
        default_by = {r["pl_name"]: r for r in sol.rows}
        nsol_by = {r["pl_name"]: r.get("n_solutions") for r in nsol.rows}
        planets_out = []
        for r in res.rows:
            d = default_by.get(r["pl_name"], {})
            planets_out.append({
                "composite": r,
                "default_solution": d,
                "n_published_solutions": nsol_by.get(r["pl_name"]),
                "mass_provenance": d.get("pl_bmassprov") or r.get("pl_bmassprov"),
                "measured_mass_earth": d.get("pl_masse"),
                "msini_earth": d.get("pl_msinie"),
                "default_reference": _strip_html(d.get("pl_refname")),
            })
        summary = {"found": True, "n_planets": len(planets_out), "planets": planets_out,
                   "tables": "pscomppars (composite: one row per planet, values may come from different papers) and "
                             "ps with default_flag=1 (one self-consistent published solution)",
                   "retrieved_utc": _utc_now()}
        warnings = []
        if any(p["mass_provenance"] and "relation" in str(p["mass_provenance"]).lower() for p in planets_out):
            warnings.append("some masses are estimates from a mass-radius relation, not measurements")
        return {"success": True, "summary": summary, "rows": res.rows, "columns": planet_cols,
                "warnings": warnings, "provenance": _merge_prov(results, {"table": "pscomppars"})}

    def exoplanet_census(self, *, filters: Optional[Dict[str, Any]] = None, discovery_facility: Optional[str] = None,
                         discovery_method: Optional[str] = None, cuts: Any = None, columns: Optional[List[str]] = None,
                         count_only: bool = False, limit: Any = 200, order_by: Optional[str] = None) -> Dict[str, Any]:
        results: List[TapResult] = []
        try:
            cols = self._columns("exoplanet", "pscomppars")
            where: List[str] = []
            applied: List[str] = []
            for key, value in (filters or {}).items():
                if value is None or value == "":
                    continue
                if key not in EXO_TYPED_FILTERS:
                    raise ValueError(f"unknown filter {key!r}; known: {', '.join(sorted(EXO_TYPED_FILTERS))}")
                col, op = EXO_TYPED_FILTERS[key]
                num = _float(value)
                if num is None:
                    raise ValueError(f"filter {key} needs a number")
                where.append(f"{col} {op} {num!r}")
                applied.append(f"{col} {op} {num:g}")
            if discovery_facility:
                where.append(f"LOWER(disc_facility) LIKE {sql_literal('%' + str(discovery_facility).lower() + '%')}")
                applied.append(f"disc_facility contains '{discovery_facility}'")
            if discovery_method:
                where.append(f"LOWER(discoverymethod) = {sql_literal(str(discovery_method).lower())}")
                applied.append(f"discoverymethod = '{discovery_method}'")
            for col, op, val in parse_cuts(cuts):
                where.append(_cut_sql(cols.sql(col), op, val))
                applied.append(f"{cols.resolve(col)} {op} {val}")
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""
            cres = self.client.query("exoplanet", f"SELECT COUNT(*) AS n FROM pscomppars{where_sql}", maxrec=5)
            results.append(cres)
            n = int((cres.rows[0] or {}).get("n") or 0) if cres.rows else 0
            rows: List[Dict[str, Any]] = []
            out_cols = [cols.resolve(c) for c in (columns or [])] or [c for c in EXO_CENSUS_COLUMNS
                                                                        if c.lower() in cols.by_lower]
            for extra in [c for c, _, _ in parse_cuts(cuts)] + [EXO_TYPED_FILTERS[k][0] for k in (filters or {})
                                                                 if (filters or {}).get(k) not in (None, "")]:
                real = cols.resolve(extra)
                if real not in out_cols:
                    out_cols.append(real)
            lim = max(1, min(int(_float(limit) or 200), 2000))
            truncated = False
            if not count_only and n:
                order = cols.sql(order_by) if order_by else "pl_name"
                rres = self.client.query(
                    "exoplanet", f"SELECT TOP {lim} {', '.join(out_cols)} FROM pscomppars{where_sql} ORDER BY {order}",
                    maxrec=lim)
                results.append(rres)
                rows = rres.rows
                truncated = n > len(rows)
        except (TapQueryError, ValueError) as exc:
            msg = exc.message if isinstance(exc, TapQueryError) else str(exc)
            return _fail(msg, queries=[r.query for r in results], service="exoplanet")
        summary = {"n_planets": n, "table": "pscomppars (confirmed planets, one row per planet)",
                   "filters": applied, "listed": len(rows), "list_truncated": truncated,
                   "retrieved_utc": _utc_now()}
        if rows and len(rows) <= 60:
            summary["planets"] = [r.get("pl_name") for r in rows]
        warnings = ["planets with no value in a filtered column are excluded by that filter"] if applied else []
        if truncated:
            warnings.append(f"{n} planets match; {len(rows)} rows listed (the count covers all)")
        return {"success": True, "summary": summary, "rows": rows, "columns": out_cols, "warnings": warnings,
                "provenance": _merge_prov(results, {"table": "pscomppars"})}

    # ── SIMBAD ─────────────────────────────────────────────────────
    _SIMBAD_BASIC = ("b.oid, b.main_id, b.otype, b.ra, b.dec, b.rvz_redshift, b.rvz_radvel, b.rvz_type, "
                     "b.rvz_qual, b.rvz_bibcode, b.plx_value, b.plx_err, b.pmra, b.pmdec, b.sp_type, b.nbref")

    def simbad_lookup(self, identifiers: Sequence[str]) -> Dict[str, Any]:
        ids = [str(i).strip() for i in (identifiers or []) if str(i).strip()]
        if not ids:
            return _fail("give one or more identifiers", service="simbad")
        if len(ids) > 25:
            return _fail("at most 25 identifiers per call", service="simbad")
        results: List[TapResult] = []
        # One "=" query per identifier: SIMBAD normalises spacing and aliases
        # only for "=" (not IN), and reports the stored alias, not the input,
        # so a batched query could not be mapped back to what was asked.
        by_query: Dict[str, Dict[str, Any]] = {}
        try:
            for ident in ids:
                res = self.client.query(
                    "simbad",
                    f"SELECT TOP 1 {self._SIMBAD_BASIC} FROM ident AS i JOIN basic AS b ON i.oidref = b.oid "
                    f"WHERE i.id = {sql_literal(ident)}", maxrec=1)
                results.append(res)
                if res.rows:
                    by_query[_norm_id(ident)] = res.rows[0]
            oids = sorted({int(r["oid"]) for r in by_query.values() if r.get("oid") is not None})
            extra_ids: Dict[int, List[str]] = {}
            otype_desc: Dict[str, str] = {}
            if oids:
                ires = self.client.query(
                    "simbad",
                    f"SELECT oidref, id FROM ident WHERE oidref IN ({', '.join(str(o) for o in oids)}) AND "
                    "(id LIKE 'Gaia DR3 %' OR id LIKE 'NAME %' OR id LIKE '2MASS J%')", maxrec=500)
                results.append(ires)
                for r in ires.rows:
                    extra_ids.setdefault(int(r["oidref"]), []).append(str(r["id"]))
                otypes = sorted({str(r["otype"]) for r in by_query.values() if r.get("otype")})
                if otypes:
                    tres = self.client.query(
                        "simbad", f"SELECT otype, description FROM otypedef WHERE otype IN "
                                  f"({', '.join(sql_literal(o) for o in otypes)})", maxrec=100)
                    results.append(tres)
                    otype_desc = {str(r["otype"]): str(r["description"]) for r in tres.rows}
        except TapQueryError as exc:
            return _fail(exc.message, queries=[r.query for r in results] + [exc.query], service="simbad")
        objects = []
        for ident in ids:
            r = by_query.get(_norm_id(ident))
            if r is None:
                objects.append({"query": ident, "found": False,
                                "note": "SIMBAD has no object with this identifier (identifier not in the ident table)"})
                continue
            oid = int(r["oid"])
            others = extra_ids.get(oid, [])
            gaia = next((o for o in others if o.startswith("Gaia DR3 ")), None)
            objects.append({
                "query": ident, "found": True, "main_id": _squash(r.get("main_id")), "otype": r.get("otype"),
                "otype_description": otype_desc.get(str(r.get("otype"))), "ra": r.get("ra"), "dec": r.get("dec"),
                "redshift": r.get("rvz_redshift"), "radial_velocity_kms": r.get("rvz_radvel"),
                "velocity_type": r.get("rvz_type"), "redshift_quality": r.get("rvz_qual"),
                "redshift_bibcode": r.get("rvz_bibcode"), "parallax_mas": r.get("plx_value"),
                "parallax_err_mas": r.get("plx_err"), "pmra_masyr": r.get("pmra"), "pmdec_masyr": r.get("pmdec"),
                "sp_type": r.get("sp_type"), "n_references": r.get("nbref"),
                "gaia_dr3_id": gaia.replace("Gaia DR3 ", "") if gaia else None,
                "other_names": [_squash(o) for o in others if not o.startswith("Gaia DR3 ")][:5],
            })
        rows = [{k: v for k, v in o.items() if k not in ("other_names",)} for o in objects]
        summary = {"n_queried": len(ids), "n_found": sum(1 for o in objects if o["found"]), "objects": objects,
                   "retrieved_utc": _utc_now()}
        columns = ["query", "found", "main_id", "otype", "otype_description", "ra", "dec", "redshift",
                   "redshift_bibcode", "parallax_mas", "gaia_dr3_id"]
        return {"success": True, "summary": summary, "rows": rows, "columns": columns, "warnings": [],
                "provenance": _merge_prov(results, {"table": "basic, ident, otypedef"})}

    def simbad_cone(self, ra: Any, dec: Any, radius_arcsec: Any = 60, *, limit: Any = 50,
                    otype: Optional[str] = None) -> Dict[str, Any]:
        try:
            ra_f, dec_f = _validate_position(ra, dec)
        except ValueError as exc:
            return _fail(str(exc), service="simbad")
        radius = max(0.1, min(_float(radius_arcsec) or 60.0, 3600.0))
        lim = max(1, min(int(_float(limit) or 50), 500))
        where = [_cone("ra", "dec", ra_f, dec_f, radius / 3600.0)]
        if otype:
            where.append(f"otype = {sql_literal(str(otype).strip())}")
        where_sql = " AND ".join(where)
        results: List[TapResult] = []
        try:
            cres = self.client.query("simbad", f"SELECT COUNT(*) AS n FROM basic WHERE {where_sql}", maxrec=5)
            results.append(cres)
            rres = self.client.query(
                "simbad",
                f"SELECT TOP {lim} main_id, otype, ra, dec, rvz_redshift, sp_type, "
                f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra_f!r}, {dec_f!r})) * 3600.0 AS sep_arcsec "
                f"FROM basic WHERE {where_sql} ORDER BY sep_arcsec", maxrec=lim)
            results.append(rres)
        except TapQueryError as exc:
            return _fail(exc.message, queries=[r.query for r in results] + [exc.query], service="simbad")
        n = int((cres.rows[0] or {}).get("n") or 0) if cres.rows else 0
        rows = [{**r, "main_id": _squash(r.get("main_id"))} for r in rres.rows]
        summary = {"n_objects": n, "radius_arcsec": radius, "center": {"ra": ra_f, "dec": dec_f},
                   "nearest": rows[0] if rows else None, "listed": len(rows), "list_truncated": n > len(rows)}
        return {"success": True, "summary": summary, "rows": rows,
                "columns": ["main_id", "otype", "sep_arcsec", "ra", "dec", "rvz_redshift", "sp_type"],
                "warnings": [] if n else [f"SIMBAD has no object within {radius:g} arcsec (a real empty result)"],
                "provenance": _merge_prov(results, {"table": "basic"})}

    def simbad_positions(self, positions: Sequence[Any], radius_arcsec: Any = 5) -> Dict[str, Any]:
        parsed: List[Tuple[str, float, float]] = []
        for i, p in enumerate(positions or []):
            try:
                if isinstance(p, dict):
                    ra_v, dec_v = p.get("ra"), p.get("dec")
                    label = str(p.get("label") or p.get("name") or f"#{i + 1}")
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    ra_v, dec_v, label = p[0], p[1], (str(p[2]) if len(p) > 2 else f"#{i + 1}")
                else:
                    parts = re.split(r"[,\s]+", str(p).strip())
                    ra_v, dec_v, label = parts[0], parts[1], f"#{i + 1}"
                ra_f, dec_f = _validate_position(ra_v, dec_v)
            except (ValueError, IndexError, TypeError):
                return _fail(f"position {i + 1} ({p!r}) is not 'ra, dec' in decimal degrees", service="simbad")
            parsed.append((label, ra_f, dec_f))
        if not parsed:
            return _fail("give a list of positions", service="simbad")
        if len(parsed) > 50:
            return _fail("at most 50 positions per call", service="simbad")
        radius = max(0.1, min(_float(radius_arcsec) or 5.0, 600.0))
        results: List[TapResult] = []
        rows = []
        for label, ra_f, dec_f in parsed:
            sql = (f"SELECT TOP 3 main_id, otype, ra, dec, rvz_redshift, "
                   f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra_f!r}, {dec_f!r})) * 3600.0 AS sep_arcsec "
                   f"FROM basic WHERE {_cone('ra', 'dec', ra_f, dec_f, radius / 3600.0)} ORDER BY sep_arcsec")
            try:
                res = self.client.query("simbad", sql, maxrec=3)
            except TapQueryError as exc:
                rows.append({"input": label, "ra": ra_f, "dec": dec_f, "found": None, "error": exc.message})
                continue
            results.append(res)
            if res.rows:
                best = res.rows[0]
                rows.append({"input": label, "ra": ra_f, "dec": dec_f, "found": True,
                             "main_id": _squash(best.get("main_id")), "otype": best.get("otype"),
                             "sep_arcsec": round(float(best["sep_arcsec"]), 3), "redshift": best.get("rvz_redshift"),
                             "n_within_radius": len(res.rows)})
            else:
                rows.append({"input": label, "ra": ra_f, "dec": dec_f, "found": False, "main_id": None,
                             "note": f"no SIMBAD object within {radius:g} arcsec"})
        if not results:
            return _fail("every SIMBAD position query failed: " + "; ".join(r.get("error", "") for r in rows),
                         service="simbad")
        otypes = sorted({r["otype"] for r in rows if r.get("otype")})
        if otypes:
            try:
                tres = self.client.query("simbad", f"SELECT otype, description FROM otypedef WHERE otype IN "
                                                   f"({', '.join(sql_literal(o) for o in otypes)})", maxrec=100)
                desc = {str(r["otype"]): str(r["description"]) for r in tres.rows}
                for r in rows:
                    if r.get("otype"):
                        r["otype_description"] = desc.get(str(r["otype"]))
            except TapQueryError:
                pass
        errors = [r for r in rows if r.get("error")]
        summary = {"n_positions": len(rows), "n_with_match": sum(1 for r in rows if r.get("found")),
                   "radius_arcsec": radius, "matches": rows}
        warnings = [f"{len(errors)} position queries failed; see their error field"] if errors else []
        return {"success": True, "summary": summary, "rows": rows,
                "columns": ["input", "ra", "dec", "found", "main_id", "otype", "otype_description", "sep_arcsec"],
                "warnings": warnings, "provenance": _merge_prov(results, {"table": "basic"})}

    # ── Gaia ───────────────────────────────────────────────────────
    def gaia_source(self, *, source_id: Any = None, identifier: Optional[str] = None, ra: Any = None, dec: Any = None,
                    radius_arcsec: Any = 5) -> Dict[str, Any]:
        """One Gaia DR3 source by source_id, by a name SIMBAD knows, or nearest to a position."""
        results: List[TapResult] = []
        warnings: List[str] = []
        how = ""
        sid = None
        if source_id not in (None, ""):
            sid = re.sub(r"\D", "", str(source_id))
            if not sid:
                return _fail("source_id must be the numeric Gaia DR3 source_id", service="gaia")
            how = "source_id given"
        pm_info: Dict[str, Any] = {}
        if sid is None and identifier:
            lk = self.simbad_lookup([identifier])
            if not lk.get("success"):
                return lk
            results_prov = lk.get("provenance", {}).get("queries", [])
            obj = (lk["summary"]["objects"] or [{}])[0]
            if not obj.get("found"):
                return {"success": True, "summary": {"found": False, "query": identifier,
                                                     "note": "SIMBAD does not know this name, so no Gaia source "
                                                             "could be tied to it"},
                        "rows": [], "columns": [], "warnings": [], "provenance": {"service": "simbad",
                                                                                "queries": results_prov}}
            if obj.get("gaia_dr3_id"):
                sid = obj["gaia_dr3_id"]
                how = f"SIMBAD cross-identification of {identifier!r} ({obj['main_id']}) as Gaia DR3 {sid}"
            else:
                ra, dec = obj.get("ra"), obj.get("dec")
                pm_info = {"pmra": obj.get("pmra_masyr"), "pmdec": obj.get("pmdec_masyr")}
                how = f"no Gaia DR3 id in SIMBAD for {obj['main_id']}; nearest source to its position"
        cols = ", ".join(GAIA_SOURCE_COLUMNS)
        try:
            if sid is not None:
                res = self.client.query("gaia", f"SELECT {cols} FROM {GAIA_SOURCE_TABLE} WHERE source_id = {int(sid)}",
                                        maxrec=5)
                results.append(res)
                rows = res.rows
            else:
                if ra is None or dec is None:
                    return _fail("give source_id, a target name, or ra/dec", service="gaia")
                ra_f, dec_f = _validate_position(ra, dec)
                pmra, pmdec = _float(pm_info.get("pmra")), _float(pm_info.get("pmdec"))
                if pmra is not None and pmdec is not None:
                    ra_f, dec_f = _propagate(ra_f, dec_f, pmra, pmdec, 2000.0, GAIA_EPOCH)
                    warnings.append(f"position propagated from J2000 to the Gaia epoch {GAIA_EPOCH} with "
                                    f"pm=({pmra:.1f}, {pmdec:.1f}) mas/yr")
                else:
                    warnings.append("Gaia DR3 positions are at epoch 2016.0: a fast-moving star can sit arcseconds "
                                    "(or arcminutes) from its J2000 position; give a name or source_id to be safe")
                radius = max(0.5, min(_float(radius_arcsec) or 5.0, 120.0))
                res = self.client.query(
                    "gaia",
                    f"SELECT TOP 5 {cols}, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra_f!r}, {dec_f!r})) * 3600.0 "
                    f"AS sep_arcsec FROM {GAIA_SOURCE_TABLE} WHERE {_cone('ra', 'dec', ra_f, dec_f, radius / 3600.0)} "
                    "ORDER BY sep_arcsec", maxrec=5)
                results.append(res)
                rows = res.rows
                how = how or f"nearest Gaia DR3 source within {radius:g} arcsec"
        except (TapQueryError, ValueError) as exc:
            msg = exc.message if isinstance(exc, TapQueryError) else str(exc)
            return _fail(msg, queries=[r.query for r in results], service="gaia")
        # The Gaia archive returns some names upper-case (SOURCE_ID); normalise.
        rows = [{str(k).lower(): v for k, v in r.items()} for r in rows]
        if not rows:
            return {"success": True, "summary": {"found": False, "method": how,
                                                 "note": "no Gaia DR3 source matched"},
                    "rows": [], "columns": GAIA_SOURCE_COLUMNS, "warnings": warnings,
                    "provenance": _merge_prov(results, {"table": GAIA_SOURCE_TABLE})}
        best = dict(rows[0])
        foe = _float(best.get("phot_g_mean_flux_over_error"))
        if foe:
            best["phot_g_mean_mag_error"] = round(math.hypot(1.0857 / foe, 0.0028), 4)
        summary = {"found": True, "method": how, "source": best, "n_candidates": len(rows),
                   "note": "G-band error combines the flux S/N with the 2.8 mmag DR3 zero-point uncertainty"}
        return {"success": True, "summary": summary, "rows": rows, "columns": list(rows[0].keys()),
                "warnings": warnings, "provenance": _merge_prov(results, {"table": GAIA_SOURCE_TABLE})}

    def gaia_variability(self, table: str, source_id: Any) -> Dict[str, Any]:
        full = GAIA_VARI_TABLES.get(str(table or "").strip().lower())
        if not full:
            return _fail(f"unknown Gaia variability table {table!r}; use one of "
                         f"{', '.join(sorted(k for k in GAIA_VARI_TABLES if not k.startswith('vari_')))}",
                         service="gaia")
        sid = re.sub(r"\D", "", str(source_id or ""))
        if not sid:
            return _fail("a numeric Gaia DR3 source_id is needed (use mode='source' with the target name first)",
                         service="gaia")
        results: List[TapResult] = []
        try:
            res = self.client.query("gaia", f"SELECT * FROM {full} WHERE source_id = {int(sid)}", maxrec=5)
            results.append(res)
            summ = None
            if full != "gaiadr3.vari_summary":
                sres = self.client.query(
                    "gaia", f"SELECT source_id, num_selected_g_fov, mean_mag_g_fov, median_mag_g_fov, "
                            f"range_mag_g_fov, std_dev_mag_g_fov FROM gaiadr3.vari_summary WHERE source_id = {int(sid)}",
                    maxrec=5)
                results.append(sres)
                summ = sres.rows[0] if sres.rows else None
        except TapQueryError as exc:
            return _fail(exc.message, queries=[r.query for r in results] + [exc.query], service="gaia")
        rec = {str(k).lower(): v for k, v in res.rows[0].items()} if res.rows else None
        if summ:
            summ = {str(k).lower(): v for k, v in summ.items()}
        compact = {k: v for k, v in (rec or {}).items() if v is not None}
        summary = {"table": full, "source_id": sid, "found": rec is not None, "record": compact,
                   "vari_summary": summ}
        warnings = [] if rec else [f"source {sid} has no row in {full} (it was not classified into this "
                                   "variability class in DR3)"]
        return {"success": True, "summary": summary, "rows": res.rows, "columns": res.columns[:40],
                "warnings": warnings, "provenance": _merge_prov(results, {"table": full})}

    # ── generic catalogue query ────────────────────────────────────
    def catalog_query(self, service: str, table: str, *, ra: Any = None, dec: Any = None, radius_arcsec: Any = None,
                      columns: Optional[Sequence[str]] = None, cuts: Any = None, count_only: bool = False,
                      limit: Any = 100, order_by: Optional[str] = None, descending: bool = False) -> Dict[str, Any]:
        results: List[TapResult] = []
        warnings: List[str] = []
        try:
            svc = self.client.service(service)
            cols = self._columns(svc.key, table)
            tsql = quote_table(table, force=svc.quote_all)
            where: List[str] = []
            center = None
            ra_col, dec_col = radec_columns(cols.cols)
            if ra is not None and dec is not None:
                ra_f, dec_f = _validate_position(ra, dec)
                if not ra_col or not dec_col:
                    raise ValueError("this table has no RA/Dec columns that could be identified; query it with cuts "
                                     "only")
                radius = _float(radius_arcsec)
                if radius is None or radius <= 0:
                    raise ValueError("radius_arcsec is required with a position")
                if radius > 5 * 3600:
                    raise ValueError("radius_arcsec is capped at 18000 (5 degrees)")
                where.append(_cone(cols.sql(ra_col), cols.sql(dec_col), ra_f, dec_f, radius / 3600.0))
                center = (ra_f, dec_f, radius)
            applied = []
            for col, op, val in parse_cuts(cuts):
                where.append(_cut_sql(cols.sql(col), op, val))
                applied.append(f"{cols.resolve(col)} {op} {val}")
            if not where and not count_only:
                warnings.append("no position or cuts: rows are the first ones the archive returns, not a selection")
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""
            cres = self.client.query(svc.key, f"SELECT COUNT(*) AS n FROM {tsql}{where_sql}", maxrec=5)
            results.append(cres)
            n = int((cres.rows[0] or {}).get("n") or 0) if cres.rows else 0
            rows: List[Dict[str, Any]] = []
            out_cols: List[str] = []
            lim = max(1, min(int(_float(limit) or 100), 5000))
            if not count_only and n:
                if columns:
                    out_cols = [cols.resolve(c) for c in columns]
                else:
                    out_cols = cols.names()[:15]
                for extra in [ra_col, dec_col] + [c for c, _, _ in parse_cuts(cuts)]:
                    if extra:
                        real = cols.resolve(extra)
                        if real not in out_cols:
                            out_cols.append(real)
                order = ""
                if order_by:
                    order = f" ORDER BY {cols.sql(order_by)}{' DESC' if descending else ''}"
                elif center:
                    # TOP without an order is an arbitrary subset: order by distance so
                    # the listed rows really are the nearest ones.
                    order = (f" ORDER BY DISTANCE(POINT('ICRS', {cols.sql(ra_col)}, {cols.sql(dec_col)}), "
                             f"POINT('ICRS', {center[0]!r}, {center[1]!r}))")
                select = (f"SELECT TOP {lim} {', '.join(quote_ident(c, force=svc.quote_all) for c in out_cols)} "
                          f"FROM {tsql}{where_sql}")
                try:
                    rres = self.client.query(svc.key, select + order, maxrec=lim)
                except TapQueryError:
                    if not (center and not order_by):
                        raise
                    rres = self.client.query(svc.key, select, maxrec=lim)
                    if n > lim:
                        warnings.append("the archive refused ORDER BY DISTANCE, so the listed rows are not "
                                        "necessarily the nearest ones")
                results.append(rres)
                rows = rres.rows
                if center and ra_col and dec_col:
                    for r in rows:
                        rr, dd = _float(r.get(ra_col)), _float(r.get(dec_col))
                        r["sep_arcsec"] = round(_sep_arcsec(center[0], center[1], rr, dd), 3) if (
                            rr is not None and dd is not None) else None
                    if not order_by:
                        rows.sort(key=lambda r: (r["sep_arcsec"] is None, r["sep_arcsec"] or 0.0))
                    out_cols = ["sep_arcsec"] + out_cols
        except (TapQueryError, ValueError) as exc:
            msg = exc.message if isinstance(exc, TapQueryError) else str(exc)
            return _fail(msg, queries=[r.query for r in results], service=str(service))
        summary: Dict[str, Any] = {"service": KNOWN_SERVICES.get(svc.key, svc).label, "table": table.strip('"'),
                                   "n_rows_matching": n, "cuts": applied, "listed": len(rows),
                                   "list_truncated": n > len(rows) and not count_only}
        if center:
            summary["cone"] = {"ra": center[0], "dec": center[1], "radius_arcsec": center[2],
                               "ra_column": ra_col, "dec_column": dec_col}
        if rows and center:
            summary["nearest"] = rows[0]
        if n > len(rows) and rows:
            warnings.append(f"{n} rows match; {len(rows)} listed (the count is the archive's own COUNT)")
        return {"success": True, "summary": summary, "rows": rows, "columns": out_cols, "warnings": warnings,
                "provenance": _merge_prov(results, {"table": table})}

    def catalog_find(self, keywords: str, *, services: Sequence[str] = ("vizier", "heasarc", "irsa"),
                     limit: Any = 12) -> Dict[str, Any]:
        text = str(keywords or "").strip()
        if not text:
            return _fail("give keywords (survey or catalogue name, author, topic)")
        lim = max(1, min(int(_float(limit) or 12), 40))
        words = [w for w in re.split(r"\s+", text) if len(w) >= 2][:6]
        found: List[Dict[str, Any]] = []
        results: List[TapResult] = []
        errors: List[str] = []
        for service in services:
            key = str(service).lower()
            try:
                if key == "vizier":
                    found.extend(self._vizier_catalogs(text, lim, results))
                elif key in ("heasarc", "irsa"):
                    conds = " AND ".join(f"LOWER(description) LIKE {sql_literal('%' + w.lower() + '%')}" for w in words)
                    name_cond = " AND ".join(f"LOWER(table_name) LIKE {sql_literal('%' + w.lower() + '%')}"
                                             for w in words)
                    res = self.client.query(key, f"SELECT TOP {lim} table_name, description FROM TAP_SCHEMA.tables "
                                                 f"WHERE ({conds}) OR ({name_cond})", maxrec=lim)
                    results.append(res)
                    for r in res.rows:
                        found.append({"service": key, "table": str(r.get("table_name") or "").strip('"'),
                                      "description": _squash(r.get("description"))[:200]})
            except TapQueryError as exc:
                errors.append(f"{key}: {exc.message}")
            except Exception as exc:  # the VizieR keyword service is not TAP
                errors.append(f"{key}: {type(exc).__name__}: {exc}")
        if not found and errors:
            return _fail("catalogue search failed: " + " | ".join(errors))
        summary = {"keywords": text, "n_found": len(found), "catalogues": found[: lim * len(services)],
                   "next_step": "query one with catalog_query(service, table, ...) or catalog_crossmatch"}
        warnings = [f"search failed for {e}" for e in errors]
        prov = _merge_prov(results) if results else {"service": "vizier", "queries": [], "retrieved_utc": _utc_now()}
        return {"success": True, "summary": summary, "rows": found, "columns": ["service", "table", "description"],
                "warnings": warnings, "provenance": prov}

    def _vizier_catalogs(self, text: str, lim: int, results: List[TapResult]) -> List[Dict[str, Any]]:
        cats = (self._vizier_find or _astroquery_vizier_find)(text, lim)
        if not cats:
            return []
        prefixes = [str(c["catalog"]).strip("/") for c in cats][:lim]
        # A catalogue id is either a folder of tables ("I/259" -> "I/259/tyc2") or,
        # for single-table catalogues, a table name itself.
        cond = " OR ".join(f"table_name LIKE {sql_literal(chr(34) + p + '/%')} OR table_name = "
                           f"{sql_literal(chr(34) + p + chr(34))}" for p in prefixes)
        res = self.client.query("vizier", f"SELECT table_name, description FROM TAP_SCHEMA.tables WHERE {cond}",
                                maxrec=200)
        results.append(res)
        tables: Dict[str, List[Tuple[str, str]]] = {}
        for r in res.rows:
            name = str(r.get("table_name") or "").strip('"')
            parent = name if name in prefixes else name.rsplit("/", 1)[0]
            tables.setdefault(parent, []).append((name, _squash(r.get("description"))))
        out = []
        for c in cats[:lim]:
            cat = str(c["catalog"]).strip("/")
            tabs = tables.get(cat, [])
            if not tabs:
                out.append({"service": "vizier", "catalog": cat, "table": None,
                            "description": str(c.get("description", ""))[:200],
                            "note": "no TAP table listed for this catalogue"})
            for name, desc in tabs[:6]:
                out.append({"service": "vizier", "catalog": cat, "table": name,
                            "description": (str(c.get("description", "")) + " | " + desc)[:220]})
        return out

    # ── cross-match ────────────────────────────────────────────────
    def catalog_crossmatch(self, left: Dict[str, Any], right: Dict[str, Any], *, ra: Any, dec: Any,
                           radius_arcsec: Any, match_radius_arcsec: Any, max_rows_per_side: Any = 20000
                           ) -> Dict[str, Any]:
        try:
            ra_f, dec_f = _validate_position(ra, dec)
        except ValueError as exc:
            return _fail(str(exc))
        radius = _float(radius_arcsec)
        match_r = _float(match_radius_arcsec)
        if radius is None or radius <= 0 or radius > 3 * 3600:
            return _fail("radius_arcsec (the search cone) must be between 0 and 10800 (3 degrees)")
        if match_r is None or match_r <= 0 or match_r > 600:
            return _fail("match_radius_arcsec must be between 0 and 600")
        cap = max(100, min(int(_float(max_rows_per_side) or 20000), 50000))
        results: List[TapResult] = []
        sides = []
        warnings: List[str] = []
        for label, spec in (("left", left), ("right", right)):
            service = str(spec.get("service") or "").lower()
            table = str(spec.get("table") or "").strip()
            try:
                svc = self.client.service(service)
                cols = self._columns(svc.key, table)
                ra_col, dec_col = radec_columns(cols.cols)
                if not ra_col or not dec_col:
                    raise ValueError(f"{label} table {table!r} has no identifiable RA/Dec columns")
                where = [_cone(cols.sql(ra_col), cols.sql(dec_col), ra_f, dec_f, radius / 3600.0)]
                applied = []
                for col, op, val in parse_cuts(spec.get("cuts")):
                    where.append(_cut_sql(cols.sql(col), op, val))
                    applied.append(f"{cols.resolve(col)} {op} {val}")
                tsql = quote_table(table, force=svc.quote_all)
                where_sql = " AND ".join(where)
                cres = self.client.query(svc.key, f"SELECT COUNT(*) AS n FROM {tsql} WHERE {where_sql}", maxrec=5)
                results.append(cres)
                n_total = int((cres.rows[0] or {}).get("n") or 0) if cres.rows else 0
                pres = self.client.query(
                    svc.key, f"SELECT TOP {cap} {cols.sql(ra_col)}, {cols.sql(dec_col)} FROM {tsql} WHERE {where_sql}",
                    maxrec=cap)
                results.append(pres)
            except (TapQueryError, ValueError) as exc:
                msg = exc.message if isinstance(exc, TapQueryError) else str(exc)
                return _fail(f"{label} side ({service}:{table}): {msg}", queries=[r.query for r in results])
            coords = [(float(r[ra_col]), float(r[dec_col])) for r in pres.rows
                      if _float(r.get(ra_col)) is not None and _float(r.get(dec_col)) is not None]
            if n_total > len(coords):
                warnings.append(f"{label} side has {n_total} rows in the cone but only {len(coords)} were pulled "
                                f"(cap {cap}); matched counts cover the pulled rows only")
            if svc.key == "gaia" or "gaia" in table.lower():
                warnings.append("Gaia positions are epoch 2016.0; stars with large proper motion may fall outside a "
                                "small match radius against a J2000 catalogue")
            sides.append({"service": KNOWN_SERVICES[svc.key].label, "table": table.strip('"'), "cuts": applied,
                          "n_in_cone": n_total, "n_pulled": len(coords), "coords": coords})
        stats = _match(sides[0]["coords"], sides[1]["coords"], match_r)
        left_s, right_s = sides
        summary = {
            "cone": {"ra": ra_f, "dec": dec_f, "radius_arcsec": radius},
            "match_radius_arcsec": match_r,
            "left": {k: v for k, v in left_s.items() if k != "coords"},
            "right": {k: v for k, v in right_s.items() if k != "coords"},
            "n_left_with_match": stats["n_left_matched"],
            "n_right_with_match": stats["n_right_matched"],
            "n_pairs": stats["n_pairs"],
            "fraction_left_matched": (round(stats["n_left_matched"] / left_s["n_pulled"], 4)
                                      if left_s["n_pulled"] else None),
            "median_sep_arcsec": stats["median_sep"], "max_sep_arcsec": stats["max_sep"],
            "complete": left_s["n_in_cone"] == left_s["n_pulled"] and right_s["n_in_cone"] == right_s["n_pulled"],
            "method": "both sides selected server-side in the cone (with cuts), then positional match by nearest "
                      "neighbour within the match radius (astropy); counts are distinct sources",
        }
        rows = stats["sample_pairs"]
        return {"success": True, "summary": summary, "rows": rows,
                "columns": ["left_ra", "left_dec", "right_ra", "right_dec", "sep_arcsec"],
                "warnings": warnings, "provenance": _merge_prov(results)}

    # ── TNS ────────────────────────────────────────────────────────
    def tns_object(self, name: str) -> Dict[str, Any]:
        raw = str(name or "").strip()
        m = re.match(r"^(?:SN|AT|TDE|FRB|KN)?\s*((?:19|20)\d{2}[a-z]{1,4})$", raw, re.I)
        if not m:
            return _fail(f"{raw!r} is not a TNS designation (expected e.g. '2024abc' or 'SN 2024abc')", service="tns")
        designation = m.group(1).lower()
        url = f"https://www.wis-tns.org/object/{designation}"
        try:
            resp = self._get(url, timeout=30.0, headers={"User-Agent": "Mozilla/5.0 (Quasar research assistant)"})
        except Exception as exc:
            return _fail(f"TNS request failed: {type(exc).__name__}: {exc}", service="tns")
        status = getattr(resp, "status_code", 200)
        text = getattr(resp, "text", "") or ""
        prov = {"service": "tns", "endpoint": url, "query": f"GET {url}", "queries": [f"GET {url}"],
                "retrieved_utc": _utc_now()}
        if status == 404 or "Page not found" in text[:5000]:
            return {"success": True, "summary": {"found": False, "designation": designation,
                                                 "note": "TNS has no object with this designation"},
                    "rows": [], "columns": [], "warnings": [], "provenance": prov}
        if int(status) >= 400:
            return {"success": False, "error": f"TNS returned HTTP {status}", "provenance": prov}
        fields = parse_tns_fields(text)
        if not fields:
            return {"success": False, "error": "TNS page had no recognisable object fields (layout changed or "
                                               "blocked)", "provenance": prov}
        radec = fields.get("radec", "")
        dec_parts = re.findall(r"[-+]?\d+\.\d+", radec.split("|")[-1]) if radec else []
        summary = {
            "found": True, "designation": designation, "iau_name": fields.get("iauname"),
            "type": fields.get("type") or None, "redshift": fields.get("redshift") or None,
            "host": fields.get("hostname") or None, "host_redshift": fields.get("host_redshift") or None,
            "discovery_date_utc": fields.get("discoverydate") or None,
            "discovery_mag": fields.get("discoverymag") or None,
            "discovery_filter": fields.get("discmagfilter") or fields.get("discoverymagfilter") or None,
            "reporting_group": fields.get("reporting_group") or fields.get("reportinggroup") or None,
            "discovery_data_source": fields.get("discovery_data_source") or fields.get("source_group") or None,
            "internal_names": fields.get("internal_names") or None,
            "radec_sexagesimal": radec.split("|")[0].strip() if radec else None,
            "ra_deg": _float(dec_parts[0]) if len(dec_parts) >= 2 else None,
            "dec_deg": _float(dec_parts[1]) if len(dec_parts) >= 2 else None,
            "classified": bool(fields.get("type")),
            "all_fields": fields,
        }
        rows = [{"field": k, "value": v} for k, v in fields.items()]
        return {"success": True, "summary": summary, "rows": rows, "columns": ["field", "value"], "warnings": [],
                "provenance": prov}

    # ── ALeRCE ─────────────────────────────────────────────────────
    ALERCE_BASE = "https://api.alerce.online/ztf/v1"

    def ztf_object(self, *, oid: Optional[str] = None, ra: Any = None, dec: Any = None,
                   radius_arcsec: Any = 5) -> Dict[str, Any]:
        queries: List[str] = []
        warnings: List[str] = []
        oid_s = str(oid or "").strip()
        try:
            if not oid_s:
                ra_f, dec_f = _validate_position(ra, dec)
                radius = max(0.5, min(_float(radius_arcsec) or 5.0, 60.0))
                url = f"{self.ALERCE_BASE}/objects"
                params = {"ra": ra_f, "dec": dec_f, "radius": radius, "page_size": 20}
                queries.append(f"GET {url}?ra={ra_f}&dec={dec_f}&radius={radius}")
                resp = self._get(url, params=params, timeout=30.0)
                if getattr(resp, "status_code", 200) >= 400:
                    raise RuntimeError(f"ALeRCE cone returned HTTP {resp.status_code}")
                items = (resp.json() or {}).get("items") or []
                if not items:
                    return {"success": True, "summary": {"found": False, "radius_arcsec": radius,
                                                         "note": "no ZTF alert object within the radius"},
                            "rows": [], "columns": [], "warnings": [],
                            "provenance": {"service": "alerce", "query": queries[0], "queries": queries}}
                for it in items:
                    it["_sep"] = _sep_arcsec(ra_f, dec_f, float(it.get("meanra")), float(it.get("meandec"))) if (
                        it.get("meanra") is not None and it.get("meandec") is not None) else 1e9
                items.sort(key=lambda it: it["_sep"])
                oid_s = str(items[0]["oid"])
                if len(items) > 1:
                    warnings.append(f"{len(items)} ZTF objects within {radius:g} arcsec; the nearest ({oid_s}, "
                                    f"{items[0]['_sep']:.2f} arcsec) is described; others: "
                                    f"{', '.join(str(i['oid']) for i in items[1:6])}")
            if not re.fullmatch(r"ZTF\d{2}[a-z]{7}", oid_s):
                return _fail(f"{oid_s!r} is not a ZTF object id (ZTFyyxxxxxxx)", service="alerce")
            url = f"{self.ALERCE_BASE}/objects/{oid_s}"
            queries.append(f"GET {url}")
            resp = self._get(url, timeout=30.0)
            if getattr(resp, "status_code", 200) == 404:
                return {"success": True, "summary": {"found": False, "oid": oid_s,
                                                     "note": "ALeRCE has no object with this id"},
                        "rows": [], "columns": [], "warnings": [],
                        "provenance": {"service": "alerce", "query": queries[-1], "queries": queries}}
            if getattr(resp, "status_code", 200) >= 400:
                raise RuntimeError(f"ALeRCE object returned HTTP {resp.status_code}")
            obj = resp.json() or {}
            classes: Dict[str, Any] = {}
            try:
                queries.append(f"GET {url}/probabilities")
                presp = self._get(url + "/probabilities", timeout=20.0)
                if getattr(presp, "status_code", 200) < 400:
                    for p in presp.json() or []:
                        if p.get("ranking") == 1:
                            classes[str(p.get("classifier_name"))] = {"class": p.get("class_name"),
                                                                       "probability": p.get("probability")}
            except Exception:
                warnings.append("classifier probabilities could not be fetched")
        except Exception as exc:
            return {"success": False, "error": f"ALeRCE request failed: {type(exc).__name__}: {exc}",
                    "provenance": {"service": "alerce", "queries": queries}}
        summary = {
            "found": True, "oid": oid_s, "n_detections": obj.get("ndet"),
            "first_detection_mjd": obj.get("firstmjd"), "first_detection_utc": mjd_to_iso(obj.get("firstmjd")),
            "last_detection_mjd": obj.get("lastmjd"), "last_detection_utc": mjd_to_iso(obj.get("lastmjd")),
            "span_days": obj.get("deltajd"), "mean_ra": obj.get("meanra"), "mean_dec": obj.get("meandec"),
            "top_classes": classes, "stellar": obj.get("stellar"),
            "note": "n_detections counts this object's alert detections (all bands)",
        }
        rows = [{"quantity": k, "value": v} for k, v in summary.items() if k not in ("top_classes", "note")]
        return {"success": True, "summary": summary, "rows": rows, "columns": ["quantity", "value"],
                "warnings": warnings,
                "provenance": {"service": "alerce", "endpoint": self.ALERCE_BASE, "query": queries[-1],
                               "queries": queries, "retrieved_utc": _utc_now()}}


# ── module helpers ─────────────────────────────────────────────────────


def _norm_id(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _squash(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _strip_html(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _squash(_html.unescape(re.sub(r"<[^>]+>", " ", str(value)))) or None


def _propagate(ra: float, dec: float, pmra: float, pmdec: float, from_year: float, to_year: float) -> Tuple[float, float]:
    """Linear proper-motion propagation (pmra includes cos(dec)); good to well under an arcsecond over decades."""
    dt = to_year - from_year
    dec_new = dec + (pmdec * dt) / 3.6e6
    cosd = math.cos(math.radians(dec)) or 1e-9
    ra_new = (ra + (pmra * dt) / 3.6e6 / cosd) % 360.0
    return ra_new, max(-90.0, min(90.0, dec_new))


def _match(left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]], radius_arcsec: float
           ) -> Dict[str, Any]:
    empty = {"n_left_matched": 0, "n_right_matched": 0, "n_pairs": 0, "median_sep": None, "max_sep": None,
             "sample_pairs": []}
    if not left or not right:
        return empty
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    lc = SkyCoord(ra=np.array([p[0] for p in left]) * u.deg, dec=np.array([p[1] for p in left]) * u.deg)
    rc = SkyCoord(ra=np.array([p[0] for p in right]) * u.deg, dec=np.array([p[1] for p in right]) * u.deg)
    idx_l, idx_r, sep, _ = rc.search_around_sky(lc, radius_arcsec * u.arcsec)
    if len(idx_l) == 0:
        return empty
    seps = sep.to(u.arcsec).value
    # nearest right neighbour per left source
    best: Dict[int, Tuple[int, float]] = {}
    for li, ri, s in zip(idx_l.tolist(), idx_r.tolist(), seps.tolist()):
        if li not in best or s < best[li][1]:
            best[li] = (ri, s)
    nn = sorted(s for _, s in best.values())
    pairs = sorted(best.items(), key=lambda kv: kv[1][1])[:10]
    return {
        "n_left_matched": len(best),
        "n_right_matched": len(set(idx_r.tolist())),
        "n_pairs": int(len(idx_l)),
        "median_sep": round(float(np.median(nn)), 3),
        "max_sep": round(float(max(nn)), 3),
        "sample_pairs": [{"left_ra": left[li][0], "left_dec": left[li][1], "right_ra": right[ri][0],
                          "right_dec": right[ri][1], "sep_arcsec": round(s, 3)} for li, (ri, s) in pairs],
    }


def parse_tns_fields(page: str) -> Dict[str, str]:
    """Every labelled field of a public TNS object page: {field-name: text}."""
    out: Dict[str, str] = {}
    for m in re.finditer(r'<div class="field field-([a-z0-9_]+)"[^>]*>(.*?)</div>\s*</div>', page, re.S):
        key, body = m.group(1), m.group(2)
        body = re.sub(r'<span class="name">.*?</span>', "", body, count=1, flags=re.S)
        if key == "radec":
            main = re.search(r'<div class="value">(.*?)</div>', body, re.S) or re.search(r"<b>(.*?)</b>", body, re.S)
            alt = re.search(r'<div class="alter-value">(.*?)(?:</div>|$)', body, re.S)
            val = _strip_html(main.group(1)) if main else ""
            if alt:
                val = f"{val} | {_strip_html(alt.group(1))}"
        else:
            val = _strip_html(body) or ""
        if key not in out:
            out[key] = val
    return out


def _astroquery_vizier_find(text: str, lim: int) -> List[Dict[str, Any]]:
    from astroquery.vizier import Vizier

    cats = Vizier(row_limit=1).find_catalogs(text, max_catalogs=lim)
    return [{"catalog": k, "description": getattr(v, "description", "")} for k, v in list(cats.items())[:lim]]


# ── ADS (fielded search, no helper LLM) ────────────────────────────────

ADS_FIELDS = "bibcode,title,author,year,pub,doi,citation_count,property,pubdate"
ADS_SORTS = {"date": "date desc", "date desc": "date desc", "date asc": "date asc", "oldest": "date asc",
             "citations": "citation_count desc", "citation_count desc": "citation_count desc",
             "relevance": "score desc", "score desc": "score desc"}


def _ads_phrase(value: str) -> str:
    return '"' + str(value).replace('"', " ").strip() + '"'


def build_ads_query(*, author: Any = None, first_author: Optional[str] = None, bibstem: Any = None,
                    year_from: Any = None, year_to: Any = None, title_words: Optional[str] = None,
                    abstract_words: Optional[str] = None, full_text: Optional[str] = None,
                    doi: Optional[str] = None, bibcode: Optional[str] = None, refereed: Optional[bool] = None,
                    extra_query: Optional[str] = None) -> Tuple[str, List[str]]:
    """Return (q, fq) for the ADS search API from structured fields.

    Words in title/abstract/full text are AND-ed (each word must appear), not
    quoted as one phrase; quote a phrase yourself inside the string to keep it.
    """
    parts: List[str] = []
    authors = author if isinstance(author, (list, tuple)) else ([author] if author else [])
    for a in authors:
        if str(a).strip():
            parts.append(f"author:{_ads_phrase(a)}")
    if first_author:
        parts.append(f"first_author:{_ads_phrase(first_author)}")
    stems = bibstem if isinstance(bibstem, (list, tuple)) else ([bibstem] if bibstem else [])
    stems = [str(b).strip() for b in stems if str(b).strip()]
    if stems:
        parts.append("bibstem:(" + " OR ".join(_ads_phrase(b) for b in stems) + ")")
    y0, y1 = (str(year_from).strip() if year_from else ""), (str(year_to).strip() if year_to else "")
    for y in (y0, y1):
        if y and not re.fullmatch(r"\d{4}", y):
            raise ValueError("years must be four digits")
    if y0 or y1:
        parts.append(f"year:[{y0 or '*'} TO {y1 or '*'}]")

    def words(field: str, text: Optional[str]) -> None:
        if not text or not str(text).strip():
            return
        tokens = re.findall(r'"[^"]+"|\S+', str(text))
        clean = [t if t.startswith('"') else re.sub(r"[^\w\-.+]", "", t) for t in tokens]
        clean = [t for t in clean if t and t.upper() not in ("AND", "OR", "NOT")]
        if clean:
            parts.append(f"{field}:(" + " AND ".join(clean) + ")")

    words("title", title_words)
    words("abs", abstract_words)
    words("full", full_text)
    if doi:
        parts.append(f"doi:{_ads_phrase(doi)}")
    if bibcode:
        parts.append(f"bibcode:{_ads_phrase(bibcode)}")
    if extra_query and str(extra_query).strip():
        parts.append(f"({str(extra_query).strip()})")
    if not parts:
        raise ValueError("give at least one search field (author, bibstem, years, title/abstract words, doi, bibcode)")
    fq = ["property:refereed"] if refereed else []
    return " AND ".join(parts), fq
