# services/search.py
"""
Search Service — Facade over the archive clients for all archive searches.

CALLED BY: capabilities/alma.py (the migrated search_by_target /
           search_by_position / ... capabilities; injected as the
           "search_service" CallContext service by core/agent.py)
CALLS:     integrations/alminer_client.py (ALminerClient)  — ALMA (default)
           integrations/tap.py (NRAOTapClient)             — VLA/VLBA/GBT

DATA FLOW:
    SearchByTarget.run(...) → SearchService.search_by_target(name)
    → ALminerClient.search_by_target(name) → alminer.target(name)
    → Returns pd.DataFrame of matching observations

    SearchByTarget.run(..., facility="VLA")
    → NRAOTapClient.search_vla_vlba(name, instruments=["VLA", "EVLA", "JVLA"])
    → Returns pd.DataFrame of matching NRAO archive observations

Also provides: plot generation, data download, line coverage checks,
and catalog search — all delegated to ALminerClient.
"""

import logging
import re
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime, timedelta
from integrations.alminer_client import ALminerClient

logger = logging.getLogger(__name__)

# Facilities served by the NRAO TAP archive (data-query.nrao.edu)
NRAO_FACILITIES = {"VLA", "VLBA", "EVLA", "JVLA", "GBT"}


def _errored_frame(error: str) -> pd.DataFrame:
    """An empty DataFrame that carries an archive-error marker in ``.attrs``.

    Backward compatible: ``.empty`` / ``len()`` still report the frame as empty,
    so existing callers that only check ``.empty`` are unaffected. Callers that
    need to tell "archive unavailable / query failed" apart from "no matching
    data" can read ``df.attrs.get("quasar_error")``. This keeps a failed search
    from being silently reported as a successful zero-row result. (C3)
    """
    df = pd.DataFrame()
    df.attrs["quasar_error"] = error
    return df


_MJD_EPOCH = datetime(1858, 11, 17)
_DATE_TOKEN_RE = re.compile(r"\b(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?\b")


def _to_mjd(value: datetime) -> float:
    delta = value - _MJD_EPOCH
    return delta.days + delta.seconds / 86400.0


def parse_date_range(date_range: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse a user date range into an (mjd_start, mjd_end) window.

    Accepts '2019-01-01 to 2020-06-30', '2019-01-01..2020-06-30', '2019-2021',
    '2019', 'after 2020-03', 'since 2021', 'before 2018'. Returns None when no
    date token is present. Bounds are inclusive; a bare year/month spans the
    whole year/month. ObsCore t_min/t_max are MJD (skill guardrail 2).
    """
    text = str(date_range or "").strip()
    if not text:
        return None
    tokens = _DATE_TOKEN_RE.findall(text)
    if not tokens:
        return None

    def _start(tok):
        y, m, d = int(tok[0]), int(tok[1] or 1), int(tok[2] or 1)
        return datetime(y, m, d)

    def _end(tok):
        y, m, d = int(tok[0]), tok[1], tok[2]
        if d:
            return datetime(y, int(m), int(d)) + timedelta(days=1) - timedelta(seconds=1)
        if m:
            mm = int(m)
            nxt = datetime(y + 1, 1, 1) if mm == 12 else datetime(y, mm + 1, 1)
            return nxt - timedelta(seconds=1)
        return datetime(y, 12, 31, 23, 59, 59)

    lower = text.lower()
    if len(tokens) == 1:
        tok = tokens[0]
        if re.search(r"\b(after|since|from)\b", lower):
            return (_to_mjd(_start(tok)), _to_mjd(datetime(2100, 1, 1)))
        if re.search(r"\b(before|until|up to)\b", lower):
            return (0.0, _to_mjd(_end(tok)))
        return (_to_mjd(_start(tok)), _to_mjd(_end(tok)))
    first, last = tokens[0], tokens[-1]
    start, end = _start(first), _end(last)
    if end < start:
        start, end = _start(last), _end(first)
    return (_to_mjd(start), _to_mjd(end))


def apply_date_range(df: pd.DataFrame, date_range: Optional[str]) -> pd.DataFrame:
    """Client-side observation-date filter on t_min/t_max (MJD).

    Rows whose [t_min, t_max] overlaps the requested window are kept. The
    outcome is recorded in ``df.attrs['date_range']`` so a caller can disclose
    whether the filter was applied, ignored (no parseable dates) or
    impossible (no MJD columns).
    """
    window = parse_date_range(date_range)
    if df is None or window is None:
        if df is not None and date_range:
            df.attrs["date_range"] = {"requested": date_range, "applied": False,
                                      "reason": "no parseable date in date_range"}
        return df
    if df.empty:
        df.attrs["date_range"] = {"requested": date_range, "applied": True, "mjd": list(window)}
        return df
    if "t_min" not in df.columns and "t_max" not in df.columns:
        df.attrs["date_range"] = {"requested": date_range, "applied": False,
                                  "reason": "results carry no t_min/t_max (MJD) columns"}
        return df
    start, end = window
    t_min = pd.to_numeric(df["t_min"], errors="coerce") if "t_min" in df.columns else None
    t_max = pd.to_numeric(df["t_max"], errors="coerce") if "t_max" in df.columns else None
    if t_min is None:
        t_min = t_max
    if t_max is None:
        t_max = t_min
    mask = (t_min <= end) & (t_max >= start)
    out = df[mask.fillna(False)].copy()
    out.attrs.update(df.attrs)
    out.attrs["date_range"] = {"requested": date_range, "applied": True, "mjd": [start, end],
                               "rows_before": int(len(df)), "rows_after": int(len(out))}
    return out


def _call_with_optional_kwargs(fn, *args, **kwargs):
    """Call ``fn`` with the advertised keyword controls, dropping any the
    callee does not accept (older clients / test fakes keep the legacy
    positional signature). Never hides a keyword the callee DOES accept."""
    try:
        return fn(*args, **kwargs)
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" not in message:
            raise
        trimmed = {k: v for k, v in kwargs.items() if f"'{k}'" not in message}
        if len(trimmed) == len(kwargs):
            raise
        return _call_with_optional_kwargs(fn, *args, **trimmed) if trimmed else fn(*args)


class SearchService:
    """High-level search operations for NRAO data"""

    def __init__(self):
        self.alminer_client = ALminerClient()
        self._nrao_client = None

    @property
    def nrao_client(self):
        """Lazily constructed NRAO TAP client for VLA/VLBA/GBT searches."""
        if self._nrao_client is None:
            try:
                from integrations.tap import NRAOTapClient
                self._nrao_client = NRAOTapClient()
            except Exception as e:
                print(f"[SearchService] NRAO TAP client unavailable: {e}")
                self._nrao_client = False  # sentinel: do not retry every call
        return self._nrao_client or None

    @staticmethod
    def is_nrao_facility(facility: Optional[str]) -> bool:
        """True when the requested facility lives in the NRAO (non-ALMA) archive."""
        return bool(facility) and facility.strip().upper() in NRAO_FACILITIES

    def _nrao_instruments(self, facility: str) -> List[str]:
        from integrations.tap import NRAOTapClient
        return NRAOTapClient.instruments_for_facility(facility)

    def cone_search(self, ra: float, dec: float, radius: float,
                   facility: Optional[str] = None,
                   max_results: int = 100,
                   public: bool = False) -> pd.DataFrame:
        """Perform cone search. Routes to NRAO TAP for VLA/VLBA/GBT, else the ALMA
        footprint cone (TAP INTERSECTS/point union, ALminer fallback)."""
        if self.is_nrao_facility(facility):
            client = self.nrao_client
            if client is None:
                logger.warning("[SearchService] NRAO TAP unavailable — cannot search VLA/VLBA/GBT")
                return _errored_frame("NRAO TAP archive is unavailable — cannot search VLA/VLBA/GBT.")
            try:
                return client.search_by_position(
                    ra, dec, radius,
                    instruments=self._nrao_instruments(facility),
                    max_results=max_results,
                )
            except Exception as e:
                logger.warning("NRAO TAP position search failed: %s", e)
                return _errored_frame(f"NRAO TAP position search failed: {e}")

        try:
            return _call_with_optional_kwargs(
                self.alminer_client.search_by_position, ra, dec, radius,
                public=bool(public), max_results=max_results,
            )
        except Exception as e:
            logger.warning("ALminer position search failed: %s", e)
            return _errored_frame(f"ALMA archive position search failed: {e}")

    def search_by_target(self, target_name: str,
                        facility: Optional[str] = None,
                        date_range: Optional[str] = None,
                        max_results: int = 100,
                        public_only: bool = False) -> pd.DataFrame:
        """Search by target name. Routes to NRAO TAP for VLA/VLBA/GBT, else ALMA.

        ``public_only`` adds ``data_rights = 'Public'`` (advertised control,
        A-06); ``date_range`` is applied client-side on t_min/t_max MJD (A-34)."""
        if self.is_nrao_facility(facility):
            client = self.nrao_client
            if client is None:
                logger.warning("[SearchService] NRAO TAP unavailable — cannot search VLA/VLBA/GBT")
                return _errored_frame("NRAO TAP archive is unavailable — cannot search VLA/VLBA/GBT.")
            try:
                return client.search_vla_vlba(
                    target_name,
                    max_results=max_results,
                    instruments=self._nrao_instruments(facility),
                )
            except Exception as e:
                logger.warning("NRAO TAP target search failed: %s", e)
                return _errored_frame(f"NRAO TAP target search failed: {e}")

        try:
            df = _call_with_optional_kwargs(
                self.alminer_client.search_by_target, target_name,
                public=bool(public_only), max_results=max_results,
            )
        except Exception as e:
            logger.warning("ALminer target search failed: %s", e)
            return _errored_frame(f"ALMA archive target search failed: {e}")
        if date_range:
            df = apply_date_range(df, date_range)
        return df

    def search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float,
                           facility: Optional[str] = None,
                           max_results: int = 100,
                           public: bool = False) -> pd.DataFrame:
        """Search by frequency range. Routes to NRAO TAP for VLA/VLBA/GBT, else ALminer (ALMA)."""
        if self.is_nrao_facility(facility):
            client = self.nrao_client
            if client is None:
                logger.warning("[SearchService] NRAO TAP unavailable — cannot search VLA/VLBA/GBT")
                return _errored_frame("NRAO TAP archive is unavailable — cannot search VLA/VLBA/GBT.")
            try:
                return client.search_by_frequency_range(
                    min_freq_ghz, max_freq_ghz,
                    instruments=self._nrao_instruments(facility),
                    max_results=max_results,
                )
            except Exception as e:
                logger.warning("NRAO TAP frequency search failed: %s", e)
                return _errored_frame(f"NRAO TAP frequency search failed: {e}")

        try:
            return _call_with_optional_kwargs(
                self.alminer_client.search_by_frequency, min_freq_ghz, max_freq_ghz,
                public=bool(public), max_results=max_results,
            )
        except Exception as e:
            logger.warning("ALMA frequency search failed: %s", e)
            return _errored_frame(f"ALMA frequency search failed: {e}")

    def get_observation_details(self, obs_id: str) -> Dict[str, Any]:
        """Look one ALMA identifier up in ivoa.obscore.

        Accepts a MOUS UID (uid://A001/...), an EB/ASDM UID (uid://A002/...),
        an obs_publisher_did, or a project code, and returns the matching rows
        aggregated to rows / MOUS / EBs (G-13: this used to be a stub that
        returned {} as a success). The identifier kind is inferred from the
        column role, never from the A001/A002 prefix alone (skill: UID grammar).
        """
        from services.alma_science_queries import (
            OBSCORE_BASE_COLUMNS, aggregate_counts, counts_note, escape_adql,
        )

        text = str(obs_id or "").strip()
        if not text:
            return {"success": False, "error": "obs_id is required (MOUS UID, ASDM/EB UID, obs_publisher_did or project code)."}
        safe = escape_adql(text)
        if re.match(r"^\d{4}\.[0-9A-Za-z]\.\d{5}\.[A-Z]$", text):
            where = f"proposal_id = '{safe}'"
        elif text.lower().startswith("uid://"):
            where = f"(member_ous_uid = '{safe}' OR asdm_uid = '{safe}' OR group_ous_uid = '{safe}')"
        elif text.lower().startswith("ads/") or text.lower().startswith("ivo://"):
            where = f"obs_publisher_did = '{safe}'"
        else:
            where = f"(member_ous_uid = '{safe}' OR asdm_uid = '{safe}' OR obs_publisher_did = '{safe}')"
        query = (
            f"SELECT TOP 2000 {', '.join(OBSCORE_BASE_COLUMNS)} FROM ivoa.obscore WHERE {where}"
        )
        try:
            service = self.alminer_client._get_tap_service()
            df = service.search(query).to_table().to_pandas()
        except Exception as e:
            return {"success": False, "error": f"ALMA TAP lookup failed: {e}", "adql": query}
        if df.empty:
            return {
                "success": True, "found": False, "identifier": text, "adql": query,
                "message": "No ivoa.obscore row matches this identifier (checked member_ous_uid, asdm_uid, group_ous_uid, obs_publisher_did, proposal_id).",
            }
        counts = aggregate_counts(df)
        matched_as = []
        for column in ("member_ous_uid", "asdm_uid", "group_ous_uid", "obs_publisher_did", "proposal_id"):
            if column in df.columns and (df[column].astype(str) == text).any():
                matched_as.append(column)
        first = df.iloc[0]

        def _val(name):
            v = first.get(name) if name in df.columns else None
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else (v.item() if hasattr(v, "item") else v)

        return {
            "success": True,
            "found": True,
            "identifier": text,
            "matched_as": matched_as,
            "counts": counts,
            "summary": counts_note(counts),
            "proposal_ids": sorted(df["proposal_id"].dropna().astype(str).unique().tolist())[:20] if "proposal_id" in df.columns else [],
            "member_ous_uids": sorted(df["member_ous_uid"].dropna().astype(str).unique().tolist())[:20] if "member_ous_uid" in df.columns else [],
            "asdm_uids": sorted(df["asdm_uid"].dropna().astype(str).unique().tolist())[:50] if "asdm_uid" in df.columns else [],
            "target_names": sorted(df["target_name"].dropna().astype(str).unique().tolist())[:20] if "target_name" in df.columns else [],
            "band_list": sorted(df["band_list"].dropna().astype(str).unique().tolist()) if "band_list" in df.columns else [],
            "data_rights": sorted(df["data_rights"].dropna().astype(str).unique().tolist()) if "data_rights" in df.columns else [],
            "obs_release_date": {
                "min": str(df["obs_release_date"].dropna().astype(str).min()) if "obs_release_date" in df.columns and df["obs_release_date"].notna().any() else None,
                "max": str(df["obs_release_date"].dropna().astype(str).max()) if "obs_release_date" in df.columns and df["obs_release_date"].notna().any() else None,
            },
            "qa2_passed_flags": sorted(df["qa2_passed"].dropna().astype(str).unique().tolist()) if "qa2_passed" in df.columns else [],
            "access_url_is_datalink": True,
            "adql": query,
            "note": "Use list_alma_files(member_ous_uid) to enumerate deliverables; qa2_passed is a T/F flag that cannot encode PASS/SEMIPASS/FAIL.",
        }

    def advanced_search(self, query: str) -> pd.DataFrame:
        """Execute advanced ADQL/TAP query"""
        try:
            return self.alminer_client.search_by_sql(query)
        except Exception as e:
            logger.warning("Advanced ADQL/TAP search failed: %s", e)
            return _errored_frame(f"ALMA advanced ADQL/TAP query failed: {e}")

    def search_alma_with_keywords(self, keywords: Dict[str, Any]) -> pd.DataFrame:
        """Search ALMA using specific keywords (project_code, pi_name, etc.)"""
        try:
            return self.alminer_client.search_by_keywords(keywords)
        except Exception as e:
            logger.warning("ALMA keyword search failed: %s", e)
            return _errored_frame(f"ALMA keyword search failed: {e}")

    def plot_alma_results(self, df: pd.DataFrame, plot_type: str = "sky") -> bytes:
        """
        Generate plots for ALMA results
        plot_type: 'sky', 'frequency', 'overview'
        Returns image as bytes (Fix 5)
        """
        if plot_type == "sky":
            return self.alminer_client.plot_sky_distribution(df)
        elif plot_type == "frequency":
            return self.alminer_client.plot_freq_coverage(df)
        elif plot_type == "overview":
            return self.alminer_client.plot_overview(df)
        return b""

    def download_alma_data(self, df: pd.DataFrame, dry_run: bool = False) -> str:
        """Download data"""
        return self.alminer_client.download_data(df, dry_run=dry_run)

    def find_calibrators(self, ra: float, dec: float,
                        max_separation: float = 10.0) -> pd.DataFrame:
        """Find potential calibrator sources near target"""
        return pd.DataFrame()

    def search_source(self, source_name: str, max_results: int = 100) -> pd.DataFrame:
        """
        Smart search that handles any source name properly
        """
        try:
            return self.alminer_client.search_by_target(source_name)
        except Exception as e:
            logger.warning("ALMA source search failed: %s", e)
            return _errored_frame(f"ALMA source search failed: {e}")

    def check_line_coverage(self, line_freq_ghz: float, z: float = 0.0, line_name: str = "Line") -> pd.DataFrame:
        """
        Check if specific frequency line is covered in loaded observations.
        Note: This usually requires a prior search to have results, 
        but alminer.line_coverage takes a DataFrame.
        This tool might need to be chainable or take a 'last_search_context'.
        For now, let's assume the agent passes the dataframe (not possible via JSON).
        Wait, standard tools take simple types. 
        REALITY CHECK: line_coverage filters EXISTING results. 
        So this tool only makes sense if we have a stateful DataFrame.
        The Agent has `self.last_run_result`. 
        We can't pass a DataFrame to an OpenAI tool call.
        
        Alternatives:
        1. Tool doesn't take DF, instead it uses 'latest_search_results' from agent state implicitly? No, simpler to just run a fresh search + filter?
        2. Or user runs this AFTER a search. The Agent code executes it.
        
        Let's Define the tool takes (line_freq, z, line_name). 
        The Agent implementation (`core/agent.py`) will inject `self.last_search_results` when calling this method.
        So this method signature should accept DF? 
        The TOOL definition (for LLM) will hide the DF argument.
        The AGENT (execution) will provide it.
        """
        # This generic signature expects DF to be passed by caller (Agent)
        # We'll handle the injection in agent.py
        pass # The logic is in alminer_client. We just expose the method on the service.
        
    def check_line_coverage_on_last(self, df: pd.DataFrame, line_freq_ghz: float, z: float = 0.0, line_name: str = "Line") -> pd.DataFrame:
        return self.alminer_client.get_line_coverage(df, line_freq_ghz, z, line_name)

    def check_co_lines_on_last(self, df: pd.DataFrame, z: float = 0.0) -> pd.DataFrame:
        return self.alminer_client.get_co_lines(df, z)
        
    def search_catalog(self, catalog_list: List[Dict[str, Any]]) -> pd.DataFrame:
        return self.alminer_client.search_by_catalog(catalog_list)
