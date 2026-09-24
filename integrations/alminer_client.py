"""
ALminer Client Integration
Wraps alminer functionality for Quasar

Failure-state contract (INT-5): every archive failure returns an EMPTY frame
carrying ``df.attrs["quasar_error"]`` (see :func:`_errored_frame`), never a
bare empty frame. Callers (services/search.py, capabilities/alma.py) read that
marker to tell "archive unavailable / query failed" apart from "no matching
data", so an outage is never reported to the user as "no data exists".
A genuinely empty result carries no marker (and may carry
``attrs["resolver_unresolved"]`` when the name resolver found nothing).
"""

import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple
import warnings
import os
import time

# Try to import alminer, handle if missing
try:
    import alminer
    ALMINER_AVAILABLE = True
except ImportError:
    ALMINER_AVAILABLE = False

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ── Cached SIMBAD resolution (canonical impl: integrations/simbad_resolver.py) ──
from integrations.simbad_resolver import _resolve_simbad_cached
from services.alma_science_queries import (
    LINE_REST_FREQ_GHZ,
    OBSCORE_BASE_COLUMNS,
    alma_cone_adql,
    alma_cone_count_adql,
    aggregate_counts,
    counts_note,
    observation_windows_ghz,
    parse_frequency_support_windows,
    wavelength_overlap_where,
)


ALMA_TAP_URL = "https://almascience.nrao.edu/tap"
ALMA_AQ_URL = "https://almascience.nrao.edu/aq/"
DEFAULT_ROW_CAP = 5000


def alminer_filter_frame(df: pd.DataFrame, *, band: Any = None, max_resolution_arcsec: Optional[float] = None,
                         science_only: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Apply the cone filters to a full ALminer frame (it has no filter
    arguments). Token band match on band_list (A-08); unknown columns are left
    alone and simply not reported as applied."""
    from services.alma_science_queries import requested_bands, row_matches_band

    applied: List[str] = []
    if df is None or df.empty:
        return df, applied
    bands = requested_bands(band) if band is not None else []
    b_col = next((c for c in ("band_list", "Band", "band") if c in df.columns), None)
    if bands and b_col:
        df = df[df[b_col].apply(lambda x: row_matches_band(x, bands))]
        applied.append(f"band {', '.join(bands)}")
    r_col = next((c for c in ("spatial_resolution", "resolution") if c in df.columns), None)
    if max_resolution_arcsec is not None and r_col:
        df = df[pd.to_numeric(df[r_col], errors="coerce") <= float(max_resolution_arcsec)]
        applied.append(f"resolution <= {float(max_resolution_arcsec):g} arcsec")
    if science_only and "science_observation" in df.columns:
        df = df[df["science_observation"].astype(str).str.upper().isin({"T", "TRUE", "1"})]
        applied.append("science observations only")
    return df, applied


def _row_cap(value, default=DEFAULT_ROW_CAP):
    """Keep the query, transport and truncation metadata on the same limit."""
    return max(1, min(int(default if value is None else value), 20000))

# Canonical ALMA result columns — guaranteed present on BOTH the TAP and
# ALminer paths by _standardize_columns so downstream code sees a deterministic
# schema regardless of which source answered. (C14)
_CANONICAL_ALMA_COLUMNS = [
    "s_ra", "s_dec", "t_exptime", "Band", "resolution", "sensitivity",
    "bandwidth", "freq_min", "freq_max", "freq_min_ghz", "freq_max_ghz",
    "telescope", "instrument_name", "access_url", "archive_url",
]


def _errored_frame(error: str) -> pd.DataFrame:
    """Empty frame tagged with the archive error (outage != empty, C3)."""
    df = pd.DataFrame()
    df.attrs["quasar_error"] = str(error)
    df.attrs["partial"] = True
    return df


def _is_footprint_rejection(exc: Exception) -> bool:
    """True only when the service objected to the footprint predicate itself."""
    text = str(exc).lower()
    return ("intersects" in text or "s_region" in text) and ("not supported" in text or "unsupported" in text or "syntax" in text or "unknown" in text or "error" in text)



def run_bounded_source(fn, timeout: float):
    """Run one archive source search bounded by ``timeout`` (itself clamped to
    the running tool's remaining budget) on a worker that INHERITS the tool
    deadline, so the TAP session / requests hook inside it stay bounded too
    (guard CX-02: a bare thread saw no deadline and could outlive the tool).

    Returns ``(df, error_text, timed_out)``; on expiry the worker is abandoned
    with the same contract as the tool guard.
    """
    from services.tool_budgets import BudgetExhausted, bounded_timeout, call_bounded

    try:
        bound = bounded_timeout(float(timeout), minimum=2.0, label="ALMA source search")
    except BudgetExhausted as exc:
        return None, str(exc), True
    try:
        df = call_bounded(fn, bound, label="ALMA source search", thread_name="quasar-alma-source-search")
    except TimeoutError as exc:
        if "did not answer within" in str(exc):
            return None, f"timed out after {bound:.0f} s", True
        return None, str(exc) or exc.__class__.__name__, False
    except Exception as exc:  # ImportError, network, parse
        return None, str(exc) or exc.__class__.__name__, False
    return df, None, False




def alminer_public_flag(public: Optional[bool]) -> Optional[bool]:
    """Map our ``public`` flag onto ALMiner's three-state one: ALMiner reads
    True = public only, False = PROPRIETARY ONLY, None = both. Ours: True =
    public only, False/None = no restriction."""
    return True if public else None

class ALminerClient:
    """Client for interacting with ALMA archive via ALminer + the ALMA TAP"""

    # Per-source bounds for _parallel_search (seconds). 2026-09-21: TAP was
    # 390 s here — against an archive that answers a healthy cone in 2-40 s,
    # a hung mirror cost 6.5 minutes of silence per call, under a 465 s guard.
    # TAP now gets what AlmaTapService can spend (2 attempts x 40 s + backoff),
    # ALminer 30 s; both are clamped again to the running tool's remaining
    # budget (services/tool_budgets.py), so the tool guard is never the first
    # thing to fire.
    SOURCE_TIMEOUT_S = 85
    ALMINER_TIMEOUT_S = 30

    def __init__(self):
        """Initialize ALminer client"""
        if not ALMINER_AVAILABLE:
            warnings.warn("alminer not installed. ALMA searches will fall back to TAP only.")

        # Ensure we have a downloads directory
        self.download_dir = "./downloads"
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir, exist_ok=True)

        # Cached TAP service — reuse TCP connection across queries
        self._tap_service = None

    def _get_tap_service(self):
        """Get or create a cached pyvo TAPService instance."""
        if self._tap_service is None:
            from integrations.alma_tap import AlmaTapService
            self._tap_service = AlmaTapService()
        return self._tap_service

    # ── Target / cone search ────────────────────────────────────────────────
    def search_by_target(self, target_name: str, public: bool = True,
                         max_results: Optional[int] = None, *, band: Any = None,
                         max_resolution_arcsec: Optional[float] = None,
                         science_only: bool = False) -> pd.DataFrame:
        """
        Search ALMA archive by target name.
        Strategy:
        1. Resolve name to RA/Dec via SIMBAD (cached)
        2. TAP footprint cone (INTERSECTS s_region OR point) then ALminer fallback

        ``band`` / ``max_resolution_arcsec`` / ``science_only`` go INTO the ADQL
        (UI benchmark 2026-09-22, D16), and a ``COUNT(*)`` companion records
        ``total_count`` when the TOP cap truncates the result.
        """
        print(f"[ALMA] Starting search for '{target_name}'")

        # Step 1: Resolve target name using cached SIMBAD lookup
        try:
            ra_deg, dec_deg = _resolve_simbad_cached(target_name)
        except Exception as e:
            print(f"[ALMA] SIMBAD resolution failed: {e}")
            return _errored_frame(f"Name resolution (SIMBAD) failed for '{target_name}': {e}")
        if ra_deg is None:
            print(f"[ALMA] SIMBAD could not resolve '{target_name}'")
            # A genuine empty: nothing to search. Moving/solar targets need a
            # name/time-window path the caller can offer.
            df = pd.DataFrame()
            df.attrs["resolver_unresolved"] = str(target_name)
            return df
        print(f"[ALMA] Resolved to RA={ra_deg:.4f}, Dec={dec_deg:.4f}")

        # Step 2: TAP then ALminer, each bounded
        # 1 arcmin: ALminer's documented default target search radius
        # (alminer.target(search_radius=1.0)); the old 0.05 deg (3 arcmin) pulled
        # in unrelated neighbours for compact targets.
        return self._parallel_search(ra_deg, dec_deg, radius=1.0 / 60, target_name=target_name,
                                     public=public, max_results=max_results, band=band,
                                     max_resolution_arcsec=max_resolution_arcsec, science_only=science_only)

    def _parallel_search(self, ra: float, dec: float, radius: float = 0.05, target_name: str = "",
                         public: bool = True, max_results: Optional[int] = None, *, band: Any = None,
                         max_resolution_arcsec: Optional[float] = None, science_only: bool = False) -> pd.DataFrame:
        """
        Primary-plus-fallback cone search: try TAP, then ALminer, SEQUENTIALLY.
        Each source is bounded by a per-source timeout on a daemon thread, so
        there is no two-thread race and no stranded in-flight search (C14).
        Returns one canonical column schema via _standardize_columns.

        Every outcome is recorded: a source that timed out or raised is never
        silently treated as "no result" (A-19/A-59). Only when NO source
        returned an actual frame does the caller get an errored frame.
        """
        import threading

        _run_bounded = run_bounded_source

        cone_meta: Dict[str, Any] = {
            "ra": float(ra), "dec": float(dec), "radius_deg": float(radius),
            "public": bool(public), "footprint_mode": "intersects_or_point",
        }
        _filters = {"band": band, "max_resolution_arcsec": max_resolution_arcsec, "science_only": bool(science_only)}
        if band is not None or max_resolution_arcsec is not None or science_only:
            cone_meta["filters"] = {k: v for k, v in _filters.items() if v not in (None, False)}
        top = _row_cap(max_results)

        def tap_search():
            print("[ALMA] TAP search starting...")
            service = self._get_tap_service()
            footprint = True
            query = alma_cone_adql(ra, dec, radius, public=public, top=top, **_filters)
            try:
                res = service.search(query, maxrec=top)
            except Exception as exc:
                if not _is_footprint_rejection(exc):
                    raise
                # Service rejected the footprint predicate: point-only fallback,
                # disclosed through attrs so provenance tells the truth.
                print(f"[ALMA] TAP rejected INTERSECTS (falling back to point cone): {exc}")
                footprint = False
                query = alma_cone_adql(ra, dec, radius, public=public, top=top, footprint=False, **_filters)
                cone_meta["footprint_mode"] = "point_only"
                res = service.search(query, maxrec=top)
            cone_meta["adql"] = query
            cone_meta["url"] = getattr(res, "quasar_tap_url", ALMA_TAP_URL)
            df = res.to_table().to_pandas()
            df.attrs["truncated"] = bool(len(df) >= top or getattr(res, "query_status", "") == "OVERFLOW")
            df.attrs["row_cap"] = top
            df.attrs["filters_in_adql"] = dict(cone_meta.get("filters") or {})
            if df.attrs["truncated"]:
                # The cap hit: ask the server how many rows/MOUS the SAME cone
                # and filters hold, so the caller can say "N total, showing M".
                count_query = alma_cone_count_adql(ra, dec, radius, public=public, footprint=footprint, **_filters)
                try:
                    from services.tool_budgets import remaining_seconds

                    left = remaining_seconds()
                    if left is None or left >= 8.0:
                        cres = service.search(count_query, maxrec=5)
                        cdf = cres.to_table().to_pandas()
                        if not cdf.empty:
                            df.attrs["total_count"] = int(cdf.iloc[0]["total_rows"])
                            df.attrs["total_mous"] = int(cdf.iloc[0]["total_mous"])
                            cone_meta["count_adql"] = count_query
                except Exception as count_exc:  # the count is a courtesy, never a failure
                    print(f"[ALMA] COUNT(*) companion failed: {count_exc}")
            return df

        def alminer_search():
            if not ALMINER_AVAILABLE:
                raise ImportError("alminer is not installed")
            import alminer as _alminer
            print("[ALMA] ALminer search starting...")
            # ALMiner semantics (live 2026-09-23, D16): search_radius is in ARCMIN
            # (our radius is degrees -- 1/60 deg was a 1-arcsec search), and
            # public=False means PROPRIETARY ONLY (None = both). Our public=False
            # means "do not restrict to public".
            df = _alminer.conesearch(ra, dec, search_radius=float(radius) * 60.0, public=alminer_public_flag(public),
                                     print_targets=False)
            cone_meta.setdefault("footprint_mode", "point_only")
            cone_meta["adql"] = alma_cone_adql(ra, dec, radius, public=public, footprint=False)
            if cone_meta.get("filters"):
                cone_meta["note_filters"] = "alminer.conesearch cannot apply band/resolution filters; they are applied client-side."
            cone_meta["url"] = ALMA_TAP_URL
            cone_meta["note"] = (
                "-- Reproducible equivalent of the executed alminer.conesearch"
                f"(public={alminer_public_flag(public)}) call; alminer's internal ADQL is not captured byte-exactly."
            )
            df = df if df is not None else pd.DataFrame()
            # Filter BEFORE the cap: capping first kept the first `top` rows of
            # ALL bands and then filtered them client-side (D16 re-run: 100 ->
            # 28 Band 6 rows while the cone held many more).
            df, applied = alminer_filter_frame(df, **_filters)
            if applied:
                cone_meta["note_filters"] = f"alminer.conesearch has no band/resolution arguments; {', '.join(applied)} applied to the full result before the {top}-row cap."
            capped = df.head(top).copy()
            capped.attrs.update(row_cap=top, truncated=len(df) > top, total_count=len(df),
                                filters_in_adql={})
            return capped

        failures: List[str] = []
        empty_frames: List[pd.DataFrame] = []
        for name, fn in (("TAP", tap_search), ("ALminer", alminer_search)):
            df, err, timed_out = _run_bounded(fn, self.SOURCE_TIMEOUT_S if name == "TAP" else self.ALMINER_TIMEOUT_S)
            if df is None:
                failures.append(f"{name}: {err}")
                print(f"[ALMA] {name} failed: {err}")
                continue
            if not df.empty:
                print(f"[ALMA] {name} found {len(df)} rows")
                out = self._standardize_columns(df)
                out.attrs["quasar_cone"] = dict(cone_meta)
                out.attrs["source"] = name
                out.attrs["truncated"] = bool(df.attrs.get("truncated", False))
                if failures:
                    out.attrs["source_warnings"] = list(failures)
                return out
            if name == "TAP":
                # The primary footprint query answered successfully. A missing
                # optional fallback cannot turn a valid empty result into failure.
                df.attrs["quasar_cone"] = dict(cone_meta)
                return df
            empty_frames.append(df)
        if empty_frames and not failures:
            # At least one source answered with a real (empty) table — a
            # genuine empty result. Failures of the other source ride along
            # as warnings so the caller can disclose them.
            print("[ALMA] Archive answered: no rows in this cone")
            out = pd.DataFrame()
            out.attrs["quasar_cone"] = dict(cone_meta)
            if failures:
                out.attrs["source_warnings"] = list(failures)
            return out
        print("[ALMA] Every source failed: " + "; ".join(failures))
        return _errored_frame(
            "ALMA archive cone search failed on every source (" + "; ".join(failures) + ")"
        )

    def search_by_position(self, ra: float, dec: float, radius: float = 1.0 / 60, public: bool = True,
                           max_results: Optional[int] = None, *, band: Any = None,
                           max_resolution_arcsec: Optional[float] = None, science_only: bool = False) -> pd.DataFrame:
        """
        Search ALMA archive by position (cone search over footprints).
        Default radius 1 arcmin = ALminer's documented default search radius.
        radius is in degrees (default ~1 arcmin)
        """
        try:
            # Filters go INTO the TAP ADQL and are applied to the full ALminer
            # frame before the row cap (D16 re-run: Band 6 was filtered after
            # a 100-row all-band cap).
            return self._parallel_search(float(ra), float(dec), radius=float(radius),
                                         public=public, max_results=max_results, band=band,
                                         max_resolution_arcsec=max_resolution_arcsec, science_only=science_only)
        except Exception as e:
            print(f"[ALMA] Position search error: {e}")
            return _errored_frame(f"ALMA position search failed: {e}")

    # ── Keyword / ADQL / frequency ──────────────────────────────────────────
    def search_by_keywords(self, keywords: Dict[str, Any], public: bool = True) -> pd.DataFrame:
        """
        Search by ALMA keywords (PI name, proposal ID, etc.)
        Example keywords: {'pi_name': 'Smith', 'proposal_id': '2017.1.000'}
        """
        if not ALMINER_AVAILABLE:
            return _errored_frame("alminer is not installed; keyword search is unavailable.")

        try:
            search_dict = {key: value if isinstance(value, list) else [str(value)]
                           for key, value in keywords.items()}
            df = alminer.keysearch(search_dict, public=alminer_public_flag(public), print_targets=False)
            if df is None:
                return pd.DataFrame()
            return self._standardize_columns(df)
        except Exception as e:
            print(f"ALminer keyword search error: {e}")
            return _errored_frame(f"ALMA keyword search failed: {e}")

    def search_by_sql(self, query: str, maxrec: int = 20000) -> pd.DataFrame:
        """
        Execute custom ADQL through the same retrying PyVO transport as cones.
        """
        try:
            maxrec = _row_cap(maxrec, default=20000)
            res = self._get_tap_service().search(query, maxrec=maxrec)
            df = self._standardize_columns(res.to_table().to_pandas())
            df.attrs["quasar_adql"] = query
            df.attrs["quasar_tap_url"] = getattr(res, "quasar_tap_url", ALMA_TAP_URL)
            df.attrs["truncated"] = len(df) >= maxrec or getattr(res, "query_status", "") == "OVERFLOW"
            df.attrs["row_cap"] = maxrec
            return df
        except Exception as e:
            print(f"ALminer SQL/TAP search error: {e}")
            return _errored_frame(f"ALMA TAP query failed: {e}")

    def search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float, public: bool = True,
                            max_results: Optional[int] = None) -> pd.DataFrame:
        """
        Search by frequency range using the wavelength-overlap idiom.

        ``frequency`` is a single representative value; observations whose
        spectral windows overlap [min, max] but whose representative value
        lies outside were missed by ``frequency BETWEEN``. The skill's
        em_min/em_max overlap (wavelengths in METRES) is the correct coarse
        prefilter; exact SPW coverage still needs frequency_support.
        """
        try:
            lo = float(min_freq_ghz)
            hi = float(max_freq_ghz)
            if lo > hi:
                lo, hi = hi, lo
            top = _row_cap(max_results)
            where = wavelength_overlap_where(lo, hi)
            if public:
                where += " AND data_rights = 'Public'"
            query = (
                f"SELECT TOP {top} {', '.join(OBSCORE_BASE_COLUMNS)} "
                f"FROM ivoa.obscore WHERE {where} ORDER BY proposal_id"
            )
            service = self._get_tap_service()
            res = service.search(query, maxrec=top)
            df = res.to_table().to_pandas()
            print(f"[ALMA] Frequency search found {len(df)} rows")
            out = self._standardize_columns(df) if not df.empty else pd.DataFrame()
            out.attrs["quasar_adql"] = query
            out.attrs["quasar_tap_url"] = getattr(res, "quasar_tap_url", ALMA_TAP_URL)
            out.attrs["truncated"] = bool(len(df) >= top or getattr(res, "query_status", "") == "OVERFLOW")
            out.attrs["row_cap"] = top
            return out
        except ImportError:
            return _errored_frame("pyvo is not installed; ALMA frequency search is unavailable.")
        except Exception as e:
            print(f"[ALMA] Frequency search error: {e}")
            return _errored_frame(f"ALMA frequency search failed: {e}")

    def get_run_summary(self, df: pd.DataFrame) -> str:
        """Get a text summary of the results (rows vs datasets vs executions)."""
        if df is None or df.empty:
            return "No data available to summarize."
        summary = counts_note(aggregate_counts(df)) + "\n"
        if 'target_name' in df.columns:
            summary += f"Unique Targets: {df['target_name'].nunique()}\n"
        return summary

    # ── Plots ───────────────────────────────────────────────────────────────
    def plot_sky_distribution(self, df: pd.DataFrame, filename: str = "alma_sky_plot.png") -> bytes:
        """
        Generate sky distribution plot
        Returns image as bytes (Fix 5 - prevents MediaFileStorageError)
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return b""

        try:
            import tempfile
            # Save to temp file, read bytes, cleanup
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name

            alminer.plot_sky(df, savefig=tmp_path)

            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()

            # Cleanup temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            return image_bytes
        except Exception as e:
            print(f"Error plotting sky distribution: {e}")
            return b""

    def plot_freq_coverage(self, df: pd.DataFrame, filename: str = "alma_freq_plot.png") -> bytes:
        """
        Generate frequency coverage plot
        Returns image as bytes (Fix 5 - prevents MediaFileStorageError)
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return b""

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name

            alminer.plot_bands(df, savefig=tmp_path)

            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()

            try:
                os.remove(tmp_path)
            except OSError:
                pass

            return image_bytes
        except Exception as e:
            print(f"Error plotting frequency coverage: {e}")
            return b""

    def plot_overview(self, df: pd.DataFrame, filename: str = "alma_overview_plot.png") -> bytes:
        """
        Generate overview plot (Integration time vs Sensitivity usually)
        Returns image as bytes (Fix 5 - prevents MediaFileStorageError)
        """
        if not ALMINER_AVAILABLE or not MATPLOTLIB_AVAILABLE or df.empty:
            return b""

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name

            alminer.plot_overview(df, savefig=tmp_path)

            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()

            try:
                os.remove(tmp_path)
            except OSError:
                pass

            return image_bytes
        except Exception as e:
            print(f"Error plotting overview: {e}")
            return b""

    # ── Download ────────────────────────────────────────────────────────────
    def download_data(self, df: pd.DataFrame, dry_run: bool = False,
                      download_dir: Optional[str] = None) -> str:
        """
        Download FITS products for the observations in the dataframe.

        This call is SYNCHRONOUS: it returns only after alminer has finished
        (or the caller's tool guard abandoned the thread). The byte/disk
        preflight lives in capabilities/alma.py (DownloadAlmaData); this method
        only performs the transfer.
        """
        if not ALMINER_AVAILABLE or df is None or df.empty:
            return "ALminer not available or empty dataframe."

        target_dir = download_dir or self.download_dir
        try:
            os.makedirs(target_dir, exist_ok=True)
            started = time.time()
            alminer.download_data(df, download_dir=target_dir, dryrun=dry_run, fitsonly=True)
            elapsed = time.time() - started
            if dry_run:
                return f"Dry run complete. Would download FITS products to {target_dir}"
            files = []
            try:
                for root, _dirs, names in os.walk(target_dir):
                    files.extend(os.path.join(root, n) for n in names)
            except OSError:
                pass
            return (
                f"Download completed (synchronous, {elapsed:.0f} s) to {target_dir} "
                f"on the Quasar server; {len(files)} file(s) now present there."
            )
        except Exception as e:
            return f"Download failed: {e}"

    # ── Column normalisation ────────────────────────────────────────────────
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize ALminer/TAP output columns to Quasar's expected format.

        ObsCore ``access_url`` is the MOUS DataLink URL and is KEPT (skill:
        follow the returned URL, never rewrite it); the hand-built Archive
        Query page link lives in ``archive_url``. ``sensitivity_10kms`` and
        ``s_resolution`` are kept and get display aliases (``sensitivity``,
        ``resolution``) instead of being renamed away (A-22).
        """
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()

        rename_map = {
            'ra': 's_ra',
            'dec': 's_dec',
            'integration_time': 't_exptime',
            'band_number': 'Band',
            'min_freq_ghz': 'freq_min',
            'max_freq_ghz': 'freq_max',
            'min_freq_GHz': 'freq_min',
            'max_freq_GHz': 'freq_max',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Aliases (keep the archive column, add the display name)
        if 's_resolution' in df.columns and 'resolution' not in df.columns:
            df['resolution'] = df['s_resolution']
        if 'sensitivity_10kms' in df.columns and 'sensitivity' not in df.columns:
            df['sensitivity'] = df['sensitivity_10kms']

        # Ensure standard columns exist
        df['telescope'] = 'ALMA'
        df['instrument_name'] = 'ALMA'

        # Frequency span from frequency_support (TAP shape has no min/max
        # frequency columns; alminer's shape does).
        if 'frequency_support' in df.columns and 'freq_min' not in df.columns:
            spans = df['frequency_support'].apply(_frequency_span_ghz)
            df['freq_min'] = spans.apply(lambda s: s[0])
            df['freq_max'] = spans.apply(lambda s: s[1])

        # Add frequency aliases for CLI display compatibility
        if 'freq_min' in df.columns and 'freq_min_ghz' not in df.columns:
            df['freq_min_ghz'] = df['freq_min']
        if 'freq_max' in df.columns and 'freq_max_ghz' not in df.columns:
            df['freq_max_ghz'] = df['freq_max']

        # Archive Query page deep link per MOUS (the DataLink access_url stays).
        if 'member_ous_uid' in df.columns:
            from urllib.parse import quote_plus
            df['archive_url'] = df['member_ous_uid'].apply(
                lambda x: f"{ALMA_AQ_URL}?member_ous_id={quote_plus(str(x))}" if pd.notna(x) else ""
            )

        # C14: guarantee a stable canonical column set on BOTH source paths.
        for _col in _CANONICAL_ALMA_COLUMNS:
            if _col not in df.columns:
                df[_col] = pd.NA

        return df

    # ── Line coverage (frequency_support-based) ─────────────────────────────
    def get_line_coverage(self, df: pd.DataFrame, line_freq: float, z: float = 0.0,
                          line_name: str = "Line") -> pd.DataFrame:
        """
        Rows of ``df`` whose spectral windows cover ``line_freq`` GHz (rest) at
        redshift ``z``. Decided from ``frequency_support`` per row (exact SPW
        windows); rows lacking it fall back to frequency +/- bandwidth/2 and are
        labelled approximate. Returns an errored frame when coverage cannot be
        determined at all, instead of a silent "0 covered" (A-61).
        """
        if df is None or df.empty:
            return pd.DataFrame()
        try:
            observed = float(line_freq) / (1.0 + float(z or 0.0))
        except (TypeError, ValueError) as e:
            return _errored_frame(f"Invalid line frequency/redshift: {e}")
        return _coverage_rows(df, {line_name: observed})

    def get_co_lines(self, df: pd.DataFrame, z: float = 0.0) -> pd.DataFrame:
        """
        Rows of ``df`` covering any CO / 13CO / C18O transition at redshift ``z``.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        lines = {
            name: freq / (1.0 + float(z or 0.0))
            for name, freq in LINE_REST_FREQ_GHZ.items()
            if name.startswith(("12CO(", "13CO(", "C18O(", "CO("))
        }
        return _coverage_rows(df, lines)

    def search_by_catalog(self, catalog_data: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Search by a catalog of objects.
        catalog_data expected format: {"Name": [...], "RAJ2000": [...], "DEJ2000": [...]}
        """
        if not ALMINER_AVAILABLE:
            return _errored_frame("alminer is not installed; catalog search is unavailable.")

        try:
            cat_df = pd.DataFrame(catalog_data)
            result = alminer.catalog(cat_df, print_targets=False)
            if result is None:
                return pd.DataFrame()
            return self._standardize_columns(result)
        except Exception as e:
            print(f"Catalog search failed: {e}")
            return _errored_frame(f"ALMA catalog search failed: {e}")


def _frequency_span_ghz(value: Any) -> Tuple[Any, Any]:
    windows = parse_frequency_support_windows(value)
    if not windows:
        return (pd.NA, pd.NA)
    return (min(w["low_ghz"] for w in windows), max(w["high_ghz"] for w in windows))


def _coverage_rows(df: pd.DataFrame, lines: Dict[str, float]) -> pd.DataFrame:
    """Filter rows covering any of ``lines`` ({name: observed GHz})."""
    has_support = "frequency_support" in df.columns
    has_fallback = "frequency" in df.columns and "bandwidth" in df.columns
    if not has_support and not has_fallback:
        return _errored_frame(
            "Cannot determine line coverage: results carry neither frequency_support "
            "nor frequency/bandwidth columns."
        )
    keep_index: List[Any] = []
    covered_names: List[str] = []
    covering_spws: List[str] = []
    methods: List[str] = []
    decidable = 0
    for idx, row in df.iterrows():
        windows = observation_windows_ghz(row)
        if not windows:
            continue
        decidable += 1
        exact = bool(parse_frequency_support_windows(row.get("frequency_support")))
        hits: List[str] = []
        spws: List[str] = []
        for name, nu in lines.items():
            for w in windows:
                if w["low_ghz"] <= nu <= w["high_ghz"]:
                    hits.append(name)
                    spws.append(f"{w['low_ghz']:.3f}-{w['high_ghz']:.3f} GHz")
                    break
        if hits:
            keep_index.append(idx)
            covered_names.append(", ".join(hits))
            covering_spws.append("; ".join(spws))
            methods.append("frequency_support SPW windows" if exact
                           else "frequency +/- bandwidth/2 (APPROXIMATE: aggregate bandwidth treated as contiguous)")
    if decidable == 0:
        return _errored_frame(
            "Cannot determine line coverage: no row carries a parseable frequency_support "
            "or a numeric frequency/bandwidth."
        )
    out = df.loc[keep_index].copy()
    out["lines_covered"] = covered_names
    out["covering_spw_ghz"] = covering_spws
    out["coverage_method"] = methods
    out.attrs["rows_undecidable"] = int(len(df) - decidable)
    return out
