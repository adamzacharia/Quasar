"""
MASTClient — Deep MAST Archive Integration for Quasar AI
Provides full query capabilities for JWST, HST, TESS, Kepler data
via astroquery.mast.

CALLED BY: core/agent.py (tool execution: search_mast, search_mast_by_criteria)
CALLS:     astroquery.mast.Observations

Registered agent tools:
    - search_mast(target_name, mission, instrument, radius)
    - search_mast_by_criteria(mission, instrument, proposal_id, filters, ...)
    - get_mast_products()  (file-level product listing)
"""

import os
import threading
import uuid

import pandas as pd
import warnings
from typing import Optional, Dict, Any, List, Tuple

# Try to import astroquery.mast, handle if missing
try:
    from astroquery.mast import Observations
    MAST_AVAILABLE = True
except ImportError:
    MAST_AVAILABLE = False

# Canonical cached SIMBAD resolver
from integrations.simbad_resolver import _resolve_simbad_cached

# Hard cap on concurrently-abandoned MAST download workers (guard CX-01,
# mirroring services/datalab_image_service.py::_DOWNLOAD_SLOTS): a hung or
# trickling archive transfer leaves an uncancellable daemon worker behind;
# each holds a slot until astroquery finally returns/errors, and when all
# slots are held further downloads fail FAST instead of stacking
# threads+sockets without bound.
_DOWNLOAD_SLOTS = threading.BoundedSemaphore(4)

MAST_HOST = "mast.stsci.edu"


def _mast_query_timeout_default() -> float:
    raw = os.getenv("MAST_QUERY_TIMEOUT_SECONDS", "").strip()
    try:
        return max(3.0, float(raw)) if raw else 30.0
    except ValueError:
        return 30.0


MAST_QUERY_TIMEOUT_S = _mast_query_timeout_default()


def _mast_query(fn, *args, **kwargs):
    """Run one astroquery.mast metadata call bounded and breaker-guarded.

    astroquery's own MAST timeout is 600 s per call (conf.timeout) — under a
    150 s tool guard a single slow query, let alone one per source in a
    cross-match loop, made the guard the ONLY thing that could end the tool
    (live 2026-09-21 AM-H-01). The call runs on a daemon worker joined with
    ``min(MAST_QUERY_TIMEOUT_SECONDS, remaining tool budget)``
    (services/tool_budgets.py); on expiry the worker is abandoned and a
    TimeoutError is raised, which — like a refused / reset connection — opens
    the MAST host breaker (services/host_breaker.py) so the next MAST call in
    the turn fails in milliseconds instead of waiting again.
    """
    from services.host_breaker import HostBreaker
    from services.tool_budgets import bounded_timeout, call_bounded

    HostBreaker.check(MAST_HOST)
    timeout = bounded_timeout(MAST_QUERY_TIMEOUT_S, minimum=3.0, label="MAST query")
    try:
        result = call_bounded(
            lambda: fn(*args, **kwargs), timeout,
            label="MAST query", thread_name="quasar-mast-query",
        )
    except Exception as exc:
        HostBreaker.record_failure(MAST_HOST, exc)
        raise
    HostBreaker.record_success(MAST_HOST)
    return result


class MASTClient:
    """
    Deep query client for MAST archive (JWST, HST, TESS, Kepler).

    Wraps astroquery.mast.Observations to provide:
    - Target name search with mission/instrument filters
    - Positional cone search
    - Rich criteria-based search (program ID, date range, filter, instrument)
    - File-level product listing for download
    """

    # Standard MAST missions
    SUPPORTED_MISSIONS = ["JWST", "HST", "TESS", "Kepler", "K2",
                          "GALEX", "IUE", "FUSE", "Swift"]

    # JWST instruments
    JWST_INSTRUMENTS = ["NIRCAM", "NIRSPEC", "MIRI", "NIRISS", "FGS"]

    # HST instruments
    HST_INSTRUMENTS = ["ACS", "WFC3", "COS", "STIS", "NICMOS", "WFPC2"]

    def __init__(self):
        if not MAST_AVAILABLE:
            warnings.warn("astroquery.mast not installed. MAST queries will fail.")

    @staticmethod
    def _instrument_criterion(instrument: str) -> str:
        """MAST instrument_name carries the mode ("NIRSPEC/SLIT", "MIRI/IMAGE",
        "WFC3/UVIS"); a bare instrument must match the whole family, so it
        becomes a wildcard. An explicit mode ("NIRSPEC/IFU") stays exact."""
        value = str(instrument).strip().upper()
        if value and "/" not in value and "*" not in value:
            return value + "*"
        return value

    @staticmethod
    def _to_mjd(value) -> float:
        """ISO date/time (or an MJD number) -> MJD float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            from astropy.time import Time
            return float(Time(str(value).strip()).mjd)

    @staticmethod
    def _failed(message: str) -> pd.DataFrame:
        """An empty frame that CARRIES its failure, so callers never mistake an
        error (timeout, resolver miss, rejected criteria) for 'no data'."""
        out = pd.DataFrame()
        out.attrs["error"] = message
        return out

    def search_by_target(self, target: str, mission: str = None,
                         instrument: str = None, radius: str = "30s",
                         max_results: int = 500) -> pd.DataFrame:
        """
        Search MAST by target name with optional mission/instrument filters.

        Args:
            target: Astronomical target name (e.g., 'M87', 'Carina Nebula')
            mission: Filter by mission (e.g., 'JWST', 'HST')
            instrument: Filter by instrument (e.g., 'NIRCAM', 'ACS')
            radius: Search radius (e.g., '30s' for 30 arcsec)
            max_results: Maximum number of results

        Returns:
            DataFrame of matching observations
        """
        if not MAST_AVAILABLE:
            print("[MAST] ERROR: astroquery.mast not available")
            return pd.DataFrame()

        try:
            print(f"[MAST] Searching for '{target}'"
                  f"{f' [{mission}]' if mission else ''}"
                  f"{f' [{instrument}]' if instrument else ''}")

            # Build criteria dict
            criteria = {}
            if mission:
                criteria["obs_collection"] = mission.upper()
            if instrument:
                criteria["instrument_name"] = self._instrument_criterion(instrument)

            if criteria:
                # Use query_criteria with target coordinates
                ra, dec = _resolve_simbad_cached(target)
                if ra is None:
                    print(f"[MAST] Could not resolve target '{target}' via SIMBAD")
                    return self._failed(f"could not resolve target {target!r} to coordinates (SIMBAD); "
                                        "retry with ra/dec or another name")

                from astropy.coordinates import SkyCoord
                import astropy.units as u
                coord = SkyCoord(ra=ra, dec=dec, unit="deg")

                # Parse radius
                radius_val = self._parse_radius(radius)

                obs = _mast_query(Observations.query_criteria,
                    coordinates=coord,
                    radius=radius_val,
                    **criteria
                )
            else:
                # Simple target query
                obs = _mast_query(Observations.query_object, target, radius=radius)

            if obs is None or len(obs) == 0:
                print(f"[MAST] No results found for '{target}'")
                return pd.DataFrame()

            df = obs.to_pandas()

            # Limit results
            if len(df) > max_results:
                df = df.head(max_results)

            print(f"[MAST] Found {len(df)} observations")
            return self._standardize_columns(df)

        except Exception as e:
            print(f"[MAST] Search error: {e}")
            import traceback
            traceback.print_exc()
            return self._failed(f"{type(e).__name__}: {e}")

    def search_by_position(self, ra: float, dec: float, radius_arcmin: float = 1.0,
                           mission: str = None, instrument: str = None,
                           max_results: int = 500) -> pd.DataFrame:
        """
        Cone search by RA/Dec with optional mission/instrument filters.

        Args:
            ra: Right Ascension in degrees (ICRS)
            dec: Declination in degrees (ICRS)
            radius_arcmin: Search radius in arcminutes
            mission: Filter by mission
            instrument: Filter by instrument
            max_results: Maximum results
        """
        if not MAST_AVAILABLE:
            return pd.DataFrame()

        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            radius = radius_arcmin * u.arcmin

            criteria = {}
            if mission:
                criteria["obs_collection"] = mission.upper()
            if instrument:
                criteria["instrument_name"] = self._instrument_criterion(instrument)

            print(f"[MAST] Cone search: RA={ra:.4f}, Dec={dec:.4f}, "
                  f"radius={radius_arcmin}'")

            obs = _mast_query(Observations.query_criteria,
                coordinates=coord,
                radius=radius,
                **criteria
            )

            if obs is None or len(obs) == 0:
                return pd.DataFrame()

            df = obs.to_pandas()
            if len(df) > max_results:
                df = df.head(max_results)

            print(f"[MAST] Found {len(df)} observations")
            return self._standardize_columns(df)

        except Exception as e:
            print(f"[MAST] Position search error: {e}")
            out = pd.DataFrame()
            out.attrs["error"] = f"{type(e).__name__}: {e}"
            return out

    def search_by_criteria(self, mission: str = None, instrument: str = None,
                           proposal_id: str = None, filters: str = None,
                           target_name: str = None, dataproduct_type: str = None,
                           date_range: Tuple[str, str] = None,
                           max_results: int = 500, ra: float = None, dec: float = None,
                           radius: str = None, exptime_range: Tuple[Optional[float], Optional[float]] = None
                           ) -> pd.DataFrame:
        """
        Advanced criteria-based search with rich filtering.

        Args:
            mission: Mission name (JWST, HST, etc.)
            instrument: Instrument name (a bare name matches every mode)
            proposal_id: Specific proposal/program ID
            filters: Filter name (e.g., 'F200W', 'F444W')
            target_name: Target name (resolves to coordinates internally)
            dataproduct_type: 'image', 'spectrum', 'cube', etc.
            date_range: Tuple of (start_date, end_date) as ISO strings
                (converted to MJD: MAST's t_min is an MJD)
            max_results: Maximum results
            ra, dec: position for a cone (instead of target_name)
            radius: cone radius, e.g. '30s' or '2m' (default 30 arcsec)
            exptime_range: (min, max) exposure time in seconds; either may be None
        """
        if not MAST_AVAILABLE:
            return pd.DataFrame()

        try:
            criteria = {}

            if mission:
                criteria["obs_collection"] = mission.upper()
            if instrument:
                criteria["instrument_name"] = self._instrument_criterion(instrument)
            if proposal_id:
                criteria["proposal_id"] = proposal_id
            if filters:
                criteria["filters"] = filters
            if dataproduct_type:
                criteria["dataproduct_type"] = dataproduct_type
            if ra is not None and dec is not None:
                from astropy.coordinates import SkyCoord
                criteria["coordinates"] = SkyCoord(ra=float(ra), dec=float(dec), unit="deg")
                criteria["radius"] = self._parse_radius(radius or "30s")
            elif target_name:
                criteria["objectname"] = target_name
                criteria["radius"] = self._parse_radius(radius or "30s")
            if date_range and len(date_range) == 2:
                criteria["t_min"] = [self._to_mjd(date_range[0]), self._to_mjd(date_range[1])]
            if exptime_range and any(v is not None for v in exptime_range):
                lo, hi = exptime_range
                criteria["t_exptime"] = [float(lo) if lo is not None else 0.0,
                                         float(hi) if hi is not None else 1.0e9]

            if not criteria:
                print("[MAST] No search criteria provided")
                return pd.DataFrame()

            criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
            print(f"[MAST] Criteria search: {criteria_str}")

            obs = _mast_query(Observations.query_criteria, **criteria)

            if obs is None or len(obs) == 0:
                print("[MAST] No results found")
                return pd.DataFrame()

            df = obs.to_pandas()
            if len(df) > max_results:
                df = df.head(max_results)

            print(f"[MAST] Found {len(df)} observations")
            return self._standardize_columns(df)

        except Exception as e:
            print(f"[MAST] Criteria search error: {e}")
            import traceback
            traceback.print_exc()
            return self._failed(f"{type(e).__name__}: {e}")

    def get_product_list(self, observations: pd.DataFrame,
                         productType: str = None,
                         extension: str = None) -> pd.DataFrame:
        """
        Get file-level product list for observations.

        Args:
            observations: DataFrame from a previous MAST search
            productType: Filter by type ('SCIENCE', 'CALIBRATION', 'PREVIEW')
            extension: Filter by file extension ('fits', 'jpg', etc.)

        Returns:
            DataFrame with file-level product info (filenames, sizes, URLs)
        """
        if not MAST_AVAILABLE or observations.empty:
            return pd.DataFrame()

        try:
            from astropy.table import Table

            # Convert back to astropy table for MAST API
            obs_table = Table.from_pandas(observations)

            products = _mast_query(Observations.get_product_list, obs_table)

            if products is None or len(products) == 0:
                return pd.DataFrame()

            df = products.to_pandas()

            # Apply filters
            if productType:
                df = df[df["productType"].str.upper() == productType.upper()]
            if extension:
                df = df[df["productFilename"].str.endswith(f".{extension}")]

            print(f"[MAST] Found {len(df)} data products")
            return df

        except Exception as e:
            print(f"[MAST] Product list error: {e}")
            return pd.DataFrame()

    # L9 (tmp/scan-2026-07-17): period-fold/lightcurve turns stalled to the
    # no-progress watchdog while download_products blocked on TESS FITS pulls
    # with no timebox. Astroquery performs the transfer internally (no
    # response object exists here to watchdog-close), so the bound must come
    # from OUTSIDE the call — a daemon worker joined with a wall budget, the
    # same pattern as services/datalab_image_service.py::_download_fits.
    DOWNLOAD_WALL_DEFAULT_SECONDS = 480.0
    # Ceiling on accepted MAST_DOWNLOAD_WALL_SECONDS values (guard CX-03):
    # download_mast_data runs under a 600 s tool-guard override
    # (core/agent.py _TOOL_TIMEOUT_OVERRIDES). Clamping to 540 s keeps 60 s
    # of headroom so the specific structured MAST timeout always fires — and
    # reaches the model — before the generic guard timeout can swallow it.
    DOWNLOAD_WALL_MAX_SECONDS = 540.0

    def _download_wall_seconds(self) -> float:
        """Total wall-clock budget for one download_products transfer.

        Malformed/NaN/inf/negative MAST_DOWNLOAD_WALL_SECONDS values fall
        back to the derived default instead of disabling or crashing
        enforcement (same contract as DATALAB_IMAGE_DOWNLOAD_WALL_SECONDS).
        Accepted values are clamped to DOWNLOAD_WALL_MAX_SECONDS so the
        MAST-specific timeout message always beats the 600 s tool guard
        (CX-03).
        """
        import math
        import os

        try:
            wall = float(os.getenv("MAST_DOWNLOAD_WALL_SECONDS", "0") or 0)
        except ValueError:
            wall = 0.0
        if not math.isfinite(wall) or wall <= 0:
            wall = 0.0
        wall = min(wall, self.DOWNLOAD_WALL_MAX_SECONDS)
        return wall or self.DOWNLOAD_WALL_DEFAULT_SECONDS

    def _download_products_bounded(self, prod_table, download_dir: str,
                                    downloader=None):
        """Run astroquery's downloader on a worker thread joined with the
        wall budget (L9). On expiry the daemon worker is abandoned and an
        honest TimeoutError is raised, so a trickling/hung MAST transfer can
        no longer outlive even the tool guard's 600 s budget. Abandoned
        workers are capped by _DOWNLOAD_SLOTS (CX-01): each holds a permit
        until astroquery returns, and when every slot is held the next call
        fails fast instead of stacking another thread."""
        wall_seconds = self._download_wall_seconds()
        if downloader is None:
            def downloader(table, ddir):
                return Observations.download_products(table, download_dir=ddir)
        box: Dict[str, Any] = {}
        if not _DOWNLOAD_SLOTS.acquire(blocking=False):
            raise TimeoutError(
                "MAST download refused: all download worker slots are held "
                "by previous unresponsive transfers. The archive is not "
                "keeping up — do NOT retry immediately."
            )

        def _run():
            try:
                box["manifest"] = downloader(prod_table, download_dir)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                box["exc"] = exc
            finally:
                _DOWNLOAD_SLOTS.release()

        worker = threading.Thread(
            target=_run, daemon=True, name="quasar-mast-download"
        )
        try:
            worker.start()
        except BaseException:
            # A failed start (e.g. "can't start new thread" under the very
            # exhaustion this cap contains) means _run() never executes — the
            # permit must be handed back here or it leaks forever (mirrors
            # datalab_image_service CX-34).
            _DOWNLOAD_SLOTS.release()
            raise
        worker.join(wall_seconds)
        if worker.is_alive():
            raise TimeoutError(
                f"MAST product download exceeded its {wall_seconds:.0f}s "
                "wall-clock deadline while the transfer was still running "
                "(hung or trickling; worker abandoned). Partial files may "
                f"remain under {download_dir}."
            )
        if "exc" in box:
            raise box["exc"]
        return box.get("manifest")

    @staticmethod
    def _make_call_dir(base_dir: str) -> str:
        """Create a unique per-call download directory under base_dir (CX-02).

        A timed-out worker is abandoned but keeps writing into ITS call's
        directory, so a later call must NEVER be handed an existing one:
        the FULL 128-bit uuid4 hex (not a truncated fragment) plus
        ``exist_ok=False`` makes silent reuse impossible. On the
        astronomically-unlikely name collision, retry once with a fresh
        uuid before giving up.
        """
        import os

        last_exc: Optional[BaseException] = None
        for _ in range(2):
            call_dir = os.path.join(base_dir, f"dl-{uuid.uuid4().hex}")
            try:
                os.makedirs(call_dir, exist_ok=False)
            except FileExistsError as exc:
                last_exc = exc
                continue
            return call_dir
        raise FileExistsError(
            f"could not create a unique MAST download directory under "
            f"{base_dir} after 2 attempts"
        ) from last_exc

    def download_products(self, products: pd.DataFrame = None,
                          observations: pd.DataFrame = None,
                          download_dir: str = None,
                          productType: str = "SCIENCE",
                          extension: str = "fits",
                          max_files: int = 10) -> Dict[str, Any]:
        """
        Download FITS files and data products from MAST.

        Args:
            products: DataFrame from get_product_list (preferred)
            observations: DataFrame from search (will get products first)
            download_dir: Directory to save files (default ~/quasar_data/mast/)
            productType: Filter by type: 'SCIENCE', 'CALIBRATION', 'PREVIEW'
            extension: Filter by extension: 'fits', 'jpg', etc.
            max_files: Maximum number of files to download (safety limit)

        Returns:
            Dict with download paths, file count, and total size
        """
        if not MAST_AVAILABLE:
            return {"success": False, "error": "astroquery.mast not available"}

        import os
        base_dir = download_dir or os.path.join(
            os.path.expanduser("~"), "quasar_data", "mast"
        )

        try:
            from astropy.table import Table

            # Get products if only observations provided
            if products is None or products.empty:
                if observations is None or observations.empty:
                    return {"success": False, "error": "No data to download. Run search_mast and get_mast_products first."}
                products = self.get_product_list(observations,
                                                 productType=productType,
                                                 extension=extension)

            if products.empty:
                return {"success": False, "error": "No matching data products found."}

            # Apply safety limit
            if len(products) > max_files:
                print(f"[MAST] Limiting download to {max_files} files (of {len(products)} available)")
                products = products.head(max_files)

            # CX-02: every call transfers into its own per-call subdirectory.
            # A timed-out worker is abandoned but keeps writing (L9) — a
            # shared directory would let those late writes collide with a
            # retry or a later call. The returned manifest/paths point inside
            # this per-call directory, so the caller only ever sees files
            # from ITS transfer. Created only once there is actually data to
            # download, so no-data early returns never leak empty dirs.
            download_dir = self._make_call_dir(base_dir)

            # Convert to astropy table for MAST API
            prod_table = Table.from_pandas(products)

            print(f"[MAST] Downloading {len(products)} files to {download_dir}")

            # L9: wall-clock-bounded from outside — astroquery's downloader
            # has no per-transfer deadline of its own.
            manifest = self._download_products_bounded(prod_table, download_dir)

            if manifest is None:
                return {"success": False, "error": "Download returned no results"}

            manifest_df = manifest.to_pandas()
            downloaded = manifest_df[manifest_df["Status"] == "COMPLETE"] if "Status" in manifest_df.columns else manifest_df

            paths = list(downloaded["Local Path"].values) if "Local Path" in downloaded.columns else []

            print(f"[MAST] Downloaded {len(downloaded)} files")

            return {
                "success": True,
                "downloaded_files": len(downloaded),
                "total_attempted": len(products),
                "download_dir": download_dir,
                "file_paths": paths[:20],  # Limit paths in response
                "note": f"Downloaded {len(downloaded)} files to {download_dir}"
            }

        except TimeoutError as e:
            # L9: structured timeout — core/runner.py keys step closure and
            # SSE deadline exclusion on timeout: True
            # (_result_indicates_timeout).
            print(f"[MAST] Download timed out: {e}")
            return {
                "success": False,
                "timeout": True,
                "error": (
                    f"MAST download TIMEOUT: {e} "
                    "Do NOT retry this exact call — the archive transfer is "
                    "not keeping up. Reduce max_files or narrow the product "
                    "filters, or answer from data already downloaded."
                ),
            }
        except Exception as e:
            print(f"[MAST] Download error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Download failed: {str(e)}"}

    def _parse_radius(self, radius_str: str):
        """Parse radius string like '30s', '1m', '0.5d' into astropy quantity."""
        import astropy.units as u

        if isinstance(radius_str, (int, float)):
            return radius_str * u.deg

        radius_str = str(radius_str).strip().lower()
        if radius_str.endswith("s"):
            return float(radius_str[:-1]) * u.arcsec
        elif radius_str.endswith("m"):
            return float(radius_str[:-1]) * u.arcmin
        elif radius_str.endswith("d"):
            return float(radius_str[:-1]) * u.deg
        else:
            try:
                return float(radius_str) * u.arcsec
            except ValueError:
                return 30 * u.arcsec

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize MAST output columns to match Quasar's expected format.
        Aligns with the same column conventions as ALMA data cards.
        """
        if df.empty:
            return df

        # Rename key columns for consistency
        rename_map = {
            "s_ra": "s_ra",
            "s_dec": "s_dec",
            "target_name": "target_name",
            "obs_collection": "telescope",
            "instrument_name": "instrument_name",
            "filters": "filters",
            "proposal_id": "project_code",
            "t_exptime": "t_exptime",
            "dataproduct_type": "dataproduct_type",
            "obs_id": "obs_id",
            "calib_level": "calib_level",
            "t_min": "t_min",
            "t_max": "t_max",
        }

        df = df.rename(columns={k: v for k, v in rename_map.items()
                                if k in df.columns and k != v})

        # Ensure telescope column exists
        if "telescope" not in df.columns and "obs_collection" in df.columns:
            df["telescope"] = df["obs_collection"]
        elif "telescope" not in df.columns:
            df["telescope"] = "MAST"

        return df
