"""
VLASS multi-epoch comparison via CADC.

Per-epoch VLASS Quicklook images are NOT published as HiPS (only the median
stack is), so epoch comparison goes through CADC instead:

  1. TAP query (argus) for VLASS Quicklook planes containing the position,
     grouped into epochs (VLASS1.1, VLASS1.2, VLASS2.1, ...);
  2. per-epoch SODA sync cutouts by science-artifact URI;
  3. a shared-normalization panel + per-epoch peak/RMS table, a simple
     variability verdict, and per-epoch frames for the blink comparator card.

Anonymous access; both endpoints verified live 2026-07-14.
"""

from __future__ import annotations

import gc
import io
import logging
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import requests

from services import plotting

logger = logging.getLogger(__name__)

CADC_TAP_SYNC_URL = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"
CADC_SODA_SYNC_URL = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/sync"

# VLASS Quicklook photometry carries known systematics (~15% at low flux,
# ~8% peak-flux low bias pre-2.1); flag variability conservatively.
QL_SYSTEMATIC_FRAC = 0.15
VARIABILITY_RATIO = 1.3
VARIABILITY_SIGMA = 5.0


class VlassEpochError(ValueError):
    """Raised for user-correctable VLASS epoch-comparison failures."""


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    try:
        ra_f, dec_f = float(ra), float(dec)
    except (TypeError, ValueError) as exc:
        raise VlassEpochError("ra and dec must be numeric ICRS degrees.") from exc
    if not (math.isfinite(ra_f) and math.isfinite(dec_f)):
        raise VlassEpochError("ra and dec must be finite ICRS degrees.")
    if not 0.0 <= ra_f < 360.0 or not -90.0 <= dec_f <= 90.0:
        raise VlassEpochError("ra must satisfy 0 <= ra < 360 and -90 <= dec <= 90.")
    if dec_f <= -40.0:
        raise VlassEpochError("VLASS covers Dec > -40 deg only.")
    return ra_f, dec_f


def _epoch_key(observation_id: str) -> str:
    """'VLASS2.1.T18t19.J133218+303000' -> 'VLASS2.1'."""
    parts = str(observation_id).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else str(observation_id)


def _tile_key(observation_id: str) -> str:
    """'VLASS2.1.T18t19.J133218+303000' -> 'T18t19.J133218+303000'."""
    parts = str(observation_id).split(".")
    return ".".join(parts[2:]) if len(parts) > 2 else ""


def _target_pixel(header: Any, ra: float, dec: float) -> Optional[Tuple[int, int]]:
    """(x, y) pixel of the target in a cutout, or None without a usable WCS."""
    try:
        from astropy.coordinates import SkyCoord
        from astropy.wcs import WCS
        import astropy.units as u

        wcs = WCS(header, naxis=2)
        if not wcs.has_celestial:
            return None
        x, y = wcs.world_to_pixel(SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"))
        return int(round(float(x))), int(round(float(y)))
    except Exception:
        return None


def _measure_at_target(
    data: np.ndarray, header: Any, ra: float, dec: float
) -> Optional[Tuple[float, int]]:
    """Peak within a box around the TARGET pixel (not the global cutout max,
    which a bright unrelated neighbor could dominate) plus the box half-width.

    Returns None when the target itself sits on blank (NaN-trimmed) pixels —
    the caller should try another subtile.
    """
    ny, nx = data.shape
    pix = _target_pixel(header, ra, dec)
    if pix is None:
        cx, cy = nx // 2, ny // 2  # cutouts are target-centered by construction
    else:
        cx, cy = pix
    if not (0 <= cx < nx and 0 <= cy < ny):
        return None
    # The target pixel itself must be finite — a subtile can be NaN-trimmed
    # exactly at the target while unrelated signal survives elsewhere (CX-17).
    if not np.isfinite(data[cy, cx]):
        return None
    # Peak search box: ~15 arcsec (QL pixels are 1"), at least 6 beam widths.
    try:
        cdelt = abs(float(header.get("CDELT2") or header.get("CDELT1") or 0)) * 3600.0
    except (TypeError, ValueError):
        cdelt = 0.0
    half = int(round(15.0 / cdelt)) if cdelt > 0 else 15
    half = max(8, min(half, max(nx, ny)))
    box = data[max(0, cy - half):cy + half + 1, max(0, cx - half):cx + half + 1]
    finite = box[np.isfinite(box)]
    if finite.size == 0:
        return None
    return float(np.max(finite)), half


class VlassEpochService:
    """Compare VLASS epochs at a sky position through CADC TAP + SODA."""

    def __init__(self, *, tap_url: Optional[str] = None, soda_url: Optional[str] = None,
                 timeout: Optional[float] = None):
        self.tap_url = str(tap_url or os.getenv("CADC_TAP_SYNC_URL") or CADC_TAP_SYNC_URL)
        self.soda_url = str(soda_url or os.getenv("CADC_SODA_SYNC_URL") or CADC_SODA_SYNC_URL)
        self.timeout = float(timeout if timeout is not None else _env_float("CADC_TAP_TIMEOUT", 90.0))

    # ── CADC queries ─────────────────────────────────────────────────────
    def _tap_csv(self, adql: str):
        import pandas as pd

        try:
            response = requests.post(
                self.tap_url,
                data={"LANG": "ADQL", "REQUEST": "doQuery", "RESPONSEFORMAT": "csv", "QUERY": adql},
                timeout=self.timeout,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            raise VlassEpochError(f"CADC TAP request failed: {exc}") from exc
        if int(response.status_code or 0) != 200:
            raise VlassEpochError(f"CADC TAP returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            return pd.read_csv(io.StringIO(response.text))
        except Exception as exc:
            raise VlassEpochError(f"CADC TAP returned unparseable CSV: {exc}") from exc

    def _query_planes(self, ra: float, dec: float):
        adql = (
            "SELECT o.observationID, p.productID, p.time_bounds_lower "
            "FROM caom2.Plane p JOIN caom2.Observation o ON p.obsID=o.obsID "
            "WHERE o.collection='VLASS' AND p.dataProductType='image' "
            f"AND p.productID LIKE '%.quicklook' "
            f"AND CONTAINS(POINT('ICRS', {ra:.8f}, {dec:.8f}), p.position_bounds)=1"
        )
        return self._tap_csv(adql)

    def _query_science_artifacts(self, observation_ids: List[str]) -> Dict[str, str]:
        """observationID -> science artifact URI."""
        if not observation_ids:
            return {}
        id_list = ", ".join("'" + str(o).replace("'", "") + "'" for o in observation_ids)
        adql = (
            "SELECT o.observationID, a.uri "
            "FROM caom2.Artifact a "
            "JOIN caom2.Plane p ON a.planeID=p.planeID "
            "JOIN caom2.Observation o ON p.obsID=o.obsID "
            f"WHERE o.observationID IN ({id_list}) AND a.productType='science' "
            "AND p.productID LIKE '%.quicklook'"
        )
        frame = self._tap_csv(adql)
        mapping: Dict[str, str] = {}
        for _, row in frame.iterrows():
            mapping.setdefault(str(row["observationID"]), str(row["uri"]))
        return mapping

    def _cutout_url(self, artifact_uri: str, ra: float, dec: float, radius_deg: float) -> str:
        return (
            f"{self.soda_url}?ID={quote(str(artifact_uri), safe='')}"
            f"&CIRCLE={ra:.8f}+{dec:.8f}+{radius_deg:.6f}"
        )

    @staticmethod
    def _register_epochs(images, warnings):
        """Reproject each epoch onto the first epoch's WCS so the blink panel
        compares identical sky pixels (CX-18). Best-effort: on any failure the
        original arrays are kept (they are already target-centered), with a note.

        images: list of (label, data, header); returns the same shape with
        each data array on the reference grid.
        """
        if len(images) < 2:
            return images
        try:
            from astropy.wcs import WCS
            from reproject import reproject_interp
        except Exception:
            warnings.append("reproject unavailable — epochs shown on native grids (may be slightly misregistered).")
            return images
        try:
            ref_label, ref_data, ref_header = images[0]
            ref_wcs = WCS(ref_header, naxis=2)
            if not ref_wcs.has_celestial:
                warnings.append("First epoch lacks a celestial WCS — epochs not co-registered.")
                return images
            out = [(ref_label, ref_data, ref_header)]
            for label, data, header in images[1:]:
                try:
                    wcs = WCS(header, naxis=2)
                    if not wcs.has_celestial:
                        raise ValueError("no celestial WCS")
                    reprojected, footprint = reproject_interp((data, wcs), ref_wcs, shape_out=ref_data.shape)
                    # A non-overlapping epoch reprojects to (near-)all NaN. Never
                    # emit that as a blink frame — keep the native cutout, which
                    # still has valid target data, and warn (CX-34).
                    overlap_frac = float(np.mean(footprint > 0)) if footprint is not None else 0.0
                    if overlap_frac < 0.2 or not np.isfinite(reprojected).any():
                        warnings.append(
                            f"{label}: reprojection onto the reference grid barely overlaps "
                            f"({overlap_frac * 100:.0f}%); shown on its native grid instead."
                        )
                        out.append((label, data, header))
                    else:
                        out.append((label, reprojected, ref_header))
                except Exception as exc:
                    warnings.append(f"{label}: reprojection failed ({exc}); shown on its native grid.")
                    out.append((label, data, header))
            return out
        except Exception as exc:
            warnings.append(f"Epoch co-registration skipped ({exc}).")
            return images

    # ── Main entry point ─────────────────────────────────────────────────
    def epoch_comparison(
        self,
        ra: Any,
        dec: Any,
        radius_arcsec: Any = 60.0,
        max_epochs: int = 6,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.io import fits as afits
        from astropy.stats import mad_std
        from astropy.time import Time
        from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval

        from services.fits_service import _download_fits

        temp_paths: List[str] = []
        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            try:
                radius = float(radius_arcsec)
            except (TypeError, ValueError):
                radius = 60.0
            radius = min(600.0, max(10.0, radius))
            radius_deg = radius / 3600.0
            max_epochs = int(max(2, min(8, int(max_epochs or 6))))

            planes = self._query_planes(ra_f, dec_f)
            if planes is None or planes.empty:
                return {"success": False,
                        "error": "No VLASS Quicklook coverage at this position (CADC returned no planes)."}

            # Group planes into epochs; keep a few candidate subtiles per epoch
            # (preferred tile first) because Quicklook subtiles have blank
            # (NaN-trimmed) edges — a plane can "contain" the point yet be
            # blank there, in which case the next subtile is tried.
            planes = planes.assign(
                epoch=planes["observationID"].map(_epoch_key),
                tile=planes["observationID"].map(_tile_key),
            )
            tile_counts = planes.groupby("tile")["epoch"].nunique().sort_values(ascending=False)
            best_tile = str(tile_counts.index[0]) if len(tile_counts) else ""
            candidates: Dict[str, List[Dict[str, Any]]] = {}
            for _, row in planes.sort_values("time_bounds_lower").iterrows():
                epoch = str(row["epoch"])
                entry = {
                    "observationID": str(row["observationID"]),
                    "tile": str(row["tile"]),
                    "mjd": float(row["time_bounds_lower"]),
                }
                bucket = candidates.setdefault(epoch, [])
                if entry["tile"] == best_tile:
                    bucket.insert(0, entry)
                else:
                    bucket.append(entry)
            epochs = sorted(
                ((epoch, bucket[:3]) for epoch, bucket in candidates.items()),
                key=lambda kv: kv[1][0]["mjd"],
            )[:max_epochs]
            uri_map = self._query_science_artifacts(
                [info["observationID"] for _, bucket in epochs for info in bucket]
            )

            warnings: List[str] = []
            measured: List[Dict[str, Any]] = []
            images: List[Tuple[str, np.ndarray]] = []
            for epoch, bucket in epochs:
                data = None
                info = None
                finite = None
                peak = None
                peak_box_half = None
                for candidate in bucket:
                    uri = uri_map.get(candidate["observationID"])
                    if not uri:
                        continue
                    cut_url = self._cutout_url(uri, ra_f, dec_f, radius_deg)
                    try:
                        path = _download_fits(cut_url, f"VLASS {epoch}")
                        temp_paths.append(path)
                        with afits.open(path, memmap=False) as hdul:
                            cand_data = None
                            cand_header = None
                            for hdu in hdul:
                                if hdu.data is not None and hdu.data.ndim >= 2:
                                    cand_data = np.asarray(hdu.data, dtype=float)
                                    cand_header = hdu.header
                                    while cand_data.ndim > 2:
                                        cand_data = cand_data[0]
                                    break
                            if cand_data is None:
                                raise ValueError("cutout has no image HDU")
                    except Exception as exc:
                        warnings.append(f"{epoch}/{candidate['tile']}: cutout failed ({exc}).")
                        continue
                    # "Usable" means the TARGET has data, not merely any pixel
                    # somewhere in the cutout (a subtile edge can clip the
                    # target while the far side of the cutout has signal).
                    measured_peak = _measure_at_target(cand_data, cand_header, ra_f, dec_f)
                    if measured_peak is None:
                        warnings.append(
                            f"{epoch}/{candidate['tile']}: target position is blank (subtile edge); trying next subtile."
                        )
                        continue
                    data, info, header_used = cand_data, candidate, cand_header
                    finite = cand_data[np.isfinite(cand_data)]
                    peak, peak_box_half = measured_peak
                    break
                if data is None or info is None or peak is None:
                    warnings.append(f"{epoch}: no usable cutout in any subtile; skipped.")
                    continue

                finite_frac = float(finite.size) / float(data.size) if data.size else 0.0
                if finite_frac < 0.2:
                    warnings.append(f"{epoch}: only {finite_frac * 100:.0f}% coverage in the cutout (subtile edge).")
                rms = float(mad_std(finite))
                date_iso = Time(info["mjd"], format="mjd").iso[:10]
                measured.append({
                    "epoch": epoch,
                    "observation_id": info["observationID"],
                    "mjd": round(info["mjd"], 3),
                    "date": date_iso,
                    "peak_jy_per_beam": round(peak, 6),
                    "rms_jy_per_beam": round(rms, 7),
                    "peak_snr": round(peak / rms, 1) if rms > 0 else None,
                    "finite_fraction": round(finite_frac, 3),
                })
                images.append((f"{epoch} ({date_iso})", data, header_used))

            if not images:
                return {"success": False,
                        "error": "All epoch cutouts were blank or failed — likely a subtile edge; "
                                 "try a slightly different position or larger radius.",
                        "warnings": warnings}

            # Reproject every epoch onto the first epoch's WCS so the blink
            # compares the SAME sky pixels — otherwise slightly different
            # Quicklook grids make a fixed source appear to shift (CX-18).
            images = self._register_epochs(images, warnings)

            # Variability verdict across epochs
            variability: Dict[str, Any] = {"n_epochs": len(measured)}
            if len(measured) >= 2:
                peaks = [m["peak_jy_per_beam"] for m in measured]
                rmses = [m["rms_jy_per_beam"] for m in measured]
                lo_i = int(np.argmin(peaks))
                hi_i = int(np.argmax(peaks))
                ratio = peaks[hi_i] / peaks[lo_i] if peaks[lo_i] > 0 else float("inf")
                noise = math.sqrt(
                    rmses[hi_i] ** 2 + rmses[lo_i] ** 2
                    + (QL_SYSTEMATIC_FRAC * peaks[hi_i]) ** 2
                    + (QL_SYSTEMATIC_FRAC * peaks[lo_i]) ** 2
                )
                sigma = abs(peaks[hi_i] - peaks[lo_i]) / noise if noise > 0 else 0.0
                variability.update({
                    "peak_ratio_max_min": round(ratio, 3),
                    "significance_sigma": round(sigma, 2),
                    "faintest_epoch": measured[lo_i]["epoch"],
                    "brightest_epoch": measured[hi_i]["epoch"],
                    "variable_candidate": bool(ratio > VARIABILITY_RATIO and sigma > VARIABILITY_SIGMA),
                    "note": (
                        "Peak-flux comparison includes a 15% Quicklook systematic term; "
                        "confirm candidates against Single-Epoch products before publication."
                    ),
                })

            # Shared-normalization panel + per-epoch blink frames
            sample = np.concatenate([img[np.isfinite(img)].ravel() for _, img, _ in images])
            norm = ImageNormalize(sample, interval=ZScaleInterval(), stretch=SqrtStretch())
            n = len(images)
            fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.9), facecolor="#0f172a", squeeze=False)
            frames: List[Dict[str, str]] = []
            os.makedirs(plotting.PLOT_OUTPUT_DIR, exist_ok=True)
            for k, (label, data, _hdr) in enumerate(images):
                ax = axes[0][k]
                ax.imshow(data, origin="lower", cmap="inferno", norm=norm)
                ax.set_title(label, color="white", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("black")
                # Individual frame for the blink card
                ffig, fax = plt.subplots(figsize=(4.2, 4.2), facecolor="black")
                fax.imshow(data, origin="lower", cmap="inferno", norm=norm)
                fax.set_axis_off()
                fname = f"vlass_epoch_{uuid.uuid4().hex[:8]}.png"
                fpath = os.path.join(plotting.PLOT_OUTPUT_DIR, fname)
                ffig.savefig(fpath, dpi=130, bbox_inches="tight", pad_inches=0.02, facecolor="black")
                plt.close(ffig)
                frames.append({"url": f"/plots/{fname}", "label": label})
            fig.suptitle(
                title or f"VLASS epochs at RA={ra_f:.5f}, Dec={dec_f:.5f}",
                color="white", fontsize=12,
            )
            panel_name = f"vlass_epochs_{uuid.uuid4().hex[:10]}.png"
            panel_path = os.path.join(plotting.PLOT_OUTPUT_DIR, panel_name)
            fig.savefig(panel_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            return {
                "success": True,
                "path": f"/plots/{panel_name}",
                "png_path": panel_path,
                "ra": ra_f,
                "dec": dec_f,
                "radius_arcsec": radius,
                "epochs": measured,
                "variability": variability,
                "frames": frames,
                "warnings": warnings,
                "provenance": {
                    "service": "CADC TAP (argus) + SODA cutouts, collection=VLASS Quicklook",
                    "tap_url": self.tap_url,
                    "soda_url": self.soda_url,
                },
            }
        except Exception as exc:
            logger.error(f"[VLASS-EPOCHS] epoch_comparison failed: {exc}")
            return {"success": False, "error": str(exc)}
        finally:
            for path in temp_paths:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            gc.collect()


__all__ = ["VlassEpochError", "VlassEpochService"]
