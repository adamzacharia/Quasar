"""Radio continuum SED compilation, spectral-index fitting, and plotting."""

from __future__ import annotations

import io
import math
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from services.plotting import PlottingService


RADIO_SED_DEFAULT_BASE_URL = "https://vizier.cds.unistra.fr/viz-bin/conesearch"
RADIO_SED_DEFAULT_RADIUS_ARCSEC = 30.0
RADIO_SED_MAX_RADIUS_ARCSEC = 120.0
RADIO_SED_DEFAULT_TIMEOUT = 30.0

# id, VizieR path, freq MHz, flux cols, unit-to-mJy, beam arcsec, epoch, err cols
SURVEY_REGISTRY = [
    ("TGSS", "J/A+A/598/A78/table3", 150.0, ["Stotal"], 1.0, 25.0, 2016, ["e_Stotal"]),
    ("GLEAM", "VIII/100/gleamegc", 200.0, ["Fintwide"], 1000.0, 120.0, 2014, ["e_Fintwide"]),
    ("SUMSS", "VIII/81B/sumss212", 843.0, ["St"], 1.0, 45.0, 2003, ["e_St"]),
    ("NVSS", "VIII/65/nvss", 1400.0, ["S1.4"], 1.0, 45.0, 1995, ["e_S1.4"]),
    ("FIRST", "VIII/92/first14", 1400.0, ["Fint"], 1.0, 5.0, 2000, []),
]


class RadioSedService:
    """Timeout-bound VizieR radio SED service for compact-source catalog fluxes."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        plotting_service: Optional[PlottingService] = None,
        fetcher: Optional[Callable[[str, Dict[str, Any], float], bytes]] = None,
    ):
        env_base = os.getenv("RADIO_SED_BASE_URL") or os.getenv("VIZIER_CONESEARCH_BASE_URL")
        self.base_url = str(base_url or env_base or RADIO_SED_DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout if timeout is not None else _env_float("RADIO_SED_TIMEOUT", RADIO_SED_DEFAULT_TIMEOUT))
        self.plotting_service = plotting_service or PlottingService()
        self.fetcher = fetcher or self._fetch_bytes

    def compile_sed(self, ra: Any, dec: Any, radius_arcsec: Any = RADIO_SED_DEFAULT_RADIUS_ARCSEC) -> Dict[str, Any]:
        """Query configured radio catalogs and return one nearest positive-flux point per survey."""
        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            radius, warnings = _normalize_radius(radius_arcsec)
            sr_deg = radius / 3600.0
            points: List[Dict[str, Any]] = []
            survey_status: List[Dict[str, Any]] = []

            for survey in SURVEY_REGISTRY:
                survey_id, catalog_path, freq_mhz, flux_cols, scale_mjy, resolution, epoch, err_cols = survey
                url = self._survey_url(catalog_path)
                params = {"RA": ra_f, "DEC": dec_f, "SR": sr_deg, "VERB": 2}
                status = {
                    "survey": survey_id,
                    "catalog_path": catalog_path,
                    "url": url,
                    "params": dict(params),
                    "status": "queried",
                }
                try:
                    content = self.fetcher(url, params, self.timeout)
                    votable, table = _parse_votable(content)
                    row_count = int(len(table))
                    status["row_count"] = row_count
                    if row_count == 0:
                        status["status"] = "empty"
                        survey_status.append(status)
                        continue

                    idx, sep_arcsec, sep_warning = _nearest_row_index(votable, table, ra_f, dec_f)
                    if sep_warning:
                        warnings.append(f"{survey_id}: {sep_warning}")
                    if row_count > 1:
                        warnings.append(
                            f"{survey_id} returned {row_count} candidates within {radius:g} arcsec; "
                            "using the nearest row and flagging possible confusion."
                        )

                    flux_col, flux_raw = _first_positive_column(table, idx, flux_cols)
                    if flux_col is None or flux_raw is None:
                        tried = ", ".join(flux_cols)
                        warnings.append(f"{survey_id}: no positive unmasked flux column found; tried {tried}.")
                        status["status"] = "skipped_missing_flux"
                        status["tried_flux_columns"] = list(flux_cols)
                        survey_status.append(status)
                        continue

                    err_col, err_raw = _first_positive_column(table, idx, err_cols)
                    flux_mjy = float(flux_raw * scale_mjy)
                    err_mjy = float(err_raw * scale_mjy) if err_raw is not None else None
                    point = {
                        "survey": str(survey_id),
                        "freq_mhz": float(freq_mhz),
                        "flux_mjy": flux_mjy,
                        "flux_err_mjy": err_mjy,
                        "sep_arcsec": sep_arcsec,
                        "resolution_arcsec": float(resolution),
                        "epoch": int(epoch),
                        "candidate_count": row_count,
                    }
                    points.append(point)
                    status.update(
                        {
                            "status": "matched",
                            "selected_row": int(idx),
                            "sep_arcsec": sep_arcsec,
                            "flux_column": flux_col,
                            "err_column": err_col,
                        }
                    )
                    survey_status.append(status)
                except Exception as exc:
                    warnings.append(f"{survey_id}: skipped {catalog_path} after VizieR failure: {exc}")
                    status["status"] = "failed"
                    status["error"] = str(exc)
                    survey_status.append(status)

            if not points and survey_status and all(row.get("status") == "failed" for row in survey_status):
                warnings.append("All radio catalog queries failed; this is not evidence of a radio nondetection.")

            return {
                "success": True,
                "points": points,
                "count": len(points),
                "warnings": warnings,
                "provenance": {
                    "service": "VizieR radio continuum cone searches",
                    "base_url": self.base_url,
                    "ra": ra_f,
                    "dec": dec_f,
                    "radius_arcsec": radius,
                    "sr_deg": sr_deg,
                    "survey_status": survey_status,
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def fit_spectral_index(self, points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Fit S proportional to nu^alpha in log10 space, preserving caveats as flags."""
        try:
            warnings: List[str] = []
            flags: List[str] = []
            valid_points, dropped = _valid_points(points)
            if dropped:
                warnings.append(f"Skipped {dropped} point(s) with nonpositive or non-finite frequency/flux.")
            if len(valid_points) < 2:
                return {
                    "success": False,
                    "error": "Spectral-index fit needs at least 2 valid positive-flux points.",
                    "n_points": len(valid_points),
                    "flags": flags,
                    "warnings": warnings,
                    "provenance": {"service": "Radio SED spectral-index fit"},
                }

            x = np.asarray([math.log10(float(p["freq_mhz"])) for p in valid_points], dtype=float)
            y = np.asarray([math.log10(float(p["flux_mjy"])) for p in valid_points], dtype=float)
            if int(np.unique(np.round(x, 12)).size) < 2:
                return {
                    "success": False,
                    "error": "Spectral-index fit needs at least two distinct frequencies.",
                    "n_points": len(valid_points),
                    "flags": flags,
                    "warnings": warnings,
                    "provenance": {"service": "Radio SED spectral-index fit"},
                }

            sigma_log = _log_flux_sigmas(valid_points)
            min_errors = int(math.ceil(0.7 * len(valid_points)))
            n_errors = int(np.isfinite(sigma_log).sum())
            weighted = n_errors >= min_errors
            if weighted:
                finite = sigma_log[np.isfinite(sigma_log) & (sigma_log > 0)]
                fill = float(np.median(finite)) if finite.size else 1.0
                sigma_used = np.where(np.isfinite(sigma_log) & (sigma_log > 0), sigma_log, fill)
                if n_errors < len(valid_points):
                    warnings.append("Some radio flux errors are missing; filled them with the median log uncertainty for weighting.")
                weights = 1.0 / np.square(sigma_used)
            else:
                sigma_used = np.full(len(valid_points), np.nan, dtype=float)
                weights = np.ones(len(valid_points), dtype=float)
                warnings.append("Fewer than 70% of radio SED points have flux errors; fit is unweighted.")

            alpha, intercept, alpha_err, chi2_red, residuals = _linear_fit(x, y, weights, weighted)
            s_1400 = float(10 ** (alpha * math.log10(1400.0) + intercept))
            if alpha_err is None and len(valid_points) == 2:
                warnings.append("Only two valid frequency points; alpha_err is not estimated.")

            flags.extend(_fit_flags(valid_points, residuals, sigma_used, weighted))

            return {
                "success": True,
                "alpha": float(alpha),
                "alpha_err": alpha_err,
                "chi2_red": chi2_red,
                "n_points": len(valid_points),
                "s_1400_mjy_predicted": s_1400,
                "flags": flags,
                "warnings": warnings,
                "provenance": {
                    "service": "Radio SED spectral-index fit",
                    "weighted": bool(weighted),
                    "n_errors": n_errors,
                    "n_points": len(valid_points),
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def plot_sed(self, points: Sequence[Dict[str, Any]], fit: Optional[Dict[str, Any]] = None, title: str = "Radio SED") -> Dict[str, Any]:
        """Render a log-log radio continuum SED plot."""
        try:
            warnings: List[str] = []
            valid_points, dropped = _valid_points(points)
            if dropped:
                warnings.append(f"Skipped {dropped} point(s) with nonpositive or non-finite frequency/flux before plotting.")
            if not valid_points:
                return {"success": False, "error": "No positive radio SED points to plot.", "warnings": warnings}

            valid_points = sorted(valid_points, key=lambda row: float(row["freq_mhz"]))
            freq = np.asarray([float(row["freq_mhz"]) for row in valid_points], dtype=float)
            flux = np.asarray([float(row["flux_mjy"]) for row in valid_points], dtype=float)
            err = np.asarray([
                float(row["flux_err_mjy"]) if _positive_float(row.get("flux_err_mjy")) is not None else np.nan
                for row in valid_points
            ])

            plt = self.plotting_service._apply_style(dark=False)
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            has_err = np.isfinite(err) & (err > 0)
            if has_err.any():
                ax.errorbar(
                    freq[has_err],
                    flux[has_err],
                    yerr=err[has_err],
                    fmt="o",
                    markersize=5,
                    capsize=3,
                    color="#0072B2",
                    ecolor="#555555",
                    label="Catalog flux",
                )
            if (~has_err).any():
                ax.scatter(freq[~has_err], flux[~has_err], s=28, color="#D55E00", label="Catalog flux (no error)")

            for row in valid_points:
                ax.annotate(
                    str(row.get("survey") or ""),
                    (float(row["freq_mhz"]), float(row["flux_mjy"])),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

            if fit and fit.get("success") and fit.get("alpha") is not None and fit.get("s_1400_mjy_predicted") is not None:
                alpha = float(fit["alpha"])
                s_1400 = float(fit["s_1400_mjy_predicted"])
                x_model = np.geomspace(max(float(freq.min()) * 0.75, 1e-6), float(freq.max()) * 1.25, 200)
                y_model = s_1400 * np.power(x_model / 1400.0, alpha)
                alpha_err = _positive_float(fit.get("alpha_err"))
                label = f"alpha = {alpha:.2f}" + (f" +/- {alpha_err:.2f}" if alpha_err is not None else "")
                ax.plot(x_model, y_model, "--", color="#009E73", label=label)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Flux density (mJy)")
            ax.set_title(str(title or "Radio SED"))
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            render = self.plotting_service._save_and_encode(fig, f"radio_sed_{uuid.uuid4().hex[:10]}")
            return {
                "success": True,
                "image_base64": render.get("base64_png"),
                "path": render.get("web_url"),
                "png_path": render.get("png_path"),
                "pdf_path": render.get("pdf_path"),
                "n_points": len(valid_points),
                "warnings": warnings,
                "provenance": {"service": "Radio SED plot", "title": str(title or "Radio SED")},
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _survey_url(self, catalog_path: str) -> str:
        return f"{self.base_url}/{str(catalog_path).lstrip('/')}"

    @staticmethod
    def _fetch_bytes(url: str, params: Dict[str, Any], timeout: float) -> bytes:
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"VizieR cone search request failed: {exc}") from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            text = str(getattr(response, "text", "") or "")[:200]
            raise RuntimeError(f"VizieR cone search returned HTTP {status}: {text}")
        content = bytes(getattr(response, "content", b"") or b"")
        if not content:
            raise RuntimeError("VizieR cone search returned an empty response.")
        return content


def _parse_votable(content: bytes) -> Tuple[Any, Any]:
    from astropy.io.votable import parse_single_table

    votable = parse_single_table(io.BytesIO(bytes(content)))
    return votable, votable.to_table(use_names_over_ids=True)


def _nearest_row_index(votable: Any, table: Any, ra: float, dec: float) -> Tuple[int, Optional[float], Optional[str]]:
    colnames = list(getattr(table, "colnames", []))
    if "_r" in colnames:
        distances = [_positive_or_zero_float(table["_r"][idx]) for idx in range(len(table))]
        finite = [(idx, value) for idx, value in enumerate(distances) if value is not None]
        if finite:
            idx, arcmin = min(finite, key=lambda item: item[1])
            return int(idx), float(arcmin * 60.0), None

    ra_col, dec_col = _position_columns(votable, colnames)
    if ra_col and dec_col:
        distances = []
        for idx in range(len(table)):
            row_ra = _finite_float(table[ra_col][idx])
            row_dec = _finite_float(table[dec_col][idx])
            if row_ra is None or row_dec is None:
                distances.append(None)
            else:
                distances.append(_angular_separation_arcsec(ra, dec, row_ra, row_dec))
        finite = [(idx, value) for idx, value in enumerate(distances) if value is not None]
        if finite:
            idx, sep = min(finite, key=lambda item: item[1])
            return int(idx), float(sep), "VOTable lacked _r; computed separation from UCD position columns."

    return 0, None, "VOTable lacked _r and usable UCD position columns; using first row."


def _position_columns(votable: Any, colnames: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    names = set(colnames)
    ra_col = None
    dec_col = None
    for field in getattr(votable, "fields", []) or []:
        name = getattr(field, "name", None) or getattr(field, "ID", None)
        if name not in names:
            continue
        ucd = str(getattr(field, "ucd", "") or "").lower()
        if ra_col is None and "pos.eq.ra" in ucd:
            ra_col = str(name)
        if dec_col is None and "pos.eq.dec" in ucd:
            dec_col = str(name)
    return ra_col, dec_col


def _first_positive_column(table: Any, row_index: int, candidates: Sequence[str]) -> Tuple[Optional[str], Optional[float]]:
    colnames = list(getattr(table, "colnames", []))
    for col in candidates:
        if col not in colnames:
            continue
        value = _positive_float(table[col][row_index])
        if value is not None:
            return str(col), value
    return None, None


def _linear_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray, weighted: bool) -> Tuple[float, float, Optional[float], Optional[float], np.ndarray]:
    design = np.column_stack([x, np.ones_like(x)])
    sqrt_w = np.sqrt(weights)
    lhs = design * sqrt_w[:, None]
    rhs = y * sqrt_w
    beta, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    residuals = y - design @ beta
    dof = int(len(x) - 2)
    alpha_err = None
    chi2_red = None
    try:
        normal_inv = np.linalg.inv(design.T @ (weights[:, None] * design))
        if dof > 0:
            if weighted:
                chi2 = float(np.sum(weights * np.square(residuals)))
                chi2_red = float(chi2 / dof)
                cov = normal_inv
            else:
                sigma2 = float(np.sum(np.square(residuals)) / dof)
                cov = normal_inv * sigma2
            alpha_var = float(cov[0, 0])
            if math.isfinite(alpha_var) and alpha_var >= 0:
                alpha_err = float(math.sqrt(alpha_var))
    except np.linalg.LinAlgError:
        alpha_err = None
    return float(beta[0]), float(beta[1]), alpha_err, chi2_red, residuals


def _fit_flags(points: Sequence[Dict[str, Any]], residuals: np.ndarray, sigma_log: np.ndarray, weighted: bool) -> List[str]:
    flags: List[str] = []
    resolutions = [_positive_float(row.get("resolution_arcsec")) for row in points]
    resolutions = [value for value in resolutions if value is not None]
    if resolutions and min(resolutions) > 0 and max(resolutions) / min(resolutions) > 3.0:
        flags.append(
            f"Beam sizes span {min(resolutions):g}-{max(resolutions):g} arcsec; "
            "extended sources may be resolved out in the high-resolution surveys."
        )

    epochs = [_finite_float(row.get("epoch")) for row in points]
    epochs = [value for value in epochs if value is not None]
    if epochs and max(epochs) - min(epochs) > 5:
        flags.append(f"Survey epochs span {int(min(epochs))}-{int(max(epochs))}; variability can bias a non-simultaneous radio spectral index.")

    by_survey = {str(row.get("survey")): row for row in points}
    if "NVSS" in by_survey and "FIRST" in by_survey:
        nvss = _positive_float(by_survey["NVSS"].get("flux_mjy"))
        first = _positive_float(by_survey["FIRST"].get("flux_mjy"))
        if nvss is not None and first is not None and (nvss + first) > 0:
            pct = abs(nvss - first) / ((nvss + first) / 2.0) * 100.0
            if pct > 30.0:
                flags.append(f"NVSS/FIRST 1.4 GHz fluxes differ by {pct:.0f}% - resolution or variability.")

    for row in points:
        count = _finite_float(row.get("candidate_count"))
        if count is not None and count > 1:
            flags.append(f"{row.get('survey')} returned {int(count)} radio candidates within the cone; the nearest match may be confused or blended.")

    if weighted:
        for row, resid, sig in zip(points, residuals, sigma_log):
            if math.isfinite(float(sig)) and float(sig) > 0 and abs(float(resid)) / float(sig) > 3.0:
                flags.append(f"{row.get('survey')} lies more than 3 sigma from the fitted radio power law.")
    elif len(points) > 3:
        scatter = float(np.std(residuals, ddof=2))
        if math.isfinite(scatter) and scatter > 0:
            for row, resid in zip(points, residuals):
                if abs(float(resid)) / scatter > 3.0:
                    flags.append(f"{row.get('survey')} lies more than 3 sigma from the fitted radio power law.")

    return flags


def _valid_points(points: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    valid: List[Dict[str, Any]] = []
    dropped = 0
    for row in points or []:
        if not isinstance(row, dict):
            dropped += 1
            continue
        freq = _positive_float(row.get("freq_mhz"))
        flux = _positive_float(row.get("flux_mjy"))
        if freq is None or flux is None:
            dropped += 1
            continue
        clean = dict(row)
        clean["freq_mhz"] = freq
        clean["flux_mjy"] = flux
        clean["flux_err_mjy"] = _positive_float(row.get("flux_err_mjy"))
        valid.append(clean)
    return valid, dropped


def _log_flux_sigmas(points: Sequence[Dict[str, Any]]) -> np.ndarray:
    sigmas = []
    for row in points:
        flux = _positive_float(row.get("flux_mjy"))
        err = _positive_float(row.get("flux_err_mjy"))
        if flux is None or err is None:
            sigmas.append(np.nan)
        else:
            sigmas.append(float(err / (flux * math.log(10.0))))
    return np.asarray(sigmas, dtype=float)


def _normalize_radius(radius_arcsec: Any) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    try:
        radius = float(radius_arcsec)
    except (TypeError, ValueError):
        radius = RADIO_SED_DEFAULT_RADIUS_ARCSEC
        warnings.append("radius_arcsec was not numeric; using 30 arcsec.")
    if not math.isfinite(radius) or radius <= 0:
        radius = RADIO_SED_DEFAULT_RADIUS_ARCSEC
        warnings.append("radius_arcsec must be positive; using 30 arcsec.")
    elif radius > RADIO_SED_MAX_RADIUS_ARCSEC:
        warnings.append(f"radius_arcsec {radius:g} exceeds 120; clamped to 120 arcsec.")
        radius = RADIO_SED_MAX_RADIUS_ARCSEC
    return float(radius), warnings


def _validate_coords(ra: Any, dec: Any) -> Tuple[float, float]:
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except (TypeError, ValueError) as exc:
        raise ValueError("ra and dec must be numeric ICRS degrees.") from exc
    if not math.isfinite(ra_f) or not math.isfinite(dec_f):
        raise ValueError("ra and dec must be finite ICRS degrees.")
    if not 0.0 <= ra_f < 360.0:
        raise ValueError("ra must satisfy 0 <= ra < 360 degrees.")
    if not -90.0 <= dec_f <= 90.0:
        raise ValueError("dec must satisfy -90 <= dec <= 90 degrees.")
    return ra_f, dec_f


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


def _angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1 = math.radians(ra1)
    d1 = math.radians(dec1)
    r2 = math.radians(ra2)
    d2 = math.radians(dec2)
    sin_ddec = math.sin((d2 - d1) / 2.0)
    sin_dra = math.sin((r2 - r1) / 2.0)
    a = sin_ddec * sin_ddec + math.cos(d1) * math.cos(d2) * sin_dra * sin_dra
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))) * 3600.0


def _positive_float(value: Any) -> Optional[float]:
    out = _finite_float(value)
    if out is None or out <= 0:
        return None
    return out


def _positive_or_zero_float(value: Any) -> Optional[float]:
    out = _finite_float(value)
    if out is None or out < 0:
        return None
    return out


def _finite_float(value: Any) -> Optional[float]:
    if np.ma.is_masked(value):
        return None
    try:
        if hasattr(value, "to_value"):
            value = value.to_value()
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


__all__ = ["RadioSedService", "SURVEY_REGISTRY"]

