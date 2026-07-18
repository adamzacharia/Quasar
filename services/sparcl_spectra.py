"""NOIRLab SparCL spectrum search and plotting helpers."""

from __future__ import annotations

import importlib
import math
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from services import line_lists
from services.plotting import PlottingService


# DESI-EDR is superseded by DESI-DR1; BOSS/SDSS-DR16 are deprecated server-side.
DEFAULT_RELEASES = ["DESI-DR1", "SDSS-DR17"]
OUTFIELDS = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"]
RETRIEVE_FIELDS = ["wavelength", "flux", "ivar", "model", "redshift", "spectype"]
# Fields the constraint search accepts for spectype (SPARCL vocabulary).
KNOWN_SPECTYPES = {"GALAXY", "STAR", "QSO"}
MAX_SEARCH_LIMIT = 500
MAX_RETRIEVE_IDS = 50
RETRIEVE_INCLUDE_DEFAULT = [
    "sparcl_id", "wavelength", "flux", "ivar", "model",
    "redshift", "spectype", "data_release",
]
# Back-compat re-export: the curated line list moved to services/line_lists.py
# (uniform VACUUM wavelengths keyed by spectype — the old literal here mixed
# air and vacuum values). Same ordered (label, wavelength) tuple shape.
COMMON_LINES = line_lists.common_lines()
# Plotly spectra above this many points are stride-downsampled so the SSE
# payload stays lean; the PNG keeps full resolution.
PLOTLY_MAX_POINTS = 4000


class SparclSpectraService:
    """Lazy cached SparCL client wrapper for source search and spectrum plots."""

    def __init__(self, *, plotting_service: Optional[PlottingService] = None):
        self.plotting_service = plotting_service or PlottingService()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            module = importlib.import_module("sparcl.client")
            client_cls = getattr(module, "SparclClient")
        except Exception as exc:
            raise RuntimeError(f"SparCL unavailable: pip install sparclclient. Underlying error: {exc}") from exc
        try:
            self._client = client_cls()
        except Exception as exc:
            raise RuntimeError(f"SparCL client initialization failed: {exc}") from exc
        return self._client

    def find_spectra(
        self,
        ra: Any,
        dec: Any,
        radius_arcsec: Any = 60,
        data_release: Optional[Sequence[str] | str] = None,
        limit: Any = 20,
    ) -> Dict[str, Any]:
        try:
            ra_f, dec_f = _validate_coords(ra, dec)
            radius = _positive_float(radius_arcsec, 60.0)
            if radius > 3600.0:
                radius = 3600.0
                warnings = ["radius_arcsec exceeds 3600; clamped to 3600 arcsec."]
            else:
                warnings = []
            limit_i = _positive_int(limit, 20, 200)
            releases = _normalize_releases(data_release)
            client = self._get_client()
            ranges, range_warnings = _search_ranges(ra_f, dec_f, radius)
            warnings.extend(range_warnings)
            records: List[Dict[str, Any]] = []
            box_capped = False
            for ra_min, ra_max, dec_min, dec_max in ranges:
                result = client.find(
                    outfields=list(OUTFIELDS),
                    constraints={"data_release": releases, "ra": [ra_min, ra_max], "dec": [dec_min, dec_max]},
                    limit=limit_i,
                )
                box_records = _records_list(result)
                box_capped = box_capped or len(box_records) >= limit_i
                records.extend(box_records)
            if box_capped:
                # sparcl-box-limit-silent-truncation: the limit applies to the
                # remote bounding-box query BEFORE the local cone cut and
                # distance sort, so hitting the cap means the box was truncated
                # in server storage order — the true nearest spectrum may be
                # missing from the result.
                warnings.append(
                    f"The remote bounding-box query returned the row cap "
                    f"(limit={limit_i}) before the local cone filter; the result "
                    "may be truncated and nearest-neighbor ordering is not "
                    "guaranteed. Retry with a larger limit or smaller radius."
                )

            rows: List[Dict[str, Any]] = []
            seen = set()
            for record in records:
                row = _normalize_record(record)
                if row is None:
                    continue
                distance = _angular_distance_arcsec(ra_f, dec_f, row["ra"], row["dec"])
                if distance > radius:
                    continue
                sid = str(row.get("sparcl_id") or "")
                if sid and sid in seen:
                    continue
                if sid:
                    seen.add(sid)
                row["distance_arcsec"] = distance
                rows.append(row)
            rows.sort(key=lambda row: row.get("distance_arcsec", float("inf")))
            rows = rows[:limit_i]
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": warnings,
                "provenance": {
                    "service": "NOIRLab SparCL",
                    "data_release": releases,
                    "ra": ra_f,
                    "dec": dec_f,
                    "radius_arcsec": radius,
                    "limit": limit_i,
                    "query_boxes": [list(rng) for rng in ranges],
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def search_spectra(
        self,
        spectype: Optional[Sequence[str] | str] = None,
        redshift_min: Optional[float] = None,
        redshift_max: Optional[float] = None,
        data_release: Optional[Sequence[str] | str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius_arcsec: Optional[float] = None,
        extra_constraints: Optional[Dict[str, Any]] = None,
        limit: Any = 100,
    ) -> Dict[str, Any]:
        """Survey-scale SPARCL constraint search (the blog's ``client.find`` with
        physical constraints), with an optional cone to combine position+physics.

        Categorical constraints take a value list; numeric constraints take a
        ``[min, max]`` pair — the SPARCL find contract."""
        try:
            warnings: List[str] = []
            limit_i = _positive_int(limit, 100, MAX_SEARCH_LIMIT)
            releases = _normalize_releases(data_release)
            constraints: Dict[str, Any] = {"data_release": releases}

            spectypes = _normalize_spectypes(spectype)
            if spectypes:
                unknown = [s for s in spectypes if s not in KNOWN_SPECTYPES]
                if unknown:
                    raise ValueError(
                        f"Unknown spectype value(s) {unknown}; valid: {sorted(KNOWN_SPECTYPES)}."
                    )
                constraints["spectype"] = spectypes

            z_min = _float_or_none(redshift_min)
            z_max = _float_or_none(redshift_max)
            if z_min is not None or z_max is not None:
                lo = z_min if z_min is not None else -0.5
                hi = z_max if z_max is not None else 15.0
                if lo > hi:
                    raise ValueError("redshift_min must be <= redshift_max.")
                if z_min is None:
                    warnings.append("redshift_min not given; using -0.5.")
                if z_max is None:
                    warnings.append("redshift_max not given; using 15.")
                constraints["redshift"] = [lo, hi]

            if extra_constraints:
                for key, value in dict(extra_constraints).items():
                    constraints[str(key)] = value

            cone = ra is not None and dec is not None
            ranges: List[Tuple[float, float, float, float]] = []
            radius = 0.0
            if cone:
                ra_f, dec_f = _validate_coords(ra, dec)
                radius = _positive_float(radius_arcsec, 60.0)
                if radius > 3600.0:
                    radius = 3600.0
                    warnings.append("radius_arcsec exceeds 3600; clamped to 3600 arcsec.")
                ranges, range_warnings = _search_ranges(ra_f, dec_f, radius)
                warnings.extend(range_warnings)
            elif radius_arcsec is not None:
                warnings.append("radius_arcsec ignored: both ra and dec are required for a cone.")

            client = self._get_client()
            records: List[Dict[str, Any]] = []
            if cone:
                box_capped = False
                for ra_min, ra_max, dec_min, dec_max in ranges:
                    box = dict(constraints)
                    box["ra"] = [ra_min, ra_max]
                    box["dec"] = [dec_min, dec_max]
                    result = client.find(outfields=list(OUTFIELDS), constraints=box, limit=limit_i)
                    box_records = _records_list(result)
                    box_capped = box_capped or len(box_records) >= limit_i
                    records.extend(box_records)
                if box_capped:
                    # sparcl-box-limit-silent-truncation: see find_spectra — a
                    # capped box query truncates BEFORE the cone cut/sort.
                    warnings.append(
                        f"The remote bounding-box query returned the row cap "
                        f"(limit={limit_i}) before the local cone filter; the "
                        "result may be truncated and nearest-neighbor ordering "
                        "is not guaranteed. Retry with a larger limit or "
                        "smaller radius."
                    )
            else:
                result = client.find(outfields=list(OUTFIELDS), constraints=dict(constraints), limit=limit_i)
                records.extend(_records_list(result))

            rows: List[Dict[str, Any]] = []
            seen = set()
            for record in records:
                row = _normalize_record(record)
                if row is None:
                    continue
                sid = str(row.get("sparcl_id") or "")
                if sid and sid in seen:
                    continue
                if sid:
                    seen.add(sid)
                if cone:
                    distance = _angular_distance_arcsec(ra_f, dec_f, row["ra"], row["dec"])
                    if distance > radius:
                        continue
                    row["distance_arcsec"] = distance
                rows.append(row)
            if cone:
                rows.sort(key=lambda row: row.get("distance_arcsec", float("inf")))
            rows = rows[:limit_i]
            provenance: Dict[str, Any] = {
                "service": "NOIRLab SparCL",
                "constraints": _jsonable_constraints(constraints),
                "limit": limit_i,
            }
            if cone:
                provenance.update(
                    {"ra": ra_f, "dec": dec_f, "radius_arcsec": radius,
                     "query_boxes": [list(rng) for rng in ranges]}
                )
            return {
                "success": True,
                "rows": rows,
                "count": len(rows),
                "warnings": warnings,
                "provenance": provenance,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def retrieve_spectra(
        self,
        sparcl_ids: Sequence[str],
        include: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Bulk-retrieve spectrum arrays (flux/wavelength/ivar/model) as a real
        data product. Arrays stay numpy; callers store them, never inline them."""
        try:
            ids = [str(sid).strip() for sid in (sparcl_ids or []) if str(sid).strip()]
            if not ids:
                return {"success": False, "error": "sparcl_ids is required (1-50 SPARCL UUIDs)."}
            warnings: List[str] = []
            if len(ids) > MAX_RETRIEVE_IDS:
                warnings.append(
                    f"Requested {len(ids)} spectra; retrieving the first {MAX_RETRIEVE_IDS}."
                )
                ids = ids[:MAX_RETRIEVE_IDS]
            include_list = [str(f).strip() for f in (include or RETRIEVE_INCLUDE_DEFAULT) if str(f).strip()]
            if "sparcl_id" not in include_list:
                include_list.insert(0, "sparcl_id")
            client = self._get_client()
            result = client.retrieve(uuid_list=list(ids), include=list(include_list))
            records = _records_list(result)
            if not records:
                return {"success": False, "error": "SPARCL returned no spectra for the given ids."}

            spectra: List[Dict[str, Any]] = []
            returned_ids = set()
            for record in records:
                spectrum = _normalize_spectrum_record(record)
                if spectrum is None:
                    warnings.append("Skipped one SPARCL record with invalid wavelength/flux arrays.")
                    continue
                returned_ids.add(spectrum["sparcl_id"])
                spectra.append(spectrum)
            missing = [sid for sid in ids if sid not in returned_ids]
            if missing:
                warnings.append(f"{len(missing)} id(s) returned no spectrum: {missing[:5]}")
            if not spectra:
                return {"success": False, "error": "No valid SPARCL spectra in the response."}
            return {
                "success": True,
                "spectra": spectra,
                "count": len(spectra),
                "warnings": warnings,
                "provenance": {
                    "service": "NOIRLab SparCL",
                    "include": include_list,
                    "requested": len(ids),
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def plot_spectrum(self, sparcl_id: str, mark_lines: bool = True, smooth: Any = 0) -> Dict[str, Any]:
        sid = str(sparcl_id or "").strip()
        if not sid:
            return {"success": False, "error": "sparcl_id is required."}
        try:
            client = self._get_client()
            # ra/dec ride along so the interactive card can deep-link the target
            # into the Spectral Line Explorer.
            include = list(RETRIEVE_FIELDS) + ["ra", "dec"]
            result = client.retrieve(uuid_list=[sid], include=include)
            records = _records_list(result)
            if not records:
                return {"success": False, "error": f"No SparCL spectrum returned for {sid}."}
            record = records[0]
            wavelength_raw = record.get("wavelength")
            flux_raw = record.get("flux")
            wavelength = np.asarray(wavelength_raw if wavelength_raw is not None else [], dtype=float)
            flux = np.asarray(flux_raw if flux_raw is not None else [], dtype=float)
            if wavelength.size == 0 or flux.size == 0 or wavelength.size != flux.size:
                return {"success": False, "error": f"SparCL spectrum {sid} has invalid wavelength/flux arrays."}
            ivar_full = _aligned_array(record.get("ivar"), int(np.asarray(wavelength_raw).size) if wavelength_raw is not None else 0)
            finite = np.isfinite(wavelength) & np.isfinite(flux)
            wavelength = wavelength[finite]
            flux = flux[finite]
            ivar = ivar_full[finite] if ivar_full is not None else None
            if wavelength.size == 0:
                return {"success": False, "error": f"SparCL spectrum {sid} has no finite wavelength/flux points."}
            warnings: List[str] = []
            smooth_i = _positive_int(smooth, 0, 1000, allow_zero=True)
            # sparcl-smooth-before-ivar-mask: blank ivar<=0 (bad/sky-line)
            # pixels BEFORE smoothing so a masked artifact (e.g. a cosmic-ray
            # spike at an ivar=0 pixel) cannot leak into its neighbors'
            # smoothed values; the boxcar is renormalized over the good pixels
            # actually inside each window.
            bad = None
            if ivar is not None:
                candidate_bad = ~(np.isfinite(ivar) & (ivar > 0))
                if np.any(candidate_bad) and not np.all(candidate_bad):
                    bad = candidate_bad
            masked_flux = flux
            if bad is not None:
                masked_flux = np.array(flux, dtype=float, copy=True)
                masked_flux[bad] = np.nan
            plot_flux = masked_flux
            if smooth_i > 1:
                if smooth_i >= flux.size:
                    warnings.append("smooth exceeds spectrum length; smoothing skipped.")
                else:
                    kernel = np.ones(smooth_i, dtype=float)
                    finite_mask = np.isfinite(masked_flux)
                    values = np.where(finite_mask, masked_flux, 0.0)
                    weight = np.convolve(finite_mask.astype(float), kernel, mode="same")
                    summed = np.convolve(values, kernel, mode="same")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        plot_flux = np.where(weight > 0, summed / weight, np.nan)
                    if bad is not None:
                        # Masked pixels themselves stay blanked in the trace.
                        plot_flux = np.array(plot_flux, dtype=float, copy=True)
                        plot_flux[bad] = np.nan

            model = record.get("model")
            model_arr = None
            if model is not None:
                candidate = np.asarray(model, dtype=float)
                if candidate.size == finite.size:
                    model_arr = candidate[finite]
                elif candidate.size == wavelength.size:
                    model_arr = candidate

            redshift = _float_or_none(record.get("redshift"))
            spectype = str(record.get("spectype") or "Spectrum")
            plt = self.plotting_service._apply_style(dark=False)
            fig, ax = plt.subplots(figsize=(9.0, 4.8))
            ax.plot(wavelength, plot_flux, lw=0.8, alpha=0.65, label="flux")
            if ivar is not None:
                good = np.isfinite(ivar) & (ivar > 0)
                if np.any(good):
                    sigma = np.zeros_like(flux)
                    sigma[good] = 1.0 / np.sqrt(ivar[good])
                    ax.fill_between(
                        wavelength, plot_flux - sigma, plot_flux + sigma,
                        where=good, color="0.6", alpha=0.25, linewidth=0, label="±1σ",
                    )
            if model_arr is not None and model_arr.size == wavelength.size:
                ax.plot(wavelength, model_arr, lw=1.0, alpha=0.85, label="model")
            if mark_lines and redshift is not None:
                self._mark_lines(ax, wavelength, redshift, spectype)
            ax.set_xlabel("Wavelength [A]")
            ax.set_ylabel("Flux")
            z_label = "unknown" if redshift is None else f"{redshift:.5g}"
            ax.set_title(f"{spectype} z={z_label} - {sid[:8]}")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            render = self.plotting_service._save_and_encode(fig, f"sparcl_spectrum_{uuid.uuid4().hex[:10]}")
            ra = _float_or_none(record.get("ra"))
            dec = _float_or_none(record.get("dec"))
            plotly_spec = _spectrum_plotly_spec(
                wavelength, plot_flux, model_arr, sid=sid, spectype=spectype,
                redshift=redshift, mark_lines=bool(mark_lines),
            )
            # plotly-downsample-unannotated: also surface the decimation as a
            # card-level warning, mirroring the in-figure annotation.
            plot_stride = _downsample_stride(int(wavelength.size))
            if plot_stride > 1:
                warnings.append(
                    f"Interactive plot downsampled (stride {plot_stride}) to "
                    f"~{int(math.ceil(wavelength.size / plot_stride)):,} of "
                    f"{int(wavelength.size):,} points; the PNG export keeps "
                    "full resolution."
                )
            return {
                "success": True,
                "image_base64": render.get("base64_png"),
                "path": render.get("web_url"),
                "png_path": render.get("png_path"),
                "pdf_path": render.get("pdf_path"),
                "sparcl_id": sid,
                "n_points": int(wavelength.size),
                "redshift": redshift,
                "spectype": spectype,
                "warnings": warnings,
                "plotly_spec": plotly_spec,
                "meta": _spectrum_card_meta(sid, spectype=spectype, redshift=redshift, ra=ra, dec=dec),
                "provenance": {"service": "NOIRLab SparCL", "include": include},
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @staticmethod
    def _mark_lines(ax: Any, wavelength: np.ndarray, redshift: float, spectype: Optional[str] = None) -> None:
        lo = float(np.nanmin(wavelength))
        hi = float(np.nanmax(wavelength))
        ymax = ax.get_ylim()[1]
        for entry in line_lists.observed_lines(spectype, redshift, wavelength_min=lo, wavelength_max=hi):
            observed = entry["wavelength_observed"]
            ax.axvline(observed, color="0.4", linestyle="--", linewidth=0.7, alpha=0.45)
            ax.text(observed, ymax, entry["label"], rotation=90, va="top", ha="right", fontsize=7, alpha=0.7)


def _downsample_stride(size: int, cap: int = PLOTLY_MAX_POINTS) -> int:
    return max(1, int(math.ceil(size / float(cap)))) if size > cap else 1


def _json_safe_list(values: np.ndarray) -> List[Optional[float]]:
    """NaN/inf -> None: json.dumps would otherwise emit bare NaN tokens that
    break the SSE JSON parse on the frontend (raw-token fallback corrupts the
    visible answer text)."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=float)]


def _spectrum_plotly_spec(
    wavelength: np.ndarray,
    flux: np.ndarray,
    model: Optional[np.ndarray],
    *,
    sid: str,
    spectype: str,
    redshift: Optional[float],
    mark_lines: bool,
) -> Dict[str, Any]:
    """Interactive Plotly figure for the spectrum card: flux + model traces and
    redshifted line overlays as layout shapes/annotations (hover keeps working
    because the lines are layout objects, not traces)."""
    stride = _downsample_stride(int(wavelength.size))
    wl = wavelength[::stride]
    fx = np.asarray(flux, dtype=float)[::stride]
    data: List[Dict[str, Any]] = [{
        "type": "scatter",
        "mode": "lines",
        "name": "flux",
        "x": _json_safe_list(wl),
        "y": _json_safe_list(fx),
        "line": {"width": 1},
        "hovertemplate": "λ %{x:.1f} Å<br>flux %{y:.3g}<extra></extra>",
    }]
    if model is not None and np.asarray(model).size == wavelength.size:
        data.append({
            "type": "scatter",
            "mode": "lines",
            "name": "model",
            "x": _json_safe_list(wl),
            "y": _json_safe_list(np.asarray(model, dtype=float)[::stride]),
            "line": {"width": 1.2},
            "hovertemplate": "λ %{x:.1f} Å<br>model %{y:.3g}<extra></extra>",
        })
    shapes: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    if mark_lines and redshift is not None and wl.size:
        lo, hi = float(np.nanmin(wl)), float(np.nanmax(wl))
        for entry in line_lists.observed_lines(spectype, redshift, wavelength_min=lo, wavelength_max=hi):
            observed = entry["wavelength_observed"]
            dash = "dot" if entry["kind"] == line_lists.ABSORPTION else "dash"
            shapes.append({
                "type": "line", "xref": "x", "yref": "paper",
                "x0": observed, "x1": observed, "y0": 0, "y1": 1,
                "line": {"width": 1, "dash": dash, "color": "rgba(148,163,184,0.55)"},
            })
            annotations.append({
                "x": observed, "xref": "x", "y": 1, "yref": "paper",
                "text": entry["label"], "showarrow": False,
                "textangle": -90, "yanchor": "top", "xanchor": "left",
                "font": {"size": 9, "color": "rgba(148,163,184,0.9)"},
            })
    z_label = "unknown" if redshift is None else f"{redshift:.5g}"
    layout = {
        "title": {"text": f"{spectype} z={z_label} — {sid[:8]}"},
        "xaxis": {"title": {"text": "Wavelength [Å] (vacuum)"}},
        "yaxis": {"title": {"text": "Flux"}},
        "shapes": shapes,
        "annotations": annotations,
        "legend": {"orientation": "h"},
        "margin": {"t": 48, "r": 16},
    }
    if stride > 1:
        # plotly-downsample-unannotated: disclose the stride decimation in the
        # interactive card so narrow features missing from the trace are never
        # mistaken for absent data; the PNG export keeps full resolution.
        layout["meta"] = {
            "downsample": {
                "stride": stride,
                "points_shown": int(wl.size),
                "points_total": int(wavelength.size),
            }
        }
        layout["annotations"] = annotations + [{
            "text": (
                f"Interactive view downsampled to {int(wl.size):,} of "
                f"{int(wavelength.size):,} points — narrow features may be "
                "missing; the PNG keeps full resolution"
            ),
            "xref": "paper", "yref": "paper",
            "x": 0, "y": -0.22, "xanchor": "left", "yanchor": "top",
            "showarrow": False,
            "font": {"size": 10, "color": "rgba(148,163,184,0.95)"},
        }]
    return {"data": data, "layout": layout}


def _spectrum_card_meta(
    sid: str,
    *,
    spectype: str,
    redshift: Optional[float],
    ra: Optional[float],
    dec: Optional[float],
) -> Dict[str, Any]:
    """Card meta for the UI: SPARCL identity + a /spectral-lines deep link.

    The link prefills target (ICRS degrees — SPARCL records carry no name) and
    the pipeline redshift; no autorun, so the user confirms before a coverage
    job fires on a coordinate-string target."""
    from urllib.parse import urlencode

    meta: Dict[str, Any] = {"sparcl_id": sid, "spectype": spectype, "redshift": redshift}
    if ra is not None and dec is not None:
        params = {
            "mode": "basic",
            "target": f"{ra:.5f} {dec:+.5f}",
            "ra_deg": f"{ra:.8g}",
            "dec_deg": f"{dec:.8g}",
        }
        params["z"] = "auto" if redshift is None else f"{redshift:.6g}"
        meta["line_explorer_url"] = "/spectral-lines?" + urlencode(params)
    return meta


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


def _normalize_spectypes(value: Optional[Sequence[str] | str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]
    return [part.upper() for part in parts if part]


def _jsonable_constraints(constraints: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in constraints.items():
        if isinstance(value, (list, tuple)):
            out[key] = [v if isinstance(v, (int, float, str, bool)) else str(v) for v in value]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _normalize_spectrum_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sid = record.get("sparcl_id")
    wavelength_raw = record.get("wavelength")
    flux_raw = record.get("flux")
    wavelength = np.asarray(wavelength_raw if wavelength_raw is not None else [], dtype=float)
    flux = np.asarray(flux_raw if flux_raw is not None else [], dtype=float)
    if sid is None or wavelength.size == 0 or flux.size == 0 or wavelength.size != flux.size:
        return None
    ivar = _aligned_array(record.get("ivar"), wavelength.size)
    model = _aligned_array(record.get("model"), wavelength.size)
    mask = _aligned_array(record.get("mask"), wavelength.size)
    median_snr = None
    if ivar is not None:
        good = np.isfinite(ivar) & (ivar > 0) & np.isfinite(flux)
        if np.any(good):
            median_snr = float(np.median(flux[good] * np.sqrt(ivar[good])))
    finite_wave = wavelength[np.isfinite(wavelength)]
    return {
        "sparcl_id": str(sid),
        "wavelength": wavelength,
        "flux": flux,
        "ivar": ivar,
        "model": model,
        "mask": mask,
        "redshift": _float_or_none(record.get("redshift")),
        "spectype": record.get("spectype"),
        "data_release": record.get("data_release"),
        "n_points": int(wavelength.size),
        "wavelength_min": float(finite_wave.min()) if finite_wave.size else None,
        "wavelength_max": float(finite_wave.max()) if finite_wave.size else None,
        "median_snr": median_snr,
    }


def _aligned_array(value: Any, size: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        candidate = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return candidate if candidate.size == size else None


def _normalize_releases(value: Optional[Sequence[str] | str]) -> List[str]:
    if value is None:
        return list(DEFAULT_RELEASES)
    if isinstance(value, str):
        releases = [part.strip() for part in value.split(",") if part.strip()]
    else:
        releases = [str(part).strip() for part in value if str(part).strip()]
    return releases or list(DEFAULT_RELEASES)


def _positive_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) and out > 0 else default


def _positive_int(value: Any, default: int, cap: int, *, allow_zero: bool = False) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    if allow_zero and out <= 0:
        return 0
    if out <= 0:
        out = default
    return min(out, cap)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _search_ranges(ra: float, dec: float, radius_arcsec: float) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
    radius_deg = radius_arcsec / 3600.0
    cos_dec = abs(math.cos(math.radians(dec)))
    dra = 180.0 if cos_dec < 1e-8 else min(180.0, radius_deg / cos_dec)
    dec_min = max(-90.0, dec - radius_deg)
    dec_max = min(90.0, dec + radius_deg)
    ra_min = ra - dra
    ra_max = ra + dra
    warnings: List[str] = []
    if dra >= 180.0:
        return [(0.0, 360.0, dec_min, dec_max)], ["RA search box spans all right ascensions near the pole."]
    if ra_min < 0.0:
        warnings.append("RA search box crosses 0 deg; split into two SparCL queries.")
        return [(0.0, ra_max, dec_min, dec_max), (360.0 + ra_min, 360.0, dec_min, dec_max)], warnings
    if ra_max > 360.0:
        warnings.append("RA search box crosses 360 deg; split into two SparCL queries.")
        return [(ra_min, 360.0, dec_min, dec_max), (0.0, ra_max - 360.0, dec_min, dec_max)], warnings
    return [(ra_min, ra_max, dec_min, dec_max)], warnings


def _records_list(result: Any) -> List[Dict[str, Any]]:
    records = getattr(result, "records", result)
    if records is None:
        return []
    return [dict(record) for record in records]


def _normalize_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ra = _float_or_none(record.get("ra"))
    dec = _float_or_none(record.get("dec"))
    sid = record.get("sparcl_id")
    if sid is None or ra is None or dec is None:
        return None
    return {
        "sparcl_id": str(sid),
        "ra": ra,
        "dec": dec,
        "redshift": _float_or_none(record.get("redshift")),
        "spectype": record.get("spectype"),
        "data_release": record.get("data_release"),
    }


def _angular_distance_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1 = math.radians(ra1)
    d1 = math.radians(dec1)
    r2 = math.radians(ra2)
    d2 = math.radians(dec2)
    sd = math.sin((d2 - d1) / 2.0)
    sr = math.sin((r2 - r1) / 2.0)
    a = sd * sd + math.cos(d1) * math.cos(d2) * sr * sr
    angle = 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))
    return math.degrees(angle) * 3600.0


__all__ = ["SparclSpectraService"]
