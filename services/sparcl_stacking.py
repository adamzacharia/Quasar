"""SPARCL spectral stacking / co-adding with binning (Data Lab parity 2026-07).

Implements the Astro Data Lab blog's signature workflow: bin a spectrum sample
(by redshift constraints or by any numeric column chained from a stored
result), bulk-retrieve N spectra per bin, shift to rest frame, resample onto a
common wavelength grid, and average so noise cancels and weak features emerge.

Pure numpy — no scipy/specutils dependency. All SPARCL I/O goes through the
injected :class:`services.sparcl_spectra.SparclSpectraService`.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from services.plotting import PlottingService
from services.sparcl_spectra import SparclSpectraService

# Retrieval batch size (SparclSpectraService caps a single retrieve at 50 ids).
_BATCH = 50
MAX_BINS = 12
MAX_PER_BIN = 200
# Total retrieval budget per stack call — keeps worst-case runtime bounded.
MAX_TOTAL_SPECTRA = 600
DEFAULT_GRID_STEP_A = 2.0


class SparclStackingService:
    """Bin → retrieve → rest-frame → resample → stack."""

    def __init__(
        self,
        *,
        spectra_service: Optional[SparclSpectraService] = None,
        plotting_service: Optional[PlottingService] = None,
    ):
        self.spectra_service = spectra_service or SparclSpectraService()
        self.plotting_service = plotting_service or PlottingService()

    # ── Public API ───────────────────────────────────────────────────────────
    def stack(
        self,
        *,
        # Mode A: SPARCL constraint search binned on redshift
        spectype: Optional[str] = None,
        redshift_min: Optional[float] = None,
        redshift_max: Optional[float] = None,
        data_release: Optional[Sequence[str]] = None,
        # Mode B: rows chained from a stored result (sparcl_id + numeric bin column)
        rows: Optional[Sequence[Dict[str, Any]]] = None,
        bin_column: Optional[str] = None,
        bin_edges: Optional[Sequence[float]] = None,
        n_bins: int = 4,
        n_per_bin: int = 25,
        rest_frame: bool = True,
        weighting: str = "ivar",
        normalize: bool = True,
        grid_step: float = DEFAULT_GRID_STEP_A,
        title: Optional[str] = None,
        status: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        try:
            warnings: List[str] = []
            notify = status or (lambda _msg: None)
            weighting = str(weighting or "ivar").strip().lower()
            if weighting not in {"ivar", "uniform", "median"}:
                raise ValueError("weighting must be one of: ivar, uniform, median")
            n_bins_i = max(1, min(int(n_bins or 4), MAX_BINS))
            n_per_bin_i = max(1, min(int(n_per_bin or 25), MAX_PER_BIN))
            if n_bins_i * n_per_bin_i > MAX_TOTAL_SPECTRA:
                n_per_bin_i = max(1, MAX_TOTAL_SPECTRA // n_bins_i)
                warnings.append(
                    f"Total spectra capped at {MAX_TOTAL_SPECTRA}; using {n_per_bin_i} per bin."
                )
            step = float(grid_step) if grid_step else DEFAULT_GRID_STEP_A
            if not math.isfinite(step) or step <= 0:
                raise ValueError("grid_step must be a positive wavelength step in Angstrom")

            if rows is not None:
                bins = self._bins_from_rows(
                    rows, bin_column=bin_column, bin_edges=bin_edges,
                    n_bins=n_bins_i, n_per_bin=n_per_bin_i, warnings=warnings,
                )
                bin_quantity = str(bin_column)
            else:
                bins = self._bins_from_constraints(
                    spectype=spectype, redshift_min=redshift_min, redshift_max=redshift_max,
                    data_release=data_release, n_bins=n_bins_i, n_per_bin=n_per_bin_i,
                    warnings=warnings, status=notify,
                )
                bin_quantity = "redshift"
            bins = [b for b in bins if b["sparcl_ids"]]
            if not bins:
                raise ValueError("No spectra fell into any bin; widen the constraints or bins.")

            # ── Retrieve per bin ────────────────────────────────────────────
            for idx, spec_bin in enumerate(bins, start=1):
                ids = spec_bin["sparcl_ids"]
                notify(f"Retrieving bin {idx}/{len(bins)} — {len(ids)} spectra")
                spectra: List[Dict[str, Any]] = []
                for start in range(0, len(ids), _BATCH):
                    chunk = ids[start:start + _BATCH]
                    got = self.spectra_service.retrieve_spectra(chunk)
                    if not got.get("success"):
                        warnings.append(
                            f"Bin {idx}: retrieval failed for {len(chunk)} ids ({got.get('error')})."
                        )
                        continue
                    spectra.extend(got.get("spectra") or [])
                spec_bin["spectra"] = spectra

            bins = [b for b in bins if b.get("spectra")]
            if not bins:
                raise ValueError("Spectrum retrieval returned nothing for any bin.")

            # ── Rest-frame shift + common grid ─────────────────────────────
            all_ranges: List[Tuple[float, float]] = []
            for spec_bin in bins:
                usable = []
                for spectrum in spec_bin["spectra"]:
                    shifted = self._shift(spectrum, rest_frame=rest_frame)
                    if shifted is None:
                        continue
                    usable.append(shifted)
                    all_ranges.append((float(np.nanmin(shifted["wavelength"])),
                                       float(np.nanmax(shifted["wavelength"]))))
                dropped = len(spec_bin["spectra"]) - len(usable)
                if dropped:
                    warnings.append(
                        f"Bin '{spec_bin['label']}': dropped {dropped} spectra without a finite redshift."
                        if rest_frame else
                        f"Bin '{spec_bin['label']}': dropped {dropped} invalid spectra."
                    )
                spec_bin["spectra"] = usable
            bins = [b for b in bins if b["spectra"]]
            if not bins or not all_ranges:
                raise ValueError("No usable spectra after rest-frame filtering.")

            # Grid covered by the median spectrum (robust to outlier redshifts).
            grid_lo = float(np.median([r[0] for r in all_ranges]))
            grid_hi = float(np.median([r[1] for r in all_ranges]))
            if grid_hi - grid_lo < 10 * step:
                raise ValueError("Common wavelength coverage is too narrow to stack.")
            grid = np.arange(grid_lo, grid_hi + step, step)

            # ── Resample + stack per bin ───────────────────────────────────
            notify(f"Stacking {sum(len(b['spectra']) for b in bins)} spectra onto "
                   f"{grid.size}-point grid")
            stacked_bins: List[Dict[str, Any]] = []
            for spec_bin in bins:
                flux_rows, weight_rows = [], []
                for spectrum in spec_bin["spectra"]:
                    flux_i, weight_i = self._resample(spectrum, grid, normalize=normalize)
                    if flux_i is None:
                        continue
                    flux_rows.append(flux_i)
                    weight_rows.append(weight_i)
                if not flux_rows:
                    continue
                flux_matrix = np.vstack(flux_rows)
                weight_matrix = np.vstack(weight_rows)
                stacked, n_pix = self._combine(flux_matrix, weight_matrix, weighting)
                stacked_bins.append({
                    "label": spec_bin["label"],
                    "lo": spec_bin.get("lo"),
                    "hi": spec_bin.get("hi"),
                    "center": spec_bin.get("center"),
                    "n_spectra": int(flux_matrix.shape[0]),
                    "flux": stacked,
                    "n_pix": n_pix,
                })
            if not stacked_bins:
                raise ValueError("Stacking produced no usable bins.")

            plot = self._plot(grid, stacked_bins, bin_quantity=bin_quantity,
                              rest_frame=rest_frame, title=title)

            return {
                "success": True,
                "bins": [
                    {k: b[k] for k in ("label", "lo", "hi", "center", "n_spectra")}
                    for b in stacked_bins
                ],
                "grid": {
                    "lo": float(grid[0]), "hi": float(grid[-1]),
                    "step": step, "n_points": int(grid.size),
                    "frame": "rest" if rest_frame else "observed",
                },
                "stack_rows": self._long_rows(grid, stacked_bins),
                "weighting": weighting,
                "normalized": bool(normalize),
                "warnings": warnings,
                **plot,
                "provenance": {
                    "service": "NOIRLab SparCL",
                    "workflow": "spectral_stacking",
                    "bin_quantity": bin_quantity,
                    "n_bins": len(stacked_bins),
                },
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Binning ──────────────────────────────────────────────────────────────
    def _bins_from_constraints(
        self, *, spectype, redshift_min, redshift_max, data_release,
        n_bins: int, n_per_bin: int, warnings: List[str], status: Callable[[str], None],
    ) -> List[Dict[str, Any]]:
        try:
            z_lo = float(redshift_min)
            z_hi = float(redshift_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Constraint mode needs redshift_min and redshift_max (the bin axis); "
                "or pass rows/result_id with a bin_column."
            ) from exc
        if not (math.isfinite(z_lo) and math.isfinite(z_hi)) or z_lo >= z_hi:
            raise ValueError("redshift_min must be < redshift_max.")
        edges = np.linspace(z_lo, z_hi, n_bins + 1)
        bins: List[Dict[str, Any]] = []
        for k in range(n_bins):
            lo, hi = float(edges[k]), float(edges[k + 1])
            status(f"Searching bin {k + 1}/{n_bins} — {lo:.3g} <= z < {hi:.3g}")
            found = self.spectra_service.search_spectra(
                spectype=spectype, redshift_min=lo, redshift_max=hi,
                data_release=data_release, limit=n_per_bin,
            )
            if not found.get("success"):
                warnings.append(f"Bin {k + 1} search failed: {found.get('error')}")
                ids: List[str] = []
            else:
                ids = [str(r["sparcl_id"]) for r in (found.get("rows") or [])][:n_per_bin]
            bins.append({
                "label": f"{lo:.3g} <= z < {hi:.3g}",
                "lo": lo, "hi": hi, "center": 0.5 * (lo + hi),
                "sparcl_ids": ids,
            })
        return bins

    @staticmethod
    def _bins_from_rows(
        rows: Sequence[Dict[str, Any]], *, bin_column, bin_edges,
        n_bins: int, n_per_bin: int, warnings: List[str],
    ) -> List[Dict[str, Any]]:
        if not bin_column:
            raise ValueError("bin_column is required when stacking from stored rows.")
        col = str(bin_column)
        pairs: List[Tuple[str, float]] = []
        for row in rows or []:
            sid = str(row.get("sparcl_id") or "").strip()
            try:
                value = float(row.get(col))
            except (TypeError, ValueError):
                continue
            if sid and math.isfinite(value):
                pairs.append((sid, value))
        if not pairs:
            raise ValueError(
                f"No rows carry both sparcl_id and a finite '{col}' value; "
                "check the result_id and bin_column."
            )
        values = np.array([v for _, v in pairs], dtype=float)
        if bin_edges:
            edges = np.asarray(sorted(float(e) for e in bin_edges), dtype=float)
            if edges.size < 2:
                raise ValueError("bin_edges needs at least two values.")
        else:
            edges = np.linspace(float(values.min()), float(values.max()), n_bins + 1)
            # Guard a constant column (all values equal → zero-width bins).
            if edges[0] == edges[-1]:
                raise ValueError(f"Column '{col}' is constant; nothing to bin on.")
        bins: List[Dict[str, Any]] = []
        for k in range(edges.size - 1):
            lo, hi = float(edges[k]), float(edges[k + 1])
            last = k == edges.size - 2
            ids = [
                sid for sid, v in pairs
                if (lo <= v < hi) or (last and v == hi)
            ][:n_per_bin]
            bins.append({
                "label": f"{lo:.3g} <= {col} < {hi:.3g}" if not last else f"{lo:.3g} <= {col} <= {hi:.3g}",
                "lo": lo, "hi": hi, "center": 0.5 * (lo + hi),
                "sparcl_ids": ids,
            })
        empty = sum(1 for b in bins if not b["sparcl_ids"])
        if empty:
            warnings.append(f"{empty} of {len(bins)} bins are empty.")
        return bins

    # ── Per-spectrum transforms ──────────────────────────────────────────────
    @staticmethod
    def _shift(spectrum: Dict[str, Any], *, rest_frame: bool) -> Optional[Dict[str, Any]]:
        wavelength = np.asarray(spectrum.get("wavelength"), dtype=float)
        flux = np.asarray(spectrum.get("flux"), dtype=float)
        if wavelength.size == 0 or wavelength.size != flux.size:
            return None
        ivar_raw = spectrum.get("ivar")
        ivar = np.asarray(ivar_raw, dtype=float) if ivar_raw is not None else None
        if ivar is not None and ivar.size != wavelength.size:
            ivar = None
        if rest_frame:
            redshift = spectrum.get("redshift")
            try:
                z = float(redshift)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(z) or z <= -0.99:
                return None
            wavelength = wavelength / (1.0 + z)
        finite = np.isfinite(wavelength) & np.isfinite(flux)
        if not np.any(finite):
            return None
        return {
            "wavelength": wavelength[finite],
            "flux": flux[finite],
            "ivar": ivar[finite] if ivar is not None else None,
        }

    @staticmethod
    def _resample(
        spectrum: Dict[str, Any], grid: np.ndarray, *, normalize: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        wavelength = spectrum["wavelength"]
        flux = spectrum["flux"]
        ivar = spectrum["ivar"]
        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        flux = flux[order]
        if ivar is not None:
            ivar = ivar[order]
            good = np.isfinite(ivar) & (ivar > 0)
            if not np.any(good):
                return None, None
            wavelength, flux, ivar = wavelength[good], flux[good], ivar[good]
        if wavelength.size < 2:
            return None, None
        if normalize:
            scale = float(np.nanmedian(flux))
            if not math.isfinite(scale) or scale == 0:
                return None, None
            flux = flux / scale
            if ivar is not None:
                ivar = ivar * scale * scale
        flux_i = np.interp(grid, wavelength, flux, left=np.nan, right=np.nan)
        if ivar is not None:
            weight_i = np.interp(grid, wavelength, ivar, left=0.0, right=0.0)
        else:
            weight_i = np.ones_like(grid)
        weight_i = np.where(np.isfinite(flux_i), weight_i, 0.0)
        if not np.any(weight_i > 0):
            return None, None
        return flux_i, weight_i

    @staticmethod
    def _combine(
        flux_matrix: np.ndarray, weight_matrix: np.ndarray, weighting: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_pix = (weight_matrix > 0).sum(axis=0)
        masked = np.where(weight_matrix > 0, flux_matrix, np.nan)
        if weighting == "median":
            with np.errstate(all="ignore"):
                stacked = np.nanmedian(masked, axis=0)
        elif weighting == "uniform":
            with np.errstate(all="ignore"):
                stacked = np.nanmean(masked, axis=0)
        else:  # ivar-weighted mean
            weights = np.where(np.isfinite(flux_matrix), weight_matrix, 0.0)
            total = weights.sum(axis=0)
            with np.errstate(all="ignore"):
                stacked = np.where(
                    total > 0,
                    np.nansum(np.nan_to_num(flux_matrix) * weights, axis=0) / np.where(total > 0, total, 1.0),
                    np.nan,
                )
        return stacked, n_pix.astype(int)

    # ── Output shaping ───────────────────────────────────────────────────────
    @staticmethod
    def _long_rows(grid: np.ndarray, stacked_bins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for bin_index, spec_bin in enumerate(stacked_bins):
            flux = spec_bin["flux"]
            n_pix = spec_bin["n_pix"]
            for j in range(grid.size):
                value = flux[j]
                if not np.isfinite(value):
                    continue
                rows.append({
                    "bin_index": bin_index,
                    "bin_label": spec_bin["label"],
                    "wavelength": float(grid[j]),
                    "flux_stacked": float(value),
                    "n_spectra_pix": int(n_pix[j]),
                })
        return rows

    def _plot(
        self, grid: np.ndarray, stacked_bins: List[Dict[str, Any]], *,
        bin_quantity: str, rest_frame: bool, title: Optional[str],
    ) -> Dict[str, Any]:
        plt = self.plotting_service._apply_style(dark=False)
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        cmap = plt.get_cmap("viridis")
        n = len(stacked_bins)
        for k, spec_bin in enumerate(stacked_bins):
            color = cmap(0.1 + 0.8 * (k / max(1, n - 1)))
            ax.plot(
                grid, spec_bin["flux"], lw=0.9, color=color,
                label=f"{spec_bin['label']} (N={spec_bin['n_spectra']})",
            )
        frame = "Rest-frame" if rest_frame else "Observed"
        ax.set_xlabel(f"{frame} wavelength [A]")
        ax.set_ylabel("Normalized stacked flux")
        ax.set_title(title or f"Stacked spectra binned by {bin_quantity}")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        render = self.plotting_service._save_and_encode(fig, f"sparcl_stack_{uuid.uuid4().hex[:10]}")
        return {
            "image_base64": render.get("base64_png"),
            "path": render.get("web_url"),
            "png_path": render.get("png_path"),
            "pdf_path": render.get("pdf_path"),
        }


__all__ = ["SparclStackingService", "MAX_BINS", "MAX_PER_BIN", "MAX_TOTAL_SPECTRA"]
