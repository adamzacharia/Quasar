"""
capabilities/spectra.py — SPARCL survey-scale spectroscopy capabilities
(Data Lab parity work, 2026-07).

New family (not a legacy migration): survey-scale constraint search over the
31M-spectrum SPARCL holdings and bulk array retrieval as a real data product.
Follows the transport-pure capability pattern of ``capabilities/vo.py``:

  * the lazy service getter is injected as a bound callable
    (``get_sparcl_spectra_service`` → the agent's
    ``_get_sparcl_spectra_service``) and called INSIDE the ``try`` so a failing
    constructor produces a caught-and-typed error dict;
  * the shared agent helpers stay agent-side and are injected bound:
    ``external_catalog_table_result`` (sets the UI table card) and
    ``live_imagery_coordinates`` (target → ra/dec resolution);
  * spectra arrays NEVER enter the LLM context: bulk retrieval stores a
    long-format DataFrame in the injected ``result_store`` and returns only
    per-spectrum scalars plus the ``result_id`` handle (fetch rows with
    ``datalab_get_result``, or chain into stacking).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, ToolResult

logger = logging.getLogger(__name__)


def _native(out: Dict[str, Any]) -> ToolResult:
    """Wrap a legacy-shaped output dict as a ToolResult (vo.py pattern)."""
    ok = bool(isinstance(out, dict) and out.get("success"))
    err = out.get("error") if isinstance(out, dict) else None
    return ToolResult(
        success=ok,
        error=(str(err) if (err is not None and not ok) else None),
        native=out,
    )


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _store_rows(ctx, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> Optional[str]:
    """Best-effort result-store put; a missing/failing store never fails the tool."""
    store = getattr(ctx, "result_store", None)
    if store is None or not rows:
        return None
    try:
        import pandas as pd

        # Scope the stored rows to their requesting user so the export
        # route's owner guard can serve them (f2-CX-01 / dl-export-owner-gap).
        user_id = getattr(ctx, "user_id", None)
        if user_id:
            meta = {**meta, "owner_id": str(user_id)}
        return store.put(pd.DataFrame(rows), meta=meta)
    except Exception:
        logger.debug("SPARCL result-store put failed", exc_info=True)
        return None


class SparclSearchSpectraInput(_In):
    spectype: Optional[str] = None
    redshift_min: Optional[float] = None
    redshift_max: Optional[float] = None
    data_release: Optional[List[str]] = None
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: Optional[float] = None
    limit: Optional[int] = 100


class SparclSearchSpectra(BaseCapability):
    name = "sparcl_search_spectra"
    description = (
        "Survey-scale SPARCL spectrum search by physical constraints: spectype "
        "(GALAXY/STAR/QSO), redshift range, and data release (DESI-DR1, SDSS-DR17) "
        "across 31M spectra — optionally combined with a cone around a target/position. "
        "Returns sparcl_ids ready for sparcl_get_spectrum / sparcl_plot_spectrum. "
        "Use sparcl_find_spectra instead for a pure position-only cone lookup."
    )
    category = "archive"
    InputModel = SparclSearchSpectraInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_service = ctx.service("get_sparcl_spectra_service")
        table_result = ctx.service("external_catalog_table_result")
        try:
            cone: Dict[str, Any] = {}
            label = None
            if inp.target_name or (inp.ra is not None and inp.dec is not None):
                resolve = ctx.service("live_imagery_coordinates")
                ra_f, dec_f, label = resolve(target_name=inp.target_name, ra=inp.ra, dec=inp.dec)
                cone = {"ra": ra_f, "dec": dec_f, "radius_arcsec": inp.radius_arcsec}
            result = get_service().search_spectra(
                spectype=inp.spectype,
                redshift_min=inp.redshift_min,
                redshift_max=inp.redshift_max,
                data_release=inp.data_release,
                limit=inp.limit,
                **cone,
            )
            if not result.get("success"):
                return _native(result)
            rows = result.get("rows") or []
            constraints = (result.get("provenance") or {}).get("constraints") or {}
            result_id = _store_rows(
                ctx, rows,
                meta={"tool_name": self.name, "source": "NOIRLab SparCL",
                      "constraints": constraints},
            )
            parts: List[str] = []
            if inp.spectype:
                parts.append(f"spectype={inp.spectype}")
            if "redshift" in constraints:
                lo, hi = constraints["redshift"]
                parts.append(f"{lo:g} <= z <= {hi:g}")
            releases = constraints.get("data_release") or []
            if releases:
                parts.append("+".join(str(r) for r in releases))
            if label:
                radius = (result.get("provenance") or {}).get("radius_arcsec")
                parts.append(f"cone {label}" + (f", r={float(radius):g}\"" if radius else ""))
            columns = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"]
            if cone:
                columns.insert(3, "distance_arcsec")
            out = table_result(
                rows,
                columns=columns,
                source="NOIRLab SparCL spectra",
                filter_label="SPARCL: " + (", ".join(parts) if parts else "all spectra"),
                tool_name=self.name,
                warnings=result.get("warnings", []),
                provenance=result.get("provenance", {}),
            )
            if result_id:
                out["result_id"] = result_id
                out["note"] = (
                    out.get("note", "")
                    + " Rows stored under result_id for chaining (datalab_get_result, stacking)."
                ).strip()
            return _native(out)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SparclGetSpectrumInput(_In):
    sparcl_ids: Optional[List[str]] = None
    include: Optional[List[str]] = None


class SparclGetSpectrum(BaseCapability):
    name = "sparcl_get_spectrum"
    description = (
        "Bulk-retrieve SPARCL spectrum arrays (wavelength, flux, ivar, model) for up to "
        "50 sparcl_ids as a stored data product. Returns per-spectrum scalars (n_points, "
        "wavelength range, redshift, spectype, median S/N) plus a result_id whose rows are "
        "the long-format arrays — fetch with datalab_get_result or chain into analysis. "
        "Use sparcl_plot_spectrum to just LOOK at one spectrum."
    )
    category = "analysis"
    InputModel = SparclGetSpectrumInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_service = ctx.service("get_sparcl_spectra_service")
        try:
            result = get_service().retrieve_spectra(inp.sparcl_ids or [], include=inp.include)
            if not result.get("success"):
                return _native(result)
            spectra = result.get("spectra") or []
            long_rows: List[Dict[str, Any]] = []
            summary: List[Dict[str, Any]] = []
            for spectrum in spectra:
                sid = spectrum["sparcl_id"]
                summary.append({
                    "sparcl_id": sid,
                    "n_points": spectrum.get("n_points"),
                    "wavelength_min": spectrum.get("wavelength_min"),
                    "wavelength_max": spectrum.get("wavelength_max"),
                    "redshift": spectrum.get("redshift"),
                    "spectype": spectrum.get("spectype"),
                    "data_release": spectrum.get("data_release"),
                    "median_snr": spectrum.get("median_snr"),
                })
                wavelength = spectrum.get("wavelength")
                flux = spectrum.get("flux")
                ivar = spectrum.get("ivar")
                model = spectrum.get("model")
                n = int(spectrum.get("n_points") or 0)
                for i in range(n):
                    row: Dict[str, Any] = {
                        "sparcl_id": sid,
                        "wavelength": float(wavelength[i]),
                        "flux": float(flux[i]),
                    }
                    if ivar is not None:
                        row["ivar"] = float(ivar[i])
                    if model is not None:
                        row["model"] = float(model[i])
                    long_rows.append(row)
            result_id = _store_rows(
                ctx, long_rows,
                meta={"tool_name": self.name, "source": "NOIRLab SparCL",
                      "n_spectra": len(spectra)},
            )
            out: Dict[str, Any] = {
                "success": True,
                "count": len(spectra),
                "spectra": summary,
                "warnings": result.get("warnings", []),
                "provenance": result.get("provenance", {}),
                "note": (
                    "Arrays are NOT inlined; the long-format table (sparcl_id, wavelength, "
                    "flux, ivar, model) is stored under result_id."
                ),
            }
            if result_id:
                out["result_id"] = result_id
            else:
                out["warnings"] = list(out["warnings"]) + [
                    "Result store unavailable; arrays were retrieved but not persisted."
                ]
            return _native(out)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SparclStackSpectraInput(_In):
    spectype: Optional[str] = None
    redshift_min: Optional[float] = None
    redshift_max: Optional[float] = None
    data_release: Optional[List[str]] = None
    result_id: Optional[str] = None
    bin_column: Optional[str] = None
    bin_edges: Optional[List[float]] = None
    n_bins: int = 4
    n_per_bin: int = 25
    rest_frame: bool = True
    weighting: str = "ivar"
    normalize: bool = True
    title: Optional[str] = None


class SparclStackSpectra(BaseCapability):
    """The blog's signature demo (Data Lab parity 2026-07): bin a spectrum
    sample, retrieve N spectra per bin, rest-frame shift, resample to a common
    grid, and average so weak features emerge from the noise."""

    name = "sparcl_stack_spectra"
    description = (
        "Stack (co-add) SPARCL spectra in bins so weak features emerge: either bin by "
        "redshift with constraints (spectype + redshift_min/max), or chain from a stored "
        "result_id whose rows carry sparcl_id plus any numeric bin_column (e.g. a g-r color "
        "computed in a Data Lab query). Spectra are rest-frame shifted, median-normalized, "
        "resampled to a common grid, and ivar-weighted averaged per bin. Returns the stacked-"
        "spectra plot and a result_id with the stacked arrays. Keep n_bins*n_per_bin modest "
        "(<=600); large stacks take minutes."
    )
    category = "analysis"
    InputModel = SparclStackSpectraInput
    annotations = {"read_only": True, "cost": "network", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        get_service = ctx.service("get_sparcl_stacking_service")
        try:
            rows = None
            if inp.result_id:
                store = getattr(ctx, "result_store", None)
                if store is None:
                    return _native({"success": False,
                                    "error": "Result store unavailable; cannot read result_id."})
                res = store.get(inp.result_id)
                rows = res.dataframe.to_dict("records")
            result = get_service().stack(
                spectype=inp.spectype,
                redshift_min=inp.redshift_min,
                redshift_max=inp.redshift_max,
                data_release=inp.data_release,
                rows=rows,
                bin_column=inp.bin_column,
                bin_edges=inp.bin_edges,
                n_bins=inp.n_bins,
                n_per_bin=inp.n_per_bin,
                rest_frame=inp.rest_frame,
                weighting=inp.weighting,
                normalize=inp.normalize,
                title=inp.title,
                status=ctx.status,
            )
            if not result.get("success"):
                return _native(result)
            stack_rows = result.pop("stack_rows", None) or []
            stack_result_id = _store_rows(
                ctx, stack_rows,
                meta={"tool_name": self.name, "source": "NOIRLab SparCL",
                      "workflow": "spectral_stacking"},
            )
            if stack_result_id:
                result["result_id"] = stack_result_id
                result["note"] = (
                    "Stacked arrays (bin_label, wavelength, flux_stacked, n_spectra_pix) are "
                    "stored under result_id; fetch with datalab_get_result."
                )
            result["_caption"] = inp.title or "Stacked SPARCL spectra"
            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": str(e)})


CAPABILITIES: List[BaseCapability] = [
    SparclSearchSpectra(),
    SparclGetSpectrum(),
    SparclStackSpectra(),
]

__all__ = [
    "CAPABILITIES",
    "SparclSearchSpectra",
    "SparclGetSpectrum",
    "SparclStackSpectra",
]
