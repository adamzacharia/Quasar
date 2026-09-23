"""One-shot Data Lab tools that collapse a DataLabBench question to one call.

UI benchmark 2026-09-22 (tmp/ui-bench-2026-09-22/):

* ``datalab_healpix_density_map``   L08 -- HEALPix aggregate on the precomputed column + healpy-style map
* ``datalab_stream_selection``      L09 -- Gaia PM window (MATERIALIZED) -> q3c_join NSC -> CMD mask IN SQL -> sky plot
* ``datalab_selection_diagram``     L06 -- one query, one plot, quality presets, absolute magnitude from parallax
* ``datalab_target_class_summary``  L11 -- server-side n(z) histogram + footprint aggregate, one figure, bitmask explained
* ``datalab_satellite_search``      L07 / L15 -- confirmed sky area, tiled density with colour/morphology cuts,
                                     background sigma per peak, known-object screening, per-candidate CMDs + cutouts

Every tool states the cuts it applied in its card caption and result (so the
prose cannot drift from them), prefers a validated field when the region is
left open, and returns ``status`` ok | partial | coverage_gap | infrastructure_failure.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from capabilities.base import BaseCapability, CallContext, ToolResult
from capabilities.datalab import _In, _resolve_coords, _run_analysis_plot, datalab_error, execute_datalab_sql
from services import datalab_orchestration as orchestration
from services import datalab_query_builders as builders
from services import datalab_registry as registry

__all__ = ["CAPABILITIES", "REGION_PRESETS", "QUALITY_PRESETS"]

# Validated reference fields for open-ended prompts (the Galactic Centre and
# the poles are the worst possible defaults: L08 re-run, L13, L15).
REGION_PRESETS: Dict[str, Dict[str, Any]] = {
    "lmc": {"ra": 80.894, "dec": -69.756, "radius_deg": 10.0, "label": "Large Magellanic Cloud", "structure": "LMC disk/bar, 30 Doradus, tidal features"},
    "smc": {"ra": 13.187, "dec": -72.829, "radius_deg": 6.0, "label": "Small Magellanic Cloud", "structure": "SMC bar/wing toward the LMC"},
    "magellanic_bridge": {"ra": 40.0, "dec": -73.0, "radius_deg": 10.0, "label": "Magellanic Bridge (between SMC and LMC)", "structure": "Bridge stellar population"},
    "sgr_stream": {"ra": 30.0, "dec": -15.0, "radius_deg": 10.0, "label": "Sagittarius stream (southern arm near RA 30, Dec -15)", "structure": "Sgr stream, MW halo gradient"},
    "ngp": {"ra": 192.859, "dec": 27.128, "radius_deg": 10.0, "label": "North Galactic Pole", "structure": "smooth halo, |b| ~ 90"},
    "anticenter": {"ra": 88.7, "dec": 28.9, "radius_deg": 10.0, "label": "Galactic anticenter (l = 180)", "structure": "thin/thick disk density gradient with |b|"},
    "hydra2": {"ra": 185.425, "dec": -31.985, "radius_deg": 1.0, "label": "Hydra II field", "structure": "known ultra-faint dwarf (re-detection test)"},
    "delve_south": {"ra": 30.0, "dec": -50.0, "radius_deg": 5.0, "label": "DELVE/DES deep southern field (RA 30, Dec -50)", "structure": "clean halo field, Reticulum II / Horologium I within 10 deg"},
    "fornax_field": {"ra": 39.997, "dec": -34.449, "radius_deg": 2.0, "label": "Fornax dSph field", "structure": "known classical dwarf (re-detection test)"},
    "great_wall": {"ra_min": 150.0, "ra_max": 220.0, "dec_min": 0.0, "dec_max": 5.0, "z_max": 0.1, "label": "SDSS Great Wall (RA 150-220, Dec 0-5, z <= 0.1)"},
}

QUALITY_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "gaia_astrometric": [
        {"column": "parallax_over_error", "op": ">", "value": 5},
        {"column": "ruwe", "op": "<", "value": 1.4},
        {"column": "parallax", "op": ">", "value": 0},
    ],
    "gaia_high_pm": [
        {"column": "parallax_over_error", "op": ">", "value": 5},
        {"column": "ruwe", "op": "<", "value": 1.4},
        {"column": "parallax", "op": ">", "value": 0},
    ],
    "nsc_point_sources": [{"column": "class_star", "op": ">", "value": 0.5}],
    "none": [],
}

# Blue/old-population colour window and point-source cut for density scans.
DEFAULT_DENSITY_COLOR = {"min": -0.4, "max": 0.9}
DEFAULT_MAG_WINDOW = {"bright": 16.0, "faint": 23.5}


def _native(out: Dict[str, Any]) -> ToolResult:
    return ToolResult(success=bool(out.get("success")), error=(None if out.get("success") else out.get("error")), native=out)


def _owner(ctx: CallContext) -> Optional[str]:
    return str(ctx.user_id) if ctx.user_id else None


def _region(inp_region: Any, preset: Optional[str], *, fallback: str) -> Tuple[float, float, float, str, Optional[str]]:
    """(ra, dec, radius_deg, label, preset_used). Explicit region dict wins,
    then a named preset, then the validated fallback field."""
    if isinstance(inp_region, dict) and inp_region.get("ra") is not None and inp_region.get("dec") is not None:
        r = float(inp_region.get("radius") or inp_region.get("radius_deg") or 5.0)
        return float(inp_region["ra"]), float(inp_region["dec"]), r, f"RA={float(inp_region['ra']):.4f}, Dec={float(inp_region['dec']):.4f}, r={r:g} deg", None
    if isinstance(inp_region, dict) and inp_region.get("ra_min") is not None:
        ra = (float(inp_region["ra_min"]) + float(inp_region["ra_max"])) / 2.0
        dec = (float(inp_region["dec_min"]) + float(inp_region["dec_max"])) / 2.0
        r = max(abs(float(inp_region["ra_max"]) - float(inp_region["ra_min"])) * math.cos(math.radians(dec)), abs(float(inp_region["dec_max"]) - float(inp_region["dec_min"]))) / 2.0
        return ra, dec, r, f"box RA {inp_region['ra_min']}-{inp_region['ra_max']}, Dec {inp_region['dec_min']}-{inp_region['dec_max']}", None
    key = str(preset or (inp_region if isinstance(inp_region, str) else "") or fallback).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in REGION_PRESETS or "ra" not in REGION_PRESETS[key]:
        key = fallback
    p = REGION_PRESETS[key]
    return p["ra"], p["dec"], p["radius_deg"], p["label"], key


def _cut_text(cuts: List[Dict[str, Any]]) -> str:
    return "; ".join(f"{c.get('column')} {c.get('op')} {c.get('value')}" for c in cuts if isinstance(c, dict))


def _fetch_frame(ctx: CallContext, result_id: str) -> Optional[pd.DataFrame]:
    store = ctx.result_store
    if store is None or not result_id:
        return None
    try:
        df, _meta, _status = store.lookup(result_id)
        return df
    except Exception:
        try:
            return store.get(result_id)
        except Exception:
            return None


def _append_card(ctx: CallContext, card: Dict[str, Any]) -> None:
    fn = ctx.services.get("append_run_result")
    if callable(fn):
        try:
            fn(card)
        except Exception:
            pass


def _image_card_from_plot(plot: Dict[str, Any], caption: str) -> Optional[Dict[str, Any]]:
    if not isinstance(plot, dict) or not plot.get("success"):
        return None
    url = plot.get("path") or (f"data:image/png;base64,{plot['image_base64']}" if plot.get("image_base64") else None)
    if not url:
        return None
    card = {"type": "image", "image_url": url, "caption": caption}
    if plot.get("plotly_spec"):
        card["plotly_spec"] = plot["plotly_spec"]
    return card


def _strip_heavy(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k not in ("image_base64", "plotly_spec", "path", "png_path", "pdf_path")}


# ═════════════════════════════════════════════════════════════════════════
# 10. datalab_healpix_density_map
# ═════════════════════════════════════════════════════════════════════════
class HealpixDensityMapInput(_In):
    catalog: str = "nsc_dr2"
    table: Optional[str] = None
    region: Optional[Any] = Field(default=None, description="{'ra','dec','radius'} in deg, or a preset name: lmc | smc | magellanic_bridge | sgr_stream | ngp | anticenter")
    preset: Optional[str] = None
    nside: int = 256
    log: bool = True
    cuts: Optional[List[Dict[str, Any]]] = Field(default=None, description="value cuts; default = point sources + blue/old colour window + magnitude window")
    apply_default_cuts: bool = True
    wait_s: int = 240
    title: Optional[str] = None


class HealpixDensityMap(BaseCapability):
    name = "datalab_healpix_density_map"
    description = (
        "ONE-CALL stellar density map of a wide region (default ~20x20 deg) from a Data Lab catalog, aggregated AT THE SERVER "
        "on the precomputed HEALPix column (nside 256 by default) with the default point-source + blue/old-population colour + "
        "magnitude cuts, rendered as a log-count HEALPix map card with structure notes (LMC/SMC, disk gradient, known objects). "
        "Region presets prefer validated fields (lmc, smc, magellanic_bridge, sgr_stream, ngp, anticenter) -- never the Galactic "
        "Centre. Tiles the query so it beats the 60 s sync window; partial coverage is reported, never hidden."
    )
    category = "datalab"
    InputModel = HealpixDensityMapInput
    annotations = {"read_only": True, "cost": "expensive", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        started = time.perf_counter()
        try:
            table = inp.table or registry.default_table(inp.catalog)
            info = registry.describe_table(inp.catalog, table)
            hp_cols = info.get("healpix_columns") or []
            if not hp_cols:
                return _native({"success": False, "status": "coverage_gap", "error": f"{inp.catalog}.{table} has no precomputed HEALPix column; use datalab_density_aggregate mode='grid' instead."})
            chosen = next((h for h in hp_cols if int(h.get("nside", 0)) == int(inp.nside)), None) or min(hp_cols, key=lambda h: abs(int(h.get("nside", 0)) - int(inp.nside)))
            ra, dec, radius, label, preset_used = _region(inp.region, inp.preset, fallback="lmc")
            cuts = list(inp.cuts or [])
            applied: List[str] = []
            color_cut = None
            morphology = None
            if inp.apply_default_cuts:
                morphology = registry.point_source_cut(inp.catalog, table)
                if morphology:
                    applied.append("point sources: " + " ".join(f"{k}={v}" for k, v in morphology.items()))
                try:
                    g_col, r_col = registry.mag_column(inp.catalog, table, "g"), registry.mag_column(inp.catalog, table, "r")
                    color_cut = {"bands": [g_col, r_col], **DEFAULT_DENSITY_COLOR}
                    applied.append(f"colour {DEFAULT_DENSITY_COLOR['min']} < {g_col}-{r_col} < {DEFAULT_DENSITY_COLOR['max']} (blue/old population)")
                    if not any(c.get("column") == g_col for c in cuts):
                        cuts += [{"column": g_col, "op": ">", "value": DEFAULT_MAG_WINDOW["bright"]}, {"column": g_col, "op": "<", "value": DEFAULT_MAG_WINDOW["faint"]}]
                        applied.append(f"{DEFAULT_MAG_WINDOW['bright']} < {g_col} < {DEFAULT_MAG_WINDOW['faint']}")
                except Exception:
                    pass
            if cuts:
                applied.append("cuts: " + _cut_text(cuts))
            predicates = builders.build_catalog_predicates(inp.catalog, table, color_cut=color_cut, value_cuts=cuts, morphology=morphology)
            budget_s = max(30.0, min(float(inp.wait_s), 240.0))
            agg = orchestration.tiled_density_aggregate(
                inp.catalog, table, mode="healpix", healpix_column=chosen["name"], ra=ra, dec=dec, radius_deg=radius,
                predicates=predicates, max_seconds=budget_s, client=ctx.service("datalab_client"), result_store=ctx.result_store,
                owner_id=_owner(ctx),
            )
        except Exception as e:
            return datalab_error(e)
        if not agg.get("result_id"):
            out = {**_strip_heavy(agg), "success": False, "status": "infrastructure_failure" if agg.get("budget_exhausted") else "coverage_gap",
                   "error": agg.get("error") or "no density cells returned", "region": label, "cuts_applied": applied}
            return _native(out)
        caption = inp.title or f"{inp.catalog} stellar density, HEALPix nside {chosen['nside']} ({chosen['scheme']}), log counts — {label}"
        caption += " | " + "; ".join(applied) if applied else ""
        try:
            plot = _run_analysis_plot("sky_density_map", {
                "result_id": agg["result_id"], "mode": "healpix", "healpix_col": "healpix", "nside": int(chosen["nside"]),
                "order": str(chosen.get("scheme", "nested")).lower(), "log_scale": bool(inp.log), "title": caption[:120],
            }, ctx)
        except Exception as e:
            return datalab_error(e)
        plot_native = plot.to_native() if hasattr(plot, "to_native") else {}
        frame = _fetch_frame(ctx, agg["result_id"])
        notes = list(agg.get("warnings") or [])
        structure = self._structure_notes(frame, ra, dec, radius, preset_used)
        partial = bool(agg.get("partial"))
        out = dict(plot_native)
        out.update({
            "success": bool(plot_native.get("success")),
            "status": ("partial" if partial else "ok") if plot_native.get("success") else "infrastructure_failure",
            "result_id": agg["result_id"], "region": label, "preset": preset_used, "ra": ra, "dec": dec, "radius_deg": radius,
            "healpix_column": chosen["name"], "nside": chosen["nside"], "scheme": chosen.get("scheme"),
            "cuts_applied": applied, "cells": int(len(frame)) if frame is not None else None,
            "tiles_completed": agg.get("tiles_completed"), "tiles_total": agg.get("tiles_total"), "coverage_summary": agg.get("coverage_summary"),
            "partial": partial, "structure_notes": structure, "warnings": notes, "elapsed_s": round(time.perf_counter() - started, 1),
            "_caption": caption,
            "note": "Map cell values are source counts per HEALPix pixel after the listed cuts; unvisited tiles are unknown, not zero.",
        })
        return _native(out)

    @staticmethod
    def _structure_notes(frame: Optional[pd.DataFrame], ra: float, dec: float, radius: float, preset: Optional[str]) -> List[str]:
        notes: List[str] = []
        if frame is None or frame.empty or "source_count" not in frame.columns:
            return notes
        counts = pd.to_numeric(frame["source_count"], errors="coerce").dropna()
        if counts.empty:
            return notes
        med, p95, mx = float(counts.median()), float(counts.quantile(0.95)), float(counts.max())
        notes.append(f"{len(counts)} cells; median {med:.0f}, 95th percentile {p95:.0f}, max {mx:.0f} sources per cell (contrast max/median = {mx / max(med, 1):.1f}).")
        for entry in registry.KNOWN_MW_OBJECTS:
            sep = registry.angular_separation_deg(ra, dec, entry["ra"], entry["dec"])
            if sep <= radius + entry.get("radius_deg", 0):
                notes.append(f"{entry['name']} ({entry['kind']}) lies in or at the edge of this region ({sep:.1f} deg from centre).")
        b = registry.galactic_latitude_deg(ra, dec)
        notes.append(f"Region centre at Galactic latitude b = {b:.1f} deg; " + ("expect a strong disk gradient" if abs(b) < 30 else "high-latitude halo field, expect a smooth background"))
        if preset:
            notes.append(f"Preset '{preset}': {REGION_PRESETS[preset].get('structure', '')}")
        return notes


# ═════════════════════════════════════════════════════════════════════════
# 11. datalab_stream_selection
# ═════════════════════════════════════════════════════════════════════════
class StreamSelectionInput(_In):
    cluster_name: Optional[str] = Field(default=None, description="e.g. 'Palomar 5' (resolved), or give ra/dec")
    ra: Optional[float] = None
    dec: Optional[float] = None
    pm_window: Optional[Dict[str, float]] = Field(default=None, description="{'pmra_min','pmra_max','pmdec_min','pmdec_max'} mas/yr; omit for auto (from the cluster core)")
    cmd_mask: Optional[Dict[str, float]] = Field(default=None, description="{'color_min','color_max','g_min','g_max'} on NSC g-r / g; omit for auto")
    field_deg: float = 10.0
    match_arcsec: float = 1.0
    small_limit: int = 50000
    limit: int = 5000
    title: Optional[str] = None


class StreamSelection(BaseCapability):
    name = "datalab_stream_selection"
    description = (
        "ONE-CALL tidal-stream/stream-star selection around a cluster: Gaia DR3 cone with a PROPER-MOTION window applied "
        "in a MATERIALIZED CTE (window auto-derived from the cluster core when omitted), q3c_join to NSC DR2 photometry on "
        "the indexed side, the CMD mask (g-r colour and g magnitude, point sources) applied IN THE SQL, then an on-sky "
        "scatter over a stream-sized field with the principal-axis (tail orientation) note. Every cut applied is listed in "
        "the result and caption. Use for 'trace Palomar 5's tidal tails with Gaia proper motions and NSC photometry'."
    )
    category = "datalab"
    InputModel = StreamSelectionInput
    annotations = {"read_only": True, "cost": "expensive", "produces": "image"}

    _AUTO_PM_HALF_WIDTH = 1.0  # mas/yr around the core median

    def run(self, inp, ctx) -> ToolResult:
        started = time.perf_counter()
        try:
            ra, dec, label = _resolve_coords(inp.cluster_name, inp.ra, inp.dec, ctx)
            ra, dec = float(ra), float(dec)
            field_r = max(0.5, min(float(inp.field_deg), 15.0)) / 2.0
            pm = dict(inp.pm_window or {})
            pm_source = "user"
            if not all(k in pm for k in ("pmra_min", "pmra_max", "pmdec_min", "pmdec_max")):
                pm, pm_source = self._auto_pm_window(ctx, ra, dec)
            mask = dict(inp.cmd_mask or {})
            mask_source = "user" if mask else "auto (metal-poor old population: 0.1 < g-r < 0.9, 16 < g < 23)"
            mask = {"color_min": mask.get("color_min", 0.1), "color_max": mask.get("color_max", 0.9), "g_min": mask.get("g_min", 16.0), "g_max": mask.get("g_max", 23.0)}
            sql, meta = self.build_sql(ra, dec, field_r, pm=pm, cmd_mask=mask, match_arcsec=float(inp.match_arcsec),
                                       small_limit=int(inp.small_limit), limit=int(inp.limit))
        except Exception as e:
            return datalab_error(e)
        if field_r > self._TILE_RADIUS_DEG:
            # A 5-degree Gaia cone x NSC join returned HTTP 502 twice live
            # (2026-09-23 L09): run it as concurrent sub-cones instead.
            native = self._run_tiled(ctx, ra, dec, field_r, pm=pm, mask=mask, match_arcsec=float(inp.match_arcsec),
                                     small_limit=int(inp.small_limit), limit=int(inp.limit), meta=meta)
            sql = native.get("sql_example") or sql
        else:
            res = execute_datalab_sql(sql, meta, tool_name=self.name, ctx=ctx)
            native = res.to_native() if hasattr(res, "to_native") else {}
        if not native.get("success") or not native.get("result_id"):
            native.setdefault("status", "infrastructure_failure")
            return _native(native)
        result_id = native["result_id"]
        frame = _fetch_frame(ctx, result_id)
        n = int(len(frame)) if frame is not None else int(native.get("rowcount") or 0)
        cuts_applied = [
            f"Gaia cone radius {field_r:g} deg around {label}",
            f"proper motion {pm['pmra_min']:.2f} < pmra < {pm['pmra_max']:.2f}, {pm['pmdec_min']:.2f} < pmdec < {pm['pmdec_max']:.2f} mas/yr ({pm_source})",
            f"q3c_join to nsc_dr2.object within {float(inp.match_arcsec):g} arcsec",
            f"CMD mask {mask['color_min']:g} < g-r < {mask['color_max']:g}, {mask['g_min']:g} < g < {mask['g_max']:g}, class_star > 0.5 ({mask_source})",
        ]
        orientation = self._orientation(frame, ra, dec)
        caption = (inp.title or f"{label}: stream-star candidates (Gaia DR3 PM window + NSC DR2 CMD mask)") + " | " + "; ".join(cuts_applied)
        try:
            plot = _run_analysis_plot("catalog_scatter", {
                "result_id": result_id, "x_expr": "ra", "y_expr": "dec", "color_by": "pmra", "invert_x": True,
                "title": caption[:110], "x_label": "RA (deg)", "y_label": "Dec (deg)",
            }, ctx)
            plot_native = plot.to_native() if hasattr(plot, "to_native") else {}
        except Exception as e:
            return datalab_error(e)
        out = dict(plot_native)
        out.update({
            "success": bool(plot_native.get("success")),
            "status": "ok" if plot_native.get("success") else "infrastructure_failure",
            "result_id": result_id, "n_selected": n, "cluster": label, "ra": ra, "dec": dec, "field_deg": field_r * 2,
            "pm_window": pm, "pm_window_source": pm_source, "cmd_mask": mask, "cmd_mask_source": mask_source,
            "cuts_applied": cuts_applied, "orientation": orientation, "sql": sql,
            "warnings": list(native.get("warnings") or []), "elapsed_s": round(time.perf_counter() - started, 1),
            "_caption": caption,
            "note": ("The plotted stars are exactly the rows returned by the SQL above (every cut is in the query). "
                     "A LIMIT-capped result is nearest-to-centre first; the count is a lower bound when the cap is hit."),
        })
        return _native(out)

    _TILE_RADIUS_DEG = 2.0

    def _run_tiled(self, ctx: CallContext, ra: float, dec: float, field_r: float, *, pm: Dict[str, float],
                   mask: Dict[str, float], match_arcsec: float, small_limit: int, limit: int, meta: Dict[str, Any]) -> Dict[str, Any]:
        """The same governed cross-match per sub-cone (each also bounded by the
        parent cone), four at a time under child deadlines; rows merged on
        source_id and stored as ONE result."""
        from services import datalab_sql_policy as policy
        from services.alma_server_side import run_concurrently
        from services.tool_budgets import bounded_timeout, remaining_seconds

        client = ctx.service("datalab_client")
        parent = f"q3c_radial_query(ra, dec, {ra:.8g}, {dec:.8g}, {field_r:.8g})"
        centers = orchestration._tile_cone_centers(field_r, self._TILE_RADIUS_DEG, cell_margin_deg=0.0)
        tiles = []
        for dx, dy in centers:
            tdec = max(-89.5, min(89.5, dec + dy))
            tra = (ra + dx / max(math.cos(math.radians(tdec)), 1e-6)) % 360.0
            sql_t, meta_t = self.build_sql(tra, tdec, self._TILE_RADIUS_DEG, pm=pm, cmd_mask=mask, match_arcsec=match_arcsec,
                                           small_limit=small_limit, limit=limit, parent_bound=parent)
            tiles.append(policy.validate(sql_t, source="builder", meta=meta_t))

        def _one(validated):
            seconds = bounded_timeout(float(getattr(client, "timeout", 60.0) or 60.0), minimum=3.0, label="Data Lab stream tile")
            return client.query(sql=validated.sql, fmt="pandas", async_fallback=False, timeout=seconds)

        frames, done, failed, errors = [], 0, 0, []
        for start in range(0, len(tiles), 4):
            left = remaining_seconds()
            if left is not None and left < 12.0:
                break
            batch = tiles[start:start + 4]
            outs = run_concurrently([(lambda v=v: _one(v)) for v in batch], wall_seconds=(left - 6.0) if left is not None else 120.0)
            for out in outs:
                if isinstance(out, BaseException):
                    failed += 1
                    errors.append(f"{type(out).__name__}: {str(out)[:120]}")
                else:
                    done += 1
                    frames.append(out.dataframe)
        total = len(tiles)
        if not frames:
            return {"success": False, "status": "infrastructure_failure", "tiles_total": total, "tiles_failed": failed,
                    "error": "every stream sub-cone query failed or ran out of budget: " + "; ".join(errors[:3])}
        merged = pd.concat(frames, ignore_index=True)
        key = "source_id" if "source_id" in merged.columns else None
        merged = merged.drop_duplicates(subset=[key] if key else ["ra", "dec"]).reset_index(drop=True)
        note = (f"Field split into {total} sub-cones of {self._TILE_RADIUS_DEG:g} deg (each bounded by the {field_r:g}-deg parent cone): "
                f"{done} completed" + (f", {failed} failed" if failed else "") + (f", {total - done - failed} not run (budget)" if total - done - failed else "") + ".")
        if done < total:
            note += " PARTIAL: unqueried sub-cones are unknown, not empty."
        store_meta = {**dict(meta or {}), "validated_sql": tiles[0].sql, "tool_name": self.name, "warnings": [note],
                      "provenance": {"validated_sql": tiles[0].sql, "tiles_total": total, "tiles_completed": done, "parent_bound": parent},
                      **({"owner_id": str(ctx.user_id)} if ctx.user_id else {})}
        result_id = ctx.result_store.put(merged, store_meta)
        return {"success": True, "result_id": result_id, "rowcount": int(len(merged)), "warnings": [note],
                "tiles_total": total, "tiles_completed": done, "partial": done < total, "sql_example": tiles[0].sql}

    @classmethod
    def build_sql(cls, ra: float, dec: float, radius_deg: float, *, pm: Dict[str, float], cmd_mask: Dict[str, float],
                  match_arcsec: float, small_limit: int, limit: int, parent_bound: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """The governed q3c cross-match with the PM window spliced into the Gaia
        CTE and the CMD mask into the join, so every cut is in the executed SQL."""
        sql, meta = builders.build_q3c_crossmatch(
            small_catalog="gaia_dr3", small_table="gaia_source", big_catalog="nsc_dr2", big_table="object",
            ra=ra, dec=dec, radius_deg=radius_deg, match_radius_arcsec=match_arcsec,
            small_columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag", "bp_rp"],
            big_columns=["id", "ra", "dec", "gmag", "rmag", "imag", "class_star"],
            small_limit=small_limit, limit=limit,
        )
        pm_pred = (
            f"  AND pmra BETWEEN {float(pm['pmra_min']):.4f} AND {float(pm['pmra_max']):.4f}\n"
            f"  AND pmdec BETWEEN {float(pm['pmdec_min']):.4f} AND {float(pm['pmdec_max']):.4f}\n"
            "  AND pmra < 'Infinity'::float8 AND pmdec < 'Infinity'::float8\n"
        )
        if parent_bound:
            pm_pred += f"  AND {parent_bound}\n"
        # Splice after the CTE's WHERE line (before ORDER BY).
        sql = re.sub(r"(    WHERE q3c_radial_query\([^\n]*\n)", lambda m: m.group(1) + pm_pred, sql, count=1)
        cmd_pred = (
            f"WHERE (big.gmag - big.rmag) BETWEEN {float(cmd_mask['color_min']):.3f} AND {float(cmd_mask['color_max']):.3f}\n"
            f"  AND big.gmag BETWEEN {float(cmd_mask['g_min']):.3f} AND {float(cmd_mask['g_max']):.3f}\n"
            "  AND big.gmag < 'Infinity'::float8 AND big.rmag < 'Infinity'::float8\n"
            "  AND big.class_star > 0.5\n"
        )
        sql = re.sub(r"(  ON q3c_join\([^\n]*\n)", lambda m: m.group(1) + cmd_pred, sql, count=1)
        meta = dict(meta)
        meta["stream_selection"] = {"pm_window": dict(pm), "cmd_mask": dict(cmd_mask), "point_source": "class_star > 0.5"}
        return sql, meta

    def _auto_pm_window(self, ctx: CallContext, ra: float, dec: float) -> Tuple[Dict[str, float], str]:
        """Median Gaia proper motion of the cluster core (bright, low-parallax
        stars within 5 arcmin) +/- 1 mas/yr."""
        sql, meta = builders.build_cone_select(
            "gaia_dr3", "gaia_source", ra=ra, dec=dec, radius_deg=5.0 / 60.0,
            columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag"], limit=2000,
            predicates=["phot_g_mean_mag < 19.5", "parallax < 1.0", "pmra < 'Infinity'::float8", "pmdec < 'Infinity'::float8"],
        )
        res = execute_datalab_sql(sql, meta, tool_name=self.name, ctx=ctx)
        native = res.to_native() if hasattr(res, "to_native") else {}
        frame = _fetch_frame(ctx, native.get("result_id", "")) if native.get("success") else None
        if frame is None or frame.empty or "pmra" not in frame.columns:
            return {"pmra_min": -4.0, "pmra_max": 0.0, "pmdec_min": -4.0, "pmdec_max": 0.0}, "fallback (core query returned nothing): generic -4..0 mas/yr window"
        pmra = pd.to_numeric(frame["pmra"], errors="coerce").dropna()
        pmdec = pd.to_numeric(frame["pmdec"], errors="coerce").dropna()
        if pmra.empty or pmdec.empty:
            return {"pmra_min": -4.0, "pmra_max": 0.0, "pmdec_min": -4.0, "pmdec_max": 0.0}, "fallback: generic -4..0 mas/yr window"
        m_ra, m_dec = float(pmra.median()), float(pmdec.median())
        h = self._AUTO_PM_HALF_WIDTH
        return ({"pmra_min": round(m_ra - h, 3), "pmra_max": round(m_ra + h, 3), "pmdec_min": round(m_dec - h, 3), "pmdec_max": round(m_dec + h, 3)},
                f"auto: median PM of {len(pmra)} core stars (5 arcmin, G<19.5, parallax<1) = ({m_ra:.2f}, {m_dec:.2f}) mas/yr, +/- {h:g}")

    @staticmethod
    def _orientation(frame: Optional[pd.DataFrame], ra0: float, dec0: float) -> Dict[str, Any]:
        if frame is None or frame.empty or "ra" not in frame.columns or "dec" not in frame.columns:
            return {"note": "no rows to orient"}
        x = (pd.to_numeric(frame["ra"], errors="coerce") - ra0) * math.cos(math.radians(dec0))
        y = pd.to_numeric(frame["dec"], errors="coerce") - dec0
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            return {"note": "too few rows to orient"}
        pts = np.vstack([x[m], y[m]])
        cov = np.cov(pts)
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
        pa = (math.degrees(math.atan2(major[0], major[1])) + 360.0) % 180.0  # east of north
        ratio = float(math.sqrt(max(vals) / max(min(vals), 1e-12)))
        return {"position_angle_deg_east_of_north": round(pa, 1), "axis_ratio": round(ratio, 2),
                "note": (f"selected stars are elongated along PA {pa:.0f} deg (axis ratio {ratio:.1f}) — the tail direction"
                         if ratio > 1.3 else "no strong elongation (axis ratio < 1.3): tails not evident in this selection")}


# ═════════════════════════════════════════════════════════════════════════
# 12. datalab_selection_diagram
# ═════════════════════════════════════════════════════════════════════════
class SelectionDiagramInput(_In):
    catalog: str = "gaia_dr3"
    table: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    target_name: Optional[str] = None
    radius_deg: float = 2.0
    cuts: Optional[List[Dict[str, Any]]] = None
    x_expr: str = "bp_rp"
    y_expr: str = "phot_g_mean_mag"
    abs_mag_from_parallax: bool = False
    quality_preset: str = Field(default="gaia_astrometric", description="gaia_astrometric | gaia_high_pm | nsc_point_sources | none")
    pm_total_min_mas_yr: Optional[float] = Field(default=None, description="keep sqrt(pmra^2+pmdec^2) > this (mas/yr)")
    overlay_locus: Optional[str] = Field(default=None, description="e.g. 'white_dwarf' to draw the WD sequence")
    limit: Optional[int] = None
    title: Optional[str] = None


class SelectionDiagram(BaseCapability):
    name = "datalab_selection_diagram"
    description = (
        "ONE query, ONE plot: a CMD / HR / colour diagram of a catalog cone with the selection cuts IN THE SQL -- quality presets "
        "(gaia_astrometric = parallax_over_error > 5, ruwe < 1.4, parallax > 0), a total-proper-motion floor, arbitrary "
        "value cuts, and absolute magnitude from parallax (y = G + 5 log10(parallax) - 10) -- plus an optional white-dwarf "
        "locus overlay with side-of-line counts. Use for 'high-proper-motion white-dwarf candidates in Gaia, HR diagram'. Target <= 60 s."
    )
    category = "datalab"
    InputModel = SelectionDiagramInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        started = time.perf_counter()
        try:
            table = inp.table or registry.default_table(inp.catalog)
            ra, dec, label = _resolve_coords(inp.target_name, inp.ra, inp.dec, ctx)
            preset_key = str(inp.quality_preset or "none").strip().lower()
            if preset_key not in QUALITY_PRESETS:
                return _native({"success": False, "error": f"unknown quality_preset {inp.quality_preset!r}; use {', '.join(QUALITY_PRESETS)}"})
            cuts = [dict(c) for c in QUALITY_PRESETS[preset_key]] + [dict(c) for c in (inp.cuts or [])]
            pm_floor = float(inp.pm_total_min_mas_yr) if inp.pm_total_min_mas_yr is not None else None
            y_expr = str(inp.y_expr or "phot_g_mean_mag")
            if inp.abs_mag_from_parallax:
                y_expr = f"{y_expr} + 5*log10(parallax) - 10"
            title = inp.title or f"{inp.catalog} selection diagram: {label}"
            kwargs: Dict[str, Any] = dict(
                x_expr=str(inp.x_expr), y_expr=y_expr, value_cuts=cuts, overlay_locus=inp.overlay_locus,
                limit=(int(inp.limit) if inp.limit is not None else None), title=title,
                client=ctx.service("datalab_client"), result_store=ctx.result_store,
                owner_id=_owner(ctx),
            )
            if pm_floor is not None:
                kwargs["pm_total_min"] = pm_floor  # squared-component predicate built in the SQL
            # Wide cones are split into 1.5-degree sub-cones run concurrently
            # (a 5-degree selective Gaia cone blows the 60 s sync window: L06).
            kwargs["tile_threshold_deg"] = 1.5
            out = orchestration.color_magnitude_diagram(inp.catalog, table, float(ra), float(dec), float(inp.radius_deg), **kwargs)
        except Exception as e:
            return datalab_error(e)
        if not isinstance(out, dict):
            return _native({"success": False, "error": "diagram returned no result"})
        applied = [f"cone {float(inp.radius_deg):g} deg around {label}", f"quality preset {preset_key}: {_cut_text(QUALITY_PRESETS[preset_key]) or 'none'}"]
        if inp.cuts:
            applied.append("cuts: " + _cut_text(inp.cuts))
        if inp.pm_total_min_mas_yr is not None:
            applied.append(f"total proper motion > {float(inp.pm_total_min_mas_yr):g} mas/yr")
        applied.append(f"x = {inp.x_expr}, y = {y_expr}" + (" (absolute magnitude from parallax)" if inp.abs_mag_from_parallax else ""))
        caption = title + " | " + "; ".join(applied)
        out.update({
            "status": "ok" if out.get("success") else ("coverage_gap" if out.get("coverage_gap") else "infrastructure_failure"),
            "cuts_applied": applied, "target": label, "ra": float(ra), "dec": float(dec), "quality_preset": preset_key,
            "elapsed_s": round(time.perf_counter() - started, 1), "_caption": caption,
            "note": "Every listed cut is in the executed SQL (see the query panel); the plotted sample is exactly the query result.",
        })
        return _native(out)


# ═════════════════════════════════════════════════════════════════════════
# 13. datalab_target_class_summary
# ═════════════════════════════════════════════════════════════════════════
class TargetClassSummaryInput(_In):
    catalog: str = "desi_dr1"
    table: str = "zpix"
    target_class: Optional[str] = Field(default="LRG", description="bitmask name from the registry (LRG | ELG | QSO | BGS_ANY | MWS_ANY)")
    bitmask_expr: Optional[str] = Field(default=None, description="raw bitmask predicate, e.g. '(desi_target & 1) != 0' (validated)")
    z_range: List[float] = Field(default_factory=lambda: [0.4, 0.8])
    dz: float = 0.02
    cell_deg: float = 2.0
    title: Optional[str] = None


class TargetClassSummary(BaseCapability):
    name = "datalab_target_class_summary"
    description = (
        "ONE-CALL redshift distribution n(z) AND sky footprint of a DESI (or other bitmask-selected) target class, both "
        "computed AT THE SERVER in one pass each (GROUP BY z bin; GROUP BY sky cell), rendered as one two-panel card, with "
        "the bitmask decoded (e.g. LRG = bit 0 of desi_target -> (desi_target & 1) != 0) and the registry quality cuts stated. "
        "Use for 'DESI DR1 LRGs at 0.4 < z < 0.8: redshift distribution and sky footprint'."
    )
    category = "datalab"
    InputModel = TargetClassSummaryInput
    annotations = {"read_only": True, "cost": "moderate", "produces": "image"}

    _BITMASK_RE = re.compile(r"^\(?\s*([a-z_][a-z0-9_]*)\s*&\s*(\d+)\s*\)?\s*(?:!=|<>|>)\s*0\s*$", re.I)

    def run(self, inp, ctx) -> ToolResult:
        started = time.perf_counter()
        try:
            info = registry.describe_table(inp.catalog, inp.table)
            z_lo, z_hi = sorted(float(v) for v in (inp.z_range or [0.4, 0.8])[:2])
            dz = max(0.001, float(inp.dz))
            cell = max(0.25, float(inp.cell_deg))
            bit_pred, bit_explain = self._bitmask(info, inp.target_class, inp.bitmask_expr)
            quality = registry.default_quality_cuts(inp.catalog, inp.table)
            q_preds = builders.build_catalog_predicates(inp.catalog, inp.table, value_cuts=quality) if quality else []
            z_col = "z"
            common = [bit_pred, f"{z_col} >= {z_lo:g}", f"{z_col} < {z_hi:g}", f"{z_col} < 'Infinity'::float8"] + list(q_preds)
            where = "\n  AND ".join(f"({p})" if not p.startswith("(") else p for p in common)
            hist_sql = (
                f"SELECT FLOOR({z_col} / {dz:g}) * {dz:g} AS z_bin, COUNT(*) AS source_count\n"
                f"FROM {info['qualified_name']}\nWHERE {where}\nGROUP BY z_bin\nORDER BY z_bin\nLIMIT 5000"
            )
            hist_meta = builders._meta("zhistogram", info, aggregate=True, row_limit=5000)
            fp_sql, fp_meta = builders.build_density_aggregate(inp.catalog, inp.table, mode="grid", step_deg=cell, all_sky=True,
                                                               predicates=common, limit=builders.MAX_ROW_LIMIT)
        except Exception as e:
            return datalab_error(e)
        hist_res = execute_datalab_sql(hist_sql, hist_meta, tool_name=self.name, ctx=ctx)
        hist_native = hist_res.to_native() if hasattr(hist_res, "to_native") else {}
        if not hist_native.get("success"):
            hist_native.setdefault("status", "infrastructure_failure")
            return _native(hist_native)
        fp_res = execute_datalab_sql(fp_sql, fp_meta, tool_name=self.name, ctx=ctx)
        fp_native = fp_res.to_native() if hasattr(fp_res, "to_native") else {}
        hist = _fetch_frame(ctx, hist_native.get("result_id", ""))
        fp = _fetch_frame(ctx, fp_native.get("result_id", "")) if fp_native.get("success") else None
        total = int(pd.to_numeric(hist["source_count"], errors="coerce").sum()) if hist is not None and not hist.empty else 0
        fp_truncated = bool(fp is not None and len(fp) >= builders.MAX_ROW_LIMIT)
        # -- one two-panel figure -----------------------------------------
        try:
            from services.plotting import PlottingService

            plotting = PlottingService()
            plt = plotting._apply_style(dark=False)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
            if hist is not None and not hist.empty:
                ax1.bar(pd.to_numeric(hist["z_bin"], errors="coerce"), pd.to_numeric(hist["source_count"], errors="coerce"), width=dz * 0.95, align="edge")
            ax1.set_xlabel("redshift z"); ax1.set_ylabel(f"N per Δz = {dz:g}"); ax1.set_title(f"n(z), {inp.target_class or 'bitmask'}: {total:,} objects")
            if fp is not None and not fp.empty:
                sc = ax2.scatter(pd.to_numeric(fp["ra_bin"], errors="coerce"), pd.to_numeric(fp["dec_bin"], errors="coerce"),
                                 c=np.log10(pd.to_numeric(fp["source_count"], errors="coerce").clip(lower=1)), s=max(4, int(120 * cell / 2)), marker="s", cmap="viridis")
                fig.colorbar(sc, ax=ax2, label="log10 N per cell")
                ax2.invert_xaxis()
            ax2.set_xlabel("RA (deg)"); ax2.set_ylabel("Dec (deg)"); ax2.set_title(f"sky footprint ({cell:g}° cells)" + (" — CAPPED" if fp_truncated else ""))
            fig.tight_layout()
            saved = plotting._save_and_encode(fig, f"target_class_{int(time.time())}")
        except Exception as e:
            return datalab_error(e)
        applied = [bit_explain, f"{z_lo:g} <= z < {z_hi:g}", "registry quality cuts: " + (_cut_text(quality) or "none")]
        caption = (inp.title or f"{inp.catalog}.{inp.table} {inp.target_class or 'bitmask'}: n(z) and sky footprint") + " | " + "; ".join(applied)
        warnings = list(hist_native.get("warnings") or []) + list(fp_native.get("warnings") or [])
        if fp_truncated:
            warnings.append(f"Footprint aggregate hit the {builders.MAX_ROW_LIMIT}-cell cap (sparsest cells dropped): increase cell_deg for a complete footprint.")
        if not fp_native.get("success"):
            warnings.append("Footprint aggregate failed: " + str(fp_native.get("error"))[:200])
        out = {
            "success": True, "status": "partial" if (fp_truncated or not fp_native.get("success")) else "ok",
            "image_base64": saved.get("base64_png"), "path": saved.get("web_url"), "png_path": saved.get("png_path"),
            "histogram_result_id": hist_native.get("result_id"), "footprint_result_id": fp_native.get("result_id"),
            "n_objects": total, "n_z_bins": int(len(hist)) if hist is not None else 0, "n_footprint_cells": int(len(fp)) if fp is not None else 0,
            "z_range": [z_lo, z_hi], "dz": dz, "cell_deg": cell,
            "histogram": hist.to_dict("records")[:200] if hist is not None else [],
            "bitmask": bit_explain, "cuts_applied": applied, "sql": {"histogram": hist_sql, "footprint": fp_sql},
            "warnings": warnings, "elapsed_s": round(time.perf_counter() - started, 1), "_caption": caption,
            "note": "Both panels are server-side aggregates over the full table (no row sample); counts are exact except where a cap is flagged.",
        }
        return _native(out)

    def _bitmask(self, info: Dict[str, Any], target_class: Optional[str], expr: Optional[str]) -> Tuple[str, str]:
        if expr:
            m = self._BITMASK_RE.match(str(expr).strip())
            if not m:
                raise ValueError("bitmask_expr must look like '(desi_target & 1) != 0'")
            col, mask = m.group(1), int(m.group(2))
            if col not in (info.get("columns") or []):
                raise ValueError(f"unknown bitmask column {col!r}")
            bits = [i for i in range(64) if mask & (1 << i)]
            return f"({col} & {mask}) != 0", f"bitmask {col} & {mask} != 0 (bit(s) {bits})"
        bitmasks = info.get("bitmasks") or {}
        for col, names in bitmasks.items():
            key = str(target_class or "").upper()
            if key in {k.upper(): k for k in names}:
                real = {k.upper(): k for k in names}[key]
                bit = int(names[real])
                return f"({col} & {1 << bit}) != 0", f"{real} = bit {bit} of {col} -> ({col} & {1 << bit}) != 0"
        raise ValueError(f"target_class {target_class!r} not in the registry bitmasks {bitmasks}")


# ═════════════════════════════════════════════════════════════════════════
# 14. datalab_satellite_search
# ═════════════════════════════════════════════════════════════════════════
class SatelliteSearchInput(_In):
    region: Optional[Any] = Field(default=None, description="{'ra','dec','radius'} deg or a preset: delve_south | hydra2 | fornax_field | smc | lmc")
    preset: Optional[str] = None
    survey: str = Field(default="nsc", description="nsc | delve | des")
    tile_deg: float = 0.05
    iso_filter: str = Field(default="old_metal_poor", description="old_metal_poor (blue/old colour window) | none")
    top_n: int = 5
    vet: bool = True
    fov_deg: float = 0.05
    band: str = "g"
    cmd_candidates: int = 3
    smash_field: Optional[int] = Field(default=None, description="SMASH DR1 field id (e.g. 169): scan exactly that field (fieldid = N) instead of a cone")
    color_min: Optional[float] = Field(default=None, description="override the g-r colour window minimum (e.g. -0.3 for blue main-sequence stars)")
    color_max: Optional[float] = Field(default=None, description="override the g-r colour window maximum (e.g. 0.5)")
    mag_min: Optional[float] = None
    mag_max: Optional[float] = None


class SatelliteSearch(BaseCapability):
    name = "datalab_satellite_search"
    description = (
        "ONE-CALL Milky Way satellite (dwarf-galaxy) candidate search: confirms the sky area, runs a server-side density "
        "aggregate of POINT SOURCES inside a colour-magnitude box (old/metal-poor by default; pass color_min/color_max for "
        "e.g. blue main-sequence stars) over a validated region or ONE SMASH field (smash_field=169 scans fieldid = 169), "
        "detects peaks and scores each by its star-count EXCESS over a local annulus background (aperture significance, "
        "robust to field edges), flags known satellites/globulars as re-detections, and vets the top candidates with image "
        "cutouts and one CMD each. Returns the ranked table, the cuts applied and a density-map card. Region presets prefer clean "
        "deep fields (delve_south), never the poles. Target <= 150 s."
    )
    category = "datalab"
    InputModel = SatelliteSearchInput
    annotations = {"read_only": True, "cost": "expensive", "produces": "image"}

    _SURVEY_TABLES = {"nsc": ("nsc_dr2", "object"), "delve": ("delve_dr3", None), "des": ("des_dr1", None)}
    # Peak detection: aperture = the peak cell and its neighbours (<= 1.1 cells), background = cells
    # 0.25-0.6 deg away, variance = max(annulus variance, Poisson). Validated live on SMASH field 169:
    # Hydra II ranks first at 6.1 sigma, the next peak 3.5 sigma (2026-09-23).
    _PEAKS = {"method": "aperture excess over a plane fitted to the local annulus", "aperture_cells_radius": 1.1, "annulus_deg": [0.25, 0.6],
              "min_separation_deg": 0.15, "candidate_sigma": 5.0, "report_sigma": 3.0}

    def run(self, inp, ctx) -> ToolResult:
        from services.tool_budgets import remaining_seconds

        started = time.perf_counter()
        try:
            survey = str(inp.survey or "nsc").lower()
            if survey not in self._SURVEY_TABLES:
                return _native({"success": False, "error": "survey must be nsc, delve or des"})
            catalog, table = self._SURVEY_TABLES[survey]
            table = table or registry.default_table(catalog)
            field_cut: List[Dict[str, Any]] = []
            if inp.smash_field is not None:
                # L07: a SMASH field is a key-bounded selection, never a dummy cone.
                catalog, table, survey = "smash_dr1", "object", "smash"
                ra, dec, radius, label = self._smash_field_extent(ctx, int(inp.smash_field))
                preset_used = None
                field_cut = [{"column": "fieldid", "op": "=", "value": int(inp.smash_field)}]
            else:
                ra, dec, radius, label, preset_used = _region(inp.region, inp.preset, fallback="delve_south")
            area = orchestration.confirm_cone_area(ra, dec, radius)
            morphology = registry.point_source_cut(catalog, table)
            g_col, r_col = registry.mag_column(catalog, table, "g"), registry.mag_column(catalog, table, "r")
            value_cuts: List[Dict[str, Any]] = list(field_cut)
            applied = [f"region {label} ({area.get('area_deg2', 0):.1f} deg2)",
                       f"point sources: {morphology.get('column')} between {morphology.get('between')}" if morphology else "no morphology cut available"]
            smash = bool(field_cut)
            if smash:
                applied.append(f"SMASH field: fieldid = {int(inp.smash_field)}; depthflag > 1 (deep exposures)")
                value_cuts.append({"column": "depthflag", "op": ">", "value": 1})
            color_cut = None
            if str(inp.iso_filter or "").lower() != "none" or inp.color_min is not None or inp.color_max is not None:
                # SMASH default = blue main-sequence / turnoff box reaching g = 25
                # (distant dwarfs have their turnoff near g ~ 24.5).
                cmin = float(inp.color_min) if inp.color_min is not None else (-0.4 if smash else DEFAULT_DENSITY_COLOR["min"])
                cmax = float(inp.color_max) if inp.color_max is not None else (0.4 if smash else DEFAULT_DENSITY_COLOR["max"])
                mmin = float(inp.mag_min) if inp.mag_min is not None else (None if smash else 18.0)
                mmax = float(inp.mag_max) if inp.mag_max is not None else (25.0 if smash else DEFAULT_MAG_WINDOW["faint"])
                color_cut = {"bands": [g_col, r_col], "min": cmin, "max": cmax}
                if mmin is not None:
                    value_cuts.append({"column": g_col, "op": ">", "value": mmin})
                value_cuts.append({"column": g_col, "op": "<", "value": mmax})
                applied.append(f"colour-magnitude box: {cmin:g} < {g_col}-{r_col} < {cmax:g}, "
                               + (f"{mmin:g} < " if mmin is not None else "") + f"{g_col} < {mmax:g}")
            merged_cuts, quality_note = registry.merge_default_quality_cuts(catalog, table, value_cuts)
            predicates = builders.build_catalog_predicates(catalog, table, color_cut=color_cut, value_cuts=merged_cuts, morphology=morphology)
            step = max(0.01, float(inp.tile_deg))
            client, store = ctx.service("datalab_client"), ctx.result_store
            if radius >= 3.0:
                dens = orchestration.tiled_density_aggregate(catalog, table, mode="grid", step_deg=step, ra=ra, dec=dec, radius_deg=radius,
                                                             predicates=predicates, client=client, result_store=store, owner_id=_owner(ctx))
                result_id = dens.get("result_id")
                dens_notes = list(dens.get("warnings") or [])
                partial = bool(dens.get("partial"))
                if not result_id:
                    return _native({"success": False, "status": "infrastructure_failure", "region": label, "cuts_applied": applied,
                                    "error": dens.get("error") or "density scan returned no cells"})
            else:
                sql, meta = builders.build_density_aggregate(catalog, table, mode="grid", step_deg=step, ra=ra, dec=dec,
                                                             radius_deg=radius, predicates=predicates, limit=builders.MAX_ROW_LIMIT)
                result_id, _res = orchestration._run_builder_sql(sql, meta, client=client, result_store=store, owner_id=_owner(ctx))
                dens_notes, partial = [], False
            if quality_note:
                dens_notes.append(quality_note)
        except Exception as e:
            return datalab_error(e)
        frame = _fetch_frame(ctx, result_id)
        bg_med, bg_sigma = self._background(frame)
        # -- density map card (log counts) ---------------------------------
        try:
            dm = _run_analysis_plot("sky_density_map", {
                "result_id": result_id, "mode": "hist2d", "ra_col": "ra_bin", "dec_col": "dec_bin", "count_col": "source_count",
                "bins": max(20, min(120, int(round(2 * radius / step)))), "log_scale": True,
                "title": f"Point-source density ({step:g} deg cells) — {label}"[:110],
            }, ctx)
            dm_native = dm.to_native() if hasattr(dm, "to_native") else {}
            card = _image_card_from_plot(dm_native, f"Density of the selected point sources ({step:g} deg cells, log counts) — {label} | " + "; ".join(applied))
            if card:
                _append_card(ctx, card)
        except Exception as exc:
            dens_notes.append(f"density map failed: {type(exc).__name__}: {str(exc)[:120]}")
        # -- peak detection: aperture excess over a local annulus ------------
        ranked: List[Dict[str, Any]] = []
        claimed_known: set = set()
        for c in self.aperture_peaks(frame, step, max_candidates=max(10, 2 * int(inp.top_n))):
            if c["significance"] < self._PEAKS["report_sigma"]:
                continue
            known = registry.match_known_mw_object(float(c["ra"]), float(c["dec"]))
            near = None
            if known and known["name"] in claimed_known:
                # A known object is re-detected by its STRONGEST peak only; a
                # weaker peak nearby is not a second re-detection.
                near, known = f"near {known['name']} ({known['separation_deg']} deg away; already matched by a stronger peak)", None
            elif known:
                claimed_known.add(known["name"])
            weak = c["significance"] < self._PEAKS["candidate_sigma"]
            ranked.append({
                **c,
                "known_object": (f"{known['name']} ({known['kind']}, {known['separation_deg']} deg away)" if known else None),
                "note": near,
                "verdict": ("re-detection of a known object" if known else
                            ("weak (3-5 sigma): likely a fluctuation" if weak else f"candidate (>= {self._PEAKS['candidate_sigma']:g} sigma)")),
            })
        for i, r in enumerate(ranked, start=1):
            r["rank"] = i
        top = ranked[: max(1, int(inp.top_n))]
        # -- cutouts at the matched-filter peaks -----------------------------
        grid_slim = None
        if top and inp.vet:
            left = remaining_seconds()
            if left is None or left > 40:
                try:
                    grid = ctx.service("datalab_image_service").cutout_grid(
                        [{"ra": r["ra"], "dec": r["dec"], "label": f"#{r['rank']} {r['significance']}σ"} for r in top],
                        float(inp.fov_deg), band=str(inp.band), catalog=catalog if catalog != "smash_dr1" else "nsc_dr2",
                        title=f"Top {len(top)} density peaks — {label}"[:100],
                    )
                    gcard = _image_card_from_plot(grid, f"Image cutouts ({inp.band}-band, {float(inp.fov_deg):g} deg) at the top density peaks — {label}")
                    if gcard:
                        _append_card(ctx, gcard)
                    grid_slim = _strip_heavy(grid) if isinstance(grid, dict) else None
                except Exception as exc:
                    dens_notes.append(f"cutout grid failed: {type(exc).__name__}: {str(exc)[:120]}")
            else:
                dens_notes.append("cutouts skipped: tool budget nearly exhausted")
        # -- vetting: one CMD per top NEW candidate ---------------------------
        cmd_cards = []
        if inp.vet:
            for cand in [r for r in top if not r["known_object"]][: max(0, int(inp.cmd_candidates))] or [r for r in top][:1]:
                left = remaining_seconds()
                if left is not None and left < 30:
                    dens_notes.append("CMD vetting stopped early: tool budget nearly exhausted.")
                    break
                try:
                    cmd = orchestration.color_magnitude_diagram(
                        catalog, table, float(cand["ra"]), float(cand["dec"]), 0.1, blue_band="g", red_band="r", point_sources=True,
                        value_cuts=list(field_cut) or None,
                        title=f"CMD at peak #{cand['rank']} ({cand['ra']:.3f}, {cand['dec']:.3f}), r = 6 arcmin", limit=3000,
                        client=client, result_store=store, owner_id=_owner(ctx),
                    )
                    card = _image_card_from_plot(cmd, f"Peak #{cand['rank']} CMD (g vs g-r, point sources, 6 arcmin): {cand['verdict']}")
                    if card:
                        _append_card(ctx, card)
                        cmd_cards.append({"rank": cand["rank"], "n_stars": (cmd or {}).get("rowcount") or (cmd or {}).get("points")})
                        cand["cmd"] = "CMD card attached: judge by a coherent main sequence / RGB, not by star counts alone"
                except Exception as exc:
                    cand["cmd"] = f"CMD failed: {type(exc).__name__}"
        n_new = sum(1 for r in ranked if not r["known_object"] and r["significance"] >= self._PEAKS["candidate_sigma"])
        n_known = sum(1 for r in ranked if r["known_object"])
        out = {
            "success": True, "status": "partial" if partial else "ok", "survey": survey, "catalog": catalog, "table": table,
            "region": label, "preset": preset_used, "ra": ra, "dec": dec, "radius_deg": radius, "sky_area": area,
            "result_id": result_id, "cuts_applied": applied, "cell_size_deg": step,
            "peak_detection": dict(self._PEAKS),
            "background": {"median_per_cell": round(bg_med, 1), "sigma_mad": round(bg_sigma, 2), "cells": int(len(frame)) if frame is not None else None},
            "candidates": ranked, "n_candidates_over_5sigma": n_new, "n_known_objects": n_known,
            "headline": (f"{len(ranked)} peak(s) >= 3 sigma above the local background: {n_new} new candidate(s) >= 5 sigma, "
                         f"{n_known} known object(s) re-detected"
                         + ("" if not ranked else f"; strongest: #1 at ({ranked[0]['ra']}, {ranked[0]['dec']}), "
                            f"{ranked[0]['significance']} sigma, excess {ranked[0]['excess_stars']} stars"
                            + (f" = {ranked[0]['known_object']}" if ranked[0]["known_object"] else ""))),
            "cutout_grid": grid_slim, "cmd_cards": cmd_cards, "warnings": dens_notes,
            "elapsed_s": round(time.perf_counter() - started, 1),
            "note": ("Strategy: point-source selection + colour-magnitude box -> server-side density grid -> peaks scored by "
                     "star-count excess over a local 0.25-0.6 deg annulus (significance uses the annulus variance, not just Poisson) -> known-object screening -> "
                     "cutouts + CMD per candidate. A real satellite shows a coherent CMD sequence; a peak without one is a field "
                     "fluctuation or a background galaxy cluster."),
        }
        return _native(out)

    @staticmethod
    def _smash_field_extent(ctx: CallContext, fieldid: int) -> Tuple[float, float, float, str]:
        """(ra, dec, radius, label) of a SMASH DR1 field from one key-bounded
        aggregate (fieldid is an indexed bound column)."""
        info = registry.describe_table("smash_dr1", "object")
        sql = ("SELECT AVG(ra) AS ra, AVG(dec) AS dec, MIN(ra) AS ra_min, MAX(ra) AS ra_max, "
               "MIN(dec) AS dec_min, MAX(dec) AS dec_max, COUNT(*) AS n\n"
               f"FROM smash_dr1.object\nWHERE fieldid = {int(fieldid)}")
        meta = builders._meta("field_extent", info, aggregate=True, row_limit=1)
        _rid, result = orchestration._run_builder_sql(sql, meta, client=ctx.service("datalab_client"), result_store=ctx.result_store,
                                                     owner_id=_owner(ctx))
        df = result.dataframe
        if df is None or df.empty or int(df.iloc[0].get("n") or 0) == 0:
            raise ValueError(f"SMASH DR1 has no objects with fieldid = {int(fieldid)}")
        row = df.iloc[0]
        ra, dec = float(row["ra"]), float(row["dec"])
        half = max(abs(float(row["ra_max"]) - float(row["ra_min"])) * math.cos(math.radians(dec)), abs(float(row["dec_max"]) - float(row["dec_min"]))) / 2.0
        return ra, dec, min(max(half * 1.05, 0.3), 2.0), f"SMASH DR1 field {int(fieldid)} (centre RA={ra:.3f}, Dec={dec:.3f}, {int(row['n'])} objects)"

    @classmethod
    def aperture_peaks(cls, frame: Optional[pd.DataFrame], step: float, *, max_candidates: int = 10) -> List[Dict[str, Any]]:
        """Density peaks (greedy, >= min separation apart) scored by their
        star-count excess in a small aperture over a plane fitted to a local
        annulus; variance = max(raw annulus variance, Poisson mean).
        Peaks whose annulus is < 80 % populated (field edges) are skipped."""
        if frame is None or frame.empty or not {"ra_bin", "dec_bin", "source_count"} <= set(frame.columns):
            return []
        df = frame[["ra_bin", "dec_bin", "source_count"]].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            return []
        cosd = math.cos(math.radians(float(df["dec_bin"].median())))
        X, Y, N = df["ra_bin"].to_numpy() * cosd, df["dec_bin"].to_numpy(), df["source_count"].to_numpy(dtype=float)
        ap_r = step * float(cls._PEAKS["aperture_cells_radius"])
        # The annulus/separation scale with the cell size so a coarse grid still
        # has a clean gap between aperture and background.
        a0 = max(float(cls._PEAKS["annulus_deg"][0]), 2.5 * step)
        a1 = max(float(cls._PEAKS["annulus_deg"][1]), 6.0 * step)
        sep = max(float(cls._PEAKS["min_separation_deg"]), 3.0 * step)
        picked: List[Tuple[float, float]] = []
        out: List[Dict[str, Any]] = []
        # Evaluate many peaks (in raw-count order) before ranking by
        # significance: in a crowded region the brightest cells sit on the
        # smooth host (LMC body, disc), not on a compact dwarf.
        max_evaluated = max(200, int(max_candidates))

        def _spacing(v: np.ndarray, fallback: float) -> float:
            u = np.unique(np.round(v, 6))
            dv = np.diff(u)
            dv = dv[dv > 1e-9]
            return float(np.median(dv)) if dv.size else fallback

        cell_area = _spacing(X, step * cosd) * _spacing(Y, step)
        expected_annulus = math.pi * (a1 ** 2 - a0 ** 2) / max(cell_area, 1e-12)
        for idx in np.argsort(-N):
            if len(out) >= max_evaluated:
                break
            x0, y0 = X[idx], Y[idx]
            if any(math.hypot(x0 - px, y0 - py) < sep for px, py in picked):
                continue
            d = np.hypot(X - x0, Y - y0)
            ap, an = d <= ap_r, (d >= a0) & (d <= a1)
            n_an = int(an.sum())
            if n_an < max(8, 0.8 * expected_annulus):
                continue  # at a field edge / footprint gap: the background is not measurable
            # Background = a plane fitted to the annulus, evaluated over the
            # aperture: a large-scale gradient (host galaxy, depth variation)
            # then cancels instead of masquerading as an excess.
            dx, dy = X[an] - x0, Y[an] - y0
            A = np.column_stack([np.ones(n_an), dx, dy])
            coef, *_ = np.linalg.lstsq(A, N[an], rcond=None)
            n_ap = int(ap.sum())
            bg_ap = coef[0] * n_ap + coef[1] * float((X[ap] - x0).sum()) + coef[2] * float((Y[ap] - y0).sum())
            bg = max(float(coef[0]), 0.0)
            # Noise = the RAW annulus variance (not the plane residual): conservative,
            # so depth-pattern structure does not read as significance. Field 169:
            # Hydra II 6.1 sigma vs next 3.5 (residual variance gave 6.2 vs 5.5).
            var = max(float(N[an].var(ddof=1)), bg, 1.0)
            excess = float(N[ap].sum() - bg_ap)
            picked.append((x0, y0))
            out.append({"ra": round(float(df["ra_bin"].to_numpy()[idx]), 4), "dec": round(float(y0), 4),
                        "peak_cell_count": int(N[idx]), "excess_stars": round(excess, 1), "background_per_cell": round(bg, 1),
                        "aperture_cells": n_ap, "annulus_cells": n_an, "significance": round(excess / math.sqrt(n_ap * var), 1)})
        return sorted(out, key=lambda o: -o["significance"])[: max(1, int(max_candidates))]

    @staticmethod
    def _nearest_cell_count(frame: Optional[pd.DataFrame], ra: float, dec: float, step: float) -> Optional[int]:
        if frame is None or frame.empty or not {"ra_bin", "dec_bin", "source_count"} <= set(frame.columns):
            return None
        d = (pd.to_numeric(frame["ra_bin"], errors="coerce") - ra).abs() * math.cos(math.radians(dec)) + (pd.to_numeric(frame["dec_bin"], errors="coerce") - dec).abs()
        near = frame[d <= 1.5 * step]
        if near.empty:
            return None
        return int(pd.to_numeric(near["source_count"], errors="coerce").max())

    @staticmethod
    def _peak_count(p: Dict[str, Any]) -> Optional[int]:
        m = re.search(r"n=(\d+)", str(p.get("label") or ""))
        return int(m.group(1)) if m else None

    @staticmethod
    def _background(frame: Optional[pd.DataFrame]) -> Tuple[float, float]:
        if frame is None or frame.empty or "source_count" not in frame.columns:
            return 0.0, 0.0
        counts = pd.to_numeric(frame["source_count"], errors="coerce").dropna().astype(float)
        if counts.empty:
            return 0.0, 0.0
        med = float(counts.median())
        mad = float((counts - med).abs().median())
        return med, max(1.4826 * mad, math.sqrt(max(med, 1.0)))


CAPABILITIES: List[BaseCapability] = [
    HealpixDensityMap(),
    StreamSelection(),
    SelectionDiagram(),
    TargetClassSummary(),
    SatelliteSearch(),
]


def _self_register() -> None:
    """Append these tools to capabilities.datalab.CAPABILITIES (deduplicated)
    so the agent's _datalab_tool_fn / _datalab_image_tool_fn find them by name."""
    from capabilities import datalab as _dl

    names = {c.name for c in _dl.CAPABILITIES}
    for cap in CAPABILITIES:
        if cap.name not in names:
            _dl.CAPABILITIES.append(cap)


_self_register()
