"""Data Lab catalog-science orchestration.

Implements the Tier 6-7 workflows on top of the P0 builders/governor/result-store and
the P1 image/analysis layers:
  - density_then_cutouts  (P12): densest cells in a cone -> SIA cutout grid of the top-N.
  - tiled_sky_scan        (P15): tile a footprint into q3c cones, server-side density
                          aggregate per tile, matched-filter peaks, accumulate + rank.
  - confirm_sky_area      : HITL gate — wide scans must be confirmed before fan-out (guardrail #5).
  - rank_candidates       : sort/cap candidates.

All network-bound dependencies (TAP client, result store, image service, analysis module)
are injected so the workflows are unit-testable offline with fakes. The per-tile query is
ALWAYS a q3c-bounded aggregate built by build_density_aggregate (never a BETWEEN box), and
tile/candidate budgets cap the work; anything dropped is reported (no silent truncation).
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from services import datalab_query_builders as builders
from services import datalab_sql_policy as policy

MAX_TILES = int(os.getenv("DATALAB_MAX_TILES", "64"))
DEFAULT_CANDIDATE_BUDGET = int(os.getenv("DATALAB_CANDIDATE_BUDGET", "50"))


# ── lazy default dependencies (overridable for tests) ─────────────────────────
def _default_client():
    from integrations.datalab_client import DatalabClient
    return DatalabClient()


def _default_result_store():
    from services.datalab_result_store import default_result_store
    return default_result_store()


def _default_analysis():
    from services import datalab_analysis
    return datalab_analysis


def _default_image_service():
    from services.datalab_image_service import DatalabImageService
    return DatalabImageService()


def _run_builder_sql(sql: str, meta: Mapping[str, Any], *, client, result_store):
    """Validate builder SQL through the governor, run it, and store the frame -> result_id."""
    validated = policy.validate(sql, source="builder", meta=meta)
    result = client.query(sql=validated.sql, fmt="pandas")
    store_meta = {
        **dict(meta or {}),
        "validated_sql": validated.sql,
        "provenance": {**(getattr(result, "provenance", {}) or {}), "validated_sql": validated.sql},
    }
    result_id = result_store.put(result.dataframe, store_meta)
    return result_id, result


def _cosd(dec_deg: float) -> float:
    return max(math.cos(math.radians(float(dec_deg))), 0.02)


def footprint_area_deg2(footprint: Mapping[str, float]) -> float:
    """Proper rectangular sky area in deg² (accounts for cos(dec) convergence)."""
    dra = math.radians(float(footprint["ra_max"]) - float(footprint["ra_min"]))
    dsin = math.sin(math.radians(float(footprint["dec_max"]))) - math.sin(math.radians(float(footprint["dec_min"])))
    steradians = abs(dra * dsin)
    return steradians * (180.0 / math.pi) ** 2


def tile_footprint(footprint: Mapping[str, float], tile_radius_deg: float) -> List[tuple]:
    """Cover a rectangular footprint with cone-tile centers; RA spacing widens by 1/cos(dec)."""
    ra_min, ra_max = float(footprint["ra_min"]), float(footprint["ra_max"])
    dec_min, dec_max = float(footprint["dec_min"]), float(footprint["dec_max"])
    if ra_min > ra_max or dec_min > dec_max:
        raise ValueError("footprint must satisfy ra_min<=ra_max and dec_min<=dec_max")
    step = max(1e-3, float(tile_radius_deg))
    tiles: List[tuple] = []
    dec = dec_min
    while dec <= dec_max + 1e-9:
        ra_step = step / _cosd(dec)
        ra = ra_min
        while ra <= ra_max + 1e-9:
            tiles.append((round(ra, 6), round(dec, 6)))
            ra += ra_step
        dec += step
    return tiles


def confirm_sky_area(footprint: Mapping[str, float], tile_radius_deg: float, *, max_tiles: int = MAX_TILES) -> Dict[str, Any]:
    """HITL gate: report area + tile count and whether a wide scan needs user confirmation."""
    n = len(tile_footprint(footprint, tile_radius_deg))
    area = footprint_area_deg2(footprint)
    needs = n > int(max_tiles)
    message = ""
    if needs:
        message = (
            f"This scan spans ~{area:.1f} deg² = {n} tiles of radius {tile_radius_deg}° "
            f"(cap {max_tiles}). Confirm the sky area before fanning out, or shrink the footprint / enlarge the tiles."
        )
    return {
        "tiles": n,
        "area_deg2": round(area, 3),
        "tile_radius_deg": float(tile_radius_deg),
        "max_tiles": int(max_tiles),
        "needs_confirmation": needs,
        "message": message,
    }


def rank_candidates(candidates: Sequence[Mapping[str, Any]], *, by: str = "significance", limit: int = DEFAULT_CANDIDATE_BUDGET) -> List[Dict[str, Any]]:
    ranked = sorted((dict(c) for c in candidates), key=lambda c: float(c.get(by, 0.0) or 0.0), reverse=True)
    return ranked[: int(limit)]


def _tile_peaks(frame, *, analysis, bins: int, sigma_small: float, sigma_large: float, threshold: float, max_peaks: int) -> List[Dict[str, float]]:
    """Matched-filter peaks for one tile's density frame (ra_bin, dec_bin, source_count).

    Computes the 2D histogram + DoG peaks WITHOUT rendering a figure (rendering per tile
    would be far too slow). Reuses the analysis module's peak finder, which returns [] when
    no local maximum clears the threshold.
    """
    if frame is None or len(frame) == 0:
        return []
    ra = np.asarray(frame["ra_bin"], dtype=float)
    dec = np.asarray(frame["dec_bin"], dtype=float)
    weights = np.asarray(frame["source_count"], dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(weights)
    if good.sum() < 4:
        return []
    hist, xe, ye = np.histogram2d(ra[good], dec[good], bins=int(bins), weights=weights[good])
    return analysis._matched_filter_peaks(hist, xe, ye, sigma_small, sigma_large, threshold, max_peaks)


def tiled_sky_scan(
    catalog: str,
    table: str,
    footprint: Mapping[str, float],
    *,
    tile_radius_deg: float = 2.0,
    mode: str = "grid",
    step_deg: float = 0.05,
    color_cut: Optional[Mapping[str, Any]] = None,
    value_cuts: Optional[Sequence[Mapping[str, Any]]] = None,
    morphology: Optional[Mapping[str, Any]] = None,
    bins: int = 60,
    sigma_small: float = 1.0,
    sigma_large: float = 3.0,
    peak_threshold: float = 3.0,
    max_peaks_per_tile: int = 5,
    max_tiles: int = MAX_TILES,
    candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
    confirm: bool = False,
    client: Any = None,
    result_store: Any = None,
    analysis: Any = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """P15: tile a footprint, run a server-side density aggregate per q3c cone tile, find
    matched-filter peaks, accumulate and rank. Gates wide scans behind confirm=True."""
    client = client or _default_client()
    result_store = result_store or _default_result_store()
    analysis = analysis or _default_analysis()

    if mode != "grid":
        # Matched-filter peak detection requires the grid (hist2d) representation.
        raise ValueError("tiled_sky_scan currently supports mode='grid' (matched-filter peaks)")

    tiles = tile_footprint(footprint, tile_radius_deg)
    area = footprint_area_deg2(footprint)
    total_tiles = len(tiles)
    dropped_tiles = 0
    if total_tiles > int(max_tiles):
        if not confirm:
            decision = confirm_sky_area(footprint, tile_radius_deg, max_tiles=max_tiles)
            return {"success": False, "needs_confirmation": True, **decision}
        dropped_tiles = total_tiles - int(max_tiles)
        tiles = tiles[: int(max_tiles)]

    predicates = builders.build_catalog_predicates(catalog, table, color_cut=color_cut, value_cuts=value_cuts, morphology=morphology)
    candidates: List[Dict[str, Any]] = []
    tiles_scanned = 0
    tile_errors = 0
    for idx, (ra, dec) in enumerate(tiles):
        if cancel_check and cancel_check():
            break
        try:
            sql, meta = builders.build_density_aggregate(
                catalog, table, mode="grid", step_deg=step_deg,
                ra=ra, dec=dec, radius_deg=tile_radius_deg, predicates=predicates,
            )
            _rid, result = _run_builder_sql(sql, meta, client=client, result_store=result_store)
        except Exception:  # noqa: BLE001 - one bad tile must not kill the scan
            tile_errors += 1
            continue
        tiles_scanned += 1
        for pk in _tile_peaks(
            result.dataframe, analysis=analysis, bins=bins,
            sigma_small=sigma_small, sigma_large=sigma_large,
            threshold=peak_threshold, max_peaks=max_peaks_per_tile,
        ):
            candidates.append({
                "ra": float(pk["ra"]),
                "dec": float(pk["dec"]),
                "significance": float(pk.get("significance", 0.0)),
                "tile_index": idx,
                "tile_center": [ra, dec],
            })

    ranked = rank_candidates(candidates, limit=candidate_budget)
    notes: List[str] = []
    if dropped_tiles:
        notes.append(f"Capped at {max_tiles} tiles; {dropped_tiles} of {total_tiles} tiles were not scanned.")
    if len(candidates) > len(ranked):
        notes.append(f"Returned top {len(ranked)} of {len(candidates)} candidates (candidate_budget={candidate_budget}).")
    if tile_errors:
        notes.append(f"{tile_errors} tile queries failed and were skipped.")
    return {
        "success": True,
        "catalog": catalog,
        "table": table,
        "area_deg2": round(area, 3),
        "tiles_total": total_tiles,
        "tiles_scanned": tiles_scanned,
        "dropped_tiles": dropped_tiles,
        "candidates_found": len(candidates),
        "candidate_budget": int(candidate_budget),
        "candidates": ranked,
        "notes": notes,
    }


def density_then_cutouts(
    catalog: str,
    table: str,
    ra: float,
    dec: float,
    radius_deg: float,
    *,
    step_deg: float = 0.05,
    color_cut: Optional[Mapping[str, Any]] = None,
    value_cuts: Optional[Sequence[Mapping[str, Any]]] = None,
    morphology: Optional[Mapping[str, Any]] = None,
    top_n: int = 5,
    fov_deg: float = 0.05,
    band: str = "g",
    client: Any = None,
    result_store: Any = None,
    image_service: Any = None,
) -> Dict[str, Any]:
    """P12: densest stellar cells within a cone -> SIA cutout grid of the top-N cells."""
    client = client or _default_client()
    result_store = result_store or _default_result_store()
    image_service = image_service or _default_image_service()

    top_n = max(1, int(top_n))
    predicates = builders.build_catalog_predicates(catalog, table, color_cut=color_cut, value_cuts=value_cuts, morphology=morphology)
    sql, meta = builders.build_density_aggregate(
        catalog, table, mode="grid", step_deg=step_deg,
        ra=ra, dec=dec, radius_deg=radius_deg, predicates=predicates,
        limit=max(top_n * 4, 50),
    )
    result_id, result = _run_builder_sql(sql, meta, client=client, result_store=result_store)
    df = result.dataframe
    # The aggregate is ORDER BY source_count DESC, so head(top_n) are the densest cells.
    peaks: List[Dict[str, Any]] = []
    for _, row in df.head(top_n).iterrows():
        peaks.append({
            "ra": float(row["ra_bin"]),
            "dec": float(row["dec_bin"]),
            "label": f"n={int(row['source_count'])}",
        })
    grid = image_service.cutout_grid(peaks, fov_deg, band=band, catalog=catalog) if peaks else {"success": True, "panels": []}
    return {
        "success": True,
        "result_id": result_id,
        "n_peaks": len(peaks),
        "peaks": peaks,
        "cutout_grid": grid,
    }


__all__ = [
    "MAX_TILES",
    "DEFAULT_CANDIDATE_BUDGET",
    "confirm_sky_area",
    "density_then_cutouts",
    "footprint_area_deg2",
    "rank_candidates",
    "tile_footprint",
    "tiled_sky_scan",
]
