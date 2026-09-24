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
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from services import datalab_query_builders as builders
from services import datalab_sql_policy as policy

DATALAB_TILE_QUERY_SECONDS = 40.0

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


def _run_builder_sql(
    sql: str,
    meta: Mapping[str, Any],
    *,
    client,
    result_store,
    owner_id: Optional[str] = None,
    async_fallback: bool = True,
    timeout: Optional[float] = None,
):
    """Validate builder SQL through the governor, run it, and store the frame -> result_id."""
    from services import datalab_registry as registry
    registry.ensure_tap_schema_fresh(client)
    validated = policy.validate(sql, source="builder", meta=meta)
    result = client.query(sql=validated.sql, fmt="pandas", async_fallback=async_fallback,
                          **({"timeout": timeout} if timeout is not None else {}))
    trunc_warning = policy.limit_truncation_warning(
        len(result.dataframe), (validated.meta or {}).get("row_limit")
    )
    store_meta = {
        **dict(meta or {}),
        "validated_sql": validated.sql,
        # Scope the stored result to its requesting user when known, so the
        # export route's owner guard is not vacuously skipped for results
        # minted here (dl-export-owner-gap / UIAPI-05).
        **({"owner_id": str(owner_id)} if owner_id else {}),
        "provenance": {
            **(getattr(result, "provenance", {}) or {}),
            "validated_sql": validated.sql,
            # Platform-chosen caps are stamped in provenance so they can never
            # read as a science choice (RE-B1).
            **(
                {"platform_row_cap": int(validated.meta["platform_row_cap"])}
                if isinstance(validated.meta, dict) and validated.meta.get("platform_row_cap")
                else {}
            ),
            # Named platform budgets (crossmatch small-side, derived candidate
            # budgets) disclose the same way (guard CX-01/CX-09).
            **(
                {"platform_row_caps": dict(validated.meta["platform_row_caps"])}
                if isinstance(validated.meta, dict) and validated.meta.get("platform_row_caps")
                else {}
            ),
            **(
                {"row_limit": int(validated.meta["row_limit"]), "limit_truncated": True}
                if trunc_warning
                else {}
            ),
        },
    }
    if trunc_warning:
        store_meta.setdefault("warnings", []).append(trunc_warning)
    result_id = result_store.put(result.dataframe, store_meta)
    return result_id, result


def _cosd(dec_deg: float) -> float:
    return max(math.cos(math.radians(float(dec_deg))), 0.02)


def _ra_span_deg(ra_min: float, ra_max: float) -> float:
    """RA extent in degrees; ra_min > ra_max means the box wraps through 0/360
    (ra-wrap-rect-footprint-unsupported)."""
    span = float(ra_max) - float(ra_min)
    return span if span >= 0 else span + 360.0


def footprint_area_deg2(footprint: Mapping[str, float]) -> float:
    """Proper rectangular sky area in deg² (accounts for cos(dec) convergence)."""
    dra = math.radians(_ra_span_deg(float(footprint["ra_min"]), float(footprint["ra_max"])))
    dsin = math.sin(math.radians(float(footprint["dec_max"]))) - math.sin(math.radians(float(footprint["dec_min"])))
    steradians = abs(dra * dsin)
    return steradians * (180.0 / math.pi) ** 2


def tile_footprint(footprint: Mapping[str, float], tile_radius_deg: float) -> List[tuple]:
    """Cover a rectangular footprint with cone-tile centers; RA spacing widens by 1/cos(dec).

    ra_min > ra_max is a footprint wrapping through RA=0/360 (e.g. 358°→2°):
    tiles walk the wrapped span and centers are emitted mod 360
    (ra-wrap-rect-footprint-unsupported)."""
    ra_min, ra_max = float(footprint["ra_min"]), float(footprint["ra_max"])
    dec_min, dec_max = float(footprint["dec_min"]), float(footprint["dec_max"])
    if dec_min > dec_max:
        raise ValueError("footprint must satisfy dec_min<=dec_max")
    ra_end = ra_min + _ra_span_deg(ra_min, ra_max)
    step = max(1e-3, float(tile_radius_deg))
    tiles: List[tuple] = []
    dec = dec_min
    while dec <= dec_max + 1e-9:
        ra_step = step / _cosd(dec)
        ra = ra_min
        while ra <= ra_end + 1e-9:
            tiles.append((round(ra % 360.0, 6), round(dec, 6)))
            ra += ra_step
        dec += step
    return tiles


# Density scans wider than this radius need explicit user confirmation — a 30°
# "quick look" is ~2,800 deg² of server-side aggregation (live P15 burned three
# such scans into async ERRORs with no gate).
MAX_UNCONFIRMED_CONE_RADIUS_DEG = float(os.getenv("DATALAB_MAX_UNCONFIRMED_RADIUS_DEG", "2.5"))


def confirm_cone_area(ra: float, dec: float, radius_deg: float, *, max_radius_deg: Optional[float] = None) -> Dict[str, Any]:
    """HITL gate for cone-based density tools (same shape as confirm_sky_area)."""
    limit = float(max_radius_deg if max_radius_deg is not None else MAX_UNCONFIRMED_CONE_RADIUS_DEG)
    r = float(radius_deg)
    area = math.pi * r * r
    needs = r > limit
    message = ""
    if needs:
        message = (
            f"This density scan covers a {r:g}° cone ≈ {area:.0f} deg² (unconfirmed cap: "
            f"{limit:g}° radius). If the user explicitly asked for this region, re-call with "
            "confirm=true; otherwise shrink the radius or ask the user first."
        )
    return {
        "ra": float(ra),
        "dec": float(dec),
        "radius_deg": r,
        "area_deg2": area,
        "max_radius_deg": limit,
        "needs_confirmation": needs,
        "message": message,
    }


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


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    by: str = "significance",
    limit: int = DEFAULT_CANDIDATE_BUDGET,
    min_separation_deg: float = 0.0,
) -> List[Dict[str, Any]]:
    """Rank by significance; optionally merge near-duplicates greedily.

    Overlapping tiles find the SAME sky peak twice, and a broad clump spans
    several histogram bins — without a minimum separation the 'top N
    candidates' can be N views of one object (live P15: five sub-peaks of
    Draco; live P12: 2 of 5 panels were neighbors of another peak).
    """
    ranked = sorted((dict(c) for c in candidates), key=lambda c: float(c.get(by, 0.0) or 0.0), reverse=True)
    if not min_separation_deg or min_separation_deg <= 0:
        return ranked[: int(limit)]
    kept: List[Dict[str, Any]] = []
    for cand in ranked:
        if len(kept) >= int(limit):
            break
        ra_c, dec_c = float(cand.get("ra", 0.0)), float(cand.get("dec", 0.0))
        cosd = _cosd(dec_c)
        too_close = any(
            math.hypot((ra_c - float(k["ra"])) * cosd, dec_c - float(k["dec"])) < float(min_separation_deg)
            for k in kept
        )
        if not too_close:
            kept.append(cand)
    return kept


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
    max_seconds: Optional[float] = None,
    client: Any = None,
    result_store: Any = None,
    analysis: Any = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    owner_id: Optional[str] = None,
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

    from services import datalab_registry as registry
    merged_cuts, quality_note = registry.merge_default_quality_cuts(catalog, table, list(value_cuts or []))
    morph_warning = builders.morphology_deviation_warning(catalog, table, morphology)
    predicates = builders.build_catalog_predicates(catalog, table, color_cut=color_cut, value_cuts=merged_cuts, morphology=morphology)
    # Wall-clock budget + no per-tile async fallback: with the fallback ON, a
    # crowded field turned a 64-tile scan into an hours-long retry storm — each
    # slow tile burned the sync window AND (with a real token) spawned+polled an
    # orphan server-side job (dl-tiled-scan-no-budget-async-fallback-storm).
    if max_seconds is None:
        max_seconds = float(os.getenv("DATALAB_TILED_SCAN_MAX_SECONDS", "210"))
    started = time.monotonic()
    budget_stop = False
    candidates: List[Dict[str, Any]] = []
    tiles_scanned = 0
    tile_errors = 0
    # Per-tile cap/truncation tracking (guard CX-08): a tile whose ORDER BY
    # source_count DESC query filled its row cap silently dropped its SPARSEST
    # density cells BEFORE peak-finding — faint peaks there can be missing and
    # affected candidates' significances are lower bounds. Surface it.
    tiles_truncated = 0
    truncated_tile_indices: set = set()
    truncated_row_limit: Optional[int] = None
    for idx, (ra, dec) in enumerate(tiles):
        if cancel_check and cancel_check():
            break
        if time.monotonic() - started > max_seconds:
            budget_stop = True
            break
        try:
            sql, meta = builders.build_density_aggregate(
                catalog, table, mode="grid", step_deg=step_deg,
                ra=ra, dec=dec, radius_deg=tile_radius_deg, predicates=predicates,
            )
            _rid, result = _run_builder_sql(
                sql, meta, client=client, result_store=result_store,
                owner_id=owner_id, async_fallback=False,
            )
        except Exception:  # noqa: BLE001 - one bad tile must not kill the scan
            tile_errors += 1
            continue
        tiles_scanned += 1
        _tile_row_limit = (meta or {}).get("row_limit")
        if policy.limit_truncation_warning(len(result.dataframe), _tile_row_limit):
            tiles_truncated += 1
            truncated_tile_indices.add(idx)
            truncated_row_limit = int(_tile_row_limit)
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

    # Merge near-duplicates (overlapping tiles + multi-bin clumps) before the
    # budget cut, then flag candidates that sit inside a KNOWN MW satellite or
    # globular cluster (live P15 "discovered" Draco as candidate #1).
    merge_radius = max(0.1, 3.0 * float(step_deg))
    ranked = rank_candidates(candidates, limit=candidate_budget, min_separation_deg=merge_radius)
    known_hits = 0
    for cand in ranked:
        known = registry.match_known_mw_object(cand["ra"], cand["dec"])
        if known:
            cand["known_object"] = f"{known['name']} ({known['kind']}, {known['separation_deg']}° away)"
            known_hits += 1
        # Per-candidate cap disclosure (guard CX-08): the candidate's OWN tile
        # dropped density cells at the row cap, so its significance is a lower
        # bound and neighbouring faint peaks may be missing.
        if cand.get("tile_index") in truncated_tile_indices:
            cand["warnings"] = list(cand.get("warnings") or []) + [
                f"This candidate's tile hit the per-tile row cap (LIMIT {truncated_row_limit}); "
                "the tile's sparsest density cells were dropped BEFORE peak-finding, so the "
                "significance is a lower bound and nearby faint peaks may be missing."
            ]
    notes: List[str] = []
    if tiles_truncated:
        notes.append(
            f"{tiles_truncated} tile quer{'y' if tiles_truncated == 1 else 'ies'} hit the "
            f"per-tile row cap (LIMIT {truncated_row_limit}): the SPARSEST density cells in "
            "those tiles were dropped BEFORE peak-finding, so faint peaks there may be missing "
            "and candidates from those tiles (see their warnings) have lower-bound "
            "significances. Re-run with a coarser step_deg or smaller tiles for a complete scan."
        )
    if quality_note:
        notes.append(quality_note)
    if morph_warning:
        notes.append(morph_warning)
    if dropped_tiles:
        notes.append(f"Capped at {max_tiles} tiles; {dropped_tiles} of {total_tiles} tiles were not scanned.")
    if len(candidates) > len(ranked):
        notes.append(
            f"Returned {len(ranked)} of {len(candidates)} peaks after merging near-duplicates "
            f"(min separation {merge_radius:g}°) and applying candidate_budget={candidate_budget}."
        )
    if known_hits:
        notes.append(
            f"{known_hits} candidate(s) coincide with KNOWN MW satellites/globulars (see "
            "known_object) — they are re-detections, NOT new discoveries; say so in the answer."
        )
    if tile_errors:
        notes.append(f"{tile_errors} tile queries failed and were skipped.")
    if budget_stop:
        notes.append(
            f"Stopped at the {max_seconds:.0f}s scan budget: {tiles_scanned} of "
            f"{len(tiles)} tiles were scanned — the remaining footprint is UNSCANNED, "
            "not empty. Narrow the footprint or enlarge the tiles and re-run."
        )
    return {
        "success": True,
        "catalog": catalog,
        "table": table,
        "area_deg2": round(area, 3),
        "tiles_total": total_tiles,
        "tiles_scanned": tiles_scanned,
        "tiles_truncated": tiles_truncated,
        "dropped_tiles": dropped_tiles,
        "budget_stop": budget_stop,
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
    owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """P12: densest stellar cells within a cone -> SIA cutout grid of the top-N cells."""
    client = client or _default_client()
    result_store = result_store or _default_result_store()
    image_service = image_service or _default_image_service()

    top_n = max(1, int(top_n))
    from services import datalab_registry as registry
    merged_cuts, quality_note = registry.merge_default_quality_cuts(catalog, table, list(value_cuts or []))
    morph_warning = builders.morphology_deviation_warning(catalog, table, morphology)
    predicates = builders.build_catalog_predicates(catalog, table, color_cut=color_cut, value_cuts=merged_cuts, morphology=morphology)
    # The candidate budget is DERIVED by the platform (top_n*4, floor 50) —
    # never a user science cut, so it gets the full platform-cap treatment:
    # self-describing SQL comment + platform_row_cap(+named) metadata + a
    # result note naming it (guard CX-09 / RE-B1).
    # Clamped to the builder's hard maximum so the DISCLOSED number always
    # equals the EMITTED LIMIT — for huge top_n the builder clamps the SQL to
    # MAX_ROW_LIMIT, and an unclamped value here overstated the budget in
    # platform_row_cap and the note (guard CX-09 verify regression).
    candidate_budget = min(max(top_n * 4, 50), builders.MAX_ROW_LIMIT)
    sql, meta = builders.build_density_aggregate(
        catalog, table, mode="grid", step_deg=step_deg,
        ra=ra, dec=dec, radius_deg=radius_deg, predicates=predicates,
        limit=candidate_budget,
    )
    if sql.rstrip().endswith(f"LIMIT {candidate_budget}"):
        sql = f"{sql.rstrip()} {builders.PLATFORM_ROW_CAP_COMMENT}"
    builders._flag_platform_cap(
        meta, candidate_budget,
        f"derived candidate budget max(top_n*4, 50) = {candidate_budget}, chosen by the "
        "platform to bound the density-peak pool",
    )
    meta.setdefault("platform_row_caps", {})["candidate_budget"] = int(candidate_budget)
    if float(radius_deg) >= 5.0:
        tiled = tiled_density_aggregate(
            catalog, table, mode="grid", step_deg=step_deg,
            ra=ra, dec=dec, radius_deg=radius_deg, predicates=predicates,
            limit=candidate_budget, client=client, result_store=result_store,
            owner_id=owner_id,
        )
        if not tiled.get("result_id"):
            return tiled
        result_id = tiled["result_id"]
        df, _meta, _status = result_store.lookup(result_id)
    else:
        result_id, result = _run_builder_sql(
            sql, meta, client=client, result_store=result_store, owner_id=owner_id
        )
        df = result.dataframe
    # The aggregate is ORDER BY source_count DESC. Greedily skip cells adjacent
    # to an already-accepted peak (2 of live P12's 5 "candidates" were neighbor
    # cells of the same clump), and flag peaks inside known MW objects.
    min_sep = 2.0 * float(step_deg)
    peaks: List[Dict[str, Any]] = []
    notes: List[str] = list(tiled.get("warnings", [])) if float(radius_deg) >= 5.0 else []
    if quality_note:
        notes.append(quality_note)
    if morph_warning:
        notes.append(morph_warning)
    notes.append(
        f"Candidate budget: the density scan was capped at LIMIT {candidate_budget} cells — "
        f"a platform-derived budget (min(max(top_n*4, 50), {builders.MAX_ROW_LIMIT})), not a "
        "science cut and not user-requested; raise top_n to widen the peak pool (guard CX-09)."
    )
    for _, row in df.iterrows():
        if len(peaks) >= top_n:
            break
        ra_p, dec_p = float(row["ra_bin"]), float(row["dec_bin"])
        cosd = _cosd(dec_p)
        if any(math.hypot((ra_p - p["ra"]) * cosd, dec_p - p["dec"]) < min_sep for p in peaks):
            continue
        peak: Dict[str, Any] = {
            "ra": ra_p,
            "dec": dec_p,
            "label": f"n={int(row['source_count'])}",
        }
        known = registry.match_known_mw_object(ra_p, dec_p)
        if known:
            peak["known_object"] = f"{known['name']} ({known['kind']}, {known['separation_deg']}° away)"
            peak["label"] += f" KNOWN: {known['name']}"
        peaks.append(peak)
    if any("known_object" in p for p in peaks):
        notes.append(
            "Some peaks coincide with KNOWN MW satellites/globulars (see known_object) — "
            "they are re-detections, not new candidates."
        )
    grid = image_service.cutout_grid(peaks, fov_deg, band=band, catalog=catalog) if peaks else {"success": True, "panels": []}
    # Nested-grid timeout semantics (guard CX-11, partial-contest): density
    # peaks ARE real progress — analysis can proceed and an SSE deadline
    # extension is then correct — so peaks keep success=True even when the
    # nested grid fully timed out. But the timeout must be LOUD (warning) and
    # machine-readable (cutout_grid_timeout), and with NO peaks to show a
    # fully-timed-out grid is a timeout, not a completion.
    grid_timeout = bool(isinstance(grid, dict) and grid.get("timeout"))
    if grid_timeout:
        notes.append(
            "CUTOUT GRID TIMEOUT: the nested cutout grid timed out downloading tiles — the "
            "density peaks above are REAL results (their ranking and coordinates stand), but "
            "NO grid image was rendered. Do not describe cutouts; offer a retry with fewer "
            "peaks or a smaller fov_deg."
        )
    return {
        "success": bool(peaks) or not grid_timeout,
        "result_id": result_id,
        **({k: tiled[k] for k in ("partial", "tiles_completed", "tiles_total", "coverage_summary") if k in tiled} if float(radius_deg) >= 5.0 else {}),
        "n_peaks": len(peaks),
        "peaks": peaks,
        "cutout_grid": grid,
        **({"cutout_grid_timeout": True} if grid_timeout else {}),
        **({"timeout": True} if grid_timeout and not peaks else {}),
        **({"notes": notes} if notes else {}),
    }


def _tile_cone_centers(radius_deg: float, tile_radius_deg: float, cell_margin_deg: float = 0.3) -> List[tuple]:
    """Tangent-plane (dx, dy) offsets in degrees for cone tiles that fully cover a
    parent cone of radius_deg. Square grid with spacing s = sqrt(2)*(tile_r - margin):
    the grid covering radius (s*sqrt(2)/2 = tile_r - margin) leaves `cell_margin_deg`
    inside every tile, so any density cell (grid bin / coarse HEALPix pixel) smaller
    than the margin is fully contained in at least one tile — required for the
    max-count merge to be exact rather than undercounting along tile seams."""
    r = float(tile_radius_deg)
    margin = min(float(cell_margin_deg), 0.5 * r)
    s = (r - margin) * math.sqrt(2.0)
    keep = float(radius_deg) + s * 0.7072
    n = int(math.ceil(keep / s))
    centers = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            dx, dy = i * s, j * s
            if math.hypot(dx, dy) <= keep:
                centers.append((dx, dy))
    # Scan the dense middle first so a time-budget stop drops rim tiles, not the core.
    centers.sort(key=lambda c: math.hypot(*c))
    return centers


ASYNC_MIN_POLL_SECONDS = 15.0


def run_async_sql(
    validated_sql: str,
    *,
    client: Any,
    max_seconds: float = 240.0,
    poll_seconds: float = 5.0,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Submit ONE already-validated query as a Data Lab async job and poll it
    until it completes, errors, the budget runs out (bounded by the tool
    deadline) or the turn is cancelled (the job is then aborted). Returns
    ``{"state": COMPLETED|ERROR|ABORTED|RUNNING|CANCELLED|UNAVAILABLE, "jobid",
    "elapsed_s", "result"?, "error"?}``. Needs a login token."""
    from integrations.datalab_client import ANON_TOKEN
    from services.tool_budgets import is_cancelled, remaining_seconds

    if getattr(client, "token", ANON_TOKEN) == ANON_TOKEN:
        return {"state": "UNAVAILABLE", "error": "async jobs need a Data Lab login token"}
    # The async parser gets a rewritten statement (MATERIALIZED / ::casts
    # stripped): that EXECUTED text is what provenance must show (CX-16).
    executed_sql = validated_sql
    _strip = getattr(client, "strip_materialized_for_async", None)
    if callable(_strip):
        try:
            executed_sql = str(_strip(validated_sql))
        except Exception:
            executed_sql = validated_sql
    left = remaining_seconds()
    budget = min(float(max_seconds), max(0.0, left - 10.0) if left is not None else float(max_seconds))
    # Never submit a server job that cannot be polled or was already cancelled (CX-12).
    if is_cancelled():
        return {"state": "CANCELLED", "executed_sql": executed_sql, "error": "turn cancelled before submission"}
    # A zero/negative budget can never poll: refuse it outright (CX-12 verify:
    # max_seconds=0 submitted a job and returned RUNNING without one poll).
    if float(max_seconds) <= 0 or budget <= 0 or budget < min(ASYNC_MIN_POLL_SECONDS, float(max_seconds)):
        return {"state": "SKIPPED", "executed_sql": executed_sql,
                "error": f"only {budget:.0f} s of tool budget left: not enough to submit and poll an async job"}
    started = time.monotonic()
    jobid = str(client.submit(sql=validated_sql))
    state, status_errors = "", 0
    # Every distinct status seen while polling, with its time since submission.
    history: List[Dict[str, Any]] = [{"t_s": 0.0, "state": "SUBMITTED"}]

    def _emit(event: Dict[str, Any]) -> None:
        # Callers watching from another thread learn the job id and each
        # status as they happen, even if this poller is later abandoned.
        if on_event is not None:
            try:
                on_event(dict(event, jobid=jobid))
            except Exception:  # noqa: BLE001 - reporting must never break polling
                pass

    _emit(history[0])
    while time.monotonic() - started < budget:
        if is_cancelled():
            try:
                client.abort(jobid)
            except Exception:
                pass
            return {"state": "CANCELLED", "jobid": jobid, "status_history": history, "elapsed_s": round(time.monotonic() - started, 1),
                    "executed_sql": executed_sql, "error": "turn cancelled; async job aborted"}
        try:
            state = str(client.status(jobid) or "").upper()
            status_errors = 0
            if state and state != history[-1]["state"]:
                history.append({"t_s": round(time.monotonic() - started, 1), "state": state})
                _emit(history[-1])
        except Exception as exc:
            # A transport error must not lose the submitted job id (CX-15).
            status_errors += 1
            state = ""
            if status_errors >= 3:
                return {"state": "UNKNOWN", "jobid": jobid, "status_history": history, "elapsed_s": round(time.monotonic() - started, 1),
                        "executed_sql": executed_sql,
                        "error": f"async job {jobid}: status polling failed 3 times ({type(exc).__name__})",
                        "note": f"The job may still be running server-side; fetch it with datalab_job_results jobid={jobid}."}
        if state in ("COMPLETED", "ERROR", "ABORTED"):
            break
        time.sleep(max(0.5, min(float(poll_seconds), budget - (time.monotonic() - started))))
    elapsed = round(time.monotonic() - started, 1)
    if state == "COMPLETED":
        try:
            result = client.results(jobid, query_text=validated_sql)
        except Exception as exc:
            return {"state": "RESULT_ERROR", "jobid": jobid, "status_history": history, "elapsed_s": elapsed, "executed_sql": executed_sql,
                    "error": f"async job {jobid} completed but fetching its results failed: {type(exc).__name__}: {str(exc)[:160]}",
                    "note": f"Fetch it with datalab_job_results jobid={jobid}."}
        return {"state": "COMPLETED", "jobid": jobid, "status_history": history, "elapsed_s": elapsed, "executed_sql": executed_sql, "result": result}
    if state == "ERROR":
        try:
            detail = str(client.error(jobid))[:200]
        except Exception:
            detail = ""
        return {"state": "ERROR", "jobid": jobid, "status_history": history, "elapsed_s": elapsed, "executed_sql": executed_sql,
                "error": f"async job {jobid} ERROR: {detail}"}
    return {"state": state or "RUNNING", "jobid": jobid, "status_history": history, "elapsed_s": elapsed, "executed_sql": executed_sql,
            "error": f"async job {jobid} {state or 'still running'} after {elapsed:.0f}s",
            "note": f"The job keeps running server-side; fetch it with datalab_job_results jobid={jobid}."}


def abort_async_job(client: Any, jobid: Optional[str]) -> str:
    """Best-effort abort of a server-side job before a synchronous fallback
    re-runs the same scan (guard CX-09: an UNKNOWN job may still be running)."""
    if not jobid:
        return "no job"
    try:
        client.abort(str(jobid))
        return "abort requested"
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return f"abort failed ({type(exc).__name__})"


def async_density_aggregate(
    catalog: str,
    table: str,
    *,
    mode: str = "healpix",
    step_deg: float = 0.1,
    healpix_column: Optional[str] = None,
    ra: float,
    dec: float,
    radius_deg: float,
    predicates: Optional[Sequence[str]] = None,
    max_cells: int = 20000,
    max_seconds: float = 240.0,
    poll_seconds: float = 5.0,
    client: Any = None,
    result_store: Any = None,
    owner_id: Optional[str] = None,
) -> Dict[str, Any]:
    """ONE density aggregate over the whole cone as a Data Lab ASYNC job
    (needs a login token; anonymous tokens cannot poll jobs). Dense wide
    regions never finish in the 60 s sync window whatever the tiling -- live
    2026-09-23: even a 0.5 deg HEALPix tile at the LMC centre timed out, while
    the whole 20x20 deg LMC aggregate completed as one async job in 199 s.
    Polls until done or ``max_seconds`` (bounded by the tool deadline and
    stopped by turn cancellation). Same return shape as
    :func:`tiled_density_aggregate`; a job still running at the budget is
    reported with its ``jobid`` (fetch later with datalab_job_results)."""
    from integrations.datalab_client import ANON_TOKEN
    from services import datalab_registry as reg

    client = client or _default_client()
    result_store = result_store or _default_result_store()
    if getattr(client, "token", ANON_TOKEN) == ANON_TOKEN:
        return {"success": False, "async_unavailable": True, "error": "async jobs need a Data Lab login token"}
    mode_key = str(mode or "healpix").strip().lower()
    info = reg.describe_table(catalog, table)
    sql, meta = builders.build_density_aggregate(
        catalog, table, mode=mode_key, step_deg=step_deg, healpix_column=healpix_column,
        ra=ra, dec=dec, radius_deg=radius_deg, predicates=list(predicates or []), limit=max_cells, max_cells=max_cells,
    )
    validated = policy.validate(sql, source="builder", meta=meta)
    job = run_async_sql(validated.sql, client=client, max_seconds=max_seconds, poll_seconds=poll_seconds)
    jobid, elapsed, state = job.get("jobid"), float(job.get("elapsed_s") or 0.0), job["state"]
    executed_sql = job.get("executed_sql") or validated.sql
    if state != "COMPLETED":
        return {"success": False, "jobid": jobid, "job_state": state, "elapsed_s": elapsed,
                "budget_exhausted": state in ("RUNNING", "SKIPPED", "EXECUTING", "QUEUED", "PENDING"),
                "error": ("turn cancelled; async job aborted" if state == "CANCELLED" else job.get("error")),
                "note": job.get("note", "")}
    result = job["result"]
    frame = result.dataframe
    row_limit = int((validated.meta or {}).get("row_limit") or max_cells)
    truncated = frame is not None and len(frame) >= row_limit
    hp_meta = None
    if mode_key == "healpix":
        want = str(healpix_column or "").strip().lower()
        hp_entry = next((h for h in (info.get("healpix_columns") or []) if h.get("name") == want), None)
        if hp_entry:
            hp_meta = {"column": want, "nside": hp_entry.get("nside"), "scheme": hp_entry.get("scheme")}
    warnings = [f"Whole region aggregated server-side as one Data Lab async job ({jobid}, {elapsed:.0f} s)."]
    if truncated:
        warnings.append(f"The aggregate filled its {row_limit}-cell cap: the SPARSEST cells were dropped (ORDER BY count DESC).")
    store_meta = {
        "builder": "density_aggregate_async", "catalog": info["catalog"], "table": info["table"],
        "tool_name": "datalab_density_aggregate", "validated_sql": executed_sql, "warnings": warnings,
        **({"owner_id": str(owner_id)} if owner_id else {}),
        "provenance": {
            "catalog": info["catalog"], "table": info["table"], "mode": f"{mode_key}_async", "jobid": jobid,
            "validated_sql": executed_sql, "parent_cone": {"ra": float(ra), "dec": float(dec), "radius_deg": float(radius_deg)},
            "partial": bool(truncated), "elapsed_s": round(elapsed, 1),
            **({"limit_truncated": True, "row_limit": row_limit} if truncated else {}),
            **({"healpix": hp_meta} if hp_meta else {}),
        },
    }
    result_id = result_store.put(frame, store_meta)
    return {"success": True, "result_id": result_id, "rowcount": int(len(frame)), "partial": bool(truncated),
            "tiles_completed": 1, "tiles_total": 1, "coverage_summary": "whole region in one async job",
            "warnings": warnings, "jobid": jobid, "elapsed_s": round(elapsed, 1), "sql_example": executed_sql}


def tiled_density_aggregate(
    catalog: str,
    table: str,
    *,
    mode: str = "grid",
    step_deg: float = 0.1,
    healpix_column: Optional[str] = None,
    ra: float,
    dec: float,
    radius_deg: float,
    predicates: Optional[Sequence[str]] = None,
    # None = the builder's per-tile cell cap (5000) applies and is FLAGGED as a
    # platform cap (SQL comment + platform_row_cap provenance + warning) — an
    # explicit default of 5000 here bypassed the flag entirely (guard CX-07).
    limit: Optional[int] = None,
    # Warnings computed BEFORE the tiled fallback (quality-cut note, morphology
    # deviation) must ride into the PERSISTED result too, not only the returned
    # dict — chained consumers read the stored result_id (guard CX-06 verify).
    extra_warnings: Optional[Sequence[str]] = None,
    tile_radius_deg: Optional[float] = None,
    max_seconds: Optional[float] = None,
    client: Any = None,
    result_store: Any = None,
    owner_id: Optional[str] = None,
    max_cells: Optional[int] = None,
) -> Dict[str, Any]:
    """P8 fallback: a wide density aggregate that beats the Data Lab 60s sync window
    by tiling the parent cone into overlapping sub-cones, each ALSO bounded by the
    parent cone (so semantics match the single-shot query exactly), then merging the
    per-tile cells (dedup by cell key, keeping the max count — partial rim counts
    from overlapping tiles are subsets of the full cell). Live-proven pattern:
    deepseek manually tiled 6 cones over a 20°x20° NSC field when the single 10°
    aggregate timed out (anon tokens cannot use the async-job path).
    """
    import pandas as pd

    from services import datalab_registry as reg

    client = client or _default_client()
    result_store = result_store or _default_result_store()
    mode_key = str(mode or "grid").strip().lower()
    if mode_key not in {"grid", "healpix"}:
        raise ValueError("tiled density aggregate mode must be grid or healpix")

    info = reg.describe_table(catalog, table)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    parent_bound = (
        f"q3c_radial_query({ra_col}, {dec_col}, {float(ra):.8g}, {float(dec):.8g}, {float(radius_deg):.8g})"
    )
    base_predicates = [str(p) for p in (predicates or [])] + [parent_bound]

    from services.tool_budgets import bounded_timeout, remaining_seconds, call_bounded, BudgetExhausted

    # 2026-09-23 L08: 16 SERIAL tiles covered at most ~23 % of a 10-degree
    # region (69 tiles of 2 deg) whatever the budget. Tiles now run
    # ``DATALAB_TILED_AGG_CONCURRENCY`` at a time (default 4, each under a child
    # deadline) and every tile of the footprint is attempted while the budget
    # lasts (cap ``DATALAB_TILED_AGG_MAX_TILES``, default 128).
    max_tiles = max(1, int(os.getenv("DATALAB_TILED_AGG_MAX_TILES", "128")))
    concurrency = max(1, int(os.getenv("DATALAB_TILED_AGG_CONCURRENCY", "4")))
    # Leave time to store and render partial results before the existing guard.
    requested_budget = float(os.getenv("DATALAB_TILED_AGG_MAX_SECONDS", "120")) if max_seconds is None else float(max_seconds)
    remaining = remaining_seconds()
    max_seconds = min(requested_budget, 240.0, max(0.0, remaining - 8.0) if remaining is not None else 240.0)
    cell_margin = max(0.3, 2.0 * float(step_deg)) if mode_key == "grid" else 0.3
    tile_r = min(float(tile_radius_deg or os.getenv("DATALAB_TILE_RADIUS_DEG", "2")), 2.0)
    tile_r = min(tile_r, max(0.5, float(radius_deg) / 1.5))
    # Never enlarge expensive tiles to conceal the cap: report the unvisited
    # footprint explicitly and retain the partial map from completed queries.
    all_centers = _tile_cone_centers(radius_deg, tile_r, cell_margin)
    tiles_total = len(all_centers)
    centers = all_centers[:max_tiles]

    # Per-tile LIMIT truncation: each tile is ORDER BY source_count DESC, so a
    # tile that filled its row cap silently dropped its SPARSEST cells and the
    # merged map read them as "no data" with no stamp — the repo's row-capped-
    # as-complete bug pattern (dl-tiled-agg-silent-cell-drop /
    # tiled-agg-silent-cell-truncation). Track it and stamp the provenance.
    tiles_truncated = 0
    effective_limit: Optional[int] = None
    # Resolve the per-tile cap ONCE, exactly as the builder will: when the
    # platform chose it (limit=None default or clamped), the final provenance
    # must record platform_row_cap ALWAYS — not only when a tile happens to
    # truncate (guard CX-07). User-passed in-range limits stay unflagged.
    # ``max_cells`` raises the per-tile cell cap for fine grids: a 2-degree
    # tile at 0.05-degree cells is ~5000 cells, so 9 of 21 L15 tiles filled the
    # 5000 cap and dropped their sparsest cells (live 2026-09-24).
    cell_cap = int(max_cells) if max_cells else builders.MAX_ROW_LIMIT
    per_tile_limit, tile_cap_reason = builders._resolve_limit(
        limit, maximum=cell_cap, default=cell_cap
    )

    import threading as _threading

    _count_lock = _threading.Lock()

    def _run_tile(dx: float, dy: float, r: float) -> Optional[Any]:
        nonlocal tiles_truncated, effective_limit
        tdec = max(-89.5, min(89.5, float(dec) + dy))
        tra = (float(ra) + dx / _cosd(tdec)) % 360.0
        sql, meta = builders.build_density_aggregate(
            catalog, table, mode=mode_key, step_deg=step_deg, healpix_column=healpix_column,
            ra=tra, dec=tdec, radius_deg=r, predicates=base_predicates, limit=limit,
            **({"max_cells": cell_cap} if max_cells else {}),
        )
        validated = policy.validate(sql, source="builder", meta=meta)
        left = max_seconds - (time.monotonic() - started)
        if left < 1.0:
            raise BudgetExhausted("Data Lab density tiling", left, 1.0)
        seconds = bounded_timeout(min(DATALAB_TILE_QUERY_SECONDS, left, float(getattr(client, "timeout", DATALAB_TILE_QUERY_SECONDS))),
                                  label="Data Lab density tile")
        result = call_bounded(
            lambda: client.query(sql=validated.sql, fmt="pandas", async_fallback=False, timeout=seconds),
            seconds, label="Data Lab density tile",
        )
        row_limit = (validated.meta or {}).get("row_limit")
        if policy.limit_truncation_warning(len(result.dataframe), row_limit):
            with _count_lock:
                tiles_truncated += 1
                effective_limit = int(row_limit)
        return result

    frames: List[Any] = []
    tiles_run = 0
    tile_errors = 0
    subdivided = 0
    started = time.monotonic()
    budget_stop = False
    errors: List[str] = []
    from services.alma_server_side import run_concurrently
    from services.host_breaker import HostCircuitOpen

    breaker_open = False
    for start in range(0, len(centers), concurrency):
        left = max_seconds - (time.monotonic() - started)
        if left < 2.0 or breaker_open:
            budget_stop = budget_stop or left < 2.0
            break
        batch = centers[start:start + concurrency]
        outs = run_concurrently([(lambda c=c: _run_tile(c[0], c[1], tile_r)) for c in batch], wall_seconds=left)
        for out in outs:
            if isinstance(out, BudgetExhausted):
                budget_stop = True
            elif isinstance(out, BaseException):  # A failed tile must not discard completed tiles.
                tile_errors += 1
                errors.append(str(out)[:300])
                if isinstance(out, HostCircuitOpen):
                    breaker_open = True
            else:
                frames.append(out.dataframe)
                tiles_run += 1

    partial = tiles_run < tiles_total or bool(tiles_truncated)
    progress = f"{tiles_run} of {tiles_total} tiles completed"
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return {
            "success": tiles_run > 0, "rowcount": 0, "partial": partial,
            "tiles_completed": tiles_run, "tiles_total": tiles_total,
            "tiles_failed": tile_errors, "coverage_summary": progress,
            "budget_exhausted": budget_stop, "tile_errors": errors,
            "error": None if tiles_run else "No Data Lab density tiles completed; the bounded queries failed or exhausted their budget.",
            "note": "No nonempty density cells were returned. Unvisited tiles are unknown, not zero density.",
        }
    merged = pd.concat(frames, ignore_index=True)
    key_cols = ["healpix"] if mode_key == "healpix" else ["ra_bin", "dec_bin"]
    for col in key_cols:
        if col not in merged.columns:
            raise RuntimeError(f"Tiled aggregate missing expected column {col!r}")
    if mode_key == "grid":
        # Guard against float-repr jitter in the SQL-computed bin keys.
        merged[key_cols] = merged[key_cols].round(9)
    merged = (
        merged.groupby(key_cols, as_index=False)["source_count"].max()
        .sort_values("source_count", ascending=False)
        .reset_index(drop=True)
    )

    warnings: List[str] = [str(w) for w in (extra_warnings or []) if w]
    if tile_cap_reason:
        warnings.append(
            f"Per-tile row cap LIMIT {per_tile_limit} applied by the platform ({tile_cap_reason}) — "
            "cost governance, not a science cut and not user-requested. Disclose the cap if any "
            "tile truncates."
        )
    if partial:
        warnings.append(f"Partial map: {progress}; unvisited pixels are unknown, not zero density. Overlapping cell counts are lower bounds where no complete tile covers the cell.")
    if budget_stop:
        warnings.append(
            f"Stopped at the {max_seconds:.0f}s tiling budget: {tiles_run} tile queries ran; "
            "outer tiles were dropped, so the map underrepresents the region edges."
        )
    if tile_errors:
        warnings.append(f"{tile_errors} tile queries failed and were skipped (partial coverage).")
    if tiles_truncated:
        # dl-tiled-agg-silent-cell-drop: a truncated map must never masquerade
        # as complete — this warning + the limit_truncated provenance stamp
        # below make datalab_sky_density_map render the TRUNCATED caption.
        warnings.append(
            f"{tiles_truncated} tile quer{'y' if tiles_truncated == 1 else 'ies'} hit the "
            f"per-tile row cap (LIMIT {effective_limit}): the SPARSEST cells in those tiles "
            "were dropped, so low-density regions of the merged map are INCOMPLETE — do not "
            "read empty cells there as 'no data'. Re-run with a coarser step_deg (fewer "
            "cells per tile) or a higher limit."
        )

    hp_meta = None
    if mode_key == "healpix":
        hp_cols = info.get("healpix_columns") or []
        want = str(healpix_column or (hp_cols[0].get("name") if hp_cols else "")).strip().lower()
        hp_entry = next((h for h in hp_cols if h.get("name") == want), None)
        if hp_entry:
            hp_meta = {"column": want, "nside": hp_entry.get("nside"), "scheme": hp_entry.get("scheme")}
    store_meta = {
        "builder": "density_aggregate_tiled",
        "catalog": info["catalog"],
        "table": info["table"],
        "tool_name": "datalab_density_aggregate",
        "warnings": warnings,
        # dl-export-owner-gap: scope the stored result to its requester.
        **({"owner_id": str(owner_id)} if owner_id else {}),
        "provenance": {
            "catalog": info["catalog"],
            "table": info["table"],
            "mode": f"{mode_key}_tiled",
            "parent_cone": {"ra": float(ra), "dec": float(dec), "radius_deg": float(radius_deg)},
            "tile_radius_deg": tile_r,
            "tiles_run": tiles_run,
            "tiles_completed": tiles_run, "tiles_total": tiles_total,
            "partial": partial, "coverage_summary": progress,
            "tiles_failed": tile_errors,
            "tiles_subdivided": subdivided,
            "sync_timeout_fallback": True,
            # Platform-chosen per-tile cap is stamped ALWAYS, not only when a
            # tile happened to truncate (guard CX-07 / RE-B1).
            **({"platform_row_cap": per_tile_limit} if tile_cap_reason else {}),
            # dl-tiled-agg-silent-cell-drop: propagate the row-cap stamp so the
            # plot layer's _truncation_warnings marks the rendered map.
            **(
                {
                    "limit_truncated": True,
                    "row_limit": effective_limit,
                    "tiles_truncated": tiles_truncated,
                }
                if tiles_truncated
                else {}
            ),
            **({"healpix": hp_meta} if hp_meta else {}),
        },
    }
    result_id = result_store.put(merged, store_meta)
    preview = merged.head(10).to_dict("records")
    return {
        "success": True,
        "tool_name": "datalab_density_aggregate",
        "result_id": result_id,
        "rowcount": int(len(merged)),
        "columns": list(merged.columns),
        "catalog": info["catalog"],
        "table": info["table"],
        "tiled_fallback": True,
        "tiles_run": tiles_run,
        "tiles_completed": tiles_run, "tiles_total": tiles_total,
        "partial": partial, "coverage_summary": progress,
        "budget_exhausted": budget_stop, "tile_errors": errors,
        "tiles_failed": tile_errors,
        "tile_radius_deg": round(tile_r, 3),
        "warnings": warnings,
        "query_summary": (
            f"tiled density aggregate ({mode_key}): {tiles_run} cone tiles of {tile_r:.1f}° "
            f"covering ra={float(ra):.4g}, dec={float(dec):.4g}, radius={float(radius_deg):.4g}°"
        ),
        "preview": preview,
        "note": (
            "The wide aggregate was split into bounded sub-cones and merged; overlapping cell "
            "counts are lower bounds when a full cell was not covered. Every tile is also bounded "
            "by the parent cone. Render THIS result_id with datalab_sky_density_map NOW, before "
            "attempting any wider region — turns have a hard time budget and a rendered map beats "
            "an unrendered bigger one."
        ),
    }


def _wedge_bounds(ra_min, ra_max, dec_min, dec_max, z_min, z_max):
    bounds = [float(v) for v in (ra_min, ra_max, dec_min, dec_max, z_min, z_max)]
    if not all(math.isfinite(v) for v in bounds):
        raise ValueError("Wedge bounds must be finite")
    r0, r1, d0, d1, z0, z1 = bounds
    if not (0 <= r0 < r1 <= 360 and -90 <= d0 < d1 <= 90 and 0 <= z0 < z1 <= 10):
        raise ValueError("Invalid RA, declination or redshift bounds")
    if d1 - d0 > 5.0:
        raise ValueError("A wedge requires a thin declination slice (at most 5 degrees); use the default 2.5 degree strip or provide narrower bounds")
    return (
        "FROM sdss_dr17.specobj WHERE class = 'GALAXY' AND zwarning = 0 "
        f"AND z BETWEEN {z0:g} AND {z1:g} "
        f"AND ra BETWEEN {r0:g} AND {r1:g} AND dec BETWEEN {d0:g} AND {d1:g} "
    )


# Default slice: the SDSS equatorial stripe (Dec -1.25..+1.25, 2.5 deg thick)
# over RA 150-220 at z <= 0.1 -- where Gott et al. (2005) identified the SDSS
# Great Wall (z ~ 0.07-0.08).
WEDGE_DEFAULT = {"ra_min": 150.0, "ra_max": 220.0, "dec_min": -1.25, "dec_max": 1.25, "z_min": 0.0, "z_max": 0.1}


def build_wedge_count(*, ra_min=150.0, ra_max=220.0, dec_min=-1.25, dec_max=1.25, z_min=0.0, z_max=0.1):
    """COUNT(*) of the wedge slice, so the pull can be thinned uniformly."""
    sql = "SELECT COUNT(*) AS n " + _wedge_bounds(ra_min, ra_max, dec_min, dec_max, z_min, z_max).rstrip()
    return sql, {"catalog": "sdss_dr17", "table": "specobj", "builder": "lss_wedge_count", "aggregate": True, "row_limit": 1}


def wedge_thinning(n_rows: Optional[int], limit: int) -> int:
    """Smallest k with ceil(n / k) <= limit (1 = no thinning)."""
    try:
        n = int(n_rows or 0)
    except (TypeError, ValueError):
        return 1
    lim = max(1, int(limit))
    return max(1, -(-n // lim))


def build_wedge_selection(*, ra_min=150.0, ra_max=220.0, dec_min=-1.25, dec_max=1.25,
                          z_min=0.0, z_max=0.1, limit=5000, thin=1):
    """A bounded spectroscopic slice; query must succeed before plotting.

    The default equatorial strip is an analysis choice, not a catalogued wall
    boundary. User-specified geometry is validated, never silently cropped.
    ``thin`` = k keeps a uniform 1-in-k subsample (MOD(fiberid, k) = 0): a bare
    LIMIT returns rows in storage (plate) order, and live the 5 000-row cap of
    the RA 150-220 slice covered only RA 150-172 (2026-09-23).
    """
    where = _wedge_bounds(ra_min, ra_max, dec_min, dec_max, z_min, z_max)
    row_limit = min(5000, max(1, int(limit)))
    k = max(1, int(thin or 1))
    sql = "SELECT specobjid, ra, dec, z, class, zwarning " + where + (f"AND MOD(fiberid, {k}) = 0 " if k > 1 else "") + f"LIMIT {row_limit}"
    warnings = ["Bounded spectroscopic pilot; the default equatorial strip is an analysis choice. The row cap can omit galaxies; do not infer survey completeness."]
    if k > 1:
        warnings.append(f"Uniform 1-in-{k} subsample (MOD(fiberid, {k}) = 0) so the {row_limit}-row cap spans the whole slice "
                        "instead of the first plates in storage order.")
    return sql, {"catalog": "sdss_dr17", "table": "specobj", "builder": "lss_wedge_selection",
                 "row_limit": row_limit, "platform_row_cap": row_limit, "thinning": k,
                 "warnings": warnings}


def _diagram_dataframe(catalog, table, ra, dec, radius_deg, *, bands, extra_cols, limit, client, result_store,
                       point_sources=False, morphology=None, extra_value_cuts=None, owner_id=None,
                       default_limit=None, extra_select=None):
    """Cone-select the magnitude (+extra) columns for a diagram and return (result_id, df, magcols).

    ``limit=None`` means the caller made NO row-budget choice: the plotting
    budget ``default_limit`` is applied by the builder as a PLATFORM cap and
    flagged as such (platform_row_cap + warning + SQL comment, RE-B1).
    ``extra_select`` are trusted, builder-made SELECT expressions (e.g. the
    star/galaxy CASE column) appended to the column list."""
    from services import datalab_registry as reg
    magcols = {b: reg.mag_column(catalog, table, b) for b in bands}
    info = reg.describe_table(catalog, table)
    cols = [info["ra_column"], info["dec_column"]] + list(dict.fromkeys(magcols.values())) + [c for c in (extra_cols or []) if c]
    # Registry default quality cuts (e.g. DES flags_*=0) ride along unless the
    # caller cut the same column (live P5: garbage colors stretched CCD axes).
    value_cuts, quality_note = reg.merge_default_quality_cuts(catalog, table, [dict(vc) for vc in (extra_value_cuts or [])])
    validity_cols = list(dict.fromkeys(magcols.values()))
    # An explicit morphology cut wins over point_sources (which pulls the
    # catalog's registered star/galaxy cut). Both go through build_catalog_predicates.
    morph_cut = dict(morphology) if morphology else None
    ps_applied = False
    ps_note = None
    if morph_cut is None and point_sources:
        morph_cut = reg.point_source_cut(catalog, table)
        ps_applied = morph_cut is not None
        if morph_cut is None:
            ps_note = (f"{catalog}.{table} has no registered star/galaxy separator; "
                       "point_sources request ignored (returning ALL sources).")
    # Orders-of-magnitude morphology-threshold conflations (class_star's 0.5 on
    # spread_model) get a LOUD warning; the cut still executes as given (RE-B3).
    morph_warning = builders.morphology_deviation_warning(catalog, table, morphology)
    predicates = builders.build_catalog_predicates(catalog, table, value_cuts=value_cuts, morphology=morph_cut)
    # Server-side validity cuts: survey sentinel magnitudes (99.99 / -99) otherwise
    # blow the axes out to ±80 and waste the LIMIT budget on junk photometry.
    # Lower bound -5 keeps genuinely bright sources while excluding -9/-99 sentinels.
    # Built as a separate predicate group whose SQL carries a self-describing
    # comment, so provenance readers see sentinel removal — not science (RE-B2).
    predicates = predicates + builders._sentinel_mag_predicates(catalog, table, validity_cols)
    sql, meta = builders.build_cone_select(catalog, table, ra=ra, dec=dec, radius_deg=radius_deg,
                                           columns=cols, limit=limit, predicates=predicates,
                                           default_limit=(default_limit or builders.DEFAULT_ROW_LIMIT))
    extras = [str(e).strip() for e in (extra_select or []) if str(e or "").strip()]
    if extras:
        sql = sql.replace("\nFROM ", ", " + ", ".join(extras) + "\nFROM ", 1)
    result_id, result = _run_builder_sql(
        sql, meta, client=client, result_store=result_store, owner_id=owner_id
    )
    meta = dict(meta or {})
    meta["value_cuts_applied"] = [dict(vc) for vc in value_cuts]
    meta["point_source_cut_applied"] = ps_applied
    meta["morphology"] = morph_cut
    # The -5/50 cuts above are sentinel removal, NOT survey science — annotate
    # them so the provenance SQL cannot be imitated/rationalized as a science
    # choice ("mag < 50", "avoid the noise floor"; NOIRLab beta eval).
    meta["auto_validity_filters"] = {
        "columns": validity_cols,
        "range": list(_VALID_MAG_RANGE),
        "purpose": "sentinel-magnitude removal (99/-99); automatic data-validity filter, not a science cut",
    }
    meta.setdefault("warnings", []).append(
        f"Validity filter {_VALID_MAG_RANGE[0]:g} < mag < {_VALID_MAG_RANGE[1]:g} applied "
        f"automatically to {', '.join(validity_cols)} to drop survey sentinel values (99/-99). "
        "It is a platform data-validity guard, not a science cut — never present it as one or "
        "copy it into user-facing SQL as a magnitude selection."
    )
    if ps_note:
        meta.setdefault("warnings", []).append(ps_note)
    if quality_note:
        meta.setdefault("warnings", []).append(quality_note)
    if morph_warning:
        meta.setdefault("warnings", []).append(morph_warning)
    trunc_warning = policy.limit_truncation_warning(len(result.dataframe), meta.get("row_limit"))
    if trunc_warning:
        meta.setdefault("warnings", []).append(
            f"Diagram sample hit its row cap (LIMIT {int(meta['row_limit'])}) — it is a "
            "storage-order subsample of the cone, disclose the sample size in the answer."
        )
    return result_id, result, magcols, meta


# Single source of truth: services.datalab_registry.SENTINEL_MAG_RANGE
# (sentinel-magnitude removal, NOT a science cut — RE-B2 unification).
_VALID_MAG_RANGE = builders.SENTINEL_MAG_RANGE

# One-shot diagram plotting budgets. These are PLATFORM sample caps applied
# when the caller passes no limit — flagged in warnings/provenance by the
# builder, never presented as a science choice (RE-B1).
CCD_SAMPLE_BUDGET = 3000
CMD_SAMPLE_BUDGET = 5000


def _expr_identifiers(expr: str) -> List[str]:
    """Column names referenced by a plot expression (whitelisted AST walk).

    Uses the same grammar as datalab_analysis._eval_expression, so anything
    accepted here evaluates client-side later; function names are excluded.
    """
    import ast
    from services.datalab_analysis import _EXPR_FUNCS
    tree = ast.parse(str(expr or "").strip(), mode="eval")
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _EXPR_FUNCS and node.id not in names:
            names.append(node.id)
    if not names:
        raise ValueError(f"Expression {expr!r} references no result columns.")
    return names


def _expr_diagram(
    catalog, table, ra, dec, radius_deg, *,
    x_expr, y_expr, invert_y, prefix, limit, title,
    point_sources=False, morphology=None, value_cuts=None,
    overlay_locus=None,
    client, result_store, plotting, owner_id=None,
    default_limit=None,
    pm_total_min=None,
    tile_threshold_deg=None,
    abs_mag_cut=None,
):
    """One-shot diagram with derived axes (e.g. a Gaia HR diagram:
    x = bp_rp, y = phot_g_mean_mag + 5*log10(parallax) - 10).

    The SELECT list comes from the expressions' own identifiers instead of the
    per-band magnitude templates, and every referenced column gets a server-side
    finiteness guard so NaN rows cannot eat the LIMIT budget (the live P6
    pathology). Expressions are evaluated client-side by the same whitelisted
    evaluator datalab_catalog_scatter uses.
    """
    import numpy as np
    from services import datalab_analysis as analysis
    from services import datalab_registry as reg

    cols = list(dict.fromkeys(_expr_identifiers(x_expr) + _expr_identifiers(y_expr)))
    info = reg.describe_table(catalog, table)
    select_cols = [info["ra_column"], info["dec_column"]] + [c for c in cols if c not in (info["ra_column"], info["dec_column"])]
    morph_cut = dict(morphology) if morphology else None
    if morph_cut is None and point_sources:
        morph_cut = reg.point_source_cut(catalog, table)
    morph_warning = builders.morphology_deviation_warning(catalog, table, morphology)
    predicates = builders.build_catalog_predicates(
        catalog, table, value_cuts=[dict(vc) for vc in (value_cuts or [])], morphology=morph_cut,
    )
    # NaN rows fail every expression anyway — exclude them server-side so they
    # don't consume the row budget.
    predicates = predicates + [f"{builders._column(info, c)} < 'Infinity'::float8" for c in cols]
    # Magnitude columns in the expressions get the same sentinel guard as the
    # band diagrams (-5 < mag < 50): NSC pads missing photometry with 99, which
    # stretched a 2026-09-24 L03 CMD to g - r = -80..80 and g = 100.
    mag_cols = [c for c in cols if re.search(r"mag", c, re.I)]
    if mag_cols:
        predicates = predicates + builders._sentinel_mag_predicates(catalog, table, mag_cols)
    if pm_total_min is not None:
        # Total proper-motion floor IN THE SQL (datalab_selection_diagram, UI
        # benchmark 2026-09-22 L06): squared components, registry-validated
        # column names, a finite numeric threshold only.
        pm_floor = float(pm_total_min)
        if not math.isfinite(pm_floor) or pm_floor < 0:
            raise ValueError("pm_total_min must be a finite, non-negative number (mas/yr)")
        pmra_c, pmdec_c = builders._column(info, "pmra"), builders._column(info, "pmdec")
        predicates = predicates + [
            f"{pmra_c} < 'Infinity'::float8", f"{pmdec_c} < 'Infinity'::float8",
            f"({pmra_c}*{pmra_c} + {pmdec_c}*{pmdec_c}) > {pm_floor * pm_floor:.6g}",
        ]
    if abs_mag_cut is not None:
        # Absolute-magnitude floor IN THE SQL: (m + 5 log10(parallax) - 10) > M
        # with parallax in mas (Postgres LOG = base 10). DLB-06 C5 expects an
        # explicit abs G > ~10 cut for white-dwarf candidates.
        mag_c = builders._column(info, str(abs_mag_cut[0]))
        plx_c = builders._column(info, "parallax")
        m_min = float(abs_mag_cut[1])
        if not math.isfinite(m_min):
            raise ValueError("abs_mag_cut threshold must be finite")
        predicates = predicates + [f"{plx_c} > 0", f"({mag_c} + 5 * LOG({plx_c}) - 10) > {m_min:g}"]
    sql, meta = builders.build_cone_select(
        catalog, table, ra=ra, dec=dec, radius_deg=radius_deg,
        columns=select_cols, limit=limit, predicates=predicates,
        default_limit=(default_limit or builders.DEFAULT_ROW_LIMIT),
    )
    if tile_threshold_deg is not None and float(radius_deg) > float(tile_threshold_deg):
        # Wide selective cones scan millions of rows for a few matches and blow
        # Data Lab's 60 s sync window (UI benchmark 2026-09-23 L06: a 5-degree
        # Gaia cone with parallax/RUWE/proper-motion cuts timed out every call).
        # Split into sub-cones, each ALSO bounded by the parent cone, run them
        # concurrently and merge.
        rid, result, tile_note = _tiled_cone_select(
            catalog, table, ra=ra, dec=dec, radius_deg=radius_deg, select_cols=select_cols,
            predicates=predicates, limit=limit, default_limit=(default_limit or builders.DEFAULT_ROW_LIMIT),
            tile_radius_deg=float(tile_threshold_deg), client=client, result_store=result_store,
            owner_id=owner_id, base_meta=meta,
        )
    else:
        tile_note = None
        rid, result = _run_builder_sql(
            sql, meta, client=client, result_store=result_store, owner_id=owner_id
        )
    df = result.dataframe
    x = analysis._eval_expression(df, x_expr)
    y = analysis._eval_expression(df, y_expr)
    finite = np.isfinite(x) & np.isfinite(y)
    # Builder meta warnings (platform row cap disclosure) + the morphology
    # deviation check must reach the tool result, not die in meta (RE-B1/B3).
    warnings = list((meta or {}).get("warnings") or [])
    if morph_warning:
        warnings.append(morph_warning)
    if tile_note:
        warnings.append(tile_note)
    trunc = policy.limit_truncation_warning(len(df), (meta or {}).get("row_limit"))
    if trunc:
        warnings.append(
            f"Diagram sample hit its row cap (LIMIT {int(meta['row_limit'])}) — it is a "
            "subsample of the cone; disclose the sample size in the answer."
        )

    plt = plotting._apply_style(dark=False)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(x[finite], y[finite], s=6, alpha=0.4, edgecolors="none")
    # WD-locus overlay + side-of-line counts (B3): the agent prompt rule steers
    # models to pass overlay_locus='wd' on derived-axis HR diagrams and quote
    # the returned n_wd_candidates. This path used to DROP the argument (only
    # catalog_scatter honored it), so the model had nothing to quote and
    # eyeballed the diagram (found live 2026-07-18).
    _locus_info = None
    if overlay_locus:
        _locus_info = analysis._overlay_locus(ax, str(overlay_locus), x[finite], y[finite])
    ax.set_xlabel(x_expr)
    ax.set_ylabel(y_expr)
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plot_title = title or f"{catalog}.{table} — {y_expr} vs {x_expr}"
    ax.set_title(plot_title)
    fig.tight_layout()
    if _locus_info:
        # The interactive card must show what the PNG shows (UI 2026-09-24
        # L06: the answer described a locus line and highlighted candidates
        # the card did not have): WD candidates as their own trace + the line.
        xv, yv = np.asarray(x[finite], dtype=float), np.asarray(y[finite], dtype=float)
        is_wd = yv > (analysis._WD_INTERCEPT + analysis._WD_SLOPE * xv)
        plotly_spec = _plotly_scatter_spec(
            [("main sequence / other", xv[~is_wd].tolist(), yv[~is_wd].tolist())],
            x_label=x_expr, y_label=y_expr, title=plot_title, invert_y=invert_y,
        )
        plotly_spec["data"].append({"type": "scatter", "mode": "markers", "name": f"WD candidates (n={int(is_wd.sum())})",
                                    "x": [round(float(v), 4) for v in xv[is_wd]], "y": [round(float(v), 4) for v in yv[is_wd]],
                                    "marker": {"size": 5, "opacity": 0.9, "color": "#D55E00"}})
        if xv.size:
            lx = [float(np.nanmin(xv)), float(np.nanmax(xv))]
            plotly_spec["data"].append({"type": "scatter", "mode": "lines", "name": "WD locus",
                                        "x": lx, "y": [analysis._WD_INTERCEPT + analysis._WD_SLOPE * v for v in lx],
                                        "line": {"dash": "dash", "color": "#D55E00", "width": 1.2}})
        plotly_spec["layout"]["showlegend"] = True
    else:
        plotly_spec = _plotly_scatter_spec(
            [("selected", x[finite].tolist(), y[finite].tolist())],
            x_label=x_expr, y_label=y_expr, title=plot_title, invert_y=invert_y,
        )
    _extra = {"rowcount": int(len(df)), "x": x_expr, "y": y_expr,
              "points": int(finite.sum()), "morphology": morph_cut,
              "warnings": warnings,
              "plotly_spec": plotly_spec}
    if _locus_info:
        _extra.update(_locus_info)
    return _render_diagram(plotting, fig, prefix, rid,
                           {**(getattr(result, "provenance", {}) or {}), "catalog": catalog, "table": table,
                            # Platform-applied sample cap (RE-B1).
                            **({"platform_row_cap": int((meta or {})["platform_row_cap"])}
                               if (meta or {}).get("platform_row_cap") else {})},
                           _extra)


class _TiledResult:
    def __init__(self, dataframe, provenance):
        self.dataframe = dataframe
        self.provenance = provenance


def _tiled_cone_select(
    catalog, table, *, ra, dec, radius_deg, select_cols, predicates, limit, default_limit,
    tile_radius_deg, client, result_store, owner_id=None, base_meta=None,
):
    """Row selection over a wide cone as concurrent sub-cone queries.

    Every tile carries the parent q3c_radial_query bound too, so the union is
    exactly the parent selection; rows are de-duplicated on the catalog's id
    column (or ra/dec). Tiles run four at a time under child deadlines of the
    tool deadline; a tile that fails or runs out of budget is reported, never
    silently read as empty sky. Returns (result_id, result, note)."""
    import pandas as pd

    from services import datalab_registry as reg
    from services.alma_server_side import run_concurrently
    from services.tool_budgets import bounded_timeout, remaining_seconds

    info = reg.describe_table(catalog, table)
    ra_col, dec_col = info["ra_column"], info["dec_column"]
    parent = f"q3c_radial_query({ra_col}, {dec_col}, {float(ra):.8g}, {float(dec):.8g}, {float(radius_deg):.8g})"
    tile_r = max(0.5, float(tile_radius_deg))
    centers = _tile_cone_centers(float(radius_deg), tile_r, cell_margin_deg=0.0)
    max_tiles = max(1, int(os.getenv("DATALAB_TILED_SELECT_MAX_TILES", "24")))
    centers = centers[:max_tiles]
    queries = []
    for dx, dy in centers:
        tdec = max(-89.5, min(89.5, float(dec) + dy))
        tra = (float(ra) + dx / _cosd(tdec)) % 360.0
        sql_t, meta_t = builders.build_cone_select(
            catalog, table, ra=tra, dec=tdec, radius_deg=tile_r, columns=select_cols, limit=limit,
            predicates=list(predicates) + [parent], default_limit=default_limit,
        )
        validated = policy.validate(sql_t, source="builder", meta=meta_t)
        queries.append((validated, meta_t))

    def _one(validated):
        seconds = bounded_timeout(float(getattr(client, "timeout", 60.0) or 60.0), minimum=3.0, label="Data Lab tile select")
        return client.query(sql=validated.sql, fmt="pandas", async_fallback=False, timeout=seconds)

    frames, failed, done, truncated = [], 0, 0, 0
    for start in range(0, len(queries), 4):
        left = remaining_seconds()
        if left is not None and left < 10.0:
            break
        batch = queries[start:start + 4]
        outs = run_concurrently([(lambda v=v: _one(v)) for v, _m in batch], wall_seconds=(left - 5.0) if left is not None else 120.0)
        for (validated, meta_t), out in zip(batch, outs):
            if isinstance(out, BaseException):
                failed += 1
                continue
            done += 1
            frames.append(out.dataframe)
            if policy.limit_truncation_warning(len(out.dataframe), (validated.meta or {}).get("row_limit")):
                truncated += 1
    if not frames:
        raise RuntimeError(f"all {len(queries)} Data Lab sub-cone queries failed or ran out of budget")
    merged = pd.concat([f for f in frames if f is not None], ignore_index=True)
    id_col = next((c for c in ("source_id", "id", "objid") if c in merged.columns), None)
    merged = merged.drop_duplicates(subset=[id_col] if id_col else [ra_col, dec_col]).reset_index(drop=True)
    total = len(queries)
    note = (f"Wide cone split into {total} sub-cones of {tile_r:g} deg (each bounded by the parent cone): "
            f"{done} completed" + (f", {failed} failed" if failed else "") + (f", {total - done - failed} not run (budget)" if total - done - failed else "")
            + (f"; {truncated} sub-cone(s) hit the row cap" if truncated else "") + ".")
    if done < total:
        note += " The sample is PARTIAL: unqueried sub-cones are unknown, not empty."
    provenance = {"validated_sql": queries[0][0].sql, "tiles_total": total, "tiles_completed": done, "tiles_failed": failed,
                  "tile_radius_deg": tile_r, "parent_bound": parent}
    store_meta = {**dict(base_meta or {}), "validated_sql": queries[0][0].sql, "provenance": provenance,
                  **({"owner_id": str(owner_id)} if owner_id else {}), "warnings": [note]}
    result_id = result_store.put(merged, store_meta)
    return result_id, _TiledResult(merged, provenance), note


def _valid_mag_mask(*series):
    """Finite AND inside the physical magnitude range for every series given."""
    import numpy as np
    import pandas as pd
    mask = None
    lo, hi = _VALID_MAG_RANGE
    for s in series:
        vals = pd.to_numeric(s, errors="coerce")
        good = np.isfinite(vals) & (vals > lo) & (vals < hi)
        mask = good if mask is None else (mask & good)
    return mask


def _plotly_scatter_spec(panels, *, x_label, y_label, title, invert_y=False, max_points=4000):
    """Build a JSON-safe Plotly figure spec (traces + layout) for 1-2 scatter panels.

    panels: [(label, x_values, y_values)] — values are downsampled to max_points
    per panel so the spec stays light enough to stream over SSE.
    """
    import numpy as np
    traces = []
    n_panels = max(1, len(panels))
    for idx, (label, xs, ys) in enumerate(panels):
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if len(x) > max_points:
            sel = np.random.default_rng(0).choice(len(x), size=max_points, replace=False)
            x, y = x[sel], y[sel]
        trace = {
            # Plain SVG scatter, NOT scattergl: the frontend ships the basic
            # Plotly bundle (no WebGL renderer), and points are capped anyway.
            "type": "scatter",
            "mode": "markers",
            "name": f"{label} (n={len(xs)})",
            "x": [round(float(v), 4) for v in x],
            "y": [round(float(v), 4) for v in y],
            "marker": {"size": 3, "opacity": 0.55},
        }
        if n_panels > 1:
            trace["xaxis"] = f"x{idx + 1 if idx else ''}"
            trace["yaxis"] = f"y{idx + 1 if idx else ''}"
        traces.append(trace)
    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": x_label}},
        "yaxis": {"title": {"text": y_label}, "autorange": "reversed" if invert_y else True},
        "showlegend": n_panels > 1,
        "margin": {"l": 55, "r": 15, "t": 45, "b": 45},
    }
    if n_panels > 1:
        layout["grid"] = {"rows": 1, "columns": n_panels, "pattern": "independent"}
        layout["xaxis2"] = {"title": {"text": x_label}}
        layout["yaxis2"] = {"title": {"text": y_label}, "autorange": "reversed" if invert_y else True}
    return {"data": traces, "layout": layout}


def _render_diagram(plotting, fig, prefix, result_id, provenance, extra):
    import uuid as _uuid
    saved = plotting._save_and_encode(fig, f"{prefix}_{_uuid.uuid4().hex[:10]}")
    out = {
        "success": True,
        "result_id": result_id,
        "image_base64": saved.get("base64_png"),
        "path": saved.get("web_url"),
        "provenance": dict(provenance or {}),
    }
    out.update(extra or {})
    return out


def _resolve_split_threshold(catalog, table, split_col, split_threshold):
    """(threshold, deviation_warning) for a star/galaxy split column.

    A None threshold must NEVER silently inherit the DES spread_model 0.003:
    class_star-style classifier probabilities split at 0.5 (0.5 is
    class_star-ONLY), and a table with a registered star/galaxy convention
    (_POINT_SOURCE_CUTS) whose column family matches the split column uses
    that registered scale (guard CX-03). A caller-supplied threshold is used
    as given — never clamped — and is run through
    morphology_deviation_warning so orders-of-magnitude conflations get the
    LOUD RE-B3 warning on the split path too."""
    col = str(split_col or "").strip().lower()
    family = builders._morph_family(col)
    if split_threshold is None:
        threshold = None
        try:
            from services import datalab_registry as reg
            reference = reg.point_source_cut(catalog, table)
        except Exception:  # noqa: BLE001 - unknown table => family fallback below
            reference = None
        if reference and builders._morph_family(str(reference.get("column") or "")) == family:
            threshold = builders._morph_scale(reference)
        if threshold is None:
            threshold = 0.5 if family == "class_star" else 0.003
        return float(threshold), None
    threshold = float(split_threshold)
    warning = builders.morphology_deviation_warning(
        catalog, table, {"column": col, "op": "<", "value": threshold}
    )
    return threshold, warning


def _split_groups(split_col, s, finite, split_threshold):
    """Star/galaxy masks for a morphology split column -> (groups, note).

    delve_dr3's ext_coadd is CATEGORICAL (-9 no data, 0 hi-conf star,
    1 candidate star, 2 candidate galaxy, 3 hi-conf galaxy): the
    spread_model-style |threshold| split filed candidate stars under
    'galaxies' and no-data rows under 'stars'
    (delve-ccd-split-misclass / dl-delve-ccd-split-misclassifies).
    Categorical columns use set membership (stars IN (0,1), galaxies IN (2,3),
    -9 excluded from both panels). class_star-style classifier probabilities
    are HIGH for stars (registered convention class_star > 0.5), so the
    threshold split is s > t, not s <= t (guard CX-03). Continuous
    spread_model-style columns use the registered TWO-SIDED convention
    |s| <= t (DES: -0.003 <= spread_model_r <= 0.003) — the old one-sided
    s <= t filed negative outliers below -t under stars (guard CX-04)."""
    col = str(split_col).strip().lower()
    if col == "ext_coadd":
        groups = [("stars", finite & s.isin([0, 1])), ("galaxies", finite & s.isin([2, 3]))]
        n_no_data = int((finite & (s == -9)).sum())
        note = (
            f"ext_coadd split: stars = ext_coadd IN (0,1), galaxies = ext_coadd IN (2,3); "
            f"{n_no_data} no-data (ext_coadd = -9) source(s) excluded from both panels."
            if n_no_data
            else None
        )
        return groups, note
    if builders._morph_family(col) == "class_star":
        return (
            [("stars", finite & (s > split_threshold)), ("galaxies", finite & (s <= split_threshold))],
            f"class_star split: stars = {col} > {split_threshold:g} (classifier probability; "
            "high = star), galaxies = the rest.",
        )
    return (
        [("stars", finite & (s.abs() <= split_threshold)), ("galaxies", finite & (s.abs() > split_threshold))],
        None,
    )


def _split_case_sql(split_col, split_threshold) -> str:
    """The star/galaxy split as a SQL CASE column ``morph`` ('star' | 'galaxy';
    NULL = no morphology data), so the executed query carries the split.
    Same conventions as _split_groups: ext_coadd categories, class_star high =
    star, spread_model-style two-sided |s| <= t = star."""
    col = str(split_col or "").strip().lower()
    if not re.match(r"^[a-z_][a-z0-9_]*$", col):
        raise ValueError(f"Invalid split column {split_col!r}")
    if col == "ext_coadd":
        return ("CASE WHEN ext_coadd IN (0, 1) THEN 'star' WHEN ext_coadd IN (2, 3) THEN 'galaxy' "
                "END AS morph")
    t = float(split_threshold)
    if builders._morph_family(col) == "class_star":
        return f"CASE WHEN {col} > {t:g} THEN 'star' WHEN {col} IS NOT NULL THEN 'galaxy' END AS morph"
    return (f"CASE WHEN {col} BETWEEN {-abs(t):g} AND {abs(t):g} THEN 'star' "
            f"WHEN {col} IS NOT NULL THEN 'galaxy' END AS morph")


def _describe_cut(vc) -> str:
    val = vc.get("value")
    lit = f"'{val}'" if isinstance(val, str) else f"{val:g}" if isinstance(val, float) else str(val)
    return f"{vc.get('column')} {vc.get('op')} {lit}"


def color_color_diagram(
    catalog, table, ra, dec, radius_deg, *,
    x_bands=("g", "r"), y_bands=("r", "i"),
    split_col=None, split_threshold=None, limit=None, title=None,
    point_sources=False, morphology=None, value_cuts=None,
    x_expr=None, y_expr=None,
    client=None, result_store=None, plotting_service=None, owner_id=None,
):
    """P5: one-shot color-color diagram. Queries the cone, optionally splits into stars/galaxies
    by a morphology column (auto-detected from the registry, e.g. DES spread_model_r), and renders
    a 1- or 2-panel CCD as a single image. Sentinel magnitudes (99/-99 padding) are cut both
    server- and client-side via the unified SENTINEL_MAG_RANGE guard (-5 < mag < 50 — sentinel
    removal, not science) so the axes stay physical. An explicit morphology cut overrides
    point_sources. x_expr/y_expr switch to derived axes (single panel, no star/galaxy split)."""
    import pandas as pd
    from services import datalab_registry as reg
    from services.plotting import PlottingService
    client = client or _default_client()
    result_store = result_store or _default_result_store()
    plotting = plotting_service or PlottingService()
    if x_expr or y_expr:
        if not (x_expr and y_expr):
            raise ValueError("Provide BOTH x_expr and y_expr (or neither) for a derived-axis CCD.")
        return _expr_diagram(
            catalog, table, ra, dec, radius_deg,
            x_expr=str(x_expr), y_expr=str(y_expr), invert_y=False, prefix="datalab_ccd",
            limit=limit, title=title, point_sources=point_sources, morphology=morphology,
            value_cuts=value_cuts, client=client, result_store=result_store, plotting=plotting,
            owner_id=owner_id, default_limit=CCD_SAMPLE_BUDGET,
        )
    # A morphology-selected sample is one population — don't auto-split it into stars/galaxies.
    if split_col is None and not point_sources and not morphology:
        split_col = reg.morphology_split_column(catalog, table)
    # Threshold semantics (guard CX-03): None derives the FAMILY default
    # (class_star → 0.5; registered spread_model-style conventions → their
    # _POINT_SOURCE_CUTS scale) instead of blanket-defaulting to DES's 0.003;
    # explicit thresholds run through the morphology deviation check too.
    split_deviation_warning = None
    if split_col is not None:
        split_threshold, split_deviation_warning = _resolve_split_threshold(
            catalog, table, split_col, split_threshold
        )

    bands = list(dict.fromkeys([*x_bands, *y_bands]))
    # Registered CCD magnitude window (DES: 16 < mag_auto_i < 23), unless the
    # caller already cut that column (DLB-05 C7: flags, fluxerr AND a window).
    value_cuts = [dict(vc) for vc in (value_cuts or [])]
    window = reg.ccd_magnitude_window(catalog, table)
    window_note = None
    if window and str(window["column"]).lower() not in {str(vc.get("column", "")).lower() for vc in value_cuts}:
        value_cuts += [{"column": window["column"], "op": ">", "value": window["min"]},
                       {"column": window["column"], "op": "<", "value": window["max"]}]
        window_note = (f"Magnitude window {window['min']:g} < {window['column']} < {window['max']:g} "
                       "(drops saturated and noise-dominated sources from the colour axes); "
                       "override with your own value_cut on that column.")
    split_sql = _split_case_sql(split_col, split_threshold) if split_col else None
    rid, result, magcols, _meta = _diagram_dataframe(
        catalog, table, ra, dec, radius_deg, bands=bands,
        extra_cols=[split_col] if split_col else [], limit=limit, client=client, result_store=result_store,
        point_sources=point_sources, morphology=morphology, extra_value_cuts=value_cuts,
        owner_id=owner_id, default_limit=CCD_SAMPLE_BUDGET,
        extra_select=[split_sql] if split_sql else None,
    )
    df = result.dataframe
    _ps_applied = bool(_meta.get("point_source_cut_applied"))
    used_cols = [magcols[b] for b in dict.fromkeys([*x_bands, *y_bands])]
    valid = _valid_mag_mask(*[df[c] for c in used_cols])
    x = pd.to_numeric(df[magcols[x_bands[0]]], errors="coerce") - pd.to_numeric(df[magcols[x_bands[1]]], errors="coerce")
    y = pd.to_numeric(df[magcols[y_bands[0]]], errors="coerce") - pd.to_numeric(df[magcols[y_bands[1]]], errors="coerce")
    finite = valid
    xl, yl = f"{x_bands[0]}-{x_bands[1]}", f"{y_bands[0]}-{y_bands[1]}"

    plt = plotting._apply_style(dark=False)
    populations = []
    panels = []
    split_note = None
    if split_col and split_col in df.columns:
        s = pd.to_numeric(df[split_col], errors="coerce")
        groups, split_note = _split_groups(split_col, s, finite, split_threshold)
        if "morph" in df.columns:
            # The executed SQL classified each row: use the server's column.
            morph = df["morph"].astype(str).str.strip().str.lower()
            groups = [("stars", finite & (morph == "star")), ("galaxies", finite & (morph == "galaxy"))]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        for ax, (label, mask) in zip(axes, groups):
            ax.scatter(x[mask], y[mask], s=6, alpha=0.4, edgecolors="none")
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
            ax.set_title(f"{label} (n={int(mask.sum())})")
            populations.append({"population": label, "n": int(mask.sum())})
            panels.append((label, x[mask].tolist(), y[mask].tolist()))
    else:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.scatter(x[finite], y[finite], s=6, alpha=0.4, edgecolors="none")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=0.3)
        pop_label = "point sources" if _ps_applied else ("selected" if _meta.get("morphology") else "all")
        populations.append({"population": pop_label, "n": int(finite.sum())})
        panels.append((pop_label, x[finite].tolist(), y[finite].tolist()))
    plot_title = title or f"{catalog}.{table} — {xl} vs {yl}"
    fig.suptitle(plot_title)
    fig.tight_layout()
    plotly_spec = _plotly_scatter_spec(panels, x_label=xl, y_label=yl, title=plot_title)
    ps_applied = bool(_meta.get("point_source_cut_applied"))
    cuts_applied = [_describe_cut(vc) for vc in (_meta.get("value_cuts_applied") or [])]
    if split_sql:
        cuts_applied.append(f"star/galaxy split in the SQL: {split_sql}")
    if _meta.get("morphology"):
        cuts_applied.append(f"morphology cut: {_meta['morphology']}")
    return _render_diagram(plotting, fig, "datalab_ccd", rid,
                           {**(getattr(result, "provenance", {}) or {}), "catalog": catalog, "table": table,
                            "auto_validity_filters": _meta.get("auto_validity_filters"),
                            # Platform-applied sample cap, stamped so it can
                            # never read as a science choice (RE-B1).
                            **({"platform_row_cap": int(_meta["platform_row_cap"])}
                               if _meta.get("platform_row_cap") else {})},
                           {"rowcount": int(len(df)), "x": xl, "y": yl, "split_col": split_col,
                            "split_threshold": split_threshold,
                            "split_sql": split_sql,
                            "cuts_applied": cuts_applied,
                            "point_sources": ps_applied, "morphology": _meta.get("morphology"),
                            "populations": populations,
                            "warnings": list(_meta.get("warnings") or [])
                            + ([window_note] if window_note else [])
                            + ([split_note] if split_note else [])
                            + ([split_deviation_warning] if split_deviation_warning else []),
                            "excluded_invalid_mags": int(len(df) - int(finite.sum())),
                            "plotly_spec": plotly_spec})


def color_magnitude_diagram(
    catalog, table, ra, dec, radius_deg, *,
    blue_band="g", red_band="r", mag_band=None, limit=None, title=None,
    point_sources=False, morphology=None, value_cuts=None,
    x_expr=None, y_expr=None, overlay_locus=None,
    client=None, result_store=None, plotting_service=None, owner_id=None,
    pm_total_min=None, tile_threshold_deg=None, abs_mag_cut=None,
):
    """P3: one-shot color-magnitude diagram (mag_band vs blue-red color), magnitude axis inverted.
    Sentinel magnitudes are excluded server- and client-side; point_sources=True applies the
    catalog's registered star/galaxy cut (e.g. NSC class_star > 0.5). An explicit morphology
    cut overrides point_sources. x_expr/y_expr switch to derived axes (e.g. a Gaia HR
    diagram: x=bp_rp, y=phot_g_mean_mag + 5*log10(parallax) - 10) — previously these
    params were silently IGNORED and the band fallback 400'd on Gaia (live P6)."""
    import pandas as pd
    from services.plotting import PlottingService
    client = client or _default_client()
    result_store = result_store or _default_result_store()
    plotting = plotting_service or PlottingService()
    if x_expr or y_expr:
        if not (x_expr and y_expr):
            raise ValueError("Provide BOTH x_expr and y_expr (or neither) for a derived-axis CMD.")
        return _expr_diagram(
            catalog, table, ra, dec, radius_deg,
            x_expr=str(x_expr), y_expr=str(y_expr), invert_y=True, prefix="datalab_cmd",
            limit=limit, title=title, point_sources=point_sources, morphology=morphology,
            value_cuts=value_cuts, overlay_locus=overlay_locus, pm_total_min=pm_total_min, abs_mag_cut=abs_mag_cut,
            tile_threshold_deg=tile_threshold_deg,
            client=client, result_store=result_store, plotting=plotting,
            owner_id=owner_id, default_limit=CMD_SAMPLE_BUDGET,
        )
    mag_band = mag_band or blue_band
    bands = list(dict.fromkeys([blue_band, red_band, mag_band]))
    from services import datalab_registry as reg
    # Registered faint depth bound (NSC: g, r < 24) on every plotted band the
    # caller did not already cut (DLB-03 C6: "magnitude limits ~g,r < 24").
    value_cuts = [dict(vc) for vc in (value_cuts or [])]
    depth_bound = reg.cmd_depth_bound(catalog, table)
    depth_cuts: List[str] = []
    if depth_bound is not None:
        cut_cols = {str(vc.get("column", "")).lower() for vc in value_cuts}
        for b in bands:
            col = reg.mag_column(catalog, table, b)
            if col.lower() not in cut_cols:
                value_cuts.append({"column": col, "op": "<", "value": float(depth_bound)})
                depth_cuts.append(f"{col} < {depth_bound:g}")
                cut_cols.add(col.lower())
    rid, result, magcols, _meta = _diagram_dataframe(
        catalog, table, ra, dec, radius_deg, bands=bands, extra_cols=[], limit=limit,
        client=client, result_store=result_store, point_sources=point_sources,
        morphology=morphology, extra_value_cuts=value_cuts, owner_id=owner_id,
        default_limit=CMD_SAMPLE_BUDGET,
    )
    df = result.dataframe
    valid = _valid_mag_mask(*[df[magcols[b]] for b in dict.fromkeys([blue_band, red_band, mag_band])])
    color = pd.to_numeric(df[magcols[blue_band]], errors="coerce") - pd.to_numeric(df[magcols[red_band]], errors="coerce")
    mag = pd.to_numeric(df[magcols[mag_band]], errors="coerce")
    finite = valid
    xl, yl = f"{blue_band}-{red_band}", f"{mag_band}"
    plt = plotting._apply_style(dark=False)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(color[finite], mag[finite], s=6, alpha=0.4, edgecolors="none")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    plot_title = title or f"{catalog}.{table} CMD — {yl} vs {xl}"
    ax.set_title(plot_title)
    fig.tight_layout()
    ps_applied = bool(_meta.get("point_source_cut_applied"))
    label = "point sources" if ps_applied else ("selected" if _meta.get("morphology") else "all sources")
    plotly_spec = _plotly_scatter_spec(
        [(label, color[finite].tolist(), mag[finite].tolist())],
        x_label=xl, y_label=yl, title=plot_title, invert_y=True,
    )
    # Depth of the plotted sample: the bound applied and where the magnitude
    # histogram turns over (completeness starts to fall).
    mags = mag[finite].to_numpy(dtype=float) if hasattr(mag[finite], "to_numpy") else np.asarray(mag[finite], dtype=float)
    # A capped sample (storage-order LIMIT) cannot measure depth, and a peak
    # counts as a turnover only if the counts really fall fainter (guard CX-07).
    turnover, depth_note = None, "too few points to measure the depth"
    capped = bool(_meta.get("row_limit")) and len(df) >= int(_meta.get("row_limit") or 0)
    if capped:
        depth_note = (f"The plotted sample hit its row cap ({int(_meta['row_limit'])} rows, storage order), so its magnitude "
                      "histogram does not measure the catalogue depth"
                      + (f"; the faint limit here is the bound applied ({', '.join(depth_cuts)})." if depth_cuts else "."))
    elif mags.size >= 50:
        hist, edges = np.histogram(mags, bins=np.arange(np.floor(mags.min()), np.ceil(mags.max()) + 0.25, 0.25))
        if hist.size >= 8:
            # Smoothed counts; a turnover needs the peak in the fainter 60 %
            # of the range and the three faintest bins at <= half the peak
            # (number counts rise to the completeness limit, then fall).
            sm = np.convolve(hist.astype(float), np.ones(3) / 3.0, mode="same")
            i = int(np.argmax(sm))
            tail = sm[-3:]
            if i >= int(0.4 * sm.size) and i < sm.size - 3 and float(tail.mean()) <= 0.5 * float(sm[i]):
                turnover = round(float(0.5 * (edges[i] + edges[i + 1])), 2)
                depth_note = (f"The {mag_band}-band counts peak at {mag_band} ~ {turnover:g} and fall below half that peak "
                              "fainter: the sample is incomplete beyond it.")
            elif depth_cuts:
                depth_note = (f"No turnover within the plotted range: the faint limit here is the bound applied "
                              f"({', '.join(depth_cuts)}).")
            else:
                # No faint bound of ours: say only what the histogram shows (CX-07 verify).
                depth_note = "No turnover within the plotted range: this sample does not show where the catalogue becomes incomplete."
    depth = {"bound_applied": depth_cuts or None,
             "bound_source": ("registry default (no caller cut on these bands)" if depth_cuts else
                              "caller cuts / none registered"),
             "turnover_mag": turnover, "sample_capped": capped, "note": depth_note}
    return _render_diagram(plotting, fig, "datalab_cmd", rid,
                           {**(getattr(result, "provenance", {}) or {}), "catalog": catalog, "table": table,
                            "auto_validity_filters": _meta.get("auto_validity_filters"),
                            # Platform-applied sample cap, stamped so it can
                            # never read as a science choice (RE-B1).
                            **({"platform_row_cap": int(_meta["platform_row_cap"])}
                               if _meta.get("platform_row_cap") else {})},
                           {"rowcount": int(len(df)), "x": xl, "y": yl, "points": int(finite.sum()),
                            "point_sources": ps_applied, "morphology": _meta.get("morphology"),
                            "depth": depth,
                            "cuts_applied": [_describe_cut(vc) for vc in (_meta.get("value_cuts_applied") or [])],
                            "warnings": list(_meta.get("warnings") or []),
                            "excluded_invalid_mags": int(len(df) - int(finite.sum())),
                            "plotly_spec": plotly_spec})


__all__ = [
    "MAX_TILES",
    "DEFAULT_CANDIDATE_BUDGET",
    "color_color_diagram",
    "color_magnitude_diagram",
    "confirm_sky_area",
    "density_then_cutouts",
    "footprint_area_deg2",
    "rank_candidates",
    "tile_footprint",
    "tiled_sky_scan",
]
