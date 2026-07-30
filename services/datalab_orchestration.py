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
import time
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


def _run_builder_sql(
    sql: str,
    meta: Mapping[str, Any],
    *,
    client,
    result_store,
    owner_id: Optional[str] = None,
    async_fallback: bool = True,
):
    """Validate builder SQL through the governor, run it, and store the frame -> result_id."""
    from services import datalab_registry as registry
    registry.ensure_tap_schema_fresh(client)
    validated = policy.validate(sql, source="builder", meta=meta)
    result = client.query(sql=validated.sql, fmt="pandas", async_fallback=async_fallback)
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
    notes: List[str] = []
    if quality_note:
        notes.append(quality_note)
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
    predicates = builders.build_catalog_predicates(catalog, table, color_cut=color_cut, value_cuts=merged_cuts, morphology=morphology)
    sql, meta = builders.build_density_aggregate(
        catalog, table, mode="grid", step_deg=step_deg,
        ra=ra, dec=dec, radius_deg=radius_deg, predicates=predicates,
        limit=max(top_n * 4, 50),
    )
    result_id, result = _run_builder_sql(
        sql, meta, client=client, result_store=result_store, owner_id=owner_id
    )
    df = result.dataframe
    # The aggregate is ORDER BY source_count DESC. Greedily skip cells adjacent
    # to an already-accepted peak (2 of live P12's 5 "candidates" were neighbor
    # cells of the same clump), and flag peaks inside known MW objects.
    min_sep = 2.0 * float(step_deg)
    peaks: List[Dict[str, Any]] = []
    notes: List[str] = []
    if quality_note:
        notes.append(quality_note)
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
    return {
        "success": True,
        "result_id": result_id,
        "n_peaks": len(peaks),
        "peaks": peaks,
        "cutout_grid": grid,
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
    limit: int = 5000,
    tile_radius_deg: Optional[float] = None,
    max_seconds: Optional[float] = None,
    client: Any = None,
    result_store: Any = None,
    owner_id: Optional[str] = None,
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

    max_tiles = int(os.getenv("DATALAB_TILED_AGG_MAX_TILES", "16"))
    # Budget must leave room for the rest of the turn: the chat hard cap is
    # 900s and a model may run 2+ aggregates (live DS-P8 attempt 1 died at the
    # cap with a 360s budget: Galactic-center tiles all timed out serially).
    if max_seconds is None:
        max_seconds = float(os.getenv("DATALAB_TILED_AGG_MAX_SECONDS", "210"))
    cell_margin = max(0.3, 2.0 * float(step_deg)) if mode_key == "grid" else 0.3
    tile_r = float(tile_radius_deg or os.getenv("DATALAB_TILE_RADIUS_DEG", "5"))
    tile_r = min(tile_r, max(0.5, float(radius_deg) / 1.5))
    # Coarsen until the tile count fits the cap (bigger tiles may time out per-tile,
    # which the subdivision fallback below absorbs).
    centers = _tile_cone_centers(radius_deg, tile_r, cell_margin)
    while len(centers) > max_tiles and tile_r < float(radius_deg):
        tile_r *= 1.5
        centers = _tile_cone_centers(radius_deg, tile_r, cell_margin)

    # Per-tile LIMIT truncation: each tile is ORDER BY source_count DESC, so a
    # tile that filled its row cap silently dropped its SPARSEST cells and the
    # merged map read them as "no data" with no stamp — the repo's row-capped-
    # as-complete bug pattern (dl-tiled-agg-silent-cell-drop /
    # tiled-agg-silent-cell-truncation). Track it and stamp the provenance.
    tiles_truncated = 0
    effective_limit: Optional[int] = None

    def _run_tile(dx: float, dy: float, r: float) -> Optional[Any]:
        nonlocal tiles_truncated, effective_limit
        tdec = max(-89.5, min(89.5, float(dec) + dy))
        tra = (float(ra) + dx / _cosd(tdec)) % 360.0
        sql, meta = builders.build_density_aggregate(
            catalog, table, mode=mode_key, step_deg=step_deg, healpix_column=healpix_column,
            ra=tra, dec=tdec, radius_deg=r, predicates=base_predicates, limit=limit,
        )
        validated = policy.validate(sql, source="builder", meta=meta)
        result = client.query(sql=validated.sql, fmt="pandas", async_fallback=False)
        row_limit = (validated.meta or {}).get("row_limit")
        if policy.limit_truncation_warning(len(result.dataframe), row_limit):
            tiles_truncated += 1
            effective_limit = int(row_limit)
        return result

    frames: List[Any] = []
    tiles_run = 0
    tile_errors = 0
    subdivided = 0
    outer_tiles_done = 0
    started = time.monotonic()
    budget_stop = False
    for (dx, dy) in centers:
        # Early abort for hopeless regions (live DS-P8: every Galactic-center
        # NSC tile timed out serially, even subdivided — each failed tile costs
        # up to 5x the sync window). If the first 2 outer tiles produced no
        # data at all, this tile size cannot work; fail fast with the hint.
        if outer_tiles_done >= 2 and not frames:
            break
        if time.monotonic() - started > max_seconds:
            budget_stop = True
            break
        try:
            frames.append(_run_tile(dx, dy, tile_r).dataframe)
            tiles_run += 1
        except Exception as exc:  # noqa: BLE001 - a slow/broken tile must not kill the map
            if "timed out" not in str(exc).lower():
                tile_errors += 1
                continue
            # One level of subdivision: 4 half-area cones covering the tile
            # (child radius tile_r/sqrt(2) is the exact cover; add the cell
            # margin so seam cells stay fully contained in one child).
            subdivided += 1
            child_r = tile_r / math.sqrt(2.0) + cell_margin
            for (cx, cy) in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)):
                if time.monotonic() - started > max_seconds:
                    budget_stop = True
                    break
                try:
                    frames.append(_run_tile(dx + cx * tile_r, dy + cy * tile_r, child_r).dataframe)
                    tiles_run += 1
                except Exception:  # noqa: BLE001
                    tile_errors += 1
            if budget_stop:
                break
        finally:
            # The fail-fast check above reads this; it was never incremented,
            # so the documented two-fruitless-outer-tiles abort was dead code
            # and hopeless regions ran the whole wall budget.
            outer_tiles_done += 1

    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        raise RuntimeError(
            f"Tiled density aggregate produced no data: {tiles_run} tiles ok, "
            f"{tile_errors} failed. This field is too crowded for the anonymous 60s "
            f"sync window even at {tile_r:.1f}° tiles. What works (live-verified): a "
            "radius ≤2° cone on crowded fields, a BRIGHT magnitude cut (e.g. gmag < 18), "
            "or a field away from the Galactic centre/plane (|b| > 5°). Get a working "
            "map at radius 2° FIRST and render it, then widen if time permits."
        )
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

    warnings: List[str] = []
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
            "tiles_failed": tile_errors,
            "tiles_subdivided": subdivided,
            "sync_timeout_fallback": True,
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
        "tiles_failed": tile_errors,
        "tile_radius_deg": round(tile_r, 3),
        "warnings": warnings,
        "query_summary": (
            f"tiled density aggregate ({mode_key}): {tiles_run} cone tiles of {tile_r:.1f}° "
            f"covering ra={float(ra):.4g}, dec={float(dec):.4g}, radius={float(radius_deg):.4g}°"
        ),
        "preview": preview,
        "note": (
            "The single wide aggregate exceeded the Data Lab 60s sync window, so the cone was "
            "auto-tiled into sub-cones and merged (identical semantics: every tile is also bounded "
            "by the parent cone). Render THIS result_id with datalab_sky_density_map NOW, before "
            "attempting any wider region — turns have a hard time budget and a rendered map beats "
            "an unrendered bigger one."
        ),
    }


def _diagram_dataframe(catalog, table, ra, dec, radius_deg, *, bands, extra_cols, limit, client, result_store,
                       point_sources=False, morphology=None, extra_value_cuts=None, owner_id=None):
    """Cone-select the magnitude (+extra) columns for a diagram and return (result_id, df, magcols)."""
    from services import datalab_registry as reg
    magcols = {b: reg.mag_column(catalog, table, b) for b in bands}
    info = reg.describe_table(catalog, table)
    cols = [info["ra_column"], info["dec_column"]] + list(dict.fromkeys(magcols.values())) + [c for c in (extra_cols or []) if c]
    # Server-side validity cuts: survey sentinel magnitudes (99.99 / -99) otherwise
    # blow the axes out to ±80 and waste the LIMIT budget on junk photometry.
    # Lower bound -5 keeps genuinely bright sources while excluding -9/-99 sentinels.
    # Registry default quality cuts (e.g. DES flags_*=0) ride along unless the
    # caller cut the same column (live P5: garbage colors stretched CCD axes).
    value_cuts, quality_note = reg.merge_default_quality_cuts(catalog, table, [dict(vc) for vc in (extra_value_cuts or [])])
    validity_cols = list(dict.fromkeys(magcols.values()))
    for col in validity_cols:
        value_cuts.append({"column": col, "op": ">", "value": _VALID_MAG_RANGE[0]})
        value_cuts.append({"column": col, "op": "<", "value": _VALID_MAG_RANGE[1]})
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
    predicates = builders.build_catalog_predicates(catalog, table, value_cuts=value_cuts, morphology=morph_cut)
    sql, meta = builders.build_cone_select(catalog, table, ra=ra, dec=dec, radius_deg=radius_deg,
                                           columns=cols, limit=limit, predicates=predicates)
    result_id, result = _run_builder_sql(
        sql, meta, client=client, result_store=result_store, owner_id=owner_id
    )
    meta = dict(meta or {})
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
    trunc_warning = policy.limit_truncation_warning(len(result.dataframe), meta.get("row_limit"))
    if trunc_warning:
        meta.setdefault("warnings", []).append(
            f"Diagram sample hit its row cap (LIMIT {int(meta['row_limit'])}) — it is a "
            "storage-order subsample of the cone, disclose the sample size in the answer."
        )
    return result_id, result, magcols, meta


_VALID_MAG_RANGE = (-5.0, 50.0)


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
    predicates = builders.build_catalog_predicates(
        catalog, table, value_cuts=[dict(vc) for vc in (value_cuts or [])], morphology=morph_cut,
    )
    # NaN rows fail every expression anyway — exclude them server-side so they
    # don't consume the row budget.
    predicates = predicates + [f"{builders._column(info, c)} < 'Infinity'::float8" for c in cols]
    sql, meta = builders.build_cone_select(
        catalog, table, ra=ra, dec=dec, radius_deg=radius_deg,
        columns=select_cols, limit=limit, predicates=predicates,
    )
    rid, result = _run_builder_sql(
        sql, meta, client=client, result_store=result_store, owner_id=owner_id
    )
    df = result.dataframe
    x = analysis._eval_expression(df, x_expr)
    y = analysis._eval_expression(df, y_expr)
    finite = np.isfinite(x) & np.isfinite(y)
    warnings = []
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
                           {**(getattr(result, "provenance", {}) or {}), "catalog": catalog, "table": table},
                           _extra)


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


def _split_groups(split_col, s, finite, split_threshold):
    """Star/galaxy masks for a morphology split column -> (groups, note).

    delve_dr3's ext_coadd is CATEGORICAL (-9 no data, 0 hi-conf star,
    1 candidate star, 2 candidate galaxy, 3 hi-conf galaxy): the
    spread_model-style |threshold| split filed candidate stars under
    'galaxies' and no-data rows under 'stars'
    (delve-ccd-split-misclass / dl-delve-ccd-split-misclassifies).
    Categorical columns use set membership (stars IN (0,1), galaxies IN (2,3),
    -9 excluded from both panels); the numeric threshold applies only to
    spread_model-style continuous columns."""
    if str(split_col).strip().lower() == "ext_coadd":
        groups = [("stars", finite & s.isin([0, 1])), ("galaxies", finite & s.isin([2, 3]))]
        n_no_data = int((finite & (s == -9)).sum())
        note = (
            f"ext_coadd split: stars = ext_coadd IN (0,1), galaxies = ext_coadd IN (2,3); "
            f"{n_no_data} no-data (ext_coadd = -9) source(s) excluded from both panels."
            if n_no_data
            else None
        )
        return groups, note
    return (
        [("stars", finite & (s <= split_threshold)), ("galaxies", finite & (s > split_threshold))],
        None,
    )


def color_color_diagram(
    catalog, table, ra, dec, radius_deg, *,
    x_bands=("g", "r"), y_bands=("r", "i"),
    split_col=None, split_threshold=0.003, limit=3000, title=None,
    point_sources=False, morphology=None, value_cuts=None,
    x_expr=None, y_expr=None,
    client=None, result_store=None, plotting_service=None, owner_id=None,
):
    """P5: one-shot color-color diagram. Queries the cone, optionally splits into stars/galaxies
    by a morphology column (auto-detected from the registry, e.g. DES spread_model_r), and renders
    a 1- or 2-panel CCD as a single image. Sentinel magnitudes (99.99) are cut both server- and
    client-side so the axes stay physical. An explicit morphology cut overrides point_sources.
    x_expr/y_expr switch to derived axes (single panel, no star/galaxy split)."""
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
            owner_id=owner_id,
        )
    # A morphology-selected sample is one population — don't auto-split it into stars/galaxies.
    if split_col is None and not point_sources and not morphology:
        split_col = reg.morphology_split_column(catalog, table)

    bands = list(dict.fromkeys([*x_bands, *y_bands]))
    rid, result, magcols, _meta = _diagram_dataframe(
        catalog, table, ra, dec, radius_deg, bands=bands,
        extra_cols=[split_col] if split_col else [], limit=limit, client=client, result_store=result_store,
        point_sources=point_sources, morphology=morphology, extra_value_cuts=value_cuts,
        owner_id=owner_id,
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
    return _render_diagram(plotting, fig, "datalab_ccd", rid,
                           {**(getattr(result, "provenance", {}) or {}), "catalog": catalog, "table": table,
                            "auto_validity_filters": _meta.get("auto_validity_filters")},
                           {"rowcount": int(len(df)), "x": xl, "y": yl, "split_col": split_col,
                            "point_sources": ps_applied, "morphology": _meta.get("morphology"),
                            "populations": populations,
                            "warnings": list(_meta.get("warnings") or [])
                            + ([split_note] if split_note else []),
                            "excluded_invalid_mags": int(len(df) - int(finite.sum())),
                            "plotly_spec": plotly_spec})


def color_magnitude_diagram(
    catalog, table, ra, dec, radius_deg, *,
    blue_band="g", red_band="r", mag_band=None, limit=5000, title=None,
    point_sources=False, morphology=None, value_cuts=None,
    x_expr=None, y_expr=None, overlay_locus=None,
    client=None, result_store=None, plotting_service=None, owner_id=None,
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
            value_cuts=value_cuts, overlay_locus=overlay_locus,
            client=client, result_store=result_store, plotting=plotting,
            owner_id=owner_id,
        )
    mag_band = mag_band or blue_band
    bands = list(dict.fromkeys([blue_band, red_band, mag_band]))
    rid, result, magcols, _meta = _diagram_dataframe(
        catalog, table, ra, dec, radius_deg, bands=bands, extra_cols=[], limit=limit,
        client=client, result_store=result_store, point_sources=point_sources,
        morphology=morphology, extra_value_cuts=value_cuts, owner_id=owner_id,
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
    return _render_diagram(plotting, fig, "datalab_cmd", rid,
                           {**(getattr(result, "provenance", {}) or {}), "catalog": catalog, "table": table,
                            "auto_validity_filters": _meta.get("auto_validity_filters")},
                           {"rowcount": int(len(df)), "x": xl, "y": yl, "points": int(finite.sum()),
                            "point_sources": ps_applied, "morphology": _meta.get("morphology"),
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
