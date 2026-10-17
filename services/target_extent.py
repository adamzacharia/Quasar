"""Cone radius for an archive search sized from the source's catalogued extent.

ALMAbench 2026-10-15 A22: "all ALMA observations of M83 in Band 6" searched a
60 arcsec cone and found 22 of the 44 MOUS; mosaics of the disc and arms sit up
to 8 arcmin from the nucleus. SIMBAD's galaxy dimension (``galdim_majaxis``,
the D25-like major axis in arcmin) is 13.8 arcmin for M83, and 1.25 x the
semi-major axis (8.6 arcmin) holds all 44. Compact sources (HH 212, 0.01
arcmin) and sources without a catalogued size keep the caller's default.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

__all__ = ["extent_radius_arcsec", "major_axis_lookup", "catalogued_major_axis_arcmin", "EXTENT_MARGIN", "EXTENT_CAP_ARCSEC"]

EXTENT_MARGIN = 1.25          # x the semi-major axis: outer-disc and arm mosaics
EXTENT_CAP_ARCSEC = 1800.0    # 30 arcmin; M31 (199 arcmin) is capped and says so


_MAJOR_AXIS_CACHE: Dict[str, Optional[float]] = {}
LOOKUP_WALL_S = 8.0       # wall-clock cap on the whole SIMBAD lookup (field discovery + query)
RESERVE_S = 20.0          # tool budget left for the archive search the radius is for


def _query_major_axis(name: str) -> Optional[float]:
    """One SIMBAD lookup; raises on any transport or service failure."""
    import numpy as np
    from astroquery.simbad import Simbad

    simbad = Simbad()
    simbad.add_votable_fields("dim")
    table = simbad.query_object(name)
    if table is None or len(table) == 0:
        return None
    cols = {c.lower(): c for c in table.colnames}
    col = cols.get("galdim_majaxis") or cols.get("dim_majaxis")
    value = table[col][0] if col else None
    if value is None or np.ma.is_masked(value):
        return None
    try:
        major = float(value)
    except (TypeError, ValueError):
        return None
    return major if math.isfinite(major) and major > 0 else None


def major_axis_lookup(target: str) -> Tuple[str, Optional[float]]:
    """``("ok", arcmin or None)`` when SIMBAD answered (None = no catalogued
    size), ``("unavailable", None)`` when it failed or ran out of time, and
    ``("off", None)`` when QUASAR_TARGET_EXTENT_LOOKUP disables lookups.

    The whole lookup (astroquery's field discovery and the query) runs on a
    bounded worker (tool_budgets.call_bounded) for at most LOOKUP_WALL_S, with
    a deadline of its own that is cancelled if abandoned, and never into the last
    RESERVE_S of the tool budget: astroquery's Simbad ignores an uppercase
    TIMEOUT and its own timeout does not bound the synchronous request (guard
    CX-01). Answers are cached; failures are not, so an outage is retried."""
    name = str(target or "").strip()
    if not name:
        return "ok", None
    if os.getenv("QUASAR_TARGET_EXTENT_LOOKUP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return "off", None
    if name in _MAJOR_AXIS_CACHE:
        return "ok", _MAJOR_AXIS_CACHE[name]
    from services.tool_budgets import Deadline, TurnCancelled, adopt_deadline, call_bounded, current_deadline, remaining_seconds

    left = remaining_seconds()
    wall = LOOKUP_WALL_S if left is None else min(LOOKUP_WALL_S, left - RESERVE_S)
    if wall < 1.0:
        return "unavailable", None
    # call_bounded gives the worker a CHILD of the tool deadline (so the HTTP
    # hook clamps its requests) and cancels it when abandoned. Outside a tool
    # there is no parent, so the worker adopts its own deadline of the same
    # length, cancelled the same way (guard CX-11).
    own = Deadline(wall, label="SIMBAD size") if current_deadline() is None else None

    def work() -> Optional[float]:
        if own is not None:
            adopt_deadline(own)
        return _query_major_axis(name)

    try:
        major = call_bounded(work, wall, label="SIMBAD size", thread_name="quasar-simbad-size")
    except TurnCancelled:
        raise
    except Exception:  # noqa: BLE001 - timeout or any failure means "size unknown"
        if own is not None:
            own.cancel("SIMBAD size lookup abandoned")
        return "unavailable", None
    if len(_MAJOR_AXIS_CACHE) < 1024:
        _MAJOR_AXIS_CACHE[name] = major
    return "ok", major


def catalogued_major_axis_arcmin(target: str) -> Optional[float]:
    """SIMBAD ``galdim_majaxis`` (arcmin) for ``target``, or None (no size, unknown, or lookup unavailable)."""
    return major_axis_lookup(target)[1]


def extent_radius_arcsec(target: str, default_arcsec: float) -> Dict[str, Any]:
    """The cone radius to use for ``target`` when the caller gave none.

    Returns ``{"radius_arcsec", "basis", "major_axis_arcmin"}``: the larger of
    ``default_arcsec`` and EXTENT_MARGIN x the catalogued semi-major axis,
    capped at EXTENT_CAP_ARCSEC. The basis says when the size was unavailable
    rather than absent (guard CX-09).
    """
    status, major = major_axis_lookup(str(target or "").strip())
    if major is None:
        why = {"unavailable": "the SIMBAD size lookup did not complete, so an extended source may reach beyond it",
               "off": "source-size lookup is off"}.get(status, "no catalogued size for this source")
        return {"radius_arcsec": float(default_arcsec), "major_axis_arcmin": None,
                "basis": f"default {float(default_arcsec):g} arcsec cone ({why})"}
    sized = EXTENT_MARGIN * major / 2.0 * 60.0
    if sized <= default_arcsec:
        return {"radius_arcsec": float(default_arcsec), "major_axis_arcmin": major,
                "basis": f"default {float(default_arcsec):g} arcsec cone (source is {major:g} arcmin across)"}
    radius = min(sized, EXTENT_CAP_ARCSEC)
    basis = (f"{radius:.0f} arcsec cone = {EXTENT_MARGIN:g} x the semi-major axis of the SIMBAD size "
             f"({major:g} arcmin across), so the whole extended source is searched")
    if sized > EXTENT_CAP_ARCSEC:
        basis += f"; capped at {EXTENT_CAP_ARCSEC / 60:g} arcmin (the source is larger)"
    return {"radius_arcsec": radius, "major_axis_arcmin": major, "basis": basis}
