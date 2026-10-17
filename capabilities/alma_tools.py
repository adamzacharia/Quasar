"""One-shot ALMA-side tools that collapse a benchmark question to one call.

UI benchmark 2026-09-22 (tmp/ui-bench-2026-09-22/): every tool here is the
deterministic answer to a question the 8-round agent loop lost points on:

* ``alma_project_census``      D09 / D10 / D11 / D21 / D22 -- server-side per-project counts
* ``alma_source_summary``      D12 / D16 -- per-MOUS table with filters IN the ADQL + total count
* ``alma_public_band_status``  D08 -- live distinct public band tokens with counts, as of a date
* ``alma_archive_link``        D17 -- ASA deep link with documented parameter names, HEAD-verified
* ``alma_bibliography``        D20 -- ADS ``bibgroup:ALMA`` search with project codes from abstracts
* ``cross_archive_match``      D18 -- per-source POINT cones (never the N-way footprint query)
* ``archive_overlay``          D19 -- size-aware base image (MAST product or HiPS fallback) + ALMA contours
* ``code_recipe``              D13 / D14 / D15 -- curated, tested snippets (docs/recipes/*.py)
* ``alma_reference``           D01-D08 -- Handbook tables + documentation lookup with citations

Every result carries ``status`` (ok | partial | coverage_gap | infrastructure_failure),
``provenance`` (executed ADQL / URLs) and never invents numbers: a phase that
failed says so, per source / per archive.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlencode

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from capabilities.base import BaseCapability, CallContext, ToolResult
from services.alma_science_queries import (
    LINE_REST_FREQ_GHZ,
    OBSCORE_BASE_COLUMNS,
    aggregate_counts,
    alma_cone_adql,
    alma_cone_count_adql,
    alma_cone_funnel_adql,
    as_text,
    band_tokens,
    counts_note,
    normalize_target_alias,
    parse_frequency_support_intervals,
    public_band_inventory_query,
    summarize_mous,
)

__all__ = ["CAPABILITIES"]

_ALMA_TAP_URL = "https://almascience.nrao.edu/tap"
_ASA_BASE = "https://almascience.nrao.edu/aq/"


def _native(out: Dict[str, Any]) -> ToolResult:
    return ToolResult(success=bool(out.get("success")), error=(None if out.get("success") else out.get("error")), native=out)


def _norm_name(name: Any) -> str:
    """Source / field name reduced to lowercase letters and digits ('UDF_mosaic_1mm' -> 'udfmosaic1mm')."""
    return re.sub(r"[^a-z0-9]", "", as_text(name).lower())


def _source_key(name: Any) -> str:
    """Source name for matching an ObsCore target_name to a product filename, compared for EQUALITY: lowercase, signs,
    dots and hyphens kept (J1234+0100 and J1234-0100 stay distinct), spaces, underscores and other punctuation
    dropped, since product filenames write 'NGC 1068' as 'NGC1068' or 'NGC_1068' (guard CX-12)."""
    return re.sub(r"[^a-z0-9+.\-]", "", as_text(name).lower())


# What a redshift census can and cannot say (ALMAbench 2026-10-15 A29: answers gave the census count and the
# formula but never said how to get from coverage to galaxies at that redshift).
_REDSHIFT_METHOD_NOTE = (
    "ivoa.obscore has no redshift column: these projects have spectral windows that WOULD cover the line at some z in "
    "the range, not targets with a measured redshift there. To list galaxies at that redshift, cross-match the ALMA "
    "pointings by position with a redshift catalogue (NED, or the survey catalogue of the field), prefer spectroscopic "
    "redshifts (a photometric error can move the line out of the window), and for each galaxy compute nu_obs = "
    "nu_rest/(1+z) per transition (allowing for the redshift error and line width) and check it against each spectral "
    "window in frequency_support, not the aggregate band range. band_mapping gives the ALMA band each transition falls "
    "in across the requested z range."
)


# Caveats every census answer can quote (ObsCore semantics, not facts about any one query).
_CENSUS_CAVEATS_COMMON = [
    "ivoa.obscore rows are finer than datasets (one per spectral window and execution block), so counts here are "
    "DISTINCT member_ous_uid (datasets) grouped by proposal_id (projects), never raw rows.",
    "A cycle is selected by project-code prefix (main call YYYY.1, supplemental call YYYY.2, DDT YYYY.A; see the "
    "warnings for which were included); the archive only lists data that exist, so projects not yet observed are absent.",
]
_CENSUS_CAVEATS = {
    "solar": [
        "The Sun is an ephemeris (moving) target, so a position or cone search cannot find solar data; select by the "
        "ALMA scientific category 'Sun'.",
        "Free-text matching on 'sun' or 'solar' gives false hits (Sunyaev-Zel'dovich projects, 'Solar system' bodies, "
        "names containing the letters), so text matches are only a word-anchored fallback here.",
    ],
    "arrays": [
        "Array per dataset comes from the scheduling-block suffix or the antenna names; a PM (Total Power) antenna that "
        "appears in a 12m or 7m antenna list does not make the data Total Power (see array_definition).",
    ],
    "lines": [
        "Coverage is decided per spectral window (frequency_support), not from the overall frequency range; covering a "
        "line is not a detection, and science keywords are chosen by the PI.",
        "The same object can appear under different target names; group by project and target position before "
        "counting objects.",
    ],
}


def _census_reproduce(kind: str, cycle: Optional[int], *, science_only: bool = True, public_only: bool = False) -> Dict[str, str]:
    """A short query a user can run themselves for the SAME population the census counted (every project-code
    prefix of the cycle, the same science/public filters; guard CX-06), and the ASA web-interface equivalent (checked
    in the ASA on 2026-10-16: projectCode=2021.1.*%7C2021.2.*%7C2021.A.*&observationsScientificCategory=Sun filtered
    the Observations table to 8 rows; a raw '|' in the URL is rejected with HTTP 400; the Projects tab count ignores
    column filters)."""
    if kind != "solar" or cycle is None:
        return {}
    from services.alma_science_queries import cycle_project_prefixes

    prefixes = cycle_project_prefixes(int(cycle))
    codes = " OR ".join(f"proposal_id LIKE '{p}%'" for p in prefixes)
    filters = (" AND science_observation = 'T'" if science_only else "") + (" AND data_rights = 'Public'" if public_only else "")
    asa_codes = "|".join(f"{p}*" for p in prefixes)
    return {
        "simple_adql": f"SELECT DISTINCT proposal_id FROM ivoa.obscore WHERE scientific_category = 'Sun' AND ({codes}){filters}",
        "asa_web": (f"In the ASA set Project code = {asa_codes} ('|' = or) and filter the Observations table's Scientific "
                    f"category column to Sun (https://almascience.org/aq/?projectCode={asa_codes.replace('|', '%7C')}"
                    f"&observationsScientificCategory=Sun{'&publicData=true' if public_only else ''}); count distinct "
                    "project codes in the Observations table, because the Projects tab count ignores column filters. "
                    "The census also accepts word-anchored Sun/solar text matches, so its count can be slightly larger."),
    }


def _redshift_band_mapping(windows: List[Dict[str, Any]], z_min: float, z_max: float) -> List[Dict[str, Any]]:
    """For each transition window, the ALMA receiver band(s) its observed frequency falls in and the z sub-range for
    each, from the reference band table (nominal edges)."""
    try:
        from services.alma_reference import load_reference_tables

        bands = load_reference_tables()["bands"]
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for w in windows:
        nu = float(w.get("rest_frequency_ghz") or 0)
        if nu <= 0:
            continue
        parts = []
        for b in bands:
            lo, hi = float(b["frequency_min_ghz"]), float(b["frequency_max_ghz"])
            z_lo, z_hi = max(z_min, nu / hi - 1.0), min(z_max, nu / lo - 1.0)
            if z_lo <= z_hi:
                parts.append({"band": int(b["band"]), "z_min": round(z_lo, 3), "z_max": round(z_hi, 3)})
        out.append({"transition": w.get("transition"), "rest_frequency_ghz": round(nu, 4),
                    "observed_ghz": [round(nu / (1 + z_max), 2), round(nu / (1 + z_min), 2)],
                    "bands": parts or "outside the ALMA bands over this z range"})
    return out


# Names a field's ALMA pointings go by (normalised). The HUDF is "HUDF" in the
# Dunlop et al. mosaic but "UDF_mosaic_1mm" / "UDF_mosaic_3mm" in ASPECS.
_UDF_NAMES = {"hudf", "udf", "hubbleultradeepfield"}
_FIELD_NAME_ALIASES = {"hudf": _UDF_NAMES, "hubbleultradeepfield": _UDF_NAMES, "ultradeepfield": _UDF_NAMES}


def _sep_arcmin(ra: float, dec: float, ras: pd.Series, decs: pd.Series) -> pd.Series:
    """Great-circle separation (arcmin) from (ra, dec) to each (ras, decs), all in degrees."""
    r0, d0 = math.radians(ra), math.radians(dec)
    r1, d1 = ras.map(math.radians), decs.map(math.radians)
    cos = (d1.map(math.sin) * math.sin(d0) + d1.map(math.cos) * math.cos(d0) * (r1 - r0).map(math.cos)).clip(-1.0, 1.0)
    return cos.map(lambda c: math.degrees(math.acos(c)) * 60.0)


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _run_adql_factory(ctx: CallContext, executed: List[str]):
    """Budgeted TAP runner recording every query into ``executed`` and the
    request-scoped ALMA provenance (Show-query panel)."""
    from capabilities.alma import _server_side_runner

    prov_state = ctx.service("alma_tap_provenance")
    return _server_side_runner(ctx, prov_state, executed)


def _infra_failure(out: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    from services.host_breaker import HostCircuitOpen, classify_infrastructure_error

    if isinstance(exc, HostCircuitOpen):
        out.update(status="infrastructure_failure", infrastructure_failure=True, circuit_breaker=True,
                   host=exc.host, retry_after_s=int(round(exc.retry_after)))
    elif classify_infrastructure_error(exc) is not None:
        out.update(status="infrastructure_failure", infrastructure_failure=True)
    return out


def _data_card(ctx: CallContext, frame: pd.DataFrame, *, source: str, tool_name: str, label: str = "",
               warnings: Optional[List[str]] = None, partial: bool = False) -> None:
    try:
        ctx.service("set_last_search_results")(frame)
        ctx.service("set_last_run_result")({
            "type": "data", "data": frame, "source": source, "filter_label": label or source,
            "tool_name": tool_name, "warnings": list(warnings or []), "partial": bool(partial),
        })
    except Exception:  # cards are transport; never fail the tool over them
        pass


# ═════════════════════════════════════════════════════════════════════════
# 1. alma_project_census
# ═════════════════════════════════════════════════════════════════════════
class ProjectCensusConstraint(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = Field(description="solar | arrays | lines | redshift | bandwidth_switching")
    arrays: Optional[List[str]] = Field(default=None, description="for type=arrays: e.g. ['12m','7m','TP']")
    lines: Optional[List[str]] = Field(default=None, description="for type=lines: e.g. ['12CO','13CO','C18O']")
    band: Any = Field(default=None, description="for type=lines: ALMA band(s) the lines must sit in")
    zmin: Optional[float] = Field(default=None, description="for type=redshift")
    zmax: Optional[float] = Field(default=None, description="for type=redshift")
    species: Optional[str] = Field(default="CO", description="for type=redshift: rest-frame species (CO ladder default)")
    science_category: Optional[str] = Field(default="", description="for type=redshift: override the extragalactic category filter")


class ProjectCensusInput(_In):
    cycle: Optional[int] = Field(default=None, description="ALMA cycle (0-13). Required for solar/arrays; optional otherwise (whole archive).")
    constraint: ProjectCensusConstraint
    public_only: bool = False
    science_only: bool = True


class AlmaProjectCensus(BaseCapability):
    name = "alma_project_census"
    description = (
        "ONE-CALL per-project census of the ALMA Science Archive computed AT THE SERVER (GROUP BY / DISTINCT / HAVING), "
        "never a TOP-capped row pull: which projects of a cycle observed the Sun (constraint.type='solar'); used ALL of "
        "the 12m, 7m and Total Power arrays ('arrays'); cover a set of lines such as 12CO/13CO/C18O in the SAME MOUS "
        "('lines' + band); have science spectral windows coverage-compatible with a species at z in [zmin, zmax] "
        "('redshift'); or have narrow setups below the Handbook bandwidth-switching threshold ('bandwidth_switching'). "
        "Returns a per-project table with MOUS/EB counts, a complete/partial flag with the cycles scanned, the exact ADQL, "
        "rest frequencies / coverage windows, and a headline count you can quote. Science scans only by default "
        "(calibrators excluded). Target <= 60 s."
    )
    category = "archive"
    InputModel = ProjectCensusInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        from services import alma_server_side as ss

        c = inp.constraint
        kind = str(c.type or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {"sun": "solar", "array": "arrays", "array_combo": "arrays", "line": "lines", "line_set": "lines",
                   "z": "redshift", "redshifted": "redshift", "bwsw": "bandwidth_switching", "bandwidth": "bandwidth_switching"}
        kind = aliases.get(kind, kind)
        extra = "data_rights = 'Public'" if inp.public_only else ""
        executed: List[str] = []
        started = time.perf_counter()
        try:
            run_query = _run_adql_factory(ctx, executed)
            if kind == "solar":
                if inp.cycle is None:
                    return _native({"success": False, "error": "cycle is required for a solar census"})
                res = ss.solar_projects_server_side(run_query, int(inp.cycle), science_only=inp.science_only, extra_where=extra)
                headline = f"{len(res.frame)} project(s) with science observations of the Sun in Cycle {inp.cycle}"
            elif kind == "arrays":
                if inp.cycle is None:
                    return _native({"success": False, "error": "cycle is required for an array-combination census"})
                res = ss.array_combo_projects_server_side(run_query, int(inp.cycle), c.arrays or ["12m", "7m", "TP"],
                                                          science_only=inp.science_only, extra_where=extra)
                headline = (f"{res.extras.get('matching_projects', len(res.frame))} project(s) used ALL of "
                            f"{', '.join(c.arrays or ['12m', '7m', 'TP'])} in Cycle {inp.cycle} "
                            f"(of {res.extras.get('projects_in_window')} projects in the cycle window)")
            elif kind == "lines":
                if not c.lines:
                    return _native({"success": False, "error": "constraint.lines is required for a line-set census"})
                res = ss.line_set_projects_server_side(run_query, c.lines, band=c.band, science_only=inp.science_only,
                                                       cycle=inp.cycle, topic_filter="", extra_where=extra)
                headline = (f"{len(res.frame)} project(s) have at least one MOUS covering every requested line "
                            f"({', '.join(res.extras.get('lines', []))}); {res.n_units} MOUS in total")
            elif kind == "redshift":
                if c.zmin is None or c.zmax is None:
                    return _native({"success": False, "error": "constraint.zmin and zmax are required for a redshift census"})
                res = ss.redshifted_line_projects_server_side(
                    run_query, rest_species=c.species or "CO", z_min=float(c.zmin), z_max=float(c.zmax),
                    science_category=str(c.science_category or ""), cycle=inp.cycle, extra_where=extra,
                )
                headline = (f"{len(res.frame)} project(s) with science spectral windows coverage-compatible with "
                            f"{c.species or 'CO'} at z={float(c.zmin):g}-{float(c.zmax):g} (spectral coverage, NOT measured redshifts)")
            elif kind == "bandwidth_switching":
                res = ss.bandwidth_switching_server_side(run_query, cycle=inp.cycle, extra_where=extra)
                headline = (f"{len(res.frame)} project(s) have science MOUS below the bandwidth-switching threshold "
                            f"({res.extras.get('threshold_mhz'):g} MHz aggregate bandwidth); {res.n_units} candidate MOUS")
            else:
                return _native({"success": False, "error": f"unknown constraint.type {c.type!r}; use solar | arrays | lines | redshift | bandwidth_switching"})
        except Exception as exc:
            out = {"success": False, "status": "infrastructure_failure", "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                   "executed_queries": executed[-3:], "elapsed_s": round(time.perf_counter() - started, 1)}
            return _native(_infra_failure(out, exc))

        frame = res.frame
        warnings = list(res.notes) + [("Completeness: " if res.complete else "INCOMPLETE: ") + res.completeness_note()]
        redshift_extras: Dict[str, Any] = {}
        if kind == "redshift":
            redshift_extras = {"band_mapping": _redshift_band_mapping(res.extras.get("windows") or [], float(c.zmin), float(c.zmax)),
                               "method_note": _REDSHIFT_METHOD_NOTE}
        _data_card(ctx, frame, source=f"ALMA project census ({kind})", tool_name=self.name,
                   label=f"{kind}" + (f" Cycle {inp.cycle}" if inp.cycle is not None else " (whole archive)"),
                   warnings=warnings, partial=not res.complete)
        out = {
            "success": True,
            "status": res.status,
            "constraint": kind,
            "cycle": inp.cycle,
            "headline": headline,
            "n_projects": int(frame["proposal_id"].nunique()) if not frame.empty and "proposal_id" in frame else 0,
            "n_units": res.n_units,
            "unit": res.unit,
            "complete": res.complete,
            "completeness": res.completeness_note(),
            "scanned": res.scanned,
            "not_scanned": res.unscanned,
            "method": res.extras.get("method"),
            "results": frame.head(60).to_dict("records") if not frame.empty else [],
            "results_returned": min(60, len(frame)),
            "results_truncated": len(frame) > 60,
            "warnings": warnings,
            "provenance": {"archive": "ALMA Science Archive", "tap_url": _ALMA_TAP_URL, "adql": res.queries,
                           "executed_queries": len(res.queries), "elapsed_s": round(res.elapsed_s, 1),
                           "computation": "server-side"},
            "note": (
                "Counts are computed at the archive server; 'partial' means some cycles were not scanned within the "
                "tool budget and counts are lower bounds. The full table is in the UI data card."
            ),
            "method_caveats": _CENSUS_CAVEATS.get(kind, []) + _CENSUS_CAVEATS_COMMON,
            **({"reproduce": _census_reproduce(kind, inp.cycle, science_only=inp.science_only, public_only=inp.public_only)}
               if kind == "solar" and inp.cycle is not None else {}),
            **redshift_extras,
        }
        for key in ("rest_frequencies_ghz", "lines", "mous_per_line", "distinct_objects_by_position", "windows", "threshold_mhz", "citation",
                    "projects_in_window", "per_array_project_counts", "matching_projects", "matching_project_ids",
                    "alternative_counts", "array_definition", "cycles_scanned"):
            if key in res.extras:
                out[key] = res.extras[key]
        return _native(out)


# ═════════════════════════════════════════════════════════════════════════
# 2. alma_source_summary
# ═════════════════════════════════════════════════════════════════════════
class SourceSummaryInput(_In):
    target: Optional[str] = Field(default=None, description="Target name (SIMBAD-resolvable), or give ra/dec")
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: Optional[float] = None  # None: sized from the source's catalogued extent, at least 60 arcsec
    released_before: Optional[str] = None  # ISO date: keep rows released before it (filter in the ADQL)
    bands: Optional[List[int]] = Field(default=None, description="ALMA bands to keep (in the ADQL), e.g. [7]")
    max_resolution_arcsec: Optional[float] = Field(default=None, description="keep MOUS with spatial_resolution <= this (arcsec)")
    public_only: bool = False
    science_only: bool = True
    max_rows: int = 2000


_EXTENT_CONE_MODE = (
    "fields overlapping the cone, computed at the archive: pointings whose field (s_fov/2) reaches the cone, or whose "
    "footprint contains the source position (field footprints treated as circles of diameter s_fov, inclusive by 2 "
    "arcsec). Complete for fields up to 1 degree across; a larger mosaic centred more than 30 arcmin outside the cone "
    "would not be found.")


class AlmaSourceSummary(BaseCapability):
    name = "alma_source_summary"
    description = (
        "ONE-CALL per-MOUS summary of the ALMA data on a source: band, spatial resolution, aggregate bandwidth, continuum "
        "sensitivity, integration time, release date, data rights, PI and project, with the band/resolution/public/science "
        "filters applied IN the ADQL (so a row cap never hides matching data) and a server-side total count when the cap "
        "is hit, and the band frequency edges. Omit radius_arcsec to search the source's whole catalogued extent; "
        "released_before=YYYY-MM-DD for 'released before'. Target <= 30 s."
    )
    category = "archive"
    InputModel = SourceSummaryInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        started = time.perf_counter()
        if inp.released_before:
            try:
                _dt.date.fromisoformat(str(inp.released_before).strip()[:10])
            except ValueError:
                # Caller input, not an archive outage (guard CX-07): no query is sent.
                return _native({"success": False, "status": "invalid_input",
                                "error": f"released_before must be an ISO date (YYYY-MM-DD), got {inp.released_before!r}"})
        prov_state = ctx.service("alma_tap_provenance")
        try:
            ra, dec, label = inp.ra, inp.dec, (inp.target or "").strip()
            if (ra is None) != (dec is None):
                return _native({"success": False, "error": "give BOTH ra and dec, or a target name"})
            if ra is None:
                if not label:
                    return _native({"success": False, "error": "target or ra/dec is required"})
                resolved = ctx.service("resolve_target")(normalize_target_alias(label))
                ra, dec = resolved.get("ra_deg"), resolved.get("dec_deg")
                if ra is None or dec is None:
                    return _native({"success": False, "status": "coverage_gap", "error": f"could not resolve target {label!r}"})
            radius_basis = None
            if inp.radius_arcsec is not None:
                radius_arcsec = float(inp.radius_arcsec)
            elif label:
                from services.target_extent import extent_radius_arcsec

                sized = extent_radius_arcsec(label, 60.0)
                radius_arcsec, radius_basis = float(sized["radius_arcsec"]), sized["basis"]
            else:
                radius_arcsec = 60.0
            label = label or f"RA={float(ra):.5f}, Dec={float(dec):.5f}"
            radius_deg = max(1.0, radius_arcsec) / 3600.0
            top = max(50, min(int(inp.max_rows), 20000))
            # A cone sized from an extended source's extent is searched on pointing centres plus footprints that
            # contain the source position: the full s_region INTERSECTS form took 54 s for an 8.6-arcmin M83 cone
            # (ReadTimeouts in the 2026-10-16 re-run); this form 34 s, same 44 MOUS (guard CX-04).
            footprint: Any = "contains" if (radius_basis and radius_arcsec > 120.0) else True
            kw = dict(public=bool(inp.public_only), band=inp.bands or None,
                      max_resolution_arcsec=inp.max_resolution_arcsec, science_only=bool(inp.science_only),
                      released_before=inp.released_before or None, footprint=footprint)
            query = alma_cone_adql(float(ra), float(dec), radius_deg, top=top, columns=OBSCORE_BASE_COLUMNS, **kw)
            prov_state["query"] = query
            prov_state["url"] = _ALMA_TAP_URL
            service = ctx.service("search_service").alminer_client._get_tap_service()
            try:
                res = service.search(query, maxrec=top)
            except TypeError:
                res = service.search(query)
            prov_state["url"] = getattr(res, "quasar_tap_url", _ALMA_TAP_URL)
            df = res.to_table().to_pandas()
            for column in df.select_dtypes(include=["object"]):
                df[column] = df[column].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
            truncated = len(df) >= top or getattr(res, "query_status", "") == "OVERFLOW"
            total_rows = total_mous = None
            from services.tool_budgets import remaining_seconds as _remaining

            _left = _remaining()
            if truncated and (_left is None or _left >= 8.0):
                count_q = alma_cone_count_adql(float(ra), float(dec), radius_deg, **kw)
                prov_state["query"] = query + "\n\n" + count_q
                try:
                    cres = service.search(count_q, maxrec=5)
                except TypeError:
                    cres = service.search(count_q)
                cdf = cres.to_table().to_pandas()
                if not cdf.empty:
                    total_rows, total_mous = int(cdf.iloc[0]["total_rows"]), int(cdf.iloc[0]["total_mous"])
            before_cuts = self._before_quality_cuts(service, float(ra), float(dec), radius_deg, kw, prov_state, query)
        except Exception as exc:
            out = {"success": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "status": "infrastructure_failure"}
            return _native(_infra_failure(out, exc))

        band_edges = self._band_edges(inp.bands)
        if df.empty:
            out = {
                "success": True, "status": "coverage_gap", "target": label, "ra": float(ra), "dec": float(dec),
                "radius_arcsec": round(radius_arcsec, 1), **({"radius_basis": radius_basis} if radius_basis else {}),
                "n_mous": 0, "n_rows": 0, "filters_in_adql": {k: v for k, v in kw.items() if k != "footprint" and v not in (None, False)},
                **({"before_quality_cuts": before_cuts} if before_cuts else {}),
                "cone_mode": (_EXTENT_CONE_MODE if kw.get("footprint") == "contains" else "footprint (s_region) or pointing centre"),
                "band_frequency_edges": band_edges,
                "note": "No ALMA rows match this cone with these filters (the query completed; this is a real empty result).",
                "provenance": {"tap_url": prov_state.get("url"), "adql": query, "elapsed_s": round(time.perf_counter() - started, 1)},
            }
            _data_card(ctx, df, source=f"ALMA source summary: {label}", tool_name=self.name)
            return _native(out)

        mous = summarize_mous(df)
        # Per-MOUS aggregate bandwidth (sum of distinct SPW widths) and continuum sensitivity / exptime extremes.
        agg_bw: Dict[str, float] = {}
        sens: Dict[str, float] = {}
        texp: Dict[str, float] = {}
        rel: Dict[str, str] = {}
        rights: Dict[str, str] = {}
        arrays_of: Dict[str, str] = {}
        from services.alma_science_queries import classify_array_row

        for uid, g in df.groupby(df["member_ous_uid"].map(as_text)):
            labels: List[str] = []
            for sb, ants in zip(g.get("schedblock_name", pd.Series([""] * len(g))), g.get("antenna_arrays", pd.Series([""] * len(g)))):
                for lab in classify_array_row(ants, sb):
                    if lab not in labels:
                        labels.append(lab)
            arrays_of[uid] = "+".join(labels)
            widths = set()
            for fs in g.get("frequency_support", pd.Series(dtype=str)).dropna().astype(str).unique():
                for lo, hi in parse_frequency_support_intervals(fs):
                    widths.add((round(lo, 4), round(hi, 4)))
            if widths:
                agg_bw[uid] = round(sum(hi - lo for lo, hi in widths) * 1e3, 1)
            else:
                bw = pd.to_numeric(g.get("bandwidth"), errors="coerce").dropna()
                freqs = pd.to_numeric(g.get("frequency"), errors="coerce").dropna()
                if not bw.empty:
                    agg_bw[uid] = round(float(bw.groupby(freqs.round(4) if len(freqs) == len(bw) else bw.index).first().sum()) / 1e6, 1)
            s = pd.to_numeric(g.get("cont_sensitivity_bandwidth"), errors="coerce").dropna()
            if not s.empty:
                sens[uid] = round(float(s.min()), 4)
            t = pd.to_numeric(g.get("t_exptime"), errors="coerce").dropna()
            if not t.empty:
                texp[uid] = round(float(t.max()), 1)
            r = [as_text(v) for v in g.get("obs_release_date", pd.Series(dtype=str)) if as_text(v)]
            if r:
                rel[uid] = min(r)[:10]
            dr = {as_text(v).lower() for v in g.get("data_rights", pd.Series(dtype=str)) if as_text(v)}
            if dr:
                rights[uid] = "/".join(sorted(dr))
        mous["member_ous_uid"] = mous["member_ous_uid"].map(as_text)
        table = pd.DataFrame({
            "member_ous_uid": mous["member_ous_uid"],
            "proposal_id": mous.get("proposal_id", ""),
            "pi_name": mous.get("pi_name", ""),
            "target_name": mous.get("target_name", ""),
            "band_list": mous.get("band_list", ""),
            "array": mous["member_ous_uid"].map(arrays_of),
            "spatial_resolution_arcsec": pd.to_numeric(mous.get("spatial_resolution"), errors="coerce").round(3),
            "aggregate_bandwidth_mhz": mous["member_ous_uid"].map(agg_bw),
            "cont_sensitivity_mjy_beam": mous["member_ous_uid"].map(sens),
            "t_exptime_s": mous["member_ous_uid"].map(texp),
            "obs_release_date": mous["member_ous_uid"].map(rel),
            "data_rights": mous["member_ous_uid"].map(rights),
            "science_observation": mous.get("science_observation", ""),
            "qa2_passed": mous.get("qa2_passed", ""),
            "n_rows": mous.get("rows", 0),
        }).sort_values(["spatial_resolution_arcsec", "t_exptime_s"], ascending=[True, False], na_position="last").reset_index(drop=True)
        counts = aggregate_counts(df)
        warnings: List[str] = []
        if truncated:
            warnings.append(
                f"Row cap {top} reached: the archive holds {total_rows} matching rows ({total_mous} MOUS) for this cone and filters; "
                f"showing {len(df)} rows / {len(table)} MOUS. Say 'N rows total, showing M'."
                if total_rows is not None else f"Row cap {top} reached; more rows exist."
            )
        _data_card(ctx, table, source=f"ALMA source summary: {label}", tool_name=self.name,
                   label=f"{label}" + (f" Band {','.join(map(str, inp.bands))}" if inp.bands else ""), warnings=warnings, partial=truncated)
        out = {
            "success": True,
            "status": "partial" if truncated else "ok",
            "target": label, "ra": float(ra), "dec": float(dec), "radius_arcsec": round(radius_arcsec, 1),
            **({"radius_basis": radius_basis} if radius_basis else {}),
            "filters_in_adql": {k: v for k, v in kw.items() if k != "footprint" and v not in (None, False)},
            "cone_mode": (_EXTENT_CONE_MODE if kw.get("footprint") == "contains" else "footprint (s_region) or pointing centre"),
            "n_mous": int(len(table)), "n_rows": int(len(df)),
            # Every MOUS id, compactly: 'find all observations' answers must list the datasets, and results stop at 40
            # rows (ALMAbench 2026-10-15 A22: 44 MOUS, answers gave the count only or the first 40). Placed before the
            # verbose rows so a truncated tool result still carries the whole list.
            "mous_list": [f"{r.member_ous_uid} {r.proposal_id} {as_text(r.obs_release_date)[:10]} {as_text(r.array)} "
                          f"{'' if pd.isna(r.spatial_resolution_arcsec) else str(r.spatial_resolution_arcsec) + chr(34)}".strip()
                          for r in table.head(500).itertuples()],
            "n_projects": int(table["proposal_id"].map(as_text).nunique()),
            **({"before_quality_cuts": before_cuts} if before_cuts else {}),
            "by_array": self._by_array(table),
            "total_rows_in_archive": total_rows, "total_mous_in_archive": total_mous,
            "counts": counts, "count_summary": counts_note(counts),
            "band_frequency_edges": band_edges,
            "best_resolution_arcsec": float(table["spatial_resolution_arcsec"].min()) if table["spatial_resolution_arcsec"].notna().any() else None,
            "results": table.head(40).to_dict("records"),
            "results_returned": min(40, len(table)),
            "warnings": warnings,
            "provenance": {"tap_url": prov_state.get("url"), "adql": prov_state.get("query"), "elapsed_s": round(time.perf_counter() - started, 1)},
            "note": ("One row per MOUS (dataset). aggregate_bandwidth_mhz sums the distinct spectral windows; cont_sensitivity "
                     "is the archive's estimate (mJy/beam). Full table in the UI data card. When the user asks to find or list "
                     "observations, list every MOUS in mous_list (uid, project, release date, array, resolution), not only "
                     "the count; by_array splits the counts by array."),
        }
        return _native(out)

    @staticmethod
    def _before_quality_cuts(service, ra: float, dec: float, radius_deg: float, kw: Dict[str, Any],
                             prov_state: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Counts for the same cone and scope without the user's resolution limit, so the answer can give
        the funnel (how many datasets the target has in this band, and how many pass the user's cut). None when no
        cut is set, the turn is short of time, or the count fails (the main result stands on its own)."""
        from services.tool_budgets import remaining_seconds as _remaining

        funnel_q = alma_cone_funnel_adql(ra, dec, radius_deg, **kw)
        left = _remaining()
        if not funnel_q or (left is not None and left < 8.0):
            return None
        try:
            try:
                fres = service.search(funnel_q, maxrec=5)
            except TypeError:
                fres = service.search(funnel_q)
            fdf = fres.to_table().to_pandas()
            n_mous, n_projects = int(fdf.iloc[0]["n_mous"]), int(fdf.iloc[0]["n_projects"])
        except Exception as exc:  # noqa: BLE001 - optional context
            print(f"[ALMA SOURCE SUMMARY] before-cuts count failed (non-fatal): {type(exc).__name__}: {str(exc)[:120]}")
            return None
        prov_state["query"] = (prov_state.get("query") or query) + "\n\n" + funnel_q
        cuts = [k for k in ("max_resolution_arcsec",) if kw.get(k) not in (None, False)]
        return {
            "cuts_removed": cuts,
            "n_mous": n_mous,
            "n_projects": n_projects,
            "note": ("Counts for the same target, band and scope BEFORE the quality cuts listed in cuts_removed. "
                     "Report both: how many datasets (MOUS) and projects the target has, then how many pass the cut."),
        }

    @staticmethod
    def _by_array(table: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """MOUS / project counts and the resolution range per array (12m, 7m, TP, or a '+' mix)."""
        out: Dict[str, Dict[str, Any]] = {}
        for arr, g in table.groupby(table["array"].map(lambda v: as_text(v) or "unknown")):
            res = pd.to_numeric(g["spatial_resolution_arcsec"], errors="coerce").dropna()
            out[arr] = {"n_mous": int(len(g)), "n_projects": int(g["proposal_id"].map(as_text).nunique()),
                        "resolution_arcsec_range": [float(res.min()), float(res.max())] if not res.empty else None}
        return out

    @staticmethod
    def _band_edges(bands: Optional[List[int]]) -> List[Dict[str, Any]]:
        try:
            from services.alma_reference import alma_reference_table

            table = alma_reference_table("bands")
            rows = table.get("bands", [])
            keep = [r for r in rows if not bands or int(r.get("band", -1)) in set(int(b) for b in bands)]
            return [{"band": r.get("band"), "frequency_min_ghz": r.get("frequency_min_ghz"), "frequency_max_ghz": r.get("frequency_max_ghz")} for r in keep]
        except Exception:
            return []


# ═════════════════════════════════════════════════════════════════════════
# 3. alma_public_band_status
# ═════════════════════════════════════════════════════════════════════════
class PublicBandStatusInput(_In):
    as_of: Optional[str] = Field(default=None, description="ISO date (YYYY-MM-DD): count only data released on or before this date")


class AlmaPublicBandStatus(BaseCapability):
    name = "alma_public_band_status"
    description = (
        "LIVE archive-state answer to 'how many ALMA bands have public data (as of a date)': the distinct band_list "
        "tokens of publicly released rows, per-band counts of public MOUS and projects, the query date and the exact "
        "ADQL, plus the nominal receiver-band table for context. Never answer archive-holding questions from memory."
    )
    category = "archive"
    InputModel = PublicBandStatusInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        started = time.perf_counter()
        executed: List[str] = []
        as_of = None
        if inp.as_of:
            try:
                as_of = _dt.date.fromisoformat(str(inp.as_of)[:10]).isoformat()
            except ValueError:
                return _native({"success": False, "error": f"as_of must be an ISO date (YYYY-MM-DD), got {inp.as_of!r}"})
        try:
            run_query = _run_adql_factory(ctx, executed)
            where = "data_rights = 'Public' AND band_list IS NOT NULL"
            if as_of:
                nxt = (_dt.date.fromisoformat(as_of) + _dt.timedelta(days=1)).isoformat()
                where += f" AND obs_release_date < '{nxt}'"
            agg = (
                "SELECT band_list, COUNT(DISTINCT member_ous_uid) AS n_mous, COUNT(DISTINCT proposal_id) AS n_projects, COUNT(*) AS n_rows "
                f"FROM ivoa.obscore WHERE {where} GROUP BY band_list"
            )
            df = run_query(agg)
            distinct_q = public_band_inventory_query(as_of)
            executed.append("-- equivalent distinct-token query: " + distinct_q.strip())
        except Exception as exc:
            out = {"success": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "status": "infrastructure_failure"}
            return _native(_infra_failure(out, exc))
        per_band: Dict[str, Dict[str, int]] = {}
        for _, row in df.iterrows():
            for token in band_tokens(row.get("band_list")):
                if not token.isdigit():
                    continue
                slot = per_band.setdefault(token, {"n_mous": 0, "n_projects": 0, "n_rows": 0})
                slot["n_mous"] += int(row.get("n_mous") or 0)
                slot["n_projects"] += int(row.get("n_projects") or 0)
                slot["n_rows"] += int(row.get("n_rows") or 0)
        bands_sorted = sorted(per_band, key=int)
        multi = sorted({as_text(v) for v in df["band_list"] if len(band_tokens(v)) > 1}) if not df.empty else []
        table = pd.DataFrame([{"band": int(b), **per_band[b]} for b in bands_sorted])
        reference = {}
        without: List[Dict[str, Any]] = []
        try:
            from services.alma_reference import alma_reference_table

            ref = alma_reference_table("bands")
            reference = {"receiver_band_count": ref.get("receiver_band_count"),
                         "bands": [{"band": r.get("band"), "frequency_min_ghz": r.get("frequency_min_ghz"), "frequency_max_ghz": r.get("frequency_max_ghz")} for r in ref.get("bands", [])],
                         "verified_on": ref.get("verified_on"), "sources": ref.get("sources"), "notes": ref.get("notes")}
            without = self._bands_without_public_data(ref, {int(b) for b in bands_sorted})
        except Exception:
            pass
        _data_card(ctx, table, source="ALMA public data by band" + (f" (released by {as_of})" if as_of else ""), tool_name=self.name)
        out = {
            "success": True, "status": "ok",
            "as_of": as_of or _dt.date.today().isoformat(),
            "query_date": _dt.date.today().isoformat(),
            "bands_with_public_data": [int(b) for b in bands_sorted],
            "n_bands_with_public_data": len(bands_sorted),
            "per_band": {int(b): per_band[b] for b in bands_sorted},
            "band_to_band_combinations": multi,
            "bands_without_public_data": without,
            "reference": reference,
            "results": table.to_dict("records"),
            "provenance": {"tap_url": _ALMA_TAP_URL, "adql": executed, "elapsed_s": round(time.perf_counter() - started, 1)},
            "note": (
                "Counts are per band token of publicly released ObsCore rows; a band-to-band row ('5 10') counts for both bands. "
                "Per-band n_mous/n_projects sum the token groups (a project spanning bands appears once per band). "
                + ("as_of filters on obs_release_date <= that date: it describes CURRENT records released by then, not a historical snapshot." if as_of else "")
            ),
        }
        return _native(out)

    @staticmethod
    def _bands_without_public_data(ref: Dict[str, Any], with_data: set) -> List[Dict[str, Any]]:
        """Each receiver band with no public rows, with the cycles the reference tables offer it in. A band first
        offered in the latest tabulated cycle had not operated for science before it, which explains the empty archive
        (ALMAbench 2026-10-15 A14: once the tables loaded, answers said 'Band 2 has none' and dropped the reason)."""
        offered = {int(c): {int(b) for b in bands} for c, bands in (ref.get("offered_bands_by_cycle") or {}).items()}
        out: List[Dict[str, Any]] = []
        for row in ref.get("bands", []):
            band = int(row.get("band"))
            if band in with_data:
                continue
            cycles = sorted(c for c, bands in offered.items() if band in bands)
            entry: Dict[str, Any] = {"band": band, "offered_in_tabulated_cycles": cycles,
                                     "tabulated_cycles": sorted(offered)}
            if offered and cycles and min(cycles) > min(offered):
                entry["explanation"] = (f"Band {band} is first offered in Cycle {min(cycles)} (not in Cycle "
                                        f"{', '.join(str(c) for c in sorted(offered) if c < min(cycles))}), so it had not been "
                                        "operating for science observations long enough for data to be public.")
            elif offered and not cycles:
                entry["explanation"] = f"Band {band} is not offered in the tabulated cycles {sorted(offered)}."
            out.append(entry)
        return out


# ═════════════════════════════════════════════════════════════════════════
# 4. alma_archive_link
# ═════════════════════════════════════════════════════════════════════════
class ArchiveLinkInput(_In):
    target: Optional[str] = Field(default=None, description="Source name for the ASA name resolver (sourceNameResolver)")
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcmin: Optional[float] = Field(default=None, description="cone radius in arcmin (ASA default 10)")
    band: Any = Field(default=None, description="ALMA band(s), e.g. 6 or [6,7]")
    public_only: bool = False
    released_before: Optional[str] = None  # ISO date -> observationsReleaseDate=<date filter
    extra: Dict[str, Any] = Field(default_factory=dict, description="additional documented ASA URL parameters, verbatim")


# Read from the ASA application (almascience.eso.org/aq/main.js) and checked live in a browser on 2026-10-16:
# sourceName=NGC253&band=3,7&observationsReleaseDate=%3C2025-01-01 applied all three filters (the page showed
# "ALMA source name: NGC253", "Band: 3, 7", "Release date: <2025-01-01"); almascience.org/aq/ answered 302 to a mirror.
_ASA_LINK_NOTES = {
    "url_structure": (
        "Base: the ASA query page https://almascience.org/aq/ (redirects to the nearest mirror: almascience.nrao.edu, "
        ".eso.org or .nao.ac.jp). Filters follow the '?' as key=value pairs joined by '&'."),
    "url_encoding": (
        "Values are URL-encoded: '<' is %3C, '>' %3E, a space %20 or '+', so 'release date < 2026-01-01' is "
        "observationsReleaseDate=%3C2026-01-01. Several values of one filter are comma-separated (band=6,7); the "
        "column filters take the ASA operators '<', '>', '..' (range) and '|' (or)."),
    "name_vs_resolver": (
        "sourceNameResolver resolves the name (SIMBAD/NED) to a position and returns every observation whose footprint "
        "overlaps a cone around it (10 arcmin default, ', N' = N arcmin), whatever name the PI typed, so it can also "
        "return pointings near the source. sourceName matches the source name the PI entered (wildcards '*', '?'), so "
        "it misses data registered under another name (a catalogue number, underscores) and finds nothing for a "
        "renamed target. The two searches can therefore give different results."),
    "sharing": (
        "The ASA writes every active filter into the page URL, so the address bar after setting a query in the "
        "interface is the shareable link. Opening it re-runs the query live, so the results can change as data are "
        "added or released; record the date with any result list."),
}


class AlmaArchiveLink(BaseCapability):
    name = "alma_archive_link"
    description = (
        "Build the ALMA Science Archive (ASA) query-interface deep link for a target/position and band using the ASA "
        "filter parameter names (sourceNameResolver, sourceName, raDec, band, publicData), plus the equivalent "
        "TAP sync URL. Both are HEAD-verified live and returned with a verified/unverified label. Use whenever the user asks "
        "for a URL/link into the ALMA archive (a source or position, bands, public data, a release-date cut)."
    )
    category = "archive"
    InputModel = ArchiveLinkInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        from capabilities.alma_science_helpers import head_verify  # local helper module (below)

        # Filter ids read from the ASA application (almascience.eso.org/aq/main.js, 2026-10-16) and checked live: the
        # band filter is `band` (comma list); `bandList` and `resultView` are silently ignored (ALMAbench A24: the old
        # link returned every band). A trailing ", N" on sourceNameResolver / raDec is the radius in ARCMIN (default 10).
        params: Dict[str, str] = {}
        if inp.target:
            params["sourceNameResolver"] = str(inp.target).strip() + (f", {float(inp.radius_arcmin):g}" if inp.radius_arcmin else "")
        if inp.ra is not None and inp.dec is not None:
            r = float(inp.radius_arcmin) if inp.radius_arcmin else 1.0
            params["raDec"] = f"{float(inp.ra):.6f} {float(inp.dec):+.6f}, {r:g}"
        bands: List[str] = []
        if inp.band is not None:
            raw = inp.band if isinstance(inp.band, (list, tuple)) else re.split(r"[,\s]+", str(inp.band))
            bands = [str(int(b)) for b in raw if str(b).strip().isdigit() and 1 <= int(b) <= 10]
            if bands:
                params["band"] = ",".join(bands)
        if inp.public_only:
            params["publicData"] = "true"
        if inp.released_before:
            try:
                params["observationsReleaseDate"] = "<" + _dt.date.fromisoformat(str(inp.released_before).strip()[:10]).isoformat()
            except ValueError:
                return _native({"success": False, "status": "invalid_input",
                                "error": f"released_before must be an ISO date (YYYY-MM-DD), got {inp.released_before!r}"})
        for k, v in (inp.extra or {}).items():
            if k and v is not None:
                params[str(k)] = str(v)
        if not (inp.target or (inp.ra is not None and inp.dec is not None)):
            return _native({"success": False, "error": "target or ra/dec is required"})
        asa_url = _ASA_BASE + "?" + urlencode(params, quote_via=quote_plus, safe=",")
        # The same filters with a PI-name match instead of the resolver (see _ASA_LINK_NOTES name_vs_resolver).
        name_url = None
        if inp.target and "sourceNameResolver" in params:
            name_params = {("sourceName" if k == "sourceNameResolver" else k): (str(inp.target).strip() if k == "sourceNameResolver" else v)
                           for k, v in params.items()}
            name_url = _ASA_BASE + "?" + urlencode(name_params, quote_via=quote_plus, safe=",")

        # TAP equivalent: needs coordinates.
        tap_url = None
        adql = None
        resolved_note = ""
        ra, dec = inp.ra, inp.dec
        if ra is None and inp.target:
            try:
                res = ctx.service("resolve_target")(normalize_target_alias(inp.target))
                ra, dec = res.get("ra_deg"), res.get("dec_deg")
                resolved_note = f"{inp.target} resolved to RA={float(ra):.5f}, Dec={float(dec):.5f}" if ra is not None else ""
            except Exception as exc:
                resolved_note = f"name resolution failed ({type(exc).__name__}); TAP link omitted"
        if ra is not None and dec is not None:
            radius_deg = (float(inp.radius_arcmin) if inp.radius_arcmin else 1.0) / 60.0
            adql = alma_cone_adql(float(ra), float(dec), radius_deg, public=bool(inp.public_only), top=500,
                                  columns=("proposal_id", "member_ous_uid", "target_name", "band_list", "s_ra", "s_dec",
                                           "spatial_resolution", "obs_release_date", "data_rights"),
                                  band=[int(b) for b in bands] or None, released_before=inp.released_before or None)
            tap_url = _ALMA_TAP_URL + "/sync?" + urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}, quote_via=quote_plus)

        checks = {"asa": head_verify(asa_url), "tap": head_verify(tap_url) if tap_url else None}
        out = {
            "success": True, "status": "ok",
            "asa_url": asa_url,
            "asa_url_verified": checks["asa"][0], "asa_url_check": checks["asa"][1],
            "asa_parameters": params,
            "tap_sync_url": tap_url, "tap_url_verified": (checks["tap"][0] if checks["tap"] else None),
            "tap_url_check": (checks["tap"][1] if checks["tap"] else None), "tap_adql": adql,
            "resolved": resolved_note,
            "asa_name_match_url": name_url,
            "links": [asa_url] + ([name_url] if name_url else []) + ([tap_url] if tap_url else []),
            "url_notes": _ASA_LINK_NOTES,
            "note": (
                "Parameter names are the ASA filter ids (sourceNameResolver = name resolved to a position, 10 arcmin default "
                "radius or ', N' arcmin; sourceName = the PI's source name; raDec = 'ra dec, radius_arcmin'; band = comma list; "
                "publicData=true; observationsReleaseDate='<YYYY-MM-DD'), checked live in the ASA application on 2026-10-16; "
                "see url_notes for the URL structure, encoding, resolver vs name match and sharing. "
                "'verified' means the endpoint answered the URL with an HTTP 2xx/3xx; the ASA is a browser application, so the server "
                "does not validate individual parameters in that response — the TAP link is the mechanically checkable equivalent."
            ),
        }
        return _native(out)


# ═════════════════════════════════════════════════════════════════════════
# 5. alma_bibliography
# ═════════════════════════════════════════════════════════════════════════
_PROJECT_CODE_RE = re.compile(r"\b(20\d{2}\.[0-9A-Z]\.\d{5}\.[A-Z])\b")


class BibliographyInput(_In):
    topic: str
    synonyms: Optional[List[str]] = None  # spelling / acronym variants ADS does not map
    n: int = 10
    since: Optional[int] = Field(default=None, description="earliest publication year")
    archival: str = Field(default="any", description="any | only | ignore (archival use of ALMA data)")
    refereed_only: bool = True


class AlmaBibliography(BaseCapability):
    name = "alma_bibliography"
    description = (
        "Refereed papers on a topic that USED ALMA DATA (ADS bibgroup:ALMA = the ALMA telescope bibliography), newest "
        "first, each classified for ARCHIVAL use the way ALMA's publication statistics define it (no author is PI or "
        "co-I of the ALMA project the paper used; projects, PIs and co-Is from the ALMA archive). Per paper: bibcode, "
        "title, date, first author, ALMA projects with PI, topic evidence, archival class + confidence, and the abstract (whole sentences; give each paper a short summary of its target, data and main result from it). "
        "archival='any' (default) keeps archival only + archival+PI papers, 'only' archival only, 'ignore' all. The list "
        "is closed: never add papers that are not in it."
    )
    category = "literature"
    InputModel = BibliographyInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        from services import alma_bibliography as ab

        ads = ctx.services.get("ads_client")
        if ads is None:
            return _native({"success": False, "status": "infrastructure_failure", "error": "ADS client is not configured"})
        topic = str(inp.topic or "").strip()
        if not topic:
            return _native({"success": False, "error": "topic is required"})
        mode = str(inp.archival or "any").strip().lower()
        if mode not in {"any", "only", "ignore"}:
            return _native({"success": False, "status": "invalid_input", "error": "archival must be any | only | ignore"})
        n = max(1, min(int(inp.n), 50))
        phrases = [topic] + [str(x).strip() for x in (inp.synonyms or []) if str(x).strip()][:8]
        q = ab.ads_query(phrases)
        filters = ["doctype:article"]
        if inp.refereed_only:
            filters.append("property:refereed")
        if inp.since:
            filters.append(f"year:{int(inp.since)}-")
        started = time.perf_counter()

        mapping: Dict[str, List[Dict[str, Any]]] = {}
        map_note = ""
        map_complete = True
        if True:  # classify in every mode (the map is cached); `archival` only decides what is kept
            executed: List[str] = []
            try:
                mapping, built, map_complete = ab.bib_map(_run_adql_factory(ctx, executed))
                map_note = f"ALMA paper-project map from ObsCore bib_reference ({len(mapping)} papers), built {built}."
                if not map_complete:
                    map_note += " The map query hit its row cap: some paper-project links may be missing, so archival classes are provisional."
            except Exception as exc:  # noqa: BLE001 - without the map the archival class is unknown, said so below
                map_note = f"ALMA paper-project map unavailable ({type(exc).__name__}): archival use could not be classified."

        patterns = ab.topic_patterns(phrases)
        kept: List[Dict[str, Any]] = []
        scanned = off_topic = wrong_class = 0
        fields = ["bibcode", "title", "author", "pubdate", "year", "pub", "abstract", "doi", "keyword"]
        try:
            raw = getattr(ads, "search_raw", None)
            for start in (0, 200, 400):
                if raw is not None:   # full author lists (search_papers shortens them for display; guard CX-01)
                    page = raw(q, rows=200, sort="date desc", fields=fields, filters=filters, start=start)
                else:
                    page = ads.search_papers(q, max_results=200, sort="date desc", fields=fields, filters=filters, start=start)
                for p in page:
                    scanned += 1
                    evidence = ab.topic_evidence(p, patterns)
                    if evidence is None:
                        off_topic += 1
                        continue
                    authors = p.get("author") if isinstance(p.get("author"), list) else None
                    full_authors = authors is not None
                    if authors is None:   # display-formatted record ('A et al.'): cannot be checked author by author
                        shown = p.get("authors") or p.get("author") or ""
                        authors = [shown] if isinstance(shown, str) else list(shown)
                    cls, proj = ab.classify(authors, mapping.get(str(p.get("bibcode") or ""), []))
                    if not full_authors and cls != ab.UNKNOWN:
                        cls = ab.UNKNOWN
                    if mode == "any" and cls not in (ab.ARCHIVAL_ONLY, ab.ARCHIVAL_PLUS_PI):
                        wrong_class += 1
                        continue
                    if mode == "only" and cls != ab.ARCHIVAL_ONLY:
                        wrong_class += 1
                        continue
                    kept.append(self._row(p, authors, cls, proj, evidence))
                    if len(kept) >= n:
                        break
                if len(kept) >= n or len(page) < 200 or time.perf_counter() - started > 40.0:
                    break
        except Exception as exc:
            out = {"success": False, "error": f"ADS search failed: {type(exc).__name__}: {str(exc)[:200]}",
                   "status": "infrastructure_failure", "query": q}
            return _native(_infra_failure(out, exc))

        table = pd.DataFrame(kept)
        _data_card(ctx, table, source=f"ALMA bibliography: {topic}", tool_name=self.name)
        try:
            ctx.service("set_last_run_result")({"type": "papers", "papers": kept, "source": "ADS bibgroup:ALMA", "tool_name": self.name})
        except Exception:
            pass
        as_of = _dt.date.today().isoformat()
        return _native({
            "success": True, "status": "ok" if (mapping and map_complete) else "partial",
            "topic": topic, "phrases": phrases, "archival_filter": mode, "as_of": as_of,
            "n_returned": len(kept), "results": kept,
            "search": {"ads_query": q, "filters": filters, "papers_scanned": scanned, "off_topic_dropped": off_topic,
                       "dropped_by_archival_class": wrong_class},
            "source": "NASA ADS bibgroup:ALMA (the ALMA telescope bibliography) + ALMA archive ObsCore bib_reference",
            "method_note": ab.METHOD_NOTE + (" " + map_note if map_note else ""),
            "note": (f"Search run {as_of}, newest first; 'most recent' depends on that date and on the ALMA bibliography's "
                     "curation lag. These are the ONLY papers found: do not add, rename or describe any paper that is not "
                     "in results, and quote each archival_class with its confidence."),
        })

    @staticmethod
    def _row(p: Dict[str, Any], authors: List[str], cls: str, proj: List[Dict[str, Any]], evidence: str) -> Dict[str, Any]:
        from services import alma_bibliography as ab

        title = p.get("title")
        title = (title[0] if title else "") if isinstance(title, list) else title
        first = authors[0] if authors else ""
        abstract = str(p.get("abstract") or "")
        codes = [e["project"] for e in proj] or sorted(set(_PROJECT_CODE_RE.findall(abstract)))
        doi = p.get("doi")
        return {
            "bibcode": p.get("bibcode"), "title": title, "pubdate": str(p.get("pubdate") or p.get("year") or "")[:7],
            "first_author": f"{first} et al." if len(authors) > 1 else first, "journal": p.get("pub") or p.get("journal"),
            "alma_project_codes": ", ".join(codes) if codes else "not linked",
            "alma_projects": proj, "archival_class": cls, "confidence": ab.CONFIDENCE[cls],
            "topic_evidence": evidence, "summary": ab.summarize(abstract),
            "doi": (doi[0] if doi else None) if isinstance(doi, list) else doi,
            "ads_url": f"https://ui.adsabs.harvard.edu/abs/{p.get('bibcode')}/abstract" if p.get("bibcode") else None,
        }


# ═════════════════════════════════════════════════════════════════════════
# 6. cross_archive_match
# ═════════════════════════════════════════════════════════════════════════
class CrossArchiveMatchInput(_In):
    catalog_name: Optional[str] = Field(default="perseus_protostars", description="'vizier:<catalogue id>' (any published list), a built-in name, or 'inline' with sources")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="[{source_name, ra, dec}, ...]")
    archives: Optional[List[str]] = Field(default=None, description="e.g. ['ALMA','JWST','HST']")
    radius_arcsec: float = 5.0
    max_sources: int = 100  # the built-in Perseus list is the 97-source VANDAM sample
    require_all_archives: Any = False


class CrossArchiveMatch(BaseCapability):
    name = "cross_archive_match"
    description = (
        "Cross-match a source list against ALMA and MAST missions (JWST/HST/...) with ONE POINT CONE PER SOURCE for ALMA "
        "(never the many-source footprint query that times out on every mirror) and one MAST query per source, run "
        "concurrently under the tool budget. Returns a per-source table with a STATUS COLUMN PER ARCHIVE (ok | no_match | "
        "timeout | skipped | unreachable), a map card labelled ALMA-only / JWST-only / both, and an explicit 'ALMA status "
        "unknown for k sources' line when a phase failed. Use for 'which sources of a catalogue or region have both ALMA and JWST data'. Target <= 90 s."
    )
    category = "archive"
    InputModel = CrossArchiveMatchInput
    annotations = {"read_only": True, "cost": "network"}

    @staticmethod
    def _observation_count(row: Dict[str, Any], archive: str) -> Optional[int]:
        """Observation count for one archive from the matcher's summary row
        (services/cross_archive_matcher.py: alma_observations,
        jwst_observations, mast_observations)."""
        a = archive.lower()
        keys = [f"{a}_observations"]
        if a != "alma":
            keys.append("mast_observations")
        for k in keys:
            v = row.get(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip().isdigit():
                return int(v.strip())
        return None

    def run(self, inp, ctx) -> ToolResult:
        from capabilities.alma import _match_cross_archive_sources_impl

        archives = [str(a).upper() for a in (inp.archives or ["ALMA", "JWST"])]
        raw = _match_cross_archive_sources_impl(
            ctx, catalog_name=inp.catalog_name or "perseus_protostars", sources=inp.sources, archives=archives,
            radius_arcsec=float(inp.radius_arcsec), max_sources=int(inp.max_sources),
            require_all_archives=inp.require_all_archives, alma_mode="point_only",
        )
        if not isinstance(raw, dict):
            return _native({"success": False, "error": "cross-match returned no result"})
        errors = [str(e) for e in raw.get("archive_errors") or []]
        dead = list(raw.get("dead_hosts") or [])
        budget_out = bool(raw.get("budget_exhausted"))
        n_sources = int(raw.get("sources_tested") or 0)
        # Per-archive phase status from the error notes.
        def _phase_status(archive: str) -> str:
            a = archive.lower()
            texts = " ".join(errors).lower()
            if any(a in h.lower() or ("alma" == a and "almascience" in h.lower()) or ("mast" in h.lower() and a != "alma") for h in dead):
                return "unreachable"
            if a == "alma":
                if "alma box query hit the" in texts:
                    # Capped box answer (guard CX-02 reopen): a zero count is unknown, not absence.
                    return "truncated"
                if "alma phase did not finish" in texts or "alma not queried" in texts or "alma point cones" in texts:
                    return "timeout"
                if "alma tap failed" in texts or "alma tap unreachable" in texts:
                    return "failed"
                return "ok"
            if "mast not queried" in texts or "mast queries" in texts:
                return "timeout"
            if "mast unreachable" in texts:
                return "unreachable"
            if "mast query failed" in texts:
                return "partial"
            return "ok"

        phase_status = {a: _phase_status(a) for a in archives}
        # ALMA sources not queried: parse "ALMA not queried for N source(s)" / "k of N sources answered".
        unknown_alma = 0
        for e in errors:
            m = re.search(r"ALMA not queried for (\d+) source", e)
            if m:
                unknown_alma = max(unknown_alma, int(m.group(1)))
            m2 = re.search(r"(\d+) of (\d+) sources answered", e)
            if m2:
                unknown_alma = max(unknown_alma, int(m2.group(2)) - int(m2.group(1)))
        if phase_status.get("ALMA") in {"unreachable", "failed"}:
            unknown_alma = n_sources
        results = raw.get("results") or []
        require_all = bool(inp.require_all_archives) and str(inp.require_all_archives).lower() not in {"false", "0", "no"}
        for row in results:
            for a in archives:
                key = f"{a.lower()}_status"
                if key in row:
                    continue
                n = self._observation_count(row, a)
                st = phase_status.get(a, "ok")
                if n is None:
                    # No count column for this archive: with require_all_archives
                    # every returned row matched EVERY archive by construction.
                    row[key] = "ok" if require_all else ("unknown" if st == "ok" else st)
                else:
                    row[key] = "ok" if n > 0 else ("no_match" if st == "ok" else ("unknown_truncated" if st == "truncated" else st))
                if n is not None:
                    row[f"{a.lower()}_n_observations"] = n
        if phase_status.get("ALMA") == "truncated":
            # Under a capped box answer every source WITHOUT a returned ALMA match is unknown, including the ones a
            # require_all_archives filter already dropped from `results` (guard CX-02 reopen): an upper bound.
            alma_known = sum(1 for row in results if row.get("alma_status") == "ok")
            unknown_alma = max(unknown_alma, n_sources - alma_known)
        status = "ok"
        if raw.get("success") is False:
            status = "infrastructure_failure" if (dead or raw.get("infrastructure_failure")) else "partial"
        elif errors or budget_out or unknown_alma:
            status = "partial"
        require_all_h = bool(inp.require_all_archives) and str(inp.require_all_archives).lower() not in {"false", "0", "no"}
        headline_bits = [
            f"{raw.get('matched_sources', 0)} of {n_sources} sources have data in ALL of {', '.join(archives)}"
            if require_all_h else f"{raw.get('matched_sources', 0)} of {n_sources} sources have data in at least one of {', '.join(archives)}"
        ]
        if unknown_alma:
            headline_bits.append(f"ALMA status unknown for {'up to ' if phase_status.get('ALMA') == 'truncated' else ''}"
                                 f"{unknown_alma} source(s) ("
                                 + ("ALMA box query hit its row cap" if phase_status.get("ALMA") == "truncated"
                                    else "phase timed out / not queried") + ")")
        out = dict(raw)
        out.update({
            "status": status,
            "phase_status": phase_status,
            "alma_status_unknown_for": unknown_alma,
            "alma_query_mode": ("one box query for every science pointing around the catalogue; a source matches when a "
                                "pointing's field of view (or the match radius) contains it"
                                if raw.get("footprint_mode") == "box_fov" else
                                "one point cone per source (footprint INTERSECTS deliberately not used)"),
            "headline": "; ".join(headline_bits),
            "map_legend": {"ALMA-only": "sources with ALMA rows only", "JWST-only": "sources with JWST/MAST rows only", "both": "sources with both"},
            "results": results,
        })
        if unknown_alma:
            out["note"] = (out.get("note") or "") + f" ALMA status is UNKNOWN for {'up to ' if phase_status.get('ALMA') == 'truncated' else ''}{unknown_alma} source(s): do not state that they lack ALMA data."
        # Relabel the card with the per-archive legend.
        try:
            lrr = ctx.service("get_last_run_result")()
            if isinstance(lrr, dict) and lrr.get("tool_name") == "match_cross_archive_sources":
                lrr = dict(lrr, tool_name=self.name, filter_label=f"{lrr.get('filter_label', '')} — ALMA-only / JWST-only / both", partial=status != "ok")
                ctx.service("set_last_run_result")(lrr)
        except Exception:
            pass
        out["success"] = bool(raw.get("success"))
        return _native(out)


# ═════════════════════════════════════════════════════════════════════════
# 7. archive_overlay
# ═════════════════════════════════════════════════════════════════════════
class OverlayLayer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mission: str = Field(description="JWST | HST | ALMA")
    instrument: Optional[str] = None


class ArchiveOverlayInput(_In):
    field: Optional[str] = Field(default=None, description="named field/region (e.g. 'Hubble Ultra Deep Field') or target name")
    ra: Optional[float] = None
    dec: Optional[float] = None
    base: OverlayLayer = Field(default_factory=lambda: OverlayLayer(mission="JWST"))
    contour: OverlayLayer = Field(default_factory=lambda: OverlayLayer(mission="ALMA"))
    size_arcmin: float = 3.0
    max_product_mb: float = 150.0


class ArchiveOverlay(BaseCapability):
    name = "archive_overlay"
    description = (
        "Overlay ALMA continuum CONTOURS on a JWST/HST COLOURSCALE image of a field in one call: picks a MAST science FITS "
        "product by size (<= max_product_mb), otherwise falls back to a hips2fits cutout of the mission's HiPS, fetches the "
        "ALMA continuum product via DataLink, reprojects with WCS, and renders colourbar + contour legend. Returns "
        "status='coverage_gap' naming the missing side when either image is unavailable, instead of looping. "
        "Use for 'show ALMA contours on a JWST/HST image of a field or source'. Target <= 120 s."
    )
    category = "analysis"
    InputModel = ArchiveOverlayInput
    annotations = {"read_only": False, "cost": "network"}

    # Single-band HiPS maps that exist at CDS (MocServer, 2026-09-23); the one
    # covering the position is chosen with a MocServer coverage query.
    _HIPS_FALLBACK = {
        "JWST": ["CDS/P/JWST/F200W", "CDS/P/JWST/F444W", "CDS/P/JWST/F150W", "CDS/P/JWST/F115W", "ESAVO/P/JWST/NIRCam_Imaging"],
        "HST": ["CDS/P/HST/GOODS/z", "CDS/P/HST/GOODS/i", "CDS/P/HST/I", "ESAVO/P/HST/ACS-blue", "CDS/P/HST/wideV"],
    }

    def run(self, inp, ctx) -> ToolResult:
        from capabilities.viz import overlay_region_coordinates, pick_mast_fits_product
        from services.alma_server_side import run_concurrently
        from services.tool_budgets import remaining_seconds

        started = time.perf_counter()
        coords = overlay_region_coordinates(inp.field or "", ra_deg=inp.ra, dec_deg=inp.dec)
        if not coords and inp.field:
            try:
                res = ctx.service("resolve_target")(normalize_target_alias(inp.field))
                if res.get("ra_deg") is not None:
                    coords = (float(res["ra_deg"]), float(res["dec_deg"]), str(inp.field))
            except Exception:
                coords = None
        if not coords:
            return _native({"success": False, "status": "coverage_gap", "error": f"could not resolve the field {inp.field!r}; give ra/dec"})
        ra, dec, label = coords
        base_mission = str(inp.base.mission or "JWST").upper()
        if str(inp.contour.mission or "ALMA").upper() != "ALMA":
            return _native({"success": False, "error": "contour.mission must be ALMA (continuum products via DataLink)"})
        size_arcmin = max(0.5, float(inp.size_arcmin))
        max_mb = float(inp.max_product_mb)
        # Phase walls: a clear answer inside ~100 s even when an archive hangs.
        left = remaining_seconds()
        phase_wall = min(90.0, max(20.0, (left - 25.0) if left is not None else 90.0))

        def base_phase() -> Dict[str, Any]:
            notes: List[str] = []
            mast = ctx.services.get("mast_client")
            if mast is not None and base_mission in {"JWST", "HST"}:
                try:
                    obs = mast.search_by_position(ra, dec, radius_arcmin=size_arcmin, mission=base_mission, max_results=80)
                    err = (getattr(obs, "attrs", None) or {}).get("error") or (getattr(obs, "attrs", None) or {}).get("quasar_error")
                    if err:
                        notes.append(f"MAST {base_mission} search did not complete ({str(err)[:100]}): {base_mission} coverage UNKNOWN, using HiPS")
                    elif obs is None or obs.empty:
                        notes.append(f"MAST lists no {base_mission} observations within {size_arcmin:g} arcmin of {label}; using HiPS")
                    else:
                        if inp.base.instrument and "instrument_name" in obs.columns:
                            sel = obs[obs["instrument_name"].astype(str).str.contains(str(inp.base.instrument), case=False, na=False)]
                            obs = sel if not sel.empty else obs
                        products = mast.get_product_list(obs, productType="SCIENCE", extension="fits")
                        prod = pick_mast_fits_product(products, max_product_mb=max_mb)
                        if prod and prod.get("access_url"):
                            return {"url": prod["access_url"], "source": {"kind": "MAST product", "filename": prod.get("productFilename") or prod.get("filename"), "access_url": prod["access_url"]}, "notes": notes}
                        notes.append(f"{len(obs)} {base_mission} observations exist but no science FITS product <= {max_mb:g} MB; using HiPS")
                except Exception as exc:
                    notes.append(f"MAST lookup failed ({type(exc).__name__}: {str(exc)[:100]}); {base_mission} coverage UNKNOWN, using HiPS")
            sid = self._covering_hips(ra, dec, base_mission)
            if sid:
                from services.hips_images import HipsImageService

                url = HipsImageService().fits_url(ra, dec, fov_deg=size_arcmin / 60.0, survey=sid, width=768)
                return {"url": url, "source": {"kind": "HiPS (hips2fits) fallback", "survey": sid, "access_url": url}, "notes": notes}
            notes.append(f"no {base_mission} HiPS map covers {label}")
            return {"url": None, "source": None, "notes": notes}

        def contour_phase() -> Dict[str, Any]:
            return self._find_alma_contour(ctx, ra, dec, size_arcmin, max_mb, label)

        base_out, contour_out = run_concurrently([base_phase, contour_phase], wall_seconds=phase_wall)
        gaps: List[str] = []
        if isinstance(base_out, BaseException):
            gaps.append(f"base-image phase did not finish within {phase_wall:.0f} s ({type(base_out).__name__})")
            base_out = {"url": None, "source": None, "notes": []}
        if isinstance(contour_out, BaseException):
            gaps.append(f"ALMA phase did not finish within {phase_wall:.0f} s ({type(contour_out).__name__}): ALMA coverage UNKNOWN")
            contour_out = {"product": None, "notes": [], "unknown": True}
        gaps = list(base_out.get("notes") or []) + list(contour_out.get("notes") or []) + gaps
        base_url, base_source, contour = base_out.get("url"), base_out.get("source"), contour_out.get("product")
        if base_url is None or not contour:
            missing = [side for side, ok in (("base image", base_url is not None), ("ALMA contour product", bool(contour))) if not ok]
            return _native({
                "success": False, "status": "coverage_gap", "field": label, "ra": ra, "dec": dec,
                "missing": missing, "base_source": base_source, "gaps": gaps,
                "alma_status": "unknown (lookup did not complete)" if contour_out.get("unknown") else ("no product found" if not contour else "ok"),
                "error": f"overlay not possible: missing {', '.join(missing)}. " + "; ".join(gaps),
                "elapsed_s": round(time.perf_counter() - started, 1),
                "note": ("This is a coverage/availability gap, not a tool error: do not retry with the same inputs. Where a phase "
                         "did not complete, say the coverage is UNKNOWN (not absent); offer another field, mission, or a single-archive image."),
            })
        try:
            from services.fits_service import overlay_fits_images

            is_hips = bool(base_source and str(base_source.get("kind", "")).startswith("HiPS"))
            result = overlay_fits_images(base_url, contour["access_url"],
                                         base_label=f"{base_mission} {label}" + (f" ({base_source.get('survey')})" if is_hips else ""),
                                         contour_label=f"ALMA {label}", base_cmap="inferno", contour_levels=8)
        except Exception as exc:
            return _native(_infra_failure({"success": False, "status": "infrastructure_failure", "error": f"overlay render failed: {type(exc).__name__}: {str(exc)[:200]}",
                                            "base_source": base_source, "contour": contour, "gaps": gaps}, exc))
        if result.get("success"):
            try:
                ctx.service("set_last_run_result")({"type": "image", "image_url": result["image_path"],
                                                    "caption": result.get("caption") or f"ALMA contours on {base_mission} {label}", "tool_name": self.name})
            except Exception:
                pass
        result.update({
            "status": "ok" if result.get("success") else "infrastructure_failure", "field": label, "ra": ra, "dec": dec,
            "base_source": base_source, "contour_product": {"filename": contour.get("filename"), "access_url": contour.get("access_url"),
                                                            "member_ous_uid": contour.get("member_ous_uid"), "proposal_id": contour.get("proposal_id"),
                                                            "target_name": contour.get("target_name"),
                                                            "field_of_view_arcmin": contour.get("field_of_view_arcmin"),
                                                            "footprint_check": contour.get("footprint_check")},
            "gaps": gaps, "elapsed_s": round(time.perf_counter() - started, 1),
        })
        if result.get("success"):
            result["checks"] = self._overlay_checks(result, base_mission)
        return _native(result)

    @staticmethod
    def _overlay_checks(result: Dict[str, Any], base_mission: str) -> List[str]:
        """What a careful overlay answer reports (ALMAbench 2026-10-15 A26: the tool drew noise-based contours but the
        answers never said so, nor discussed resolution or registration)."""
        cont = result.get("contours") or {}
        beam = result.get("contour_beam") or {}
        checks = []
        if cont.get("rms"):
            checks.append(f"Contours are at {', '.join(f'{s:g}' for s in cont.get('sigmas') or [])} x the ALMA rms "
                          f"({cont['rms']:.3g} map units, robust MAD over the map)"
                          + ("; dashed contours are -3 rms, which show the noise and any negative sidelobes" if cont.get("negative_contours") else "")
                          + (". The ALMA image is primary-beam corrected, so its noise rises toward the field edge: judge "
                             "significance near the centre or on the uncorrected image." if ".pbcor." in str((result.get("contour_product") or {}).get("filename") or "").lower() else "."))
        if beam.get("bmaj_arcsec"):
            checks.append(f"Resolution: the ALMA beam is {beam['bmaj_arcsec']:g} x {beam['bmin_arcsec']:g} arcsec (PA {beam['bpa_deg']:g} deg)"
                          + (f" against {result['base_pixel_arcsec']:g} arcsec pixels in the {base_mission} image" if result.get("base_pixel_arcsec") else "")
                          + "; compare like with like (smooth the sharper image to the ALMA beam before comparing sizes or fluxes).")
        if result.get("reprojected") is False:
            # The renderer could not reproject (reproject missing): the contours are NOT aligned (guard CX-06).
            checks.append("Alignment: the ALMA image could NOT be reprojected onto the base image here, so the contours "
                          "are drawn in their own pixel grid and are not aligned; do not read positions from this figure.")
        else:
            checks.append("Registration: the ALMA image was reprojected onto the base image's WCS; check the astrometry "
                          "against a common frame (Gaia) using bright compact sources before reading offsets. Older "
                          "HST-based astrometry in deep fields can be offset from ALMA positions by a few tenths of an arcsec.")
        checks.append("Headers: ALMA images carry degenerate frequency and Stokes axes, dropped here (2-D celestial WCS) before "
                      "reprojection; both images must be in the same frame (ICRS).")
        return checks

    @classmethod
    def _covering_hips(cls, ra: float, dec: float, mission: str) -> Optional[str]:
        """First fallback HiPS whose MOC covers (ra, dec) -- one MocServer query."""
        candidates = cls._HIPS_FALLBACK.get(mission, [])
        if not candidates:
            return None
        try:
            import requests

            from services.tool_budgets import bounded_timeout

            r = requests.get(
                "https://alasky.cds.unistra.fr/MocServer/query",
                params={"RA": f"{ra:.6f}", "DEC": f"{dec:.6f}", "SR": "0.005", "expr": " || ".join(f"ID={c}" for c in candidates),
                        "get": "id", "fmt": "ascii"},
                timeout=bounded_timeout(15.0, minimum=2.0, label="MocServer coverage"),
            )
            covering = {line.strip() for line in r.text.splitlines() if line.strip()}
            for c in candidates:
                if c in covering:
                    return c
            return None
        except Exception:
            return candidates[0]  # coverage check unavailable: try the first map

    # member.uid___A001_X2fe_Xb2a.ari_l.HUDF-JVLA-ALMA_sci.spw0_1_2_3_262521MHz.12m.cont.I.pbcor.fits: the source
    # sits between the uid token (and the optional ari_l. prefix) and "_sci"; it may itself contain dots (G31.41+0.31).
    _FILE_SOURCE_RE = re.compile(r"^(?:member\.)?uid___[a-z0-9]+_[a-z0-9]+_[a-z0-9]+\.(?:ari_l\.)?(.+?)_sci[._]", re.I)

    @staticmethod
    def _file_target(filename: str) -> Optional[str]:
        """The source key (:func:`_source_key`) of an ALMA product filename, or None when it is not a science image
        (``member.uid___A001_X1_X1.HUDF_sci.spw25.cont.I.pbcor.fits`` -> ``hudf``)."""
        m = ArchiveOverlay._FILE_SOURCE_RE.search(str(filename or ""))
        return _source_key(m.group(1)) if m else None

    @staticmethod
    def _pick_science_continuum(files: List[Dict[str, Any]], max_mb: float,
                                target: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """The best SCIENCE continuum image of a MOUS: ``_sci`` target, a
        continuum product (``.cont.`` or ``.mfs.``), primary-beam corrected,
        never a calibrator (``_bp``/``_ph``/``_chk``/``_flux``/``_pol``) or a
        diagnostic (mask/pb/psf/residual/model), within the size limit.
        ``target`` (a :func:`_source_key`) keeps only that source's images, by
        exact key: one MOUS can hold dozens of pointings (2016.1.01079.S has 64),
        and a file whose source cannot be parsed or differs is never taken."""
        def name(f):
            return str(f.get("filename") or "").lower()

        cands = []
        for f in files:
            n = name(f)
            if not (n.endswith(".fits") or n.endswith(".fits.gz")) or not f.get("access_url"):
                continue
            if "_sci." not in n and "_sci_" not in n:
                continue
            if target is not None and ArchiveOverlay._file_target(n) != target:
                continue
            if any(tok in n for tok in (".mask.", ".pb.", ".psf.", ".residual.", ".model.", ".weight.", ".sumwt.")):
                continue
            if ".cont." not in n and ".mfs." not in n:
                continue
            size = float(f.get("size_mb") or 0)
            if size and size > max_mb:
                continue
            rank = (0 if ".cont." in n else 1, 0 if ".pbcor." in n else 1, 0 if n.endswith(".fits") else 1, size)
            cands.append((rank, f))
        if not cands:
            return None
        cands.sort(key=lambda rf: rf[0])
        return cands[0][1]

    @staticmethod
    def _find_alma_contour(ctx: CallContext, ra: float, dec: float, size_arcmin: float, max_mb: float, label: str) -> Dict[str, Any]:
        """Bounded ALMA continuum-product lookup whose image COVERS the field.

        One TAP query for science pointings whose footprint (``s_region``)
        contains the field centre (no ORDER BY -- it stalls the NRAO proxy;
        2.6 s for the HUDF where the old point cone took 22 s), ranked per
        (MOUS, source) by a source-name match with the field, then by a field of
        view known to span the requested size, then by resolution. At most 4
        MOUS are listed via DataLink, each only while budget remains, and in
        each the covering sources are tried in rank order, taking only files
        whose parsed source name equals that source. ALMAbench 2026-10-15 A26:
        a point cone on s_ra/s_dec plus "any _sci image of the MOUS" picked
        scuba2-035, one of 64 17-arcsec pointings in 2016.1.01079.S, about 6
        arcmin from the HUDF. If the footprint query fails, a point cone is used
        and each pointing's s_fov is checked against the centre instead; a
        pointing without s_fov or coordinates is unverifiable, not absent."""
        from services.tool_budgets import remaining_seconds

        notes: List[str] = []
        cols = ("member_ous_uid, proposal_id, target_name, s_ra, s_dec, s_fov, band_list, spatial_resolution, "
                "dataproduct_type, data_rights")
        where = "science_observation = 'T' AND data_rights = 'Public'"
        footprint_q = (f"SELECT TOP 500 {cols} FROM ivoa.obscore WHERE "
                       f"CONTAINS(POINT('ICRS', {ra:.8f}, {dec:.8f}), s_region) = 1 AND {where}")
        radius_deg = max(size_arcmin / 2.0, 0.25) / 60.0
        cone_q = (f"SELECT TOP 500 {cols} FROM ivoa.obscore WHERE CONTAINS(POINT('ICRS', s_ra, s_dec), "
                  f"CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius_deg:.8f})) = 1 AND {where}")
        prov = ctx.service("alma_tap_provenance")
        prov["url"] = _ALMA_TAP_URL

        def search(q: str) -> pd.DataFrame:
            prov["query"] = q
            service = ctx.service("search_service").alminer_client._get_tap_service()
            try:
                return service.search(q, maxrec=500).to_table().to_pandas()
            except TypeError:
                return service.search(q).to_table().to_pandas()

        footprint = True
        try:
            df = search(footprint_q)
        except Exception as exc:
            left = remaining_seconds()
            if left is not None and left < 30:
                notes.append(f"ALMA archive query did not complete ({type(exc).__name__}: {str(exc)[:100]}): ALMA coverage UNKNOWN")
                return {"product": None, "notes": notes, "unknown": True}
            footprint = False
            try:
                df = search(cone_q)
            except Exception as exc2:
                notes.append(f"ALMA archive query did not complete ({type(exc2).__name__}: {str(exc2)[:100]}): ALMA coverage UNKNOWN")
                return {"product": None, "notes": notes, "unknown": True}
            notes.append(f"footprint query failed ({type(exc).__name__}); used a {radius_deg * 60:.2f} arcmin point cone and each pointing's field of view")
        if df.empty:
            notes.append(f"no public ALMA science pointing covers the centre of {label}" if footprint
                         else f"the ALMA archive lists no public science observations within {radius_deg * 60:.2f} arcmin of {label}")
            return {"product": None, "notes": notes, "unknown": False}
        for column in ("member_ous_uid", "proposal_id", "target_name"):
            if column in df:
                df[column] = df[column].map(lambda v: v.decode() if isinstance(v, bytes) else v)
        if "target_name" not in df:
            df["target_name"] = ""
        df["_res"] = pd.to_numeric(df.get("spatial_resolution"), errors="coerce")
        fov = pd.to_numeric(df["s_fov"], errors="coerce") * 60.0 if "s_fov" in df else pd.Series(float("nan"), index=df.index)
        if {"s_ra", "s_dec"} <= set(df.columns):
            sep = _sep_arcmin(ra, dec, pd.to_numeric(df["s_ra"], errors="coerce"), pd.to_numeric(df["s_dec"], errors="coerce"))
        else:
            sep = pd.Series(float("nan"), index=df.index)
        if not footprint:
            verifiable = sep.notna() & fov.notna()
            covers = verifiable & (sep <= fov / 2.0)
            if not covers.any():
                n_unverifiable = int((~verifiable).sum())
                if n_unverifiable:
                    notes.append(f"{n_unverifiable} ALMA pointing(s) near {label} lack a field of view or position, so whether "
                                 "ALMA covers its centre is UNKNOWN")
                    return {"product": None, "notes": notes, "unknown": True}
                notes.append(f"{df['member_ous_uid'].nunique()} ALMA MOUS have pointings near {label}, but none covers its centre")
                return {"product": None, "notes": notes, "unknown": False}
            df, fov = df[covers], fov[covers]
        df["_key"] = df["target_name"].map(_source_key)
        unnamed = df["_key"] == ""
        if unnamed.all():
            notes.append(f"the ALMA pointings covering {label} carry no source name, so their images cannot be matched: ALMA product UNKNOWN")
            return {"product": None, "notes": notes, "unknown": True}
        df, fov = df[~unnamed], fov[~unnamed]
        want = _FIELD_NAME_ALIASES.get(_norm_name(label), {_norm_name(label)}) - {""}

        def name_rank(target: Any) -> int:
            t = _norm_name(target)
            return 0 if t and any(w in t or (len(t) >= 3 and t in w) for w in want) else 1

        df["_name"] = df["target_name"].map(name_rank)
        # A missing field of view is not evidence that it spans the request.
        df["_span"] = (~(fov.notna() & (fov >= 0.8 * float(size_arcmin)))).astype(int)
        df["_fov"] = fov
        ranked = df.sort_values(["_name", "_span", "_res"], na_position="last").drop_duplicates(["member_ous_uid", "_key"])
        datalink = ctx.services.get("datalink_client")
        if datalink is None:
            notes.append("DataLink client unavailable: cannot list ALMA products")
            return {"product": None, "notes": notes, "unknown": True}
        # Sources are tried in GLOBAL rank order (a better source in another
        # MOUS beats a worse one in a MOUS already listed; guard CX-13), each
        # MOUS listed once. At most four MOUS are listed, not three: early-cycle
        # MOUS (the 2012.1.00173.S HUDF mosaic) carry no imaging products.
        listings: Dict[str, Optional[List[Dict[str, Any]]]] = {}
        stopped = False
        for _, row in ranked.iterrows():
            uid = row["member_ous_uid"]
            if uid not in listings:
                if len(listings) >= 4 or stopped:
                    continue
                left = remaining_seconds()
                if left is not None and left < 25:
                    notes.append(f"stopped listing ALMA products after {len(listings)} MOUS (tool budget)")
                    stopped = True
                    continue
                listing = datalink.list_files(mous_uid=str(uid))
                listings[uid] = listing.get("files", []) if listing.get("success") else None
            if not listings[uid]:
                continue
            pick = ArchiveOverlay._pick_science_continuum(listings[uid], max_mb, target=row["_key"])
            if pick is not None:
                return {"product": {**pick, "member_ous_uid": uid, "proposal_id": row.get("proposal_id"),
                                    "target_name": row.get("target_name"),
                                    "field_of_view_arcmin": None if pd.isna(row["_fov"]) else round(float(row["_fov"]), 2),
                                    "covers_field_centre": True,
                                    "footprint_check": "s_region contains the field centre" if footprint else "pointing s_fov covers the field centre"},
                        "notes": notes, "unknown": False}
        listed = len(listings)
        n_mous = ranked["member_ous_uid"].nunique()
        notes.append(f"{n_mous} ALMA MOUS cover {label} but none of the {listed} listed has a continuum FITS of the covering source <= {max_mb:g} MB")
        return {"product": None, "notes": notes, "unknown": False}


# ═════════════════════════════════════════════════════════════════════════
# 8. code_recipe
# ═════════════════════════════════════════════════════════════════════════
RECIPES_DIR = Path(__file__).resolve().parents[1] / "docs" / "recipes"

_RECIPE_TASKS = {
    "cone_search": ["cone", "region", "position", "coordinates", "radius"],
    "object_search": ["object", "target", "source", "name", "by name"],
    "adql": ["adql", "tap", "sql", "query_tap", "custom query"],
    "datalink": ["datalink", "files", "list files", "download", "products", "fits"],
    "alminer": ["alminer", "conesearch", "run_query", "keysearch"],
    "dataset_list": ["list", "dataset", "mous", "released", "release date", "before", "all observations", "every observation"],
}
# Recipes of the same library tried when nothing matches the task.
_LIBRARY_DEFAULT_RECIPES = {
    "astroquery": ["object_search", "adql"],
    "pyvo": ["adql", "cone_search"],
    "alminer": ["alminer"],
}
_RECIPE_DOCS = {
    "astroquery": "https://astroquery.readthedocs.io/en/latest/alma/alma.html",
    "pyvo": "https://pyvo.readthedocs.io/en/latest/dal/index.html#tap",
    "alminer": "https://alminer.readthedocs.io/en/latest/",
}
# Documented public calls (checked against the installed libraries in
# tests/unit/test_code_recipes.py) and real ivoa.obscore columns. Without a
# recipe the model refused to write any code (ALMAbench 2026-10-15 A19-A21);
# standard calls from this list are fine, invented ones are not.
_STANDARD_API_CALLS = {
    "astroquery": ["Alma.query_object(name, public=...)", "Alma.query_region(SkyCoord, radius=Quantity, public=...)",
                   "Alma.query_tap(adql)", "Alma.get_data_info(member_ous_uids)", "Alma.download_files(urls)"],
    "pyvo": ["pyvo.dal.TAPService('https://almascience.nrao.edu/tap')", "service.search(adql, maxrec=N)",
             "result.to_table()"],
    "alminer": ["alminer.target([names], search_radius=arcmin, public=...)", "alminer.conesearch(ra=, dec=, search_radius=arcmin)",
                "alminer.keysearch({'column': [values]})", "alminer.run_query(adql)", "alminer.summary(df)",
                "alminer.download_data(df, fitsonly=True, dryrun=True)"],
}
OBSCORE_STANDARD_COLUMNS = ["proposal_id", "member_ous_uid", "target_name", "s_ra", "s_dec", "s_fov", "band_list",
                            "frequency", "spatial_resolution", "t_exptime", "obs_release_date", "data_rights",
                            "science_observation", "pi_name", "scientific_category", "antenna_arrays"]


def _standard_api_fallback(lib: str) -> Dict[str, Any]:
    return {
        "documentation": _RECIPE_DOCS.get(lib),
        "standard_calls": _STANDARD_API_CALLS.get(lib, []),
        "obscore_columns": OBSCORE_STANDARD_COLUMNS,
        "instruction": ("No tested recipe fits, but you SHOULD still give working code: write a short, standard snippet with "
                        "the documented calls listed here and real ivoa.obscore columns (table ivoa.obscore; band_list is a "
                        "space-separated token list). Do not use any other API name. Say it is written from the library "
                        "documentation, not a tested recipe. If a Python execution tool is available, run it once first."),
    }


class CodeRecipeInput(_In):
    task: str = Field(description="what the code should do, e.g. 'cone search', 'query by object name', 'ADQL via TAP', 'list and download DataLink files'")
    library: str = Field(default="astroquery", description="astroquery | pyvo | alminer")


class CodeRecipe(BaseCapability):
    name = "code_recipe"
    description = (
        "Return a TESTED, curated Python snippet for querying the ALMA Science Archive (astroquery.alma, pyvo TAP, or "
        "ALMiner): cone/object search, ADQL via query_tap, DataLink listing and download, ALMiner target/conesearch/"
        "run_query, and a complete 'list every dataset (MOUS) of a source in a band released before a date' script. Snippets come from docs/recipes/*.py which are exercised by the test suite (recorded responses; live "
        "weekly) -- never invent API names such as Alma.get_data_links or query_sql. When no recipe fits, the result "
        "names the library's documented calls so you can still write standard code. Use for 'how do I query ALMA "
        "with Python / astroquery / TAP / ALMiner'."
    )
    category = "general"
    InputModel = CodeRecipeInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        lib = str(inp.library or "astroquery").strip().lower()
        if lib not in {"astroquery", "pyvo", "alminer"}:
            return _native({"success": False, "error": "library must be astroquery, pyvo or alminer"})
        if not RECIPES_DIR.is_dir() or not any(RECIPES_DIR.glob("*.py")):
            # A deployment without docs/recipes is broken, not a coverage gap
            # (ALMAbench 2026-10-15: the benchmark snapshots had no docs/ and
            # every code question came back "no curated recipe").
            return _native({"success": False, "status": "infrastructure_failure",
                            "error": f"the recipe library {RECIPES_DIR} is missing from this deployment",
                            "fallback": _standard_api_fallback(lib)})
        task_text = str(inp.task or "").lower()
        scores = {k: sum(1 for kw in kws if kw in task_text) for k, kws in _RECIPE_TASKS.items()}
        ordered = [k for k, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0]
        recipes = self._load_recipes(lib, ordered)
        nearest = False
        if not recipes:
            # No recipe for this task in this library (or no task keyword at
            # all): the library's own recipes come next, flagged as nearest (a
            # pyvo cone or ADQL recipe answers "query by object name" with pyvo
            # better than no code; ALMAbench 2026-10-15 A21; guard CX-10).
            recipes = self._load_recipes(lib, [k for k in _LIBRARY_DEFAULT_RECIPES[lib] if k not in ordered])
            nearest = bool(recipes)
        if not recipes:
            return _native({"success": False, "status": "coverage_gap", "error": f"no curated recipe for library={lib} task={inp.task!r}",
                            "available": sorted(p.name for p in RECIPES_DIR.glob("*.py")),
                            "fallback": _standard_api_fallback(lib)})
        out = {
            "success": True, "status": "ok", "library": lib, "recipes": recipes[:2],
            "documentation": _RECIPE_DOCS[lib],
            "note": ("Return each `code` VERBATIM in a ```python block (it is complete and runnable); do not rename functions or "
                     "re-type any metadata. API names in these files are the tested ones."),
        }
        if nearest:
            out["nearest_recipe"] = True
            out["note"] += (f" No recipe matches the task {inp.task!r} exactly; these are the closest tested {lib} recipes. "
                            "Show the tested recipe verbatim first, then, if the task needs it, an adapted version labelled "
                            "'adapted from the tested recipe' that uses the same calls (only the target, columns or WHERE "
                            "clause change) and says what changed.")
        return _native(out)

    @staticmethod
    def _load_recipes(lib: str, keys: List[str]) -> List[Dict[str, Any]]:
        recipes: List[Dict[str, Any]] = []
        seen: set = set()
        for key in keys:
            path = RECIPES_DIR / f"{lib}_{key}.py"
            if not path.exists():
                if lib == "alminer":
                    path = RECIPES_DIR / "alminer_alminer.py"
                elif key in ("adql",) and lib == "astroquery":
                    path = RECIPES_DIR / "astroquery_adql.py"
                if not path.exists():
                    continue
            if path.name in seen:
                continue
            seen.add(path.name)
            text = path.read_text(encoding="utf-8")
            header = re.search(r'"""(.*?)"""', text, re.S)
            meta = {}
            if header:
                for line in header.group(1).splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        mkey = k.strip().lower()  # own name: `key` is the recipe task id (guard CX-08)
                        # Repeated keys (multi-line 'notes:') accumulate; only the last line used to survive (guard CX-03).
                        meta[mkey] = f"{meta[mkey]} {v.strip()}" if mkey in meta and mkey == "notes" else v.strip()
            # The user-facing snippet is the RUNNABLE code only: the metadata
            # docstring stays in its own fields. Live 2026-09-23 D15: the model
            # re-typed the docstring header and dropped its closing quotes, so
            # the displayed pyvo snippet would not even parse (and it leaked a
            # tests/ path). A short leading comment keeps the provenance.
            code = text[header.end():].lstrip("\n") if header else text
            # The header states what THIS recipe was tested against (the
            # DataLink recipe is not a TAP recipe; guard CX-14).
            against = str(meta.get("tested_against") or "").split(" and recorded responses")[0].strip()
            code = (f"# ALMA archive recipe ({lib}; last tested {meta.get('last_tested') or 'n/a'}"
                    + (f" against {against}" if against else "") + ")\n" + code.rstrip() + "\n")
            recipes.append({"task": key, "library": lib, "file": f"docs/recipes/{path.name}", "code": code,
                            "last_tested": meta.get("last_tested"), "tested_against": meta.get("tested_against"),
                            "library_requirement": meta.get("library"),
                            "notes": meta.get("notes")})
        return recipes


# ═════════════════════════════════════════════════════════════════════════
# 9. alma_reference
# ═════════════════════════════════════════════════════════════════════════
_REFERENCE_TOPICS = {
    "bands": "ALMA receiver bands frequency ranges Band 1 2 3 4 5 6 7 8 9 10 GHz",
    "configurations": "ALMA 12-m Array configurations C-1 to C-10 maximum baseline angular resolution maximum recoverable scale",
    "cycles": "ALMA cycle project code year mapping",
    "correlator": "ALMA Baseline Correlator BLC ACA correlator spectral windows channels bandwidth modes FDM TDM 64-antenna",
    "bandwidth_switching": "bandwidth switching BWSW phase calibrator narrow spectral windows aggregate bandwidth 937.5 MHz differential gain calibration",
    "proprietary_period": "ALMA proprietary period 12 months data rights public release policy DDT 6 months",
    "data_products": "ALMA archive data products delivered calibrated measurement set imaging products FITS weblog README QA2 report scripts",
    "weblog": "ALMA pipeline weblog QA2 report contents stages",
    "casa_version": "ALMA CASA version pipeline version used README weblog casa_pipeline manifest",
    "manifest": "ALMA data delivery manifest member_ous package contents README",
    "sensitivity": "ALMA sensitivity calculator radiometer equation Tsys precipitable water vapour PWV octiles",
}


class AlmaReferenceInput(_In):
    topic: str = Field(description="bands | configurations | cycles | correlator | bandwidth_switching | proprietary_period | data_products | weblog | casa_version | manifest | sensitivity | free text")
    cycle: Optional[int] = None
    frequency_ghz: Optional[float] = None
    k: int = 6


class AlmaReference(BaseCapability):
    name = "alma_reference"
    description = (
        "Forced documentation lookup for ALMA facts with citations: the official cycle-labelled tables (bands with frequency "
        "edges, configurations with baselines and resolutions, cycle/project-code mapping) PLUS the Technical Handbook / "
        "Proposer's Guide / archive documentation passages on the correlator, bandwidth switching, proprietary period, "
        "delivered data products, weblog, CASA/pipeline version and delivery manifest — each passage with its source file "
        "and page. Use for any knowledge question about ALMA instead of answering from memory."
    )
    category = "general"
    InputModel = AlmaReferenceInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        topic_raw = str(inp.topic or "").strip()
        topic = topic_raw.lower().replace(" ", "_").replace("-", "_")
        out: Dict[str, Any] = {"success": True, "status": "ok", "topic": topic_raw, "tables": None, "passages": [], "citations": []}
        if topic in {"bands", "band", "configurations", "configuration", "cycles", "cycle"}:
            key = {"band": "bands", "configuration": "configurations", "cycle": "cycles"}.get(topic, topic)
            try:
                from services.alma_reference import alma_reference_table

                out["tables"] = alma_reference_table(key, cycle=inp.cycle, frequency_ghz=inp.frequency_ghz)
                out["citations"].extend([s.get("url") or s.get("title") for s in out["tables"].get("sources", []) if isinstance(s, dict)])
            except Exception as exc:
                out["table_error"] = f"{type(exc).__name__}: {exc}"
        query = _REFERENCE_TOPICS.get(topic, topic_raw)
        rag = ctx.services.get("rag_service")
        if rag is not None:
            try:
                docs, diag = rag.search_with_diagnostics(query, k=max(1, min(int(inp.k), 10)))
                for d in docs:
                    meta = getattr(d, "metadata", {}) or {}
                    src = meta.get("source_file") or meta.get("source") or meta.get("filename") or "documentation"
                    page = meta.get("page") or meta.get("page_number") or meta.get("page_label")
                    cite = f"[Source: {src}" + (f", Page {page}" if page else "") + "]"
                    out["passages"].append({"text": str(getattr(d, "page_content", ""))[:1200], "source": src, "page": page, "cite_as": cite,
                                            "score": meta.get("_semantic_score") or meta.get("score")})
                    if cite not in out["citations"]:
                        out["citations"].append(cite)
                if isinstance(diag, dict) and diag.get("year_conflict"):
                    out["freshness_notice"] = diag["year_conflict"].get("message")
            except Exception as exc:
                out["documentation_error"] = f"documentation search failed: {type(exc).__name__}: {str(exc)[:160]}"
        else:
            out["documentation_error"] = "documentation search service unavailable in this context"
        if out["tables"] is None and not out["passages"]:
            out.update(success=False, status="coverage_gap", error=f"no reference table or documentation passage found for {topic_raw!r}")
        out["note"] = "Quote the tables and cite passages with their cite_as tag; do not add numbers that are not in them."
        return _native(out)


CAPABILITIES: List[BaseCapability] = [
    AlmaProjectCensus(),
    AlmaSourceSummary(),
    AlmaPublicBandStatus(),
    AlmaArchiveLink(),
    AlmaBibliography(),
    CrossArchiveMatch(),
    ArchiveOverlay(),
    CodeRecipe(),
    AlmaReference(),
]
