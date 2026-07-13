"""
capabilities/alma.py — the ALMA / archive-search family as transport-pure
capabilities (P1 family migration #2, after datalab).

Logic relocated VERBATIM from ``core/agent.py`` (the ``_search_by_*`` /
``_search_cadc`` / ``_query_alma_science_archive`` / cross-match / line-coverage
/ ``_filter_results`` / ``_merged_plot_alma_results`` methods plus the
``_filter_by_scan_intent`` / ``_tap_obscore_dataframe`` / ``_redshifted_line_where``
helpers). The only changes are structural:

  * clients/services arrive via ``CallContext.services`` (``search_service``,
    ``analysis_service``, ``plotting_service``, ``mast_client``) — never
    ``self.*`` on the agent;
  * the agent's thread-local result state (``last_search_results`` /
    ``last_run_result``) is reached ONLY through injected bound accessors
    (``get_/set_last_search_results``, ``get_/set_last_run_result``) so the
    request-scoped TLS semantics live agent-side and the capability stays pure;
  * the legacy ``print()`` filter traces go through an injected ``console_log``
    callable (the agent binds it to ``print``) — byte-identical stdout, no
    ``print()`` in the capability;
  * the legacy ``self._last_alma_tap_query`` / ``_last_alma_tap_url`` plain
    instance attributes become an injected agent-owned mutable dict
    (``alma_tap_provenance``, keys ``query``/``url``) — same instance-wide,
    last-write-wins visibility;
  * SIMBAD fallback resolution goes through the injected ``resolve_target``
    (the agent's bound ``_resolve_target``) — byte-identical resolution;
  * every path returns a :class:`ToolResult` whose ``native`` payload is the
    exact legacy output dict (byte-parity). Exceptions the legacy method did
    NOT catch still propagate (the adapter does not swallow them), so the
    raise-vs-error-dict surface is unchanged.

Known preserved-verbatim legacy quirks (do NOT "fix" without a benchmark gate):
  * ``SearchByTarget``'s positional fallback gate reads ``resolved.get("ra")``
    but ``resolve_target`` returns ``ra_deg``/``dec_deg`` — the fallback is
    dead (logged in docs/v2 OPEN_ISSUES as a discovered-but-unfixed issue).
  * ``QueryAlmaScienceArchive``'s ``require_same_project is False`` identity
    check (a stringy "false" is NOT ``False``) — the field is typed ``Any``
    so pydantic cannot coerce it and change the outcome. Same reasoning for
    every truthiness-gated boolean below (``dry_run``, ``dark_mode``,
    ``require_all_archives``, ``include_adql``, ``public_only``).
  * ``SearchCadc``'s runtime default ``radius=0.02`` vs. its registration
    schema's documented default 0.05 — the signature wins, as it always has.

Logging: the legacy methods logged through ``core.logger``'s loguru logger and
four carried ``@log_tool`` (Langfuse spans). Capabilities use a stdlib logger
(no core.* import); the agent re-applies ``log_tool`` at the adapter boundary
for the four tools that had it (see ``QuasarAgent._alma_tool_fn(log_name=...)``),
so tool spans/entry-exit logs keep their legacy names.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, CallContext, ToolResult
from services.alma_science_queries import (
    LINE_REST_FREQ_GHZ,
    bandwidth_switching_candidates,
    filter_band as science_filter_band,
    filter_resolution as science_filter_resolution,
    line_names_for_species,
    normalize_target_alias,
    projects_covering_all_lines,
    projects_with_array_combo,
    project_prefix_where,
    redshifted_line_projects,
    select_obscore_query,
    summarize_projects,
)
from services.cross_archive_matcher import (
    alma_bulk_cone_adql,
    attach_nearest_source,
    normalize_source_catalog,
    summarize_cross_archive_matches,
)

logger = logging.getLogger(__name__)


def _native(out: Dict[str, Any]) -> ToolResult:
    """Wrap a legacy output dict as a byte-parity ToolResult."""
    ok = bool(isinstance(out, dict) and out.get("success"))
    err = out.get("error") if isinstance(out, dict) else None
    return ToolResult(
        success=ok,
        error=(str(err) if (err is not None and not ok) else None),
        native=out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (relocated verbatim from agent.py)
# ─────────────────────────────────────────────────────────────────────────────
def filter_by_scan_intent(results: pd.DataFrame, scan_intent: Optional[Any]) -> Tuple[pd.DataFrame, str]:
    """Filter ALMA rows by the archive scan_intent column when requested."""
    raw = str(scan_intent or "").strip()
    if not raw or results is None or results.empty:
        return results, ""

    intent_col = next((c for c in ["scan_intent", "Scan Intent", "intent"] if c in results.columns), None)
    if not intent_col:
        return results, ""

    known_intents = re.findall(
        r"\b(TARGET|BANDPASS|PHASE|FLUX|WVR|CHECK|POINTING|FOCUS|AMPLITUDE|ATMOSPHERE)\b",
        raw.upper(),
    )
    requested = known_intents or [part.strip().upper() for part in re.split(r"[,;/]+|\s+and\s+", raw) if part.strip()]
    requested = [part for part in requested if part and part not in {"ONLY", "EXCLUDE", "EXCLUDING", "CALIBRATORS"}]
    if not requested:
        return results, ""

    mask = results[intent_col].astype(str).str.upper().apply(
        lambda value: any(intent in value for intent in requested)
    )
    label = "Scan Intent " + ",".join(requested)
    return results[mask].copy(), label


def _tap_obscore_dataframe(
    where_clause: str,
    *,
    max_results: int = 5000,
    order_by: str = "proposal_id",
    ctx: CallContext,
) -> pd.DataFrame:
    query = select_obscore_query(where_clause, top=max_results, order_by=order_by)
    prov = ctx.service("alma_tap_provenance")
    prov["query"] = query
    prov["url"] = "https://almascience.nrao.edu/tap"
    search_service = ctx.service("search_service")
    service = search_service.alminer_client._get_tap_service()
    result = service.search(query)
    df = result.to_table().to_pandas()
    if hasattr(search_service.alminer_client, "_standardize_columns"):
        return search_service.alminer_client._standardize_columns(df)
    return df


def _redshifted_line_where(
    rest_species: str,
    redshift_min: float,
    redshift_max: float,
    science_category: str = "",
) -> Tuple[str, List[str]]:
    z_min = float(redshift_min)
    z_max = float(redshift_max)
    line_names = line_names_for_species(rest_species or "CO")
    conditions = []
    for line_name in line_names:
        rest_freq = LINE_REST_FREQ_GHZ[line_name]
        nu_low = rest_freq / (1.0 + max(z_min, z_max))
        nu_high = rest_freq / (1.0 + min(z_min, z_max))
        conditions.append(
            f"((frequency - 0.5*bandwidth/1e9) < {nu_high:.6f} "
            f"AND (frequency + 0.5*bandwidth/1e9) > {nu_low:.6f})"
        )
    where = "(" + " OR ".join(conditions) + ")"
    if science_category:
        safe_category = str(science_category).replace("'", "''")
        where += f" AND LOWER(scientific_category) LIKE '%{safe_category.lower()}%'"
    else:
        where += (
            " AND (LOWER(scientific_category) LIKE '%galaxy%' "
            "OR LOWER(scientific_category) LIKE '%cosmology%' "
            "OR LOWER(scientific_category) LIKE '%active%')"
        )
    return where, line_names


# ─────────────────────────────────────────────────────────────────────────────
# Input models + capabilities (one Input right above its capability; every
# truthiness-gated boolean is typed Any so pydantic cannot coerce a stringy
# "false" and flip a legacy truthiness/identity gate)
# ─────────────────────────────────────────────────────────────────────────────
class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SearchByPositionInput(_In):
    # Required-but-nullable (no default): a MISSING key is invalid (as legacy),
    # but an explicit JSON null must flow into the legacy body — the inline
    # method ran cone_search(None, ...) inside its try and returned the same
    # typed error dict. Same pattern for every required field in this family
    # (established by DensityVettingInput.radius_deg). (guard CX-01)
    ra: Optional[float]
    dec: Optional[float]
    radius: Optional[float] = 0.5
    facility: Optional[str] = None
    band: Any = None
    scan_intent: Any = None
    max_results: Optional[int] = 100


class SearchByPosition(BaseCapability):
    name = "search_by_position"
    description = "Search NRAO archives by sky position (cone search)"
    category = "general"
    InputModel = SearchByPositionInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        search_service = ctx.service("search_service")
        _log = ctx.service("console_log")
        ra, dec, radius = inp.ra, inp.dec, inp.radius
        facility, band, scan_intent, max_results = inp.facility, inp.band, inp.scan_intent, inp.max_results

        facility_label = (facility or "ALMA").strip().upper()
        if facility_label in ("EVLA", "JVLA"):
            facility_label = "VLA"
        if facility_label not in ("VLA", "VLBA", "GBT"):
            facility_label = "ALMA"

        try:
            results = search_service.cone_search(
                ra, dec, radius, facility, max_results
            )

            # If the archive query failed, surface a typed error rather than
            # masking the outage as a confirmed empty result. (C3)
            _pos_err = results.attrs.get("quasar_error") if hasattr(results, "attrs") else None
            if results.empty and _pos_err:
                return _native({
                    "success": False,
                    "error": _pos_err,
                    "ra": ra, "dec": dec, "radius_deg": radius,
                    "note": (
                        "The archive query did not complete, so it is unknown whether "
                        "data exist at this position — this is an archive/service error, "
                        "not a confirmed 'no data' result."
                    ),
                })

            # Post-filter by band if specified
            if band is not None and not results.empty:
                import re as _re_band
                band_vals = []
                if isinstance(band, str):
                    parts = _re_band.split(r'[,\s]+and[\s]+|[,\s]+', band.strip())
                    band_vals = [int(p) for p in parts if p.strip().isdigit()]
                elif isinstance(band, (int, float)):
                    band_vals = [int(band)]
                if band_vals:
                    b_col = next((c for c in ['band_list', 'Band', 'band'] if c in results.columns), None)
                    if b_col:
                        before = len(results)
                        results = results[results[b_col].astype(str).apply(
                            lambda x: any(str(b) in [v.strip() for v in x.split(',')] for b in band_vals)
                        )]
                        _log(f"[FILTER] Band {band}: {before} → {len(results)} rows")

            scan_filter_label = ""
            if facility_label == "ALMA" and scan_intent:
                before = len(results)
                results, scan_filter_label = filter_by_scan_intent(results, scan_intent)
                if scan_filter_label:
                    _log(f"[FILTER] {scan_filter_label}: {before} → {len(results)} rows")

            ctx.service("set_last_search_results")(results)
            ctx.service("set_last_run_result")({
                "type": "data",
                "data": results,
                "source": facility_label,
                "filter_label": f"{facility_label} › position" + (f" [{scan_filter_label}]" if scan_filter_label else ""),
                "tool_name": "search_by_position",
            })

            # Include top MOUS UIDs + access URLs so Conductor subtasks
            # can use them for list_alma_files / render_fits_image.
            top_mous = []
            if hasattr(results, 'columns') and 'member_ous_uid' in results.columns:
                top_mous = results['member_ous_uid'].dropna().unique()[:5].tolist()
            top_urls = []
            if hasattr(results, 'columns') and 'access_url' in results.columns:
                top_urls = results['access_url'].dropna().head(5).tolist()

            # Extract unique project codes
            top_projects = []
            if hasattr(results, 'columns'):
                project_col = next((col for col in ["proposal_id", "project_code", "Project", "obs_publisher_did"] if col in results.columns), None)
                if project_col:
                    raw_projects = results[project_col].dropna().astype(str).str.strip().unique()
                    import re as _re_proj
                    proj_regex = _re_proj.compile(r"\b\d{4}\.\d\.\d{5}\.[A-Za-z]\b")
                    for p in raw_projects:
                        match = proj_regex.search(p)
                        if match:
                            code = match.group(0).upper()
                            if code not in top_projects:
                                top_projects.append(code)
                        elif len(p) >= 10 and '.' in p:
                            p_upper = p.upper()
                            if p_upper not in top_projects:
                                top_projects.append(p_upper)
                    top_projects = top_projects[:3]

            return _native({
                "success": True,
                "total_results": len(results),
                "ra": ra, "dec": dec, "radius_deg": radius,
                "top_mous_uids": top_mous,
                "top_access_urls": top_urls,
                "top_project_codes": top_projects,
                "filters_applied": [scan_filter_label] if scan_filter_label else [],
                "note": f"Found {len(results)} observations. Full dataset with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SearchByTargetInput(_In):
    # Required-but-nullable — an explicit null reaches re.split(...) inside the
    # legacy try and maps to the identical typed error. (guard CX-01)
    target_name: Optional[str]
    facility: Optional[str] = None
    date_range: Optional[str] = None
    max_results: Optional[int] = 100
    band: Any = None
    max_resolution: Optional[float] = None
    min_resolution: Optional[float] = None
    min_freq_ghz: Optional[float] = None
    max_freq_ghz: Optional[float] = None
    min_exp_s: Optional[float] = None
    scan_intent: Any = None
    public_only: Any = False  # accepted-and-unused, verbatim with the legacy signature


class SearchByTarget(BaseCapability):
    name = "search_by_target"
    description = (
        "Search the ALMA archive (default) by target name. Supports multiple targets "
        "separated by 'and' or comma (e.g. 'M87 and Sz65' or 'M87, NGC 1068').\n"
        "Pass facility='VLA', 'VLBA', or 'GBT' to search the NRAO archive instead — "
        "ONLY when the user explicitly asks for those telescopes.\n"
        "CRITICAL: ONLY pass optional filter parameters (band, resolution, frequency, scan_intent) "
        "if the user EXPLICITLY mentions them. Do NOT invent default values. "
        "If the user just says 'Find ALMA data of M87', pass ONLY target_name='M87' "
        "with NO other parameters — this returns ALL observations across all bands.\n"
        "Only pass band=6 if the user says 'Band 6'. Only pass max_resolution if "
        "the user specifies a resolution constraint. Omitting a filter means 'no filter'."
    )
    category = "general"
    InputModel = SearchByTargetInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        search_service = ctx.service("search_service")
        _log = ctx.service("console_log")
        target_name = inp.target_name
        facility, date_range, max_results = inp.facility, inp.date_range, inp.max_results
        band = inp.band
        max_resolution, min_resolution = inp.max_resolution, inp.min_resolution
        min_freq_ghz, max_freq_ghz, min_exp_s = inp.min_freq_ghz, inp.max_freq_ghz, inp.min_exp_s
        scan_intent = inp.scan_intent

        # ── Normalize facility for routing + result labeling ──
        facility_label = (facility or "ALMA").strip().upper()
        if facility_label in ("EVLA", "JVLA"):
            facility_label = "VLA"
        if facility_label not in ("VLA", "VLBA", "GBT"):
            facility_label = "ALMA"

        # ── Normalize band to a list (multi-band support) ──
        band_list_input: List[int] = []
        if band is not None:
            if isinstance(band, list):
                # LLM passed a list (e.g. [6, 7])
                band_list_input = [int(b) for b in band if str(b).strip().isdigit() and 1 <= int(b) <= 10]
            elif isinstance(band, (int, float)):
                # Single integer
                b = int(band)
                if 1 <= b <= 10:
                    band_list_input = [b]
                else:
                    _log(f"[FILTER] Ignoring invalid band={band} (must be 3-10)")
            elif isinstance(band, str):
                # String like "6" or "6,7" or "6 and 7"
                import re as _re_band
                parts = _re_band.split(r'[,\s]+and[\s]+|[,\s]+', band.strip())
                for p in parts:
                    p = p.strip()
                    if p.isdigit():
                        b = int(p)
                        if 1 <= b <= 10:
                            band_list_input.append(b)
            else:
                _log(f"[FILTER] Ignoring unrecognized band={band}")

        # ── Safety: ignore near-zero/zero min values (LLM default filling) ──
        # The LLM often fills 0 for optional params despite being told not to.
        # A value of 0 for resolution/frequency/exptime means "no filter".
        if min_resolution is not None and min_resolution <= 0:
            min_resolution = None
        if max_resolution is not None and max_resolution <= 0:
            max_resolution = None  # 0 arcsec = impossible, treat as no filter
        if min_freq_ghz is not None and min_freq_ghz <= 0:
            min_freq_ghz = None
        if max_freq_ghz is not None and max_freq_ghz <= 0:
            max_freq_ghz = None
        if min_exp_s is not None and min_exp_s <= 0:
            min_exp_s = None
        # Ignore absurdly wide max values (LLM defaults)
        if max_resolution is not None and max_resolution >= 100:
            max_resolution = None
        if max_freq_ghz is not None and max_freq_ghz >= 5000:
            max_freq_ghz = None

        try:
            # ── Multi-target support ──────────────────────────────────
            # Detect "M87 and Sz65" or "M87, NGC 1068" patterns
            # Also handles per-target band specs like:
            #   "M87 in band 6, Sz65 in band 7"  →  per-target bands
            #   "M87, Sz65, NGC23"               →  shared bands (from band= param)
            import re as _re
            raw_names = [n.strip() for n in _re.split(r'\s+and\s+|\s*,\s*', target_name) if n.strip()]

            # ── Parse per-target band specifications ──────────────────
            # If target_name contains inline band specs (e.g. "M87 in band 6"),
            # extract them so each target gets its own filter.
            _per_target_specs = []  # list of (name, [bands]) tuples
            _has_per_target_bands = False
            _inline_band_pattern = _re.compile(
                r'^(.+?)\s+(?:in\s+)?band\s*([\d,\s]+(?:\s*(?:and|,)\s*\d+)*)$',
                _re.IGNORECASE,
            )
            for raw in raw_names:
                m = _inline_band_pattern.match(raw.strip())
                if m:
                    _tgt_name = m.group(1).strip()
                    _band_str = m.group(2)
                    _bands = [int(b.strip()) for b in _re.split(r'[,\s]+and[\s]+|[,\s]+', _band_str) if b.strip().isdigit()]
                    _bands = [b for b in _bands if 1 <= b <= 10]
                    _per_target_specs.append((_tgt_name, _bands))
                    if _bands:
                        _has_per_target_bands = True
                else:
                    _per_target_specs.append((raw.strip(), []))

            if _has_per_target_bands:
                # Per-target band mode: search each target with its own band filter
                all_frames = []
                searched_names = []
                _sub_errs = []  # archive errors per target (C3 — don't mask outages)
                for _tgt, _tgt_bands in _per_target_specs:
                    if not _tgt:
                        continue
                    try:
                        df = search_service.search_by_target(
                            _tgt, facility, date_range, max_results
                        )
                        _e = df.attrs.get("quasar_error") if hasattr(df, "attrs") else None
                        if _e:
                            _sub_errs.append(f"{_tgt}: {_e}")
                        if not df.empty and _tgt_bands:
                            # Apply per-target band filter
                            b_col = next((c for c in ["band_list", "Band", "band"] if c in df.columns), None)
                            if b_col:
                                df = df[df[b_col].astype(str).str.split(",").apply(
                                    lambda bands: any(str(b).strip() == x.strip() for x in bands for b in _tgt_bands)
                                )]
                        if not df.empty:
                            all_frames.append(df)
                            band_label = ",".join(str(b) for b in _tgt_bands) if _tgt_bands else "all"
                            searched_names.append(_tgt)
                            _log(f"[PER-TARGET] '{_tgt}' band={band_label} → {len(df)} results")
                        else:
                            _log(f"[PER-TARGET] '{_tgt}' → 0 results")
                    except Exception as e:
                        _sub_errs.append(f"{_tgt}: {e}")
                        _log(f"[PER-TARGET] '{_tgt}' failed: {e}")

                if all_frames:
                    results = pd.concat(all_frames, ignore_index=True)
                    target_name = " + ".join(searched_names)
                else:
                    results = pd.DataFrame()
                    if _sub_errs:  # preserve the archive-error signal (C3)
                        results.attrs["quasar_error"] = "; ".join(_sub_errs)
                # Skip shared band filtering below — bands already applied per target
                band_list_input = []

            elif len(raw_names) > 1:
                # Search each target independently, concatenate results
                all_frames = []
                searched_names = []
                _sub_errs = []  # archive errors per target (C3 — don't mask outages)
                for name in raw_names[:10]:  # Cap at 10 targets
                    try:
                        df = search_service.search_by_target(
                            name, facility, date_range, max_results
                        )
                        _e = df.attrs.get("quasar_error") if hasattr(df, "attrs") else None
                        if _e:
                            _sub_errs.append(f"{name}: {_e}")
                        if not df.empty:
                            all_frames.append(df)
                            searched_names.append(name)
                            _log(f"[MULTI] '{name}' → {len(df)} results")
                        else:
                            _log(f"[MULTI] '{name}' → 0 results")
                    except Exception as e:
                        _sub_errs.append(f"{name}: {e}")
                        _log(f"[MULTI] '{name}' failed: {e}")

                if all_frames:
                    results = pd.concat(all_frames, ignore_index=True)
                    target_name = " + ".join(searched_names)  # update label
                else:
                    results = pd.DataFrame()
                    if _sub_errs:  # preserve the archive-error signal (C3)
                        results.attrs["quasar_error"] = "; ".join(_sub_errs)
            else:
                results = search_service.search_by_target(
                    target_name, facility, date_range, max_results
                )

            if results.empty:
                # ── Automatic positional fallback ─────────────────────
                # The name-based search uses a tight radius (0.05°).
                # Many ALMA observations have offset pointing centers, so
                # retry with a wider cone search to avoid losing results
                # (and critically, to keep band/filter params applied).
                # NB (preserved verbatim): resolve_target returns ra_deg/dec_deg,
                # so the .get("ra") gate below never passes — the fallback is
                # DEAD, exactly as in the legacy inline method (documented in
                # docs/v2 OPEN_ISSUES; enabling it is a behaviour change).
                _log(f"[FALLBACK] Name search empty for '{target_name}', trying positional fallback...")
                try:
                    resolved = ctx.service("resolve_target")(target_name)
                    if resolved.get("success") and resolved.get("ra") is not None:
                        _fb_ra, _fb_dec = resolved["ra"], resolved["dec"]
                        _log(f"[FALLBACK] Resolved to RA={_fb_ra:.4f}, Dec={_fb_dec:.4f} — cone search 0.14°")
                        results = search_service.cone_search(
                            _fb_ra, _fb_dec, radius=0.14,
                            facility=facility, max_results=max_results
                        )
                        if not results.empty:
                            _log(f"[FALLBACK] Cone search found {len(results)} results — continuing with filters")
                except Exception as _fb_err:
                    _log(f"[FALLBACK] Positional fallback failed: {_fb_err}")

            if results.empty:
                ctx.service("set_last_run_result")({"type": "data", "data": results, "source": facility_label, "tool_name": "search_by_target"})
                # Distinguish an archive failure from a genuine empty result:
                # search.py tags failed/unavailable searches via df.attrs so we
                # don't report "no data" when the archive was simply down. (C3)
                try:
                    _search_err = results.attrs.get("quasar_error")
                except Exception:
                    _search_err = None
                if _search_err:
                    return _native({
                        "success": False,
                        "error": _search_err,
                        "target": target_name,
                        "note": (
                            "The archive query did not complete, so it is unknown whether "
                            "data exist for this target — this is an archive/service error, "
                            "not a confirmed 'no data' result."
                        ),
                    })
                return _native({"success": True, "total_results": 0, "target": target_name, "note": "No results found."})

            # ── Tier 2: Pandas post-filters (non-band) ─────────────────
            filter_parts = []

            # Resolution filter
            res_col = next((c for c in ["spatial_resolution", "s_resolution", "resolution"] if c in results.columns), None)
            if res_col:
                if max_resolution is not None:
                    before = len(results)
                    results = results[pd.to_numeric(results[res_col], errors="coerce") <= max_resolution]
                    filter_parts.append(f"res ≤ {max_resolution}\"")
                    _log(f"[FILTER] max_resolution {max_resolution}: {before} → {len(results)} rows")
                if min_resolution is not None:
                    before = len(results)
                    results = results[pd.to_numeric(results[res_col], errors="coerce") >= min_resolution]
                    filter_parts.append(f"res ≥ {min_resolution}\"")

            # Frequency filter
            freq_col = next((c for c in ["frequency", "min_frequency", "freq_min"] if c in results.columns), None)
            if freq_col:
                if min_freq_ghz is not None:
                    results = results[pd.to_numeric(results[freq_col], errors="coerce") >= min_freq_ghz]
                    filter_parts.append(f"freq ≥ {min_freq_ghz} GHz")
                if max_freq_ghz is not None:
                    results = results[pd.to_numeric(results[freq_col], errors="coerce") <= max_freq_ghz]
                    filter_parts.append(f"freq ≤ {max_freq_ghz} GHz")

            # Integration time filter
            exp_col = next((c for c in ["t_exptime", "integration"] if c in results.columns), None)
            if exp_col and min_exp_s is not None:
                results = results[pd.to_numeric(results[exp_col], errors="coerce") >= min_exp_s]
                filter_parts.append(f"exp ≥ {min_exp_s}s")

            # ── Multi-band handling ────────────────────────────────────
            # When multiple bands are requested (e.g. [6, 7]), produce a
            # separate data card for each band via _accumulated_run_results.
            if facility_label == "ALMA" and scan_intent:
                before = len(results)
                results, scan_filter_label = filter_by_scan_intent(results, scan_intent)
                if scan_filter_label:
                    filter_parts.append(scan_filter_label)
                    _log(f"[FILTER] {scan_filter_label}: {before} → {len(results)} rows")

            band_col = next((c for c in ["band_list", "Band", "band"] if c in results.columns), None)

            if len(band_list_input) > 1 and band_col:
                # Multi-band: combine into a single result with all matching bands
                band_str_list = [str(b) for b in band_list_input]
                combined = results[results[band_col].astype(str).str.split(",").apply(
                    lambda bands: any(x.strip() in band_str_list for x in bands)
                )]
                for b in band_list_input:
                    ct = len(results[results[band_col].astype(str).str.split(",").apply(
                        lambda bands, _b=b: any(str(_b).strip() == x.strip() for x in bands)
                    )])
                    _log(f"[MULTI-BAND] Band {b}: {ct} rows")

                if not combined.empty:
                    results = combined
                    band_label = ", ".join(f"Band {b}" for b in band_list_input)
                    filter_label = f"{facility_label} › {target_name} [{band_label}" + (", ".join([""] + filter_parts) if filter_parts else "") + "]"
                else:
                    # None of the bands matched — show unfiltered
                    filter_label = f"{facility_label} › {target_name}"
                    if filter_parts:
                        filter_label += " [" + ", ".join(filter_parts) + "]"

                ctx.service("set_last_search_results")(results)
                ctx.service("set_last_run_result")({
                    "type": "data", "data": results,
                    "source": facility_label, "filter_label": filter_label,
                    "tool_name": "search_by_target"
                })
            elif len(band_list_input) == 1 and band_col:
                # Single band filter
                b = band_list_input[0]
                before = len(results)
                results = results[results[band_col].astype(str).str.split(",").apply(
                    lambda bands, _b=b: any(str(_b).strip() == x.strip() for x in bands)
                )]
                filter_parts.append(f"Band {b}")
                _log(f"[FILTER] Band {b}: {before} → {len(results)} rows")

                filter_label = f"{facility_label} › {target_name}"
                if filter_parts:
                    filter_label += " [" + ", ".join(filter_parts) + "]"
                ctx.service("set_last_search_results")(results)
                ctx.service("set_last_run_result")({
                    "type": "data", "data": results,
                    "source": facility_label, "filter_label": filter_label,
                    "tool_name": "search_by_target"
                })
            else:
                # No band filter
                filter_label = f"{facility_label} › {target_name}"
                if filter_parts:
                    filter_label += " [" + ", ".join(filter_parts) + "]"
                ctx.service("set_last_search_results")(results)
                ctx.service("set_last_run_result")({
                    "type": "data", "data": results,
                    "source": facility_label, "filter_label": filter_label,
                    "tool_name": "search_by_target"
                })

            # Compact summary + top MOUS UIDs & URLs for Conductor subtask chaining
            top_mous = []
            if 'member_ous_uid' in results.columns:
                top_mous = results['member_ous_uid'].dropna().unique()[:5].tolist()
            top_urls = []
            if 'access_url' in results.columns:
                top_urls = results['access_url'].dropna().head(5).tolist()

            # Extract unique project codes
            top_projects = []
            if hasattr(results, 'columns'):
                project_col = next((col for col in ["proposal_id", "project_code", "Project", "obs_publisher_did"] if col in results.columns), None)
                if project_col:
                    raw_projects = results[project_col].dropna().astype(str).str.strip().unique()
                    import re as _re_proj
                    proj_regex = _re_proj.compile(r"\b\d{4}\.\d\.\d{5}\.[A-Za-z]\b")
                    for p in raw_projects:
                        match = proj_regex.search(p)
                        if match:
                            code = match.group(0).upper()
                            if code not in top_projects:
                                top_projects.append(code)
                        elif len(p) >= 10 and '.' in p:
                            p_upper = p.upper()
                            if p_upper not in top_projects:
                                top_projects.append(p_upper)
                    top_projects = top_projects[:3]

            return _native({
                "success": True,
                "total_results": len(results),
                "filters_applied": filter_parts,
                "target": target_name,
                "top_mous_uids": top_mous,
                "top_access_urls": top_urls,
                "top_project_codes": top_projects,
                "note": f"Found {len(results)} observations matching your constraints. Full data with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SearchByFrequencyInput(_In):
    # Required-but-nullable (guard CX-01).
    min_freq_ghz: Optional[float]
    max_freq_ghz: Optional[float]
    facility: Optional[str] = None
    max_results: Optional[int] = 100


class SearchByFrequency(BaseCapability):
    name = "search_by_frequency"
    description = "Search archives by frequency range. Defaults to ALMA; pass facility='VLA'/'VLBA'/'GBT' for the NRAO archive."
    category = "general"
    InputModel = SearchByFrequencyInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        search_service = ctx.service("search_service")
        min_freq_ghz, max_freq_ghz = inp.min_freq_ghz, inp.max_freq_ghz
        facility, max_results = inp.facility, inp.max_results

        facility_label = (facility or "ALMA").strip().upper()
        if facility_label in ("EVLA", "JVLA"):
            facility_label = "VLA"
        if facility_label not in ("VLA", "VLBA", "GBT"):
            facility_label = "ALMA"

        try:
            results = search_service.search_by_frequency(
                min_freq_ghz, max_freq_ghz, facility, max_results
            )
            ctx.service("set_last_search_results")(results)
            ctx.service("set_last_run_result")({"type": "data", "data": results,
                                                "source": facility_label,
                                                "filter_label": f"{facility_label} › {min_freq_ghz}–{max_freq_ghz} GHz",
                                                "tool_name": "search_by_frequency"})
            return _native({
                "success": True,
                "total_results": len(results),
                "note": f"Found {len(results)} observations at {min_freq_ghz}–{max_freq_ghz} GHz. Full data with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SearchCadcInput(_In):
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius: Optional[float] = 0.02
    collection: Optional[str] = None
    max_results: Optional[int] = 500


class SearchCadc(BaseCapability):
    name = "search_cadc_archive"
    description = (
        "Search the Canadian Astronomy Data Centre (CADC) for multi-wavelength observations "
        "from JWST, HST, JCMT, CFHT, Gemini, and other telescopes. Uses the IVOA ObsCore TAP "
        "service. Complements ALMA searches with optical/infrared/submm data."
    )
    category = "general"
    InputModel = SearchCadcInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        target_name, ra, dec = inp.target_name, inp.ra, inp.dec
        radius, collection, max_results = inp.radius, inp.collection, inp.max_results
        try:
            import pyvo

            # Resolve target name to coordinates if needed
            if target_name and (ra is None or dec is None):
                try:
                    from astroquery.simbad import Simbad
                    result = Simbad.query_object(target_name)
                    if result is not None and len(result) > 0:
                        ra = float(result["RA"][0].replace(" ", ":").split(":")[0]) * 15 + \
                             float(result["RA"][0].replace(" ", ":").split(":")[1]) * 15/60 + \
                             float(result["RA"][0].replace(" ", ":").split(":")[2]) * 15/3600
                        dec_parts = result["DEC"][0].replace(" ", ":").split(":")
                        dec_sign = -1 if dec_parts[0].startswith("-") else 1
                        dec = dec_sign * (abs(float(dec_parts[0])) + float(dec_parts[1])/60 + float(dec_parts[2])/3600)
                    else:
                        return _native({"success": False, "error": f"Could not resolve target '{target_name}' via SIMBAD."})
                except Exception as e:
                    # Fallback: try using our existing resolve_target
                    try:
                        resolved = ctx.service("resolve_target")(target_name)
                        if resolved.get("success") and resolved.get("ra") is not None:
                            ra = resolved["ra"]
                            dec = resolved["dec"]
                        else:
                            return _native({"success": False, "error": f"Could not resolve '{target_name}': {e}"})
                    except Exception:
                        return _native({"success": False, "error": f"Could not resolve '{target_name}': {e}"})

            if ra is None or dec is None:
                return _native({"success": False, "error": "Provide target_name or (ra, dec) coordinates."})

            # Build ADQL query against CADC's IVOA ObsCore
            collection_filter = ""
            if collection:
                collection_filter = f"AND obs_collection = '{collection.upper()}'"

            adql = f"""
            SELECT TOP {max_results} *
            FROM ivoa.ObsCore
            WHERE CONTAINS(POINT('ICRS', s_ra, s_dec),
                           CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius})) = 1
            {collection_filter}
            """

            # Query CADC TAP
            tap_url = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus"
            tap = pyvo.dal.TAPService(tap_url)
            result = tap.search(adql)
            df = result.to_table().to_pandas()

            if df.empty:
                ctx.service("set_last_run_result")({"type": "data", "data": df, "source": "CADC", "tool_name": "search_cadc_archive"})
                note = f"No CADC observations found near "
                note += f"'{target_name}'" if target_name else f"RA={ra:.4f}, Dec={dec:.4f}"
                note += f" (radius={radius}°)"
                if collection:
                    note += f" for {collection}"
                return _native({"success": True, "total_results": 0, "note": note})

            # Build source label
            telescopes = df["obs_collection"].unique().tolist() if "obs_collection" in df.columns else []
            filter_label = "CADC"
            if target_name:
                filter_label += f" › {target_name}"
            if collection:
                filter_label += f" [{collection}]"
            elif telescopes:
                filter_label += f" [{', '.join(telescopes[:3])}]"

            ctx.service("set_last_search_results")(df)
            ctx.service("set_last_run_result")({
                "type": "data", "data": df,
                "source": "CADC", "filter_label": filter_label,
                "tool_name": "search_cadc_archive"
            })

            # Build compact summary for LLM
            # Count by telescope
            tel_summary = df["obs_collection"].value_counts().to_dict() if "obs_collection" in df.columns else {}

            # Include top access_urls so Conductor subtasks can download/render
            top_urls = []
            if "access_url" in df.columns:
                top_urls = df["access_url"].dropna().head(5).tolist()

            return _native({
                "success": True,
                "total_results": len(df),
                "telescopes": tel_summary,
                "top_access_urls": top_urls,
                "note": (
                    f"Found {len(df)} observations from {len(telescopes)} telescope(s): "
                    f"{', '.join(f'{t} ({c})' for t, c in tel_summary.items())}. "
                    f"Full data with sky previews shown in UI table. Do NOT render a table — the UI already displays one."
                )
            })
        except ImportError:
            return _native({"success": False, "error": "pyvo library not installed. Run: pip install pyvo"})
        except Exception as e:
            return _native({"success": False, "error": f"CADC TAP query failed: {str(e)}"})


class GetObservationDetailsInput(_In):
    # Required-but-nullable (guard CX-01): null reaches the service inside the
    # legacy try → the service's own error string, not a pydantic dict.
    obs_id: Optional[str]


class GetObservationDetails(BaseCapability):
    name = "get_observation_details"
    description = "Get detailed information about a specific observation"
    category = "general"
    InputModel = GetObservationDetailsInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            details = ctx.service("search_service").get_observation_details(inp.obs_id)
            return _native({"success": True, "details": details})
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class DownloadDataInput(_In):
    # Required-but-nullable (guard CX-01): the legacy stub echoed obs_id=None.
    obs_id: Optional[str]
    output_dir: Optional[str] = "./data"


class DownloadData(BaseCapability):
    name = "download_data"
    description = "Legacy download tool (Use download_alma_data instead)"
    category = "general"
    InputModel = DownloadDataInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        # Legacy stub preserved verbatim (the original kept it to avoid breaking
        # older tests/tools; download_alma_data is the real one).
        try:
            return _native({
                "success": False,
                "error": "Basic download not supported. Use download_alma_data for ALMA observations.",
                "obs_id": inp.obs_id
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class AnalyzeUvCoverageInput(_In):
    # Required-but-nullable (guard CX-01).
    ms_path: Optional[str]


class AnalyzeUvCoverage(BaseCapability):
    name = "analyze_uv_coverage"
    description = "Analyze UV coverage for an observation"
    category = "general"
    InputModel = AnalyzeUvCoverageInput
    annotations = {"read_only": True, "cost": "compute"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            analysis = ctx.service("analysis_service").analyze_uv_coverage(inp.ms_path)
            # Store analysis result potentially?
            ctx.service("set_last_run_result")({"type": "analysis", "data": analysis})
            # Reflect the service's own success flag so a CASA-unavailable
            # error isn't surfaced to the model as a successful analysis. (C1)
            if isinstance(analysis, dict) and analysis.get("success") is False:
                return _native({
                    "success": False,
                    "error": analysis.get("message") or analysis.get("error") or "UV analysis unavailable",
                    "analysis": analysis,
                })
            return _native({"success": True, "analysis": analysis})
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SearchAlmaWithKeywordsInput(_In):
    keywords: Any


class SearchAlmaWithKeywords(BaseCapability):
    name = "search_alma_with_keywords"
    description = "Search ALMA archives using specific keywords (pi_name, project_code, etc.)"
    category = "general"
    InputModel = SearchAlmaWithKeywordsInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        keywords = inp.keywords
        try:
            # Handle string input if LLM passed JSON string
            if isinstance(keywords, str):
                keywords = json.loads(keywords)

            results = ctx.service("search_service").search_alma_with_keywords(keywords)
            ctx.service("set_last_search_results")(results)
            ctx.service("set_last_run_result")({"type": "data", "data": results, "source": f"Keywords: {keywords}"})

            return _native({
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class AdvancedSearchInput(_In):
    # Required-but-nullable (guard CX-01): the legacy body null-coerced via
    # str(query or "") and returned its SPECIFIC obscore error + hint for a
    # null query — a pydantic rejection would change output keys/control flow.
    query: Optional[str]


class AdvancedSearch(BaseCapability):
    name = "advanced_search"
    description = (
        "Execute a custom ADQL/TAP query directly on the ALMA Science Archive (ivoa.obscore table).\n"
        "ALMA ONLY — NOT for NOIRLab Data Lab catalogs (gaia_dr3/des_dr1/desi_dr1/nsc_dr2/smash/...): "
        "use datalab_sql_query for those.\n"
        "IMPORTANT: The obscore table has NO 'redshift' column. Use frequency/bandwidth containment instead.\n"
        "To find observations covering a specific frequency nu_ghz:\n"
        "  WHERE (frequency - 0.5*bandwidth/1e9) < {nu_ghz} AND (frequency + 0.5*bandwidth/1e9) > {nu_ghz}\n"
        "Key columns: target_name, s_ra, s_dec, frequency (GHz), bandwidth (Hz), scientific_category,\n"
        "  science_keyword, proposal_id, member_ous_uid, t_exptime, s_resolution, band_list.\n"
        "Add OFFSET 0 ROWS FETCH NEXT 500 ROWS ONLY to limit results."
    )
    category = "general"
    InputModel = AdvancedSearchInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        query = inp.query
        try:
            # This tool queries the ALMA Science Archive TAP service ONLY. Queries
            # against Data Lab schemas used to be forwarded and came back as a
            # silent success with 0 rows, which the model then reported as "no
            # data exists" (2026-07-04 live test P11: a DESI LRG selection).
            query_lower = str(query or "").lower()
            try:
                from services import datalab_registry as _dl_reg
                _dl_schemas = sorted(_dl_reg.DATALAB_CATALOGS.keys())
            except Exception:
                _dl_schemas = []
            _foreign = [s for s in _dl_schemas if re.search(rf"\b{re.escape(s)}\s*\.", query_lower)]
            if _foreign:
                return _native({
                    "success": False,
                    "error": (
                        f"advanced_search only queries the ALMA Science Archive (ivoa.obscore). "
                        f"The query references NOIRLab Data Lab schema(s): {', '.join(_foreign)}."
                    ),
                    "hint": "Run this SQL with datalab_sql_query (or a datalab_* builder tool) instead.",
                })
            if "obscore" not in query_lower:
                return _native({
                    "success": False,
                    "error": "advanced_search queries the ALMA ivoa.obscore table; the query does not reference it.",
                    "hint": (
                        "Use FROM ivoa.obscore for ALMA archive searches. For survey-catalog SQL "
                        "(Gaia/DES/DESI/NSC/SMASH/...), use datalab_sql_query instead."
                    ),
                })
            results = ctx.service("search_service").advanced_search(query)
            ctx.service("set_last_search_results")(results)
            ctx.service("set_last_run_result")({"type": "data", "data": results, "source": f"SQL: {query}"})

            return _native({
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SearchAlmaCoInRedshiftRangeInput(_In):
    # Required-but-nullable (guard CX-01): legacy nulls hit the ADQL arithmetic
    # OUTSIDE the try and raised (the tool loop formats it) — nullable typing
    # reproduces that exact raise instead of a pydantic dict.
    z_min: Optional[float]
    z_max: Optional[float]
    science_category: Optional[str] = ""
    max_results: Optional[int] = 500


class SearchAlmaCoInRedshiftRange(BaseCapability):
    name = "search_alma_co_in_redshift_range"
    description = (
        "Search the ALMA archive for observations that cover CO emission lines "
        "for galaxies at a given redshift range. Handles the CO rest-frequency → "
        "observed-frequency conversion and TAP frequency-containment query automatically. "
        "Use this for any query like 'galaxies at z=1-2 with CO coverage' or "
        "'ALMA CO detections at high redshift'.\n"
        "CO transitions checked: J=1-0 (115.3 GHz), J=2-1 (230.5), J=3-2 (345.8), "
        "J=4-3 (461.0), J=5-4 (576.3), J=6-5 (691.5), J=7-6 (806.7).\n"
        "Returns: target_name, proposal_id, CO_transition, obs_frequency_ghz, bandwidth_ghz."
    )
    category = "general"
    InputModel = SearchAlmaCoInRedshiftRangeInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        z_min, z_max = inp.z_min, inp.z_max
        science_category, max_results = inp.science_category, inp.max_results

        # CO rotational transitions: (J_upper, rest_freq_GHz)
        CO_TRANSITIONS = [
            ("CO(1-0)",  115.2712018),
            ("CO(2-1)",  230.5380000),
            ("CO(3-2)",  345.7959899),
            ("CO(4-3)",  461.0407682),
            ("CO(5-4)",  576.2679305),
            ("CO(6-5)",  691.4730763),
            ("CO(7-6)",  806.6518060),
        ]

        # Build ADQL frequency-range OR conditions for all transitions
        # Each transition covers a *range* of frequencies depending on the z range.
        # ν_obs_min (at z_max) to ν_obs_max (at z_min)
        freq_conditions = []
        transition_map = {}  # (freq_min, freq_max) -> transition label

        for label, nu_rest in CO_TRANSITIONS:
            nu_at_z_max = nu_rest / (1.0 + z_max)   # lower observed freq (higher z)
            nu_at_z_min = nu_rest / (1.0 + z_min)   # higher observed freq (lower z)

            # We want any observation whose spectral window overlaps [nu_at_z_max, nu_at_z_min]
            # (frequency - 0.5*bw/1e9) < nu_at_z_min  AND  (frequency + 0.5*bw/1e9) > nu_at_z_max
            cond = (
                f"((frequency - 0.5*bandwidth/1e9) < {nu_at_z_min:.4f} "
                f"AND (frequency + 0.5*bandwidth/1e9) > {nu_at_z_max:.4f})"
            )
            freq_conditions.append(cond)
            transition_map[(round(nu_at_z_max, 4), round(nu_at_z_min, 4))] = label

        freq_where = " OR ".join(freq_conditions)

        # Optional science category filter
        cat_clause = ""
        if science_category:
            cat_clause = f" AND scientific_category LIKE '%{science_category}%'"
        else:
            # Default: restrict to extragalactic categories
            cat_clause = (
                " AND (scientific_category LIKE '%Galaxy%' "
                "OR scientific_category LIKE '%Cosmology%' "
                "OR scientific_category LIKE '%Active%')"
            )

        adql_query = f"""
SELECT TOP {max_results}
       target_name, proposal_id, member_ous_uid,
       frequency, bandwidth, scientific_category, science_keyword,
       s_ra, s_dec, t_exptime, s_resolution
FROM ivoa.obscore
WHERE ({freq_where})
{cat_clause}
ORDER BY target_name
"""

        logger.info("CO redshift search ADQL:\n%s", adql_query.strip())

        try:
            import pyvo
            tap_url = "https://almascience.eso.org/tap"
            service = pyvo.dal.TAPService(tap_url)
            res = service.search(adql_query)
            df = res.to_table().to_pandas()

            if df.empty:
                return _native({
                    "success": True,
                    "count": 0,
                    "message": (
                        f"No ALMA observations found covering CO lines at z={z_min}–{z_max}. "
                        "The archive may not have public data for this parameter space, or "
                        "the frequency range falls outside ALMA's standard bands."
                    ),
                    "z_range": [z_min, z_max],
                    "co_obs_freq_ranges_ghz": {
                        label: {
                            "nu_min_ghz": round(nu_rest / (1 + z_max), 2),
                            "nu_max_ghz": round(nu_rest / (1 + z_min), 2),
                        }
                        for label, nu_rest in CO_TRANSITIONS
                    },
                })

            # Annotate which CO transition each observation covers
            def _which_co(row):
                obs_nu = float(row.get("frequency", 0))
                bw_ghz = float(row.get("bandwidth", 0)) / 1e9
                covered = []
                for label, nu_rest in CO_TRANSITIONS:
                    nu_lo = nu_rest / (1 + z_max)
                    nu_hi = nu_rest / (1 + z_min)
                    obs_lo = obs_nu - 0.5 * bw_ghz
                    obs_hi = obs_nu + 0.5 * bw_ghz
                    if obs_lo < nu_hi and obs_hi > nu_lo:
                        covered.append(label)
                return ", ".join(covered) if covered else "unknown"

            df["CO_transitions_covered"] = df.apply(_which_co, axis=1)
            df["obs_freq_ghz"] = df["frequency"].round(3)
            df["bandwidth_ghz"] = (df["bandwidth"] / 1e9).round(3)

            # Store in agent cache for follow-up plotting
            ctx.service("set_last_search_results")(df)
            ctx.service("set_last_run_result")({"type": "data", "data": df, "source": f"CO z={z_min}-{z_max}"})

            # Build summary
            summary_cols = ["target_name", "proposal_id", "CO_transitions_covered",
                            "obs_freq_ghz", "bandwidth_ghz", "scientific_category"]
            available_cols = [c for c in summary_cols if c in df.columns]
            summary = df[available_cols].drop_duplicates().to_dict("records")

            return _native({
                "success": True,
                "count": len(df),
                "unique_targets": int(df["target_name"].nunique()) if "target_name" in df.columns else None,
                "z_range": [z_min, z_max],
                "co_transitions_searched": [t[0] for t in CO_TRANSITIONS],
                "results": summary[:100],  # cap at 100 for LLM context
                "note": (
                    "Observations found where CO line at given z falls inside the ALMA spectral window. "
                    "Use plot_alma_results() to visualize sky distribution."
                )
            })

        except ImportError:
            return _native({
                "success": False,
                "error": "pyvo is not installed. Run: pip install pyvo",
            })
        except Exception as e:
            import traceback
            logger.error("CO redshift search failed: %s\n%s", e, traceback.format_exc())
            return _native({"success": False, "error": str(e)})


class QueryAlmaScienceArchiveInput(_In):
    # Required-but-nullable (guard CX-01): legacy str(query_type or "").strip()
    # turned a null into "" → the typed "Unknown query_type: " error.
    query_type: Optional[str]
    cycle: Optional[int] = None
    target: Optional[str] = ""
    # Any, NOT Optional[int] (guard CX-02): the body's truthiness gates
    # (`band or 6`, `Band {band or 'any'}`) ran on the RAW value in legacy —
    # int-coercion turned "0" into 0 (falsy) and silently queried Band 6, and
    # rejected exotic-but-legacy-tolerated shapes.
    band: Any = None
    max_resolution_arcsec: Optional[float] = None
    arrays: Optional[List[str]] = None
    lines: Optional[List[str]] = None
    topic_filter: Optional[str] = ""
    redshift_min: Optional[float] = None
    redshift_max: Optional[float] = None
    rest_species: Optional[str] = "CO"
    science_category: Optional[str] = ""
    # `is False` identity gate + truthiness gate in the body — Any, no coercion.
    require_same_project: Any = True
    include_adql: Any = True
    max_results: Optional[int] = 5000


class QueryAlmaScienceArchive(BaseCapability):
    name = "query_alma_science_archive"
    description = (
        "Run deterministic ALMA Science Archive query templates for hard archive-science questions. "
        "Use this instead of raw ADQL for: Cycle N project counts, Sun/solar projects, projects using "
        "12m+7m+total-power arrays, high-resolution Band N continuum candidates for a target, projects "
        "covering a required molecular line set such as 12CO/13CO/C18O in the same project, and "
        "bandwidth-switching calibration diagnostics."
    )
    category = "archive"
    InputModel = QueryAlmaScienceArchiveInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        """Run deterministic ALMA archive query templates for science questions."""
        # search_service is fetched INSIDE the branch that needs it (guard
        # CX-04): the unknown-query_type / cycle-required early returns must
        # fire without touching the service, exactly like the legacy method.
        prov_state = ctx.service("alma_tap_provenance")
        cycle, target, band = inp.cycle, inp.target, inp.band
        max_resolution_arcsec = inp.max_resolution_arcsec
        arrays, lines, topic_filter = inp.arrays, inp.lines, inp.topic_filter
        redshift_min, redshift_max = inp.redshift_min, inp.redshift_max
        rest_species, science_category = inp.rest_species, inp.science_category
        require_same_project, include_adql = inp.require_same_project, inp.include_adql

        started = time.perf_counter()
        query_type = str(inp.query_type or "").strip()
        max_results = max(1, min(int(inp.max_results or 5000), 20000))
        prov_state["query"] = None
        prov_state["url"] = None
        warnings: List[str] = []
        query_summary = ""
        try:
            if query_type == "cycle_solar_projects":
                if cycle is None:
                    return _native({"success": False, "error": "cycle is required"})
                where = (
                    f"{project_prefix_where(int(cycle))} AND ("
                    "LOWER(target_name) LIKE '%sun%' "
                    "OR LOWER(science_keyword) LIKE '%sun%' "
                    "OR LOWER(scientific_category) LIKE '%sun%' "
                    "OR LOWER(obs_title) LIKE '%sun%'"
                    ")"
                )
                df = _tap_obscore_dataframe(where, max_results=max_results, ctx=ctx)
                result_df = summarize_projects(df)
                source = f"ALMA Cycle {cycle} solar projects"
                mode = "cycle_solar_projects"
                query_summary = f"Cycle {cycle} projects with solar/Sun terms in target, keyword, category, or title."

            elif query_type == "cycle_array_combo_projects":
                if cycle is None:
                    return _native({"success": False, "error": "cycle is required"})
                required_arrays = arrays or ["12m", "7m", "TP"]
                df = _tap_obscore_dataframe(project_prefix_where(int(cycle)), max_results=max_results, ctx=ctx)
                result_df = projects_with_array_combo(df, required_arrays)
                source = f"ALMA Cycle {cycle} array combo projects"
                mode = "cycle_array_combo_projects"
                query_summary = f"Cycle {cycle} projects grouped by proposal_id requiring arrays {', '.join(required_arrays)}."

            elif query_type == "high_resolution_band_data":
                if not target:
                    return _native({"success": False, "error": "target is required"})
                normalized_target = normalize_target_alias(target)
                if max_resolution_arcsec is None:
                    max_resolution_arcsec = 0.1
                    warnings.append("Defaulted high-resolution threshold to <0.1 arcsec.")
                df = ctx.service("search_service").search_by_target(normalized_target, facility="ALMA", max_results=max_results)
                df = science_filter_band(df, band)
                df = science_filter_resolution(df, max_resolution_arcsec)
                if "dataproduct_type" in df.columns:
                    image_mask = df["dataproduct_type"].astype(str).str.contains("image|cube", case=False, regex=True, na=False)
                    df = df[image_mask].copy()
                result_df = summarize_projects(df)
                source = f"ALMA {normalized_target} Band {band or 'any'} high-resolution candidates"
                mode = "high_resolution_band_data"
                query_summary = (
                    f"Target search for {normalized_target}, Band {band or 'any'}, "
                    f"resolution < {max_resolution_arcsec} arcsec, image/cube products when available."
                )

            elif query_type == "line_set_projects":
                required_lines = lines or ["12CO", "13CO", "C18O"]
                requested_band = band or 6
                topic = str(topic_filter or "").strip()
                where_parts = [f"(band_list LIKE '%{requested_band}%')"]
                if topic:
                    safe_topic = topic.replace("'", "''").lower()
                    where_parts.append(
                        "("
                        f"LOWER(science_keyword) LIKE '%{safe_topic}%' "
                        f"OR LOWER(scientific_category) LIKE '%{safe_topic}%' "
                        f"OR LOWER(obs_title) LIKE '%{safe_topic}%'"
                        ")"
                    )
                where = " AND ".join(where_parts)
                df = _tap_obscore_dataframe(where, max_results=max_results, ctx=ctx)
                result_df = projects_covering_all_lines(df, required_lines, z=0.0)
                source = f"ALMA Band {requested_band} projects covering {', '.join(required_lines)}"
                mode = "line_set_projects"
                query_summary = (
                    f"Band {requested_band} rows grouped by proposal_id; retained projects covering all requested "
                    f"rest-frame lines: {', '.join(required_lines)}."
                )

            elif query_type == "redshifted_line_projects":
                z_min = 1.0 if redshift_min is None else float(redshift_min)
                z_max = 2.0 if redshift_max is None else float(redshift_max)
                where, line_names = _redshifted_line_where(rest_species or "CO", z_min, z_max, science_category)
                df = _tap_obscore_dataframe(where, max_results=max_results, ctx=ctx)
                result_df = redshifted_line_projects(df, rest_species=rest_species or "CO", z_min=z_min, z_max=z_max)
                source = f"ALMA {rest_species or 'CO'} redshifted line projects z={z_min:g}-{z_max:g}"
                mode = "redshifted_line_projects"
                if require_same_project is False:
                    warnings.append("require_same_project=False is accepted for API compatibility; this summary is still grouped by proposal_id.")
                query_summary = (
                    f"Frequency-containment query for {', '.join(line_names)} shifted to z={z_min:g}-{z_max:g}, "
                    "restricted to extragalactic science categories unless science_category is supplied."
                )

            elif query_type == "bandwidth_switching_candidates":
                where = project_prefix_where(int(cycle)) if cycle is not None else "proposal_id IS NOT NULL"
                df = _tap_obscore_dataframe(where, max_results=max_results, ctx=ctx)
                result_df = bandwidth_switching_candidates(df)
                source = "ALMA bandwidth-switching calibration candidates" + (f" Cycle {cycle}" if cycle is not None else "")
                mode = "bandwidth_switching_candidates"
                warnings.append("Bandwidth Switching likelihood is inferred from public spectral setup metadata; it is not proof of calibration intent.")
                query_summary = "Projects scored by spectral-window count, bandwidth diversity, tuning diversity, and calibration-like metadata."

            else:
                return _native({"success": False, "error": f"Unknown query_type: {query_type}"})

            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            provenance = {
                "archive": "ALMA Science Archive",
                "tap_url": prov_state.get("url"),
                "adql": prov_state.get("query") if include_adql else None,
                "elapsed_ms": elapsed_ms,
                "fresh_query": True,
            }
            ctx.service("set_last_search_results")(result_df)
            ctx.service("set_last_run_result")({
                "type": "data",
                "data": result_df,
                "source": source,
                "filter_label": source,
                "tool_name": "query_alma_science_archive",
            })
            unique_projects = int(result_df["proposal_id"].nunique()) if "proposal_id" in result_df.columns else len(result_df)
            return _native({
                "success": True,
                "mode": mode,
                "count": len(result_df),
                "unique_projects": unique_projects,
                "source": source,
                "query_summary": query_summary,
                "results": result_df.head(100).to_dict("records") if not result_df.empty else [],
                "warnings": warnings,
                "provenance": provenance,
                "note": "Full result table is shown in the UI data card.",
            })
        except Exception as e:
            import traceback
            logger.error("ALMA science query failed: %s\n%s", e, traceback.format_exc())
            return _native({"success": False, "error": str(e), "query_type": query_type})


def _match_cross_archive_sources_impl(
    ctx: CallContext,
    *,
    catalog_name: str = "perseus_protostars",
    sources: Optional[List[Dict[str, Any]]] = None,
    archives: Optional[List[str]] = None,
    radius_arcsec: float = 5.0,
    max_sources: int = 12,
    max_alma_rows: int = 5000,
    max_mast_results_per_source: int = 80,
    require_all_archives: Any = False,
) -> Dict[str, Any]:
    """Cross-match a built-in or inline source catalog against archives.

    The legacy method body, verbatim — shared by MatchCrossArchiveSources and
    MatchPerseusProtostarsAlmaJwst (which called the method directly).
    search_service is fetched inside the ALMA branch's try (guard CX-04): the
    legacy touched self.search_service only there, so a degraded agent still
    gets normalize_source_catalog's typed error / the MAST-only path."""
    prov_state = ctx.service("alma_tap_provenance")
    requested_archives = {str(a).upper() for a in (archives or ["ALMA", "JWST"])}
    radius_arcsec = max(0.5, min(float(radius_arcsec or 5.0), 60.0))
    requested_mast_missions = sorted(requested_archives & {"JWST", "HST", "TESS", "KEPLER", "K2", "GALEX", "SWIFT"})
    try:
        catalog_label, source_catalog = normalize_source_catalog(
            catalog_name=catalog_name,
            sources=sources,
            max_sources=max_sources,
        )
    except ValueError as e:
        return {"success": False, "error": str(e)}

    archive_errors: List[str] = []
    alma_df = pd.DataFrame()
    alma_matches = pd.DataFrame()
    mast_by_source: Dict[str, pd.DataFrame] = {}
    if "ALMA" in requested_archives:
        try:
            search_service = ctx.service("search_service")
            service = search_service.alminer_client._get_tap_service()
            query = alma_bulk_cone_adql(source_catalog, radius_arcsec=radius_arcsec, top=max_alma_rows)
            prov_state["query"] = query
            prov_state["url"] = "https://almascience.nrao.edu/tap"
            alma_result = service.search(query)
            alma_df = alma_result.to_table().to_pandas()
            if hasattr(search_service.alminer_client, "_standardize_columns"):
                alma_df = search_service.alminer_client._standardize_columns(alma_df)
            alma_matches = attach_nearest_source(alma_df, source_catalog, radius_arcsec=radius_arcsec)
        except Exception as e:
            import traceback
            logger.error("ALMA cross-match query failed: %s\n%s", e, traceback.format_exc())
            archive_errors.append(f"ALMA TAP failed: {e}")

    if {"MAST", "JWST", "HST", "TESS", "KEPLER", "K2", "GALEX", "SWIFT"} & requested_archives:
        mission = requested_mast_missions[0] if len(requested_mast_missions) == 1 else None
        for source in source_catalog:
            try:
                mast_by_source[source["source_name"]] = ctx.service("mast_client").search_by_position(
                    float(source["ra"]),
                    float(source["dec"]),
                    radius_arcmin=radius_arcsec / 60.0,
                    mission=mission,
                    max_results=max_mast_results_per_source,
                )
            except Exception as e:
                logger.error("MAST cross-match query failed for %s: %s", source["source_name"], e)
                archive_errors.append(f"MAST query failed for {source['source_name']}: {e}")
                mast_by_source[source["source_name"]] = pd.DataFrame()

    summary = summarize_cross_archive_matches(
        source_catalog,
        alma_matches,
        mast_by_source,
        sorted(requested_archives),
        require_all_archives=bool(require_all_archives),
    )
    ctx.service("set_last_search_results")(summary)
    ctx.service("set_last_run_result")({
        "type": "data",
        "data": summary,
        "source": f"{' + '.join(sorted(requested_archives))} {catalog_label} Cross-match",
        "filter_label": f"{catalog_label} within {radius_arcsec:g} arcsec",
        "tool_name": "match_cross_archive_sources",
        "warnings": archive_errors,
        "partial": bool(archive_errors),
    })

    return {
        "success": True,
        "mode": "cross_archive_source_match",
        "catalog_name": catalog_label,
        "archives": sorted(requested_archives),
        "sources_tested": len(source_catalog),
        "matched_sources": len(summary),
        "radius_arcsec": radius_arcsec,
        "alma_rows": len(alma_df) if alma_df is not None else 0,
        "archive_errors": archive_errors,
        "results": summary.head(100).to_dict("records") if not summary.empty else [],
        "note": "Full cross-match table is shown in the UI data card with sky coordinates.",
    }


class MatchCrossArchiveSourcesInput(_In):
    catalog_name: Optional[str] = "perseus_protostars"
    sources: Optional[List[Dict[str, Any]]] = None
    archives: Optional[List[str]] = None
    radius_arcsec: Optional[float] = 5.0
    max_sources: Optional[int] = 12
    max_alma_rows: Optional[int] = 5000
    max_mast_results_per_source: Optional[int] = 80
    require_all_archives: Any = False  # bool(...) truthiness in the impl — Any


class MatchCrossArchiveSources(BaseCapability):
    name = "match_cross_archive_sources"
    description = (
        "Cross-match a source catalog against ALMA and MAST/JWST observations. "
        "Use this for source-list location questions such as Perseus protostars observed with ALMA and JWST, "
        "or pass explicit source coordinates for any catalog."
    )
    category = "archive"
    InputModel = MatchCrossArchiveSourcesInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        out = _match_cross_archive_sources_impl(
            ctx,
            catalog_name=inp.catalog_name,
            sources=inp.sources,
            archives=inp.archives,
            radius_arcsec=inp.radius_arcsec,
            max_sources=inp.max_sources,
            max_alma_rows=inp.max_alma_rows,
            max_mast_results_per_source=inp.max_mast_results_per_source,
            require_all_archives=inp.require_all_archives,
        )
        return _native(out)


class MatchPerseusProtostarsAlmaJwstInput(_In):
    radius_arcsec: Optional[float] = 5.0
    max_sources: Optional[int] = 12
    max_alma_rows: Optional[int] = 5000
    max_mast_results_per_source: Optional[int] = 80


class MatchPerseusProtostarsAlmaJwst(BaseCapability):
    name = "match_perseus_protostars_alma_jwst"
    description = (
        "Cross-match a built-in Perseus protostar source list against ALMA and MAST/JWST observations. "
        "Use this for questions like 'Show locations of protostars in Perseus observed with ALMA and JWST'. "
        "Returns sources with both ALMA and JWST matches, counts, project/program IDs, and sky coordinates."
    )
    category = "archive"
    InputModel = MatchPerseusProtostarsAlmaJwstInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        """Backward-compatible wrapper for the generic cross-archive matcher."""
        result = _match_cross_archive_sources_impl(
            ctx,
            catalog_name="perseus_protostars",
            archives=["ALMA", "JWST"],
            radius_arcsec=inp.radius_arcsec,
            max_sources=inp.max_sources,
            max_alma_rows=inp.max_alma_rows,
            max_mast_results_per_source=inp.max_mast_results_per_source,
            require_all_archives=True,
        )
        # Legacy: rewrite the tool_name on the SAME card dict the agent holds.
        last_run_result = ctx.service("get_last_run_result")()
        if result.get("success") and last_run_result:
            last_run_result["tool_name"] = "match_perseus_protostars_alma_jwst"
        if result.get("success"):
            result["mode"] = "perseus_alma_jwst_cross_match"
        return _native(result)


class PlotAlmaResultsInput(_In):
    plot_type: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_by: Optional[str] = None
    title: Optional[str] = None
    dark_mode: Any = False  # `if dark_mode:` truthiness — Any


class PlotAlmaResults(BaseCapability):
    name = "plot_alma_results"
    description = (
        "Generate a plot from the last ALMA search results. Two modes:\n"
        "1. Quick overview: pass plot_type='sky', 'frequency', or 'overview' for pre-built plots.\n"
        "2. Publication-quality scatter: pass x_column and y_column for a custom ApJ/MNRAS-style "
        "scatter plot (300 DPI, colorblind-safe).\n"
        "If plot_type is given, x_column/y_column are ignored. Use after any search."
    )
    category = "general"
    InputModel = PlotAlmaResultsInput
    annotations = {"read_only": True, "cost": "compute", "produces": "image"}

    def run(self, inp, ctx) -> ToolResult:
        """Unified plot handler — supports quick overview and publication scatter modes."""
        plot_type = inp.plot_type
        x_column, y_column, color_by, title, dark_mode = inp.x_column, inp.y_column, inp.color_by, inp.title, inp.dark_mode
        try:
            # (Legacy also guarded hasattr(self, 'last_search_results'); the TLS
            # property always exists, so the None/empty checks are the live ones.)
            last_search_results = ctx.service("get_last_search_results")()
            if last_search_results is None or last_search_results.empty:
                return _native({"success": False, "error": "No results available to plot. Please run a search first."})

            # Mode 1: Quick overview plot (sky, frequency, overview)
            if plot_type:
                image_bytes = ctx.service("search_service").plot_alma_results(last_search_results, plot_type)
                if image_bytes:
                    ctx.service("set_last_run_result")({
                        "type": "image",
                        "image_bytes": image_bytes,
                        "caption": f"ALMA {plot_type.capitalize()} Plot"
                    })
                    return _native({"success": True, "message": f"Generated {plot_type} plot successfully"})
                return _native({"success": False, "error": "Plot generation returned empty"})

            # Mode 2: Publication-quality scatter plot
            data_records = last_search_results.to_dict("records")
            kw = {}
            if x_column: kw["x_column"] = x_column
            if y_column: kw["y_column"] = y_column
            if color_by: kw["color_by"] = color_by
            if title: kw["title"] = title
            if dark_mode: kw["dark_mode"] = dark_mode
            return _native(ctx.service("plotting_service").plot_alma_results(data_records=data_records, **kw))

        except Exception as e:
            return _native({"success": False, "error": str(e)})


class DownloadAlmaDataInput(_In):
    dry_run: Any = False  # forwarded into a service truthiness gate — Any


class DownloadAlmaData(BaseCapability):
    name = "download_alma_data"
    description = "Download ALMA data (FITS) for current results"
    category = "general"
    InputModel = DownloadAlmaDataInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        """Download ALMA data for observations in current context"""
        try:
            last_search_results = ctx.service("get_last_search_results")()
            if last_search_results is None or last_search_results.empty:
                return _native({"success": False, "error": "No results available to download."})

            msg = ctx.service("search_service").download_alma_data(last_search_results, dry_run=inp.dry_run)
            return _native({"success": True, "message": msg})
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class FindAlmaLineCoverageInput(_In):
    # Required-but-nullable (guard CX-01): the legacy error dict echoes the
    # null values back under target_name/species/transition.
    target_name: Optional[str]
    species: Optional[str]
    transition: Optional[str]
    redshift: Optional[float] = None
    tolerance_mhz: Optional[float] = 0.0
    velocity_width_kms: Optional[float] = None


class FindAlmaLineCoverage(BaseCapability):
    name = "find_alma_line_coverage"
    description = (
        "Resolve one named spectral transition with Splatalogue and return only "
        "ALMA projects whose exact spectral windows cover its observed frequency."
    )
    category = "archive"
    InputModel = FindAlmaLineCoverageInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        """Resolve one transition and verify exact ALMA SPW coverage."""
        target_name, species, transition = inp.target_name, inp.species, inp.transition
        try:
            from services.spectral_line_explorer import (
                find_alma_line_coverage as run_alma_line_coverage,
            )

            result = run_alma_line_coverage(
                target_name=target_name,
                species=species,
                transition=transition,
                redshift=inp.redshift,
                tolerance_mhz=inp.tolerance_mhz,
                velocity_width_kms=inp.velocity_width_kms,
            )
            if not result.get("success"):
                return _native(result)
            project_rows = []
            for project in result.get("projects") or []:
                project_rows.append(
                    {
                        "proposal_id": project.get("proposal_id"),
                        "target_name": project.get("target_name"),
                        "covered_line_count": project.get("covered_line_count"),
                        "all_lines_full": project.get("all_lines_full"),
                        "minimum_edge_margin_mhz": project.get(
                            "minimum_edge_margin_mhz"
                        ),
                        "angular_separation_arcsec": project.get(
                            "angular_separation_arcsec"
                        ),
                        "best_angular_resolution_arcsec": project.get(
                            "best_angular_resolution_arcsec"
                        ),
                        "total_exposure_seconds": project.get(
                            "total_exposure_seconds"
                        ),
                        "archive_url": project.get("archive_url"),
                    }
                )
            frame = pd.DataFrame(project_rows)
            ctx.service("set_last_search_results")(frame)
            ctx.service("set_last_run_result")({
                "type": "data",
                "data": frame,
                "source": (
                    f"Exact ALMA coverage: {target_name} "
                    f"{species}({transition})"
                ),
            })
            selected = result.get("selected_line") or {}
            target = result.get("target") or {}
            return _native({
                "success": True,
                "target_name": target_name,
                "species": species,
                "transition": transition,
                "rest_frequency_ghz": selected.get("frequency_ghz"),
                "observed_frequency_ghz": selected.get(
                    "observed_frequency_ghz"
                ),
                "redshift": target.get("redshift"),
                "redshift_source": target.get("redshift_source"),
                "coordinates": {
                    "ra_deg": target.get("ra_deg"),
                    "dec_deg": target.get("dec_deg"),
                    "source": target.get("coordinate_source"),
                },
                "project_count": result.get("project_count", 0),
                "projects": project_rows[:25],
                "backend": result.get("backend"),
                "degraded": result.get("degraded", False),
                "warnings": result.get("warnings") or [],
                "line_explorer_url": result.get("deep_link"),
            })
        except Exception as e:
            return _native({
                "success": False,
                "error": str(e),
                "target_name": target_name,
                "species": species,
                "transition": transition,
            })


class CheckLineCoverageInput(_In):
    # Required-but-nullable (guard CX-01): with no prior results the legacy
    # "No previous search results" error fires FIRST, even for a null freq.
    line_freq_ghz: Optional[float]
    z: Optional[float] = 0.0
    line_name: Optional[str] = "Line"


class CheckLineCoverage(BaseCapability):
    name = "check_line_coverage"
    description = "Check if specific lines are covered in the LAST search results."
    category = "general"
    InputModel = CheckLineCoverageInput
    annotations = {"read_only": True, "cost": "compute"}

    def run(self, inp, ctx) -> ToolResult:
        """Check line coverage on cache"""
        last_search_results = ctx.service("get_last_search_results")()
        if last_search_results is None or last_search_results.empty:
            return _native({"success": False, "error": "No previous search results to check. Run a search first."})

        try:
            results = ctx.service("search_service").check_line_coverage_on_last(
                last_search_results, inp.line_freq_ghz, inp.z, inp.line_name
            )
            # Don't overwrite last_search_results, just return analysis?
            # Or do we overwrite context? Let's overwrite so we can plot THIS result.
            ctx.service("set_last_search_results")(results)
            ctx.service("set_last_run_result")({"type": "data", "data": results, "source": f"Line Check: {inp.line_name} @ {inp.line_freq_ghz}GHz"})

            return _native({
                "success": True,
                "count": len(results),
                "results": results.to_dict("records")
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class CheckCoLinesInput(_In):
    z: Optional[float] = 0.0


class CheckCoLines(BaseCapability):
    name = "check_co_lines"
    description = "Check for CO, 13CO, and C18O lines in the LAST search results."
    category = "general"
    InputModel = CheckCoLinesInput
    annotations = {"read_only": True, "cost": "compute"}

    def run(self, inp, ctx) -> ToolResult:
        """Check CO lines on cache"""
        last_search_results = ctx.service("get_last_search_results")()
        if last_search_results is None or last_search_results.empty:
            return _native({"success": False, "error": "No previous search results to check. Run a search first."})

        try:
            results = ctx.service("search_service").check_co_lines_on_last(last_search_results, inp.z)
            ctx.service("set_last_search_results")(results)
            ctx.service("set_last_run_result")({"type": "data", "data": results, "source": "CO Lines Check"})

            return _native({
                "success": True,
                "count": len(results),
                "results": results.to_dict("records")
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class SearchCatalogInput(_In):
    # Required-but-nullable: the legacy body ran `if objects:` on whatever
    # arrived, so an explicit null pivots to an empty catalog dict verbatim.
    objects: Optional[List[Dict[str, Any]]]


class SearchCatalog(BaseCapability):
    name = "search_catalog"
    description = "Search for a catalog of objects (Name, RA, Dec)"
    category = "general"
    InputModel = SearchCatalogInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        """Search by catalog"""
        objects = inp.objects
        try:
            # restructure for service: dict of lists
            # Input: [{"Name": "A", "RA": 1}, {"Name": "B"}]
            # Output needed: {"Name": ["A", "B"], ...}

            # Simple pivot
            catalog = {}
            if objects:
                keys = objects[0].keys()
                for k in keys:
                    catalog[k] = [o.get(k) for o in objects]

            results = ctx.service("search_service").search_catalog(catalog)
            ctx.service("set_last_search_results")(results)
            ctx.service("set_last_run_result")({"type": "data", "data": results, "source": "Catalog Search"})

            return _native({
                "success": True,
                "count": len(results),
                "results": results.to_dict("records")
            })
        except Exception as e:
            return _native({"success": False, "error": str(e)})


class FilterResultsInput(_In):
    # Required-but-nullable (guard CX-01): the no-results guard runs before any
    # use of these, and a null column hits .lower() inside the legacy try →
    # the "Filter failed: ..." dict.
    column: Optional[str]
    operator: Optional[str]
    # Any: the legacy compared whatever arrived verbatim; typed-float coercion
    # of a stringy value would silently CHANGE the comparison outcome.
    value: Any


class FilterResults(BaseCapability):
    name = "filter_results"
    description = "Apply numeric filters to the LAST ALMA/archive search results table (e.g. 'resolution < 0.05 arcsec', 'sensitivity > 10 mJy'). This does NOT see Data Lab catalog results — for those, put the cut in the query itself (datalab_select_catalog_rows value_cuts, datalab_sql_query WHERE) or use the one-shot diagram tools' point_sources/morphology options."
    category = "general"
    InputModel = FilterResultsInput
    annotations = {"read_only": True, "cost": "cheap"}

    def run(self, inp, ctx) -> ToolResult:
        """Apply deterministic numeric filter to last search results (Fix 2)"""
        column, operator, value = inp.column, inp.operator, inp.value
        last_search_results = ctx.service("get_last_search_results")()
        if last_search_results is None or last_search_results.empty:
            return _native({
                "success": False,
                "error": "No ALMA/archive search results to filter. This tool only sees archive search tables.",
                "hint": (
                    "For Data Lab catalog data, apply the cut inside the query instead: "
                    "datalab_select_catalog_rows(value_cuts=[{'column': ..., 'op': ..., 'value': ...}]), "
                    "a WHERE clause in datalab_sql_query, or point_sources=true on the one-shot diagram tools."
                ),
            })

        try:
            df = last_search_results.copy()

            # Find the actual column name (case-insensitive match)
            actual_column = None
            for col in df.columns:
                if col.lower() == column.lower():
                    actual_column = col
                    break

            if actual_column is None:
                # Try common aliases
                aliases = {
                    'resolution': ['resolution', 's_resolution', 'angular_resolution'],
                    'sensitivity': ['sensitivity', 'sensitivity_10kms', 'cont_sens_bandwidth'],
                    'frequency': ['freq_min', 'freq_max', 'frequency', 'min_freq_ghz', 'max_freq_ghz'],
                    'band': ['Band', 'band_number', 'band_list']
                }
                for alias_key, alias_list in aliases.items():
                    if column.lower() == alias_key:
                        for alias in alias_list:
                            if alias in df.columns:
                                actual_column = alias
                                break
                        break

            if actual_column is None:
                return _native({
                    "success": False,
                    "error": f"Column '{column}' not found. Available: {list(df.columns)}"
                })

            # Apply filter using operator
            original_count = len(df)
            if operator == "<":
                df = df[df[actual_column] < value]
            elif operator == ">":
                df = df[df[actual_column] > value]
            elif operator == "<=":
                df = df[df[actual_column] <= value]
            elif operator == ">=":
                df = df[df[actual_column] >= value]
            elif operator == "==":
                df = df[df[actual_column] == value]
            elif operator == "!=":
                df = df[df[actual_column] != value]
            else:
                return _native({"success": False, "error": f"Unknown operator: {operator}"})

            # Update cached results
            ctx.service("set_last_search_results")(df)
            ctx.service("set_last_run_result")({
                "type": "data",
                "data": df,
                "source": f"Filtered: {actual_column} {operator} {value}"
            })

            return _native({
                "success": True,
                "original_count": original_count,
                "filtered_count": len(df),
                "filter_applied": f"{actual_column} {operator} {value}",
                "results": df.to_dict("records") if len(df) < 100 else f"[{len(df)} rows - too large to display]"
            })
        except Exception as e:
            return _native({"success": False, "error": f"Filter failed: {str(e)}"})


# The migrated ALMA / archive-search family. resolve_target stays agent-side
# (it is shared cross-family and is the injected resolver); list_alma_files /
# triage_alma_data_products belong to the files family (later migration).
CAPABILITIES: List[BaseCapability] = [
    SearchByPosition(),
    SearchByTarget(),
    SearchByFrequency(),
    SearchCadc(),
    GetObservationDetails(),
    DownloadData(),
    AnalyzeUvCoverage(),
    SearchAlmaWithKeywords(),
    AdvancedSearch(),
    SearchAlmaCoInRedshiftRange(),
    QueryAlmaScienceArchive(),
    MatchCrossArchiveSources(),
    MatchPerseusProtostarsAlmaJwst(),
    PlotAlmaResults(),
    DownloadAlmaData(),
    FindAlmaLineCoverage(),
    CheckLineCoverage(),
    CheckCoLines(),
    SearchCatalog(),
    FilterResults(),
]

__all__ = [
    "CAPABILITIES",
    "filter_by_scan_intent",
    "SearchByPosition", "SearchByTarget", "SearchByFrequency", "SearchCadc",
    "GetObservationDetails", "DownloadData", "AnalyzeUvCoverage",
    "SearchAlmaWithKeywords", "AdvancedSearch", "SearchAlmaCoInRedshiftRange",
    "QueryAlmaScienceArchive", "MatchCrossArchiveSources",
    "MatchPerseusProtostarsAlmaJwst", "PlotAlmaResults", "DownloadAlmaData",
    "FindAlmaLineCoverage", "CheckLineCoverage", "CheckCoLines",
    "SearchCatalog", "FilterResults",
]
