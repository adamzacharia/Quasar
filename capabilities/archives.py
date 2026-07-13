"""
capabilities/archives.py — the MAST / ESO / IRSA archive-search family as
transport-pure capabilities (P1 family migration #4).

Logic relocated VERBATIM from ``core/agent.py`` (the ``_search_mast`` /
``_search_mast_by_criteria`` / ``_get_mast_products`` / ``_download_mast_data``
/ ``_search_eso`` / ``_search_irsa`` methods). The only changes are structural
(same rules as capabilities/alma.py):

  * clients arrive via ``CallContext.services`` (``mast_client``,
    ``eso_client``, ``irsa_client``) — fetched with ``ctx.services.get(...)``
    INSIDE the legacy ``try`` at the position the body first used them, so a
    missing client becomes the same caught-and-typed error dict the inline
    method produced (and the service-free early returns of
    ``get_mast_products`` still need no client — the CX-04 locality rule);
  * the agent's thread-local ``last_search_results`` / ``last_run_result``
    are reached ONLY through the injected bound accessors;
  * every path returns a :class:`ToolResult` whose ``native`` payload is the
    exact legacy output dict (byte-parity).

None of these methods carried ``@log_tool``, so no log_name re-wrap is
needed at the adapter boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, ToolResult

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


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MAST
# ─────────────────────────────────────────────────────────────────────────────
class SearchMastInput(_In):
    target_name: Optional[str] = None
    mission: Optional[str] = None
    instrument: Optional[str] = None
    radius: Optional[str] = "30s"
    ra: Optional[float] = None
    dec: Optional[float] = None


class SearchMast(BaseCapability):
    name = "search_mast"
    description = (
        "Search the MAST archive for observations from JWST, HST, TESS, Kepler, "
        "and other space telescopes. Use this for any JWST or HST data queries. "
        "Returns observation metadata including target, instrument, filters, "
        "exposure time, and data product type."
    )
    category = "general"
    InputModel = SearchMastInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lsr = ctx.service("set_last_search_results")
        set_lrr = ctx.service("set_last_run_result")
        target_name, mission, instrument = inp.target_name, inp.mission, inp.instrument
        radius, ra, dec = inp.radius, inp.ra, inp.dec
        try:
            mast_client = ctx.services.get("mast_client")
            if target_name:
                df = mast_client.search_by_target(
                    target=target_name, mission=mission,
                    instrument=instrument, radius=radius
                )
            elif ra is not None and dec is not None:
                df = mast_client.search_by_position(
                    ra=ra, dec=dec,
                    radius_arcmin=mast_client._parse_radius(radius).to("arcmin").value,
                    mission=mission, instrument=instrument
                )
            else:
                return _native({"success": False, "error": "Provide target_name or (ra, dec) coordinates."})

            if df.empty:
                note = f"No MAST observations found"
                if target_name:
                    note += f" for '{target_name}'"
                if mission:
                    note += f" [{mission}]"
                if instrument:
                    note += f" [{instrument}]"
                set_lrr({"type": "data", "data": df, "source": "MAST", "tool_name": "search_mast"})
                return _native({"success": True, "total_results": 0, "note": note})

            # Build source label
            filter_label = "MAST"
            if target_name:
                filter_label += f" › {target_name}"
            if mission:
                filter_label += f" [{mission}]"
            if instrument:
                filter_label += f" [{instrument}]"

            set_lsr(df)
            set_lrr({
                "type": "data", "data": df,
                "source": "MAST", "filter_label": filter_label,
                "tool_name": "search_mast"
            })

            # Build summary for LLM
            mission_summary = df["telescope"].value_counts().to_dict() if "telescope" in df.columns else {}
            instr_summary = df["instrument_name"].value_counts().to_dict() if "instrument_name" in df.columns else {}

            return _native({
                "success": True,
                "total_results": len(df),
                "missions": mission_summary,
                "instruments": instr_summary,
                "unique_targets": int(df["target_name"].nunique()) if "target_name" in df.columns else 0,
                "unique_programs": int(df["project_code"].nunique()) if "project_code" in df.columns else 0,
                "note": (
                    f"Found {len(df)} MAST observations. "
                    f"Missions: {', '.join(f'{m} ({c})' for m, c in mission_summary.items())}. "
                    f"Full data shown in UI table. Do NOT render a table — the UI already displays one."
                )
            })
        except Exception as e:
            return _native({"success": False, "error": f"MAST search failed: {str(e)}"})


class SearchMastByCriteriaInput(_In):
    mission: Optional[str] = None
    instrument: Optional[str] = None
    proposal_id: Optional[str] = None
    filters: Optional[str] = None
    target_name: Optional[str] = None
    dataproduct_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SearchMastByCriteria(BaseCapability):
    name = "search_mast_by_criteria"
    description = (
        "Advanced MAST search with rich filtering: program ID, date range, "
        "filter name, data product type, etc. Use this when users ask for "
        "specific JWST/HST programs, particular filters (F200W, F444W), "
        "or time-constrained searches."
    )
    category = "general"
    InputModel = SearchMastByCriteriaInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lsr = ctx.service("set_last_search_results")
        set_lrr = ctx.service("set_last_run_result")
        mission, instrument, proposal_id = inp.mission, inp.instrument, inp.proposal_id
        filters, target_name = inp.filters, inp.target_name
        dataproduct_type, start_date, end_date = inp.dataproduct_type, inp.start_date, inp.end_date
        try:
            date_range = None
            if start_date and end_date:
                date_range = (start_date, end_date)

            mast_client = ctx.services.get("mast_client")
            df = mast_client.search_by_criteria(
                mission=mission, instrument=instrument,
                proposal_id=proposal_id, filters=filters,
                target_name=target_name,
                dataproduct_type=dataproduct_type,
                date_range=date_range
            )

            if df.empty:
                criteria_parts = []
                if mission: criteria_parts.append(f"mission={mission}")
                if instrument: criteria_parts.append(f"instrument={instrument}")
                if proposal_id: criteria_parts.append(f"program={proposal_id}")
                if filters: criteria_parts.append(f"filter={filters}")
                note = f"No MAST observations found for criteria: {', '.join(criteria_parts) or 'unspecified'}"
                set_lrr({"type": "data", "data": df, "source": "MAST", "tool_name": "search_mast_by_criteria"})
                return _native({"success": True, "total_results": 0, "note": note})

            # Build label
            filter_label = "MAST Criteria"
            if mission: filter_label += f" [{mission}]"
            if proposal_id: filter_label += f" Program {proposal_id}"
            if target_name: filter_label += f" › {target_name}"

            set_lsr(df)
            set_lrr({
                "type": "data", "data": df,
                "source": "MAST", "filter_label": filter_label,
                "tool_name": "search_mast_by_criteria"
            })

            mission_summary = df["telescope"].value_counts().to_dict() if "telescope" in df.columns else {}
            instr_summary = df["instrument_name"].value_counts().to_dict() if "instrument_name" in df.columns else {}
            filter_summary = df["filters"].value_counts().head(10).to_dict() if "filters" in df.columns else {}

            return _native({
                "success": True,
                "total_results": len(df),
                "missions": mission_summary,
                "instruments": instr_summary,
                "filters_used": filter_summary,
                "unique_targets": int(df["target_name"].nunique()) if "target_name" in df.columns else 0,
                "note": (
                    f"Found {len(df)} observations. "
                    f"Instruments: {', '.join(f'{i} ({c})' for i, c in instr_summary.items())}. "
                    f"Full data shown in UI table."
                )
            })
        except Exception as e:
            return _native({"success": False, "error": f"MAST criteria search failed: {str(e)}"})


class GetMastProductsInput(_In):
    product_type: Optional[str] = None
    extension: Optional[str] = None


class GetMastProducts(BaseCapability):
    name = "get_mast_products"
    description = (
        "Get file-level product list for the LAST MAST search results. "
        "Shows individual data files available for download (filenames, sizes, URLs). "
        "Call this AFTER a search_mast or search_mast_by_criteria call."
    )
    category = "general"
    InputModel = GetMastProductsInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_lsr = ctx.service("get_last_search_results")
        set_lsr = ctx.service("set_last_search_results")
        set_lrr = ctx.service("set_last_run_result")
        product_type, extension = inp.product_type, inp.extension
        try:
            last_search_results = get_lsr()
            if last_search_results is None or last_search_results.empty:
                return _native({
                    "success": False,
                    "error": "No MAST search results to get products for. Run search_mast or search_mast_by_criteria first.",
                })
            if "obsid" not in last_search_results.columns:
                return _native({
                    "success": False,
                    "error": "Last cached results are not MAST observation results. Run search_mast or search_mast_by_criteria first.",
                })

            # Fetched branch-lazily AFTER the service-free early returns (CX-04).
            mast_client = ctx.services.get("mast_client")
            df = mast_client.get_product_list(
                last_search_results,
                productType=product_type,
                extension=extension
            )

            if df.empty:
                return _native({"success": True, "total_products": 0, "note": "No data products found."})

            set_lsr(df)
            set_lrr({
                "type": "data", "data": df,
                "source": "MAST Products",
                "filter_label": "MAST Products",
                "tool_name": "get_mast_products"
            })

            type_summary = df["productType"].value_counts().to_dict() if "productType" in df.columns else {}

            return _native({
                "success": True,
                "total_products": len(df),
                "product_types": type_summary,
                "note": (
                    f"Found {len(df)} data products. "
                    f"Types: {', '.join(f'{t} ({c})' for t, c in type_summary.items())}."
                )
            })
        except Exception as e:
            return _native({"success": False, "error": f"MAST product listing failed: {str(e)}"})


class DownloadMastDataInput(_In):
    product_type: Optional[str] = "SCIENCE"
    extension: Optional[str] = "fits"
    max_files: Optional[int] = 10


class DownloadMastData(BaseCapability):
    name = "download_mast_data"
    description = (
        "Download FITS files from the MAST archive (JWST/HST data). "
        "Call this AFTER search_mast to download actual science data files. "
        "Downloads to ~/quasar_data/mast/ by default. Has a safety limit of 10 files."
    )
    category = "general"
    InputModel = DownloadMastDataInput
    annotations = {"read_only": False, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        get_lsr = ctx.service("get_last_search_results")
        product_type, extension, max_files = inp.product_type, inp.extension, inp.max_files
        try:
            last_search_results = get_lsr()
            if last_search_results is None or last_search_results.empty:
                return _native({"success": False, "error": "No search results. Run search_mast first."})

            # Fetched branch-lazily AFTER the service-free early return (CX-04).
            mast_client = ctx.services.get("mast_client")
            result = mast_client.download_products(
                observations=last_search_results,
                productType=product_type,
                extension=extension,
                max_files=max_files
            )

            return _native(result)
        except Exception as e:
            return _native({"success": False, "error": f"MAST download failed: {str(e)}"})


# ─────────────────────────────────────────────────────────────────────────────
# ESO
# ─────────────────────────────────────────────────────────────────────────────
class SearchEsoInput(_In):
    target_name: Optional[str] = None
    instrument: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcmin: Optional[float] = 1.0


class SearchEso(BaseCapability):
    name = "search_eso_archive"
    description = (
        "Search the ESO Science Archive for VLT instrument observations. "
        "Supports instruments: MUSE, KMOS, X-Shooter, FORS2, HAWK-I, UVES, "
        "SPHERE, GRAVITY, ESPRESSO, and more. Uses TAP/ADQL queries against "
        "the ESO ObsCore table."
    )
    category = "general"
    InputModel = SearchEsoInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lsr = ctx.service("set_last_search_results")
        set_lrr = ctx.service("set_last_run_result")
        target_name, instrument = inp.target_name, inp.instrument
        ra, dec, radius_arcmin = inp.ra, inp.dec, inp.radius_arcmin
        try:
            eso_client = ctx.services.get("eso_client")
            if target_name:
                df = eso_client.search_by_target(
                    target=target_name, instrument=instrument,
                    radius_arcmin=radius_arcmin
                )
            elif ra is not None and dec is not None:
                df = eso_client.search_by_position(
                    ra=ra, dec=dec, radius_arcmin=radius_arcmin,
                    instrument=instrument
                )
            else:
                return _native({"success": False, "error": "Provide target_name or (ra, dec) coordinates."})

            if df.empty:
                note = f"No ESO observations found"
                if target_name:
                    note += f" for '{target_name}'"
                if instrument:
                    note += f" [{instrument}]"
                set_lrr({"type": "data", "data": df, "source": "ESO", "tool_name": "search_eso_archive"})
                return _native({"success": True, "total_results": 0, "note": note})

            # Build source label
            filter_label = "ESO"
            if target_name:
                filter_label += f" › {target_name}"
            if instrument:
                filter_label += f" [{instrument}]"

            set_lsr(df)
            set_lrr({
                "type": "data", "data": df,
                "source": "ESO", "filter_label": filter_label,
                "tool_name": "search_eso_archive"
            })

            instr_summary = df["instrument_name"].value_counts().to_dict() if "instrument_name" in df.columns else {}
            dptype_summary = df["dataproduct_type"].value_counts().to_dict() if "dataproduct_type" in df.columns else {}

            return _native({
                "success": True,
                "total_results": len(df),
                "instruments": instr_summary,
                "data_types": dptype_summary,
                "note": (
                    f"Found {len(df)} ESO observations. "
                    f"Instruments: {', '.join(f'{i} ({c})' for i, c in instr_summary.items())}. "
                    f"Full data shown in UI table. Do NOT render a table — the UI already displays one."
                )
            })
        except Exception as e:
            return _native({"success": False, "error": f"ESO archive search failed: {str(e)}"})


# ─────────────────────────────────────────────────────────────────────────────
# IRSA
# ─────────────────────────────────────────────────────────────────────────────
class SearchIrsaInput(_In):
    target_name: Optional[str] = None
    catalog: Optional[str] = None
    radius_arcsec: Optional[float] = 30.0
    ra: Optional[float] = None
    dec: Optional[float] = None


class SearchIrsa(BaseCapability):
    name = "search_irsa"
    description = (
        "Search the IRSA (Infrared Science Archive) catalog services for "
        "infrared source catalogs. Catalogs include AllWISE, 2MASS Point "
        "Source, 2MASS Extended Source, and Spitzer SEIP."
    )
    category = "general"
    InputModel = SearchIrsaInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        set_lsr = ctx.service("set_last_search_results")
        set_lrr = ctx.service("set_last_run_result")
        target_name, catalog = inp.target_name, inp.catalog
        radius_arcsec, ra, dec = inp.radius_arcsec, inp.ra, inp.dec
        try:
            catalog = catalog or "allwise"

            irsa_client = ctx.services.get("irsa_client")
            if target_name:
                df = irsa_client.search_by_target(
                    target=target_name, catalog=catalog,
                    radius_arcsec=radius_arcsec
                )
            elif ra is not None and dec is not None:
                df = irsa_client.search_by_position(
                    ra=ra, dec=dec,
                    radius_arcsec=radius_arcsec, catalog=catalog
                )
            else:
                return _native({"success": False, "error": "Provide target_name or (ra, dec) coordinates."})

            if df.empty:
                note = f"No IRSA sources found"
                if target_name:
                    note += f" for '{target_name}'"
                note += f" in catalog '{catalog}'"
                set_lrr({"type": "data", "data": df, "source": "IRSA", "tool_name": "search_irsa"})
                return _native({"success": True, "total_results": 0, "note": note})

            filter_label = f"IRSA › {catalog.upper()}"
            if target_name:
                filter_label += f" › {target_name}"

            set_lsr(df)
            set_lrr({
                "type": "data", "data": df,
                "source": "IRSA", "filter_label": filter_label,
                "tool_name": "search_irsa"
            })

            return _native({
                "success": True,
                "total_results": len(df),
                "catalog": catalog,
                "columns": list(df.columns[:15]),  # First 15 columns for LLM context
                "note": (
                    f"Found {len(df)} sources in IRSA {catalog.upper()} catalog. "
                    f"Full data shown in UI table. Do NOT render a table — the UI already displays one."
                )
            })
        except Exception as e:
            return _native({"success": False, "error": f"IRSA search failed: {str(e)}"})


CAPABILITIES: List[BaseCapability] = [
    SearchMast(),
    SearchMastByCriteria(),
    GetMastProducts(),
    DownloadMastData(),
    SearchEso(),
    SearchIrsa(),
]

__all__ = [
    "CAPABILITIES",
    "SearchMast", "SearchMastByCriteria", "GetMastProducts",
    "DownloadMastData", "SearchEso", "SearchIrsa",
]
