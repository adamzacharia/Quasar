"""
capabilities/catalogs.py — dedicated archive tools over well-known TAP services
and public archive APIs (ArchiveBench gap closing, 2026-09).

New family (not a legacy migration), same transport-pure shape as
``capabilities/spectra.py``:

  * the service getter (``get_archive_catalog_service``), the target resolver
    (``live_imagery_coordinates``), the table-card helper
    (``external_catalog_table_result``) and the ADS client (``ads_client``)
    are injected through the CallContext;
  * every tool returns the service's scalar ``summary`` to the model (the
    table card preview is only 10 rows) plus the executed queries in
    ``provenance`` so the UI "Show query" panel shows them;
  * errors are errors: a failed archive call never becomes an empty result.

See services/archive_catalogs.py for the query builders.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from capabilities.base import BaseCapability, ToolResult

logger = logging.getLogger(__name__)

_NOTE = ("The key values are in `summary`: state them (numbers, identifiers, dates) in the answer and name the "
         "archive and selection used. The full table is also shown to the user as a data card.")


def _native(out: Dict[str, Any]) -> ToolResult:
    ok = bool(isinstance(out, dict) and out.get("success"))
    err = out.get("error") if isinstance(out, dict) else None
    return ToolResult(success=ok, error=(str(err) if (err is not None and not ok) else None), native=out)


class _In(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _resolve(ctx, target_name: Optional[str], ra: Optional[float], dec: Optional[float]):
    if ra is not None and dec is not None:
        return float(ra), float(dec), (target_name or f"RA {float(ra):.5f}, Dec {float(dec):.5f}")
    if not target_name:
        raise ValueError("give target_name or ra and dec")
    resolve = ctx.service("live_imagery_coordinates")
    return resolve(target_name=target_name, ra=None, dec=None)


def _finish(ctx, result: Dict[str, Any], *, tool: str, source: str, label: str) -> ToolResult:
    """Attach the table card and return the summary-first payload."""
    if not result.get("success"):
        return _native(result)
    table_result = ctx.service("external_catalog_table_result")
    rows = result.get("rows") or []
    columns = result.get("columns") or (list(rows[0].keys()) if rows else [])
    try:
        out = table_result(rows, columns=columns, source=source, filter_label=label, tool_name=tool,
                           warnings=result.get("warnings") or [], provenance=result.get("provenance") or {})
    except Exception:  # a card failure must not lose the answer
        logger.debug("table card failed for %s", tool, exc_info=True)
        out = {"success": True, "warnings": result.get("warnings") or [],
               "provenance": result.get("provenance") or {}}
    out["summary"] = result.get("summary") or {}
    out["note"] = _NOTE
    return _native(out)


def _fail(message: str) -> ToolResult:
    return _native({"success": False, "error": message})


# ── HEASARC ─────────────────────────────────────────────────────────────────
class HeasarcObservationsInput(_In):
    mission: str
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcmin: Optional[float] = None
    instrument: Optional[str] = None
    public_only: bool = False
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: Optional[int] = 100


class HeasarcObservations(BaseCapability):
    name = "heasarc_observations"
    description = (
        "Search the HEASARC mission master observation tables (Chandra, XMM-Newton, Swift, NuSTAR, NICER, "
        "Suzaku, ROSAT) for pointed X-ray observations near a target or position. Returns the archive's own "
        "COUNT and total exposure, first/last observation dates, public status, and the ObsID list, with "
        "optional instrument/grating filter (Chandra ACIS/HRC/HETG/LETG, XMM pn/mos/rgs/om, Swift xrt/uvot/bat), "
        "public-only and date window. Use for any 'has Chandra/XMM/Swift/NuSTAR observed X' or ObsID question."
    )
    category = "archive"
    InputModel = HeasarcObservationsInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            ra, dec, label = _resolve(ctx, inp.target_name, inp.ra, inp.dec)
            svc = ctx.service("get_archive_catalog_service")()
            res = svc.heasarc_observations(inp.mission, ra, dec, radius_arcmin=inp.radius_arcmin,
                                           instrument=inp.instrument, public_only=inp.public_only,
                                           start_date=inp.start_date, end_date=inp.end_date, limit=inp.limit,
                                           target_label=label)
            mission = (res.get("summary") or {}).get("mission") or inp.mission
            return _finish(ctx, res, tool=self.name, source=f"HEASARC {mission} master table",
                           label=f"{mission} observations near {label}")
        except Exception as exc:
            return _fail(f"heasarc_observations failed: {exc}")


# ── Exoplanet Archive ───────────────────────────────────────────────────────
class ExoplanetArchiveInput(_In):
    mode: Optional[str] = None
    planet_name: Optional[str] = None
    hostname: Optional[str] = None
    max_distance_pc: Optional[float] = None
    min_distance_pc: Optional[float] = None
    min_host_teff_k: Optional[float] = None
    max_host_teff_k: Optional[float] = None
    min_eq_temp_k: Optional[float] = None
    max_eq_temp_k: Optional[float] = None
    min_radius_earth: Optional[float] = None
    max_radius_earth: Optional[float] = None
    min_mass_earth: Optional[float] = None
    max_mass_earth: Optional[float] = None
    min_period_days: Optional[float] = None
    max_period_days: Optional[float] = None
    min_disc_year: Optional[int] = None
    max_disc_year: Optional[int] = None
    discovery_facility: Optional[str] = None
    discovery_method: Optional[str] = None
    cuts: Optional[Any] = None
    columns: Optional[List[str]] = None
    count_only: bool = False
    limit: Optional[int] = 200
    order_by: Optional[str] = None


_EXO_FILTER_KEYS = ["max_distance_pc", "min_distance_pc", "min_host_teff_k", "max_host_teff_k", "min_eq_temp_k",
                    "max_eq_temp_k", "min_radius_earth", "max_radius_earth", "min_mass_earth", "max_mass_earth",
                    "min_period_days", "max_period_days", "min_disc_year", "max_disc_year"]


class ExoplanetArchive(BaseCapability):
    name = "exoplanet_archive"
    description = (
        "Query the NASA Exoplanet Archive (confirmed planets). mode='lookup': one planet (planet_name) or all "
        "planets of a host (hostname) with composite parameters and uncertainties, the default published solution "
        "and its reference, the number of published solutions, and mass provenance (measured mass vs msini vs "
        "mass-radius estimate). mode='census': the archive's own COUNT (plus rows) of planets passing typed filters "
        "(distance, host Teff, equilibrium temperature, radius, mass, period, discovery year, discovery facility or "
        "method) or extra column cuts on pscomppars."
    )
    category = "archive"
    InputModel = ExoplanetArchiveInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            svc = ctx.service("get_archive_catalog_service")()
            mode = (inp.mode or ("lookup" if (inp.planet_name or inp.hostname) else "census")).lower()
            if mode == "lookup":
                res = svc.exoplanet_lookup(planet_name=inp.planet_name, hostname=inp.hostname)
                label = f"Exoplanet Archive: {inp.planet_name or inp.hostname}"
            elif mode == "census":
                filters = {k: getattr(inp, k) for k in _EXO_FILTER_KEYS if getattr(inp, k) is not None}
                res = svc.exoplanet_census(filters=filters, discovery_facility=inp.discovery_facility,
                                           discovery_method=inp.discovery_method, cuts=inp.cuts, columns=inp.columns,
                                           count_only=inp.count_only, limit=inp.limit, order_by=inp.order_by)
                label = "Exoplanet Archive census: " + (", ".join((res.get("summary") or {}).get("filters") or [])
                                                        or "all confirmed planets")
            else:
                return _fail("mode must be 'lookup' or 'census'")
            return _finish(ctx, res, tool=self.name, source="NASA Exoplanet Archive", label=label)
        except Exception as exc:
            return _fail(f"exoplanet_archive failed: {exc}")


# ── SIMBAD ──────────────────────────────────────────────────────────────────
class SimbadQueryInput(_In):
    mode: Optional[str] = None
    identifiers: Optional[List[str]] = None
    identifier: Optional[str] = None
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: Optional[float] = None
    positions: Optional[List[Any]] = None
    otype: Optional[str] = None
    limit: Optional[int] = 50


class SimbadQuery(BaseCapability):
    name = "simbad_query"
    description = (
        "Query the SIMBAD database itself (not a web page). mode='lookup': identifiers -> main id, object type "
        "with its meaning, coordinates, redshift or radial velocity with its bibcode and quality, parallax, proper "
        "motion, spectral type, Gaia DR3 id, and an explicit not-found for unknown names. mode='cone': objects "
        "around a target or position (count + nearest, optional object type). mode='positions': a list of "
        "'ra, dec' positions -> nearest SIMBAD object within radius_arcsec for each, with separations."
    )
    category = "archive"
    InputModel = SimbadQueryInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            svc = ctx.service("get_archive_catalog_service")()
            ids = list(inp.identifiers or [])
            if inp.identifier:
                ids.insert(0, inp.identifier)
            mode = (inp.mode or ("positions" if inp.positions else "lookup" if ids else "cone")).lower()
            if mode == "lookup":
                if not ids and inp.target_name:
                    ids = [inp.target_name]
                res = svc.simbad_lookup(ids)
                label = "SIMBAD lookup: " + ", ".join(ids[:5])
            elif mode == "positions":
                res = svc.simbad_positions(inp.positions or [], inp.radius_arcsec if inp.radius_arcsec else 5)
                label = f"SIMBAD nearest objects for {len(inp.positions or [])} positions"
            elif mode == "cone":
                ra, dec, label0 = _resolve(ctx, inp.target_name, inp.ra, inp.dec)
                res = svc.simbad_cone(ra, dec, inp.radius_arcsec or 60, limit=inp.limit, otype=inp.otype)
                label = f"SIMBAD cone around {label0}"
            else:
                return _fail("mode must be 'lookup', 'cone' or 'positions'")
            return _finish(ctx, res, tool=self.name, source="CDS SIMBAD", label=label)
        except Exception as exc:
            return _fail(f"simbad_query failed: {exc}")


# ── Gaia ────────────────────────────────────────────────────────────────────
class GaiaArchiveInput(_In):
    mode: Optional[str] = None
    target_name: Optional[str] = None
    source_id: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: Optional[float] = None
    cuts: Optional[Any] = None
    columns: Optional[List[str]] = None
    count_only: bool = False
    limit: Optional[int] = 100
    variability_table: Optional[str] = None
    order_by: Optional[str] = None


class GaiaArchiveQuery(BaseCapability):
    name = "gaia_archive_query"
    description = (
        "Query the ESA Gaia DR3 archive. mode='source': one star by name (via SIMBAD's Gaia DR3 cross-id, else a "
        "cone at the position propagated to epoch 2016.0 with its proper motion), by source_id, or nearest to a "
        "position -> parallax, proper motion, G/BP/RP with errors, RUWE, RV. mode='cone': the archive's own COUNT "
        "(and rows) of gaia_source in a cone with column cuts such as 'parallax BETWEEN 5 AND 8', "
        "'phot_g_mean_mag < 18'. mode='variability': the DR3 variability record (rrlyrae, cepheid, "
        "eclipsing_binary, long_period_variable, summary, classifier, ...) of a star (target_name or source_id), "
        "with its period and mean magnitudes."
    )
    category = "archive"
    InputModel = GaiaArchiveInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            svc = ctx.service("get_archive_catalog_service")()
            mode = (inp.mode or ("variability" if inp.variability_table else
                                 "cone" if (inp.cuts or inp.count_only) else "source")).lower()
            if mode == "source":
                if inp.source_id or inp.target_name:
                    res = svc.gaia_source(source_id=inp.source_id, identifier=inp.target_name,
                                          ra=inp.ra, dec=inp.dec, radius_arcsec=inp.radius_arcsec or 5)
                else:
                    res = svc.gaia_source(ra=inp.ra, dec=inp.dec, radius_arcsec=inp.radius_arcsec or 5)
                label = f"Gaia DR3 source: {inp.target_name or inp.source_id or (inp.ra, inp.dec)}"
            elif mode == "cone":
                ra, dec, label0 = _resolve(ctx, inp.target_name, inp.ra, inp.dec)
                if not inp.radius_arcsec:
                    return _fail("mode='cone' needs radius_arcsec")
                res = svc.catalog_query("gaia", "gaiadr3.gaia_source", ra=ra, dec=dec, radius_arcsec=inp.radius_arcsec,
                                        columns=inp.columns or ["source_id", "ra", "dec", "parallax", "pmra", "pmdec",
                                                                "phot_g_mean_mag", "bp_rp"],
                                        cuts=inp.cuts, count_only=inp.count_only, limit=inp.limit,
                                        order_by=inp.order_by)
                label = f"Gaia DR3 cone around {label0}, r={inp.radius_arcsec:g} arcsec"
                if res.get("success"):
                    res.setdefault("warnings", []).append("Gaia DR3 positions are epoch 2016.0")
            elif mode == "variability":
                sid = inp.source_id
                prelim: Dict[str, Any] = {}
                if not sid:
                    prelim = svc.gaia_source(identifier=inp.target_name, ra=inp.ra, dec=inp.dec,
                                             radius_arcsec=inp.radius_arcsec or 5)
                    if not prelim.get("success"):
                        return _native(prelim)
                    sid = ((prelim.get("summary") or {}).get("source") or {}).get("source_id")
                    if not sid:
                        return _native({**prelim, "note": "no Gaia DR3 source found, so no variability record"})
                res = svc.gaia_variability(inp.variability_table or "summary", sid)
                if res.get("success") and prelim:
                    res["summary"]["source_match"] = (prelim.get("summary") or {}).get("method")
                    res["provenance"]["queries"] = ((prelim.get("provenance") or {}).get("queries") or []) + \
                        (res["provenance"].get("queries") or [])
                    res["provenance"]["query"] = "\n\n".join(res["provenance"]["queries"])
                label = f"Gaia DR3 {inp.variability_table or 'summary'} for {inp.target_name or sid}"
            else:
                return _fail("mode must be 'source', 'cone' or 'variability'")
            return _finish(ctx, res, tool=self.name, source="ESA Gaia DR3 archive", label=label)
        except Exception as exc:
            return _fail(f"gaia_archive_query failed: {exc}")


# ── catalogue finder / query / cross-match ─────────────────────────────────
class CatalogFindInput(_In):
    keywords: str
    services: Optional[List[str]] = None
    limit: Optional[int] = 12


class CatalogFind(BaseCapability):
    name = "catalog_find"
    description = (
        "Find the exact table name of a published catalogue or survey by keywords (survey name, author, topic) "
        "across VizieR (ranked by VizieR's own search), HEASARC and IRSA table metadata. Returns service + table "
        "names to pass to catalog_query or catalog_crossmatch."
    )
    category = "archive"
    InputModel = CatalogFindInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            svc = ctx.service("get_archive_catalog_service")()
            services = [s.lower() for s in (inp.services or ["vizier", "heasarc", "irsa"])]
            res = svc.catalog_find(inp.keywords, services=services, limit=inp.limit)
            return _finish(ctx, res, tool=self.name, source="VizieR / HEASARC / IRSA table metadata",
                           label=f"catalogues matching '{inp.keywords}'")
        except Exception as exc:
            return _fail(f"catalog_find failed: {exc}")


class CatalogQueryInput(_In):
    service: str
    table: str
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: Optional[float] = None
    columns: Optional[List[str]] = None
    cuts: Optional[Any] = None
    count_only: bool = False
    limit: Optional[int] = 100
    order_by: Optional[str] = None
    descending: bool = False


class CatalogQuery(BaseCapability):
    name = "catalog_query"
    description = (
        "Query one catalogue table on VizieR, IRSA, HEASARC, Gaia, SIMBAD or the Exoplanet Archive (service = "
        "vizier|irsa|heasarc|gaia|simbad|exoplanet): optional cone (target or ra/dec + radius_arcsec; the RA/Dec "
        "columns are found automatically), column cuts like 'Jmag < 14' or {column, op, value}, and returns the "
        "archive's own COUNT of matching rows plus the nearest/ordered rows. Column names are checked against the "
        "table schema. Use catalog_find first when the exact table name is unknown."
    )
    category = "archive"
    InputModel = CatalogQueryInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            ra = dec = None
            where = ""
            if inp.target_name or (inp.ra is not None and inp.dec is not None):
                ra, dec, label0 = _resolve(ctx, inp.target_name, inp.ra, inp.dec)
                where = f" around {label0}"
            svc = ctx.service("get_archive_catalog_service")()
            res = svc.catalog_query(inp.service, inp.table, ra=ra, dec=dec, radius_arcsec=inp.radius_arcsec,
                                    columns=inp.columns, cuts=inp.cuts, count_only=inp.count_only, limit=inp.limit,
                                    order_by=inp.order_by, descending=inp.descending)
            return _finish(ctx, res, tool=self.name, source=f"{inp.service}:{inp.table}",
                           label=f"{inp.table}{where}")
        except Exception as exc:
            return _fail(f"catalog_query failed: {exc}")


class CatalogCrossmatchInput(_In):
    left_service: str
    left_table: str
    right_service: str
    right_table: str
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: float
    match_radius_arcsec: float
    left_cuts: Optional[Any] = None
    right_cuts: Optional[Any] = None
    max_rows_per_side: Optional[int] = 20000


class CatalogCrossmatch(BaseCapability):
    name = "catalog_crossmatch"
    description = (
        "Positionally cross-match two catalogue tables (each on vizier|irsa|heasarc|gaia|simbad|exoplanet, may be "
        "different services) inside one cone: both sides are selected by the archives (cone + optional cuts), "
        "then matched within match_radius_arcsec. Returns how many LEFT sources have a match, out of how many, "
        "the matched fraction, how many right sources are matched, separation statistics and whether both sides "
        "were complete. Use for 'how many X have a Y counterpart within N arcsec'."
    )
    category = "archive"
    InputModel = CatalogCrossmatchInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            ra, dec, label0 = _resolve(ctx, inp.target_name, inp.ra, inp.dec)
            svc = ctx.service("get_archive_catalog_service")()
            res = svc.catalog_crossmatch(
                {"service": inp.left_service, "table": inp.left_table, "cuts": inp.left_cuts},
                {"service": inp.right_service, "table": inp.right_table, "cuts": inp.right_cuts},
                ra=ra, dec=dec, radius_arcsec=inp.radius_arcsec, match_radius_arcsec=inp.match_radius_arcsec,
                max_rows_per_side=inp.max_rows_per_side)
            return _finish(ctx, res, tool=self.name, source="catalogue cross-match",
                           label=f"{inp.left_table} x {inp.right_table} around {label0}, "
                                 f"match {inp.match_radius_arcsec:g} arcsec")
        except Exception as exc:
            return _fail(f"catalog_crossmatch failed: {exc}")


# ── time domain ─────────────────────────────────────────────────────────────
class TnsObjectInput(_In):
    name: str


class TnsObject(BaseCapability):
    name = "tns_object"
    description = (
        "Look up a transient on the Transient Name Server by its IAU designation (e.g. 'SN 2024abc', 'AT 2019xyz'): "
        "classification, redshift, host galaxy, discovery date and magnitude, reporting group and internal survey "
        "names, from the public TNS object record."
    )
    category = "archive"
    InputModel = TnsObjectInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            svc = ctx.service("get_archive_catalog_service")()
            res = svc.tns_object(inp.name)
            return _finish(ctx, res, tool=self.name, source="Transient Name Server", label=f"TNS {inp.name}")
        except Exception as exc:
            return _fail(f"tns_object failed: {exc}")


class ZtfObjectInput(_In):
    oid: Optional[str] = None
    target_name: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    radius_arcsec: Optional[float] = None


class ZtfObject(BaseCapability):
    name = "ztf_object"
    description = (
        "ALeRCE summary of ONE ZTF alert object, by ZTF id or the object nearest a target/position: number of "
        "alert detections, first and last detection as MJD and calendar date (UTC), time span, mean position and "
        "the top classifier labels. Use for 'how many ZTF detections / since when' questions."
    )
    category = "archive"
    InputModel = ZtfObjectInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        try:
            svc = ctx.service("get_archive_catalog_service")()
            if inp.oid:
                res = svc.ztf_object(oid=inp.oid)
                label = f"ALeRCE {inp.oid}"
            else:
                ra, dec, label0 = _resolve(ctx, inp.target_name, inp.ra, inp.dec)
                res = svc.ztf_object(ra=ra, dec=dec, radius_arcsec=inp.radius_arcsec or 5)
                label = f"ALeRCE object nearest {label0}"
            return _finish(ctx, res, tool=self.name, source="ALeRCE ZTF broker", label=label)
        except Exception as exc:
            return _fail(f"ztf_object failed: {exc}")


# ── ADS ─────────────────────────────────────────────────────────────────────
class AdsSearchInput(_In):
    author: Optional[Any] = None
    first_author: Optional[str] = None
    bibstem: Optional[Any] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    title_words: Optional[str] = None
    abstract_words: Optional[str] = None
    full_text: Optional[str] = None
    doi: Optional[str] = None
    bibcode: Optional[str] = None
    refereed: Optional[bool] = None
    extra_query: Optional[str] = None
    sort: Optional[str] = "date"
    rows: Optional[int] = 25


class AdsSearch(BaseCapability):
    name = "ads_search"
    description = (
        "Fielded NASA ADS search that returns the exact number of matching papers (numFound) plus bibcodes, "
        "titles, first authors, years, journals and DOIs. Fields: author ('Last, First'), first_author, bibstem "
        "(journal abbreviation such as ApJ, MNRAS, A&A, AJ, Nature, Sci), year_from/year_to, title_words, "
        "abstract_words, full_text (every word must appear), doi, bibcode, refereed, sort (date|oldest|citations|"
        "relevance). Use for paper counts by author/journal/year and for finding a specific paper's bibcode and DOI."
    )
    category = "literature"
    InputModel = AdsSearchInput
    annotations = {"read_only": True, "cost": "network"}

    def run(self, inp, ctx) -> ToolResult:
        from services.archive_catalogs import ADS_FIELDS, ADS_SORTS, build_ads_query

        client = ctx.service("ads_client")
        if client is None:
            return _fail("ADS client not configured (no ADS API key)")
        try:
            q, fq = build_ads_query(author=inp.author, first_author=inp.first_author, bibstem=inp.bibstem,
                                    year_from=inp.year_from, year_to=inp.year_to, title_words=inp.title_words,
                                    abstract_words=inp.abstract_words, full_text=inp.full_text, doi=inp.doi,
                                    bibcode=inp.bibcode, refereed=inp.refereed, extra_query=inp.extra_query)
        except ValueError as exc:
            return _fail(str(exc))
        sort = ADS_SORTS.get(str(inp.sort or "date").lower(), "date desc")
        rows_n = max(1, min(int(inp.rows or 25), 200))
        params: Dict[str, Any] = {"q": q, "fl": ADS_FIELDS, "rows": rows_n, "sort": sort}
        if fq:
            params["fq"] = fq
        prov = {"service": "ads", "endpoint": "https://api.adsabs.harvard.edu/v1/search/query", "query": q,
                "fq": fq, "sort": sort, "rows": rows_n}
        try:
            data = client._perform_get("/search/query", params)
        except Exception as exc:
            return _native({"success": False, "error": f"ADS search failed: {exc}", "provenance": prov})
        response = (data or {}).get("response") or {}
        docs = response.get("docs") or []
        papers = []
        for d in docs:
            authors = d.get("author") or []
            papers.append({
                "bibcode": d.get("bibcode"), "title": (d.get("title") or [""])[0], "first_author": authors[0] if authors else None,
                "n_authors": len(authors), "year": d.get("year"), "journal": d.get("pub"),
                "doi": (d.get("doi") or [None])[0], "citations": d.get("citation_count"),
                "refereed": "REFEREED" in (d.get("property") or []), "pubdate": d.get("pubdate"),
            })
        num = int(response.get("numFound") or 0)
        summary = {"num_found": num, "query": q, "filters": fq, "sort": sort, "listed": len(papers),
                   "list_truncated": num > len(papers), "papers": papers[:25]}
        result = {"success": True, "summary": summary, "rows": papers,
                  "columns": ["bibcode", "year", "first_author", "title", "journal", "doi", "citations"],
                  "warnings": ([] if num else ["ADS returned no papers for this query (a real zero for these fields)"]),
                  "provenance": prov}
        return _finish(ctx, result, tool=self.name, source="NASA ADS", label=f"ADS: {q}")


CAPABILITIES = [
    HeasarcObservations(), ExoplanetArchive(), SimbadQuery(), GaiaArchiveQuery(), CatalogFind(), CatalogQuery(),
    CatalogCrossmatch(), TnsObject(), ZtfObject(), AdsSearch(),
]
