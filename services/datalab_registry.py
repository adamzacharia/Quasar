"""Static NOIRLab Astro Data Lab catalog registry and TAP_SCHEMA cache.

Table/column names here are AUTHORITATIVE — the builders and the SQL governor
trust them. They are taken from the verified notebook SQL (the suggested-prompts
PDF), not guessed. Keep them faithful to the real Data Lab schema.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd


COMMON_RA_DEC = ["ra", "dec"]


DATALAB_CATALOGS: Dict[str, Dict[str, Any]] = {
    "gaia_dr3": {
        "region_strategy": "q3c",
        "tables": {
            "gaia_source": {
                "columns": [
                    "source_id", "ra", "dec",
                    "parallax", "parallax_over_error", "pmra", "pmdec", "pm",
                    "ruwe", "ipd_frac_multi_peak", "astrometric_sigma5d_max",
                    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
                    "bp_rp", "radial_velocity",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [],
        "morphology": {},
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "All-sky (space astrometry mission; covers the LMC/SMC, Galactic plane, both hemispheres).",
        "citation": {
            "text": "Gaia Data Release 3",
            "doi": "10.1051/0004-6361/202243940",
            "url": "https://www.cosmos.esa.int/web/gaia/dr3",
        },
    },
    "nsc_dr2": {
        "region_strategy": "q3c",
        "tables": {
            "object": {
                # NSC uses gmag/rmag/... (NOT mag_auto_*); carries class_star, fwhm
                # and precomputed HEALPix columns ring256 (RING) and nest4096 (NEST).
                "columns": [
                    "id", "ra", "dec",
                    "gmag", "rmag", "imag", "zmag", "gerr", "rerr",
                    "class_star", "fwhm", "ring256", "nest4096",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [
            {"name": "ring256", "nside": 256, "scheme": "RING"},
            {"name": "nest4096", "nside": 4096, "scheme": "NEST"},
        ],
        "morphology": {"class_star": "class_star ~1 = star-like, lower = extended (point-source proxy)"},
        "bitmasks": {},
        "sia_endpoints": ["https://datalab.noirlab.edu/sia/coadd_all"],
        "footprint": "~35,000 deg² of archival DECam/Bok/Mosaic imaging — nearly all of the sky south of Dec ≈ +40° plus patchy northern coverage; includes the LMC/SMC region and much of the Galactic plane (depth varies strongly by field).",
        "citation": {
            "text": "NOIRLab Source Catalog Data Release 2",
            "doi": None,
            "url": "https://datalab.noirlab.edu/nscdr2/",
        },
    },
    "des_dr1": {
        "region_strategy": "q3c",
        "tables": {
            "main": {
                # DES DR1 columns are PER-BAND (verified against the live service): there is no bare
                # `spread_model` or `class_star` — use spread_model_r / class_star_r, etc.
                "columns": [
                    "coadd_object_id", "ra", "dec",
                    "mag_auto_g", "mag_auto_r", "mag_auto_i", "mag_auto_z",
                    "mag_auto_g_dered", "mag_auto_r_dered", "mag_auto_i_dered", "mag_auto_z_dered",
                    "magerr_auto_g", "magerr_auto_r", "magerr_auto_i", "magerr_auto_z",
                    "flags_g", "flags_r", "flags_i", "flags_z",
                    "spread_model_g", "spread_model_r", "spread_model_i", "spread_model_z",
                    "spreaderr_model_r",
                    "class_star_g", "class_star_r", "class_star_i", "class_star_z",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [],
        "morphology": {
            "spread_model_r": "DES star/galaxy via spread_model_r > 0.005 => galaxy, ~0 => star. PER-BAND column — use spread_model_r (not 'spread_model'); also spread_model_g/i/z.",
            "class_star_r": "class_star_r near 1 = star-like (per-band: class_star_g/i/z).",
        },
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "~5,000 deg² of the southern high-Galactic-latitude sky (roughly -65° < Dec < +5°, avoiding the Galactic plane); does NOT cover the LMC/SMC main bodies.",
        "citation": {"text": "Dark Energy Survey Data Release 1", "doi": "10.3847/1538-4365/ab4f2b", "url": "https://des.ncsa.illinois.edu/releases/dr1"},
    },
    "smash_dr1": {
        "region_strategy": "q3c",
        "tables": {
            # P7 overdensity uses the OBJECT table (fieldid, depthflag, sharp, gmag, rmag).
            "object": {
                "columns": [
                    "id", "ra", "dec", "fieldid", "depthflag", "sharp", "chi",
                    "gmag", "rmag", "imag", "zmag", "gerr", "rerr",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            },
            # P14 variable star uses the multi-epoch SOURCE table; error column is `cerr`.
            "source": {
                "columns": ["id", "ra", "dec", "mjd", "filter", "cmag", "cerr", "chi", "sharp"],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            },
        },
        "healpix_columns": [],
        "morphology": {"sharp": "stellar sharpness diagnostic (|sharp|<0.5 ~ stellar)", "depthflag": "exposure-depth flag (>1 = deeper)"},
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "~480 deg² of targeted DECam fields covering the Magellanic system — LMC, SMC, Bridge, and periphery; field-partitioned (query by fieldid).",
        "citation": {"text": "Survey of the MAgellanic Stellar History Data Release 1", "doi": "10.3847/1538-4365/ab6e6c", "url": "https://datalab.noirlab.edu/smash/"},
    },
    "smash_dr2": {
        "region_strategy": "q3c",
        "tables": {
            "object": {
                "columns": [
                    "id", "ra", "dec", "fieldid", "depthflag", "sharp", "chi",
                    "gmag", "rmag", "imag", "zmag", "gerr", "rerr",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            },
            "source": {
                "columns": ["id", "ra", "dec", "mjd", "filter", "cmag", "cerr", "chi", "sharp"],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            },
        },
        "healpix_columns": [],
        "morphology": {"sharp": "stellar sharpness diagnostic", "depthflag": "exposure-depth flag"},
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "~480 deg² of targeted DECam fields covering the Magellanic system — LMC, SMC, Bridge, and periphery; field-partitioned (query by fieldid).",
        "citation": {"text": "Survey of the MAgellanic Stellar History Data Release 2", "doi": None, "url": "https://datalab.noirlab.edu/smash/"},
    },
    "delve_dr3": {
        "region_strategy": "q3c",
        "tables": {
            # P15 uses delve_dr3.coadd_objects with ext_coadd star/galaxy + hpix_* columns.
            "coadd_objects": {
                "columns": [
                    "quick_object_id", "ra", "dec",
                    "mag_auto_g", "magerr_auto_g", "mag_auto_r", "magerr_auto_r", "mag_auto_i",
                    "ext_coadd", "hpix_1024", "hpix_4096", "hpix_16384",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [
            {"name": "hpix_1024", "nside": 1024, "scheme": "NEST"},
            {"name": "hpix_4096", "nside": 4096, "scheme": "NEST"},
            {"name": "hpix_16384", "nside": 16384, "scheme": "NEST"},
        ],
        "morphology": {"ext_coadd": "0=hi-conf star, 1=candidate star, 2=mostly galaxy, 3=hi-conf galaxy, -9=no data"},
        "bitmasks": {},
        "sia_endpoints": ["https://datalab.noirlab.edu/sia/delve_dr3", "https://datalab.noirlab.edu/sia/coadd_all"],
        "footprint": "~21,000 deg² of the southern high-Galactic-latitude sky (DECam, Dec ≲ +30°), including dedicated coverage of the Magellanic periphery; avoids the inner Galactic plane.",
        "citation": {"text": "DECam Local Volume Exploration Survey Data Release 3", "doi": None, "url": "https://datalab.noirlab.edu/delve/"},
    },
    "desi_dr1": {
        "region_strategy": "q3c",
        "tables": {
            # P11 uses desi_dr1.zpix; coords are mean_fiber_ra/mean_fiber_dec.
            "zpix": {
                "columns": [
                    "targetid", "mean_fiber_ra", "mean_fiber_dec",
                    "z", "zerr", "zwarn", "spectype", "desi_target", "survey", "main_primary",
                ],
                "ra_column": "mean_fiber_ra",
                "dec_column": "mean_fiber_dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [],
        "morphology": {},
        "bitmasks": {"desi_target": {"LRG": 0, "ELG": 1, "QSO": 2, "BGS_ANY": 60, "MWS_ANY": 61}},
        "sia_endpoints": [],
        "footprint": "DESI spectroscopic footprint: high-Galactic-latitude northern/equatorial sky (roughly Dec > -20°); does NOT cover the LMC/SMC or the Galactic plane.",
        "citation": {"text": "DESI Data Release 1", "doi": None, "url": "https://data.desi.lbl.gov/doc/releases/dr1/"},
    },
    "sdss_dr17": {
        "region_strategy": "q3c",
        "tables": {
            "specobj": {
                # SDSS spectro quality flag is `zwarning` (not zwarn). specobj tolerates a box.
                "columns": ["specobjid", "ra", "dec", "z", "zerr", "zwarning", "class", "subclass", "plate", "mjd", "fiberid"],
                "ra_column": "ra",
                "dec_column": "dec",
                "region_strategy": "box_ok",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [],
        "morphology": {"class": "SDSS spectroscopic class label (GALAXY/STAR/QSO)"},
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "~14,500 deg² of mostly northern high-Galactic-latitude sky (Dec ≳ -10° plus equatorial stripes); does NOT cover the LMC/SMC.",
        "citation": {"text": "Sloan Digital Sky Survey Data Release 17", "doi": "10.3847/1538-4365/acda98", "url": "https://www.sdss4.org/dr17/"},
    },
    "ls_dr9": {
        "region_strategy": "q3c",
        "tables": {
            "tractor": {
                # Forced unWISE W1/W2 only — W3/W4 are NOT in dered_mag_* (guardrail).
                "columns": [
                    "release", "brickid", "objid", "ra", "dec", "type",
                    "dered_mag_g", "dered_mag_r", "dered_mag_z", "dered_mag_w1", "dered_mag_w2",
                    "snr_g", "snr_r", "snr_z", "snr_w1", "snr_w2",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [],
        "morphology": {"type": "Legacy Surveys Tractor morphological type; type != 'PSF' => extended"},
        "bitmasks": {},
        "sia_endpoints": ["https://datalab.noirlab.edu/sia/coadd_all"],
        "footprint": "~19,700 deg² of high-Galactic-latitude sky in both hemispheres (roughly -68° < Dec < +84°, |b| ≳ 18°); does NOT cover the LMC/SMC or the Galactic plane.",
        "citation": {"text": "Legacy Surveys Data Release 9", "doi": None, "url": "https://www.legacysurvey.org/dr9/"},
    },
    "vhs_dr5": {
        "region_strategy": "q3c",
        "tables": {
            # Verified against live tap_schema.tables (2026-07-05): the DR5 relation is
            # vhs_dr5.vhs_cat_v3 — the old "source" name does not exist server-side and
            # every query against it returned HTTP 400. Coordinates are ra2000/dec2000.
            "vhs_cat_v3": {
                "columns": [
                    "sourceid", "ra2000", "dec2000",
                    "japermag3", "japermag3err", "hapermag3", "hapermag3err",
                    "ksapermag3", "ksapermag3err", "mergedclass", "pstar",
                    "ring256", "nest4096",
                ],
                "ra_column": "ra2000",
                "dec_column": "dec2000",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [
            {"name": "ring256", "nside": 256, "scheme": "RING"},
            {"name": "nest4096", "nside": 4096, "scheme": "NEST"},
        ],
        "morphology": {
            "mergedclass": "VSA merged morphology class (-1 star, -2 probable star, 1 galaxy)",
            "pstar": "probability the source is point-like (0-1)",
        },
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "Southern-hemisphere near-IR (J/H/Ks) survey (~20,000 deg², Dec < 0°); by design EXCLUDES tiles owned by other VISTA surveys — VVV (Galactic plane/bulge) and VMC (inner LMC/SMC) — so inner Magellanic coverage is incomplete; verify per-position.",
        "citation": {"text": "VISTA Hemisphere Survey Data Release 5", "doi": None, "url": "https://datalab.noirlab.edu/"},
    },
}


# Columns whose equality predicate is an indexed, selective bound on its own —
# the PDF's canonical SMASH queries are bounded by `fieldid = N` (no cone), and
# its crowding-proof variable-star idiom is `id = '169.429960'`. The SQL governor
# accepts an equality on any of these as a valid row-level bound.
INDEXED_BOUND_COLUMNS: Dict[str, List[str]] = {
    "gaia_dr3.gaia_source": ["source_id"],
    "nsc_dr2.object": ["id"],
    "smash_dr1.object": ["fieldid", "id"],
    "smash_dr1.source": ["fieldid", "id"],
    "smash_dr2.object": ["fieldid", "id"],
    "smash_dr2.source": ["fieldid", "id"],
    "desi_dr1.zpix": ["targetid"],
    "des_dr1.main": ["coadd_object_id"],
    "delve_dr3.coadd_objects": ["quick_object_id"],
    "sdss_dr17.specobj": ["specobjid"],
}


def indexed_bound_columns(catalog: str, table: str) -> List[str]:
    """Columns whose equality predicate alone bounds a row-level query."""
    qualified = describe_table(catalog, table)["qualified_name"]
    return list(INDEXED_BOUND_COLUMNS.get(qualified, []))


def default_table(catalog: str) -> str:
    """The catalog's primary table — lets one-shot tools accept catalog-only calls."""
    catalog_key = _normalize_identifier(catalog)
    if catalog_key not in DATALAB_CATALOGS:
        raise ValueError(f"Unknown Data Lab catalog: {catalog}")
    tables = sorted(DATALAB_CATALOGS[catalog_key].get("tables", {}).keys())
    if not tables:
        raise ValueError(f"No tables registered for Data Lab catalog: {catalog}")
    return tables[0]


# Default point-source (star) selection per table, in the build_catalog_predicates
# morphology-argument schema. Used when a caller asks for point_sources without an
# explicit cut. Thresholds follow each survey's documented star/galaxy convention.
_POINT_SOURCE_CUTS = {
    "nsc_dr2.object": {"column": "class_star", "op": ">", "value": 0.5},
    "des_dr1.main": {"column": "spread_model_r", "between": [-0.005, 0.005]},
    "delve_dr3.coadd_objects": {"column": "ext_coadd", "between": [0, 1]},  # 0/1 = star/candidate star
    "smash_dr1.object": {"column": "sharp", "between": [-0.5, 0.5]},
    "smash_dr2.object": {"column": "sharp", "between": [-0.5, 0.5]},
    "vhs_dr5.vhs_cat_v3": {"column": "mergedclass", "in": [-1, -2]},  # VSA: -1 star, -2 probable star
}


def point_source_cut(catalog: str, table: str) -> Optional[Dict[str, Any]]:
    """Default star/point-source morphology cut for a table, or None if the catalog
    has no registered star/galaxy separator (e.g. gaia_dr3, desi_dr1)."""
    cut = _POINT_SOURCE_CUTS.get(describe_table(catalog, table)["qualified_name"])
    if not cut:
        return None
    return {k: (list(v) if isinstance(v, list) else v) for k, v in cut.items()}


def list_catalogs() -> List[Dict[str, Any]]:
    """Return a compact catalog list for tool output."""

    rows = []
    for name, entry in DATALAB_CATALOGS.items():
        rows.append(
            {
                "catalog": name,
                "tables": sorted(entry.get("tables", {}).keys()),
                "region_strategy": entry.get("region_strategy", "q3c"),
                "footprint": entry.get("footprint"),
                "citation": entry.get("citation", {}).get("text"),
            }
        )
    return rows


def describe_table(catalog: str, table: str) -> Dict[str, Any]:
    catalog_key = _normalize_identifier(catalog)
    table_key = _normalize_identifier(table)
    if catalog_key not in DATALAB_CATALOGS:
        raise ValueError(f"Unknown Data Lab catalog: {catalog}")
    tables = DATALAB_CATALOGS[catalog_key].get("tables", {})
    if table_key not in tables:
        if len(tables) == 1:
            # Tolerate a wrong table name when the catalog has exactly one table — the agent
            # commonly guesses 'object'/'main'; resolve to the catalog's canonical table.
            table_key = next(iter(tables))
        else:
            raise ValueError(
                f"Unknown Data Lab table: {catalog_key}.{table_key}. Available tables: {sorted(tables)}"
            )
    entry = DATALAB_CATALOGS[catalog_key]
    table_entry = dict(tables[table_key])
    return {
        "catalog": catalog_key,
        "table": table_key,
        "qualified_name": f"{catalog_key}.{table_key}",
        "columns": list(table_entry.get("columns", [])),
        "ra_column": table_entry.get("ra_column", "ra"),
        "dec_column": table_entry.get("dec_column", "dec"),
        "region_strategy": table_entry.get("region_strategy", entry.get("region_strategy", "q3c")),
        "healpix_columns": list(entry.get("healpix_columns", [])),
        "morphology": dict(entry.get("morphology", {})),
        "bitmasks": dict(entry.get("bitmasks", {})),
        "aggregate_safe": bool(table_entry.get("aggregate_safe")),
        "footprint": entry.get("footprint"),
        "citation": dict(entry.get("citation", {})),
    }


# Per-table magnitude column naming (band -> column). Used by the one-shot diagram tools so
# the agent need not know each catalog's column convention.
_MAG_TEMPLATES = {
    "des_dr1.main": "mag_auto_{band}",
    "nsc_dr2.object": "{band}mag",
    "smash_dr1.object": "{band}mag",
    "smash_dr2.object": "{band}mag",
    "delve_dr3.coadd_objects": "mag_auto_{band}",
    "ls_dr9.tractor": "dered_mag_{band}",
}


def mag_column(catalog: str, table: str, band: str) -> str:
    """Resolve a band ('g','r','i','z',...) to its magnitude column for a catalog."""
    info = describe_table(catalog, table)
    template = _MAG_TEMPLATES.get(info["qualified_name"], "{band}mag")
    return template.format(band=str(band).strip().lower())


def morphology_split_column(catalog: str, table: str) -> Optional[str]:
    """Best star/galaxy separation column for a catalog (spread_model_* preferred), or None."""
    morph = dict(describe_table(catalog, table).get("morphology", {}))
    for key in morph:
        if "spread_model" in key or key == "ext_coadd":
            return key
    return None


def region_strategy(catalog: str, table: str) -> str:
    return str(describe_table(catalog, table).get("region_strategy") or "q3c")


def citation(catalog: str) -> Dict[str, Any]:
    catalog_key = _normalize_identifier(catalog)
    if catalog_key not in DATALAB_CATALOGS:
        raise ValueError(f"Unknown Data Lab catalog: {catalog}")
    return dict(DATALAB_CATALOGS[catalog_key].get("citation", {}))


def aggregate_safe_tables() -> set[str]:
    safe = set()
    for catalog, entry in DATALAB_CATALOGS.items():
        for table, meta in entry.get("tables", {}).items():
            if meta.get("aggregate_safe"):
                safe.add(f"{catalog}.{table}")
    return safe


def refresh_tap_schema(client: Any, *, cache_dir: Path | str = Path("cache") / "datalab_tap_schema") -> Dict[str, Any]:
    """Fetch and cache TAP_SCHEMA tables/columns through a supplied DatalabClient."""

    payload = {
        "tables": client.query(
            sql="SELECT schema_name, table_name, table_type, description FROM tap_schema.tables",
            fmt="pandas",
        ).dataframe,
        "columns": client.query(
            sql="SELECT schema_name, table_name, column_name, datatype, description FROM tap_schema.columns",
            fmt="pandas",
        ).dataframe,
        "refreshed_at": time.time(),
    }
    _cache_schema(payload, cache_dir=cache_dir)
    return payload


def cached_tap_schema(*, cache_dir: Path | str = Path("cache") / "datalab_tap_schema") -> Optional[Dict[str, Any]]:
    cache = _open_cache(cache_dir)
    if cache is None:
        return None
    try:
        return cache.get("tap_schema_v1")
    finally:
        try:
            cache.close()
        except Exception:
            pass


def _cache_schema(payload: Mapping[str, Any], *, cache_dir: Path | str) -> None:
    cache = _open_cache(cache_dir)
    if cache is None:
        return
    try:
        cache.set("tap_schema_v1", dict(payload), expire=int(os.getenv("DATALAB_TAP_SCHEMA_TTL_SECONDS", "86400")))
    finally:
        try:
            cache.close()
        except Exception:
            pass


def _open_cache(cache_dir: Path | str):
    try:
        from diskcache import Cache
    except Exception:
        return None
    try:
        return Cache(str(cache_dir))
    except Exception:
        return None


def _normalize_identifier(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text or not text.replace("_", "a").isalnum():
        raise ValueError(f"Invalid Data Lab identifier: {value!r}")
    return text


def dataframe_from_schema(rows: List[Mapping[str, Any]]) -> pd.DataFrame:
    """Utility for tests and callers that need a stable schema frame."""

    return pd.DataFrame(list(rows))


__all__ = [
    "DATALAB_CATALOGS",
    "aggregate_safe_tables",
    "cached_tap_schema",
    "citation",
    "dataframe_from_schema",
    "describe_table",
    "list_catalogs",
    "refresh_tap_schema",
    "region_strategy",
]
