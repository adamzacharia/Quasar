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
                "columns": [
                    "coadd_object_id", "ra", "dec",
                    "mag_auto_g", "mag_auto_r", "mag_auto_i", "mag_auto_z",
                    "magerr_auto_g", "magerr_auto_r", "magerr_auto_i",
                    "fluxerr_auto_g", "fluxerr_auto_r", "fluxerr_auto_i",
                    "spread_model_r", "flags_g", "flags_r", "flags_i",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [],
        "morphology": {"spread_model_r": "spread_model_r > 0.003 => galaxy, else star"},
        "bitmasks": {},
        "sia_endpoints": [],
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
        "citation": {"text": "Legacy Surveys Data Release 9", "doi": None, "url": "https://www.legacysurvey.org/dr9/"},
    },
    "vhs_dr5": {
        "region_strategy": "q3c",
        "tables": {
            "source": {
                "columns": ["sourceid", "ra", "dec", "japermag3", "hapermag3", "ksapermag3", "mergedclass"],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [],
        "morphology": {"mergedclass": "VISTA source morphology class (J/H/Ks near-infrared photometry)"},
        "bitmasks": {},
        "sia_endpoints": [],
        "citation": {"text": "VISTA Hemisphere Survey Data Release 5", "doi": None, "url": "https://datalab.noirlab.edu/"},
    },
}


def list_catalogs() -> List[Dict[str, Any]]:
    """Return a compact catalog list for tool output."""

    rows = []
    for name, entry in DATALAB_CATALOGS.items():
        rows.append(
            {
                "catalog": name,
                "tables": sorted(entry.get("tables", {}).keys()),
                "region_strategy": entry.get("region_strategy", "q3c"),
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
        raise ValueError(f"Unknown Data Lab table: {catalog_key}.{table_key}")
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
        "citation": dict(entry.get("citation", {})),
    }


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
