"""Static NOIRLab Astro Data Lab catalog registry and TAP_SCHEMA cache.

Table/column names here are AUTHORITATIVE — the builders and the SQL governor
trust them. They are taken from the verified notebook SQL (the suggested-prompts
PDF), not guessed. Keep them faithful to the real Data Lab schema.
"""

from __future__ import annotations

import logging
import os
import re
import threading
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
    "unwise_dr1": {
        "region_strategy": "q3c",
        "tables": {
            "object": {
                # Live tap_schema-verified 2026-07-12. unWISE photometry is Vega:
                # flux_w1/w2 in Vega nanomaggies, mag_w1_vg/mag_w2_vg precomputed
                # Vega magnitudes (both indexed). The table also has a `primary`
                # column — deliberately NOT listed because PRIMARY is a reserved
                # SQL keyword and would need quoting.
                "columns": [
                    "unwise_objid", "ra", "dec",
                    "mag_w1_vg", "mag_w2_vg", "w1_w2_vg",
                    "flux_w1", "flux_w2", "dflux_w1", "dflux_w2",
                    "qf_w1", "qf_w2", "rchi2_w1", "rchi2_w2",
                    "fracflux_w1", "fracflux_w2", "fwhm_w1", "fwhm_w2",
                    "spread_model_w1", "spread_model_w2",
                    "flags_unwise_w1", "flags_unwise_w2",
                    "flags_info_w1", "flags_info_w2",
                    "coadd_id", "ring256", "nest4096",
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
        "morphology": {
            "spread_model_w1": "crowdsource spread_model in W1: ~0 = point source, positive = extended. The WISE PSF is ~6\" — blending is common in crowded fields.",
        },
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "All-sky unWISE coadd catalog (WISE W1/W2 from time-resolved coadds, ~0.7 mag deeper than AllWISE; Vega photometry; ~2 billion sources).",
        "citation": {
            "text": "unWISE Catalog (Schlafly, Meisner & Green 2019)",
            "doi": "10.3847/1538-4365/aafbea",
            "url": "https://catalog.unwise.me/",
        },
    },
    "twomass": {
        "region_strategy": "q3c",
        "tables": {
            # Live tap_schema-verified 2026-07-12: the PSC's Ks magnitude column
            # is k_m (band prefix j/h/k with template {band}_m); NEST HEALPix here
            # is nest256 (nside 256), unlike the nest4096 used by most catalogs.
            "psc": {
                "columns": [
                    "designation", "ra", "dec",
                    "j_m", "h_m", "k_m",
                    "j_cmsig", "h_cmsig", "k_cmsig",
                    "j_msigcom", "h_msigcom", "k_msigcom",
                    "j_snr", "h_snr", "k_snr",
                    "ph_qual", "cc_flg", "rd_flg", "bl_flg",
                    "gal_contam", "mp_flg", "use_src", "dup_src", "prox",
                    "jdate", "ring256", "nest256",
                ],
                "ra_column": "ra",
                "dec_column": "dec",
                "aggregate_safe": True,
            }
        },
        "healpix_columns": [
            {"name": "ring256", "nside": 256, "scheme": "RING"},
            {"name": "nest256", "nside": 256, "scheme": "NEST"},
        ],
        "morphology": {
            "gal_contam": "extended-source contamination flag: 0 = clean point source, 1/2 = contaminated by / part of a 2MASS XSC extended source.",
        },
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "All-sky near-IR J/H/Ks point-source catalog (2MASS PSC, ~471M sources; includes the LMC/SMC and the full Galactic plane).",
        "citation": {
            "text": "Two Micron All Sky Survey (Skrutskie et al. 2006)",
            "doi": "10.1086/498708",
            "url": "https://irsa.ipac.caltech.edu/Missions/2mass.html",
        },
    },
    "allwise": {
        "region_strategy": "q3c",
        "tables": {
            # Live tap_schema-verified 2026-07-12. Vega magnitudes w1mpro..w4mpro;
            # pmra/pmdec are AllWISE apparent motions (integer mas/yr, include
            # parallax — not proper motions in the Gaia sense).
            "source": {
                "columns": [
                    "cntr", "source_id", "designation", "ra", "dec",
                    "w1mpro", "w2mpro", "w3mpro", "w4mpro",
                    "w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro",
                    "w1snr", "w2snr", "w3snr", "w4snr",
                    "w1rchi2", "w2rchi2",
                    "cc_flags", "ext_flg", "var_flg", "ph_qual", "moon_lev",
                    "pmra", "pmdec", "sigpmra", "sigpmdec",
                    "j_m_2mass", "h_m_2mass", "k_m_2mass",
                    "ring256", "nest4096",
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
        "morphology": {
            "ext_flg": "0 = point source (profile fit consistent with the PSF); >0 = extended emission or association with a 2MASS XSC source.",
        },
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "All-sky mid-IR W1/W2/W3/W4 (3.4/4.6/12/22 μm) source catalog (AllWISE, ~748M sources; Vega mags; carries 2MASS crossmatch columns).",
        "citation": {
            "text": "AllWISE Source Catalog (Cutri et al. 2013; WISE mission: Wright et al. 2010)",
            "doi": "10.1088/0004-6256/140/6/1868",
            "url": "https://wise2.ipac.caltech.edu/docs/release/allwise/",
        },
    },
    "splus_dr4": {
        "region_strategy": "q3c",
        "tables": {
            # Live tap_schema-verified 2026-07-12: dual-mode photometry table.
            # 12-band Javalambre system — u,g,r,i,z broadbands + 7 narrow bands
            # (j0378..j0861), AB mags as {band}_auto with e_{band}_auto errors.
            "dual": {
                "columns": [
                    "id", "ra", "dec", "field",
                    "class_star", "fwhm", "ebv_sch",
                    "u_auto", "g_auto", "r_auto", "i_auto", "z_auto",
                    "j0378_auto", "j0395_auto", "j0410_auto", "j0430_auto",
                    "j0515_auto", "j0660_auto", "j0861_auto",
                    "e_u_auto", "e_g_auto", "e_r_auto", "e_i_auto", "e_z_auto",
                    "e_j0378_auto", "e_j0395_auto", "e_j0410_auto", "e_j0430_auto",
                    "e_j0515_auto", "e_j0660_auto", "e_j0861_auto",
                    "ring256", "nest4096",
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
        "morphology": {
            "class_star": "SExtractor CLASS_STAR (dual-mode detection image): near 1 = star-like, near 0 = extended.",
        },
        "bitmasks": {},
        "sia_endpoints": [],
        "footprint": "S-PLUS DR4: ~3,000 deg² of the southern sky (incl. Stripe 82 and Magellanic Cloud fields) in the 12-band Javalambre filter system; AB photometry.",
        "citation": {
            "text": "Southern Photometric Local Universe Survey DR4 (Mendes de Oliveira et al. 2019; DR4: Herpich et al. 2024)",
            "doi": "10.1093/mnras/stz1985",
            "url": "https://splus.cloud/",
        },
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
    # Live tap_schema-verified indexed columns (2026-07-12): unwise_objid,
    # twomass designation, allwise cntr/source_id, and splus id all carry
    # indexed=1 server-side; allwise designation does NOT (indexed=0).
    "unwise_dr1.object": ["unwise_objid"],
    "twomass.psc": ["designation"],
    "allwise.source": ["cntr", "source_id"],
    "splus_dr4.dual": ["id"],
}


def indexed_bound_columns(catalog: str, table: str) -> List[str]:
    """Columns whose equality predicate alone bounds a row-level query."""
    qualified = describe_table(catalog, table)["qualified_name"]
    return list(INDEXED_BOUND_COLUMNS.get(qualified, []))


# Columns whose values are STRINGS even when they look numeric — SMASH ids like
# '169.429960' carry a significant trailing zero. pandas dtype inference float-
# coerces them at CSV-parse time (live P14: id '169.429960' became 169.42996 and
# the whole light-curve analysis silently ran on a DIFFERENT, 24th-mag star), so
# the Data Lab client forces dtype=str on these columns when parsing results.
STRING_ID_COLUMNS: Dict[str, List[str]] = {
    "smash_dr1.object": ["id"],
    "smash_dr1.source": ["id"],
    "smash_dr2.object": ["id"],
    "smash_dr2.source": ["id"],
    "nsc_dr2.object": ["id"],
    "twomass.psc": ["designation"],
    "allwise.source": ["designation"],
    "unwise_dr1.object": ["unwise_objid"],
    "splus_dr4.dual": ["id"],
}


def string_id_columns(catalog: Optional[str] = None, table: Optional[str] = None) -> List[str]:
    """String-typed id columns for a table.

    For a known registered table, exactly its declared columns (possibly none —
    e.g. gaia_dr3 ids are true int64). When the table is unknown (async job
    results carry no table context), the union of every registered string-id
    name: forcing a stray numeric column named like an id to str is harmless,
    while float-coercing a real string id corrupts it.
    """
    if catalog and table:
        try:
            qualified = describe_table(catalog, table)["qualified_name"]
        except (KeyError, ValueError):
            qualified = None
        if qualified is not None:
            return list(STRING_ID_COLUMNS.get(qualified, []))
    names: List[str] = []
    for cols in STRING_ID_COLUMNS.values():
        for col in cols:
            if col not in names:
                names.append(col)
    return names


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
    "allwise.source": {"column": "ext_flg", "op": "=", "value": 0},  # 0 = PSF-consistent point source
    "splus_dr4.dual": {"column": "class_star", "op": ">", "value": 0.9},  # SExtractor CLASS_STAR convention
    # unwise_dr1 and twomass get no default cut: the 2MASS PSC is point sources by
    # construction, and unWISE's 6" PSF makes a blanket spread_model cut unreliable.
}


# Default per-table QUALITY cuts (value_cuts schema), applied by the catalog
# tools unless the caller supplies its own cut on the same column. Live P11:
# DESI LRG counts without zwarn/survey/main_primary were verified 11-27%
# inflated per redshift bin; DES colors without flags stretched CCD axes.
_DEFAULT_QUALITY_CUTS: Dict[str, List[Dict[str, Any]]] = {
    "desi_dr1.zpix": [
        {"column": "zwarn", "op": "=", "value": 0},
        {"column": "survey", "op": "=", "value": "main"},
        {"column": "main_primary", "op": "=", "value": "true"},
    ],
    "des_dr1.main": [
        {"column": "flags_g", "op": "=", "value": 0},
        {"column": "flags_r", "op": "=", "value": 0},
        {"column": "flags_i", "op": "=", "value": 0},
    ],
    "sdss_dr17.specobj": [
        {"column": "zwarning", "op": "=", "value": 0},
    ],
}


def default_quality_cuts(catalog: str, table: str) -> List[Dict[str, Any]]:
    """Registry default quality cuts for a table (copies), possibly empty."""
    qualified = describe_table(catalog, table)["qualified_name"]
    return [dict(cut) for cut in _DEFAULT_QUALITY_CUTS.get(qualified, [])]


def merge_default_quality_cuts(
    catalog: str,
    table: str,
    value_cuts: Optional[List[Dict[str, Any]]],
) -> tuple:
    """(merged_value_cuts, applied_defaults_description or None).

    A caller cut on the same column OVERRIDES the default (so an explicit
    `zwarn != 0` study is still possible); everything else is appended.
    """
    explicit = [dict(vc) for vc in (value_cuts or [])]
    try:
        defaults = default_quality_cuts(catalog, table)
    except (KeyError, ValueError):
        defaults = []
    if not defaults:
        return explicit, None
    explicit_cols = {str(vc.get("column", "")).strip().lower() for vc in explicit}
    applied = [d for d in defaults if str(d["column"]).lower() not in explicit_cols]
    if not applied:
        return explicit, None
    described = ", ".join(
        f"{d['column']} {d['op']} {d['value']!r}" if isinstance(d["value"], str)
        else f"{d['column']} {d['op']} {d['value']}"
        for d in applied
    )
    note = (
        f"Applied registry default quality cuts: {described}. Override by passing "
        "your own value_cut on those columns."
    )
    return explicit + applied, note


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
    # Verified expansion catalogs, reachable through the live TAP-schema fallback
    # (columns read from the live service; footprint/citation curated).
    for entry in EXPANSION_CATALOGS:
        rows.append(
            {
                "catalog": entry["catalog"],
                "tables": [entry["table"]],
                "region_strategy": "q3c",
                "footprint": entry.get("footprint"),
                "citation": (entry.get("citation") or {}).get("text"),
            }
        )
    return rows


def describe_table(catalog: str, table: str) -> Dict[str, Any]:
    catalog_key = _normalize_identifier(catalog)
    table_key = _normalize_identifier(table)
    if catalog_key not in DATALAB_CATALOGS:
        # Survey-breadth fallback: a table that was refreshed into the live
        # TAP-schema cache is describable (and thus governable) without a
        # curated registry entry.
        live_entry = live_table_entry(catalog_key, table_key)
        if live_entry is not None:
            return live_entry
        raise ValueError(f"Unknown Data Lab catalog: {catalog}")
    tables = DATALAB_CATALOGS[catalog_key].get("tables", {})
    if table_key not in tables:
        live_entry = live_table_entry(catalog_key, table_key)
        if live_entry is not None:
            return live_entry
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
    # Gaia bands are g|bp|rp; without this entry the '{band}mag' fallback emits
    # 'gmag', which does not exist server-side — the one-shot CMD/CCD tools then
    # 400 on every Gaia call (live P6 burned 4 rounds on it).
    "gaia_dr3.gaia_source": "phot_{band}_mean_mag",
    "nsc_dr2.object": "{band}mag",
    "smash_dr1.object": "{band}mag",
    "smash_dr2.object": "{band}mag",
    "delve_dr3.coadd_objects": "mag_auto_{band}",
    "ls_dr9.tractor": "dered_mag_{band}",
    "unwise_dr1.object": "mag_{band}_vg",  # band = w1|w2 (Vega)
    "twomass.psc": "{band}_m",             # band = j|h|k (k = Ks)
    "allwise.source": "{band}mpro",        # band = w1..w4 (Vega)
    "splus_dr4.dual": "{band}_auto",       # band = u|g|r|i|z|j0378..j0861 (AB)
}


# Valid band tokens where the survey's set differs from ugriz-style defaults —
# without this, a default blue/red_band of g/r on Gaia silently produces the
# nonexistent phot_r_mean_mag and fails only at the server.
_MAG_TEMPLATE_BANDS = {
    "gaia_dr3.gaia_source": {"g", "bp", "rp"},
    "unwise_dr1.object": {"w1", "w2"},
    "twomass.psc": {"j", "h", "k"},
    "allwise.source": {"w1", "w2", "w3", "w4"},
}


def mag_column(catalog: str, table: str, band: str) -> str:
    """Resolve a band ('g','r','i','z',...) to its magnitude column for a catalog."""
    info = describe_table(catalog, table)
    qualified = info["qualified_name"]
    band_key = str(band).strip().lower()
    allowed = _MAG_TEMPLATE_BANDS.get(qualified)
    if allowed is not None and band_key not in allowed:
        raise ValueError(
            f"{qualified} has no {band_key!r} band; valid bands: {'|'.join(sorted(allowed))}. "
            "For derived axes (e.g. absolute magnitude) pass x_expr/y_expr instead."
        )
    template = _MAG_TEMPLATES.get(qualified, "{band}mag")
    return template.format(band=band_key)


# Known Milky Way satellites and prominent globular clusters, for candidate
# hygiene in discovery workflows: a density peak inside an entry's veto radius
# is a KNOWN object, not a new candidate (live P15 "discovered" Draco as its
# top-ranked new-satellite candidate). Positions J2000, veto radii generous
# (roughly a few half-light radii; giants like the Clouds and Sgr get degrees).
KNOWN_MW_OBJECTS: List[Dict[str, Any]] = [
    {"name": "LMC", "ra": 80.894, "dec": -69.756, "radius_deg": 5.0, "kind": "satellite"},
    {"name": "SMC", "ra": 13.187, "dec": -72.829, "radius_deg": 3.0, "kind": "satellite"},
    {"name": "Sagittarius dSph", "ra": 283.831, "dec": -30.545, "radius_deg": 4.0, "kind": "satellite"},
    {"name": "Fornax dSph", "ra": 39.997, "dec": -34.449, "radius_deg": 1.0, "kind": "satellite"},
    {"name": "Sculptor dSph", "ra": 15.039, "dec": -33.709, "radius_deg": 0.8, "kind": "satellite"},
    {"name": "Draco dSph", "ra": 260.052, "dec": 57.915, "radius_deg": 0.7, "kind": "satellite"},
    {"name": "Ursa Minor dSph", "ra": 227.286, "dec": 67.222, "radius_deg": 0.8, "kind": "satellite"},
    {"name": "Sextans dSph", "ra": 153.262, "dec": -1.615, "radius_deg": 1.0, "kind": "satellite"},
    {"name": "Carina dSph", "ra": 100.403, "dec": -50.966, "radius_deg": 0.8, "kind": "satellite"},
    {"name": "Leo I", "ra": 152.117, "dec": 12.306, "radius_deg": 0.4, "kind": "satellite"},
    {"name": "Leo II", "ra": 168.370, "dec": 22.152, "radius_deg": 0.3, "kind": "satellite"},
    {"name": "Bootes I", "ra": 210.025, "dec": 14.500, "radius_deg": 0.6, "kind": "satellite"},
    {"name": "Bootes II", "ra": 209.500, "dec": 12.850, "radius_deg": 0.3, "kind": "satellite"},
    {"name": "Canes Venatici I", "ra": 202.015, "dec": 33.556, "radius_deg": 0.5, "kind": "satellite"},
    {"name": "Canes Venatici II", "ra": 194.292, "dec": 34.321, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Coma Berenices", "ra": 186.746, "dec": 23.904, "radius_deg": 0.4, "kind": "satellite"},
    {"name": "Hercules", "ra": 247.758, "dec": 12.792, "radius_deg": 0.4, "kind": "satellite"},
    {"name": "Segue 1", "ra": 151.767, "dec": 16.082, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Segue 2", "ra": 34.817, "dec": 20.175, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Ursa Major I", "ra": 158.720, "dec": 51.920, "radius_deg": 0.5, "kind": "satellite"},
    {"name": "Ursa Major II", "ra": 132.875, "dec": 63.130, "radius_deg": 0.5, "kind": "satellite"},
    {"name": "Willman 1", "ra": 162.339, "dec": 51.053, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Reticulum II", "ra": 53.925, "dec": -54.049, "radius_deg": 0.3, "kind": "satellite"},
    {"name": "Tucana II", "ra": 342.980, "dec": -58.569, "radius_deg": 0.4, "kind": "satellite"},
    {"name": "Horologium I", "ra": 43.882, "dec": -54.119, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Grus I", "ra": 344.177, "dec": -50.163, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Hydra II", "ra": 185.425, "dec": -31.985, "radius_deg": 0.3, "kind": "satellite"},
    {"name": "Crater II", "ra": 177.310, "dec": -18.413, "radius_deg": 1.2, "kind": "satellite"},
    {"name": "Antlia II", "ra": 143.887, "dec": -36.767, "radius_deg": 1.8, "kind": "satellite"},
    {"name": "Leo IV", "ra": 173.238, "dec": -0.533, "radius_deg": 0.3, "kind": "satellite"},
    {"name": "Leo V", "ra": 172.790, "dec": 2.220, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Pisces II", "ra": 344.629, "dec": 5.952, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Phoenix dwarf", "ra": 27.776, "dec": -44.445, "radius_deg": 0.3, "kind": "satellite"},
    {"name": "Tucana dwarf", "ra": 340.457, "dec": -64.420, "radius_deg": 0.3, "kind": "satellite"},
    {"name": "Triangulum II", "ra": 33.322, "dec": 36.178, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Draco II", "ra": 238.198, "dec": 64.565, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Hydrus I", "ra": 37.389, "dec": -79.309, "radius_deg": 0.3, "kind": "satellite"},
    {"name": "Carina II", "ra": 114.107, "dec": -57.999, "radius_deg": 0.4, "kind": "satellite"},
    {"name": "Carina III", "ra": 114.630, "dec": -57.900, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Aquarius II", "ra": 338.481, "dec": -9.327, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Sagittarius II", "ra": 298.169, "dec": -22.068, "radius_deg": 0.2, "kind": "satellite"},
    {"name": "Palomar 5", "ra": 229.022, "dec": -0.112, "radius_deg": 0.4, "kind": "globular"},
    {"name": "Palomar 13", "ra": 346.685, "dec": 12.772, "radius_deg": 0.2, "kind": "globular"},
    {"name": "NGC 5466", "ra": 211.364, "dec": 28.534, "radius_deg": 0.3, "kind": "globular"},
    {"name": "M3", "ra": 205.548, "dec": 28.377, "radius_deg": 0.4, "kind": "globular"},
    {"name": "M5", "ra": 229.638, "dec": 2.081, "radius_deg": 0.4, "kind": "globular"},
    {"name": "M13", "ra": 250.422, "dec": 36.460, "radius_deg": 0.4, "kind": "globular"},
    {"name": "M15", "ra": 322.493, "dec": 12.167, "radius_deg": 0.4, "kind": "globular"},
    {"name": "M53", "ra": 198.230, "dec": 18.168, "radius_deg": 0.3, "kind": "globular"},
    {"name": "M54", "ra": 283.764, "dec": -30.480, "radius_deg": 0.3, "kind": "globular"},
    {"name": "M92", "ra": 259.281, "dec": 43.136, "radius_deg": 0.3, "kind": "globular"},
    {"name": "47 Tucanae", "ra": 6.024, "dec": -72.081, "radius_deg": 0.7, "kind": "globular"},
    {"name": "Omega Centauri", "ra": 201.697, "dec": -47.480, "radius_deg": 0.8, "kind": "globular"},
    {"name": "NGC 2419", "ra": 114.535, "dec": 38.882, "radius_deg": 0.2, "kind": "globular"},
    {"name": "NGC 288", "ra": 13.188, "dec": -26.583, "radius_deg": 0.3, "kind": "globular"},
    {"name": "NGC 1851", "ra": 78.528, "dec": -40.047, "radius_deg": 0.3, "kind": "globular"},
    {"name": "M79", "ra": 81.044, "dec": -24.524, "radius_deg": 0.3, "kind": "globular"},
]


def match_known_mw_object(ra: float, dec: float) -> Optional[Dict[str, Any]]:
    """The known MW satellite/globular whose veto radius contains (ra, dec), or
    None. Returns a copy with 'separation_deg' added; nearest match wins."""
    import math as _math
    best: Optional[Dict[str, Any]] = None
    best_sep = float("inf")
    cosd = max(_math.cos(_math.radians(float(dec))), 1e-6)
    for entry in KNOWN_MW_OBJECTS:
        dra = (float(ra) - entry["ra"]) * cosd
        # Handle RA wrap-around for objects near 0/360.
        if dra > 180 * cosd:
            dra -= 360 * cosd
        elif dra < -180 * cosd:
            dra += 360 * cosd
        sep = _math.hypot(dra, float(dec) - entry["dec"])
        if sep <= entry["radius_deg"] and sep < best_sep:
            best, best_sep = dict(entry), sep
    if best is not None:
        best["separation_deg"] = round(best_sep, 4)
    return best


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


DEFAULT_TAP_SCHEMA_CACHE_DIR = Path("cache") / "datalab_tap_schema"
_TAP_SCHEMA_CACHE_KEY = "tap_schema_v2"
# Minimum spacing between refresh ATTEMPTS, so a dead service is not re-polled
# on every SQL call once the cached payload has gone stale.
_MIN_REFRESH_RETRY_SECONDS = 600.0
# Tables fetched per tap_schema query — small enough that the columns query
# stays well within the ~60s sync window even for 100-200-column breadth tables.
_SCHEMA_FETCH_CHUNK = 5

_SCHEMA_REFRESH_LOCK = threading.Lock()
_SCHEMA_REFRESH_STATE: Dict[str, Any] = {"in_flight": False, "last_attempt": 0.0}


def _reset_refresh_state_for_tests() -> None:
    with _SCHEMA_REFRESH_LOCK:
        _SCHEMA_REFRESH_STATE["in_flight"] = False
        _SCHEMA_REFRESH_STATE["last_attempt"] = 0.0


def registered_qualified_tables() -> List[str]:
    """Every curated catalog.table, qualified."""
    return sorted(
        f"{catalog}.{table}"
        for catalog, entry in DATALAB_CATALOGS.items()
        for table in entry.get("tables", {})
    )


# Extra Data Lab tables reachable through the live-TAP-schema fallback
# (describe_table -> live_table_entry). Each was verified live (2026-07-13) to
# exist with an indexed q3c ra/dec pair and to answer a governed q3c cone query;
# they are kept OUT of the hand-curated DATALAB_CATALOGS on purpose — their
# column lists are large (100-200+ cols) and are better read from the live
# schema than hardcoded. Seeding them into the default refresh scope makes them
# queryable and describable out of the box: survey breadth as data, not code.
# footprint/citation are the canonical survey references (columns still come
# from the live schema, never guessed).
EXPANSION_CATALOGS: List[Dict[str, Any]] = [
    {
        "catalog": "delve_dr2", "table": "objects",
        "footprint": "DECam Local Volume Exploration Survey DR2 — ~21,000 deg² of the "
                     "high-Galactic-latitude southern sky and Magellanic periphery (grizY).",
        "citation": {
            "text": "DELVE Data Release 2 (Drlica-Wagner et al. 2022)",
            "doi": "10.3847/1538-4365/ac78eb",
            "url": "https://datalab.noirlab.edu/delve/",
        },
    },
    {
        "catalog": "catwise2020", "table": "main",
        "footprint": "CatWISE2020 — all-sky W1+W2 co-adds from WISE/NEOWISE (2010–2018).",
        "citation": {
            "text": "CatWISE2020 (Marocco et al. 2021)",
            "doi": "10.3847/1538-4365/abd805",
            "url": "https://catwise.github.io/",
        },
    },
    {
        "catalog": "ls_dr10", "table": "tractor",
        "footprint": "DESI Legacy Imaging Surveys DR10 — ~20,000 deg² of grizW1–W4 tractor "
                     "photometry (DECam + BASS/MzLS).",
        "citation": {
            "text": "DESI Legacy Imaging Surveys DR10 (Dey et al. 2019)",
            "doi": "10.3847/1538-3881/ab089d",
            "url": "https://www.legacysurvey.org/dr10/",
        },
    },
]

_EXPANSION_BY_QUALIFIED: Dict[str, Dict[str, Any]] = {
    f"{entry['catalog']}.{entry['table']}": entry for entry in EXPANSION_CATALOGS
}


def expansion_qualified_tables() -> List[str]:
    """The verified live-fallback tables, qualified."""
    return [f"{entry['catalog']}.{entry['table']}" for entry in EXPANSION_CATALOGS]


def default_refresh_tables() -> List[str]:
    """Default TAP-schema fetch scope: every curated table plus the verified
    expansion set, so breadth catalogs are cached (and thus queryable through
    the governor) after the nightly refresh. Order-stable and de-duplicated."""
    ordered: Dict[str, None] = {}
    for name in registered_qualified_tables() + expansion_qualified_tables():
        ordered.setdefault(name, None)
    return list(ordered)


def tap_schema_ttl_seconds() -> int:
    return int(os.getenv("DATALAB_TAP_SCHEMA_TTL_SECONDS", "86400"))


def refresh_tap_schema(
    client: Any,
    *,
    cache_dir: Path | str | None = None,
    tables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fetch and cache the live TAP_SCHEMA rows for the registered tables.

    Live-verified 2026-07-12: Data Lab's ``tap_schema.columns`` has NO
    ``schema_name`` column (its ``table_name`` is fully qualified), so both
    queries key on qualified ``table_name``. The fetch is scoped to the
    registered tables (plus any extra ``tables``) rather than the whole
    tap_schema — the full columns relation is hundreds of thousands of rows.

    The normalized payload is what the SQL governor grounds column names on:
    ``{"tables": {qualified: {...}}, "columns": {qualified: {col: {"datatype", "indexed"}}},
    "refreshed_at": epoch}``.
    """

    wanted = list(tables) if tables else default_refresh_tables()

    tables_map: Dict[str, Dict[str, Any]] = {}
    columns_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    # Fetch in small chunks: a single tap_schema.columns query over the full
    # default scope (curated + the 100-200-column breadth tables) exceeds the
    # ~60s sync window and times out (live-verified 2026-07-13: 17 tables in one
    # IN-list timed out, 4-5 per query returns in seconds). Per-chunk merge keeps
    # the payload identical to a single query.
    for start in range(0, len(wanted), _SCHEMA_FETCH_CHUNK):
        chunk = wanted[start:start + _SCHEMA_FETCH_CHUNK]
        # Identifier safety: every name is registry-supplied or caller code
        # (never end-user text); _normalize_qualified rejects anything but [\w.].
        in_list = ", ".join("'" + _normalize_qualified(name) + "'" for name in chunk)
        tables_df = client.query(
            sql=(
                "SELECT table_name, table_type, description FROM tap_schema.tables "
                f"WHERE table_name IN ({in_list})"
            ),
            fmt="pandas",
        ).dataframe
        columns_df = client.query(
            sql=(
                "SELECT table_name, column_name, datatype, unit, indexed FROM tap_schema.columns "
                f"WHERE table_name IN ({in_list})"
            ),
            fmt="pandas",
        ).dataframe

        for row in tables_df.to_dict("records"):
            name = str(row.get("table_name") or "").strip().lower()
            if name:
                tables_map[name] = {
                    "table_type": row.get("table_type"),
                    "description": row.get("description"),
                }
        for row in columns_df.to_dict("records"):
            name = str(row.get("table_name") or "").strip().lower()
            column = str(row.get("column_name") or "").strip().lower()
            if not name or not column:
                continue
            try:
                indexed = bool(int(row.get("indexed") or 0))
            except (TypeError, ValueError):
                indexed = False
            columns_map.setdefault(name, {})[column] = {
                "datatype": row.get("datatype"),
                "unit": row.get("unit"),
                "indexed": indexed,
            }

    # MERGE with whatever is already cached (per-table replace): a scoped
    # refresh (tables=[...]) that seeded a non-curated table must survive the
    # next nightly registered-tables refresh, and vice versa. (guard CX-03)
    existing = cached_tap_schema(cache_dir=cache_dir) or {}
    payload = {
        "tables": {**dict(existing.get("tables") or {}), **tables_map},
        "columns": {**dict(existing.get("columns") or {}), **columns_map},
        "refreshed_at": time.time(),
    }
    _cache_schema(payload, cache_dir=cache_dir)
    return payload


def cached_tap_schema(*, cache_dir: Path | str | None = None) -> Optional[Dict[str, Any]]:
    cache = _open_cache(cache_dir)
    if cache is None:
        return None
    try:
        payload = cache.get(_TAP_SCHEMA_CACHE_KEY)
        return payload if isinstance(payload, Mapping) else None
    finally:
        try:
            cache.close()
        except Exception:
            pass


def ensure_tap_schema_fresh(
    client_factory: Any,
    *,
    cache_dir: Path | str | None = None,
    background: bool = True,
) -> str:
    """Nightly-cache driver: refresh the live TAP schema when the cache is stale.

    Called on the SQL execution path, so it must never block or raise: a stale
    (or missing) cache triggers ONE daemon-thread refresh at a time, retried at
    most every ``_MIN_REFRESH_RETRY_SECONDS``. ``client_factory`` may be a
    DatalabClient instance or a zero-arg callable returning one.
    Returns one of ``fresh|refreshing|scheduled|refreshed|skipped`` (for tests/logs).
    """

    if os.getenv("DATALAB_TAP_SCHEMA_AUTOREFRESH", "1").strip().lower() in {"0", "false", "off"}:
        return "disabled"
    now = time.time()
    payload = cached_tap_schema(cache_dir=cache_dir)
    if payload and (now - float(payload.get("refreshed_at") or 0.0)) < tap_schema_ttl_seconds():
        return "fresh"
    with _SCHEMA_REFRESH_LOCK:
        if _SCHEMA_REFRESH_STATE["in_flight"]:
            return "refreshing"
        if (now - float(_SCHEMA_REFRESH_STATE["last_attempt"])) < _MIN_REFRESH_RETRY_SECONDS:
            return "skipped"
        _SCHEMA_REFRESH_STATE["in_flight"] = True
        _SCHEMA_REFRESH_STATE["last_attempt"] = now

    def _do_refresh() -> None:
        try:
            client = client_factory() if callable(client_factory) else client_factory
            refresh_tap_schema(client, cache_dir=cache_dir)
        except Exception:
            logging.getLogger(__name__).warning(
                "Data Lab TAP schema refresh failed; the SQL governor keeps using "
                "the curated registry (and any previous cache).", exc_info=True,
            )
        finally:
            with _SCHEMA_REFRESH_LOCK:
                _SCHEMA_REFRESH_STATE["in_flight"] = False

    if background:
        threading.Thread(target=_do_refresh, name="datalab-tap-schema-refresh", daemon=True).start()
        return "scheduled"
    _do_refresh()
    return "refreshed"


def live_columns(catalog: str, table: str, *, cache_dir: Path | str | None = None) -> Optional[Dict[str, Dict[str, Any]]]:
    """Column map for catalog.table from the cached live TAP schema, or None."""
    payload = cached_tap_schema(cache_dir=cache_dir)
    if not payload:
        return None
    qualified = f"{_normalize_identifier(catalog)}.{_normalize_identifier(table)}"
    columns = (payload.get("columns") or {}).get(qualified)
    return dict(columns) if isinstance(columns, Mapping) else None


def known_columns(catalog: str, table: str, *, cache_dir: Path | str | None = None) -> Optional[Dict[str, Any]]:
    """Ground truth for the SQL governor's column check.

    Returns ``{"columns": set[str], "authoritative": bool}`` — the union of the
    curated registry columns (fast path, always available offline) and the
    cached live tap_schema columns. ``authoritative`` is True only when the live
    schema for this table is cached: the curated lists are deliberate SUBSETS of
    the real tables, so without live data an unknown column is merely
    unverifiable, not provably wrong. Returns None when the table is known to
    neither source.
    """

    curated: set[str] = set()
    try:
        catalog_key = _normalize_identifier(catalog)
        table_key = _normalize_identifier(table)
    except ValueError:
        return None
    table_entry = (DATALAB_CATALOGS.get(catalog_key, {}).get("tables", {}) or {}).get(table_key)
    if table_entry:
        curated = {str(col).lower() for col in table_entry.get("columns", [])}
        entry = DATALAB_CATALOGS[catalog_key]
        curated.add(str(table_entry.get("ra_column", "ra")).lower())
        curated.add(str(table_entry.get("dec_column", "dec")).lower())
        curated.update(str(h.get("name", "")).lower() for h in entry.get("healpix_columns", []))
        curated.discard("")
    live = live_columns(catalog_key, table_key, cache_dir=cache_dir)
    if not curated and live is None:
        return None
    combined = set(curated)
    if live:
        combined.update(live.keys())
    return {"columns": combined, "authoritative": live is not None}


# RA/Dec column-name pairs seen across Data Lab catalogs, in preference order —
# used to auto-detect coordinates when building a live (non-curated) table entry.
_RA_DEC_CANDIDATES: List[tuple[str, str]] = [
    ("ra", "dec"),
    ("ra2000", "dec2000"),
    ("mean_fiber_ra", "mean_fiber_dec"),
    ("raj2000", "dej2000"),
    ("alpha_j2000", "delta_j2000"),
]

_HEALPIX_NAME_RE = re.compile(r"^(?:(?:hpix|ring|nest)_?(\d+))$")


def live_table_entry(catalog: str, table: str, *, cache_dir: Path | str | None = None) -> Optional[Dict[str, Any]]:
    """Build a describe_table-shaped entry for a NON-curated table from the
    cached live TAP schema, auto-detecting the RA/Dec pair and HEALPix columns.

    This is the survey-breadth path: ``refresh_tap_schema(client, tables=[...])``
    on any Data Lab table makes it immediately queryable through the governor
    (q3c-bounded, row-capped) without a registry code change. Conservative
    defaults: region_strategy='q3c' and aggregate_safe=False.
    """

    columns = live_columns(catalog, table, cache_dir=cache_dir)
    if not columns:
        return None
    names = list(columns.keys())
    ra_column, dec_column = None, None
    for ra_cand, dec_cand in _RA_DEC_CANDIDATES:
        if ra_cand in columns and dec_cand in columns:
            ra_column, dec_column = ra_cand, dec_cand
            break
    if ra_column is None:
        return None
    healpix = []
    for name in names:
        match = _HEALPIX_NAME_RE.match(name)
        if match:
            nside = int(match.group(1))
            scheme = "RING" if name.startswith("ring") else "NEST"
            if nside > 0 and (nside & (nside - 1)) == 0:
                healpix.append({"name": name, "nside": nside, "scheme": scheme})
    catalog_key = _normalize_identifier(catalog)
    table_key = _normalize_identifier(table)
    # Verified expansion catalogs carry a curated footprint + citation overlay
    # (the columns still come from the live schema, never hardcoded).
    overlay = _EXPANSION_BY_QUALIFIED.get(f"{catalog_key}.{table_key}", {})
    return {
        "catalog": catalog_key,
        "table": table_key,
        "qualified_name": f"{catalog_key}.{table_key}",
        "columns": names,
        "ra_column": ra_column,
        "dec_column": dec_column,
        "region_strategy": "q3c",
        "healpix_columns": healpix,
        "morphology": {},
        "bitmasks": {},
        "aggregate_safe": False,
        "footprint": overlay.get("footprint"),
        "citation": dict(overlay.get("citation", {})),
        "source": "live_tap_schema",
    }


def _cache_schema(payload: Mapping[str, Any], *, cache_dir: Path | str) -> None:
    cache = _open_cache(cache_dir)
    if cache is None:
        return
    try:
        # No expire: a stale payload is still far better ground truth than none,
        # and ensure_tap_schema_fresh() handles freshness by refreshed_at + TTL.
        cache.set(_TAP_SCHEMA_CACHE_KEY, dict(payload))
    finally:
        try:
            cache.close()
        except Exception:
            pass


def _normalize_qualified(value: str) -> str:
    text = str(value or "").strip().lower()
    parts = text.split(".")
    if len(parts) != 2:
        raise ValueError(f"Expected a qualified catalog.table name: {value!r}")
    return f"{_normalize_identifier(parts[0])}.{_normalize_identifier(parts[1])}"


def _default_cache_dir() -> Path:
    """Production default is cache/datalab_tap_schema; DATALAB_TAP_SCHEMA_CACHE_DIR
    overrides it (tests point this at an isolated temp dir so a developer's real
    cache never changes unit-test behavior)."""
    override = os.getenv("DATALAB_TAP_SCHEMA_CACHE_DIR", "").strip()
    return Path(override) if override else DEFAULT_TAP_SCHEMA_CACHE_DIR


def _open_cache(cache_dir: Path | str | None):
    try:
        from diskcache import Cache
    except Exception:
        return None
    try:
        return Cache(str(cache_dir if cache_dir is not None else _default_cache_dir()))
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


# ─────────────────────────────────────────────────────────────────────────────
# Archive-profile authoring layer (Feature 3, docs/plans/2026-07-astrofetch-
# provenance). Extends the registry with the fields DATALAB_CATALOGS lacks —
# column-level dtype/unit/role, table purpose/grain, structured pitfalls and
# golden tool invocations — WITHOUT touching the load-bearing "columns" lists.
# ``to_profile()`` assembles everything into a plain dict;
# ``services.archive_profiles.datalab`` validates it into an ArchiveProfile
# (schema converged in codex-bridge duel task-81b3b73-2145).
# ─────────────────────────────────────────────────────────────────────────────

DATALAB_TAP_ENDPOINT = "https://datalab.noirlab.edu/tap"

# Per-column authoring, keyed by qualified table. Tuple = (dtype, unit, role, desc).
#   unit: "mag"/"deg"/… = known unit (astropy-parseable); "1" = verified
#   dimensionless; None on a NUMERIC column = not_applicable (identifiers,
#   codes, bitmasks); ("?", "reason") = genuinely unknown, with the mandatory
#   note. None on a non-numeric column simply means no unit fields apply.
# Every column in a curated table's "columns" list MUST appear here and vice
# versa — to_profile() fails loudly on any mismatch, and the unit test holds
# the same line. Descriptions are canonical/common guidance, not exhaustive.
TABLE_PROFILE_INFO: Dict[str, Dict[str, Any]] = {
    "gaia_dr3.gaia_source": {
        "purpose": "All-sky space astrometry + broadband photometry + radial velocities (Gaia DR3).",
        "grain": "one row per Gaia source",
        "columns": {
            "source_id": ("integer", None, "identifier", "Unique Gaia DR3 source id (int64)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "parallax": ("float", "mas", "measurement", "Absolute parallax; can be negative, NaN when missing."),
            "parallax_over_error": ("float", "1", "quality", "Parallax S/N; e.g. > 5 for reliable distances (guard NaN)."),
            "pmra": ("float", "mas/yr", "measurement", "Proper motion in RA (mualpha* = mu_alpha cos delta)."),
            "pmdec": ("float", "mas/yr", "measurement", "Proper motion in Dec."),
            "pm": ("float", "mas/yr", "measurement", "Total proper motion."),
            "ruwe": ("float", "1", "quality", "Renormalised unit weight error; <~1.4 = well-behaved astrometric fit."),
            "ipd_frac_multi_peak": ("integer", "%", "quality", "Windows with multi-peak detections (blend indicator)."),
            "astrometric_sigma5d_max": ("float", "mas", "quality", "Worst-direction 5D astrometric uncertainty."),
            "phot_g_mean_mag": ("float", "mag", "measurement", "Mean G magnitude (Vega). Gaia bands are g|bp|rp."),
            "phot_bp_mean_mag": ("float", "mag", "measurement", "Mean BP magnitude (Vega)."),
            "phot_rp_mean_mag": ("float", "mag", "measurement", "Mean RP magnitude (Vega)."),
            "bp_rp": ("float", "mag", "measurement", "BP-RP colour."),
            "radial_velocity": ("float", "km/s", "measurement", "Spectroscopic RV (bright-star subset; NaN elsewhere)."),
        },
    },
    "nsc_dr2.object": {
        "purpose": "Mean-object photometry from ~35,000 deg2 of archival DECam/Bok/Mosaic imaging (NSC DR2).",
        "grain": "one row per unique object (mean over epochs)",
        "columns": {
            "id": ("string", None, "identifier", "NSC object id (STRING — keep quoted)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "gmag": ("float", "mag", "measurement", "Mean g magnitude (AB). NSC uses <band>mag, NOT mag_auto_<band>."),
            "rmag": ("float", "mag", "measurement", "Mean r magnitude (AB)."),
            "imag": ("float", "mag", "measurement", "Mean i magnitude (AB)."),
            "zmag": ("float", "mag", "measurement", "Mean z magnitude (AB)."),
            "gerr": ("float", "mag", "uncertainty", "g magnitude uncertainty."),
            "rerr": ("float", "mag", "uncertainty", "r magnitude uncertainty."),
            "class_star": ("float", "1", "quality", "Stellarity: ~1 star-like, ~0 extended; point-source cut > 0.5."),
            "fwhm": ("float", "arcsec", "measurement", "Mean source FWHM."),
            "ring256": ("integer", None, "healpix", "HEALPix RING nside=256 (coarse; use for wide-area density)."),
            "nest4096": ("integer", None, "healpix", "HEALPix NEST nside=4096 (fine)."),
        },
    },
    "des_dr1.main": {
        "purpose": "DES DR1 coadd photometry over ~5,000 deg2 of the southern high-latitude sky (grizY).",
        "grain": "one row per coadd object",
        "columns": {
            "coadd_object_id": ("integer", None, "identifier", "Unique DES coadd object id."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "mag_auto_g": ("float", "mag", "measurement", "AUTO g magnitude (AB). PER-BAND — there is no bare mag_auto."),
            "mag_auto_r": ("float", "mag", "measurement", "AUTO r magnitude (AB)."),
            "mag_auto_i": ("float", "mag", "measurement", "AUTO i magnitude (AB)."),
            "mag_auto_z": ("float", "mag", "measurement", "AUTO z magnitude (AB)."),
            "mag_auto_g_dered": ("float", "mag", "measurement", "Dereddened AUTO g magnitude (SFD)."),
            "mag_auto_r_dered": ("float", "mag", "measurement", "Dereddened AUTO r magnitude."),
            "mag_auto_i_dered": ("float", "mag", "measurement", "Dereddened AUTO i magnitude."),
            "mag_auto_z_dered": ("float", "mag", "measurement", "Dereddened AUTO z magnitude."),
            "magerr_auto_g": ("float", "mag", "uncertainty", "AUTO g magnitude uncertainty."),
            "magerr_auto_r": ("float", "mag", "uncertainty", "AUTO r magnitude uncertainty."),
            "magerr_auto_i": ("float", "mag", "uncertainty", "AUTO i magnitude uncertainty."),
            "magerr_auto_z": ("float", "mag", "uncertainty", "AUTO z magnitude uncertainty."),
            "flags_g": ("integer", None, "quality", "SExtractor flags in g; flags_<b> = 0 selects clean photometry."),
            "flags_r": ("integer", None, "quality", "SExtractor flags in r."),
            "flags_i": ("integer", None, "quality", "SExtractor flags in i."),
            "flags_z": ("integer", None, "quality", "SExtractor flags in z."),
            "spread_model_g": ("float", "1", "quality", "Star/galaxy separator in g; ~0 point source, > 0.005 galaxy."),
            "spread_model_r": ("float", "1", "quality", "Star/galaxy separator in r (canonical band for the cut)."),
            "spread_model_i": ("float", "1", "quality", "Star/galaxy separator in i."),
            "spread_model_z": ("float", "1", "quality", "Star/galaxy separator in z."),
            "spreaderr_model_r": ("float", "1", "uncertainty", "Uncertainty on spread_model_r."),
            "class_star_g": ("float", "1", "quality", "Stellarity in g (~1 star). PER-BAND — no bare class_star."),
            "class_star_r": ("float", "1", "quality", "Stellarity in r."),
            "class_star_i": ("float", "1", "quality", "Stellarity in i."),
            "class_star_z": ("float", "1", "quality", "Stellarity in z."),
        },
    },
    "smash_dr1.object": {
        "purpose": "SMASH DR1 mean-object photometry of targeted Magellanic-system DECam fields.",
        "grain": "one row per object (mean over epochs)",
        "columns": {
            "id": ("string", None, "identifier", "SMASH object id — a STRING like '169.429960'; never float-coerce."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "fieldid": ("integer", None, "identifier", "SMASH field number; indexed — bound whole-field queries with fieldid = N."),
            "depthflag": ("integer", None, "quality", "Exposure-depth flag (>1 = deeper imaging)."),
            "sharp": ("float", "1", "quality", "DAOPHOT sharpness; |sharp| < 0.5 ~ stellar."),
            "chi": ("float", "1", "quality", "DAOPHOT chi of the PSF fit."),
            "gmag": ("float", "mag", "measurement", "Mean g magnitude (AB)."),
            "rmag": ("float", "mag", "measurement", "Mean r magnitude (AB)."),
            "imag": ("float", "mag", "measurement", "Mean i magnitude (AB)."),
            "zmag": ("float", "mag", "measurement", "Mean z magnitude (AB)."),
            "gerr": ("float", "mag", "uncertainty", "g magnitude uncertainty."),
            "rerr": ("float", "mag", "uncertainty", "r magnitude uncertainty."),
        },
    },
    "smash_dr1.source": {
        "purpose": "SMASH DR1 multi-epoch (per-exposure) photometry for variability/light curves.",
        "grain": "one row per detection (object x epoch x band)",
        "columns": {
            "id": ("string", None, "identifier", "SMASH object id (STRING; joins to object.id)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "mjd": ("float", "d", "time", "Modified Julian Date of the epoch."),
            "filter": ("string", None, "category", "Band letter (u/g/r/i/z)."),
            "cmag": ("float", "mag", "measurement", "Calibrated per-epoch magnitude; error column is cerr."),
            "cerr": ("float", "mag", "uncertainty", "Per-epoch magnitude uncertainty."),
            "chi": ("float", "1", "quality", "PSF-fit chi."),
            "sharp": ("float", "1", "quality", "Sharpness; |sharp| < 0.5 ~ stellar."),
        },
    },
    "smash_dr2.object": {
        "purpose": "SMASH DR2 mean-object photometry of targeted Magellanic-system DECam fields.",
        "grain": "one row per object (mean over epochs)",
        "columns": {
            "id": ("string", None, "identifier", "SMASH object id — a STRING like '169.429960'; never float-coerce."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "fieldid": ("integer", None, "identifier", "SMASH field number; indexed — bound whole-field queries with fieldid = N."),
            "depthflag": ("integer", None, "quality", "Exposure-depth flag (>1 = deeper imaging)."),
            "sharp": ("float", "1", "quality", "DAOPHOT sharpness; |sharp| < 0.5 ~ stellar."),
            "chi": ("float", "1", "quality", "DAOPHOT chi of the PSF fit."),
            "gmag": ("float", "mag", "measurement", "Mean g magnitude (AB)."),
            "rmag": ("float", "mag", "measurement", "Mean r magnitude (AB)."),
            "imag": ("float", "mag", "measurement", "Mean i magnitude (AB)."),
            "zmag": ("float", "mag", "measurement", "Mean z magnitude (AB)."),
            "gerr": ("float", "mag", "uncertainty", "g magnitude uncertainty."),
            "rerr": ("float", "mag", "uncertainty", "r magnitude uncertainty."),
        },
    },
    "smash_dr2.source": {
        "purpose": "SMASH DR2 multi-epoch (per-exposure) photometry for variability/light curves.",
        "grain": "one row per detection (object x epoch x band)",
        "columns": {
            "id": ("string", None, "identifier", "SMASH object id (STRING; joins to object.id)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "mjd": ("float", "d", "time", "Modified Julian Date of the epoch."),
            "filter": ("string", None, "category", "Band letter (u/g/r/i/z)."),
            "cmag": ("float", "mag", "measurement", "Calibrated per-epoch magnitude; error column is cerr."),
            "cerr": ("float", "mag", "uncertainty", "Per-epoch magnitude uncertainty."),
            "chi": ("float", "1", "quality", "PSF-fit chi."),
            "sharp": ("float", "1", "quality", "Sharpness; |sharp| < 0.5 ~ stellar."),
        },
    },
    "delve_dr3.coadd_objects": {
        "purpose": "DELVE DR3 coadd photometry over ~21,000 deg2 of the southern sky (griz).",
        "grain": "one row per coadd object",
        "columns": {
            "quick_object_id": ("integer", None, "identifier", "Unique DELVE coadd object id."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "mag_auto_g": ("float", "mag", "measurement", "AUTO g magnitude (AB); per-band naming like DES."),
            "magerr_auto_g": ("float", "mag", "uncertainty", "AUTO g magnitude uncertainty."),
            "mag_auto_r": ("float", "mag", "measurement", "AUTO r magnitude (AB)."),
            "magerr_auto_r": ("float", "mag", "uncertainty", "AUTO r magnitude uncertainty."),
            "mag_auto_i": ("float", "mag", "measurement", "AUTO i magnitude (AB)."),
            "ext_coadd": ("integer", None, "category", "Star/galaxy class: 0 hi-conf star, 1 candidate star, 2 mostly galaxy, 3 hi-conf galaxy, -9 no data."),
            "hpix_1024": ("integer", None, "healpix", "HEALPix NEST nside=1024."),
            "hpix_4096": ("integer", None, "healpix", "HEALPix NEST nside=4096."),
            "hpix_16384": ("integer", None, "healpix", "HEALPix NEST nside=16384."),
        },
    },
    "desi_dr1.zpix": {
        "purpose": "DESI DR1 healpix-coadded spectroscopic redshift catalog.",
        "grain": "one row per targetid per survey/program coadd",
        "columns": {
            "targetid": ("integer", None, "identifier", "Unique DESI target id (int64); indexed equality bound."),
            "mean_fiber_ra": ("float", "deg", "ra", "Mean fiber RA — the coordinate columns here are NOT ra/dec."),
            "mean_fiber_dec": ("float", "deg", "dec", "Mean fiber Dec."),
            "z": ("float", "1", "measurement", "Spectroscopic redshift."),
            "zerr": ("float", "1", "uncertainty", "Redshift uncertainty."),
            "zwarn": ("integer", None, "quality", "Redshift warning bitmask; zwarn = 0 selects reliable redshifts."),
            "spectype": ("string", None, "category", "Spectral class: GALAXY | QSO | STAR."),
            "desi_target": ("integer", None, "category", "Targeting bitmask (LRG bit 0, ELG bit 1, QSO bit 2, BGS_ANY 60, MWS_ANY 61)."),
            "survey": ("string", None, "category", "Survey phase; survey = 'main' for the main survey."),
            "main_primary": ("boolean", None, "quality", "True for the primary main-survey spectrum of the target."),
        },
    },
    "sdss_dr17.specobj": {
        "purpose": "SDSS DR17 optical spectroscopic catalog (redshifts + classes).",
        "grain": "one row per spectrum (specobjid)",
        "columns": {
            "specobjid": ("integer", None, "identifier", "Unique spectrum id (int64)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "z": ("float", "1", "measurement", "Spectroscopic redshift."),
            "zerr": ("float", "1", "uncertainty", "Redshift uncertainty."),
            "zwarning": ("integer", None, "quality", "Quality bitmask — SDSS uses zwarning (NOT zwarn); 0 = clean."),
            "class": ("string", None, "category", "Spectral class: GALAXY | QSO | STAR."),
            "subclass": ("string", None, "category", "Finer spectral subclass."),
            "plate": ("integer", None, "identifier", "Plate number."),
            "mjd": ("integer", "d", "time", "MJD of the observation."),
            "fiberid": ("integer", None, "identifier", "Fiber number on the plate."),
        },
    },
    "ls_dr9.tractor": {
        "purpose": "Legacy Surveys DR9 Tractor model photometry (~19,700 deg2, g/r/z + forced unWISE W1/W2).",
        "grain": "one row per Tractor model source",
        "columns": {
            "release": ("integer", None, "provenance", "LS release number of the brick reduction."),
            "brickid": ("integer", None, "identifier", "Brick id; (brickid, objid) is unique."),
            "objid": ("integer", None, "identifier", "Object id within the brick."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "type": ("string", None, "category", "Tractor morphological type; PSF = point source, others extended."),
            "dered_mag_g": ("float", "mag", "measurement", "Dereddened model g magnitude (AB). LS DR9 has g/r/z — NO i band."),
            "dered_mag_r": ("float", "mag", "measurement", "Dereddened model r magnitude (AB)."),
            "dered_mag_z": ("float", "mag", "measurement", "Dereddened model z magnitude (AB)."),
            "dered_mag_w1": ("float", "mag", "measurement", "Forced unWISE W1 magnitude (AB); W3/W4 are NOT available."),
            "dered_mag_w2": ("float", "mag", "measurement", "Forced unWISE W2 magnitude (AB)."),
            "snr_g": ("float", "1", "quality", "Flux S/N in g."),
            "snr_r": ("float", "1", "quality", "Flux S/N in r."),
            "snr_z": ("float", "1", "quality", "Flux S/N in z."),
            "snr_w1": ("float", "1", "quality", "Flux S/N in W1."),
            "snr_w2": ("float", "1", "quality", "Flux S/N in W2."),
        },
    },
    "vhs_dr5.vhs_cat_v3": {
        "purpose": "VISTA Hemisphere Survey DR5 near-IR JHKs photometry of the southern sky.",
        "grain": "one row per merged source",
        "columns": {
            "sourceid": ("integer", None, "identifier", "Unique VSA source id."),
            "ra2000": ("float", "deg", "ra", "ICRS right ascension — coordinates here are ra2000/dec2000, not ra/dec."),
            "dec2000": ("float", "deg", "dec", "ICRS declination."),
            "japermag3": ("float", "mag", "measurement", "J aperture-3 (2 arcsec) magnitude (Vega)."),
            "japermag3err": ("float", "mag", "uncertainty", "J aperture-3 magnitude uncertainty."),
            "hapermag3": ("float", "mag", "measurement", "H aperture-3 magnitude (Vega)."),
            "hapermag3err": ("float", "mag", "uncertainty", "H aperture-3 magnitude uncertainty."),
            "ksapermag3": ("float", "mag", "measurement", "Ks aperture-3 magnitude (Vega)."),
            "ksapermag3err": ("float", "mag", "uncertainty", "Ks aperture-3 magnitude uncertainty."),
            "mergedclass": ("integer", None, "category", "VSA morphology class: -1 star, -2 probable star, 1 galaxy."),
            "pstar": ("float", "1", "quality", "Probability the source is point-like (0-1)."),
            "ring256": ("integer", None, "healpix", "HEALPix RING nside=256."),
            "nest4096": ("integer", None, "healpix", "HEALPix NEST nside=4096."),
        },
    },
    "unwise_dr1.object": {
        "purpose": "unWISE coadd catalog: deep all-sky WISE W1/W2 photometry (~2 billion sources).",
        "grain": "one row per coadd source",
        "columns": {
            "unwise_objid": ("string", None, "identifier", "unWISE object id (STRING)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "mag_w1_vg": ("float", "mag", "measurement", "W1 magnitude (VEGA, precomputed; unWISE photometry is Vega)."),
            "mag_w2_vg": ("float", "mag", "measurement", "W2 magnitude (Vega)."),
            "w1_w2_vg": ("float", "mag", "measurement", "W1-W2 colour (Vega)."),
            "flux_w1": ("float", "nmgy", "measurement", "W1 flux in Vega nanomaggies."),
            "flux_w2": ("float", "nmgy", "measurement", "W2 flux in Vega nanomaggies."),
            "dflux_w1": ("float", "nmgy", "uncertainty", "W1 flux uncertainty."),
            "dflux_w2": ("float", "nmgy", "uncertainty", "W2 flux uncertainty."),
            "qf_w1": ("float", "1", "quality", "Fraction of W1 frames unaffected by bad pixels (0-1)."),
            "qf_w2": ("float", "1", "quality", "Fraction of W2 frames unaffected by bad pixels (0-1)."),
            "rchi2_w1": ("float", "1", "quality", "Reduced chi2 of the W1 PSF fit."),
            "rchi2_w2": ("float", "1", "quality", "Reduced chi2 of the W2 PSF fit."),
            "fracflux_w1": ("float", "1", "quality", "Fraction of flux from this source (blending diagnostic; W1)."),
            "fracflux_w2": ("float", "1", "quality", "Fraction of flux from this source (W2)."),
            "fwhm_w1": ("float", "pix", "measurement", "PSF FWHM at source (unWISE coadd pixels, 2.75 arcsec/pix)."),
            "fwhm_w2": ("float", "pix", "measurement", "PSF FWHM at source (W2)."),
            "spread_model_w1": ("float", "1", "quality", "crowdsource spread_model in W1: ~0 point source, positive extended."),
            "spread_model_w2": ("float", "1", "quality", "spread_model in W2."),
            "flags_unwise_w1": ("integer", None, "quality", "unWISE coadd flags bitmask (W1); 0 = clean."),
            "flags_unwise_w2": ("integer", None, "quality", "unWISE coadd flags bitmask (W2)."),
            "flags_info_w1": ("integer", None, "quality", "Informational flags bitmask (W1)."),
            "flags_info_w2": ("integer", None, "quality", "Informational flags bitmask (W2)."),
            "coadd_id": ("string", None, "provenance", "Coadd tile id."),
            "ring256": ("integer", None, "healpix", "HEALPix RING nside=256."),
            "nest4096": ("integer", None, "healpix", "HEALPix NEST nside=4096."),
        },
    },
    "twomass.psc": {
        "purpose": "2MASS Point Source Catalog: all-sky near-IR J/H/Ks photometry (~471M sources).",
        "grain": "one row per point source",
        "columns": {
            "designation": ("string", None, "identifier", "2MASS designation (STRING)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "j_m": ("float", "mag", "measurement", "J magnitude (Vega). Band prefix j/h/k; Ks column is k_m."),
            "h_m": ("float", "mag", "measurement", "H magnitude (Vega)."),
            "k_m": ("float", "mag", "measurement", "Ks magnitude (Vega) — named k_m, not ks_m."),
            "j_cmsig": ("float", "mag", "uncertainty", "J corrected photometric uncertainty."),
            "h_cmsig": ("float", "mag", "uncertainty", "H corrected photometric uncertainty."),
            "k_cmsig": ("float", "mag", "uncertainty", "Ks corrected photometric uncertainty."),
            "j_msigcom": ("float", "mag", "uncertainty", "J total (combined) uncertainty."),
            "h_msigcom": ("float", "mag", "uncertainty", "H total uncertainty."),
            "k_msigcom": ("float", "mag", "uncertainty", "Ks total uncertainty."),
            "j_snr": ("float", "1", "quality", "J-band S/N."),
            "h_snr": ("float", "1", "quality", "H-band S/N."),
            "k_snr": ("float", "1", "quality", "Ks-band S/N."),
            "ph_qual": ("string", None, "quality", "3-char photometric quality flags (JHKs; 'AAA' best)."),
            "cc_flg": ("string", None, "quality", "Contamination/confusion flags ('000' clean)."),
            "rd_flg": ("string", None, "quality", "Read flags (detection origin per band)."),
            "bl_flg": ("string", None, "quality", "Blend flags (fit components per band)."),
            "gal_contam": ("integer", None, "quality", "Extended-source contamination: 0 clean, 1/2 contaminated by XSC source."),
            "mp_flg": ("integer", None, "quality", "Minor-planet association flag."),
            "use_src": ("integer", None, "quality", "Use-source flag (1 = primary photometry)."),
            "dup_src": ("integer", None, "quality", "Duplicate-source flag."),
            "prox": ("float", "arcsec", "quality", "Distance to the nearest PSC neighbour."),
            "jdate": ("float", "d", "time", "Julian Date of the observation."),
            "ring256": ("integer", None, "healpix", "HEALPix RING nside=256."),
            "nest256": ("integer", None, "healpix", "HEALPix NEST nside=256 — NOT nest4096 here."),
        },
    },
    "allwise.source": {
        "purpose": "AllWISE all-sky mid-IR W1-W4 source catalog (~748M sources).",
        "grain": "one row per source",
        "columns": {
            "cntr": ("integer", None, "identifier", "Unique AllWISE counter id; indexed."),
            "source_id": ("string", None, "identifier", "Source id (coadd id + sequence; STRING)."),
            "designation": ("string", None, "identifier", "AllWISE designation (STRING; not indexed)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "w1mpro": ("float", "mag", "measurement", "W1 (3.4 um) profile-fit magnitude (Vega)."),
            "w2mpro": ("float", "mag", "measurement", "W2 (4.6 um) profile-fit magnitude (Vega)."),
            "w3mpro": ("float", "mag", "measurement", "W3 (12 um) profile-fit magnitude (Vega)."),
            "w4mpro": ("float", "mag", "measurement", "W4 (22 um) profile-fit magnitude (Vega)."),
            "w1sigmpro": ("float", "mag", "uncertainty", "W1 magnitude uncertainty."),
            "w2sigmpro": ("float", "mag", "uncertainty", "W2 magnitude uncertainty."),
            "w3sigmpro": ("float", "mag", "uncertainty", "W3 magnitude uncertainty."),
            "w4sigmpro": ("float", "mag", "uncertainty", "W4 magnitude uncertainty."),
            "w1snr": ("float", "1", "quality", "W1 S/N."),
            "w2snr": ("float", "1", "quality", "W2 S/N."),
            "w3snr": ("float", "1", "quality", "W3 S/N."),
            "w4snr": ("float", "1", "quality", "W4 S/N."),
            "w1rchi2": ("float", "1", "quality", "Reduced chi2 of W1 profile fit."),
            "w2rchi2": ("float", "1", "quality", "Reduced chi2 of W2 profile fit."),
            "cc_flags": ("string", None, "quality", "Contamination/confusion flags per band ('0000' clean)."),
            "ext_flg": ("integer", None, "quality", "0 = PSF-consistent point source; >0 extended/XSC-associated."),
            "var_flg": ("string", None, "quality", "Variability flag per band (0-9; higher = more likely variable)."),
            "ph_qual": ("string", None, "quality", "Photometric quality per band (A best)."),
            "moon_lev": ("string", None, "quality", "Moon contamination level per band."),
            "pmra": ("integer", "mas/yr", "measurement", "AllWISE apparent motion in RA — includes parallax; NOT a Gaia-style proper motion."),
            "pmdec": ("integer", "mas/yr", "measurement", "AllWISE apparent motion in Dec (same caveat as pmra)."),
            "sigpmra": ("integer", "mas/yr", "uncertainty", "Uncertainty on pmra."),
            "sigpmdec": ("integer", "mas/yr", "uncertainty", "Uncertainty on pmdec."),
            "j_m_2mass": ("float", "mag", "measurement", "Associated 2MASS J magnitude (Vega)."),
            "h_m_2mass": ("float", "mag", "measurement", "Associated 2MASS H magnitude (Vega)."),
            "k_m_2mass": ("float", "mag", "measurement", "Associated 2MASS Ks magnitude (Vega)."),
            "ring256": ("integer", None, "healpix", "HEALPix RING nside=256."),
            "nest4096": ("integer", None, "healpix", "HEALPix NEST nside=4096."),
        },
    },
    "splus_dr4.dual": {
        "purpose": "S-PLUS DR4 12-band (5 broad + 7 narrow) southern photometry, dual-mode detection.",
        "grain": "one row per dual-mode detection",
        "columns": {
            "id": ("string", None, "identifier", "S-PLUS object id (STRING)."),
            "ra": ("float", "deg", "ra", "ICRS right ascension."),
            "dec": ("float", "deg", "dec", "ICRS declination."),
            "field": ("string", None, "provenance", "S-PLUS field name."),
            "class_star": ("float", "1", "quality", "SExtractor CLASS_STAR: ~1 star-like; point-source cut > 0.9."),
            "fwhm": ("float", ("?", "S-PLUS DR4 documentation is ambiguous between pixels and arcsec for this column; verify on splus.cloud before physical use."), "measurement", "Source FWHM."),
            "ebv_sch": ("float", "mag", "measurement", "Schlegel E(B-V) at the source position."),
            "u_auto": ("float", "mag", "measurement", "u AUTO magnitude (AB; Javalambre system)."),
            "g_auto": ("float", "mag", "measurement", "g AUTO magnitude (AB)."),
            "r_auto": ("float", "mag", "measurement", "r AUTO magnitude (AB)."),
            "i_auto": ("float", "mag", "measurement", "i AUTO magnitude (AB)."),
            "z_auto": ("float", "mag", "measurement", "z AUTO magnitude (AB)."),
            "j0378_auto": ("float", "mag", "measurement", "J0378 narrow-band AUTO magnitude (AB)."),
            "j0395_auto": ("float", "mag", "measurement", "J0395 narrow-band AUTO magnitude (AB)."),
            "j0410_auto": ("float", "mag", "measurement", "J0410 narrow-band AUTO magnitude (AB)."),
            "j0430_auto": ("float", "mag", "measurement", "J0430 narrow-band AUTO magnitude (AB)."),
            "j0515_auto": ("float", "mag", "measurement", "J0515 narrow-band AUTO magnitude (AB)."),
            "j0660_auto": ("float", "mag", "measurement", "J0660 narrow-band AUTO magnitude (AB)."),
            "j0861_auto": ("float", "mag", "measurement", "J0861 narrow-band AUTO magnitude (AB)."),
            "e_u_auto": ("float", "mag", "uncertainty", "u AUTO magnitude uncertainty."),
            "e_g_auto": ("float", "mag", "uncertainty", "g AUTO magnitude uncertainty."),
            "e_r_auto": ("float", "mag", "uncertainty", "r AUTO magnitude uncertainty."),
            "e_i_auto": ("float", "mag", "uncertainty", "i AUTO magnitude uncertainty."),
            "e_z_auto": ("float", "mag", "uncertainty", "z AUTO magnitude uncertainty."),
            "e_j0378_auto": ("float", "mag", "uncertainty", "J0378 magnitude uncertainty."),
            "e_j0395_auto": ("float", "mag", "uncertainty", "J0395 magnitude uncertainty."),
            "e_j0410_auto": ("float", "mag", "uncertainty", "J0410 magnitude uncertainty."),
            "e_j0430_auto": ("float", "mag", "uncertainty", "J0430 magnitude uncertainty."),
            "e_j0515_auto": ("float", "mag", "uncertainty", "J0515 magnitude uncertainty."),
            "e_j0660_auto": ("float", "mag", "uncertainty", "J0660 magnitude uncertainty."),
            "e_j0861_auto": ("float", "mag", "uncertainty", "J0861 magnitude uncertainty."),
            "ring256": ("integer", None, "healpix", "HEALPix RING nside=256."),
            "nest4096": ("integer", None, "healpix", "HEALPix NEST nside=4096."),
        },
    },
}


# Archive-level profile authoring (plain dicts; validated by
# services.archive_profiles.datalab into the pydantic ArchiveProfile).
_PROFILE_DESCRIPTION = (
    "NOIRLab Astro Data Lab survey catalogs (Gaia DR3, DES DR1, DESI DR1, NSC DR2, "
    "SMASH DR1/2, DELVE DR3, Legacy Surveys DR9, SDSS DR17, VHS DR5, unWISE, 2MASS, "
    "AllWISE, S-PLUS DR4) queried through governed TAP/SQL builder tools."
)

_PROFILE_ENDPOINTS = [
    {
        "id": "datalab_tap",
        "description": "Data Lab TAP/SQL service (reached through the governed datalab_* tools).",
        "url": DATALAB_TAP_ENDPOINT,
        "protocol": "tap",
    },
    {
        "id": "sia_coadd_all",
        "description": "Data Lab SIA cutout service over the merged coadd collection.",
        "url": "https://datalab.noirlab.edu/sia/coadd_all",
        "protocol": "sia",
    },
    {
        "id": "sia_delve_dr3",
        "description": "Data Lab SIA cutout service for DELVE DR3 coadds.",
        "url": "https://datalab.noirlab.edu/sia/delve_dr3",
        "protocol": "sia",
    },
]

_PROFILE_QUERY_SURFACES = [
    {
        "id": "discover",
        "purpose": "List registered catalogs/tables with footprints before choosing where to query.",
        "tool": "datalab_list_catalogs",
        "request_kind": "structured_args",
    },
    {
        "id": "describe",
        "purpose": "Column names/region strategy/morphology hints for one catalog table — call before writing SQL.",
        "tool": "datalab_describe_table",
        "request_kind": "structured_args",
    },
    {
        "id": "select_rows",
        "purpose": "Governed cone selection with server-side cuts — the default row-fetch route.",
        "tool": "datalab_select_catalog_rows",
        "request_kind": "structured_args",
        "parameters": [
            {"name": "radius_deg", "json_type": "number", "unit": "deg",
             "description": "Cone radius in DEGREES (not arcmin)."},
            {"name": "value_cuts", "json_type": "array", "description":
                "Structured cuts, e.g. [{'column':'parallax_over_error','op':'>','value':5}]; "
                "the builder adds the NaN finiteness guard automatically — prefer over raw SQL."},
            {"name": "limit", "json_type": "integer", "unit": "1",
             "description": "Row cap; capped results are storage-order slices — never density-map them."},
        ],
        "endpoint_ids": ["datalab_tap"],
    },
    {
        "id": "expert_sql",
        "purpose": "Expert/debug raw SQL (governed): needs a q3c cone, indexed equality, approved BETWEEN box, or aggregate bound.",
        "tool": "datalab_sql_query",
        "request_kind": "sql",
        "query_argument": "sql",
        "parameters": [
            {"name": "expert_ack", "json_type": "boolean",
             "description": "Must be true — acknowledges expert raw-SQL mode."},
            {"name": "reason", "json_type": "string",
             "description": "Free-text justification (narrative; earns no benchmark trace credit)."},
        ],
        "endpoint_ids": ["datalab_tap"],
    },
    {
        "id": "density",
        "purpose": "Server-side density aggregation (grid or HEALPix) — the ONLY correct base for sky-density maps.",
        "tool": "datalab_density_aggregate",
        "request_kind": "structured_args",
        "parameters": [
            {"name": "step_deg", "json_type": "number", "unit": "deg",
             "description": "Grid cell size in degrees (mode='grid')."},
            {"name": "healpix_column", "json_type": "string",
             "description": "Registered HEALPix column; use the COARSE one (e.g. ring256) for regions wider than a few degrees."},
            {"name": "radius_deg", "json_type": "number", "unit": "deg",
             "description": "Cone radius in degrees; wide cones auto-tile — call ONCE with the full cone."},
        ],
        "endpoint_ids": ["datalab_tap"],
    },
    {
        "id": "cone_count",
        "purpose": "Cheap server-side row count for a cone — size a query before fetching rows.",
        "tool": "datalab_cone_count",
        "request_kind": "structured_args",
        "parameters": [
            {"name": "radius_deg", "json_type": "number", "unit": "deg",
             "description": "Cone radius in degrees."},
        ],
        "endpoint_ids": ["datalab_tap"],
    },
]

_PROFILE_PITFALLS = [
    {
        "id": "nan_ordering",
        "summary": "missing floats are NaN and sort ABOVE numbers — every one-sided cut (col > x) needs AND col < 'Infinity'",
        "detail": (
            "Data Lab tables store missing float values as NaN (not SQL NULL) and Postgres orders NaN above every "
            "real number, so a bare col > x / col >= x / col != x silently ADMITS every missing-value row. Add "
            "AND col < 'Infinity' to one-sided lower-bound cuts in raw SQL; structured value_cuts add the guard "
            "automatically. Upper-bounded and BETWEEN cuts are already NaN-safe."
        ),
        "applies_to": [{"kind": "archive", "ref": "datalab"}],
        "prompt_rank": 1,
    },
    {
        "id": "per_band_columns",
        "summary": "column naming is per-survey: DES/DELVE per-band mag_auto_r/class_star_r (no bare mag_auto), NSC/SMASH gmag/rmag, Gaia phot_g_mean_mag (bands g|bp|rp)",
        "detail": (
            "There is no bare mag_auto, class_star or spread_model on des_dr1.main — use mag_auto_r, class_star_r, "
            "spread_model_r etc. NSC/SMASH use gmag/rmag/...; Gaia uses phot_g_mean_mag/phot_bp_mean_mag/"
            "phot_rp_mean_mag (bands g|bp|rp, so 'rmag' does not exist); LS DR9 uses dered_mag_g/r/z (+W1/W2, no i band)."
        ),
        "applies_to": [{"kind": "archive", "ref": "datalab"}],
        "prompt_rank": 2,
    },
    {
        "id": "density_from_aggregates",
        "summary": "never build sky-density/overdensity maps from row-capped pulls — use datalab_density_aggregate (coarse HEALPix, e.g. ring256, for wide areas)",
        "applies_to": [{"kind": "surface", "ref": "density"}],
    },
    {
        "id": "smash_fieldid",
        "summary": "bound whole-field SMASH queries with the indexed fieldid = N (no cone); multi-epoch photometry lives in .source, mean objects in .object",
        "applies_to": [
            {"kind": "table", "ref": "smash_dr1.object"},
            {"kind": "table", "ref": "smash_dr2.object"},
        ],
    },
    {
        "id": "desi_quality",
        "summary": "DESI zpix reliability cuts: zwarn = 0 AND survey = 'main' AND main_primary; object class via spectype ('GALAXY'|'QSO'|'STAR')",
        "applies_to": [{"kind": "table", "ref": "desi_dr1.zpix"}],
    },
    {
        "id": "sdss_zwarning",
        "summary": "the SDSS specobj quality flag is zwarning (not zwarn); class is GALAXY|QSO|STAR",
        "applies_to": [{"kind": "table", "ref": "sdss_dr17.specobj"}],
    },
    {
        "id": "vhs_naming",
        "summary": "the VHS DR5 relation is vhs_dr5.vhs_cat_v3 with coordinates ra2000/dec2000 (not ra/dec); JHKs are Vega",
        "applies_to": [{"kind": "table", "ref": "vhs_dr5.vhs_cat_v3"}],
    },
    {
        "id": "expansion_live_schema",
        "summary": "catalogs beyond the curated set (delve_dr2, catwise2020, ls_dr10, ...) resolve through the live TAP schema — describe them before querying",
        "applies_to": [{"kind": "archive", "ref": "datalab"}],
    },
]

_PROFILE_GOLDEN_EXAMPLES = [
    {
        "id": "gaia_quality_cone",
        "intent": "Well-measured Gaia sources with colours and proper motions around M13.",
        "invocation": {
            "tool": "datalab_select_catalog_rows",
            "arguments": {
                "catalog": "gaia_dr3",
                "table": "gaia_source",
                "ra": 250.423,
                "dec": 36.46,
                "radius_deg": 0.2,
                "columns": ["ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec"],
                "value_cuts": [{"column": "parallax_over_error", "op": ">", "value": 5}],
                "limit": 1000,
            },
        },
        "note": "Structured value_cuts add the NaN finiteness guard automatically — prefer them over raw SQL.",
    },
    {
        "id": "des_per_band_cmd",
        "intent": "Clean DES stellar photometry for a colour-magnitude selection (per-band columns).",
        "invocation": {
            "tool": "datalab_sql_query",
            "arguments": {
                "sql": (
                    "SELECT ra, dec, mag_auto_g - mag_auto_r AS gr, mag_auto_r "
                    "FROM des_dr1.main "
                    "WHERE q3c_radial_query(ra, dec, 34.0, -5.0, 0.5) "
                    "AND flags_r = 0 AND mag_auto_r < 24 AND spread_model_r < 0.005 "
                    "LIMIT 5000"
                ),
                "expert_ack": True,
                "reason": "CMD needs a computed colour column; per-band mag_auto_g/r with NaN-safe upper-bound cuts.",
            },
        },
        "request": {"kind": "sql", "argument": "sql"},
        "note": "mag_auto_r / spread_model_r / flags_r are PER-BAND; upper-bound cuts are NaN-safe as written.",
    },
    {
        "id": "nsc_lmc_density",
        "intent": "Wide-area stellar density of the LMC region without pulling rows.",
        "invocation": {
            "tool": "datalab_density_aggregate",
            "arguments": {
                "catalog": "nsc_dr2",
                "table": "object",
                "mode": "healpix",
                "healpix_column": "ring256",
                "ra": 81.28,
                "dec": -69.78,
                "radius_deg": 5.0,
            },
        },
        "note": "Coarse ring256 for a 5-degree field; the aggregate counts EVERY row server-side.",
    },
    {
        "id": "des_cone_count",
        "intent": "Size a DES query before fetching rows.",
        "invocation": {
            "tool": "datalab_cone_count",
            "arguments": {"catalog": "des_dr1", "table": "main", "ra": 34.0, "dec": -5.0, "radius_deg": 0.5},
        },
    },
]

_PROFILE_UNIT_CONVENTIONS = [
    {
        "id": "coords_icrs_deg",
        "statement": "All coordinates are ICRS decimal degrees; all cone radii in datalab_* tools are DEGREES.",
    },
    {
        "id": "mag_systems",
        "statement": (
            "Magnitudes are AB for the DECam-family surveys (DES, NSC, DELVE, LS DR9, SMASH, S-PLUS) and "
            "Vega for Gaia, 2MASS, AllWISE, unWISE and VHS."
        ),
    },
]


def _column_spec_dict(name: str, info: tuple) -> Dict[str, Any]:
    """Expand a compact (dtype, unit, role, desc) tuple into ColumnSpec kwargs."""
    dtype, unit, role, desc = info
    spec: Dict[str, Any] = {"name": name, "dtype": dtype, "description": desc}
    if role is not None:
        spec["role"] = role
    numeric = dtype in ("integer", "float", "decimal")
    if isinstance(unit, tuple):
        spec["unit_state"] = "unknown"
        spec["unit_note"] = unit[1]
    elif isinstance(unit, str):
        spec["unit"] = unit
    elif numeric:
        spec["unit_state"] = "not_applicable"
    return spec


def to_profile() -> Dict[str, Any]:
    """Assemble the Data Lab ArchiveProfile as a PLAIN dict (adapter, not a
    rewrite): tables come from the curated registry merged with
    TABLE_PROFILE_INFO; pydantic validation happens in
    ``services.archive_profiles.datalab`` so this module never imports the
    profile package. Fails loudly if the authored column metadata and the
    authoritative ``columns`` lists ever drift apart."""

    sia_endpoint_ids = {e["url"]: e["id"] for e in _PROFILE_ENDPOINTS if e["protocol"] == "sia"}

    citations: List[Dict[str, Any]] = []
    tables: Dict[str, Any] = {}
    for qualified in registered_qualified_tables():
        catalog_key, table_key = qualified.split(".", 1)
        entry = DATALAB_CATALOGS[catalog_key]
        described = describe_table(catalog_key, table_key)
        info = TABLE_PROFILE_INFO.get(qualified)
        if info is None:
            raise ValueError(f"TABLE_PROFILE_INFO is missing curated table {qualified!r}")
        authored = info["columns"]
        registry_cols = described["columns"]
        missing = [c for c in registry_cols if c not in authored]
        extra = [c for c in authored if c not in registry_cols]
        if missing or extra:
            raise ValueError(
                f"{qualified}: column metadata drift (missing={missing}, extra={extra})"
            )

        citation_id = f"cite_{catalog_key}"
        if not any(c["id"] == citation_id for c in citations):
            cite = described.get("citation") or {}
            citations.append(
                {
                    "id": citation_id,
                    "text": cite.get("text") or catalog_key,
                    "url": cite.get("url"),
                    "doi": cite.get("doi"),
                }
            )

        tables[qualified] = {
            "purpose": info["purpose"],
            "grain": info["grain"],
            "columns": [_column_spec_dict(name, authored[name]) for name in registry_cols],
            "ra_column": described["ra_column"],
            "dec_column": described["dec_column"],
            "hints": {
                "footprint": described.get("footprint"),
                "region_strategy": described.get("region_strategy"),
                "healpix_columns": described.get("healpix_columns", []),
                "morphology": described.get("morphology", {}),
                "bitmasks": described.get("bitmasks", {}),
                "aggregate_safe": described.get("aggregate_safe"),
                "endpoint_ids": [
                    sia_endpoint_ids[url]
                    for url in entry.get("sia_endpoints", [])
                    if url in sia_endpoint_ids
                ],
            },
            "citation_ids": [citation_id],
        }

    return {
        "archive": "datalab",
        "aliases": ("noirlab", "astro data lab", "noao"),
        "description": _PROFILE_DESCRIPTION,
        "endpoints": _PROFILE_ENDPOINTS,
        "query_surfaces": _PROFILE_QUERY_SURFACES,
        "tables": tables,
        "pitfalls": _PROFILE_PITFALLS,
        "golden_examples": _PROFILE_GOLDEN_EXAMPLES,
        "unit_conventions": _PROFILE_UNIT_CONVENTIONS,
        "citations": citations,
    }


__all__ = [
    "DATALAB_CATALOGS",
    "EXPANSION_CATALOGS",
    "aggregate_safe_tables",
    "cached_tap_schema",
    "citation",
    "dataframe_from_schema",
    "default_refresh_tables",
    "describe_table",
    "ensure_tap_schema_fresh",
    "expansion_qualified_tables",
    "known_columns",
    "list_catalogs",
    "live_columns",
    "live_table_entry",
    "refresh_tap_schema",
    "region_strategy",
    "registered_qualified_tables",
    "tap_schema_ttl_seconds",
    "to_profile",
    "TABLE_PROFILE_INFO",
    "DATALAB_TAP_ENDPOINT",
]
