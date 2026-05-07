"""
Archive-link helpers for Quasar UI data cards.

These helpers keep archive detection and link routing in one place so the
streaming and post-streaming UI paths stay in sync.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote_plus

MAST_MISSIONS = {
    "JWST",
    "HST",
    "TESS",
    "KEPLER",
    "K2",
    "GALEX",
    "IUE",
    "FUSE",
    "SWIFT",
}

ESO_INSTRUMENTS = {
    "MUSE",
    "KMOS",
    "XSHOOTER",
    "FORS2",
    "HAWKI",
    "UVES",
    "SPHERE",
    "GRAVITY",
    "ESPRESSO",
    "CRIRES",
    "VISIR",
    "FLAMES",
    "MATISSE",
    "PIONIER",
}


def _combined_hint(*values: str) -> str:
    return " ".join(str(v).upper() for v in values if v)


def _first_nonempty(df, column: str) -> Optional[str]:
    if df is None or not hasattr(df, "columns") or column not in df.columns:
        return None

    for value in df[column]:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return None


def _unique_upper(df, column: str, limit: int = 25) -> set[str]:
    if df is None or not hasattr(df, "columns") or column not in df.columns:
        return set()

    values = set()
    for value in df[column].head(limit):
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            values.add(text.upper())
    return values


def infer_archive_kind(df, source_hint: str = "", filter_label: str = "") -> str:
    """Infer which archive a result table belongs to."""
    hint = _combined_hint(source_hint, filter_label)

    if "IRSA" in hint:
        return "irsa"
    if "MAST" in hint:
        return "mast"
    if "CADC" in hint:
        return "cadc"
    if "ESO" in hint and "ALMA" not in hint:
        return "eso"
    if "ALMA" in hint:
        return "alma"

    columns = set(getattr(df, "columns", []))
    if "member_ous_uid" in columns:
        return "alma"

    telescopes = _unique_upper(df, "telescope")
    if "IRSA" in telescopes:
        return "irsa"
    if telescopes & MAST_MISSIONS:
        return "mast"
    if "ESO" in telescopes:
        return "eso"

    collections = _unique_upper(df, "obs_collection")
    if collections & MAST_MISSIONS:
        return "mast"

    if "obs_publisher_did" in columns:
        sample = (_first_nonempty(df, "obs_publisher_did") or "").lower()
        if "cadc" in sample:
            return "cadc"
        if "mast" in sample:
            return "mast"

    if {"obsid", "productFilename", "productType"} & columns:
        return "mast"

    instruments = {
        value.replace("-", "").replace(" ", "")
        for value in _unique_upper(df, "instrument_name")
    }
    if instruments & ESO_INSTRUMENTS:
        return "eso"

    return "unknown"


def build_archive_link(df, source_hint: str = "", filter_label: str = "") -> Optional[str]:
    """
    Build the best archive landing link for a result table.

    Returns None when there is no reliable archive page to link to.
    """
    archive_kind = infer_archive_kind(df, source_hint=source_hint, filter_label=filter_label)
    target_name = _first_nonempty(df, "target_name")

    if archive_kind == "cadc":
        if target_name:
            return (
                "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/"
                f"?Observation.target.name={quote_plus(target_name)}"
            )
        obs_id = _first_nonempty(df, "obs_id")
        if obs_id:
            return (
                "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/"
                f"?Observation.observationID={quote_plus(obs_id)}"
            )
        return "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/"

    if archive_kind == "alma":
        member_ous_uid = _first_nonempty(df, "member_ous_uid")
        if member_ous_uid:
            return (
                "https://almascience.nrao.edu/aq/"
                f"?member_ous_id={quote_plus(member_ous_uid)}"
            )
        if target_name:
            return f"https://almascience.eso.org/aq/?target={quote_plus(target_name)}"
        return "https://almascience.eso.org/aq/"

    if archive_kind == "mast":
        if target_name:
            return (
                "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html"
                f"?searchQuery={quote_plus(target_name)}"
            )
        return "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html"

    if archive_kind == "eso":
        return "https://archive.eso.org/scienceportal/home"

    if archive_kind == "irsa":
        return "https://irsa.ipac.caltech.edu/frontpage/"

    return None
