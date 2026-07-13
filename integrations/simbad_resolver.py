"""Canonical cached SIMBAD target-name -> (ra_deg, dec_deg) resolver.

One source of truth for every archive client. Only SUCCESSFUL resolutions are
memoized; a failed lookup raises internally so a transient SIMBAD outage never
poisons the LRU with (None, None) for the process lifetime. (C16)
"""
from functools import lru_cache


class _SimbadUnresolved(Exception):
    """Internal: resolution failed — raised so lru_cache does NOT memoize it."""


@lru_cache(maxsize=256)
def _resolve_simbad_success_cached(target_name: str):
    """Resolve target name → (ra_deg, dec_deg) via SIMBAD.

    Only SUCCESSFUL resolutions are cached (coordinates are stable). A failed
    lookup raises so the LRU never memoizes it. (C16)"""
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    result = Simbad.query_object(target_name)
    if result is None or len(result) == 0:
        raise _SimbadUnresolved(target_name)

    colnames = set(result.colnames)
    if {"ra", "dec"} <= colnames:
        return (float(result["ra"][0]), float(result["dec"][0]))
    if {"RA_d", "DEC_d"} <= colnames:
        return (float(result["RA_d"][0]), float(result["DEC_d"][0]))
    if {"RA", "DEC"} <= colnames:
        coord = SkyCoord(result["RA"][0], result["DEC"][0], unit=(u.hourangle, u.deg))
        return (coord.ra.deg, coord.dec.deg)
    raise _SimbadUnresolved(target_name)


def _resolve_simbad_cached(target_name: str):
    """Public resolver: same (ra, dec) / (None, None) contract as always;
    failures are simply no longer cached. Other exceptions (network errors
    raised by astroquery) keep propagating to callers unchanged."""
    try:
        return _resolve_simbad_success_cached(target_name)
    except _SimbadUnresolved:
        return (None, None)


# Callers (and tests) manage the cache through the public name.
_resolve_simbad_cached.cache_clear = _resolve_simbad_success_cached.cache_clear
