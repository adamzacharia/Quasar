"""Canonical cached SIMBAD target-name -> (ra_deg, dec_deg) resolver.

One source of truth for every archive client. Only SUCCESSFUL resolutions are
memoized; a failed lookup raises internally so a transient SIMBAD outage never
poisons the LRU with (None, None) for the process lifetime. (C16)
"""
import os
from functools import lru_cache

SIMBAD_HOST = "simbad.cds.unistra.fr"


def _simbad_timeout_default() -> float:
    raw = os.getenv("SIMBAD_TIMEOUT_SECONDS", "").strip()
    try:
        return max(2.0, float(raw)) if raw else 10.0
    except ValueError:
        return 10.0


SIMBAD_TIMEOUT_S = _simbad_timeout_default()


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

    from services.host_breaker import HostBreaker
    from services.tool_budgets import bounded_timeout, call_bounded

    # Name resolution is a 1-2 s lookup when CDS is healthy; astroquery's
    # default 60 s would eat almost half of a 150 s tool guard on its own.
    # SIMBAD is served by CDS, so a dead CDS (live 2026-09-20: SSLError /
    # ReadTimeout on alasky) fails the SECOND resolve of a turn instantly too.
    HostBreaker.check(SIMBAD_HOST)
    timeout = bounded_timeout(SIMBAD_TIMEOUT_S, minimum=2.0, label="SIMBAD resolve")
    try:
        # astroquery's Simbad.timeout is a server-side TAP execution duration,
        # not an HTTP timeout, so the request is bounded from outside.
        result = call_bounded(
            lambda: Simbad.query_object(target_name), timeout,
            label="SIMBAD resolve", thread_name="quasar-simbad-resolve",
        )
    except Exception as exc:
        HostBreaker.record_failure(SIMBAD_HOST, exc)
        raise
    HostBreaker.record_success(SIMBAD_HOST)
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
