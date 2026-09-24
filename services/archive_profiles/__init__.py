"""
services/archive_profiles — per-archive grounding profiles (Feature 3).

Adding an archive == dropping a module in this package that exposes
``PROFILE: ArchiveProfile`` (schema in ``schema.py``; converged in
codex-bridge duel task-81b3b73-2145) and adding its slug to
``schema.ArchiveSlug``. Modules are AUTO-DISCOVERED — no registration list
to maintain (guard CX-01).

Consumers:
* ``capabilities/schema.py`` (the ``browse_schema`` tool) via
  ``get_profile`` + ``projections``;
* ``core/agent.py::_build_system_prompt`` via ``prompt_index_lines()``
  (gated by ``QUASAR_SCHEMA_GROUNDING``);
* ``tests/unit/test_archive_profiles.py`` (the author-time gates).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from services.archive_profiles.schema import ArchiveProfile

def _load() -> Dict[str, ArchiveProfile]:
    """Auto-discover every profile module in this package (any module that
    exposes ``PROFILE``); ``schema``/``projections``/private modules are
    infrastructure, not profiles."""
    import importlib
    import pkgutil

    profiles: Dict[str, ArchiveProfile] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name in ("schema", "projections") or info.name.startswith("_"):
            continue
        module = importlib.import_module(f"services.archive_profiles.{info.name}")
        profile = getattr(module, "PROFILE", None)
        if profile is None:
            continue
        if not isinstance(profile, ArchiveProfile):
            raise TypeError(f"{info.name}.PROFILE is not an ArchiveProfile")
        if profile.archive in profiles:
            raise ValueError(f"duplicate profile for archive {profile.archive!r} in {info.name}")
        profiles[profile.archive] = profile
    return profiles


PROFILES: Dict[str, ArchiveProfile] = _load()

# slug + every alias -> canonical slug (lower/stripped match).
_ALIAS_TO_SLUG: Dict[str, str] = {}
for _slug, _profile in PROFILES.items():
    _ALIAS_TO_SLUG[_slug] = _slug
    for _alias in _profile.aliases:
        _ALIAS_TO_SLUG[_alias.strip().lower()] = _slug


def canonical_slug(archive: str) -> str:
    """Resolve a slug or alias to the canonical archive slug (KeyError if unknown)."""
    key = str(archive or "").strip().lower()
    if key not in _ALIAS_TO_SLUG:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown archive {archive!r}. Known archives: {known}")
    return _ALIAS_TO_SLUG[key]


def get_profile(archive: str) -> ArchiveProfile:
    """The profile for a slug or alias (KeyError with the known list if unknown)."""
    return PROFILES[canonical_slug(archive)]


def list_profiles() -> List[Dict[str, str]]:
    """Compact per-archive index rows (slug + one-line description)."""
    return [
        {"archive": slug, "description": profile.description}
        for slug, profile in sorted(PROFILES.items())
    ]


def prompt_index_lines() -> List[str]:
    """The compact system-prompt index: one line per archive, built ONLY from
    the slug and the prompt-ranked pitfall summaries (duel DX-12)."""
    lines: List[str] = []
    for slug in sorted(PROFILES):
        profile = PROFILES[slug]
        summaries = profile.prompt_pitfalls()
        line = f"- {slug}: call browse_schema('{slug}') before writing a query"
        if summaries:
            line += "; top pitfalls: " + "; ".join(summaries)
        lines.append(line + ".")
    return lines


def _split(url: str):
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return "", ""
    return (parts.hostname or "").lower().rstrip("."), (parts.path or "").rstrip("/").lower()


def profile_for_endpoint(url: str) -> Optional[ArchiveProfile]:
    """The profile whose TAP/SIA endpoint serves ``url`` (same host, or a
    declared mirror host, and the endpoint path as a path prefix). Longest
    path match wins; None for an archive Quasar has no profile for."""
    host, path = _split(url)
    if not host:
        return None
    best: Optional[ArchiveProfile] = None
    best_len = -1
    for profile in PROFILES.values():
        hosts_for_profile = {h.lower() for h in profile.mirror_hosts}
        for endpoint in profile.endpoints:
            if endpoint.protocol not in ("tap", "sia"):
                continue
            ep_host, ep_path = _split(str(endpoint.url))
            if host != ep_host and host not in hosts_for_profile:
                continue
            if path == ep_path or path.startswith(ep_path + "/"):
                if len(ep_path) > best_len:
                    best, best_len = profile, len(ep_path)
    return best


def error_hints(url: str, *texts: str) -> List[str]:
    """Summaries of the pitfalls whose ``error_triggers`` fire on ``texts``
    (the submitted ADQL and the server error) for the archive serving
    ``url``. Empty for unknown archives or when nothing fires."""
    profile = profile_for_endpoint(url)
    if profile is None:
        return []
    return [p.summary for p in profile.pitfalls if p.error_triggers and p.fires_on(*texts)]


def sia_endpoint_for(archive: str) -> Optional[str]:
    """The profile's SIA service URL for a slug or alias, or None. Resolves
    through ``canonical_slug`` but never raises for an unknown archive."""
    try:
        profile = get_profile(archive)
    except KeyError:
        return None
    for endpoint in profile.endpoints:
        if endpoint.protocol == "sia":
            return str(endpoint.url).rstrip("/")
    return None


def sia_archives() -> List[str]:
    """Slugs whose profile declares an SIA endpoint (vo_image_search archives)."""
    return sorted(slug for slug in PROFILES if sia_endpoint_for(slug))


__all__ = [
    "ArchiveProfile",
    "PROFILES",
    "canonical_slug",
    "get_profile",
    "list_profiles",
    "error_hints",
    "profile_for_endpoint",
    "prompt_index_lines",
    "sia_archives",
    "sia_endpoint_for",
]
