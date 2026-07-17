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

from typing import Dict, List

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


__all__ = [
    "ArchiveProfile",
    "PROFILES",
    "canonical_slug",
    "get_profile",
    "list_profiles",
    "prompt_index_lines",
]
