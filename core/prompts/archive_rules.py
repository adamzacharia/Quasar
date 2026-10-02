"""Cross-archive accuracy rules shared by BOTH system-prompt paths.

Injected by core/agent.py::_build_system_prompt next to the per-archive
schema-grounding index, so the legacy prompt (QUASAR_PROMPT_V2=0) and the v2
bundle carry the same text. It sits outside the v2 core token budget, like
the other generated blocks. Each rule answers a failure from the 2026-10-01
MANNA-evals UI benchmark (stems in brackets are for maintainers; the model
never sees this docstring).

- NRAO routing: prompts that said "NRAO obscore" went to almascience.nrao.edu
  with ivoa.obscore / LOWER() / bare sync reads (T10, T16, T18, T19, T25,
  T26, MQ14, MQ21).
- Row caps: a 500-row ESO cap was reported as "500 observations" (MQ17, MQ22).
- Capability questions answered from memory, with invented access (T02, T03).
- Identifiers no tool returned (a Gaia source_id after a COUNT only, MQ18).
- A local file:// path or a DataLink reported as the FITS URL (MQ09, T12, MQ16).
"""

from __future__ import annotations

from services.row_cap import COUNT_RULE

ARCHIVE_RULES = (
    "NRAO archive = VLA/EVLA/VLBA/GBT. Structured: search_by_position or search_by_target with "
    "facility='VLA'|'VLBA'|'GBT'. Raw ADQL: vo_adql_query at https://data-query.nrao.edu/tap on "
    "tap_schema.obscore (not ivoa.obscore), no LOWER/UPPER/ILIKE, mode='auto' or 'async' with a selective "
    "WHERE (a cone for positional asks). almascience.nrao.edu is ALMA's mirror, never the NRAO archive.",
    COUNT_RULE,
    "Questions about which archives you can reach or an archive's quirks: call browse_schema(<archive>) "
    "(or vo_find_services) and answer from what it returns; do not list archives, capabilities or quirks "
    "from memory.",
    "Quote an identifier (Gaia source_id, obs_id, MOUS uid, bibcode, archive URL) only if a tool returned "
    "it this turn or the user gave it; otherwise run the query that returns it.",
    "A 'direct access URL' is an https file URL a tool returned (prefer fits_url over a DataLink "
    "access_url); never a local path or a file:// link.",
)


def archive_rules_block() -> str:
    """The block appended after the schema-grounding index (both prompt paths)."""
    return "\nARCHIVE ACCURACY RULES:\n" + "\n".join(f"- {rule}" for rule in ARCHIVE_RULES) + "\n"


__all__ = ["ARCHIVE_RULES", "archive_rules_block"]
