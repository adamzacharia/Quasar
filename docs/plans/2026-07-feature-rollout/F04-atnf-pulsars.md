# F04 — ATNF pulsar catalogue (psrqpy)

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the exemplars it
names BEFORE coding. Dependency `psrqpy` is pre-installed in Phase 0.

## Objective

Cone search and name lookup against the ATNF Pulsar Catalogue.

## Files

- CREATE `services/pulsar_catalog.py`
- CREATE `tests/unit/test_pulsar_catalog.py`
- CREATE `scripts/smoke/smoke_f04_pulsars.py`
- EDIT `core/agent.py` (4 anchors)

## Design

psrqpy downloads the full catalogue on first use (a few MB, cached by the
library). Load ONCE per service instance and filter locally with astropy —
do not issue per-query remote conditions.

```python
# Verified live 2026-07-03: for B0329+54, psrqpy returns JNAME='J0332+5434',
# BNAME='B0329+54', NAME='B0329+54' (NAME = B-name when it exists, else
# J-name). RAJD/DECJD are decimal degrees. Include BNAME so name lookups on
# B-names resolve robustly.
PSR_PARAMS = ["JNAME", "NAME", "BNAME", "RAJD", "DECJD", "P0", "P1", "DM",
              "DIST", "AGE", "BSURF", "EDOT", "S1400", "BINARY", "ASSOC",
              "TYPE"]

class PulsarCatalogService:
    def __init__(self, *, table_loader=None):
        # table_loader: callable() -> pandas.DataFrame with PSR_PARAMS cols.
        # Default implementation (lazy):
        #   from psrqpy import QueryATNF
        #   df = QueryATNF(params=PSR_PARAMS).pandas
        # Cache the DataFrame on the instance after first load. Record the
        # catalogue version if available (QueryATNF(...).get_version) in
        # provenance; failure to get version is non-fatal.

    def search_pulsars(self, ra, dec, radius_deg=1.0, max_rows=25) -> dict:
        # validate coords; clamp radius_deg <= 30 (warn), max_rows <= 200
        # separation via astropy SkyCoord (RAJD/DECJD are degrees, note some
        # rows have NaN positions -> drop them first)
        # sort by separation; rows include sep_arcmin (2 dp)

    def pulsar_lookup(self, name) -> dict:
        # match JNAME, BNAME, or NAME case-insensitively, allowing a missing
        # 'PSR ' prefix and B/J-name variants ('B0329+54' matches BNAME/NAME
        # 'B0329+54'; 'J0332+5434' matches JNAME). Return the single row (or
        # several matches) with ALL PSR_PARAMS normalized.
```

Row normalization: floats rounded sensibly (P0 6 sig figs, DM 2 dp); NaN →
None; column keys lowercase (`jname`, `p0_s`, `p1`, `dm_pc_cm3`,
`dist_kpc`, `age_yr`, `bsurf_g`, `edot_erg_s`, `s1400_mjy`, `binary`,
`assoc`, `type`, `sep_arcmin`).

Provenance: `{"service": "ATNF Pulsar Catalogue (psrqpy)",
"catalogue_version": <str|None>, ...query params...}`.

Failure mode: first-load network failure → `{"success": False, "error":
"ATNF catalogue download failed: ..."}` (do not cache the failure — retry on
next call).

## agent.py wiring

Tools (category `"archive"`):

1. `search_pulsars` — `_search_pulsars(target_name=None, ra=None, dec=None,
   radius_deg=1.0, max_rows=25)`; table columns
   `["jname","p0_s","dm_pc_cm3","s1400_mjy","dist_kpc","binary","assoc",
   "sep_arcmin"]`, source `"ATNF Pulsar Catalogue"`.
   Description: "Search the ATNF pulsar catalogue around a sky position;
   returns period, DM, 1400 MHz flux, distance, binarity, associations."
2. `pulsar_lookup` — `_pulsar_lookup(name)` required. Same table helper
   (single row) with the full parameter set columns.
   Description: "Look up a pulsar by J/B name in the ATNF catalogue and
   return its full timing/derived parameters."

Status labels: `"search_pulsars": "Searching ATNF pulsar catalogue"`,
`"pulsar_lookup": "Looking up pulsar parameters"`.

Prompt bullet: `Use \`search_pulsars\` / \`pulsar_lookup\` for anything
pulsar-related (periods, DMs, S1400, associations) instead of web search.`

## Unit tests

Inject `table_loader` returning a small pandas DataFrame (5 pulsars incl.
one with NaN position, one binary, B0329+54 with P0=0.714520, DM=26.7641):
1. Cone search around (53.2475, 54.5787) (B0329+54, RAJD/DECJD deg) finds it
   first; NaN-position row dropped without error; sep_arcmin present.
2. Radius/max_rows clamps warn.
3. `pulsar_lookup("B0329+54")` and `pulsar_lookup("psr b0329+54")` both hit;
   unknown name → success True, count 0.
4. Loader raising → success False with informative error; second call
   retries loader (assert loader called twice).
5. NaN → None normalization in rows.

## Smoke expectations

- `pulsar_lookup("B0329+54")` → P0 ≈ 0.71452 s (tol 1e-3), DM ≈ 26.7 (tol 0.5).
- `search_pulsars(83.63, 22.01, 1.0)` (Crab field) → includes J0534+2200.
First live call may take ~1 min (catalogue download) — allow it.

## Chat acceptance question

"What pulsars lie within a degree of the Crab nebula, and what are their DMs?"

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done ticked
- [ ] Unit tests green; smoke passes; wiring complete (2 tools)
