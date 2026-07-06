# F02 — Sky-coverage answers via CDS MOCServer

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the two exemplar
files it names BEFORE coding. Follow the service/test/wiring contracts there.

## Objective

Answer "what surveys/datasets cover this sky position or region?" instantly,
and "does survey X cover position Y?", using the CDS MOCServer. This becomes
a preflight for all archive tools (avoid querying archives with no coverage).

## Files to create/edit

- CREATE `services/moc_coverage.py`
- CREATE `tests/unit/test_moc_coverage.py`
- CREATE `scripts/smoke/smoke_f02_moc.py`
- EDIT `core/agent.py` (4 anchor insertions per CONVENTIONS)

## Endpoint

Base: `https://alasky.unistra.fr/MocServer/query` (env override
`MOCSERVER_BASE_URL`, timeout env `MOCSERVER_TIMEOUT`, default 30 s).
CAUTION (verified live 2026-07-03): the path is CASE-SENSITIVE —
`MocServer` works, `MOCServer` returns HTTP 404. `moc_sky_fraction` comes
back as a JSON STRING ("1") — coerce to float.

Cone/point query params (HTTP GET):
- `RA=<deg>` `DEC=<deg>` `SR=<deg>` (ICRS decimal degrees; SR=0.0 is valid
  for a point — send SR=0 exactly, NEVER a tiny placeholder like 1e-6:
  MocServer returns HTTP 500 for 0 < SR < ~1e-4)
- `intersect=overlaps`
- `get=record`, `fmt=json`
- `fields=ID,obs_title,dataproduct_type,em_min,em_max,moc_sky_fraction`
- optional server-side filter via `expr=` (e.g. `dataproduct_type=image`);
  NOTE: `expr` syntax support varies — implement filtering CLIENT-SIDE as the
  authoritative path and only pass `expr` when it is a simple
  `dataproduct_type=<image|catalog|cube>` filter. If the server errors on
  `expr`, retry once without it and add a warning.

Response: JSON array of records (dicts) with the requested fields. It can be
LARGE (thousands of records for a bare cone) — cap client-side.

## Class and methods

```python
class MocCoverageService:
    def __init__(self, *, base_url=None, timeout=None): ...

    def coverage_at(self, ra, dec, radius_deg=0.0, dataproduct_type=None,
                    keyword=None, max_rows=50) -> dict:
        """Datasets whose MOC intersects the cone.
        - validate ra [0,360), dec [-90,90]; clamp radius_deg to <=30 (warn)
        - clamp max_rows to <=200 (warn)
        - keyword: case-insensitive substring filter on ID + obs_title
          (client-side)
        - dataproduct_type: one of image|catalog|cube (client-side filter;
          also passed as expr= best-effort)
        - sort rows: moc_sky_fraction ASCENDING (most specific datasets
          first — an all-sky survey covering everything is least informative)
        - each row: {"id", "title", "dataproduct_type", "regime",
          "moc_sky_fraction"}  where regime is derived from em_min/em_max
          (meters) via _regime_label(); None-safe
        - return also "total_matches" (count before the max_rows cap) and a
          warning when truncated
        """

    def survey_covers(self, survey_keyword, ra, dec) -> dict:
        """Boolean answer: any dataset whose ID or title contains
        survey_keyword (case-insensitive) covering the point.
        Returns {"success": True, "covered": bool, "matches": [ids...(<=10)],
        "warnings": [...], "provenance": {...}}."""
```

`_regime_label(em_min, em_max)`: midpoint wavelength in meters →
`"radio"` (>1e-2), `"mm/sub-mm"` (1e-3..1e-2), `"infrared"` (1e-6..1e-3),
`"optical"` (3e-7..1e-6), `"UV"` (1e-8..3e-7), `"X-ray"` (1e-11..1e-8),
`"gamma"` (<1e-11); `None` if fields missing.

## agent.py wiring

Tools (category `"archive"`):

1. `survey_coverage` — wrapper `_survey_coverage(target_name=None, ra=None,
   dec=None, radius_deg=0.0, dataproduct_type=None, keyword=None,
   max_rows=50)`. Resolve position via `_live_imagery_coordinates`. Table
   result via `_external_catalog_table_result`, columns
   `["id","title","dataproduct_type","regime","moc_sky_fraction"]`,
   source `"CDS MOCServer"`,
   filter_label `f"coverage at {label}, r={radius_deg:g} deg"`.
   Description: "List surveys/datasets whose sky coverage (MOC) includes a
   position or region — use FIRST to check what data exists before querying
   archives; filter by dataproduct_type (image|catalog|cube) or keyword."
2. `survey_covers_position` — wrapper `_survey_covers_position(
   survey_keyword, target_name=None, ra=None, dec=None)`; required:
   `survey_keyword`. Returns the service dict directly (plus resolved label
   in a `"target"` key) — no table card needed for a boolean.
   Description: "Check whether a named survey (e.g. 'VLASS', 'SDSS',
   'GLEAM') covers a given position; returns covered true/false and matching
   dataset IDs."

Status labels: `"survey_coverage": "Checking sky coverage (MOCServer)"`,
`"survey_covers_position": "Checking survey footprint"`.

Prompt bullet: `Use \`survey_coverage\` / \`survey_covers_position\` BEFORE
archive searches to check which surveys actually cover a position; prefer
them for any "is there data / which surveys observed X" question.`

## Unit tests (offline; FakeResponse pattern)

1. `coverage_at` builds correct URL/params (RA/DEC/SR/intersect/get/fmt/
   fields present; timeout passed); rows normalized + regime derived
   (feed em_min/em_max for a radio and an optical record); sorted by
   moc_sky_fraction ascending.
2. radius clamp (radius_deg=90 → 30 + warning) and max_rows clamp.
3. keyword + dataproduct_type client-side filters work on a canned 4-record
   payload.
4. truncation: 60 canned records, max_rows=50 → 50 rows,
   total_matches=60, warning present.
5. HTTP 500 → success False, error contains "HTTP 500".
6. `survey_covers` true and false cases.

## Smoke expectations (`scripts/smoke/smoke_f02_moc.py`)

- `coverage_at(187.2779, 2.0524, 0.1)` (3C 273 field): success, total
  matches > 100; some record IDs contain `CDS/`.
- `survey_covers("VLASS", 180.0, 30.0)` → covered True.
  (Verified live 2026-07-03: do NOT use 3C 273 here — the VLASS Quicklook
  MedianStack MOC has a genuine hole at 3C 273, bright-source tiles are
  excluded from the HiPS MOC. Point queries: send SR=0, never 1e-6 —
  MocServer 500s on 0 < SR < ~1e-4.)
- `survey_covers("VLASS", 10.0, -75.0)` (Dec −75, below VLASS window) →
  covered False.
Print each result; exit 0 only if all three hold.

## Chat acceptance question

"Which radio surveys cover NGC 253?" → agent should call `survey_coverage`
(dataproduct_type/keyword optional) and answer from the table.

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done all ticked
- [ ] Unit tests above all present and green
- [ ] Smoke passes live
- [ ] agent.py: 2 tools + 2 status labels + 1 prompt bullet, insertions only
