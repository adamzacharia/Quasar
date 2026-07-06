# F11 — Solar-system: JPL Horizons ephemerides + SkyBoT moving-object check

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the exemplars it
names BEFORE coding.

## Objective

(a) Ephemerides for any solar-system body via JPL Horizons; (b) "is there an
asteroid/comet in this field at this time?" contamination checks via IMCCE
SkyBoT cone search.

## Files

- CREATE `services/solar_system.py`
- CREATE `tests/unit/test_solar_system.py`
- CREATE `scripts/smoke/smoke_f11_sso.py`
- EDIT `core/agent.py` (4 anchors)

## Part A — Horizons (astroquery, lazy import)

```python
def horizons_ephemeris(self, target, start, stop, step="1d",
                       location="500") -> dict:
    from astroquery.jplhorizons import Horizons   # lazy
    obj = Horizons(id=target, location=location,
                   epochs={"start": start, "stop": stop, "step": step})
    tbl = obj.ephemerides()
```

- Validate: start/stop ISO strings; reject ranges > 370 days (warning +
  truncate stop instead of erroring is fine); step like "1d"/"6h"/"30m".
- Constructor injectable `horizons_factory=None` (callable with the same
  signature returning an object with `.ephemerides()`) so tests skip
  astroquery entirely.
- Ambiguous-target errors from Horizons include a list of matches in the
  exception text — return success False with that text intact (it is the
  useful part).
- Row schema: {"datetime", "ra_deg", "dec_deg", "delta_au" (observer
  distance), "r_au" (heliocentric), "v_mag" (from column `V`; comets may
  have `Tmag`/`Nmag` instead — take the first present, else None),
  "elong_deg", "alpha_deg" (phase angle, col `alpha`)}. Cap rows at 400
  (warn + thin by stride if exceeded).

## Part B — SkyBoT cone search (requests)

Base: `https://ssp.imcce.fr/webservices/skybot/api/conesearch.php` (env
`SKYBOT_BASE_URL`, `SKYBOT_TIMEOUT` default 30 s).

GET params (note the leading dashes in the names):
`-ep=<JD>`, `-ra=<deg>`, `-dec=<deg>`, `-rd=<radius deg, clamp to <=10 with
warning>`, `-mime=json`, `-output=object`, `-loc=500`.

VERIFIED LIVE 2026-07-03 — read carefully, the spec below was corrected from
what the endpoint actually does:

- **EPOCH MUST BE A JULIAN DATE.** The ISO form with a `T` separator
  (`2026-07-03T00:00:00`) intermittently triggers a server-side
  `calceph_compute_unit error` and returns `{"flag":-1,...}` with NO data.
  Convert every epoch to JD first: `from astropy.time import Time;
  jd = Time(epoch_or_now, scale="utc").jd` and send `-ep=<jd>`. This is
  robust across all tested dates. `epoch=None` → `Time.now().jd` (record the
  ISO form in provenance for readability).
- **RESPONSE SHAPE**: `-mime=json` returns a JSON **array** of records
  directly, `[ {...}, ... ]` (NOT wrapped in `{"data": ...}`). An empty or
  failed field returns either `[]` or an object with `"flag": -1` /
  `"data"` absent — treat BOTH as `success True, count 0` with a warning
  carrying any `"message"`. Only a transport error (non-200, timeout) is
  `success False`.
- **RECORD KEYS (exact, space- and unit-suffixed)**: `"Num"` (int or
  absent for unnumbered), `"Name"`, `"RA (hms)"`, `"DEC (dms)"`, `"Class"`,
  `"VMag (mag)"`, `"Err (arcsec)"`, `"d (arcsec)"` (angular sep from cone
  centre), plus a nested `"ssodnet"` object (ignore). Match keys
  case-insensitively AND ignoring the ` (...)` suffix so minor server
  changes don't break parsing.
- **RA/DEC ARE ALWAYS SEXAGESIMAL** in JSON: `"RA (hms)"` is in HOURS,
  `"DEC (dms)"` in degrees. Convert with
  `SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))` → degrees.
- **NO Dg/Dh (distances) in `-output=object`.** Do NOT put observer/
  heliocentric distance in the v1 row schema. (They need a richer
  `-output`; out of scope.)

Row schema (v1): {"number" (None if unnumbered), "name", "ra_deg",
"dec_deg", "class" (e.g. "MB>Middle"), "v_mag", "sep_arcsec" (from
`d (arcsec)`), "pos_err_arcsec" (from `Err (arcsec)`)}.

Method: `skybot_cone(ra, dec, radius_deg=0.2, epoch=None) -> dict`.

## agent.py wiring

Tools:
1. `solar_system_ephemeris` (archive) — `_solar_system_ephemeris(target,
   start, stop, step="1d")` (target/start/stop required). Table columns
   `["datetime","ra_deg","dec_deg","delta_au","r_au","v_mag","elong_deg"]`,
   source `"JPL Horizons"`.
   Description: "JPL Horizons ephemeris for a planet, asteroid, or comet
   over a date range: RA/Dec, distances, magnitude, elongation."
2. `moving_object_check` (archive) — `_moving_object_check(target_name=None,
   ra=None, dec=None, radius_deg=0.2, epoch=None)`. Table columns
   `["name","class","v_mag","ra_deg","dec_deg","sep_arcsec",
   "pos_err_arcsec"]`, source `"IMCCE SkyBoT"`, filter_label includes epoch.
   Description: "List known asteroids/comets inside a field at a given epoch
   (SkyBoT) — use to check whether a transient/odd detection is a known
   moving object."

Status labels: `"solar_system_ephemeris": "Querying JPL Horizons"`,
`"moving_object_check": "Checking for moving objects (SkyBoT)"`.

Prompt bullet: `Use \`moving_object_check\` when a transient could be an
asteroid, and \`solar_system_ephemeris\` for solar-system positions/
visibility.`

## Unit tests

1. Horizons: injected fake factory returns canned table (astropy Table with
   datetime_str/RA/DEC/delta/r/V/elong/alpha) → normalization; V missing but
   Tmag present → v_mag from Tmag; >400 rows → thinned + warning; factory
   raising ValueError("Ambiguous target ... matches: ...") → success False
   with that text.
2. SkyBoT: fake requests.get → assert the `-ep` param is a JULIAN DATE
   (float-parseable string, NOT ISO with a `T`), and `-ra/-dec/-rd/-mime/
   -output` present (incl. clamp of radius 30→10 + warning); canned JSON
   ARRAY with real keys (`"RA (hms)":"23 59 59.0"`, `"DEC (dms)":
   "-00 01 00"`, `"VMag (mag)"`, `"Err (arcsec)"`, `"d (arcsec)"`, `"Num"`,
   `"Name"`) normalizes to the v1 row schema with ra_deg/dec_deg in degrees
   and sep_arcsec from `d (arcsec)`; unnumbered record (`"Num"` absent) →
   number None; `{"flag": -1, "message": ...}` and `[]` BOTH → count 0 with
   warning; HTTP 500 → success False.
3. Default epoch: with epoch=None the `-ep` param parses as a float JD and
   provenance records the ISO time.

## Smoke expectations

- `horizons_ephemeris("Ceres", "2026-07-03", "2026-07-08")` → 5-6 rows,
  v_mag 6.5-9.5, delta_au 1.5-4.5.
- `skybot_cone(180.0, 0.0, 5.0)` at current epoch (a busy ecliptic field —
  verified live to return hundreds of main-belt asteroids) → success True
  AND count > 0. Print the first few names/classes.

## Chat acceptance question

"Could the uncatalogued moving source I see near RA 30, Dec -5 tonight be a
known asteroid?"

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done ticked
- [ ] Unit tests green; smoke passes; wiring complete (2 tools)
