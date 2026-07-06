# F10 — Distances done right (Bailer-Jones, NED-D, velocity frames)

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the exemplars it
names BEFORE coding.

## Objective

Three chronically hand-done distance tasks as tools: (a) Bailer-Jones (2021)
geometric/photogeometric distances for Gaia sources near a position; (b) NED
redshift-independent distances (NED-D) for a named galaxy; (c) velocity-frame
corrections (heliocentric → GSR / Local Group / CMB) with Hubble-flow
distances.

## Files

- CREATE `services/distance_service.py`
- CREATE `tests/unit/test_distance_service.py`
- CREATE `scripts/smoke/smoke_f10_distances.py`
- EDIT `core/agent.py` (4 anchors)

## Part A — Gaia / Bailer-Jones via ESA TAP

Endpoint: `https://gea.esac.esa.int/tap-server/tap/sync` (env
`GAIA_TAP_BASE_URL`, `GAIA_TAP_TIMEOUT`, default 60 s). Plain
`requests.post` (NOT astroquery) with form data:
`REQUEST=doQuery, LANG=ADQL, FORMAT=json, QUERY=<adql>`.

ADQL (interpolate floats only — validate ra/dec/radius numerically first;
clamp radius_arcsec to ≤300 with warning, max_rows ≤50):

```sql
SELECT TOP {max_rows}
  g.source_id, g.ra, g.dec, g.parallax, g.parallax_error,
  g.phot_g_mean_mag, g.pmra, g.pmdec,
  d.r_med_geo, d.r_lo_geo, d.r_hi_geo,
  d.r_med_photogeo, d.r_lo_photogeo, d.r_hi_photogeo,
  DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', {ra}, {dec})) AS sep_deg
FROM gaiadr3.gaia_source AS g
JOIN external.gaiaedr3_distance AS d ON g.source_id = d.source_id
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
ORDER BY sep_deg ASC
```

Response JSON: `{"metadata": [{"name": ...}, ...], "data": [[...], ...]}` —
zip names with row values. Row schema out: source_id (str — int64 must not
lose precision in JSON), ra, dec, parallax_mas, parallax_err_mas, g_mag,
pm_ra, pm_dec, r_geo_pc (=r_med_geo), r_geo_lo_pc, r_geo_hi_pc,
r_photogeo_pc (+lo/hi), sep_arcsec (=sep_deg*3600, rounded 3 dp).
Add warning when parallax/parallax_error < 5 ('low-significance parallax:
prefer r_photogeo / treat as uncertain').

Method: `gaia_distances(ra, dec, radius_arcsec=10, max_rows=10) -> dict`.

## Part B — NED-D via direct nDistance endpoint (NOT astroquery)

CORRECTED 2026-07-03 after live testing: astroquery 0.4.11's NED
`SEARCH_TYPE` dict has NO `'distances'` key — `Ned.get_table(table=
"distances")` raises `KeyError: 'distances'`. Use NED's own nDistance
endpoint and parse the HTML table instead:

```python
def _load_ned_distance_table(self, target):
    import pandas as pd, requests   # lazy
    r = requests.get("https://ned.ipac.caltech.edu/cgi-bin/nDistance",
                     params={"name": target}, timeout=self.timeout)
    r.raise_for_status()
    tables = pd.read_html(r.text)   # returns list[DataFrame]
    # pick the table whose columns include a distance-modulus or Mpc column
    # (verified live: NGC 253 -> cols include 'Distance Modulus (mag)' and
    #  'Metric Distance (Mpc)'). Raise a clear error if none match.
```

NED's nDistance is genuinely slow/flaky (frequent read timeouts); keep the
default `timeout` generous (>=45 s) and let a timeout surface as
`success: False` with an informative message (the outer try/except handles
it). `_normalize_ned_table` must accept BOTH a pandas DataFrame (default
path) AND an astropy Table (so the injected `ned_table_loader` test double
keeps working) — duck-type: if it has `.colnames`, call `.to_pandas()`.

Column names in that table vary between astroquery versions — normalize by
scanning `tbl.colnames` case-insensitively: modulus col contains "modulus"
but not "err"/"uncert"; its error col contains both; method col contains
"method"; refcode col contains "refcode". Rows out: {"dist_mpc"
(=10**((mu-25)/5), 4 sig figs), "dist_modulus", "dist_modulus_err",
"method", "refcode"}. Drop masked/NaN moduli. Also return "summary":
{"n", "median_mpc", "min_mpc", "max_mpc", "n_methods"}. Constructor takes
`ned_table_loader=None` injectable (callable(name) -> astropy Table) so
tests never import astroquery.

## Part C — velocity frames (pure astropy/numpy, no network)

`velocity_frames(ra, dec, v_helio_kms=None, z=None) -> dict` — exactly one
of v_helio_kms / z required (z → v = c·((1+z)²−1)/((1+z)²+1), relativistic).

Apex corrections, formula
`v_frame = v_helio + V_apex*(sin(b)sin(b_a) + cos(b)cos(b_a)cos(l−l_a))`
with (l, b) Galactic coords of the target (astropy SkyCoord):

| frame | V_apex (km/s) | l_apex | b_apex | source (cite in comment) |
|---|---|---|---|---|
| GSR | 232.3 | 87.8 | 1.7 | NED convention |
| Local Group | 316.0 | 93.0 | -4.0 | Karachentsev & Makarov 1996 |
| CMB | 369.82 | 264.021 | 48.253 | Planck 2018 dipole |

Rows: [{"frame": "heliocentric"|"GSR"|"LocalGroup"|"CMB", "v_kms",
"d_hubble_mpc" (=v/H0, Planck18 H0=67.66; None when v ≤ 0)}]. Warning when
|v_CMB| < 3000: 'peculiar velocities significant; Hubble-flow distance
unreliable'.

## agent.py wiring

Tools:
1. `gaia_distance` (archive) — `_gaia_distance(target_name=None, ra=None,
   dec=None, radius_arcsec=10, max_rows=10)`; table card columns
   `["source_id","g_mag","parallax_mas","r_geo_pc","r_geo_lo_pc",
   "r_geo_hi_pc","r_photogeo_pc","sep_arcsec"]`, source
   `"ESA Gaia DR3 x Bailer-Jones EDR3 distances"`.
   Description: "Bailer-Jones (2021) geometric/photogeometric distances for
   Gaia DR3 sources near a position — the correct way to turn parallax into
   distance for stars."
2. `ned_distance` (archive) — `_ned_distance(target_name)` required;
   table card columns `["dist_mpc","dist_modulus","dist_modulus_err",
   "method","refcode"]`, source `"NED-D redshift-independent distances"`,
   filter_label includes the summary median.
   Description: "NED redshift-independent distance measurements (Cepheids,
   TRGB, SNIa, ...) for a named galaxy, with median summary."
3. `velocity_frame_distance` (analysis) — `_velocity_frame_distance(
   target_name=None, ra=None, dec=None, v_helio_kms=None, z=None)`;
   table card columns `["frame","v_kms","d_hubble_mpc"]`, source
   `"Velocity-frame corrections (NED/Planck conventions)"`.
   Description: "Convert a heliocentric velocity or redshift to GSR, Local
   Group, and CMB frames and give Hubble-flow distances (Planck18 H0) — use
   for nearby-galaxy distances and flow corrections."

Status labels: `"gaia_distance": "Querying Gaia/Bailer-Jones distances"`,
`"ned_distance": "Fetching NED-D distances"`,
`"velocity_frame_distance": "Computing velocity-frame corrections"`.

Prompt bullet: `For distances: \`gaia_distance\` (stars, parallax),
\`ned_distance\` (galaxies, redshift-independent), and
\`velocity_frame_distance\` (flow-corrected Hubble distances) — do not
compute 1/parallax by hand.`

## Unit tests

1. Part A: fake `requests.post` → assert form fields (REQUEST/LANG/FORMAT),
   ADQL contains CIRCLE with the right numbers; canned metadata/data JSON →
   row normalization incl. source_id-as-string and sep_arcsec math; low
   parallax S/N warning fires.
2. Part A: radius clamp warning; HTTP 400 → success False.
3. Part B: injectable loader returns a fake astropy Table with awkward
   column names (e.g. "Distance Modulus", "DM Err", "Method", "Refcode") →
   normalization + median summary; masked row dropped.
4. Part C: known-value check — for (l≈264, b≈48)-ish target the CMB
   correction is near +V_apex: use ra/dec of the CMB apex itself and assert
   v_CMB ≈ v_helio + 369.82 (tolerance 0.5). z→v conversion: z=0.01 →
   v≈2982.6 km/s (tol 1). Warning below 3000 km/s fires. Errors when both/
   neither of v and z given.

## Smoke expectations

- Barnard's Star field: `gaia_distances(269.44850, 4.73780, 30)` → nearest
  row parallax ≈ 546–548 mas, r_geo_pc ≈ 1.83 (accept 1.80–1.86).
- `ned_distances("NGC 253")` → sane distance (median 2–6 Mpc). NED is flaky:
  retry once, and treat a clean timeout as SKIP (soft pass), NOT failure.
- `velocity_frames(187.70593, 12.39112, v_helio_kms=1284)` (M87) → v_CMB
  ≈ 1611 km/s (verified against NED). Do NOT use NGC 253 for the CMB check —
  near the anti-apex it gives a tiny/negative (but correct) v_CMB.
Exit 0 only if all hold.

## Chat acceptance question

"How far away is Barnard's star according to Gaia, properly?"

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done ticked
- [ ] Unit tests green; smoke passes; wiring complete (3 tools)
