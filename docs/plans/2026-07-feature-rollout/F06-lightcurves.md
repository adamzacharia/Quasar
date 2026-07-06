# F06 — Light curves + periodicity suite (v1)

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the exemplars it
names BEFORE coding. Dependency `lightkurve` is pre-installed in Phase 0.

## Objective

TESS/Kepler light curves via lightkurve, plus a generic Lomb-Scargle period
search + phase-fold that works on BOTH TESS/Kepler fluxes and ZTF magnitudes
(reusing the existing `services/alerce_client.AlerceClient.light_curve`).

OUT OF SCOPE (v2): ZTF forced photometry, ASAS-SN, multi-sector stitching
beyond simple `stitch()`, period uncertainty beyond peak-width estimate.

## Files

- CREATE `services/lightcurve_suite.py`
- CREATE `tests/unit/test_lightcurve_suite.py`
- CREATE `scripts/smoke/smoke_f06_lightcurves.py`
- EDIT `core/agent.py` (4 anchors)

## Design

```python
class LightCurveSuite:
    def __init__(self, *, plotting_service=None, alerce_client=None,
                 search_fn=None, cache_dir=None):
        # search_fn: injectable for lightkurve.search_lightcurve (tests)
        # alerce_client: injectable AlerceClient (tests); default lazy
        #   construction from services.alerce_client
        # cache_dir: set lightkurve cache once, lazily, via
        #   os.environ.setdefault("LIGHTKURVE_CACHE_DIR",
        #       <repo>/data/lightkurve_cache) BEFORE importing lightkurve

    # VERIFIED LIVE 2026-07-03: lk.search_lightcurve('Pi Mensae',
    # mission='TESS') -> 108 rows in ~70 s (MAST is slow; smoke must allow
    # >=120 s). res.table columns include: target_name, author (e.g. 'SPOC'),
    # mission (e.g. 'TESS Sector 01'), exptime AND t_exptime (seconds),
    # distance (arcsec), year. Map exptime_s from 'exptime' (fallback
    # 't_exptime').
    def search_space_lightcurves(self, target, mission=None,
                                 max_rows=20) -> dict:
        # lazy: import lightkurve as lk; res = (search_fn or
        #   lk.search_lightcurve)(target, mission=mission)
        # mission in {None,"TESS","Kepler","K2"}; rows from res.table:
        # {"index", "mission", "year", "author", "exptime_s",
        #  "target_name", "distance_arcsec"}
        # Empty -> success True, count 0, warning suggesting name variants.

    def fetch_lightcurve_arrays(self, target, mission=None, index=0):
        # internal helper: search -> res[index].download()
        # -> lc.remove_nans().normalize() for flux
        # returns (time_days: np.ndarray (BTJD/BKJD as-is),
        #          value: np.ndarray, value_kind: "flux",
        #          meta: {mission, author, exptime, label, sector_or_quarter})

    def plot_space_lightcurve(self, target, mission=None, index=0) -> dict:
        # PNG per CONVENTIONS plotting rules; title = f"{target} ({meta})";
        # returns {"success", "path", "n_points", "time_span_days",
        #          "provenance", "warnings"}

    def period_search(self, source, identifier, mission=None,
                      min_period_d=0.05, max_period_d=30.0,
                      fid=None) -> dict:
        # source: "tess"/"kepler" -> fetch_lightcurve_arrays (value=flux)
        #         "ztf"           -> alerce_client.light_curve(identifier);
        #             use detections with magpsf; if fid given filter to it,
        #             else use the band with the most points (warn which).
        #             value_kind="mag"
        # astropy.timeseries.LombScargle(time, value):
        #   freqs: autopower with minimum_frequency=1/max_period_d,
        #          maximum_frequency=1/min_period_d
        #   best_period_d = 1/f_peak
        #   fap = ls.false_alarm_probability(power.max())
        #   period_unc_d: half-width of the peak at 0.5*peak power
        #     (simple scan around the peak; None if indeterminate)
        #   top_periods: top 3 distinct peaks (separated by >5 freq bins)
        # PHASE-FOLD plot: two panels (periodogram with peak marked;
        #   folded LC at best period, mag axis inverted when value_kind=="mag")
        # returns {"success", "path", "best_period_d", "period_unc_d",
        #          "fap", "top_periods": [...], "n_points", "value_kind",
        #          "provenance", "warnings"}
        # n_points < 20 -> warning 'few points; period unreliable'
```

Numerical care: for LS on magnitudes DO NOT normalize; for flux use the
normalized flux. Strip NaN/inf everywhere. Time arrays must be float days
(TESS BTJD is fine as-is; ZTF uses MJD — fine as-is; only relative spacing
matters for LS).

## agent.py wiring

Tools:
1. `search_space_lightcurves` (archive) — table columns
   `["index","mission","year","author","exptime_s","target_name",
   "distance_arcsec"]`, source `"MAST via lightkurve"`.
   Description: "List available TESS/Kepler/K2 light curves for a target
   (use the returned index with plot_space_lightcurve / period_search)."
2. `plot_space_lightcurve` (analysis) — image result via
   `_datalab_attach_image_result(result, f"{mission or 'TESS/Kepler'} light
   curve: {target}")`.
   Description: "Download and plot a TESS/Kepler/K2 light curve for a
   target."
3. `period_search` (analysis) — image result (the two-panel figure) with
   caption `f"Period search: {identifier} — P={best_period_d:.6g} d
   (FAP={fap:.2g})"`. Args: source (enum tess|kepler|ztf, required),
   identifier (required), mission, min_period_d, max_period_d, fid.
   Description: "Lomb-Scargle period search + phase-folded plot on a TESS/
   Kepler light curve or a ZTF object (by ALeRCE oid). Returns best period,
   FAP, and top alternative periods."

Status labels: `"search_space_lightcurves": "Searching TESS/Kepler light
curves"`, `"plot_space_lightcurve": "Plotting space light curve"`,
`"period_search": "Running Lomb-Scargle period search"`.

Prompt bullet: `For variability: \`search_space_lightcurves\` /
\`plot_space_lightcurve\` (TESS/Kepler) and \`period_search\` (works on TESS
targets AND ZTF oids) — always report the FAP with any period.`

## Unit tests (offline — inject search_fn and alerce_client fakes)

1. `search_space_lightcurves`: fake search result table (3 rows) →
   normalized rows with index; empty → count 0 + warning.
2. `period_search(source="ztf")`: fake alerce_client returning a synthetic
   sinusoid in magpsf (P=1.7 d, 60 points, two fids where fid 1 has more
   points) → best_period_d within 1% of 1.7 (or a 1/P alias — assert the
   folded plot file exists and best in {1.7, 0.85, 3.4} within 1%; use a
   clean sinusoid and a tight freq grid so 1.7 wins), value_kind=="mag",
   fid-selection warning mentions fid 1.
3. `period_search(source="tess")` with a stubbed `fetch_lightcurve_arrays`
   (monkeypatch method) sinusoid in flux (P=3.21 d) → recovered within 1%;
   plot path endswith .png (PLOT_OUTPUT_DIR monkeypatched).
4. <20 points → warning present; all-NaN → success False informative.
5. source="ztf" with fid filter honored (only that fid's points used).

## Smoke expectations

- `search_space_lightcurves("Pi Mensae", mission="TESS")` → count ≥ 5.
- `period_search("tess", "Pi Mensae")` → succeeds; n_points > 1000 (any
  period is fine — Pi Men b's 6.27 d transit is shallow; do not assert it).
- `period_search("ztf", <oid>)` where the smoke script first cone-searches
  ALeRCE around (255.0, 11.0, r=300") and picks the oid with most
  detections → succeeds or cleanly reports too-few-points.
First TESS download can take ~1-2 min.

## Chat acceptance question

"Is there TESS data for AU Mic, and can you check it for periodicity?"

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done ticked
- [ ] Unit tests green; smoke passes; wiring complete (3 tools)
