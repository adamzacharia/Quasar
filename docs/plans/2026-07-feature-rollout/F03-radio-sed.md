# F03 — Radio continuum SED + spectral index engine (v1)

Read `docs/plans/2026-07-feature-rollout/CONVENTIONS.md` and the exemplars it
names BEFORE coding. This is the highest-science-value task in the rollout;
scientific honesty (flags, not silent corrections) is the design principle.

## Objective

Compile a compact-source radio continuum SED across major surveys, fit a
power-law spectral index α (S ∝ ν^α), and PLOT it — flagging (not
"correcting") resolution/epoch caveats.

v1 scope: catalog fluxes for compact sources only. NO resolution matching,
NO flux-scale corrections, NO image-plane photometry.

## Files

- CREATE `services/radio_sed.py`
- CREATE `tests/unit/test_radio_sed.py`
- CREATE `scripts/smoke/smoke_f03_radio_sed.py`
- EDIT `core/agent.py` (4 anchors)

## Survey registry

Query via VizieR CONE SEARCH protocol (uniform across catalogs; no
per-catalog RA/Dec column knowledge needed):
`https://vizier.cds.unistra.fr/viz-bin/conesearch/{catalog_path}?RA={ra}&DEC={dec}&SR={sr_deg}&VERB=2`
(env `VIZIER_CONESEARCH_BASE_URL`, `RADIO_SED_TIMEOUT` default 30 s per
catalog). Response is a VOTable — parse with
`astropy.io.votable.parse_single_table(io.BytesIO(resp.content))` (lazy
import astropy.io.votable). Field NAMES in the VOTable match the column
candidates below.

```python
# VERIFIED LIVE 2026-07-03 via VizieR conesearch at 3C 273 / Cyg A. GLEAM
# fluxes are in Jy (scale 1000 -> mJy); the rest are mJy. Nearest row: VERB=2
# adds a `_r` column (arcmin from cone centre) — use it, DON'T recompute.
SURVEY_REGISTRY = [
  # id      vizier path              freq_MHz  flux cols          unit->mJy  res"  epoch  err cols
  ("TGSS",  "J/A+A/598/A78/table3",    150.0,  ["Stotal"],           1.0,   25.0, 2016,  ["e_Stotal"]),
  ("GLEAM", "VIII/100/gleamegc",       200.0,  ["Fintwide"],      1000.0,  120.0, 2014,  ["e_Fintwide"]),
  ("SUMSS", "VIII/81B/sumss212",       843.0,  ["St"],               1.0,   45.0, 2003,  ["e_St"]),
  ("NVSS",  "VIII/65/nvss",           1400.0,  ["S1.4"],             1.0,   45.0, 1995,  ["e_S1.4"]),
  ("FIRST", "VIII/92/first14",        1400.0,  ["Fint"],             1.0,    5.0, 2000,  []),
]
```

CONFIRMED WORKING (rows returned, columns exact): TGSS `Stotal`, GLEAM
`Fintwide` (Jy→mJy), SUMSS `St`, NVSS `S1.4`, FIRST `Fint`. These 5 span
150 MHz–1.4 GHz — a full decade, enough for a real spectral index.

DROPPED FROM v1 (verified broken/unreliable via VizieR 2026-07-03, DO NOT add):
- **VLASS** `J/ApJS/255/30/comp`: cone search AND bounding-box AND TAP all
  return 0 rows for 3C 273 and even Cyg A (brightest northern source). The
  table has data elsewhere but bright compact sources are absent at these
  positions — VizieR's copy is not usable for this. 3 GHz VLASS needs a
  direct CIRADA/NRAO query — defer to v2.
- **LoTSS DR2** `J/A+A/659/A1/*`: every conesearch path variant returns
  "No table found in VOTABLE file". Defer to v2 (needs the correct table id
  or the LoTSS TAP service).

RADIUS: default `radius_arcsec=30` (NOT 15 — verified 15″ misses TGSS/GLEAM
matches for 3C 273 that appear at 30–60″). Clamp to ≤120.

RESILIENCE (still required): if any catalog path 404s or its flux column is
absent, SKIP it with a warning naming what was tried — never fail the whole
compile. If a future VizieR change breaks one of the 5, the smoke will catch
it; fix the REGISTRY (and this spec) then.

## Class and methods

```python
class RadioSedService:
    def __init__(self, *, base_url=None, timeout=None, plotting_service=None,
                 fetcher=None):
        # fetcher: injectable callable(url, params, timeout) -> bytes
        #          (VOTable content) for tests

    def compile_sed(self, ra, dec, radius_arcsec=15.0) -> dict:
        # clamp radius to <= 60 (warn). For EACH registry survey:
        #   cone search with SR = radius_arcsec/3600
        #   parse VOTable -> astropy table; if 0 rows: skip silently
        #   pick NEAREST row: VizieR adds a "_r" field (arcmin) when VERB=2;
        #     if absent compute separation from the table's first pair of
        #     columns whose UCD contains 'pos.eq.ra'/'pos.eq.dec' (VOTable
        #     field.ucd) — defensive
        #   flux: first candidate column present and unmasked; scale to mJy
        #   err: first err candidate present, else None
        # points: [{"survey", "freq_mhz", "flux_mjy", "flux_err_mjy",
        #           "sep_arcsec", "resolution_arcsec", "epoch"}]
        # per-survey failures -> warnings, never exceptions
        # returns {"success": True (even with 0 points — the fit tool
        #   handles that), "points", "count", "warnings", "provenance"}

    def fit_spectral_index(self, points) -> dict:
        # need >= 2 points; weighted least squares on log10(S) vs log10(nu):
        #   weights from flux errors when >=70% of points have errors,
        #   else unweighted (warn)
        # alpha = slope; alpha_err from covariance; also chi2_red when
        #   weighted and n>2
        # FLAGS (list[str], each a full sentence):
        #   - resolution mismatch: max/min resolution_arcsec > 3 ->
        #     "Beam sizes span X-Y arcsec; extended sources may be resolved
        #      out in the high-resolution surveys."
        #   - epoch spread: max-min epoch > 5 yr -> variability caveat
        #   - both 1.4 GHz points (NVSS & FIRST) present and differ by >30%:
        #     "NVSS/FIRST 1.4 GHz fluxes differ by N% - resolution or
        #      variability."
        #   - any point >3 sigma from the fit -> name the survey
        # returns {"success", "alpha", "alpha_err", "chi2_red", "n_points",
        #          "s_1400_mjy_predicted" (from the fit), "flags", "warnings"}

    def plot_sed(self, points, fit=None, title="Radio SED") -> dict:
        # log-log errorbar plot; survey labels annotated at each point;
        # if fit given, draw the power law + "alpha = X +/- Y" in the
        # legend; PNG per CONVENTIONS; returns {"success", "path", ...}
```

## agent.py wiring

ONE tool that composes all three (category `"analysis"`):

`radio_sed` — wrapper `_radio_sed(target_name=None, ra=None, dec=None,
radius_arcsec=15.0)`:
1. resolve position; `compile_sed`
2. 0 points → return `{"success": True, "note": "No radio catalog
   detections within radius.", "warnings": [...]}`.
3. ≥2 points → `fit_spectral_index`; `plot_sed` with fit and title
   `f"Radio SED: {label}"`
4. return `_datalab_attach_image_result(plot_result, caption, meta)` where
   caption = `f"Radio SED: {label} — α = {alpha:.2f} ± {alpha_err:.2f}
   ({n} surveys)"` and the returned dict ALSO carries "points", "alpha",
   "alpha_err", "flags", "warnings" (merge keys into the attach result
   before returning) so the model can discuss the numbers.
5. 1 point → skip fit, still plot, note single-frequency.

Description: "Compile a radio continuum SED (TGSS/GLEAM/LoTSS/SUMSS/NVSS/
FIRST/VLASS catalog fluxes), fit the spectral index alpha (S~nu^alpha), and
plot it with honesty flags for resolution/epoch mismatches. Compact sources
only."

Status label: `"radio_sed": "Compiling radio SED + spectral index"`.

Prompt bullet: `Use \`radio_sed\` for radio spectral index / radio SED
questions; ALWAYS repeat its flags (resolution/epoch caveats) in your
answer.`

## Unit tests (offline — inject fetcher returning canned VOTables)

Build tiny VOTable fixtures in-code (string templates with FIELD
name/ucd/datatype + TABLEDATA) — one per mocked survey.

1. compile: 3 surveys return one row each (NVSS 1000 mJy, TGSS 2000 mJy,
   FIRST 700 mJy), GLEAM returns Jy (2.0 → 2000 mJy scale check), one
   registry entry 404s (warning naming it), one has no flux column
   (warning) → 4 points, correct scaling, nearest-row logic via the `_r`
   column when a survey returns 2 rows (_r 0.1 vs 0.5).
2. fit: synthetic power law α=−0.7 with 2% errors across 4 freqs →
   alpha within 0.05; α_err > 0; unweighted path warns when errors missing.
3. flags: NVSS 1000 vs FIRST 500 at 1.4 GHz → percent-difference flag;
   resolution span 5" (FIRST) vs 120" (GLEAM) → resolution flag; epochs
   1995 vs 2016 → epoch flag.
4. fit with 1 point → success False informative; 0-point compile →
   success True count 0.
5. plot: PLOT_OUTPUT_DIR monkeypatched; path endswith .png; runs with and
   without fit.

## Smoke expectations (3C 273: ra=187.27792, dec=2.05239, radius_arcsec=30)

- compile_sed → ≥ 4 points (verified live: TGSS, GLEAM, NVSS, FIRST all
  return; SUMSS is southern so absent for 3C 273). Print all points.
  Sanity: TGSS 150 MHz ≈ 100,000–120,000 mJy; NVSS 1.4 GHz ≈ 40,000–56,000
  mJy (3C 273 is variable — accept a wide band, just assert > 20,000).
- fit → α in [−1.2, 0.3] (3C 273 is flat-ish/variable), flags non-empty
  (epoch spread across 1995–2016 at minimum).
- Print per-survey skips. FAIL only if fewer than 4 of the 5 registry
  catalogs return data for 3C 273 (that would signal a VizieR break to fix
  at the gate). Do NOT expect VLASS/LoTSS — they are intentionally not in
  the v1 registry.

## Chat acceptance question

"What is the radio spectral index of 3C 273, and what should I be careful
about when interpreting it?"

## Acceptance checklist

- [ ] CONVENTIONS definition-of-done ticked
- [ ] Unit tests green; smoke passes with registry fixes fed back into this
      spec if needed; wiring complete (1 tool)
