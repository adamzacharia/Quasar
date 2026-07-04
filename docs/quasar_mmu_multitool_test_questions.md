# Quasar Chat-Level Test Questions — MMU + Multi-Tool Workflows

Curated questions to paste into the chat UI and judge the **end-to-end answer**, not unit
tests. Complements `quasar_test_questions.md` (ALminer-focused). Every "expected" row count
below was **live-verified on 2026-07-02** by direct `MMUHatsService` probes against the real
Hugging Face catalogs — deviations of ±a few rows are fine; zeros where a number is promised
are not.

## How to judge a pass

For every question, a passing answer must:

1. **Route to the right tool** — MMU questions hit `search_mmu_hats_catalog` /
   `crossmatch_mmu_hats_catalogs`; "full Gaia DR3" questions hit Data Lab (`gaia_dr3`), not
   the MMU subset.
2. **Render a data card** (except deliberate-empty cases) and *not* paste raw rows as text.
3. **Surface warnings honestly** — empty MMU results must be explained as subset coverage
   holes, not "no such sources exist" and not an error.
4. **Never dump a raw provider error** (`Error with Responses API`, litellm 400s, harmony
   channel markup in the answer body). That is an automatic fail — file it.
5. Keep coordinates/units scientifically sane (arcsec vs arcmin vs deg — a known LLM slip).

**Latency note:** the first query touching a new MMU catalog/region downloads HATS
partitions from Hugging Face — 30 s to ~5 min (DESI ≈ 222 s, TESS ≈ 292 s cold; ~30 s
cached). Don't score a slow first answer as a hang.

Verified anchor coordinates used throughout:

| Field | RA (deg) | Dec (deg) |
|---|---|---|
| M87 | 187.70593 | +12.39112 |
| Pleiades | 56.75 | +24.12 |
| COSMOS | 150.10 | +2.18 |
| TESS southern CVZ | 90.0 | −66.56 |

---

## A. MMU single-catalog (verified coverage)

1. **"Search the Multimodal Universe Gaia catalog within 10 arcminutes of the Pleiades and
   show G magnitudes and parallaxes."**
   Expect: hundreds of rows (150–690 verified), flat `phot_g_mean_mag` / `parallax` columns
   (struct expansion working), parallaxes clustering near ~7.4 mas (Pleiades distance).

2. **"Find SDSS spectroscopic sources within 10 arcmin of M87 and list their redshifts and
   velocity dispersions."**
   Expect: ~14 rows, `Z` around 0.003–0.005 for Virgo members, `VDISP` populated.

3. **"Search the MMU Chandra spectra catalog within 10 arcmin of M87 and show fluxes and
   hardness ratios."**
   Expect: ~1,164 rows (M87's X-ray binaries + jet knots), `flux_aper_b` and `hard_*`
   columns. The best "wow" query for M87.

4. **"Query the MMU DESI EDR catalog within 20 arcmin of RA 150.1, Dec 2.18 and summarize
   the redshift distribution."**
   Expect: ~1,956 rows (COSMOS SV3 rosette), a sensible z summary (bulk at z ≲ 1.5), and
   the answer should note rows were truncated to the row cap if max_rows < 1956.

5. **"List TESS SPOC light-curve targets within 1 degree of RA 90, Dec −66.5."**
   Expect: ~685 rows (southern continuous viewing zone). Flat columns are only
   object_id/ra/dec — the answer should not hallucinate magnitudes.

6. **"What Multimodal Universe catalogs can you search?"**
   Expect: `list_mmu_hats_catalogs` → the 5 catalogs (Gaia, DESI, SDSS, TESS SPOC, Chandra)
   with the "ML subsets, not full surveys" caveat stated.

## B. Robustness / negative controls (empty is the *correct* answer)

7. **"Search MMU Gaia within 2 arcmin of M87."**
   Expect: **0 rows + coverage-hole warning**, gracefully. M87 sits in a real hole of the
   Gaia BP/RP bright-star subset (verified empty out to ≥10 arcmin; nearest sources ~0.42°
   away). Fail if it errors, or if it claims "there are no Gaia stars there" without the
   subset caveat. Bonus pass: it offers Data Lab full DR3 as the fallback.

8. **"Search the MMU TESS catalog around the Pleiades."**
   Expect: 0 rows + hole warning (verified). Bonus: suggests MAST for real TESS products.

9. **"Search MMU Gaia around the Pleiades with a 5 degree radius."**
   Expect: radius clamped to 3600 arcsec with an explicit clamp warning relayed in the
   answer — not a 5° scan, not silence about the clamp.

10. **"Crossmatch MMU Gaia with MMU DESI at the COSMOS field, 20 arcmin."**
    Expect: few/zero matches **with the astrophysical explanation** (bright Galactic stars
    × faint extragalactic targets barely overlap). Zero-with-explanation = pass;
    zero-as-confusion or error = fail.

## C. MMU crossmatch (margin cache path)

11. **"Crossmatch MMU SDSS with MMU Chandra within 10 arcmin of M87 using a 5 arcsec match
    radius."**
    Expect: **1 match** (verified) — an SDSS spectrum with an X-ray counterpart. Suffixed
    columns (`_sdss`, `_chandra_spectra`), canonical ra/dec present, data card renders.

12. **"Same crossmatch but with a 30 arcsec match radius."**
    Expect: match radius clamped to 10 arcsec (margin-cache limit) with a warning that the
    answer states plainly.

## D. MMU × Data Lab (subset vs full catalog — the original failure case)

13. **"Find ALMA data for M87 and Gaia sources in the field."** *(the question that broke
    on 2026-07-02: gpt-oss-120b emitted a JSON comment in the Data Lab fallback call →
    raw litellm 400 in chat)*
    Expect: ALMA results card (~436 results) **and** Gaia sources via **Data Lab
    `gaia_dr3`** (full DR3 — no hole at M87). Trying MMU Gaia first is fine *if* it then
    pivots on the hole warning. Regression check: no raw provider error.

14. **"Compare: how many Gaia sources within 10 arcmin of M87 in the full DR3 catalog vs
    the Multimodal Universe Gaia subset?"**
    Expect: Data Lab count (hundreds; full DR3 has no hole) vs MMU 0, with the correct
    interpretation (subset ≠ sky truth).

15. **"Get MMU Gaia stars within 10 arcmin of the Pleiades and plot a color–magnitude
    diagram."**
    Expect: real CMD (BP−RP vs G) showing the cluster sequence. Acceptable route: pivot to
    `datalab_color_magnitude_diagram` on `gaia_dr3` for the same field — the plot is the
    deliverable, the answer should say which catalog it used.

## E. MMU × ALMA archive × spectral lines

16. **"Use MMU SDSS to get the redshift of the brightest galaxy near M87, then check which
    ALMA band covers redshifted CO(2−1) at that redshift."**
    Expect: z ≈ 0.004 from the SDSS rows, CO(2−1) 230.538 GHz → ~229.6 GHz observed →
    **Band 6**, via the line-coverage tools (not from memory).

17. **"Find ALMA observations of M87, then list the X-ray point sources in the same field
    from the Multimodal Universe."**
    Expect: ALMA card + MMU Chandra card (~1,164 sources at 10 arcmin), and a coherent
    joint summary (jet/ISM in radio vs X-ray binary population).

## F. MMU × literature

18. **"Search the MMU Chandra catalog around M87, then find recent papers on M87's
    X-ray binary population and summarize how my source count compares."**
    Expect: MMU card + ADS search (`search_papers`) with real citations — no fabricated
    references; the comparison must use the actual returned row count.

19. **"What is the Multimodal Universe dataset and who built it? Cite the paper."**
    Expect: literature/web tools → the MMU paper (Audenaert et al. / UniverseTBD,
    NeurIPS 2024 Datasets & Benchmarks); the answer should distinguish the HF datasets
    Quasar queries from the paper itself.

## G. Grand tour (one prompt, many subsystems)

20. **"I'm writing a proposal on M87. Give me: available ALMA band coverage, the X-ray
    source population from the Multimodal Universe, SDSS redshifts of nearby galaxies, and
    3 recent papers on the jet — then estimate the Band 6 continuum sensitivity for a
    2-hour integration."**
    Expect (order-independent): ALMA archive summary, MMU Chandra (~1,164) + SDSS (~14)
    cards, real ADS citations, and `calculate_alma_sensitivity` output with stated
    assumptions. This is the conductor stress test — partial silent drops of a requested
    item = fail.

---

## Scorecard template

| # | Routed right tool | Data card | Warnings honest | No raw errors | Science sane | Notes |
|---|---|---|---|---|---|---|
| 1 | | | | | | |

Resolved (2026-07): gpt-oss-120b occasionally emits malformed tool-call JSON (arithmetic or
comments inside arguments) → used to surface as a raw `Error with Responses API … Failed to
parse tool call` dump. The harness now (a) instructs gpt-oss models to emit strict-JSON tool
arguments, (b) retries the turn once on a tool-call parse 400, and (c) never returns raw
provider payloads to chat (`_user_facing_provider_error`). If a sanitized provider-error
message still appears, score it as a *harness* failure and file it.
