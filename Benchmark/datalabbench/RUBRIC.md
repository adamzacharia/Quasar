# DataLabBench v1.0 — Scoring Rubric

> GENERATED from `dlb_dataset_v1.py` by `run_datalabbench.py --emit-rubric`.
> Do not edit by hand — edit the dataset and regenerate.

## How scoring works

Every question is worth **100 points**, split across weighted **checkpoints**:

- **auto** checkpoints are scored deterministically against the captured evidence
  (tool-call trace incl. arguments, extracted SQL, response text, emitted images).
  A checkpoint contains N checks; earned credit = `points × passed/N` → **partial
  credit for partially completed steps**.
- **judge** checkpoints are scored by an LLM judge that returns fractional credit
  in [0, 1] per its guidance, grounded in the tool trace (unsupported claims = 0).
- **Penalties** (guardrail violations) subtract points after summing; floor at 0.

**Overall bench score** = tier-weighted mean of question percentages:

| Tier | Weight | Focus |
|---|---|---|
| T1 | 1.0 | Discovery & single trivial query |
| T2 | 1.2 | Single-survey selection / one image |
| T3 | 1.4 | Quality cuts, classification, kinematics |
| T4 | 1.6 | Spatial structure (matched filter, density maps) |
| T5 | 1.8 | Cross-survey combination (server-side TAP joins) |
| T6 | 2.0 | Full TAP + SIA combination workflows |
| T7 | 2.4 | Open-ended research (LLM designs the whole strategy) |

Grades: A+ ≥97, A ≥90, B ≥75, C ≥60, D ≥40, F <40.

## Global guardrail penalties (every question)

| ID | Points | Trigger |
|---|---|---|
| GP-00 | −20 | Fabricated result: the answer presents concrete results but NO Data Lab tool call succeeded during the run. |
| GP-01 | −10 | Unbounded row-level scan: executed SQL selects rows from a catalog with no q3c spatial bound and no key-equality filter (guardrail #1 — a LIMIT alone does not make the scan cheap; row-cap discipline is scored separately as positive criteria). |
| GP-02 | −15 | q3c_join anti-pattern: a statement with a flat q3c_join and no MATERIALIZED CTE reduction of the small side (guardrail #2 — the exact anti-example the PDF says never to run). Evaluated per executed statement. |
| GP-03 | −8 | ra/dec BETWEEN box used as the ONLY spatial bound of a row-level (non-aggregate) query — the q3c functional index cannot serve it (guardrail #1). |

---

## DLB-01 (Tier 1, weight 1.0): Catalog discovery (LMC + near-infrared)

**Prompt:** Which Data Lab catalogs cover the Large Magellanic Cloud, and which of those include near-infrared photometry? List the relevant table names.

**Exercises:** Dataset/schema metadata (no row query). Candidates: nsc_dr2, smash_dr2, vhs_dr5, gaia_dr3.

**Expected actions:** Call schema/metadata listing (qc.schema / dataset catalog), reason over coverage + bandpasses, return a curated table list.

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 20 | Used schema/metadata tooling instead of answering purely from memory (2 checks, each worth 20/2 pts) |
| C2 | auto | 20 | Names the candidate LMC catalogs (VHS + at least one of SMASH/NSC/Gaia) (2 checks, each worth 20/2 pts) |
| C3 | judge | 25 | Near-infrared identification is correct — *Full credit only if VHS (J/H/Ks bands) is identified as the near-infrared catalog. Deduct if NIR capability is wrongly attributed to optical-only surveys (SMASH ugriz, NSC, DES grizY is only barely NIR at Y — attributing NIR to Gaia G/BP/RP is wrong). Half credit if VHS named but bands not stated.* |
| C4 | judge | 20 | Lists concrete TABLE names, not just catalog/schema names — *Full credit for real table names (e.g. vhs_dr5.vhs_cat_v3, smash_dr2.object, nsc_dr2.object, gaia_dr3.gaia_source). Half credit if only schemas listed. Zero if tables are invented (verify against the tool outputs in the trace).* |
| C5 | judge | 15 | Coverage reasoning (why these catalogs cover the LMC) — *Credit for correctly reasoning about footprints: SMASH targets the Magellanic system, VHS covers the southern hemisphere incl. LMC, Gaia is all-sky, NSC is wide-southern. Partial credit for weaker but not wrong reasoning.* |

## DLB-02 (Tier 1, weight 1.0): Cone-search count (Gaia DR3 around Palomar 5)

**Prompt:** How many Gaia DR3 sources lie within 10 arcminutes of Palomar 5 (RA = 229.022, Dec = −0.112)?

**Exercises:** gaia_dr3.gaia_source via TAP; q3c_radial_query.

**Expected actions:** One COUNT(*) with q3c_radial_query; return the number.

**Reference query:**

```sql
SELECT COUNT(*) FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 10.0/60.0)
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 20 | Queried gaia_dr3.gaia_source (1 checks, each worth 20/1 pts) |
| C2 | auto | 20 | Cone centered on Palomar 5 (229.022, -0.112) (1 checks, each worth 20/1 pts) |
| C3 | auto | 15 | Radius = 10 arcmin converted correctly to 0.1667 deg (1 checks, each worth 15/1 pts) |
| C4 | auto | 15 | Count computed server-side (COUNT(*) / cone-count tool), not by pulling rows (1 checks, each worth 15/1 pts) |
| C5 | judge | 15 | A specific integer count is reported as THE answer — *Full credit if a single concrete source count is stated plainly. Half if buried/hedged. Zero if no number or only a code sketch.* |
| C6 | judge | 15 | Reported number is faithful to the tool output — *Compare the stated count against reported_count / rowcount in the tool trace. Full credit iff they match exactly. Zero if the number does not appear in any tool output (fabrication).* |

## DLB-03 (Tier 2, weight 1.2): Color–magnitude diagram (Draco dwarf, NSC DR2)

**Prompt:** Get g and r magnitudes for point sources within 0.4° of the Draco dwarf (RA = 260.06, Dec = +57.92) from NSC DR2 and plot a g vs (g−r) CMD.

**Exercises:** nsc_dr2.object TAP; cone + star/morphology cut; scatter CMD.

**Expected actions:** Cone + class_star/quality cuts -> dataframe -> scatter CMD.

**Reference query:**

```sql
SELECT gmag, rmag, gmag - rmag AS gr FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 260.06, 57.92, 0.4) AND class_star > 0.5 AND gmag < 24 AND rmag < 24
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | Queried nsc_dr2.object (1 checks, each worth 15/1 pts) |
| C2 | auto | 15 | Cone at Draco (260.06, +57.92) with radius 0.4 deg (2 checks, each worth 15/2 pts) |
| C3 | auto | 15 | Point-source (morphology) cut actually applied in the executed query (1 checks, each worth 15/1 pts) |
| C4 | auto | 20 | A CMD plot was actually produced (2 checks, each worth 20/2 pts) |
| C5 | judge | 20 | CMD is correctly constructed (g vs g-r, magnitude axis inverted, g/r from NSC) — *Full credit: y-axis g magnitude (inverted, bright up), x-axis g-r color. Half credit for axes swapped or inversion missing/unstated. Check tool args (mag_band/blue_band/red_band or x_expr/y_expr/invert_y).* |
| C6 | judge | 15 | Sensible depth/quality handling and interpretation — *Credit for magnitude limits (~g,r < 24) or equivalent quality bounds, and for any correct interpretation (Draco's old population / MSTO / RGB visible). Partial credit if only one of the two.* |

## DLB-04 (Tier 2, weight 1.2): Single color cutout (center of M31)

**Prompt:** Show me a color image of the center of M31 from the DECam Legacy Surveys.

**Exercises:** SIA service (coadd_all), 3-band cutout; FOV choice is a decision.

**Expected actions:** Resolve M31 -> choose modest FOV (~0.1-0.2 deg for 'center'; D25~3 deg) -> dec-corrected size -> deepest Stack/image per band -> Lupton RGB. NB: M31 at dec +41 may have little/no DECam coadd coverage — 0 rows is a coverage gap, not a code bug.

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 20 | Used the SIA color-image path (not just text) (1 checks, each worth 20/1 pts) |
| C2 | auto | 15 | Pointed at M31 (10.6847, +41.2687) or resolved it by name (decision check: a coverage-gap failure must not zero it) (1 checks, each worth 15/1 pts) |
| C3 | auto | 15 | FOV chosen for the CENTER (modest, ~0.05-0.5 deg; not the whole 3-deg disk; decision check: counts even if the SIA search finds no coverage) (1 checks, each worth 15/1 pts) |
| C4 | judge | 30 | Correct outcome handling: a color image OR an honest coverage-gap explanation — *Full credit if a color image of M31's center is rendered, OR if the search returned no DECam coverage and the answer explains it as a coverage gap (M31 at dec +41 is at the northern edge of DECam surveys) and offers an alternative. ZERO credit if it claims an image that no tool produced, or blames its own code for what is a coverage gap.* |
| C5 | judge | 10 | FOV reasoning is explicit (center vs. D25 ~ 3 deg) — *Full credit if the answer justifies the FOV from M31's apparent size / the 'center' intent. Half credit for a sensible FOV with no reasoning.* |
| C6 | judge | 10 | Color composition handled correctly — *Credit for correct RGB band assignment (red=i/z, green=r, blue=g) or for the tool's auto-selection being described; Lupton stretch parameters are a plus. No credit if bands are scrambled.* |

## DLB-05 (Tier 3, weight 1.4): Star/galaxy separation + color-color (DES DR1)

**Prompt:** Using DES DR1 around RA = 30, Dec = −50, separate stars from galaxies and show me g−r vs r−i color–color diagrams for each population.

**Exercises:** des_dr1.main TAP; spread_model_r morphology; two-panel CCD.

**Expected actions:** Query with morphology flag + colors -> two-panel color-color diagram.

**Reference query:**

```sql
SELECT mag_auto_g - mag_auto_r AS gr, mag_auto_r - mag_auto_i AS ri, CASE WHEN spread_model_r > 0.003 THEN 'galaxy' ELSE 'star' END AS morph FROM des_dr1.main WHERE q3c_radial_query(ra, dec, 30.0, -50.0, 0.5) AND mag_auto_i BETWEEN 16 AND 23 AND flags_g = 0 AND flags_r = 0 AND flags_i = 0 AND fluxerr_auto_g > 0 AND fluxerr_auto_r > 0 AND fluxerr_auto_i > 0
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | Queried des_dr1.main (1 checks, each worth 15/1 pts) |
| C2 | auto | 10 | Cone near (30, -50) (1 checks, each worth 10/1 pts) |
| C3 | auto | 20 | Morphological star/galaxy split via spread_model (in the executed query/tool args) (2 checks, each worth 20/2 pts) |
| C4 | auto | 15 | Both colors formed: g-r AND r-i (1 checks, each worth 15/1 pts) |
| C5 | auto | 10 | A color-color plot was rendered (1 checks, each worth 10/1 pts) |
| C6 | judge | 15 | Stars and galaxies shown as SEPARATE populations (two panels or clearly split) — *Full credit for a two-panel CCD (stars \| galaxies) or an explicit split. Half if both are in one panel but distinguishable. Zero if populations mixed.* |
| C7 | judge | 15 | Photometric quality cuts applied or discussed — *Credit for flags_g/r/i = 0, fluxerr_auto_* > 0, and a magnitude window (~16 < i < 23) — or the structured tool's documented equivalents. Partial credit proportional to how many of the three appear.* |

## DLB-06 (Tier 3, weight 1.4): Proper-motion + parallax selection (Gaia white dwarfs)

**Prompt:** Find high-proper-motion white-dwarf candidates in Gaia DR3 in a 5°-radius patch of the southern sky (say around RA = 60, Dec = −50): significant parallax, large total proper motion, and absolute magnitudes on the WD sequence. Give me the HR diagram.

**Exercises:** gaia_dr3.gaia_source; pm, bp_rp, parallax, astrometric quality; HR diagram.

**Expected actions:** Cone + real Gaia astrometric-quality cuts -> absolute mag -> HR diagram with WD locus.

**Reference query:**

```sql
SELECT bp_rp, phot_g_mean_mag + 5*LOG10(parallax) - 10 AS abs_g, parallax, pm FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 60.0, -50.0, 5.0) AND parallax_over_error > 4 AND parallax > 0.25 AND ruwe < 1.4 AND ipd_frac_multi_peak <= 2 AND astrometric_sigma5d_max < 1.5 AND pm > 100 AND phot_g_mean_mag + 5*LOG10(parallax) - 10 > 10
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | Gaia DR3 cone at (60, -50) with ~5 deg radius (3 checks, each worth 15/3 pts) |
| C2 | auto | 20 | Real astrometric-quality + kinematic cuts applied in the executed query (3 checks, each worth 20/3 pts) |
| C3 | auto | 15 | Absolute magnitude derived from parallax in the executed query/plot expression (1 checks, each worth 15/1 pts) |
| C4 | auto | 10 | An HR diagram was plotted (2 checks, each worth 10/2 pts) |
| C5 | judge | 20 | White-dwarf selection is physically sound — *Full credit: positive significant parallax, LARGE total PM (~>100 mas/yr), and a faint absolute-magnitude cut (abs G > ~10) placing candidates on the WD sequence — all three present. Deduct ~1/3 for each missing/wrong element.* |
| C6 | judge | 20 | HR diagram correct + WD locus highlighted/interpreted — *Full credit: bp_rp on x, absolute G on y (inverted), WD sequence below the main sequence identified or highlighted. Half for a correct plot without WD identification.* |

## DLB-07 (Tier 4, weight 1.6): Overdensity hunt / matched filter (SMASH field 169)

**Prompt:** Help me look for a stellar overdensity — a possible dwarf companion — in SMASH DR1 field 169. Select blue main-sequence stars and find where they clump on the sky.

**Exercises:** smash_dr1.object; color box; 2D density / Mexican-hat convolution. Field 169 is the Hydra II field.

**Expected actions:** Blue stars over the field -> 2D histogram + difference-of-Gaussians filter -> peak coords + significance.

**Reference query:**

```sql
SELECT ra, dec, gmag, gmag - rmag AS gr FROM smash_dr1.object WHERE fieldid = 169 AND depthflag > 1 AND ABS(sharp) < 0.5 AND gmag - rmag BETWEEN -0.5 AND 0.5 AND gmag BETWEEN 9 AND 25
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | SMASH DR1 object table, field 169 (or the Hydra II cone) (2 checks, each worth 15/2 pts) |
| C2 | auto | 15 | Stellar morphology + blue main-sequence color box in the executed query (2 checks, each worth 15/2 pts) |
| C3 | auto | 15 | Sky-density analysis actually run (server-side aggregate or density tool) (1 checks, each worth 15/1 pts) |
| C4 | auto | 10 | Matched-filter / peak detection actually executed (1 checks, each worth 10/1 pts) |
| C5 | judge | 25 | Recovers the Hydra II overdensity with concrete peak coordinates — *Full credit if a peak is reported within ~0.15 deg of (185.43, -31.99) (Hydra II) with some significance measure. Half credit for a peak list that contains it less prominently, or coordinates without significance. Zero if no concrete peak position is reported.* |
| C6 | judge | 20 | Method quality: binning/kernel/significance explained; density map shown — *Credit for a rendered density map, a sensible bin size (arcmin-scale), a smoothing/matched-filter justification, and a significance estimate. Score proportionally.* |

## DLB-08 (Tier 4, weight 1.6): HEALPix stellar-density map (NSC DR2, Milky Way structure)

**Prompt:** Make a stellar density map of a ~20° × 20° region from NSC DR2 to reveal Milky Way structure — bin by HEALPix and show log counts.

**Exercises:** nsc_dr2 TAP aggregate; precomputed HEALPix column (ring256/nest4096) + COUNT.

**Expected actions:** GROUP BY the HEALPix index server-side (one row per pixel), bounded footprint -> healpy map, log10 counts.

**Reference query:**

```sql
SELECT ring256, AVG(ra) AS ra0, AVG(dec) AS dec0, COUNT(ring256) AS nb FROM nsc_dr2.object WHERE ra BETWEEN 70 AND 90 AND dec BETWEEN -70 AND -50 AND class_star > 0.5 GROUP BY ring256
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | NSC DR2 aggregated server-side (2 checks, each worth 15/2 pts) |
| C2 | auto | 20 | HEALPix binning via a precomputed column, in the executed aggregate (1 checks, each worth 20/1 pts) |
| C3 | auto | 15 | Aggregate returns one row per pixel (COUNT + GROUP BY / healpix mode), not raw rows (1 checks, each worth 15/1 pts) |
| C4 | judge | 15 | Region is a bounded ~20x20 deg footprint (not all-sky, not a tiny cone) — *Full credit for a defined ~400 deg^2 region (e.g. 70<ra<90, -70<dec<-50, or a ~10 deg-radius cone). Half for a materially smaller/larger but still bounded region. Zero for an unbounded all-sky scan.* |
| C5 | auto | 10 | A density map was rendered (2 checks, each worth 10/2 pts) |
| C6 | judge | 15 | Log-scaled counts + true HEALPix pixel geometry — *Full credit: log10(counts) color scale AND a real HEALPix map (healpy-style, pixel shapes) rather than a scatter of pixel centers. Half for one of the two.* |
| C7 | judge | 10 | Milky Way structure interpretation — *Credit for pointing at visible structure (disk gradient, LMC/SMC if in view, globular clusters, dust lanes). Brief but correct = full.* |

- Global penalties disabled here: GP-03 (the reference solution itself uses that pattern legitimately)

## DLB-09 (Tier 5, weight 1.8): Stream tracing via crossmatch (Palomar 5 tidal tails)

**Prompt:** Combine NSC DR2 photometry with Gaia DR3 proper motions around Palomar 5 to trace its tidal tails — select stream stars by proper motion and CMD, then plot their on-sky distribution.

**Exercises:** nsc_dr2.object x gaia_dr3.gaia_source via q3c_join; PM cut + CMD mask. Must follow q3c's execution model (materialize small side; indexed side second).

**Expected actions:** Reduce Gaia (cone + PM cut) in a MATERIALIZED CTE -> q3c_join against NSC (indexed side) -> CMD mask -> on-sky scatter.

**Reference query:**

```sql
WITH g AS MATERIALIZED (SELECT ra, dec, pmra, pmdec, pm FROM gaia_dr3.gaia_source WHERE q3c_radial_query(ra, dec, 229.022, -0.112, 0.5) AND pm < 5) SELECT n.ra, n.dec, n.gmag, n.gmag - n.rmag AS gr, g.pmra, g.pmdec FROM g, nsc_dr2.object AS n WHERE q3c_join(g.ra, g.dec, n.ra, n.dec, 1.0/3600.0)
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 20 | Planner-safe crossmatch executed (builder tool OR MATERIALIZED-CTE SQL) (1 checks, each worth 20/1 pts) |
| C2 | auto | 15 | Small side is Gaia, big indexed side is NSC (correct join orientation) (1 checks, each worth 15/1 pts) |
| C3 | auto | 10 | Cone centered on Palomar 5 (1 checks, each worth 10/1 pts) |
| C4 | auto | 10 | Proper-motion selection applied in the executed crossmatch (Pal 5 is LOW-PM) (1 checks, each worth 10/1 pts) |
| C5 | auto | 10 | CMD/photometric selection on the NSC side (executed query/plot expressions) (1 checks, each worth 10/1 pts) |
| C6 | judge | 20 | On-sky distribution of stream candidates plotted; tidal tails discussed — *Full credit: RA/Dec scatter of the selected stars with the leading/trailing tail orientation discussed. Half: plot without interpretation, or interpretation without plot.* |
| C7 | judge | 15 | Execution-model awareness (why materialize the small side; why NSC is the indexed side) — *Full credit if the answer (or its tool choice) reflects the q3c execution model: reduce small side first, big catalog on the indexed side, ~1 arcsec match radius. Half credit for a correct result with no rationale.* |

## DLB-10 (Tier 5, weight 1.8): Multi-survey SED from photometry (Coma red galaxies)

**Prompt:** Build optical-to-mid-infrared SEDs for a small sample (a few hundred) of red galaxies within 1° of the Coma cluster (RA = 194.95, Dec = +27.98) by combining Legacy Surveys DR9 grz photometry with the survey's forced WISE (W1/W2) photometry. Give me magnitude vs wavelength.

**Exercises:** ls_dr9.tractor carries dereddened grz AND forced unWISE W1/W2 in ONE table (no cross-catalog join!); wavelengths from the SVO Filter Profile Service.

**Expected actions:** Cone + extended + red cut + LIMIT -> SVO effective wavelengths (a property of the FILTER, not memorized constants) -> per-object mag vs lambda.

**Reference query:**

```sql
SELECT ra, dec, dered_mag_g, dered_mag_r, dered_mag_z, dered_mag_w1, dered_mag_w2 FROM ls_dr9.tractor WHERE q3c_radial_query(ra, dec, 194.95, 27.98, 1.0) AND type != 'PSF' AND snr_g > 5 AND snr_r > 5 AND snr_z > 5 AND dered_mag_g - dered_mag_r > 1.0 LIMIT 500
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | Single-table ls_dr9.tractor source (grz + forced W1/W2 together) (2 checks, each worth 15/2 pts) |
| C2 | auto | 10 | Cone at Coma (194.95, +27.98) ~1 deg (2 checks, each worth 10/2 pts) |
| C3 | auto | 15 | Red, extended-source selection with S/N floor in the executed query (3 checks, each worth 15/3 pts) |
| C4 | auto | 10 | Sample capped to 'a few hundred' (LIMIT / limit arg <= 1000) (1 checks, each worth 10/1 pts) |
| C5 | auto | 20 | Wavelengths resolved from the SVO Filter Profile Service, not hardcoded (datalab_sed_plot is SVO-backed internally, so either tool qualifies) (1 checks, each worth 20/1 pts) |
| C6 | auto | 10 | An SED plot was rendered (1 checks, each worth 10/1 pts) |
| C7 | judge | 20 | SED is correctly constructed — *Full credit: magnitude vs wavelength (microns), log-x, inverted y (brighter up), all five bands g/r/z/W1/W2 present, per-object curves. Deduct proportionally for missing bands, linear-x over 0.4-4.6 um, or upright mag axis.* |

- **QP-10a (−10)**: Joined LS DR9 with a separate WISE catalog even though tractor already carries forced W1/W2 (explicit PDF guardrail).

## DLB-11 (Tier 5, weight 1.8): Targeting-bitmask catalog selection (DESI DR1 LRGs)

**Prompt:** From the DESI DR1 redshift catalog, select luminous red galaxies (LRG target class) between z = 0.4 and 0.8, and show their redshift distribution and sky footprint.

**Exercises:** desi_dr1.zpix + desi_target bitmask + zwarn via TAP (no spectra).

**Expected actions:** Bitmask AND + quality cuts; server-side GROUP BY z-bin for the histogram; GROUP BY rounded mean_fiber_ra/dec for the footprint.

**Reference query:**

```sql
SELECT ROUND(z::numeric, 2) AS z_bin, COUNT(*) AS n FROM desi_dr1.zpix WHERE survey = 'main' AND main_primary AND (desi_target & 1) != 0 AND zwarn = 0 AND spectype = 'GALAXY' AND z BETWEEN 0.4 AND 0.8 GROUP BY z_bin ORDER BY z_bin
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | Queried desi_dr1.zpix (1 checks, each worth 15/1 pts) |
| C2 | auto | 20 | LRG bitmask + spectroscopic quality cuts in the executed query (4 checks, each worth 20/4 pts) |
| C3 | auto | 10 | Redshift window 0.4 <= z <= 0.8 in the executed query (2 checks, each worth 10/2 pts) |
| C4 | auto | 15 | Server-side aggregation (GROUP BY z-bin / sky cell), not a multi-million-row pull (1 checks, each worth 15/1 pts) |
| C5 | auto | 10 | BOTH deliverables rendered (z-distribution AND sky footprint) (1 checks, each worth 10/1 pts) |
| C6 | judge | 20 | Deliverables are scientifically correct — *z-histogram: n(z) over 0.4-0.8 with sensible bins (~0.01-0.05). Footprint: RA/Dec (mean_fiber_ra/dec) density or scatter showing the DESI main-survey footprint. Full credit needs both correct; half for one.* |
| C7 | judge | 10 | Bitmask semantics explained (LRG = bit 0 of desi_target; main survey primacy) — *Full credit for explaining the bitwise AND selection and the survey='main' / main_primary quality context. Half for using it correctly without explanation.* |

- Global penalties disabled here: GP-03 (the reference solution itself uses that pattern legitimately)

## DLB-12 (Tier 6, weight 2.0): Density search → image vetting (Hydra II, chained)

**Prompt:** Search NSC DR2 for the densest stellar clump within 1° of the Hydra II dwarf region, then pull DECam image cutouts of the top few candidate locations so I can eyeball them.

**Exercises:** nsc_dr2.object TAP overdensity + SIA cutouts at peaks (chained TAP -> SIA).

**Expected actions:** Density aggregate (0.05-deg bins, star+blue cuts) -> top-N cells -> per cell deepest g-band Stack cutout -> multi-panel grid (label no-coverage panels).

**Reference query:**

```sql
SELECT ROUND(ra/0.05)*0.05 AS ra_bin, ROUND(dec/0.05)*0.05 AS dec_bin, COUNT(*) AS n FROM nsc_dr2.object WHERE q3c_radial_query(ra, dec, 185.41, -31.98, 1.0) AND class_star > 0.5 AND gmag - rmag BETWEEN -0.5 AND 0.5 GROUP BY ra_bin, dec_bin ORDER BY n DESC LIMIT 5
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 20 | Chained density->cutout workflow executed (one-shot vetting tool OR aggregate+grid) (1 checks, each worth 20/1 pts) |
| C2 | auto | 10 | Cone within 1 deg of Hydra II (185.41, -31.98) (2 checks, each worth 10/2 pts) |
| C3 | auto | 10 | Stellar + blue color cuts in the executed density query (2 checks, each worth 10/2 pts) |
| C4 | auto | 10 | Top-N kept small (a 'few' candidates, N <= 10) (1 checks, each worth 10/1 pts) |
| C5 | auto | 15 | Cutout imagery rendered (2 checks, each worth 15/2 pts) |
| C6 | judge | 20 | Candidates ranked with coordinates + counts; densest clump identified — *Full credit: an ordered list of peak (ra, dec) with source counts, densest first (the densest clump should be Hydra II itself, ~185.43, -31.99). Half: peaks without ordering or counts.* |
| C7 | judge | 15 | Vetting guidance + honest handling of missing coverage — *Credit for telling the user what to look for in the cutouts (compact stellar concentration vs. chance clump / cluster contamination) and for labeling panels without SIA coverage instead of failing or faking them.* |

## DLB-13 (Tier 6, weight 2.0): Large-scale-structure wedge (SDSS Great Wall)

**Prompt:** Select galaxies from SDSS/BOSS in a thin redshift slice and make a cone/wedge plot to show the cosmic web — pick a region like the SDSS Great Wall.

**Exercises:** sdss_dr17.specobj via TAP; comoving distance; wedge/3D scatter.

**Expected actions:** ra/dec/z + quality query -> astropy comoving distance -> spherical->Cartesian -> wedge (thin Dec slice) with equal axis scaling, or 2D pie slice.

**Reference query:**

```sql
SELECT ra, dec, z FROM sdss_dr17.specobj WHERE class = 'GALAXY' AND zwarning = 0 AND ra BETWEEN 190 AND 240 AND dec BETWEEN 0 AND 5 AND z BETWEEN 0.0 AND 0.10
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | Queried sdss_dr17.specobj (1 checks, each worth 15/1 pts) |
| C2 | auto | 15 | Galaxy + quality + thin-z selection in the executed query (3 checks, each worth 15/3 pts) |
| C3 | judge | 15 | Region choice targets the SDSS Great Wall geometry — *Full credit for RA ~150-250, a THIN Dec slice (a few deg), z <~ 0.1 — or an equivalent well-justified slice. The Dec slice being thin is what makes the wedge readable; deduct half if the slice is fat (>~10 deg).* |
| C4 | auto | 15 | Wedge plot actually rendered (2 checks, each worth 15/2 pts) |
| C5 | judge | 20 | Comoving-space construction is correct — *Full credit: redshifts converted to comoving distance with a named cosmology (e.g. Planck18), spherical->Cartesian (or polar r=d, theta=RA pie slice), and axis scaling that preserves Mpc-per-unit (no stretched thin slab). Deduct ~1/3 for each missing element.* |
| C6 | judge | 10 | Cosmic-web interpretation — *Credit for identifying walls/filaments/voids and naming the Great Wall feature if visible.* |
| C7 | auto | 10 | Pull is bounded (region + z cuts and/or LIMIT / tool row cap) (1 checks, each worth 10/1 pts) |

- Global penalties disabled here: GP-01, GP-03 (the reference solution itself uses that pattern legitimately)

## DLB-14 (Tier 6, weight 2.0): Variable-star characterization (RR Lyrae in Hydra II)

**Prompt:** I have a candidate variable star at RA = 185.4311, Dec = −31.9953 (an RR Lyrae in the Hydra II field). Find its multi-epoch SMASH photometry, phase-fold the light curve to get the period, and pull an image cutout of the field.

**Exercises:** smash_dr1.source multi-epoch TAP time series + Lomb-Scargle + SIA cutout. This star is SMASH object 169.429960.

**Expected actions:** 1-arcsec cone (or id key) -> g-band valid epochs (cmag<99) ordered by mjd -> Lomb-Scargle over P=0.1-1 d -> phase-fold -> field cutout via coadd_all.

**Reference query:**

```sql
SELECT mjd, cmag, cerr, filter FROM smash_dr1.source WHERE q3c_radial_query(ra, dec, 185.4311, -31.9953, 1.0/3600.0) AND filter = 'g' AND cmag < 99 ORDER BY mjd
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | auto | 15 | Multi-epoch smash_dr1.source queried (not the coadd object table) (1 checks, each worth 15/1 pts) |
| C2 | auto | 10 | Star isolated by a ~1-arcsec cone at (185.4311, -31.9953) or by object id (1 checks, each worth 10/1 pts) |
| C3 | auto | 10 | Valid-epoch filtering (cmag < 99) and a single band, in the executed query (2 checks, each worth 10/2 pts) |
| C4 | auto | 20 | Lomb-Scargle period search + phase fold executed (2 checks, each worth 20/2 pts) |
| C5 | auto | 10 | Folded light curve rendered (1 checks, each worth 10/1 pts) |
| C6 | judge | 15 | Period is reported and plausible for an RR Lyrae — *Full credit: a concrete period in days within 0.1-1.0 d (RRab typically ~0.4-0.9 d, RRc ~0.2-0.45 d), consistent with the tool output. Zero for a period outside the physical range or none reported.* |
| C7 | auto | 10 | Field image cutout also pulled (1 checks, each worth 10/1 pts) |
| C8 | judge | 10 | Coherent end-to-end narrative (epochs -> period -> fold -> field) — *Credit for reporting number of epochs used, the folded curve's sawtooth shape if visible, and tying the cutout back to the star's field.* |

## DLB-15 (Tier 7, weight 2.4): Discover new Milky Way satellites (open-ended strategy)

**Prompt:** I want to discover new Milky Way satellite dwarf-galaxy candidates. Devise and carry out a search strategy using Data Lab's deep imaging catalogs, and give me a ranked list of candidate positions with supporting CMDs and image cutouts.

**Exercises:** Full agency: choose survey (DELVE/NSC/DES), tiled TAP density + matched filter, CMD verification, SIA cutouts, ranking. Highest scan risk -> confirm area first.

**Expected actions:** Schema discovery -> pick catalog -> tiled region-bounded density GROUP BY (hpix_1024, old/metal-poor point-source cuts) -> matched filter -> rank -> per-candidate CMD + cutout -> ranked report. Confirm sky area before fanning out (guardrail #5).

**Reference query:**

```sql
SELECT AVG(ra) AS ra0, AVG(dec) AS dec0, hpix_1024, COUNT(*) AS n FROM delve_dr3.coadd_objects WHERE q3c_radial_query(ra, dec, :tile_ra, :tile_dec, 2.0) AND (mag_auto_g - mag_auto_r) < 0.75 AND mag_auto_g > 19.5 AND magerr_auto_g < 0.2 AND ext_coadd BETWEEN 0 AND 1 GROUP BY hpix_1024
```

| Checkpoint | Type | Points | Criterion |
|---|---|---|---|
| C1 | judge | 15 | A real search strategy is designed before executing — *Full credit: names the survey choice (DELVE/NSC/DES) with a reason (depth, southern coverage), describes the matched-filter density approach and the old/metal-poor CMD selection, and states how candidates will be vetted and ranked. Deduct proportionally for missing elements.* |
| C2 | auto | 15 | Guardrail: sky area confirmed/estimated BEFORE the wide fan-out (1 checks, each worth 15/1 pts) |
| C3 | auto | 20 | Tiled, region-bounded, server-side density search with point-source + color cuts (3 checks, each worth 20/3 pts) |
| C4 | auto | 10 | Background-job orchestration for the long scan (status/results polling) (1 checks, each worth 10/1 pts) |
| C5 | judge | 15 | Ranked candidate list with positions + significance — *Full credit: an ordered table of candidate (ra, dec) with a density/matched-filter significance per candidate. Known satellites recovered (e.g. Hydra II if in the footprint) count as validation, not failure. Half: unranked or significance-free list.* |
| C6 | auto | 10 | Per-candidate verification evidence gathered (CMD and/or cutouts) (2 checks, each worth 10/2 pts) |
| C7 | judge | 15 | CMD-based vetting interpreted correctly — *Full credit if candidate CMDs are checked for an old, metal-poor population (tight MSTO/RGB, possible BHB) and obvious false positives (clusters of galaxies, chip artifacts) are screened out or flagged.* |
