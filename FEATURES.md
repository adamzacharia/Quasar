# 🔭 Quasar Feature List — Astronomer Wishlist

A living document tracking features astronomers and astrophysicists want most.
Features are grouped by discipline. Items marked ✅ are implemented.

---

## 🌟 Universal (All Disciplines)

| # | Feature | Status | Priority |
|---|---|---|---|
| U1 | **Publication-Quality Plotting** — ApJ/MNRAS-style figures at 300 DPI, LaTeX labels, colorblind-safe palettes, PDF/PNG export | ✅ Built | 🔴 Critical |
| U2 | **Multi-archive Cross-matcher** — Query ALMA, Chandra, MAST, VizieR, NED, Simbad simultaneously for any source | ✅ Built | 🔴 Critical |
| U3 | **Literature-to-Code** — Read arXiv paper methods section, output equivalent Python/CASA/CIAO analysis code | 🔲 Planned | 🟠 High |
| U4 | **Grant / Proposal Writer** — RAG over NSF/NASA calls, successful funded abstracts, and reviewer guidelines | 🔲 Planned | 🟠 High |
| U5 | **Observing Run Log Analyzer** — Upload telescope night log → AI flags anomalies, weather losses, pointing failures | 🔲 Planned | 🟡 Medium |
| U6 | **Splatalogue Line ID** — Query molecular line database by frequency: *"What line is at 230.538 GHz?"* | ✅ Built | 🔴 Critical |
| U7 | **NED / Simbad Natural Language Search** — *"Find all Seyfert 2 galaxies within 100 Mpc with ALMA data"* | 🔲 Planned | 🟠 High |
| U8 | **SED Fitter Integration** — Wrap CIGALE/Bagpipes for photometric redshift and galaxy SED fitting | 🔲 Planned | 🟡 Medium |
| U9 | **Redshift Calculator** — Convert between cosmological quantities (z, Mpc, lookback time, physical scale) | 🔲 Planned | 🟡 Medium |
| U10 | **Coordinate Converter** — RA/Dec ↔ Galactic ↔ Ecliptic, epoch conversions (J2000 ↔ B1950) | 🔲 Planned | 🟢 Low |

---

## 📡 Radio Astronomy (ALMA / VLA / VLBA / GBT)

| # | Feature | Status | Priority |
|---|---|---|---|
| R1 | **ALMA Archive Search** — Search by target, position, frequency, PI, project code | ✅ Built | 🔴 Critical |
| R2 | **CASA Script Generator** — Generate calibration and imaging scripts from obs parameters | ✅ Built | 🔴 Critical |
| R3 | **Spectral Line Profile Plotter** — Plot velocity spectrum from data cube with Gaussian fit | 🔲 Planned | 🟠 High |
| R4 | **ALMA Sensitivity Calculator** — Wrapper around the online ALMA OT sensitivity formula | 🔲 Planned | 🟠 High |
| R5 | **ALMA Proposal Advisor** — RAG over the ALMA Cycle call and proposer's guide | 🔲 Planned | 🟠 High |
| R6 | **Beam Calculator** — Given array config and frequency, compute synthesized beam size | 🔲 Planned | 🟡 Medium |
| R7 | **Moment Map Generator** — From FITS data cube, produce moment 0/1/2 maps | 🔲 Planned | 🟡 Medium |
| R8 | **Channel Map Viewer** — Generate grid of velocity channel maps from 3D cubes | 🔲 Planned | 🟡 Medium |
| R9 | **uv-Coverage Plotter** — Visualize baseline coverage for a given array configuration | 🔲 Planned | 🟢 Low |
| R10 | **Primary Beam Correction Explainer** — Guide through PB correction for mosaics | 🔲 Planned | 🟢 Low |

---

## 🌊 Gravitational Wave Astronomy (LIGO / Virgo / KAGRA / LISA)

| # | Feature | Status | Priority |
|---|---|---|---|
| G1 | **GCN Alert Monitor** — Parse GW/GRB/neutrino alerts, extract key parameters, notify researcher | ✅ Built | 🔴 Critical |
| G2 | **GWTC Catalog Search** — Query the Gravitational Wave Transient Catalog by mass, distance, type | 🔲 Planned | 🟠 High |
| G3 | **Multi-messenger Sky Matcher** — Cross-match GW HEALPix sky map against telescope archives | 🔲 Planned | 🟠 High |
| G4 | **GW Parameter Estimator** — Given event params (M_chirp, distance), estimate expected SNR | 🔲 Planned | 🟡 Medium |
| G5 | **Bilby / LALInference Config Generator** — Generate PE config files from event parameters | 🔲 Planned | 🟡 Medium |
| G6 | **GW Strain Plotter** — Plot GW timeseries with bandpass filter and matched filter template | 🔲 Planned | 🟡 Medium |
| G7 | **Corner Plot Generator** — Plot posterior distributions from MCMC output (wraps `corner.py`) | 🔲 Planned | 🟡 Medium |

---

## 🔴 Infrared Astronomy (JWST / Spitzer / WISE / Herschel)

| # | Feature | Status | Priority |
|---|---|---|---|
| I1 | **MAST Archive Search** — Query Mikulski Archive by instrument/filter/target | 🔲 Planned | 🟠 High |
| I2 | **JWST Pipeline Advisor** — Recommend CRDS context, pipeline stages, parameters for given obs mode | 🔲 Planned | 🟠 High |
| I3 | **JWST Proposal Advisor** — RAG over successful JWST proposals and Cycle call | 🔲 Planned | 🟠 High |
| I4 | **Photometric SED Plotter** — Plot multi-band photometry with model SED overlay | 🔲 Planned | 🟡 Medium |
| I5 | **Dust Temperature Estimator** — Modified blackbody fit to far-IR photometry | 🔲 Planned | 🟡 Medium |
| I6 | **WISE / Spitzer Catalog Query** — Cone search and cross-match with WISE/Spitzer catalogs | 🔲 Planned | 🟢 Low |

---

## ⚡ High-Energy Astronomy (Chandra / XMM-Newton / Fermi / NuSTAR)

| # | Feature | Status | Priority |
|---|---|---|---|
| H1 | **CIAO Script Generator** — Generate CIAO data reduction scripts for Chandra observations | 🔲 Planned | 🟠 High |
| H2 | **XSPEC Model Advisor** — Recommend spectral models for given source type / energy range | 🔲 Planned | 🟠 High |
| H3 | **Fermi LAT 4FGL Search** — Query the Fermi 4FGL catalog for sources near a target | 🔲 Planned | 🟡 Medium |
| H4 | **X-ray Image Display** — Display Chandra/XMM event images with smoothing options | 🔲 Planned | 🟡 Medium |
| H5 | **Timing Analysis Tool** — Generate power spectra, lightcurves, periodograms from event files | 🔲 Planned | 🟡 Medium |

---

## 🌟 Optical / UV Astronomy (VLT / Keck / Rubin / ZTF / HST)

| # | Feature | Status | Priority |
|---|---|---|---|
| O1 | **Transient Classifier** — Given ZTF/LSST alert JSON, classify as SN Ia, AGN, CV, asteroid, etc. | 🔲 Planned | 🟠 High |
| O2 | **Target Visibility Planner** — Given site, date, and target list, generate observability chart | 🔲 Planned | 🟠 High |
| O3 | **Spectral Redshift Assistant** — Upload 1D spectrum, AI identifies lines and estimates redshift | 🔲 Planned | 🟠 High |
| O4 | **Color-Magnitude Diagram Builder** — Generate CMD from catalog photometry in standard systems | 🔲 Planned | 🟡 Medium |
| O5 | **Extinction Calculator** — Query Schlafly & Finkbeiner dust maps for E(B-V) at any position | 🔲 Planned | 🟡 Medium |
| O6 | **Finding Chart Generator** — Create DSS/PanSTARRS finding chart with target circle and N/E arrows | 🔲 Planned | 🟡 Medium |
| O7 | **Transient Name Server Search** — Query IAU TNS for registered transients by type/date/position | 🔲 Planned | 🟢 Low |

---

## 🔧 Developer / Platform Features

| # | Feature | Status | Priority |
|---|---|---|---|
| P1 | **Session Pruning** — Auto reset context window at 90k tokens, save summary to Mem0 | ✅ Built | 🔴 Critical |
| P2 | **Telegram Multi-channel** — Receive queries via Telegram bot, respond inline | ✅ Built | 🟠 High |
| P3 | **Browser Control** — Autonomous web browsing via Playwright for portals with no API | ✅ Built | 🟠 High |
| P4 | **Skills Plugin System** — Install new tools as self-contained skill packages at runtime | 🔲 Planned | 🟠 High |
| P5 | **Model Failover** — Auto-fallback to `gpt-4o-mini` if primary model fails | 🔲 Planned | 🟡 Medium |
| P6 | **WhatsApp Channel** — Route queries via Meta WhatsApp Cloud API | 🔲 Planned | 🟢 Low |

---

## 📊 Implementation Progress

| Category | Total Features | Built | % Complete |
|---|---|---|---|
| Universal | 10 | 3 | 30% |
| Radio | 10 | 2 | 20% |
| Gravitational Wave | 7 | 1 | 14% |
| Infrared | 6 | 0 | 0% |
| High-Energy | 5 | 0 | 0% |
| Optical / UV | 7 | 0 | 0% |
| Platform | 6 | 3 | 50% |
| **Total** | **51** | **9** | **18%** |

---

*Last updated: 2026-02-24 — Quasar v3.0.0*
