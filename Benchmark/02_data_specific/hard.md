# Data-Specific — Hard

---

## DS-H-01: HH212 Band 7 Deep Continuum Data

**Question:**
> I'm interested in the source HH212. Give me a summary of the Band 7 data from any project that can be used for creating a deep, high-resolution (better than 1 arcsec) image of the continuum.

**Skills Tested:**
- Source-specific archive search
- Understanding of angular resolution and its relation to array configuration/baseline length
- Knowledge of what makes data suitable for "deep continuum" imaging
- Ability to aggregate and summarize observational metadata
- Multi-criteria filtering (source + band + resolution + continuum suitability)

**Expected Tools:**
- ALMA Science Archive TAP query or Astroquery
- Possibly ALMiner for source lookup
- Multi-step reasoning (analysis of results)

**Evaluation Criteria:**
- [ ] Searches for source "HH212" (or coordinates of HH 212) in Band 7 (~275–373 GHz)
- [ ] Filters for observations with angular resolution better than 1 arcsec (`s_resolution < 1.0`)
- [ ] Identifies projects suitable for continuum imaging (considers bandwidth, sensitivity, integration time)
- [ ] Provides a structured summary including:
  - Project code(s) / proposal ID(s)
  - PI name(s)
  - Angular resolution achieved
  - Total bandwidth / continuum bandwidth
  - Integration time / sensitivity
  - Array configuration used
  - Date of observations
- [ ] Mentions the possibility of combining data from multiple projects for a deeper image
- [ ] Discusses or identifies any calibration or compatibility considerations
