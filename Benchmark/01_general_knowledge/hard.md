# General Knowledge — Hard

---

## GK-H-01: Bandwidth Switching for Calibration

**Question:**
> Which projects in the ALMA Science Archive likely needed Bandwidth Switching for calibration?

**Skills Tested:**
- Advanced knowledge of ALMA calibration strategies
- Understanding of when bandwidth switching is required (e.g., bright calibrators, high spectral resolution, specific correlator modes)
- Ability to construct complex archive queries
- Reasoning about observational metadata

**Expected Tools:**
- RAG pipeline (documentation on bandwidth switching)
- ALMA Science Archive TAP query or programmatic search
- Possibly multi-step reasoning (Conductor DAG)

**Evaluation Criteria:**
- [ ] Correctly explains what **bandwidth switching** is (switching between narrow/wide bandwidth correlator modes during calibration to cover dynamic range issues)
- [ ] Identifies the conditions that trigger bandwidth switching:
  - High spectral resolution observations with narrow channel widths
  - Observations where flux/bandpass calibrators are too bright for the narrow-bandwidth science setup
  - Typically associated with high spectral resolution spectral line observations
- [ ] Attempts to query the archive for projects matching these criteria
- [ ] Provides a strategy or query for identifying such projects (e.g., filtering by spectral resolution, correlator setup, or calibration metadata)
- [ ] Demonstrates understanding that this information may not be directly tagged in archive metadata and requires inference
