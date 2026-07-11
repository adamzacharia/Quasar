# Scientific Questions — Hard

---

## SQ-H-01: Galaxies at z=1–2 with CO Observations

**Question:**
> Show me all galaxies at a redshift between z=1 and z=2 that were observed with ALMA and included the Carbon Monoxide rest frequency in their spectral setup.

**Skills Tested:**
- Redshift-aware frequency calculations
- Understanding of CO rotational transitions and their rest frequencies
- Knowledge of how observed frequency relates to rest frequency via redshift: f_obs = f_rest / (1 + z)
- Complex multi-condition archive querying
- Cross-referencing with extragalactic source catalogs
- Understanding of ALMA's frequency coverage across bands

**Expected Tools:**
- ALMA Science Archive TAP query with frequency filtering
- Redshift-to-frequency conversion calculations
- Possibly NED or catalog cross-matching
- Multi-step reasoning (Conductor DAG)

**Evaluation Criteria:**
- [ ] Correctly computes the observed frequency ranges for CO transitions at z=1 to z=2:
  - CO(1-0) at 115.271 GHz → observed at 38.4–57.6 GHz (Band 1 range, mostly below ALMA coverage)
  - CO(2-1) at 230.538 GHz → observed at 76.8–115.3 GHz (Band 3)
  - CO(3-2) at 345.796 GHz → observed at 115.3–172.9 GHz (Band 4/5)
  - CO(4-3) at 461.041 GHz → observed at 153.7–230.5 GHz (Band 5/6)
  - CO(5-4) at 576.268 GHz → observed at 192.1–288.1 GHz (Band 6/7)
  - CO(6-5) at 691.473 GHz → observed at 230.5–345.7 GHz (Band 7)
  - Higher-J transitions at correspondingly higher frequencies
- [ ] Constructs queries covering the appropriate observed frequency ranges
- [ ] Filters for extragalactic / galaxy targets
- [ ] Handles the complexity of multiple possible CO transitions being redshifted into ALMA bands
- [ ] Returns a meaningful list of projects / sources
- [ ] Explains the methodology and which CO transitions are accessible at these redshifts
