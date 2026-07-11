# Data-Specific — Medium

---

## DS-M-01: Cycle 9 Multi-Array Projects

**Question:**
> How many projects in Cycle 9 used the 12m, 7m, and total power array for their observations?

**Skills Tested:**
- Understanding of ALMA array types (12m, 7m / ACA, Total Power / TP)
- Ability to construct multi-condition queries
- Knowledge of how array information is encoded in archive metadata
- Joining or intersecting query results

**Expected Tools:**
- ALMA Science Archive TAP query
- Possibly Astroquery or ALMiner
- Multi-step reasoning to combine filters

**Evaluation Criteria:**
- [ ] Correctly identifies all three array types: 12m (main array), 7m (ACA), and Total Power (TP)
- [ ] Constructs a query that finds projects using **all three** arrays (not just any one)
- [ ] Filters correctly for Cycle 9 (proposal IDs starting with `2021.` or `2022.` depending on convention, or using `obs_release_date` / `proposal_id` patterns)
- [ ] Returns a specific count
- [ ] Explains the methodology used
