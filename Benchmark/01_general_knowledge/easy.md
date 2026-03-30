# General Knowledge — Easy

---

## GK-E-01: ALMA Configurations and Angular Resolutions

**Question:**
> What are the different ALMA configurations and their corresponding angular resolutions?

**Skills Tested:**
- Factual recall of ALMA instrument specifications
- RAG retrieval from documentation / knowledge base

**Expected Tools:**
- RAG pipeline (documentation retrieval)
- Possibly web search for supplementary info

**Evaluation Criteria:**
- [ ] Lists the main 12m array configurations (C-1 through C-10, or compact to extended)
- [ ] Mentions that compact configurations (e.g., C-1) give lower angular resolution (~few arcsec) and extended configurations (e.g., C-10) give higher angular resolution (~tens of mas)
- [ ] Mentions that angular resolution depends on observing frequency/band
- [ ] Optionally mentions ACA (7m array) and Total Power configurations
- [ ] Provides approximate resolution ranges for at least a few configurations
- [ ] Information is factually correct

---

## GK-E-02: CASA Version Used for Data Processing

**Question:**
> How do I find out the information that indicates which CASA version was used to process the data from a given ALMA project?

**Skills Tested:**
- Knowledge of ALMA data products and metadata
- Understanding of CASA pipeline processing

**Expected Tools:**
- RAG pipeline (documentation retrieval)

**Evaluation Criteria:**
- [ ] Mentions checking the `README` file or processing logs in the delivered data package
- [ ] References the `casa*.log` files or pipeline weblog
- [ ] Mentions that the ALMA Science Archive metadata or QA2 reports may contain this information
- [ ] Mentions the `pipeline_manifest.xml` or equivalent metadata file
- [ ] Factually accurate guidance
