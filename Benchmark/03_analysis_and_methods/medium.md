# Analysis & Methods — Medium

---

## AM-M-01: Astroquery for M83 Observations

**Question:**
> How do I use Astroquery to find ALMA observations of the source M83?

**Skills Tested:**
- Specific knowledge of the `astroquery.alma` module
- Code generation with correct API usage
- Understanding of query results and metadata columns

**Expected Tools:**
- RAG pipeline or documentation lookup
- Code generation

**Evaluation Criteria:**
- [ ] Provides a working `astroquery.alma` code example
- [ ] Uses `Alma.query_object('M83')` or equivalent
- [ ] Explains the returned table structure (columns like `proposal_id`, `target_name`, `band_list`, etc.)
- [ ] Code is syntactically correct and uses current API
- [ ] Mentions any setup requirements (installation, imports)

---

## AM-M-02: TAP vs ALMiner Source Query

**Question:**
> Show me two ways to query for a source in the ALMA Science Archive, with one using TAP and the other using ALMiner.

**Skills Tested:**
- Knowledge of multiple query interfaces
- Ability to write code for both TAP (pyvo or ADQL) and ALMiner
- Comparative understanding of different tools

**Expected Tools:**
- Code generation
- RAG pipeline (documentation)

**Evaluation Criteria:**
- [ ] Provides a working **TAP/ADQL** query example (using pyvo, astroquery TAP, or raw ADQL)
- [ ] Provides a working **ALMiner** query example
- [ ] Both examples query for the same source/concept
- [ ] Explains the differences or trade-offs between the two approaches
- [ ] Code is syntactically correct for both methods

---

## AM-M-03: M83 Band 6 Observations

**Question:**
> Find ALMA observations of the source M83 using Band 6.

**Skills Tested:**
- Combining source and band filters in archive queries
- Understanding of ALMA band numbering
- Practical query execution

**Expected Tools:**
- ALMA Science Archive query (TAP, Astroquery, or ALMiner)
- Code generation and/or execution

**Evaluation Criteria:**
- [ ] Constructs a query with both source (M83) and band (Band 6) filters
- [ ] Returns actual results from the archive
- [ ] Presents results in a readable format (table, summary)
- [ ] Mentions Band 6 frequency range (~211–275 GHz) for context
- [ ] Code or method used is correct

---

## AM-M-04: Generate ALMA Archive URL for M83 Band 6

**Question:**
> Generate the URL for the ALMA Science Archive for the query of M83 using Band 6.

**Skills Tested:**
- Understanding of the ALMA Science Archive web interface URL structure
- Ability to construct parameterized URLs
- Knowledge of archive query parameters

**Expected Tools:**
- RAG pipeline (URL structure documentation)
- URL construction logic

**Evaluation Criteria:**
- [ ] Generates a valid, clickable URL pointing to the ALMA Science Archive
- [ ] URL includes source name parameter (M83)
- [ ] URL includes band filter (Band 6)
- [ ] URL is for the correct archive endpoint (almascience.eso.org, almascience.nrao.edu, or almascience.nao.ac.jp)
- [ ] URL, when opened, returns relevant results
