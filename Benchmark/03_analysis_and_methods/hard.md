# Analysis & Methods — Hard

---

## AM-H-01: ALMA and JWST Protostars in Perseus

**Question:**
> Show me locations of protostars in Perseus that have been observed with ALMA and JWST.

**Skills Tested:**
- Cross-archive query (ALMA + JWST/MAST)
- Coordinate matching and cross-referencing
- Astronomical source catalog knowledge (protostar catalogs in Perseus)
- Visualization / map generation
- Multi-step reasoning and data synthesis

**Expected Tools:**
- ALMA Science Archive query (TAP or Astroquery)
- MAST archive query (for JWST data) or Astroquery MAST
- Coordinate cross-matching (astropy, catalog matching)
- Visualization tool (matplotlib, plotly) for sky map
- Multi-agent orchestration (Conductor DAG)

**Evaluation Criteria:**
- [ ] Queries ALMA archive for observations in the Perseus star-forming region
- [ ] Queries JWST / MAST archive for observations in the same region
- [ ] Cross-matches sources observed by both observatories
- [ ] Identifies which sources are protostars (uses known catalogs or classification)
- [ ] Generates a visual map showing source locations (RA/Dec plot)
- [ ] Labels or distinguishes ALMA-only, JWST-only, and both-observed sources
- [ ] Output is visually clear and informative

---

## AM-H-02: ALMA and JWST Hubble Ultra Deep Field Overlay

**Question:**
> I want to see two images of the Hubble Ultra Deep Field, one with ALMA and one with JWST. Ideally, overlap the two images in a way that I can see ALMA in contours and JWST in colorscale.

**Skills Tested:**
- Multi-archive image retrieval (ALMA + JWST archives)
- FITS image handling and WCS alignment
- Contour overlay visualization techniques
- Knowledge of the Hubble Ultra Deep Field (HUDF) coordinates
- Advanced astronomical visualization (aplpy, astropy, matplotlib)

**Expected Tools:**
- ALMA Science Archive query + image download
- MAST / JWST archive query + image download
- Python code generation (astropy, aplpy/matplotlib for overlay)
- Multi-agent orchestration (Conductor DAG)

**Evaluation Criteria:**
- [ ] Correctly identifies the HUDF coordinates (RA ~ 03h32m39s, Dec ~ -27°47'29")
- [ ] Retrieves or references ALMA data/image of the HUDF (e.g., ASPECS or A2HUDF projects)
- [ ] Retrieves or references JWST data/image of the HUDF
- [ ] Generates (or provides code to generate) an overlay image:
  - JWST image as colorscale background
  - ALMA data shown as contours
  - Proper WCS alignment between the two
- [ ] Output image is scientifically meaningful and visually appealing
- [ ] Handles coordinate alignment (WCS reprojection if needed)
- [ ] Labels axes, provides colorbar, and adds legend/annotations
