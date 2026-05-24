# Disclaimer Page
**Effective Date: May 24, 2026**
**Version: 2.0**

The information, scripts, calculations, and astronomical summaries provided by the QUASAR platform ("the Service") are for general scientific research, academic study, and planning reference purposes only. 

By accessing or using Quasar, you explicitly acknowledge and agree to the following scientific disclaimers and limitations of liability.

---

## 1. Large Language Model & AI Hallucinations

Quasar relies on advanced Large Language Models (LLMs) and custom automated agents to plan search workflows and summarize literature. 
* **Probabilistic Outputs:** Generative artificial intelligence operates on statistical word probabilities. It does not possess real physical intelligence, celestial context, or true comprehension of astronomical data.
* **Hallucinations:** AI outputs are subject to "hallucinations." This means Quasar may generate literature summaries, Gauss-fits, line identifications, coordinate conversions, and bibliography citations that appear highly authentic but are completely false, incorrect, or fabricated.
* **Mandatory Fact-Checking:** You must independently verify all literature claims, citations, target coordinates, and research facts against verified astronomical databases (e.g., SIMBAD, NASA ADS, NED) before using them in publications or funding proposals.

---

## 2. Scientific & Observatory Operations Disclaimer

### 2.1 Telescope Time & Proposal Runs
Quasar is **not** an official interface for any astronomical observatory, including but not limited to the Joint ALMA Observatory (JAO), European Southern Observatory (ESO), National Radio Astronomy Observatory (NRAO), or the Canadian Astronomy Data Centre (CADC).
* **Observing Proposal Liability:** Using Quasar to generate, critique, or refine observing proposals is done at your own risk. We are not responsible for rejected proposals, poor scientific reviews, or the loss of highly competitive telescope time.
* **Physical Run Errors:** Do **not** input AI-generated Right Ascension (RA) coordinates, Declinations (Dec), or correlator configurations directly into telescope observing tools (such as the ALMA Observing Tool) without rigorous, manual double-checking. Quasar Observatory is not liable for ruined observing runs, pointing errors, or misplaced telescope pointings.

### 2.2 CASA & CARTA Script Generation
Quasar can generate Common Astronomy Software Applications (CASA) calibration and imaging scripts or CARTA visualization configurations.
* **No Quality Guarantees:** AI-generated CASA scripts are templates and may contain syntax errors, deprecated parameter calls, or sub-optimal imaging inputs. 
* **Safe Sandbox Testing:** You should review and test all generated scripts in isolated local environments before running them on valuable observation data sets. We are not responsible for corrupted data, crashed CASA sessions, or incorrect image reconstructions.

---

## 3. Mathematical & Calculations Accuracy

Calculations performed via the Sandboxed Python REPL (such as redshift conversions, synthesized beam calculators, and spectral line profiles) are provided for convenience.
* **System Limitations:** While our sandboxed REPL runs actual Python packages (e.g., `astropy`, `numpy`), inputs from LLMs or round-off errors in floating-point math can introduce subtle, scientific inaccuracies.
* **No Official Endorsement:** Users must execute official pipeline calibrations and utilize certified computational standards for publication-quality astronomical calculations.

---

## 4. Academic Integrity & Publication Claims

### 4.1 Peer-Review Compliance
Many academic publishers (such as AAS, MNRAS, and Elsevier) have strict guidelines regarding the use of generative AI tools in manuscript preparation and data analysis. You are solely responsible for ensuring that your use of Quasar complies with the guidelines of your target publisher, academic institution, or funding body.

### 4.2 Citation Responsibility
Quasar is an AI assistant, not an author. AI tools should not be listed as co-authors on scientific manuscripts. If Quasar contributes significantly to your literature review or pipeline generation, you agree to credit the platform appropriately according to standard academic practices, while retaining full personal accountability for the published science.

---

## 5. Contact Us

If you have any questions or require further details regarding scientific validation and safe computational boundaries, please contact:
**Email:** `science@quasarassistant.com`
