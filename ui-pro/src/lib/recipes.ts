/** Recipe Gallery content: questions from Quasar's benchmark suites that
 * gpt-oss-120b answered well, grouped by what they ask for.
 *
 * Source: the latest UI-run, blind-graded results (2026-09-26 DeepSeek vs
 * gpt-oss comparison and its same-day re-runs); a question is included when
 * the mean of the two graders was >= 70/100. Dev / public splits only:
 * ArchiveBench held-out questions (AB-H-*) are never included, and only the
 * question text ships (no rubric, golden or expected answer).
 *
 * Left out on purpose: DS-H-01 (its pass relies on a hard-coded router rule
 * flagged as contamination) and the WebBench two-turn follow-ups FUP-03/04/05
 * (a single prefilled prompt cannot reproduce them). ALMABench has no
 * gpt-oss-120b results yet, so nothing comes from it.
 *
 * Clicking a card routes to `/?prompt=<prompt>`: the chat composer prefills
 * and the user presses send (no auto-fire).
 *
 * Generated 2026-09-27 from tmp/orbita-redesign-2026-09-27/benchmark_passes.json
 * (71 questions: DataLabBench 15, domain bench 10, ArchiveBench dev 32,
 * WebBench 14).
 */

export type RecipeDifficulty = "starter" | "intermediate" | "advanced";

export interface Recipe {
    id: string;
    title: string;
    /** The question exactly as it is typed into the chat. */
    prompt: string;
    topic: string;
    difficulty: RecipeDifficulty;
    /** Archives and catalogs the question touches (may be empty). */
    catalogs: string[];
}

export const RECIPE_TOPICS = [
    "ALMA archive",
    "Multi-mission archives",
    "Catalog lookups",
    "Stellar populations & Milky Way",
    "Galaxies & cosmology",
    "Sky images",
    "Time domain & transients",
    "Exoplanets",
    "Star formation",
    "Literature & people",
    "Proposals & policy",
] as const;

export const RECIPES: Recipe[] = [
    {"id": "gk-e-02", "title": "Find the CASA version used for ALMA data", "prompt": "How do I find out the information that indicates which CASA version was used to process the data from a given ALMA project?", "topic": "ALMA archive", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "ds-e-01", "title": "ALMA Cycle 10 solar projects count", "prompt": "How many projects in Cycle 10 observed the sun?", "topic": "ALMA archive", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "am-e-01", "title": "Query the ALMA archive from Python", "prompt": "Is there a way to make a query of the ALMA Science Archive to retrieve ALMA data for a specific object through Python?", "topic": "ALMA archive", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "ab-d-48", "title": "PI and MOUS count of an ALMA project", "prompt": "Who is the PI of ALMA project 2016.1.00484.L, what is the project called, and how many member observing units does the archive hold for it?", "topic": "ALMA archive", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "nos-02", "title": "ALMA Band 7 observations of NGC 1068", "prompt": "Find ALMA observations of NGC 1068 in Band 7.", "topic": "ALMA archive", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "ds-m-01", "title": "Cycle 9 projects using all three arrays", "prompt": "How many projects in Cycle 9 used the 12m, 7m, and total power array for their observations?", "topic": "ALMA archive", "difficulty": "intermediate", "catalogs": ["ALMA"]},
    {"id": "am-m-01", "title": "Astroquery search for M83 ALMA data", "prompt": "How do I use Astroquery to find ALMA observations of the source M83?", "topic": "ALMA archive", "difficulty": "intermediate", "catalogs": ["ALMA"]},
    {"id": "am-m-02", "title": "Query ALMA with TAP and ALMiner", "prompt": "Show me two ways to query for a source in the ALMA Science Archive, with one using TAP and the other using ALMiner.", "topic": "ALMA archive", "difficulty": "intermediate", "catalogs": ["ALMA"]},
    {"id": "am-m-03", "title": "ALMA Band 6 observations of M83", "prompt": "Find ALMA observations of the source M83 using Band 6.", "topic": "ALMA archive", "difficulty": "intermediate", "catalogs": ["ALMA"]},
    {"id": "am-m-04", "title": "ALMA archive URL for M83 Band 6", "prompt": "Generate the URL for the ALMA Science Archive for the query of M83 using Band 6.", "topic": "ALMA archive", "difficulty": "intermediate", "catalogs": ["ALMA"]},
    {"id": "ab-d-52", "title": "ALMA observations of HL Tau in 2026", "prompt": "List the public ALMA observations of HL Tau taken during 2026.", "topic": "ALMA archive", "difficulty": "intermediate", "catalogs": ["ALMA"]},
    {"id": "ab-d-39", "title": "SS 433 in VLASS and ALMA", "prompt": "Is the microquasar SS 433 inside the VLASS footprint, and does ALMA have any public observations of it? Give the ALMA project codes if there are any.", "topic": "ALMA archive", "difficulty": "advanced", "catalogs": ["NRAO", "VLASS", "ALMA"]},
    {"id": "ab-d-51", "title": "ALMA CO(3-2) projects for NGC 253", "prompt": "Which public ALMA projects cover the CO(3-2) line for the starburst galaxy NGC 253, taking its systemic velocity as 243 km/s?", "topic": "ALMA archive", "difficulty": "advanced", "catalogs": ["ALMA"]},
    {"id": "ab-d-03", "title": "TESS 2-minute sectors for Proxima Centauri", "prompt": "Which TESS sectors have 2-minute cadence data for Proxima Centauri? Count only sectors up to and including Sector 90.", "topic": "Multi-mission archives", "difficulty": "starter", "catalogs": ["MAST", "TESS"]},
    {"id": "ab-d-02", "title": "JWST NIRSpec spectra of a z=6.3 quasar", "prompt": "Are there public JWST NIRSpec spectra of the z = 6.3 quasar SDSS J0100+2802? If so, which observing modes or gratings were used and under which program?", "topic": "Multi-mission archives", "difficulty": "intermediate", "catalogs": ["MAST", "JWST NIRSpec"]},
    {"id": "ab-d-11", "title": "Spitzer IRAC coverage of the Bullet Cluster", "prompt": "Has Spitzer's IRAC camera imaged the Bullet Cluster (1E 0657-56)? Which IRAC channels are covered, and are there Spitzer Enhanced Imaging Products for the field at IRSA?", "topic": "Multi-mission archives", "difficulty": "intermediate", "catalogs": ["IRSA", "Spitzer IRAC"]},
    {"id": "ab-d-17", "title": "Chandra HETG observations of Capella", "prompt": "Which Chandra HETG grating observations of Capella are public, and what is their total exposure time?", "topic": "Multi-mission archives", "difficulty": "intermediate", "catalogs": ["HEASARC", "Chandra"]},
    {"id": "ab-d-21", "title": "Chandra data on a Betelgeuse remnant", "prompt": "Pull the Chandra observations of the Betelgeuse supernova remnant and tell me how large the remnant is in X-rays.", "topic": "Multi-mission archives", "difficulty": "intermediate", "catalogs": ["HEASARC", "Chandra"]},
    {"id": "ab-d-20", "title": "X-ray observations of NGC 5907 ULX-1", "prompt": "For the ultraluminous X-ray source NGC 5907 ULX-1, list the public XMM-Newton and Chandra observation IDs that cover it, and give NED's distance to NGC 5907.", "topic": "Multi-mission archives", "difficulty": "advanced", "catalogs": ["HEASARC", "XMM-Newton", "Chandra", "NED"]},
    {"id": "dlb-01", "title": "LMC catalogs with near-infrared photometry", "prompt": "Which Data Lab catalogs cover the Large Magellanic Cloud, and which of those include near-infrared photometry? List the relevant table names.", "topic": "Catalog lookups", "difficulty": "starter", "catalogs": ["Data Lab"]},
    {"id": "dlb-02", "title": "Gaia DR3 cone count around Palomar 5", "prompt": "How many Gaia DR3 sources lie within 10 arcminutes of Palomar 5 (RA = 229.022, Dec = −0.112)?", "topic": "Catalog lookups", "difficulty": "starter", "catalogs": ["Data Lab", "Gaia DR3"]},
    {"id": "ab-d-07", "title": "AllWISE W1 and W2 magnitudes of 3C 273", "prompt": "What are the AllWISE W1 and W2 magnitudes of the quasar 3C 273?", "topic": "Catalog lookups", "difficulty": "starter", "catalogs": ["IRSA", "AllWISE"]},
    {"id": "ab-d-12", "title": "Gaia DR3 parallax of Barnard's Star", "prompt": "What parallax and G-band magnitude does Gaia DR3 give for Barnard's Star?", "topic": "Catalog lookups", "difficulty": "starter", "catalogs": ["Gaia DR3"]},
    {"id": "ab-d-22", "title": "SIMBAD type and redshift of PKS 2155-304", "prompt": "What object type and redshift does SIMBAD give for PKS 2155-304?", "topic": "Catalog lookups", "difficulty": "starter", "catalogs": ["SIMBAD"]},
    {"id": "ab-d-26", "title": "SIMBAD redshift for NGC 9999", "prompt": "What redshift does SIMBAD give for the galaxy NGC 9999?", "topic": "Catalog lookups", "difficulty": "starter", "catalogs": ["SIMBAD"]},
    {"id": "ab-d-30", "title": "SDSS spectroscopic redshift of 3C 186", "prompt": "What spectroscopic redshift does SDSS give for the quasar 3C 186?", "topic": "Catalog lookups", "difficulty": "starter", "catalogs": ["SDSS"]},
    {"id": "ab-d-53", "title": "Legacy Surveys DR9 count in Draco", "prompt": "How many Legacy Surveys DR9 sources brighter than r = 20 lie within 5 arcminutes of the centre of the Draco dwarf galaxy (RA 260.05, Dec +57.92)?", "topic": "Catalog lookups", "difficulty": "starter", "catalogs": ["Legacy Surveys DR9", "Data Lab"]},
    {"id": "ab-d-13", "title": "Hyades members by Gaia parallax", "prompt": "How many Gaia DR3 stars within 2 degrees of the Hyades centre (RA 66.75, Dec +15.87) have parallaxes between 19 and 24 mas?", "topic": "Catalog lookups", "difficulty": "intermediate", "catalogs": ["Gaia DR3"]},
    {"id": "ab-d-16", "title": "Python Gaia CMD of M67", "prompt": "Give me Python that pulls the Gaia DR3 sources within 0.5 degrees of M67 whose parallaxes lie between 1.0 and 1.4 mas, and plots BP-RP against G. How many stars does that selection return?", "topic": "Catalog lookups", "difficulty": "intermediate", "catalogs": ["Gaia DR3"]},
    {"id": "ab-d-23", "title": "Hipparcos stars within 10 parsecs", "prompt": "How many stars in the Hipparcos main catalogue have a parallax larger than 100 milliarcseconds?", "topic": "Catalog lookups", "difficulty": "intermediate", "catalogs": ["CDS", "Hipparcos"]},
    {"id": "dlb-03", "title": "Draco dwarf colour-magnitude diagram", "prompt": "Get g and r magnitudes for point sources within 0.4° of the Draco dwarf (RA = 260.06, Dec = +57.92) from NSC DR2 and plot a g vs (g−r) CMD.", "topic": "Stellar populations & Milky Way", "difficulty": "starter", "catalogs": ["Data Lab", "NSC DR2"]},
    {"id": "dlb-05", "title": "DES star/galaxy separation and colour-colour", "prompt": "Using DES DR1 around RA = 30, Dec = −50, separate stars from galaxies and show me g−r vs r−i color–color diagrams for each population.", "topic": "Stellar populations & Milky Way", "difficulty": "intermediate", "catalogs": ["Data Lab", "DES DR1"]},
    {"id": "dlb-06", "title": "Gaia white-dwarf proper-motion candidates", "prompt": "Find high-proper-motion white-dwarf candidates in Gaia DR3 in a 5°-radius patch of the southern sky (say around RA = 60, Dec = −50): significant parallax, large total proper motion, and absolute magnitudes on the WD sequence. Give me the HR diagram.", "topic": "Stellar populations & Milky Way", "difficulty": "intermediate", "catalogs": ["Data Lab", "Gaia DR3"]},
    {"id": "dlb-07", "title": "Stellar overdensity hunt in SMASH field 169", "prompt": "Help me look for a stellar overdensity — a possible dwarf companion — in SMASH DR1 field 169. Select blue main-sequence stars and find where they clump on the sky.", "topic": "Stellar populations & Milky Way", "difficulty": "intermediate", "catalogs": ["Data Lab", "SMASH DR1"]},
    {"id": "dlb-08", "title": "HEALPix stellar density map from NSC", "prompt": "Make a stellar density map of a ~20° × 20° region from NSC DR2 to reveal Milky Way structure — bin by HEALPix and show log counts.", "topic": "Stellar populations & Milky Way", "difficulty": "intermediate", "catalogs": ["Data Lab", "NSC DR2"]},
    {"id": "ab-d-55", "title": "SMASH colour-magnitude diagram of the SMC bar", "prompt": "Build a g against g-i colour-magnitude diagram from the SMASH survey for the SMC bar, using everything within 0.3 degrees of RA 13.19, Dec -72.83. How many stars go into it?", "topic": "Stellar populations & Milky Way", "difficulty": "intermediate", "catalogs": ["SMASH", "Data Lab"]},
    {"id": "dlb-09", "title": "Palomar 5 tidal tails via crossmatch", "prompt": "Combine NSC DR2 photometry with Gaia DR3 proper motions around Palomar 5 to trace its tidal tails — select stream stars by proper motion and CMD, then plot their on-sky distribution.", "topic": "Stellar populations & Milky Way", "difficulty": "advanced", "catalogs": ["Data Lab", "NSC DR2", "Gaia DR3"]},
    {"id": "dlb-12", "title": "Hydra II density search and cutouts", "prompt": "Search NSC DR2 for the densest stellar clump within 1° of the Hydra II dwarf region, then pull DECam image cutouts of the top few candidate locations so I can eyeball them.", "topic": "Stellar populations & Milky Way", "difficulty": "advanced", "catalogs": ["Data Lab", "NSC DR2", "DECam"]},
    {"id": "dlb-15", "title": "Search for new Milky Way satellites", "prompt": "I want to discover new Milky Way satellite dwarf-galaxy candidates. Devise and carry out a search strategy using Data Lab's deep imaging catalogs, and give me a ranked list of candidate positions with supporting CMDs and image cutouts.", "topic": "Stellar populations & Milky Way", "difficulty": "advanced", "catalogs": ["Data Lab"]},
    {"id": "ab-d-34", "title": "DESI DR2 spectrum of IC 1101", "prompt": "Show me the DESI DR2 spectrum of IC 1101, the giant elliptical at the centre of Abell 2029, and plot it.", "topic": "Galaxies & cosmology", "difficulty": "intermediate", "catalogs": ["DESI DR2", "Data Lab"]},
    {"id": "dlb-10", "title": "Optical to mid-infrared SEDs of Coma galaxies", "prompt": "Build optical-to-mid-infrared SEDs for a small sample (a few hundred) of red galaxies within 1° of the Coma cluster (RA = 194.95, Dec = +27.98) by combining Legacy Surveys DR9 grz photometry with the survey's forced WISE (W1/W2) photometry. Give me magnitude vs wavelength.", "topic": "Galaxies & cosmology", "difficulty": "advanced", "catalogs": ["Data Lab", "Legacy Surveys DR9", "WISE"]},
    {"id": "dlb-11", "title": "DESI DR1 LRG redshift distribution", "prompt": "From the DESI DR1 redshift catalog, select luminous red galaxies (LRG target class) between z = 0.4 and 0.8, and show their redshift distribution and sky footprint.", "topic": "Galaxies & cosmology", "difficulty": "advanced", "catalogs": ["Data Lab", "DESI DR1"]},
    {"id": "dlb-13", "title": "SDSS Great Wall wedge plot", "prompt": "Select galaxies from SDSS/BOSS in a thin redshift slice and make a cone/wedge plot to show the cosmic web — pick a region like the SDSS Great Wall.", "topic": "Galaxies & cosmology", "difficulty": "advanced", "catalogs": ["Data Lab", "SDSS", "BOSS"]},
    {"id": "dlb-04", "title": "Colour image of the M31 centre", "prompt": "Show me a color image of the center of M31 from the DECam Legacy Surveys.", "topic": "Sky images", "difficulty": "starter", "catalogs": ["Data Lab", "DECam Legacy Surveys"]},
    {"id": "ab-d-10", "title": "2MASS colour image of the Galactic Centre", "prompt": "Show me a 2MASS colour image of the Galactic Centre, about 10 arcminutes across, centred on Sgr A*.", "topic": "Sky images", "difficulty": "starter", "catalogs": ["IRSA", "CDS", "2MASS"]},
    {"id": "ab-d-25", "title": "Crab Nebula in DSS2 and 2MASS", "prompt": "Show me the Crab Nebula in DSS2 red and in 2MASS, side by side, each about 10 arcminutes across.", "topic": "Sky images", "difficulty": "intermediate", "catalogs": ["CDS", "DSS2", "2MASS"]},
    {"id": "ab-d-54", "title": "DES colour image of Fornax A", "prompt": "Show me a DES colour image of Fornax A (NGC 1316), about 15 arcminutes across.", "topic": "Sky images", "difficulty": "intermediate", "catalogs": ["DES", "Data Lab"]},
    {"id": "trn-01", "title": "Latest news on comet 3I/ATLAS", "prompt": "What is the latest news about the interstellar comet 3I/ATLAS?", "topic": "Time domain & transients", "difficulty": "starter", "catalogs": []},
    {"id": "ab-d-09", "title": "ZTF period of an RR Lyrae star", "prompt": "What pulsation period does the ZTF photometry give for the RR Lyrae star ZTFJ163108.64+365823.9 (RA 247.7860, Dec 36.9733), and is it an RRab or RRc type?", "topic": "Time domain & transients", "difficulty": "intermediate", "catalogs": ["IRSA", "ZTF"]},
    {"id": "ab-d-46", "title": "ZTF alerts near NGC 4993 in 2017", "prompt": "Are there any ZTF alerts within 30 arcseconds of NGC 4993 from August 2017, when the GW170817 kilonova was visible?", "topic": "Time domain & transients", "difficulty": "intermediate", "catalogs": ["ZTF"]},
    {"id": "dlb-14", "title": "Phase-fold an RR Lyrae light curve", "prompt": "I have a candidate variable star at RA = 185.4311, Dec = −31.9953 (an RR Lyrae in the Hydra II field). Find its multi-epoch SMASH photometry, phase-fold the light curve to get the period, and pull an image cutout of the field.", "topic": "Time domain & transients", "difficulty": "advanced", "catalogs": ["Data Lab", "SMASH"]},
    {"id": "ab-d-47", "title": "TDE AT2019dsg across TNS, ALeRCE, ALMA", "prompt": "For the tidal disruption event AT2019dsg: what classification and redshift does TNS give, how many ZTF detections does ALeRCE list for it, and does ALMA have any public data at its position?", "topic": "Time domain & transients", "difficulty": "advanced", "catalogs": ["TNS", "ALeRCE", "ZTF", "ALMA"]},
    {"id": "ab-d-41", "title": "Confirmed planets within 10 parsecs", "prompt": "How many confirmed planets orbit host stars within 10 parsecs, according to the NASA Exoplanet Archive as of 24 September 2026?", "topic": "Exoplanets", "difficulty": "intermediate", "catalogs": ["NASA Exoplanet Archive"]},
    {"id": "ab-d-42", "title": "Temperate small TESS planets around M dwarfs", "prompt": "Which confirmed planets discovered by TESS orbit M dwarfs (host temperature below 3900 K), have equilibrium temperatures below 300 K and radii under 2 Earth radii? List them with their orbital periods.", "topic": "Exoplanets", "difficulty": "intermediate", "catalogs": ["NASA Exoplanet Archive", "TESS"]},
    {"id": "ab-d-43", "title": "Radial-velocity mass of Kepler-186f", "prompt": "What is the radial-velocity mass of Kepler-186f?", "topic": "Exoplanets", "difficulty": "intermediate", "catalogs": ["NASA Exoplanet Archive"]},
    {"id": "nos-01", "title": "Derive the isothermal Jeans mass", "prompt": "Derive the Jeans mass for an isothermal, uniform-density gas cloud.", "topic": "Star formation", "difficulty": "starter", "catalogs": []},
    {"id": "am-h-01", "title": "Perseus protostars seen by ALMA and JWST", "prompt": "Show me locations of protostars in Perseus that have been observed with ALMA and JWST.", "topic": "Star formation", "difficulty": "advanced", "catalogs": ["ALMA", "JWST"]},
    {"id": "sq-e-01", "title": "Recent ALMA papers on protostellar outflows", "prompt": "I'm interested in studying protostellar outflows. Please make a table of the 10 most recent publications on this topic that used data from the ALMA Science Archive, and summarize them. Use the ALMA Science Archive to get this information.", "topic": "Literature & people", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "res-01", "title": "Who is Paola Caselli", "prompt": "Who is Paola Caselli?", "topic": "Literature & people", "difficulty": "starter", "catalogs": []},
    {"id": "res-02", "title": "Who is Crystal Brogan", "prompt": "Who is Crystal Brogan?", "topic": "Literature & people", "difficulty": "starter", "catalogs": []},
    {"id": "ab-d-59", "title": "Phosphine on Venus papers with bibcodes", "prompt": "Which paper first reported phosphine in the atmosphere of Venus, and which papers pushed back on that detection? Give me bibcodes.", "topic": "Literature & people", "difficulty": "intermediate", "catalogs": ["ADS"]},
    {"id": "ab-d-60", "title": "Discovery paper of PSR J1748-2446ad", "prompt": "Give me the bibcode and DOI of the discovery paper for the fastest-spinning known pulsar, PSR J1748-2446ad.", "topic": "Literature & people", "difficulty": "advanced", "catalogs": ["ADS"]},
    {"id": "pol-01", "title": "ALMA Cycle 13 proprietary period", "prompt": "What is the proprietary period for ALMA Cycle 13 data?", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "pol-02", "title": "JWST Cycle 5 GO proposal deadline", "prompt": "When is the JWST Cycle 5 General Observer proposal deadline?", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["JWST"]},
    {"id": "pol-03", "title": "Current VLA configuration and next move", "prompt": "Which configuration is the VLA in right now, and when is the next configuration move?", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["VLA"]},
    {"id": "nos-03", "title": "ALMA proprietary period without web search", "prompt": "Don't search the web: explain what the proprietary period means for ALMA data.", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "pol-04", "title": "Next VLA proposal deadline", "prompt": "When is the next VLA proposal deadline?", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["VLA"]},
    {"id": "pol-05", "title": "HST Cycle 34 exclusive access period", "prompt": "What is the default exclusive access period for HST GO data in Cycle 34?", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["HST"]},
    {"id": "pol-06", "title": "ALMA Cycle 13 observing start and end", "prompt": "When does ALMA Cycle 13 science observing start, and when does the cycle end?", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["ALMA"]},
    {"id": "pol-07", "title": "VLA configuration in March 2027", "prompt": "Which VLA configuration will be in use in March 2027?", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["VLA"]},
    {"id": "pol-08", "title": "ALMA Cycle 13 Large Program threshold", "prompt": "What is the Large Program threshold in the ALMA Cycle 13 Call for Proposals?", "topic": "Proposals & policy", "difficulty": "starter", "catalogs": ["ALMA"]},
];
