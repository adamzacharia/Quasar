/** Curated one-click recipe library for the /gallery page.
 *
 * Each recipe is a ready-to-run chat prompt exercising a verified Quasar
 * capability chain (Data Lab TAP + SIA, SPARCL, CDS X-Match, time domain).
 * Clicking a card routes to `/?prompt=<prompt>` — the chat composer prefills
 * and the user presses send (no auto-fire).
 */

export type RecipeDifficulty = "starter" | "intermediate" | "advanced";

export interface Recipe {
    id: string;
    title: string;
    description: string;
    prompt: string;
    topic: string;
    difficulty: RecipeDifficulty;
    /** Unified Astronomy Thesaurus-style keywords for filtering. */
    uat: string[];
    catalogs: string[];
}

export const RECIPE_TOPICS = [
    "Stellar populations",
    "Time domain",
    "Galaxies & LSS",
    "Spectroscopy",
    "Crossmatch & SEDs",
    "Imaging",
] as const;

export const RECIPES: Recipe[] = [
    {
        id: "smash-hydra-ii-overdensity",
        title: "Dwarf-galaxy overdensity in SMASH",
        description: "Reproduce the classic Hydra II-style search: density map of point sources in a SMASH field, then eyeball the densest cells with image cutouts.",
        prompt: "Using SMASH DR2 field 169, map the stellar density of point sources (|sharp| < 0.5, 20 < gmag < 24.5) on a 0.05 deg grid, find the strongest overdensity, and show me image cutouts of the top 3 peaks.",
        topic: "Stellar populations",
        difficulty: "intermediate",
        uat: ["dwarf galaxies", "stellar density", "Magellanic Clouds", "local group"],
        catalogs: ["smash_dr2"],
    },
    {
        id: "delve-stream-map",
        title: "Stellar stream matched-filter map",
        description: "Tiled overdensity scan over a DELVE footprint with a color-magnitude filter tuned to an old, metal-poor population — the stream-hunting workflow, run as a background job.",
        prompt: "Run a tiled overdensity search in delve_dr3 over RA 20-30, Dec -35 to -30 for point sources (ext_coadd 0-1) with 17 < mag_auto_g < 23.5 and g-r < 0.6. Use 1-degree tiles and rank the candidate peaks.",
        topic: "Stellar populations",
        difficulty: "advanced",
        uat: ["stellar streams", "galactic halo", "tidal disruption"],
        catalogs: ["delve_dr3"],
    },
    {
        id: "lmc-cmd",
        title: "LMC color-magnitude diagram",
        description: "One-shot CMD of a crowded Magellanic field from SMASH photometry — the quick-look diagnostic for any resolved stellar population.",
        prompt: "Plot a g vs g-i color-magnitude diagram of SMASH DR2 point sources within 0.3 degrees of RA 80.894, Dec -69.756 in the LMC.",
        topic: "Stellar populations",
        difficulty: "starter",
        uat: ["color-magnitude diagrams", "stellar photometry", "Magellanic Clouds"],
        catalogs: ["smash_dr2"],
    },
    {
        id: "rr-lyrae-fold",
        title: "RR Lyrae discovery & period folding",
        description: "The full variability chain: rank variable candidates in a SMASH cone, pull one star's multi-epoch light curve, and Lomb-Scargle fold it.",
        prompt: "Find the most variable stars in smash_dr1 within 0.2 deg of RA 80.9, Dec -69.75 (at least 15 epochs), fetch the light curve of the top candidate, and period-fold it — is it an RR Lyrae?",
        topic: "Time domain",
        difficulty: "intermediate",
        uat: ["RR Lyrae variable stars", "periodic variable stars", "light curves", "Lomb-Scargle periodogram"],
        catalogs: ["smash_dr1"],
    },
    {
        id: "async-wide-density",
        title: "Wide-area density map as a background job",
        description: "Submit a wide HEALPix density aggregate with async_submit and track it in the Jobs panel instead of holding the chat turn open.",
        prompt: "Submit a background job (async_submit) computing the ring256 HEALPix source-density map of nsc_dr2 within 5 degrees of the LMC center (RA 80.89, Dec -69.76), then tell me the job id so I can check it later.",
        topic: "Imaging",
        difficulty: "advanced",
        uat: ["surveys", "sky surveys", "HEALPix"],
        catalogs: ["nsc_dr2"],
    },
    {
        id: "desi-lss-wedge",
        title: "Large-scale structure wedge from DESI",
        description: "A redshift-space wedge of DESI DR1 galaxies through a rich field — cosmic web filaments and voids straight out of the spectroscopic catalog.",
        prompt: "Make a large-scale-structure wedge plot of desi_dr1 galaxies (zwarn = 0, 0 < z < 0.5) within 2 degrees of RA 150, Dec 2.2 (the COSMOS field).",
        topic: "Galaxies & LSS",
        difficulty: "intermediate",
        uat: ["large-scale structure", "redshift surveys", "cosmic web"],
        catalogs: ["desi_dr1"],
    },
    {
        id: "sparcl-qso-stack",
        title: "Stack DESI quasar spectra",
        description: "SPARCL constraint search for moderate-redshift QSOs, then a rest-frame median stack — watch the broad lines sharpen as the noise averages down.",
        prompt: "Search SPARCL for DESI-DR1 QSO spectra with redshift between 2.0 and 2.2, retrieve 20 of them, and stack them in the rest frame with a median combine. Show the stacked spectrum.",
        topic: "Spectroscopy",
        difficulty: "advanced",
        uat: ["quasars", "spectroscopy", "composite spectra"],
        catalogs: ["SPARCL (DESI-DR1)"],
    },
    {
        id: "sparcl-galaxy-lines",
        title: "Galaxy spectrum with line identifications",
        description: "Pull one DESI galaxy spectrum near a position and plot it interactively with redshifted emission/absorption line overlays (vacuum wavelengths).",
        prompt: "Find SPARCL spectra within 60 arcsec of RA 150.1, Dec 2.2, plot the nearest GALAXY spectrum with spectral lines marked, and tell me which emission lines are visible.",
        topic: "Spectroscopy",
        difficulty: "starter",
        uat: ["galaxy spectroscopy", "emission line galaxies", "redshift"],
        catalogs: ["SPARCL (DESI-DR1, SDSS-DR17)"],
    },
    {
        id: "xmatch-2mass",
        title: "Crossmatch your list against 2MASS",
        description: "Upload a handful of targets and match them against the 2MASS point-source catalog via the CDS X-Match service — J/H/Ks photometry in one step.",
        prompt: "Crossmatch this object list against 2MASS with a 5 arcsec radius and give me their J, H, Ks magnitudes: M31 (ra 10.6847, dec 41.2690), SN 1987A (ra 83.8667, dec -69.2697), 3C 273 (ra 187.2779, dec 2.0524).",
        topic: "Crossmatch & SEDs",
        difficulty: "starter",
        uat: ["astronomy databases", "catalogs", "infrared photometry"],
        catalogs: ["2MASS via CDS X-Match"],
    },
    {
        id: "gaia-nsc-xmatch-sed",
        title: "Gaia × NSC crossmatch + SED",
        description: "Server-side q3c crossmatch of Gaia DR3 against the NOIRLab Source Catalog in a cone, then an SED for the brightest match using SVO filter curves.",
        prompt: "Crossmatch gaia_dr3 against nsc_dr2 within 0.1 deg of RA 150.1, Dec 2.2 (1 arcsec match radius), then plot the SED of the brightest matched source.",
        topic: "Crossmatch & SEDs",
        difficulty: "intermediate",
        uat: ["cross-match", "spectral energy distribution", "photometry"],
        catalogs: ["gaia_dr3", "nsc_dr2"],
    },
    {
        id: "allwise-agn-colors",
        title: "AGN selection with WISE colors",
        description: "The W1-W2 vs W2-W3 color-color diagram of AllWISE point sources — the classic mid-IR AGN wedge, straight from the newly registered allwise catalog.",
        prompt: "Select allwise point sources (ext_flg = 0, w1snr > 5, w2snr > 5) within 1 degree of RA 150.1, Dec 2.2 and plot the W1-W2 vs W2-W3 color-color diagram. Which sources fall in the AGN wedge (W1-W2 > 0.8)?",
        topic: "Galaxies & LSS",
        difficulty: "intermediate",
        uat: ["active galactic nuclei", "infrared color", "WISE"],
        catalogs: ["allwise"],
    },
    {
        id: "splus-sed",
        title: "12-band S-PLUS photometric SED",
        description: "Exercise the 12-band Javalambre system: pull a bright S-PLUS DR4 star's u→z + narrow-band AB magnitudes and inspect the photometric SED.",
        prompt: "Get the 12-band S-PLUS DR4 photometry (u_auto through z_auto plus the 7 narrow bands) for the brightest star-like source (class_star > 0.9) within 0.05 deg of RA 10.0, Dec 0.0 in Stripe 82, and describe its SED shape.",
        topic: "Crossmatch & SEDs",
        difficulty: "intermediate",
        uat: ["photometric systems", "narrow band photometry", "stellar classification"],
        catalogs: ["splus_dr4"],
    },
    {
        id: "color-cutout-vetting",
        title: "Density peaks with cutout vetting",
        description: "Find the densest cells in a cone with color cuts applied, then automatically pull a Legacy Surveys cutout grid of the top peaks to eyeball.",
        prompt: "In nsc_dr2 within 1 degree of RA 56.09, Dec -55.95, find the densest 0.05-degree cells of blue point sources (class_star > 0.8, g-r < 0.4) and show me cutouts of the top 5 peaks.",
        topic: "Imaging",
        difficulty: "intermediate",
        uat: ["stellar density", "astronomical images", "visual inspection"],
        catalogs: ["nsc_dr2"],
    },
    {
        id: "footprint-check",
        title: "Which surveys cover my target?",
        description: "MOC coverage check + footprint overlays on the sky viewer for an arbitrary position — answer 'do I have data there?' before writing any query.",
        prompt: "Which of the registered Data Lab surveys cover RA 80.894, Dec -69.756? Show their footprints on the sky viewer and tell me which ones include the LMC.",
        topic: "Imaging",
        difficulty: "starter",
        uat: ["sky coverage", "surveys", "virtual observatory"],
        catalogs: ["all registered"],
    },
    {
        id: "save-my-table",
        title: "Save a result as a durable table",
        description: "Run a governed cone query, save the result as a named My-table that outlives the 1-hour cache, and reload it later for follow-up plots.",
        prompt: "Select gaia_dr3 sources with parallax_over_error > 10 within 0.5 deg of RA 56.75, Dec 24.12 (the Pleiades), save the result as a table named pleiades_members, then list my saved tables.",
        topic: "Stellar populations",
        difficulty: "starter",
        uat: ["astrometry", "open star clusters", "data management"],
        catalogs: ["gaia_dr3"],
    },
];
