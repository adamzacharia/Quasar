/**
 * Pure HiPS imagery helpers for chat image-card controls.
 * Plain JS so node --test can import it directly.
 */

export const HIPS2FITS_BASE_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits";
export const HIPS_FOV_MIN = 0.01;
export const HIPS_FOV_MAX = 60;
export const HIPS_DEFAULT_FOV = 0.25;
export const HIPS_DEFAULT_SIZE = 800;

export function clampHipsFov(fovDeg) {
    const value = Number(fovDeg);
    if (!Number.isFinite(value)) return HIPS_DEFAULT_FOV;
    return Math.max(HIPS_FOV_MIN, Math.min(HIPS_FOV_MAX, value));
}

export function normalizeHipsSurveyId(survey) {
    const raw = String(survey || "").trim();
    if (!raw) return "";
    return raw.startsWith("CDS/P/") ? raw.slice(4) : raw;
}

function hips2fitsSurveyId(survey) {
    const normalized = normalizeHipsSurveyId(survey);
    if (!normalized) return "";
    return normalized.startsWith("P/") ? `CDS/${normalized}` : normalized;
}

function validSkyPosition(ra, dec) {
    return Number.isFinite(ra) && Number.isFinite(dec)
        && ra >= 0 && ra < 360
        && dec >= -90 && dec <= 90;
}

function imageDimension(value, fallback) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
    return Math.max(1, Math.round(parsed));
}

/**
 * @param {{ra?: number|string, dec?: number|string, fovDeg?: number|string, survey?: string, width?: number|string, height?: number|string}} [options]
 * @returns {string | null}
 */
export function hips2fitsUrl({ ra, dec, fovDeg, survey, width = HIPS_DEFAULT_SIZE, height = HIPS_DEFAULT_SIZE } = {}) {
    const raNum = Number(ra);
    const decNum = Number(dec);
    if (!validSkyPosition(raNum, decNum)) return null;

    const surveyId = hips2fitsSurveyId(survey);
    if (!surveyId) return null;

    const url = new URL(HIPS2FITS_BASE_URL);
    url.searchParams.set("hips", surveyId);
    url.searchParams.set("ra", String(raNum));
    url.searchParams.set("dec", String(decNum));
    url.searchParams.set("fov", String(clampHipsFov(fovDeg)));
    url.searchParams.set("width", String(imageDimension(width, HIPS_DEFAULT_SIZE)));
    url.searchParams.set("height", String(imageDimension(height, HIPS_DEFAULT_SIZE)));
    url.searchParams.set("projection", "TAN");
    url.searchParams.set("format", "png");
    return url.toString();
}

/**
 * @param {unknown} meta
 * @returns {{kind?: string, ra?: number, dec?: number, fovDeg?: number, survey?: string} | undefined}
 */
export function normalizeHipsImageMeta(meta) {
    if (!meta || typeof meta !== "object") return undefined;
    const kind = String(meta.kind || "").trim();
    if (!kind) return undefined;

    const ra = Number(meta.ra);
    const dec = Number(meta.dec);
    const rawFov = meta.fovDeg ?? meta.fov_deg;
    const fov = Number(rawFov);
    const out = { kind };

    if (Number.isFinite(ra)) out.ra = ra;
    if (Number.isFinite(dec)) out.dec = dec;
    if (Number.isFinite(fov)) out.fovDeg = clampHipsFov(fov);

    const survey = normalizeHipsSurveyId(meta.survey);
    if (survey) out.survey = survey;

    return out;
}
