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

const HIPS2FITS_FORMATS = new Set(["png", "fits", "jpg"]);

/**
 * @param {{ra?: number|string, dec?: number|string, fovDeg?: number|string, survey?: string, width?: number|string, height?: number|string, format?: string}} [options]
 * @returns {string | null}
 */
export function hips2fitsUrl({ ra, dec, fovDeg, survey, width = HIPS_DEFAULT_SIZE, height = HIPS_DEFAULT_SIZE, format = "png" } = {}) {
    const raNum = Number(ra);
    const decNum = Number(dec);
    if (!validSkyPosition(raNum, decNum)) return null;

    const surveyId = hips2fitsSurveyId(survey);
    if (!surveyId) return null;

    const fmt = String(format || "png").toLowerCase();
    const outFormat = HIPS2FITS_FORMATS.has(fmt) ? fmt : "png";

    const url = new URL(HIPS2FITS_BASE_URL);
    url.searchParams.set("hips", surveyId);
    url.searchParams.set("ra", String(raNum));
    url.searchParams.set("dec", String(decNum));
    url.searchParams.set("fov", String(clampHipsFov(fovDeg)));
    url.searchParams.set("width", String(imageDimension(width, HIPS_DEFAULT_SIZE)));
    url.searchParams.set("height", String(imageDimension(height, HIPS_DEFAULT_SIZE)));
    url.searchParams.set("projection", "TAN");
    url.searchParams.set("format", outFormat);
    return url.toString();
}

/**
 * @param {unknown} meta
 * @returns {{kind?: string, ra?: number, dec?: number, fovDeg?: number, survey?: string,
 *            fitsUrl?: string,
 *            frames?: {url: string, label?: string}[],
 *            warnings?: string[],
 *            mocs?: {id: string, name?: string, color?: string, order?: number,
 *                    mocJson: Record<string, number[]>}[]} | undefined}
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

    // Raw FITS download URL (T7.1): single-band SIA cutouts carry the selected
    // tile's Data Lab access URL. HiPS cards omit it and build their own
    // hips2fits format=fits URL client-side from ra/dec/fov/survey instead.
    const fitsUrl = meta.fitsUrl ?? meta.fits_url;
    if (typeof fitsUrl === "string" && /^https?:\/\//i.test(fitsUrl)) {
        out.fitsUrl = fitsUrl;
    }

    // Ordered frames for the blink comparator card (kind "blink"): each is a
    // served PNG at matched geometry (e.g. per-epoch VLASS cutouts).
    if (Array.isArray(meta.frames)) {
        const frames = [];
        for (const f of meta.frames) {
            if (!f || typeof f !== "object") continue;
            const frameUrl = String(f.url || "").trim();
            if (!frameUrl) continue;
            const entry = { url: frameUrl };
            if (f.label) entry.label = String(f.label);
            frames.push(entry);
        }
        if (frames.length) out.frames = frames;
    }

    // Registration/coverage caveats from the epoch service (A3 CX-18): the
    // normalizer used to strip this field, so the on-card amber warning never
    // rendered and an unregistered epoch could read as a real transient.
    if (Array.isArray(meta.warnings)) {
        // Strings only (CX-36): coercing objects/numbers through String()
        // rendered "[object Object]" garbage on the card.
        const warnings = meta.warnings
            .filter((w) => typeof w === "string")
            .map((w) => w.trim())
            .filter(Boolean)
            .slice(0, 10);
        if (warnings.length) out.warnings = warnings;
    }

    // Survey footprint overlays (MOC geometry) for the interactive view.
    if (Array.isArray(meta.mocs)) {
        const mocs = [];
        for (const m of meta.mocs) {
            if (!m || typeof m !== "object") continue;
            const id = String(m.id || "").trim();
            const mocJson = m.mocJson ?? m.moc_json;
            if (!id || !mocJson || typeof mocJson !== "object" || Array.isArray(mocJson)) continue;
            const entry = { id, mocJson };
            if (m.name) entry.name = String(m.name);
            if (m.color) entry.color = String(m.color);
            const order = Number(m.order);
            if (Number.isFinite(order)) entry.order = order;
            mocs.push(entry);
        }
        if (mocs.length) out.mocs = mocs;
    }

    return out;
}
