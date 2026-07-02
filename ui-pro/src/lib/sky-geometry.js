/**
 * Pure spherical-geometry helpers for the interactive sky view.
 *
 * Plain JS (like research-graph.js) so `node --test` can import it directly;
 * consumed by AladinSkyView.tsx. All angles are ICRS degrees.
 */

const DEG = Math.PI / 180;

/** Keep only finite coords with |dec| <= 90, normalizing RA into [0, 360). */
export function normalizeSkyCoords(coords) {
    const out = [];
    for (const c of coords || []) {
        if (!c || typeof c !== "object") continue;
        const ra = Number(c.ra);
        const dec = Number(c.dec);
        if (!Number.isFinite(ra) || !Number.isFinite(dec)) continue;
        if (dec < -90 || dec > 90) continue;
        out.push({ ra: ((ra % 360) + 360) % 360, dec });
    }
    return out;
}

/**
 * Compute the center, field of view and marker set for a collection of sky
 * positions, safely across the RA 0/360 wrap and at the poles.
 *
 * Uses a unit-vector centroid (never raw min/max RA — 359.9° and 0.1° must
 * yield a ~0.2° span, not ~360°). A near-zero centroid (antipodal/all-sky
 * spreads) or a very wide max separation is reported as `allSky`, which the
 * caller should treat as "use the all-sky canvas instead".
 *
 * @param {{ra:number, dec:number}[]} coords
 * @param {{maxMarkers?: number}} [opts]
 * @returns {{ok:boolean, allSky:boolean, raCenter:number, decCenter:number,
 *            fovDeg:number, markers:{ra:number,dec:number}[], truncated:boolean,
 *            count:number}}
 */
export function skyViewGeometry(coords, opts) {
    const maxMarkers = Math.max(1, (opts && opts.maxMarkers) || 2000);
    const valid = normalizeSkyCoords(coords);
    const base = {
        ok: false, allSky: false, raCenter: 0, decCenter: 0,
        fovDeg: 0.25, markers: [], truncated: false, count: valid.length,
    };
    if (valid.length === 0) return base;

    let sx = 0, sy = 0, sz = 0;
    const vecs = valid.map(({ ra, dec }) => {
        const x = Math.cos(dec * DEG) * Math.cos(ra * DEG);
        const y = Math.cos(dec * DEG) * Math.sin(ra * DEG);
        const z = Math.sin(dec * DEG);
        sx += x; sy += y; sz += z;
        return [x, y, z];
    });

    const norm = Math.sqrt(sx * sx + sy * sy + sz * sz);
    if (norm / valid.length < 1e-8) {
        // Antipodal / uniformly spread positions: no meaningful center exists.
        return { ...base, ok: true, allSky: true };
    }
    const cx = sx / norm, cy = sy / norm, cz = sz / norm;
    const raCenter = ((Math.atan2(cy, cx) / DEG) + 360) % 360;
    const decCenter = Math.asin(Math.max(-1, Math.min(1, cz))) / DEG;

    let maxSepDeg = 0;
    for (const [x, y, z] of vecs) {
        const dot = Math.max(-1, Math.min(1, x * cx + y * cy + z * cz));
        const sep = Math.acos(dot) / DEG;
        if (sep > maxSepDeg) maxSepDeg = sep;
    }
    if (maxSepDeg > 100) {
        return { ...base, ok: true, allSky: true, raCenter, decCenter };
    }

    const fovDeg = maxSepDeg < 0.01
        ? 0.25
        : Math.max(0.05, Math.min(180, maxSepDeg * 2.5));

    const truncated = valid.length > maxMarkers;
    return {
        ok: true,
        allSky: false,
        raCenter,
        decCenter,
        fovDeg,
        markers: truncated ? valid.slice(0, maxMarkers) : valid,
        truncated,
        count: valid.length,
    };
}
