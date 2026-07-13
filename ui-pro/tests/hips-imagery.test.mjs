import test from "node:test";
import assert from "node:assert/strict";

import { hips2fitsUrl, normalizeHipsImageMeta } from "../src/lib/hips-imagery.js";

test("builds a hips2fits URL with encoded survey and default dimensions", () => {
    const urlString = hips2fitsUrl({ ra: 10.5, dec: -2.25, fovDeg: 0.2, survey: "P/SDSS9/color" });

    assert.match(urlString, /hips=CDS%2FP%2FSDSS9%2Fcolor/);
    const url = new URL(urlString);
    assert.equal(url.origin + url.pathname, "https://alasky.cds.unistra.fr/hips-image-services/hips2fits");
    assert.equal(url.searchParams.get("hips"), "CDS/P/SDSS9/color");
    assert.equal(url.searchParams.get("ra"), "10.5");
    assert.equal(url.searchParams.get("dec"), "-2.25");
    assert.equal(url.searchParams.get("fov"), "0.2");
    assert.equal(url.searchParams.get("width"), "800");
    assert.equal(url.searchParams.get("height"), "800");
    assert.equal(url.searchParams.get("projection"), "TAN");
    assert.equal(url.searchParams.get("format"), "png");
});

test("clamps fov at both ends", () => {
    const tiny = new URL(hips2fitsUrl({ ra: 1, dec: 2, fovDeg: 0.001, survey: "P/DSS2/color" }));
    const huge = new URL(hips2fitsUrl({ ra: 1, dec: 2, fovDeg: 100, survey: "P/DSS2/color" }));

    assert.equal(tiny.searchParams.get("fov"), "0.01");
    assert.equal(huge.searchParams.get("fov"), "60");
});

test("returns null for invalid coordinates", () => {
    assert.equal(hips2fitsUrl({ ra: NaN, dec: 0, survey: "P/DSS2/color" }), null);
    assert.equal(hips2fitsUrl({ ra: -1, dec: 0, survey: "P/DSS2/color" }), null);
    assert.equal(hips2fitsUrl({ ra: 1, dec: 91, survey: "P/DSS2/color" }), null);
});

test("uses custom width and height when valid", () => {
    const url = new URL(hips2fitsUrl({ ra: 1, dec: 2, fovDeg: 0.5, survey: "NRAO/P/VLASS-Quicklook-MedianStack", width: 512, height: 256 }));

    assert.equal(url.searchParams.get("hips"), "NRAO/P/VLASS-Quicklook-MedianStack");
    assert.equal(url.searchParams.get("width"), "512");
    assert.equal(url.searchParams.get("height"), "256");
});

test("normalizes backend snake-case image metadata for the frontend", () => {
    const meta = normalizeHipsImageMeta({ kind: "hips", ra: 42, dec: -5, fov_deg: 0.2, survey: "CDS/P/DSS2/color" });

    assert.deepEqual(meta, { kind: "hips", ra: 42, dec: -5, fovDeg: 0.2, survey: "P/DSS2/color" });
});

test("emits format=fits when requested (T7.1 FITS download)", () => {
    const url = new URL(hips2fitsUrl({ ra: 10.5, dec: -2.25, fovDeg: 0.2, survey: "P/DSS2/color", format: "fits" }));
    assert.equal(url.searchParams.get("format"), "fits");
});

test("falls back to png for an unknown format", () => {
    const url = new URL(hips2fitsUrl({ ra: 1, dec: 2, fovDeg: 0.2, survey: "P/DSS2/color", format: "tiff" }));
    assert.equal(url.searchParams.get("format"), "png");
});

test("passes a valid SIA fits_url through as fitsUrl", () => {
    const meta = normalizeHipsImageMeta({
        kind: "hips",
        ra: 42,
        dec: -5,
        fits_url: "https://datalab.noirlab.edu/svc/cutout?id=abc&format=fits",
    });
    assert.equal(meta.fitsUrl, "https://datalab.noirlab.edu/svc/cutout?id=abc&format=fits");
});

test("drops a non-http fits_url", () => {
    const meta = normalizeHipsImageMeta({ kind: "hips", ra: 42, dec: -5, fits_url: "/plots/local.fits" });
    assert.equal(meta.fitsUrl, undefined);
});
