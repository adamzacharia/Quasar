import test from "node:test";
import assert from "node:assert/strict";

import { normalizeSkyCoords, skyViewGeometry } from "../src/lib/sky-geometry.js";

function angularDistanceFromZero(ra) {
    return Math.min(Math.abs(ra), Math.abs(360 - ra));
}

test("centers a cluster across the RA wrap without treating it as all-sky", () => {
    const geometry = skyViewGeometry([
        { ra: 359.9, dec: 0 },
        { ra: 0.1, dec: 0 },
    ]);

    assert.equal(geometry.ok, true);
    assert.equal(geometry.allSky, false);
    assert.ok(angularDistanceFromZero(geometry.raCenter) < 0.2);
    assert.ok(geometry.fovDeg < 1);
});

test("keeps pole clusters compact despite scattered RA values", () => {
    const geometry = skyViewGeometry([
        { ra: 0, dec: 89.9 },
        { ra: 90, dec: 89.9 },
        { ra: 180, dec: 89.9 },
        { ra: 270, dec: 89.9 },
    ]);

    assert.equal(geometry.ok, true);
    assert.equal(geometry.allSky, false);
    assert.ok(Number.isFinite(geometry.raCenter));
    assert.ok(geometry.decCenter > 89.8);
    assert.ok(geometry.fovDeg < 1);
});

test("filters invalid coordinates and reports the kept count", () => {
    const geometry = skyViewGeometry([
        { ra: NaN, dec: 0 },
        { ra: 12, dec: 91 },
        { ra: 18, dec: -91 },
        { ra: "not a number", dec: 4 },
        { ra: 42, dec: -5 },
        { ra: -1, dec: 2 },
    ]);

    assert.equal(geometry.ok, true);
    assert.equal(geometry.count, 2);
    assert.deepEqual(geometry.markers.map((marker) => marker.ra), [42, 359]);
});

test("normalizes RA values into the zero to 360 degree range", () => {
    const normalized = normalizeSkyCoords([
        { ra: -0.5, dec: 0 },
        { ra: 360.5, dec: 0 },
    ]);

    assert.deepEqual(normalized.map((coord) => coord.ra), [359.5, 0.5]);
});

test("uses a small field of view for a single point", () => {
    const geometry = skyViewGeometry([{ ra: 123.4, dec: -22.5 }]);

    assert.equal(geometry.ok, true);
    assert.equal(geometry.allSky, false);
    assert.equal(geometry.fovDeg, 0.25);
});

test("treats antipodal pairs as all-sky", () => {
    const geometry = skyViewGeometry([
        { ra: 10, dec: 0 },
        { ra: 190, dec: 0 },
    ]);

    assert.equal(geometry.ok, true);
    assert.equal(geometry.allSky, true);
});

test("caps markers while preserving the valid coordinate count", () => {
    const coords = Array.from({ length: 7 }, (_, index) => ({
        ra: 360 + index * 0.001,
        dec: 0,
    }));
    const geometry = skyViewGeometry(coords, { maxMarkers: 3 });

    assert.equal(geometry.ok, true);
    assert.equal(geometry.allSky, false);
    assert.equal(geometry.truncated, true);
    assert.equal(geometry.count, 7);
    assert.equal(geometry.markers.length, 3);
    assert.ok(geometry.markers[0].ra < 1e-12);
    assert.ok(Math.abs(geometry.markers[1].ra - 0.001) < 1e-10);
    assert.ok(Math.abs(geometry.markers[2].ra - 0.002) < 1e-10);
});

test("reports empty and all-invalid input as not ok", () => {
    assert.equal(skyViewGeometry([]).ok, false);

    const geometry = skyViewGeometry([
        null,
        {},
        { ra: Infinity, dec: 0 },
        { ra: 12, dec: -92 },
    ]);

    assert.equal(geometry.ok, false);
    assert.equal(geometry.count, 0);
});