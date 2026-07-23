import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const sidebar = fs.readFileSync(new URL("../src/components/Sidebar.tsx", import.meta.url), "utf8");
const spectralPage = fs.readFileSync(new URL("../src/app/spectral-lines/page.tsx", import.meta.url), "utf8");

// E4 decision 2026-07-23: sidebar links to the explorer are hidden for now.
// The page stays reachable via deep links (/spectral-lines?...) from chat tools.
// When the links are restored, flip these back to assert.match.
test("spectral line explorer is hidden from sidebar navigation", () => {
    assert.doesNotMatch(sidebar, /href=["']\/spectral-lines["']/);
    assert.doesNotMatch(sidebar, />\s*Spectral Line Explorer\s*</);
});

test("spectral line explorer page is restored, not a redirect", () => {
    assert.doesNotMatch(spectralPage, /redirect\(["']\/["']\)/);
    // The restored page must parse the deep-link query params so that
    // /spectral-lines?... links from chat tools open the explorer instead of bouncing home.
    assert.match(spectralPage, /useSearchParams/);
});
