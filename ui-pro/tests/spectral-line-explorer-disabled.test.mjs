import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const sidebar = fs.readFileSync(new URL("../src/components/Sidebar.tsx", import.meta.url), "utf8");
const spectralPage = fs.readFileSync(new URL("../src/app/spectral-lines/page.tsx", import.meta.url), "utf8");

test("spectral line explorer is hidden from navigation", () => {
    assert.doesNotMatch(sidebar, /href=["']\/spectral-lines["']/);
    assert.doesNotMatch(sidebar, />\s*Spectral Line Explorer\s*</);
});

test("legacy spectral line explorer URL returns users to chat", () => {
    assert.match(spectralPage, /redirect\(["']\/["']\)/);
});
