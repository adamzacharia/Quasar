/**
 * Wiring assertions for the model selector and the Provider Keys panel.
 *
 * This repo has no DOM/testing-library harness — component tests here assert
 * against component source, and the behaviour those components delegate to is
 * unit-tested for real in model-catalog.test.mjs.
 */
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const dropdown = fs.readFileSync(new URL("../src/components/ModelDropdown.tsx", import.meta.url), "utf8");
const sidebar = fs.readFileSync(new URL("../src/components/Sidebar.tsx", import.meta.url), "utf8");
const settings = fs.readFileSync(new URL("../src/components/SettingsModal.tsx", import.meta.url), "utf8");
const hook = fs.readFileSync(new URL("../src/lib/useAvailableModels.ts", import.meta.url), "utf8");

/* ── data source ────────────────────────────────────────────────────────── */

test("the selector reads the live catalog instead of a hardcoded list", () => {
    assert.match(dropdown, /useAvailableModels\(\)/);
    assert.doesNotMatch(dropdown, /MODEL_PRICING/, "prices come from the catalog, not a local table");
    assert.doesNotMatch(sidebar, /MODEL_PRICING/, "the old sidebar pricing table is gone");
    assert.doesNotMatch(
        sidebar,
        /<ModelDropdown[\s\S]{0,200}availableModels=/,
        "the dropdown no longer takes a string[] of models",
    );
});

test("the hook is backed by GET /api/models/available and can be invalidated", () => {
    assert.match(hook, /\/api\/models\/available/);
    assert.match(hook, /\/api\/providers\/\$\{encodeURIComponent\(provider\)\}\/refresh-models/);
    assert.match(hook, /export function invalidateAvailableModels/);
});

test("catalog requests are authenticated", () => {
    assert.match(hook, /headers: authBearerHeaders\(\)/);
    assert.match(hook, /credentials: "include"/);
});

/* ── search ─────────────────────────────────────────────────────────────── */

test("a search box is pinned at the top of the panel and autofocused", () => {
    assert.match(dropdown, /autoFocus/);
    assert.match(dropdown, /placeholder="Search models/);
    assert.match(dropdown, /onChange=\{\(e\) => \{\s*setQuery\(e\.target\.value\)/);
});

test("search results are produced by the shared fuzzy scorer", () => {
    assert.match(dropdown, /buildGroups\(query, providers, \{ recent \}\)/);
});

test("matched characters are highlighted", () => {
    assert.match(dropdown, /function Highlight\(/);
    assert.match(dropdown, /<Highlight text=\{label\} ranges=\{labelRanges\}/);
    assert.match(dropdown, /<Highlight text=\{model\.id\} ranges=\{row\.idRanges\}/);
});

test("keyboard navigation covers arrows, Enter and the two-stage Escape", () => {
    assert.match(dropdown, /event\.key === "ArrowDown"/);
    assert.match(dropdown, /event\.key === "ArrowUp"/);
    assert.match(dropdown, /event\.key === "Enter"/);
    assert.match(dropdown, /event\.key === "Escape"/);
    assert.match(
        dropdown,
        /if \(query\) setQuery\(""\);\s*\n\s*else setOpen\(false\);/,
        "Escape should clear the query first and close on the second press",
    );
});

test("typing anywhere in the open panel focuses the search box", () => {
    assert.match(dropdown, /document\.activeElement !== inputRef\.current/);
    assert.match(dropdown, /inputRef\.current\?\.focus\(\)/);
});

/* ── layout ─────────────────────────────────────────────────────────────── */

test("the established price line format is preserved", () => {
    const lib = fs.readFileSync(new URL("../src/lib/model-catalog.js", import.meta.url), "utf8");
    assert.match(lib, /In: \$\$\{input\}\/M · Out: \$\$\{output\}\/M/);
    assert.match(dropdown, /In: \$\{currentIn\}\/M/, "the trigger keeps its own two-span price line");
    assert.match(dropdown, /Out: \$\{currentOut\}\/M/);
});

test("connected providers are badged BYOK and stale catalogs are marked", () => {
    assert.match(dropdown, /\{group\.byok && \(/);
    assert.match(dropdown, />\s*BYOK\s*</);
    assert.match(dropdown, /\{group\.stale && \(/);
});

test("not-connected providers collapse to a single unlock row at the bottom", () => {
    assert.match(dropdown, /\{entry\.label\} — Add API key to unlock \{entry\.unlockCount\} models/);
    assert.match(dropdown, /onClick=\{openKeys\}/);
});

test("the unlock row deep-links to Settings > Provider Keys", () => {
    assert.match(dropdown, /onAddProviderKey\?\.\(\)/);
    assert.match(sidebar, /setSettingsTab\("providerKeys"\); setSettingsOpen\(true\)/);
    assert.match(settings, /initialTab\?: TabType/);
});

test("model rows carry the id, price and context chip", () => {
    assert.match(dropdown, /const price = priceLine\(model\);/);
    assert.match(dropdown, /const ctx = contextChip\(model\);/);
    assert.match(dropdown, /showId && \(/, "the id is shown when it differs from the display name");
});

test("the empty state names connected providers and offers a key link", () => {
    assert.match(dropdown, /No models match/);
    assert.match(dropdown, /Connected providers: \{connected\.join\(", "\)\}/);
    assert.match(dropdown, /Add a\{\/\^\[aeiou\]\/i\.test\(lockedHint\.label\) \? "n" : ""\} \{lockedHint\.label\} API key/);
});

test("the trigger still shows the selected model as a sticky footer", () => {
    assert.match(dropdown, /Model: \{selectedModel\.startsWith\("local\/"\)/);
    assert.match(dropdown, /<ModelIcon model=\{selectedModel\}/);
});

/* ── accessibility ──────────────────────────────────────────────────────── */

test("the dropdown exposes listbox semantics", () => {
    assert.match(dropdown, /role="combobox"/);
    assert.match(dropdown, /role="listbox"/);
    assert.match(dropdown, /role="option"/);
    assert.match(dropdown, /role="group"/);
    assert.match(dropdown, /aria-activedescendant=\{flat\[active\] \? `\$\{listId\}-opt-\$\{active\}` : undefined\}/);
    assert.match(dropdown, /aria-selected=\{isSelected\}/);
    assert.match(dropdown, /aria-label="Search models"/);
    assert.match(dropdown, /aria-label="Available models"/);
    assert.match(dropdown, /aria-haspopup="listbox"/);
    assert.match(dropdown, /aria-expanded=\{open\}/);
});

/* ── recently used ──────────────────────────────────────────────────────── */

test("selecting a model records it in the recent list", () => {
    assert.match(dropdown, /nextRecentModels\(recent, modelId\)/);
    assert.match(dropdown, /persistRecentModels\(updated\)/);
});

/* ── provider keys panel ────────────────────────────────────────────────── */

test("a connected provider reports its model count with a refresh control", () => {
    assert.match(settings, /Connected · \{modelCount\(providerInfo\.id\)\} models available/);
    assert.match(settings, /aria-label="Re-check key and refresh models"/);
    assert.match(settings, /revalidateProvider\(providerInfo\.id\)/);
});

test("the refresh control is an icon that spins while working, not a text link", () => {
    assert.match(settings, /<RefreshCw className=\{`w-3 h-3 \$\{busy === `\$\{providerInfo\.id\}:refresh` \? "animate-spin" : ""\}`\}/);
    assert.doesNotMatch(settings, />\s*Refresh models\s*</, "the label is a tooltip, not visible text");
});

test("Test is merged into the refresh control", () => {
    assert.doesNotMatch(settings, /const testKey =/, "the separate Test handler is gone");
    assert.doesNotMatch(settings, /"Testing\.\.\." : "Test"/, "the Test button is gone");
    // The merged control uses the endpoint that does BOTH jobs.
    assert.match(
        settings,
        /revalidateProvider[\s\S]{0,600}\/api\/provider-keys\/\$\{provider\}\/test/,
        "re-validating the key is what also refreshes the catalog",
    );
});

test("saving, testing or deleting a key invalidates the shared catalog", () => {
    const reloads = settings.match(/void reloadCatalog\(\);/g) || [];
    assert.equal(reloads.length, 3, "save, test and delete should each refresh the catalog");
});

test("an invalid key reports inline and keeps the input populated", () => {
    assert.match(settings, /setProviderError\(prev => \(\{ \.\.\.prev, \[provider\]: text \}\)\)/);
    assert.match(settings, /\{providerError\[providerInfo\.id\] && \(/);
    assert.match(
        settings,
        /if \(!res\.ok\) throw new Error\(data\.detail \|\| "Save failed\."\);\s*\n\s*setApiKeys/,
        "the input is only cleared after a successful save",
    );
});
