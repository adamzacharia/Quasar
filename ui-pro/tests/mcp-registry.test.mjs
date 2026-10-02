import test from "node:test";
import assert from "node:assert/strict";

import {
    registryAction, registryBadges, registryName, overlapNote, registrySearchParams, mergeResults, sameEndpoint,
} from "../src/lib/mcp-registry.js";
import { ASTRONOMY_PRESETS, ASTRONOMY_PRESETS_VERIFIED_ON, MCP_PRESETS, presetForUrl, iconDomain } from "../src/lib/mcp-presets.js";
import { parseMcpInput } from "../src/lib/mcp-paste.js";
import { buildConnectBody } from "../src/lib/mcp-connect.js";

const FORM = { advancedOpen: false, transport: "http", headerText: "", envText: "", cmd: "npx", cmdArgs: "",
    apiKey: "", keyHeader: undefined, name: "", nameTyped: false };

const item = (over = {}) => ({
    id: "io.github.example/thing-mcp", title: "Thing", description: "Does things", version: "1.0.0",
    url: "https://thing.example.com/mcp", transport: "streamable-http", headers: [], needs_key: false,
    templated: false, website: null, repository: null, overlaps: [], updated_at: null, ...over,
});

test("a plain hosted server connects right away", () => {
    assert.deepEqual(registryAction(item()), { kind: "connect", url: "https://thing.example.com/mcp", name: "Thing" });
});

test("long registry titles become short server names", () => {
    assert.equal(registryName(item({ title: "NASA Image of the Day — space & astronomy photos" })), "NASA Image of the Day");
    assert.equal(registryName(item({ title: "", id: "io.github.x/arxiv-mcp" })), "arxiv-mcp");
    assert.ok(registryName(item({ title: "x".repeat(90) })).length <= 40);
});

test("a required key header fills the box so the key field asks for it", () => {
    const act = registryAction(item({
        needs_key: true,
        headers: [{ name: "X-API-Key", required: true, secret: true, description: "Get one at example.com", value_template: null }],
    }));
    assert.equal(act.kind, "fill");
    assert.equal(act.focusKey, true);
    assert.match(act.note, /X-API-Key/);
    // The paste parser sees a placeholder and asks for the key in that header.
    const parsed = parseMcpInput(act.input);
    assert.equal(parsed.kind, "url");
    assert.equal(parsed.url, "https://thing.example.com/mcp");
    assert.equal(parsed.keyHeader, "X-API-Key");
    assert.deepEqual(parsed.headers, {}); // the placeholder itself is never sent
    // With a key typed, the key goes in that header, as typed.
    const built = buildConnectBody({ parsed, form: { ...FORM, apiKey: "k123", keyHeader: parsed.keyHeader } });
    assert.equal(built.body.headers["X-API-Key"], "k123");
});

test("an Authorization template becomes a Bearer key", () => {
    const act = registryAction(item({
        needs_key: true,
        headers: [{ name: "Authorization", required: true, secret: true, description: "", value_template: "Bearer {api_key}" }],
    }));
    const parsed = parseMcpInput(act.input);
    assert.equal(parsed.keyHeader, "Authorization");
    const built = buildConnectBody({ parsed, form: { ...FORM, apiKey: "abc", keyHeader: parsed.keyHeader } });
    assert.equal(built.body.headers.Authorization, "Bearer abc");
});

test("an address with parts to fill in is never connected as is", () => {
    const act = registryAction(item({ url: "https://gitmcp.io/{owner}/{repo}", templated: true }));
    assert.equal(act.kind, "fill");
    assert.match(act.note, /owner, repo/);
    assert.equal(parseMcpInput(act.input).kind, "invalid");
});

test("badges: checked presets, already added, needs setup, unverified", () => {
    const astro = ASTRONOMY_PRESETS[0];
    assert.deepEqual(registryBadges(item({ url: astro.url + "/" })).map((b) => b.text), ["Checked by Quasar"]);
    const added = registryBadges(item(), [{ url: "https://THING.example.com/mcp/" }]).map((b) => b.text);
    assert.deepEqual(added, ["Added", "Unverified"]);
    const key = registryBadges(item({ needs_key: true, templated: true, headers: [
        { name: "X-API-Key", required: true, secret: true, description: "", value_template: null }] })).map((b) => b.text);
    assert.deepEqual(key, ["Needs API key", "Address to fill in", "Unverified"]);
});

test("overlap note names the built-in archives", () => {
    assert.equal(overlapNote(item()), "");
    assert.match(overlapNote(item({ overlaps: ["SIMBAD"] })), /already has SIMBAD built in/);
    assert.match(overlapNote(item({ overlaps: ["SIMBAD", "MAST", "Gaia"] })), /SIMBAD, MAST and Gaia/);
});

test("search params: free text pages with a cursor, a topic never does", () => {
    assert.equal(registrySearchParams({ kind: "q", q: "  arxiv " }), "q=arxiv");
    assert.equal(registrySearchParams({ kind: "q", q: "a b" }, "c/1"), "q=a+b&cursor=c%2F1");
    assert.equal(registrySearchParams({ kind: "topic", topic: "astronomy" }, "ignored"), "topic=astronomy");
});

test("load more never repeats a server", () => {
    const a = item({ id: "a" }), b = item({ id: "b" }), b2 = item({ id: "b", title: "B again" });
    assert.deepEqual(mergeResults([a, b], [b2, item({ id: "c" })]).map((x) => x.id), ["a", "b", "c"]);
});

test("astronomy presets are https, keyless, dated, distinct from the popular row", () => {
    assert.match(ASTRONOMY_PRESETS_VERIFIED_ON, /^\d{4}-\d{2}-\d{2}$/);
    const popular = new Set(MCP_PRESETS.map((p) => p.id));
    const ids = new Set();
    for (const p of ASTRONOMY_PRESETS) {
        assert.ok(!ids.has(p.id) && !popular.has(p.id), p.id); ids.add(p.id);
        assert.match(p.url, /^https:\/\//);
        assert.equal(p.auth, "none");
        assert.equal(p.verified, "tool-call");
        assert.ok(p.domain && !p.domain.includes("/"));
        assert.equal(presetForUrl(p.url).id, p.id);
    }
    // A third-party server shows its publisher's icon, not the data provider's.
    assert.equal(iconDomain("https://noaa-spaceweather.caseyjhand.com/mcp"), "caseyjhand.com");
    assert.equal(iconDomain("https://cmr.earthdata.nasa.gov/mcp/v1"), "earthdata.nasa.gov");
});

// ── guard task-4558f74-244 regressions ──────────────────────────────────────

const keyFor = (act, apiKey) => {
    const parsed = parseMcpInput(act.input);
    assert.equal(parsed.kind, "url", JSON.stringify(parsed));
    return { parsed, built: buildConnectBody({ parsed, form: { ...FORM, apiKey, keyHeader: parsed.keyHeader, keyPrefix: parsed.keyPrefix } }) };
};

test("CX-01: a hyphenated {api-key} template is replaced by the typed key", () => {
    const act = registryAction(item({ needs_key: true, headers: [
        { name: "Authorization", required: true, secret: true, description: "", value_template: "Bearer {api-key}" }] }));
    const { built } = keyFor(act, "k1");
    assert.equal(built.body.headers.Authorization, "Bearer k1");
    assert.ok(!JSON.stringify(built.body).includes("api-key"));
});

test("CX-02: a Basic template keeps its scheme", () => {
    const act = registryAction(item({ needs_key: true, headers: [
        { name: "Authorization", required: true, secret: true, description: "", value_template: "Basic {token}" }] }));
    const { parsed, built } = keyFor(act, "dXNlcjpwYXNz");
    assert.equal(parsed.keyPrefix, "Basic ");
    assert.equal(built.body.headers.Authorization, "Basic dXNlcjpwYXNz");
    // Typed with the scheme already: sent as typed, not doubled.
    assert.equal(keyFor(act, "Basic abc").built.body.headers.Authorization, "Basic abc");
    // A pasted docs config gets the same treatment.
    const pasted = parseMcpInput('{"url":"https://x.example.com/mcp","headers":{"Authorization":"Token ${API_TOKEN}"}}');
    assert.equal(pasted.keyPrefix, "Token ");
});

test("CX-03: a required non-secret header with a fixed value is sent as is, not used for the key", () => {
    const act = registryAction(item({ needs_key: true, headers: [
        { name: "X-Client-Version", required: true, secret: false, description: "", value_template: "2026-01" },
        { name: "X-API-Key", required: true, secret: true, description: "", value_template: null }] }));
    const { parsed, built } = keyFor(act, "secret-1");
    assert.equal(parsed.keyHeader, "X-API-Key");
    assert.equal(built.body.headers["X-Client-Version"], "2026-01");
    assert.equal(built.body.headers["X-API-Key"], "secret-1");
    assert.match(act.note, /X-Client-Version is filled in from the registry/);
    // Only a fixed header: nothing to type, and no key field focus.
    const fixedOnly = registryAction(item({ needs_key: true, headers: [
        { name: "X-Client-Version", required: true, secret: false, description: "", value_template: "2026-01" }] }));
    assert.equal(fixedOnly.focusKey, false);
    assert.equal(parseMcpInput(fixedOnly.input).headers["X-Client-Version"], "2026-01");
});

test("CX-04: a templated URL keeps its required headers", () => {
    const act = registryAction(item({ url: "https://api.example.com/{tenant}/mcp", templated: true, needs_key: true, headers: [
        { name: "X-API-Key", required: true, secret: true, description: "", value_template: null }] }));
    assert.equal(act.focusKey, false); // the address comes first
    assert.equal(parseMcpInput(act.input).kind, "invalid");
    const filled = act.input.replace("{tenant}", "acme");
    const parsed = parseMcpInput(filled);
    assert.equal(parsed.url, "https://api.example.com/acme/mcp");
    assert.equal(parsed.keyHeader, "X-API-Key");
});

test("CX-05: a <tenant> address the backend did not flag is still never connected directly", () => {
    const act = registryAction(item({ url: "https://vendor.example/<tenant>/mcp", templated: false }));
    assert.equal(act.kind, "fill");
    assert.match(act.note, /tenant/);
});

test("CX-10: endpoints compare host case-insensitively and path/query exactly", () => {
    assert.ok(sameEndpoint("https://X.example.com/TenantA/mcp/", "https://x.example.com/TenantA/mcp"));
    assert.ok(!sameEndpoint("https://x.example.com/TenantA/mcp", "https://x.example.com/tenanta/mcp"));
    assert.ok(!sameEndpoint("https://x.example.com/mcp?t=A", "https://x.example.com/mcp?t=a"));
    assert.deepEqual(registryBadges(item({ url: "https://thing.example.com/MCP" }), [{ url: "https://thing.example.com/mcp" }]).map((b) => b.text), ["Unverified"]);
});

test("CX-11: the arXiv full-text preset says what it adds", () => {
    const arxiv = ASTRONOMY_PRESETS.find((p) => p.id === "arxiv-fulltext");
    const note = overlapNote(item({ url: arxiv.url, overlaps: ["arXiv search"] }));
    assert.match(note, /reading the paper itself/);
    assert.doesNotMatch(overlapNote(item({ overlaps: ["SIMBAD"] })), /mostly duplicates/);
});

test("CX-12: an unseen gitmcp repo is not 'Checked by Quasar'", () => {
    const texts = registryBadges(item({ url: "https://gitmcp.io/someone/unseen" })).map((b) => b.text);
    assert.ok(!texts.includes("Checked by Quasar"));
    assert.ok(texts.includes("Unverified"));
});

// ── guard verify round 1 reopen ─────────────────────────────────────────────

import { readFileSync } from "node:fs";
import { hasPlaceholder } from "../src/lib/mcp-paste.js";

test("CX-03: a fixed non-secret header is not 'Needs API key'", () => {
    const fixedOnly = item({ needs_key: false, headers: [
        { name: "X-Client-Version", required: true, secret: false, description: "", value_template: "2026-01" }] });
    assert.deepEqual(registryBadges(fixedOnly).map((b) => b.text), ["Unverified"]);
    const other = item({ headers: [{ name: "X-Region", required: true, secret: false, description: "", value_template: null }] });
    assert.deepEqual(registryBadges(other).map((b) => b.text), ["Needs setup", "Unverified"]);
});

test("CX-16: a secret header's literal listing value is never sent", () => {
    const act = registryAction(item({ needs_key: true, headers: [
        { name: "Authorization", required: true, secret: true, description: "", value_template: "Bearer registry-published-token" }] }));
    assert.ok(!act.input.includes("registry-published-token"));
    const { parsed, built } = keyFor(act, "mine");
    assert.equal(parsed.keyHeader, "Authorization");
    assert.equal(built.body.headers.Authorization, "Bearer mine");
    assert.doesNotMatch(act.note, /filled in from the registry/);
    assert.deepEqual(registryBadges(item({ headers: [
        { name: "X-Token", required: true, secret: false, description: "", value_template: "abc" }] })).map((b) => b.text),
    ["Needs API key", "Unverified"]);
});

test("CX-17: the paste parser and the backend agree on URL placeholders", () => {
    const cases = JSON.parse(readFileSync(new URL("./fixtures/mcp-placeholder-cases.json", import.meta.url), "utf8"));
    for (const u of cases.placeholder) assert.equal(hasPlaceholder(u), true, u);
    for (const u of cases.plain) assert.equal(hasPlaceholder(u), false, u);
    // A templated address with a colon inside the braces stays blocked.
    const act = registryAction(item({ url: "https://v.example/{tenant:slug}/mcp", templated: true }));
    assert.equal(parseMcpInput(act.input).kind, "invalid");
});
