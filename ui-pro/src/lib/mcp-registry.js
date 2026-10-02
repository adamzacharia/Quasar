// Registry search results for Settings > MCP servers.
//
// The backend (GET /api/mcp-registry/search) returns hosted servers from the
// official MCP Registry. Listings are published by their authors and nobody
// vets them, so every result is labelled, and a click goes through the same
// Connect flow as a pasted URL. This file decides what a click does and
// which labels a result gets.
//
// Plain JS so ui-pro/tests/*.test.mjs can import it.

import { ASTRONOMY_PRESETS, MCP_PRESETS } from "./mcp-presets.js";
import { hasPlaceholder } from "./mcp-paste.js";

export const REGISTRY_HOME = "https://registry.modelcontextprotocol.io";

/** Same endpoint: scheme and host compared case-insensitively, path and
 *  query exactly (they can be case-sensitive tenant ids); a trailing slash
 *  does not count. */
export function sameEndpoint(a, b) {
    const key = (s) => {
        try {
            const u = new URL(String(s || ""));
            return `${u.protocol}//${u.host.toLowerCase()}${u.pathname.replace(/\/+$/, "")}${u.search}`;
        } catch { return null; }
    };
    const ka = key(a);
    return ka !== null && ka === key(b);
}

/** The checked preset with exactly this URL (never a pattern such as any
 *  gitmcp.io repo: only the exact endpoint was checked). */
function checkedPresetFor(url) {
    return [...MCP_PRESETS, ...ASTRONOMY_PRESETS].find((p) => !p.needsRepo && sameEndpoint(p.url, url));
}

/** A short server name for the Name field (registry titles can be long). */
export function registryName(item) {
    const title = String(item?.title || "").replace(/\s+[-|:—–].*$/, "").trim();
    return (title || String(item?.id || "").split("/").pop() || "server").slice(0, 40);
}

/** "{api-key}" -> "<api-key>": one placeholder form the paste parser always
 *  recognises, whatever characters the registry used inside the braces. */
const angleize = (s) => String(s).replace(/\{([^{}]+)\}/g, "<$1>");

const looksLikeKey = (name) => /authorization|api[-_]?key|token|secret|password/i.test(name);

/**
 * The value put in the box for one required header:
 *   fixed   a literal value from the registry (sent as is, e.g. a version)
 *   key     a placeholder the API key field replaces, keeping a scheme such
 *           as "Basic <token>" from the template
 *   other   a placeholder for a non-secret value the user adds under Advanced
 */
function headerEntry(h) {
    const tpl = h.value_template ? angleize(h.value_template) : null;
    if (h.secret || looksLikeKey(h.name)) {
        // A secret is always asked for, never taken from a public listing,
        // even when the listing carries a literal value (guard CX-16). A
        // literal "Bearer xyz" keeps only its scheme.
        if (tpl && hasPlaceholder(tpl)) return { value: tpl, kind: "key" };
        const scheme = tpl && /^([A-Za-z][A-Za-z0-9_-]*)\s+\S/.exec(tpl);
        if (scheme) return { value: `${scheme[1]} <your key>`, kind: "key" };
        return { value: h.name.toLowerCase() === "authorization" ? "Bearer <your key>" : "<your key>", kind: "key" };
    }
    if (tpl && !hasPlaceholder(tpl)) return { value: tpl, kind: "fixed" };
    return { value: tpl || "<value>", kind: "other" };
}

/** What a listing's required headers ask of the user (one classification
 *  shared by the Set up config and the badges). */
export function headerNeeds(item) {
    const kinds = (item.headers || []).filter((h) => h.required && h.name).map((h) => headerEntry(h).kind);
    return { key: kinds.includes("key"), other: kinds.includes("other") };
}

/**
 * What clicking a result does:
 *   { kind: "connect", url, name }          connect right away (like a pasted URL)
 *   { kind: "fill", input, note, focusKey } put a config in the box for the user
 *                                            to finish: an address to fill in,
 *                                            a key, or required headers
 */
export function registryAction(item) {
    const url = String(item.url || "");
    // The backend flag, or anything the paste parser would call a placeholder.
    const templated = !!item.templated || hasPlaceholder(url.replace(/^https?:\/\//i, ""));
    const required = (item.headers || []).filter((h) => h.required && h.name);
    if (!templated && !required.length) return { kind: "connect", url, name: registryName(item) };

    const name = registryName(item);
    const vars = [...url.matchAll(/\{([^{}]+)\}|<([^<>]+)>/g)].map((m) => m[1] || m[2]);
    const addressNote = templated ? `Fill in ${vars.length ? vars.join(", ") : "the marked part"} in the address.` : "";
    if (!required.length) {
        return { kind: "fill", input: url, focusKey: false, note: `${name} needs part of its address filled in. ${addressNote} Then press Connect.` };
    }
    // Key headers first, so the paste parser picks one of them for the key field.
    const entries = required.map((h) => ({ h, ...headerEntry(h) }));
    const order = { key: 0, other: 1, fixed: 2 };
    entries.sort((a, b) => order[a.kind] - order[b.kind]);
    const headers = {};
    for (const e of entries) headers[e.h.name] = e.value;
    const input = JSON.stringify({ url, type: item.transport === "sse" ? "sse" : "http", headers }, null, 2);
    // The paste parser puts the first placeholder header (Authorization if
    // there is one) in the API key field; any other placeholder must be
    // added under Advanced. Mirror that here so the note matches the form.
    const placeholders = entries.filter((e) => e.kind !== "fixed");
    const field = placeholders.find((e) => e.h.name.toLowerCase() === "authorization") || placeholders[0];
    const advanced = placeholders.filter((e) => e !== field);
    const fixed = entries.filter((e) => e.kind === "fixed");
    const hint = field && field.h.description;
    const note = [
        `${name} needs some setup.`,
        addressNote,
        field ? `Paste the value for ${field.h.name} in the API key field.` : "",
        advanced.length ? `Add real values for ${advanced.map((e) => e.h.name).join(", ")} under Advanced, Request headers.` : "",
        fixed.length ? `${fixed.map((e) => e.h.name).join(", ")} ${fixed.length === 1 ? "is" : "are"} filled in from the registry listing.` : "",
        "Then press Connect.",
        hint ? `(${hint})` : "",
    ].filter(Boolean).join(" ");
    return { kind: "fill", input, focusKey: !!field && !templated, note };
}

/**
 * Labels for one result, most important first. `tone` drives the colour:
 * ok (checked by Quasar / already added), info, warn (needs setup), muted.
 */
export function registryBadges(item, servers = []) {
    const out = [];
    const preset = checkedPresetFor(item.url);
    if (preset) out.push({ text: "Checked by Quasar", tone: "ok" });
    if ((servers || []).some((s) => sameEndpoint(s.url, item.url))) out.push({ text: "Added", tone: "ok" });
    // From the headers themselves: a fixed value needs nothing (guard CX-03).
    const needs = headerNeeds(item);
    if (needs.key) out.push({ text: "Needs API key", tone: "warn" });
    else if (needs.other) out.push({ text: "Needs setup", tone: "warn" });
    if (item.templated) out.push({ text: "Address to fill in", tone: "warn" });
    if (!preset) out.push({ text: "Unverified", tone: "muted" });
    return out;
}

/** "Quasar already has SIMBAD and MAST built in..." or "". A checked preset
 *  that adds something beyond the built-in says what, instead. */
export function overlapNote(item) {
    const labels = item.overlaps || [];
    if (!labels.length) return "";
    const preset = checkedPresetFor(item.url);
    if (preset && preset.builtinOverlap) return `Quasar already has ${preset.builtinOverlap} built in. ${preset.blurb}.`;
    const list = labels.length === 1 ? labels[0] : `${labels.slice(0, -1).join(", ")} and ${labels[labels.length - 1]}`;
    return `Quasar already has ${list} built in. Check the description for anything this adds before connecting it.`;
}

/** Query string for the search endpoint. */
export function registrySearchParams(mode, cursor) {
    const p = new URLSearchParams();
    if (mode.kind === "topic") p.set("topic", mode.topic);
    else p.set("q", mode.q.trim());
    if (cursor && mode.kind !== "topic") p.set("cursor", cursor);
    return p.toString();
}

/** Merge a "Load more" page into what is shown, without repeats. */
export function mergeResults(current, next) {
    const seen = new Set((current || []).map((it) => it.id));
    return [...(current || []), ...(next || []).filter((it) => !seen.has(it.id))];
}
