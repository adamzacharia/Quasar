/**
 * Web evidence citations in the chat UI (web search redesign, Phase 1).
 *
 * The backend numbers every web page of a turn (W1, W2, ...) and the model
 * cites them inline as `[W2]`. This module is pure so node tests cover it:
 *
 *  - `normalizeWebSource`: one source from a `web_sources` event or from
 *    persisted `rich_meta.webSources`, keeping id / cited / date / domain.
 *  - `mergeTurnWebSources`: combine events of one turn. A `replace` event
 *    (the final, post-answer listing) wins; otherwise sources are merged by
 *    URL. Registry order (cited first) is kept when sources carry ids.
 *  - `tokenizeWebCitations`: `[W3]`, `[W1, W3]`, `[W1][W2]` -> markdown links
 *    `[3](#web-cite-W3)` that ChatMessage renders as citation chips. Code
 *    and math are never touched; paper references like `[1]` are not
 *    citations (the `W` prefix is what marks web evidence).
 *  - `splitCitedSources`, `webSourcesForTurn`, `citationLabel`.
 */

export const WEB_CITE_HREF_PREFIX = "#web-cite-";

function cleanUrl(value) {
    const raw = String(value || "").trim().replace(/^<|>$/g, "").replace(/[.,;:)\]}"'】]+$/g, "");
    if (!raw) return "";
    if (/^https?:\/\//i.test(raw)) return raw;
    if (raw.startsWith("www.")) return `https://${raw}`;
    if (/^[A-Za-z0-9.-]+\.[A-Za-z]{2,}\/\S+$/.test(raw)) return `https://${raw}`;
    return "";
}

export function domainOf(url) {
    try {
        return new URL(url).hostname.replace(/^www\./, "").replace(/^m\./, "");
    } catch {
        return "";
    }
}

function normalizeId(value) {
    const m = /^\s*W?\s*(\d{1,3})\s*$/i.exec(String(value ?? ""));
    return m ? `W${Number(m[1])}` : undefined;
}

/** One web source in the UI shape, or null when it has no usable URL. */
export function normalizeWebSource(item) {
    if (typeof item === "string") {
        const url = cleanUrl(item);
        return url ? { title: domainOf(url) || url, url, snippet: "", domain: domainOf(url) } : null;
    }
    if (!item || typeof item !== "object") return null;
    const url = cleanUrl(item.url || item.link || item.href || item.source_url || "");
    if (!url) return null;
    const out = {
        title: String(item.title || item.name || domainOf(url) || url).trim(),
        url,
        snippet: String(item.snippet || item.content || item.text || item.description || "").trim(),
    };
    const quality = item.evidenceQuality || item.evidence_quality;
    if (quality && typeof quality === "object") out.evidenceQuality = quality;
    const id = normalizeId(item.id);
    if (id) out.id = id;
    if (item.cited !== undefined) out.cited = Boolean(item.cited);
    const date = item.publishedDate || item.published_date || "";
    if (date) out.publishedDate = String(date);
    out.domain = String(item.domain || domainOf(url) || "");
    if (item.provider) out.provider = String(item.provider);
    return out;
}

const TRACKING_PARAMS = new Set(["fbclid", "gclid", "ref", "ref_src", "mc_cid", "mc_eid", "igshid", "yclid", "msclkid"]);

/** Same key as the backend's canonicalize_url: host case, www./m., http/https,
 *  tracking params, fragment and trailing slash do not matter; path case and
 *  the remaining query do (guard CX-21: ?id=1 and ?id=2 are different pages). */
export function urlKey(url) {
    try {
        const u = new URL(String(url || ""));
        let host = u.hostname.toLowerCase().replace(/\.$/, "");
        for (const prefix of ["www.", "m."]) {
            if (host.startsWith(prefix) && host.split(".").length >= 3) host = host.slice(prefix.length);
        }
        const port = u.port && !((u.protocol === "http:" && u.port === "80") || (u.protocol === "https:" && u.port === "443"))
            ? `:${u.port}` : "";
        const params = [...u.searchParams.entries()].filter(([k]) => !k.toLowerCase().startsWith("utm_") && !TRACKING_PARAMS.has(k.toLowerCase()));
        const query = params.length ? `?${new URLSearchParams(params).toString()}` : "";
        const path = u.pathname.replace(/\/{2,}/g, "/").replace(/\/+$/, "");
        return `${host}${port}${path}${query}`;
    } catch {
        return String(url || "").trim();
    }
}

function idNumber(source) {
    const m = /^W(\d+)$/.exec(source?.id || "");
    return m ? Number(m[1]) : Number.POSITIVE_INFINITY;
}

/** Cited sources first (in the order the backend listed them), then the rest by id. */
export function orderById(sources) {
    const cited = sources.filter((s) => s.cited);
    const rest = sources.filter((s) => !s.cited).sort((a, b) => idNumber(a) - idNumber(b));
    return [...cited, ...rest];
}

/**
 * Merge a `web_sources` event into the sources already attached to a turn.
 * `replace: true` (the final listing, after the answer) replaces the list so
 * the cited flags and order are exactly the backend's.
 */
export function mergeTurnWebSources(existing = [], incoming = [], options = {}) {
    const inc = (incoming || []).map(normalizeWebSource).filter(Boolean);
    if (options.replace && inc.length) return inc;
    const merged = [];
    const byKey = new Map();
    for (const source of [...(existing || []).map(normalizeWebSource).filter(Boolean), ...inc]) {
        const key = urlKey(source.url);
        if (byKey.has(key)) {
            const cur = merged[byKey.get(key)];
            for (const field of ["title", "snippet", "publishedDate", "provider", "domain", "evidenceQuality"]) {
                if (!cur[field] && source[field]) cur[field] = source[field];
            }
            if (!cur.id && source.id) cur.id = source.id;
            if (source.cited) cur.cited = true;
            continue;
        }
        byKey.set(key, merged.length);
        merged.push({ ...source });
    }
    return merged.some((s) => s.id) ? orderById(merged) : merged;
}

/** { cited, uncited } for the sources grid: cited in first-appearance order. */
export function splitCitedSources(sources = []) {
    const cited = [];
    const uncited = [];
    for (const s of sources || []) (s && s.cited ? cited : uncited).push(s);
    return { cited, uncited };
}

/** The display number of a citation id ("W3" -> "3"). */
export function citationLabel(id) {
    const m = /^W(\d+)$/i.exec(String(id || ""));
    return m ? m[1] : String(id || "");
}

/** All web sources attached to the turn that `messages[index]` belongs to. */
export function webSourcesForTurn(messages, index) {
    if (!Array.isArray(messages) || index < 0 || index >= messages.length) return [];
    let start = index;
    while (start > 0 && messages[start - 1]?.role !== "user") start -= 1;
    let end = index;
    while (end + 1 < messages.length && messages[end + 1]?.role !== "user") end += 1;
    let sources = [];
    for (let i = start; i <= end; i += 1) {
        const m = messages[i];
        if (m?.role === "assistant" && m?.type === "web_sources" && Array.isArray(m.webSources)) {
            sources = mergeTurnWebSources(sources, m.webSources);
        }
    }
    return sources;
}

/** True when the sources come from the grounded pipeline (they carry W# ids). */
export function hasEvidenceIds(sources) {
    return Array.isArray(sources) && sources.some((s) => s && /^W\d+$/.test(s.id || ""));
}

/** True when the turn of `messages[index]` has an assistant text message. */
export function turnHasTextMessage(messages, index) {
    if (!Array.isArray(messages)) return false;
    for (let j = index; j >= 0 && messages[j]?.role !== "user"; j--) {
        if (messages[j]?.role === "assistant" && messages[j]?.type === "text") return true;
    }
    for (let j = index + 1; j < messages.length && messages[j]?.role !== "user"; j++) {
        if (messages[j]?.role === "assistant" && messages[j]?.type === "text") return true;
    }
    return false;
}

/**
 * How a `web_sources` message renders. Grounded turns (sources with W# ids)
 * show their sources inside the answer message, so the separate message is
 * hidden, except for its web image grid, which only this message carries
 * (guard CX-20). Returns "full" | "images-only" | "hide".
 */
export function webSourcesMessageView(messages, index) {
    const m = Array.isArray(messages) ? messages[index] : null;
    if (!m || m.type !== "web_sources") return "full";
    if (!hasEvidenceIds(webSourcesForTurn(messages, index)) || !turnHasTextMessage(messages, index)) return "full";
    return Array.isArray(m.webImages) && m.webImages.length ? "images-only" : "hide";
}

// ── markdown tokenization ──────────────────────────────────────────────────

const FENCE_RE = /^ {0,3}(`{3,}|~{3,})/;
const INLINE_CODE_RE = /(`+)(?:[^\n]|\n(?![ \t]*\n))+?\1/g;
const MATH_RE = /\$\$[\s\S]*?\$\$|(?<![\\$])\$[^$\n]+\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)/g;
// Not a citation: an existing link "[W1](https://...)" or an escaped "\[W1\]" (guard CX-22).
const CITE_GROUP_RE = /(?<!\\)\[\s*W\s*\d{1,3}(?:\s*[,;]\s*W?\s*\d{1,3})*\s*\](?!\()/gi;
const INDENTED_RE = /^(?: {4}|\t)/;
const LIST_ITEM_RE = /^\s*(?:[-*+]|\d+[.)])\s/;

function mapProse(text, fn) {
    // Split on fenced and indented code blocks first (line based; the same
    // indented-code rule as the backend: 4+ spaces / tab after a blank line,
    // not a list item), then protect inline code and math with placeholders.
    const lines = String(text || "").split("\n");
    const out = [];
    let buf = [];
    let fence = null;
    let prevBlank = true;
    let inIndented = false;
    const flush = () => {
        if (!buf.length) return;
        out.push(transformProse(buf.join("\n"), fn));
        buf = [];
    };
    for (const line of lines) {
        const m = FENCE_RE.exec(line);
        const blank = !line.trim();
        if (fence) {
            out.push(line);
            if (m && m[1][0] === fence[0] && m[1].length >= fence.length && line.trim() === m[1].trim()) fence = null;
            prevBlank = blank;
            continue;
        }
        if (m) {
            flush();
            fence = m[1];
            out.push(line);
            prevBlank = blank;
            continue;
        }
        if (!blank && INDENTED_RE.test(line) && !LIST_ITEM_RE.test(line) && (prevBlank || inIndented)) {
            flush();
            out.push(line);
            inIndented = true;
            prevBlank = false;
            continue;
        }
        if (!blank) inIndented = false;
        buf.push(line);
        prevBlank = blank;
    }
    flush();
    return out.join("\n");
}

function transformProse(prose, fn) {
    const saved = [];
    const protect = (s) => {
        saved.push(s);
        return `\u0000${saved.length - 1}\u0000`;
    };
    let t = prose.replace(INLINE_CODE_RE, protect).replace(MATH_RE, protect);
    t = fn(t);
    return t.replace(/\u0000(\d+)\u0000/g, (_, i) => saved[Number(i)]);
}

/**
 * `[W3]` -> `[3](#web-cite-W3)`; `[W1, W3]` -> two adjacent chip links.
 * `knownIds` (optional Set/array): ids outside it are left as plain text so a
 * stale or invented tag never renders as a working chip.
 */
export function tokenizeWebCitations(text, knownIds) {
    const known = knownIds ? new Set([...knownIds].map((k) => String(k).toUpperCase())) : null;
    return mapProse(text, (prose) =>
        prose.replace(CITE_GROUP_RE, (group) => {
            const ids = [...group.matchAll(/\d{1,3}/g)].map((m) => `W${Number(m[0])}`);
            const usable = known ? ids.filter((id) => known.has(id)) : ids;
            if (!usable.length) return group;
            return [...new Set(usable)].map((id) => `[${citationLabel(id)}](${WEB_CITE_HREF_PREFIX}${id})`).join("");
        }),
    );
}

/** The citation id carried by a chip link href, or null. */
export function citationIdFromHref(href) {
    const h = String(href || "");
    if (!h.startsWith(WEB_CITE_HREF_PREFIX)) return null;
    return normalizeId(h.slice(WEB_CITE_HREF_PREFIX.length)) || null;
}

/** DOM id of a source card in the sources grid (scroll target of a chip). */
export function sourceCardDomId(messageId, citationId) {
    return `web-src-${String(messageId || "m").replace(/[^A-Za-z0-9_-]/g, "")}-${citationId}`;
}

const WEB_SEARCH_MODES = ["off", "auto", "always"];

/** off | auto | always; anything else falls back to `fallback` (default auto). */
export function normalizeWebSearchMode(value, fallback = "auto") {
    const v = String(value ?? "").trim().toLowerCase();
    return WEB_SEARCH_MODES.includes(v) ? v : fallback;
}

/** One `web_decision` event (or persisted rich_meta.webDecision) in the UI
 *  shape, or null when it carries no decision. */
export function normalizeWebDecision(raw) {
    if (!raw || typeof raw !== "object" || typeof raw.need_web !== "boolean") return null;
    const queries = Array.isArray(raw.queries)
        ? raw.queries.map((q) => String(q || "").trim()).filter(Boolean).slice(0, 3)
        : [];
    const out = { need_web: raw.need_web, queries };
    if (raw.mode) out.mode = normalizeWebSearchMode(raw.mode, String(raw.mode));
    if (raw.reason) out.reason = String(raw.reason).slice(0, 160);
    if (raw.source) out.source = String(raw.source);
    if (typeof raw.planner_ms === "number") out.planner_ms = raw.planner_ms;
    if (raw.domain_pack) out.domain_pack = String(raw.domain_pack);
    if (raw.freshness) out.freshness = String(raw.freshness);
    if (raw.follow_up !== undefined) out.follow_up = Boolean(raw.follow_up);
    return out;
}

/** Badge text under the answer: "Searched the web (2 queries)" or
 *  "Web skipped: answered from ALMA archive". */
export function webDecisionLabel(decision) {
    const d = normalizeWebDecision(decision);
    if (!d) return "";
    if (d.need_web) {
        const n = d.queries.length;
        return n > 0 ? `Searched the web (${n} ${n === 1 ? "query" : "queries"})` : "Searched the web";
    }
    return d.reason ? `Web skipped: ${d.reason}` : "Web skipped";
}

/** Tooltip for the badge: the real queries, the pack and the planner time. */
export function webDecisionTitle(decision) {
    const d = normalizeWebDecision(decision);
    if (!d) return "";
    const parts = [];
    if (d.queries.length) parts.push(d.queries.map((q) => `"${q}"`).join("\n"));
    if (d.domain_pack && d.domain_pack !== "general") parts.push(`official sites: ${d.domain_pack}`);
    if (d.freshness && d.freshness !== "any") parts.push(`freshness: ${d.freshness}`);
    if (d.follow_up) parts.push("follow-up resolved from the conversation");
    if (d.reason && d.need_web) parts.push(d.reason);
    if (typeof d.planner_ms === "number") parts.push(`planner ${(d.planner_ms / 1000).toFixed(1)} s`);
    return parts.join("\n");
}
