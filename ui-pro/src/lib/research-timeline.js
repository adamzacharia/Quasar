// Research timeline: turns the flat stream of backend status steps into the
// grouped, Perplexity-style view the chat shows while a turn is working
// ("Searching the web" query chips, "Reading" source chips, "Querying"
// archive chips, "Wrapping up").
//
// The backend already streams human-readable labels (core/agent.py
// _tool_status_label / _web_tool_status_label, core/runner.py web pre-pass),
// so this only classifies and groups them; nothing here invents a step the
// backend did not report, and no reported step is dropped except plumbing.
//
// Plain JS (not TS) so ui-pro/tests/*.test.mjs can import it directly,
// same pattern as active-phase.js.

import { domainOf } from "./web-citations.js";
import { isSafeWebSource } from "./content-safety.js";

/**
 * ADS papers returned in the turn that `messages[index]` belongs to (same
 * turn bounds as web-citations.js webSourcesForTurn).
 */
export function turnPaperCount(messages, index) {
    if (!Array.isArray(messages) || index < 0 || index >= messages.length) return 0;
    let start = index;
    while (start > 0 && messages[start - 1]?.role !== "user") start -= 1;
    let n = 0;
    for (let i = start; i < messages.length; i += 1) {
        const m = messages[i];
        if (i > index && m?.role === "user") break;
        if (m?.role === "assistant" && m?.type === "papers" && Array.isArray(m.papers)) n += m.papers.length;
    }
    return n;
}

/** Steps that are plumbing, not research the user should read. */
const HIDDEN_STEP_RES = [
    /^__/,                                  // internal liveness / data events
    /^connecting to quasar engine$/i,
    /^calling \S+$/i,                       // "Calling gpt-oss-120b" (the model call)
    /^generating response$/i,
    /^🧠\s*reasoning$/iu,                   // reasoning marker; its lines are shown instead
    /^complex query detected\b/i,           // the workforce widget shows this turn's plan
];

const WRAP_RE = /^(composing (the )?final answer|adding the requested details|tool budget reached|resuming after provider cutoff|provider stream interrupted)/i;
// Pre-pass query: Searching the web: "q" [(official sites)]
const WEB_QUERY_RE = /^searching the web:\s*"?(.+?)"?(\s*\(official sites\))?$/i;
// Pre-pass umbrella sentence (opened before, closed after the queries). The
// planner's queries say more than it does, so it is replaced by them.
const WEB_PREPASS_RE = /^(🌐|searching the web (for|in parallel)\b)/iu;
const WEB_TIMEOUT_RE = /^web search timed out\b/i;
// In-loop web tools (agent.py _web_tool_status_label), optionally with a
// ' ("hint")' suffix naming the query or URL.
const WEB_TOOL_RE = /\bweb search\b|\bweb research\b|\bweb source\b|\bmapping website\b|\bcrawling website\b/i;
const HINT_RE = /\s*\("(.+)"\)$/;
const READING_RE = /^reading \d+ pages?$/i;
const NOTICE_RE = /\bis degraded\b|\bunavailable\b.*\busing\b/i;
const TOOL_CALL_RE = /^calling tool:\s*/i;
const REASON_RE = /^💭\s*/u;

/** How many reasoning lines show before "Show N earlier thoughts". */
export const REASONING_VISIBLE = 3;

/**
 * Icon hint for a chip. The component maps it to a lucide icon; keeping the
 * keyword logic here lets the tests cover it.
 * @param {string} text
 */
export function stepIcon(text) {
    const t = String(text || "").toLowerCase();
    if (/literature|papers?\b|\bads\b|researcher|manuals?\b|documentation|handbook|reference|knowledge base/.test(t)) return "book";
    if (/download/.test(t)) return "download";
    if (/image|cutout|render|overlay|stamps|hips|rgb|panel/.test(t)) return "image";
    if (/plot|light curve|spectrum|spectral|moment map|fit|period|sed\b|variability|\bmmdc\b.*\bmodel\b/.test(t)) return "chart";
    if (/cross-?match/.test(t)) return "merge";
    if (/sql|tap|catalog|archive|query|querying|search|listing|counting|inspecting|fetching/.test(t)) return "database";
    return "spark";
}

/**
 * Dedupe steps by label, keeping the LAST status. The persisted history
 * (rich_meta.thinkingSteps) records every transition, so a reloaded turn has
 * both "running" and "completed" entries for the same label.
 * @param {Array<{text: string, status: string}>} steps
 */
function dedupeSteps(steps) {
    const order = [];
    const byText = new Map();
    for (const s of steps) {
        if (!s || typeof s.text !== "string") continue;
        const text = s.text.trim();
        if (!text) continue;
        if (!byText.has(text)) order.push(text);
        byText.set(text, { ...s, text });
    }
    return order.map((t) => byText.get(t));
}

function normalizeStatus(status, turnRunning) {
    if (status === "error") return "error";
    if (status === "running" && turnRunning) return "running";
    return "completed";
}

const RANK = { running: 2, completed: 1, error: 0 };

/** Push a chip, merging an identical one (status: running > completed > error). */
function pushUnique(items, item) {
    const i = items.findIndex((x) => x.text === item.text);
    if (i < 0) items.push(item);
    else if (RANK[item.status] > RANK[items[i].status]) items[i] = { ...items[i], ...item };
}

/**
 * Build the grouped timeline.
 *
 * @param {object} input
 * @param {Array<{text: string, status: string, elapsedSeconds?: number}>} [input.steps]
 * @param {boolean} [input.running]      the turn is still streaming
 * @param {{need_web?: boolean, queries?: string[]}} [input.webDecision]
 * @param {Array<{url: string, title?: string}>} [input.webSources]
 * @param {number} [input.paperCount]    ADS papers returned this turn
 * @returns {{
 *   sections: Array<{kind: string, title: string, status: string,
 *     items: Array<{text: string, status: string, icon: string, url?: string, domain?: string, elapsedSeconds?: number}>,
 *     overflow?: number, hiddenCount?: number}>,
 *   sourceCount: number,
 *   stepCount: number,
 *   current: string | undefined,
 *   favicons: string[],
 * }}
 */
export function buildResearchTimeline(input = {}) {
    const running = Boolean(input.running);
    const raw = Array.isArray(input.steps) ? input.steps : [];
    const steps = dedupeSteps(raw).filter((s) => !HIDDEN_STEP_RES.some((re) => re.test(s.text)));

    /** @type {Map<string, {kind: string, title: string, items: any[], order: number, generic?: any[]}>} */
    const sections = new Map();
    let order = 0;
    const section = (kind, title) => {
        if (!sections.has(kind)) sections.set(kind, { kind, title, items: [], order: order++ });
        return sections.get(kind);
    };

    const isNamedTool = (t) => !TOOL_CALL_RE.test(t) && !REASON_RE.test(t) && !WRAP_RE.test(t)
        && !WEB_QUERY_RE.test(t) && !WEB_PREPASS_RE.test(t) && !WEB_TOOL_RE.test(t) && !WEB_TIMEOUT_RE.test(t)
        && !READING_RE.test(t) && !NOTICE_RE.test(t);
    const hasNamedTool = steps.some((s) => isNamedTool(s.text));
    let readingStatus = null;
    let readingLabel = "";
    let current;

    for (const s of steps) {
        const status = normalizeStatus(s.status, running);
        const text = s.text;
        const base = { status, ...(typeof s.elapsedSeconds === "number" ? { elapsedSeconds: s.elapsedSeconds } : {}) };
        if (status === "running") current = text.replace(REASON_RE, "").replace(/^🌐\s*/u, "");

        if (REASON_RE.test(text)) {
            const line = text.replace(REASON_RE, "").trim();
            if (line) section("reasoning", "Thinking").items.push({ ...base, text: line, icon: "spark" });
        } else if (READING_RE.test(text)) {
            readingStatus = status;
            readingLabel = text.replace(/^reading\s+/i, "");
            section("reading", "Reading");
        } else if (WEB_QUERY_RE.test(text)) {
            // The pre-pass runs a query twice (official sites, then open web):
            // one chip per query.
            const q = WEB_QUERY_RE.exec(text)[1].trim();
            pushUnique(section("searching", "Searching the web").items, { ...base, text: q, icon: "search" });
        } else if (WEB_TIMEOUT_RE.test(text)) {
            // runner.py _close_web_step: the pass gave up; say so, as a failure.
            section("searching", "Searching the web").items.push({ ...base, status: "error", text, icon: "search" });
        } else if (WEB_PREPASS_RE.test(text)) {
            const sec = section("searching", "Searching the web");
            (sec.generic = sec.generic || []).push({ ...base, text: text.replace(/^🌐\s*/u, ""), icon: "search" });
        } else if (WEB_TOOL_RE.test(text)) {
            // An in-loop web tool is its own step: show its query/URL hint.
            const hint = HINT_RE.exec(text);
            pushUnique(section("searching", "Searching the web").items, { ...base, text: hint ? hint[1] : text, icon: "search" });
        } else if (WRAP_RE.test(text)) {
            section("wrapping", "Wrapping up").items.push({ ...base, text, icon: "spark" });
        } else if (NOTICE_RE.test(text)) {
            section("notice", "Heads up").items.push({ ...base, status: status === "running" ? "running" : "completed", text, icon: "alert" });
        } else if (TOOL_CALL_RE.test(text)) {
            // Post-hoc "Calling tool: X" duplicates the archive-aware label.
            if (!hasNamedTool) section("querying", "Querying").items.push({ ...base, text: text.replace(TOOL_CALL_RE, ""), icon: stepIcon(text) });
        } else {
            section("querying", "Querying").items.push({ ...base, text, icon: stepIcon(text) });
        }
    }

    // Searching: the umbrella sentence becomes the planner's real queries
    // when there are no per-query labels; it keeps the section live while
    // open, and its failure is never hidden.
    const queries = [...new Set(
        (Array.isArray(input.webDecision?.queries) ? input.webDecision.queries : [])
            .map((q) => String(q || "").trim()).filter(Boolean),
    )];
    const searching = sections.get("searching");
    if (searching) {
        const generic = searching.generic || [];
        const gRunning = generic.some((g) => g.status === "running");
        const gFailed = !gRunning && generic.some((g) => g.status === "error");
        if (!searching.items.length && queries.length && generic.length) {
            const st = gRunning ? "running" : gFailed ? "error" : "completed";
            for (const q of queries) pushUnique(searching.items, { text: q, status: st, icon: "search" });
        } else if (!searching.items.length || gFailed) {
            for (const g of generic) searching.items.push(g);
        }
        if (gRunning) searching.live = true;
        delete searching.generic;
    }

    // Reading: the turn's actual sources, as favicon chips. Chips are one per
    // site; the count is per page, matching the answer's "View all N sources".
    // Sources the other views withhold (content-safety) are withheld here too.
    const seen = new Set();
    const urls = new Set();
    const sourceItems = [];
    for (const src of Array.isArray(input.webSources) ? input.webSources : []) {
        if (!isSafeWebSource(src)) continue;
        if (src.url) urls.add(src.url);
        const domain = domainOf(src.url || "");
        if (!domain || seen.has(domain)) continue;
        seen.add(domain);
        sourceItems.push({ text: domain.replace(/^www\./, ""), status: "completed", icon: "favicon", url: src.url, domain });
    }
    const paperCount = Number(input.paperCount) || 0;
    if (sourceItems.length || paperCount || sections.has("reading")) {
        const reading = section("reading", "Reading");
        reading.items = sourceItems.slice(0, 6);
        if (sourceItems.length > 6) reading.overflow = sourceItems.length - 6;
        if (paperCount) reading.items.push({ text: `${paperCount} paper${paperCount === 1 ? "" : "s"} from ADS`, status: "completed", icon: "book" });
        if (!reading.items.length && readingStatus) {
            // Sources not known yet (still streaming): name what is being read.
            reading.items.push({ text: readingLabel || "pages", status: readingStatus, icon: "page" });
        }
        // Reading comes right after the searches that produced it.
        const anchor = sections.get("searching") || sections.get("querying");
        if (anchor) reading.order = anchor.order + 0.5;
    }

    // Wrapping up always closes the timeline.
    const wrapping = sections.get("wrapping");
    if (wrapping) wrapping.order = Number.MAX_SAFE_INTEGER;

    // Reasoning keeps every line; the component shows the most recent ones
    // and lets the reader expand the rest.
    const reasoning = sections.get("reasoning");
    if (reasoning && reasoning.items.length > REASONING_VISIBLE) {
        reasoning.hiddenCount = reasoning.items.length - REASONING_VISIBLE;
    }

    const out = [...sections.values()]
        .filter((s) => s.items.length > 0)
        .sort((a, b) => a.order - b.order)
        .map(({ kind, title, items, overflow, hiddenCount, live }) => ({
            kind,
            title,
            items,
            status: live || items.some((i) => i.status === "running")
                ? "running"
                : items.some((i) => i.status === "error") && items.every((i) => i.status !== "completed")
                    ? "error"
                    : "completed",
            ...(overflow ? { overflow } : {}),
            ...(hiddenCount ? { hiddenCount } : {}),
        }));

    const stepCount = out.reduce((n, s) => n + (s.kind === "reading" ? 0 : s.items.length), 0);
    return {
        sections: out,
        sourceCount: urls.size + paperCount,
        stepCount,
        current,
        favicons: sourceItems.slice(0, 4).map((i) => i.domain),
    };
}
