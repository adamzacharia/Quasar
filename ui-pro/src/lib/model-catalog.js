/**
 * Model-catalog presentation logic: fuzzy search, grouping, and price lines.
 *
 * Kept as plain JS beside the other src/lib logic modules so
 * `node --test tests/*.test.mjs` can exercise it without a React or DOM
 * harness — the scorer and the grouping rules are the parts with sharp edges,
 * so they are the parts that need tests. Types live in model-catalog.d.ts.
 *
 * Wire shape comes from GET /api/models/available
 * (services/provider_models.py :: AvailableModels).
 */

/** Section order and labels, preserved from the pre-catalog dropdown. */
export const PROVIDER_ORDER = ["openai", "deepseek", "tacc", "anthropic", "google", "local"];

export const PROVIDER_LABELS = {
    openai: "OpenAI",
    deepseek: "DeepSeek",
    tacc: "TACC Tejas",
    anthropic: "Anthropic Claude",
    google: "Google Gemini",
    local: "Local LLM",
};

/** Short names used in the empty state and the "add a key" hint. */
export const PROVIDER_SHORT_LABELS = {
    openai: "OpenAI",
    deepseek: "DeepSeek",
    tacc: "TACC",
    anthropic: "Anthropic",
    google: "Google Gemini",
    local: "Local",
};

/**
 * Friendly query -> id-substring table. These exist for terms a user types that
 * are not literally in the model id (or that should outrank an incidental
 * substring hit elsewhere). Matching is case-insensitive.
 */
export const MODEL_ALIASES = {
    opus: ["claude-opus"],
    sonnet: ["claude-sonnet"],
    haiku: ["claude-haiku"],
    fable: ["claude-fable"],
    claude: ["claude-"],
    anthropic: ["claude-"],
    gpt: ["gpt-"],
    "4o": ["gpt-4o"],
    "4.1": ["gpt-4.1"],
    mini: ["-mini"],
    nano: ["-nano"],
    oss: ["gpt-oss"],
    flash: ["flash"],
    pro: ["-pro"],
    r1: ["deepseek-reasoner", "deepseek-r1"],
    reasoner: ["deepseek-reasoner"],
    v4: ["deepseek-v4"],
    gemini: ["gemini-"],
    google: ["gemini-", "gemma-"],
    deepseek: ["deepseek"],
};

const SCORE_EXACT = 1000;
const SCORE_PREFIX = 800;
const SCORE_ALIAS = 700;
const SCORE_WORD_START = 600;
const SCORE_SUBSTRING = 400;
const SCORE_SUBSEQUENCE = 200;
/** A provider-name hit is weaker evidence than a model-id hit. */
const PROVIDER_WEIGHT = 0.45;

const WORD_BOUNDARY = /[-_./ :]/;

function isWordStart(text, index) {
    return index === 0 || WORD_BOUNDARY.test(text.charAt(index - 1));
}

/** Merge sorted, possibly adjacent character indices into [start, end) runs. */
function toRanges(indices) {
    const ranges = [];
    for (const index of indices) {
        const last = ranges[ranges.length - 1];
        if (last && last[1] === index) last[1] = index + 1;
        else ranges.push([index, index + 1]);
    }
    return ranges;
}

/**
 * Score one query against one string, case-insensitively.
 * Ordering is prefix > word-start > substring > subsequence, per the spec.
 * Returns score 0 (and no ranges) when the query does not match at all.
 */
export function fuzzyMatch(query, text) {
    const q = String(query || "").trim().toLowerCase();
    const raw = String(text || "");
    const t = raw.toLowerCase();
    if (!q) return { score: 0, ranges: [] };
    if (!t) return { score: 0, ranges: [] };

    if (t === q) return { score: SCORE_EXACT, ranges: [[0, raw.length]] };
    if (t.startsWith(q)) return { score: SCORE_PREFIX, ranges: [[0, q.length]] };

    // Best literal occurrence: prefer one that starts a word.
    let best = -1;
    let bestWordStart = false;
    for (let i = t.indexOf(q); i !== -1; i = t.indexOf(q, i + 1)) {
        const wordStart = isWordStart(t, i);
        if (best === -1 || (wordStart && !bestWordStart)) {
            best = i;
            bestWordStart = wordStart;
        }
        if (bestWordStart) break;
    }
    if (best !== -1) {
        return {
            score: bestWordStart ? SCORE_WORD_START : SCORE_SUBSTRING,
            ranges: [[best, best + q.length]],
        };
    }

    // Subsequence: every query char in order, penalised by how scattered it is.
    const hits = [];
    let cursor = 0;
    for (const char of q) {
        const found = t.indexOf(char, cursor);
        if (found === -1) return { score: 0, ranges: [] };
        hits.push(found);
        cursor = found + 1;
    }
    const span = hits[hits.length - 1] - hits[0] + 1;
    const penalty = Math.min(SCORE_SUBSEQUENCE - 1, span - q.length);
    return { score: SCORE_SUBSEQUENCE - penalty, ranges: toRanges(hits) };
}

/** Alias hit: the query names a family whose id substring this model carries. */
function aliasMatch(query, modelId) {
    const q = String(query || "").trim().toLowerCase();
    const patterns = Object.prototype.hasOwnProperty.call(MODEL_ALIASES, q)
        ? MODEL_ALIASES[q]
        : null;
    if (!patterns) return { score: 0, ranges: [] };
    const id = String(modelId || "").toLowerCase();
    for (const pattern of patterns) {
        const at = id.indexOf(pattern);
        if (at !== -1) return { score: SCORE_ALIAS, ranges: [[at, at + pattern.length]] };
    }
    return { score: 0, ranges: [] };
}

/**
 * Score a model against a query across its id, display name, provider label
 * and the alias table. Ranges are reported per field so the UI can highlight
 * the matched characters in whichever string it renders.
 */
export function scoreModel(query, model, providerLabel) {
    const q = String(query || "").trim();
    if (!q) return { score: 0, idRanges: [], nameRanges: [] };

    const id = model && model.id ? model.id : "";
    const displayName = model && model.displayName ? model.displayName : "";

    const byId = fuzzyMatch(q, id);
    // Always scored, even when it equals the id: the display name is the line
    // the row actually renders, so it needs its own highlight ranges.
    const byName = displayName ? fuzzyMatch(q, displayName) : { score: 0, ranges: [] };
    const byAlias = aliasMatch(q, id);
    const byProvider = fuzzyMatch(q, providerLabel || (model && model.provider) || "");

    const score = Math.max(
        byId.score,
        byName.score,
        byAlias.score,
        Math.round(byProvider.score * PROVIDER_WEIGHT),
    );
    if (score <= 0) return { score: 0, idRanges: [], nameRanges: [] };

    // Highlight what the user actually typed whenever it occurs in the id; the
    // alias pattern is only highlighted when it is the sole reason for the hit
    // (e.g. "r1" matching deepseek-reasoner), never in place of a literal hit.
    const idRanges = byId.score > 0 ? byId.ranges : byAlias.ranges;
    return { score, idRanges, nameRanges: byName.ranges };
}

/** Price line for a model row, or null when no line should be rendered. */
export function priceLine(model) {
    if (!model) return null;
    // TACC is included-quota and has never shown a price line; local is free.
    if (model.provider === "tacc") return null;
    if (model.provider === "local") return "In: $0/M · Out: $0/M";
    const input = model.inputPricePerM;
    const output = model.outputPricePerM;
    if (typeof input === "number" && typeof output === "number") {
        return `In: $${input}/M · Out: $${output}/M`;
    }
    return "Pricing unknown";
}

/** Context-window chip text, e.g. 1000000 -> "1M ctx". */
export function contextChip(model) {
    const window = model && model.contextWindow;
    if (typeof window !== "number" || !Number.isFinite(window) || window <= 0) return null;
    if (window >= 1000000) {
        const millions = window / 1000000;
        return `${Number.isInteger(millions) ? millions : millions.toFixed(1)}M ctx`;
    }
    if (window >= 1000) return `${Math.round(window / 1000)}K ctx`;
    return `${window} ctx`;
}

/** Row label: local models drop their routing prefix, as they always have. */
export function modelLabel(model) {
    if (!model) return "";
    const name = model.displayName || model.id || "";
    return name.startsWith("local/") ? name.slice("local/".length) : name;
}

function orderProviders(providers) {
    const known = [];
    for (const provider of PROVIDER_ORDER) {
        const match = providers.find((entry) => entry && entry.provider === provider);
        if (match) known.push(match);
    }
    const rest = providers.filter((entry) => entry && !PROVIDER_ORDER.includes(entry.provider));
    return [...known, ...rest];
}

/** A provider is selectable when it is connected or carries included quota. */
export function isSelectable(catalog) {
    return !!catalog && (catalog.status === "connected" || catalog.status === "included_quota");
}

/**
 * Build the dropdown's groups for a query.
 *
 * Selectable providers become model groups; empty groups are hidden while
 * searching. Providers with no key and no included quota never contribute
 * selectable rows — they are returned separately as locked rows so the UI can
 * collapse them at the bottom.
 */
export function buildGroups(query, providers, options) {
    const list = Array.isArray(providers) ? providers : [];
    const q = String(query || "").trim();
    const recentIds = (options && options.recent) || [];
    const groups = [];

    for (const catalog of orderProviders(list)) {
        if (!isSelectable(catalog)) continue;
        const label = PROVIDER_LABELS[catalog.provider] || catalog.provider;
        const rows = [];
        for (const model of catalog.models || []) {
            if (!q) {
                rows.push({ model, score: 0, idRanges: [], nameRanges: [] });
                continue;
            }
            const scored = scoreModel(q, model, label);
            if (scored.score > 0) rows.push({ model, ...scored });
        }
        if (q) rows.sort((a, b) => b.score - a.score);
        if (rows.length === 0) continue;
        groups.push({
            key: catalog.provider,
            provider: catalog.provider,
            label,
            byok: catalog.status === "connected",
            stale: !!catalog.stale,
            rows,
        });
    }

    // "Recent" only makes sense as an unfiltered shortcut list.
    if (!q && recentIds.length > 0) {
        const selectable = new Map();
        for (const catalog of list) {
            if (!isSelectable(catalog)) continue;
            for (const model of catalog.models || []) selectable.set(model.id, model);
        }
        const rows = [];
        for (const id of recentIds) {
            const model = selectable.get(id);
            if (model) rows.push({ model, score: 0, idRanges: [], nameRanges: [] });
        }
        if (rows.length > 0) {
            groups.unshift({ key: "recent", provider: "recent", label: "Recent", byok: false, stale: false, rows });
        }
    }

    const locked = orderProviders(list)
        .filter((catalog) => catalog && catalog.status === "not_connected")
        .map((catalog) => ({
            provider: catalog.provider,
            label: PROVIDER_SHORT_LABELS[catalog.provider] || catalog.provider,
            unlockCount: catalog.unlockCount || 0,
        }));

    return { groups, locked };
}

/** Flattened selectable rows, in render order — the keyboard navigation model. */
export function flattenRows(groups) {
    const flat = [];
    for (const group of groups) {
        for (const row of group.rows) {
            flat.push({ groupKey: group.key, model: row.model });
        }
    }
    return flat;
}

/** Providers currently offering models, for the empty-state sentence. */
export function connectedLabels(providers) {
    return (Array.isArray(providers) ? providers : [])
        .filter((catalog) => isSelectable(catalog) && (catalog.models || []).length > 0)
        .map((catalog) => PROVIDER_SHORT_LABELS[catalog.provider] || catalog.provider);
}

/** Which provider a family substring belongs to, for locked-provider hints. */
function providerOwnsAlias(provider, pattern) {
    if (provider === "anthropic") return pattern.startsWith("claude");
    if (provider === "google") return pattern.startsWith("gemini") || pattern.startsWith("gemma");
    if (provider === "deepseek") return pattern.startsWith("deepseek");
    if (provider === "openai") return pattern.startsWith("gpt") || pattern.startsWith("o1");
    return false;
}

/**
 * A not-connected provider the query appears to be naming, so the empty state
 * can offer the key link instead of a dead end. Matches the provider's own
 * name ("anthropic", "gemini") or a family term from the alias table ("opus").
 */
export function suggestedLockedProvider(query, providers) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return null;
    const aliasPatterns = Object.prototype.hasOwnProperty.call(MODEL_ALIASES, q)
        ? MODEL_ALIASES[q]
        : [];
    for (const catalog of Array.isArray(providers) ? providers : []) {
        if (!catalog || catalog.status !== "not_connected") continue;
        const label = PROVIDER_SHORT_LABELS[catalog.provider] || catalog.provider;
        const names = [String(catalog.provider).toLowerCase(), label.toLowerCase()];
        const namesMatch = names.some((name) => name.includes(q) || q.includes(name));
        const aliasOwns = aliasPatterns.some((pattern) => providerOwnsAlias(catalog.provider, pattern));
        if (namesMatch || aliasOwns) {
            return { provider: catalog.provider, label, unlockCount: catalog.unlockCount || 0 };
        }
    }
    return null;
}

export const RECENT_MODELS_KEY = "quasar_recent_models";
export const RECENT_LIMIT = 3;

/** Read the recent-model ids. SSR- and private-mode-safe, like the other prefs. */
export function loadRecentModels() {
    if (typeof window === "undefined") return [];
    try {
        const raw = window.localStorage.getItem(RECENT_MODELS_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed.filter((id) => typeof id === "string" && id).slice(0, RECENT_LIMIT);
    } catch {
        return [];
    }
}

/** Most-recent-first, de-duplicated, capped at RECENT_LIMIT. */
export function nextRecentModels(current, modelId) {
    const id = typeof modelId === "string" ? modelId : "";
    const list = Array.isArray(current) ? current.filter((entry) => typeof entry === "string" && entry) : [];
    if (!id) return list.slice(0, RECENT_LIMIT);
    return [id, ...list.filter((entry) => entry !== id)].slice(0, RECENT_LIMIT);
}

export function persistRecentModels(ids) {
    if (typeof window === "undefined") return;
    try {
        window.localStorage.setItem(RECENT_MODELS_KEY, JSON.stringify(ids.slice(0, RECENT_LIMIT)));
    } catch {
        /* private mode */
    }
}
