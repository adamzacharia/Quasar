const BLOCKED_HOSTS = new Set([
    "4chan.org",
    "adultfriendfinder.com",
    "bangbros.com",
    "brazzers.com",
    "chaturbate.com",
    "erome.com",
    "fansly.com",
    "hclips.com",
    "hentaihaven.xxx",
    "livejasmin.com",
    "manyvids.com",
    "motherless.com",
    "nhentai.net",
    "naughtyamerica.com",
    "onlyfans.com",
    "playboy.com",
    "pornhub.com",
    "porntrex.com",
    "redgifs.com",
    "redtube.com",
    "rule34.xxx",
    "sex.com",
    "spankbang.com",
    "stripchat.com",
    "theporndude.com",
    "xhamster.com",
    "xnxx.com",
    "xvideos.com",
    "youporn.com",
]);

const EXPLICIT_PATTERNS = [
    /\b(?:hardcore|softcore)\s+(?:porn|video|scene|content)\b/i,
    /\b(?:porn|pornographic|pornography)\b/i,
    /\bnsfw\b/i,
    /\bhentai\b/i,
    /\brule\s*34\b/i,
    /\b(?:adult|porn)\s+(?:actor|actress|star|performer|film|movie|video|site|website|content|entertainment)\b/i,
    /\b(?:nude|nudity|naked)\s+(?:photo|image|picture|video|scene|content|model)\b/i,
    /\b(?:explicit|graphic)\s+sexual\s+(?:content|image|video|material|scene)\b/i,
    /\bsexual(?:ly)?\s+explicit\b/i,
    /\berotic(?:a)?\b/i,
    /\bsex\s+(?:tape|video|scene|cam|chat|show|site|website)\b/i,
    /\b(?:camgirl|cam-girl|webcam model)\b/i,
    /\b(?:escort|hookup)\s+(?:site|service|directory)\b/i,
    /\b(?:onlyfans|pornhub|xvideos|xnxx|xhamster|redtube|youporn|spankbang)\b/i,
];

function hostFromUrl(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    try {
        const normalized = /^https?:\/\//i.test(raw) ? raw : `https://${raw.replace(/^\/+/, "")}`;
        return new URL(normalized).hostname.toLowerCase().replace(/^www\./, "");
    } catch {
        return "";
    }
}

export function isBlockedWebUrl(value) {
    const host = hostFromUrl(value);
    if (!host) return false;
    return [...BLOCKED_HOSTS].some(blocked => host === blocked || host.endsWith(`.${blocked}`));
}

// "xxx" is explicit EXCEPT as a Roman numeral in a paper series ("Planck
// intermediate results. XXX." is the main BICEP2 rebuttal); mirrors
// services/content_safety.py _explicit_xxx (2026-10-03).
// The ONLY exemption: the Planck paper series ("Planck 2013
// results. XXX." or "Planck intermediate results. XXX."; mirrors
// services/content_safety.py (guard CX-19 rounds 1-4).
const ROMAN_XXX_CONTEXT = /\bPlanck\s+(?:(?:19|20)\d{2}|intermediate|early|legacy)\s+results\.\s*$/i;
const ADULT_NEXT_WORDS = new Set(["explicit", "hot", "gallery", "galleries", "adult", "nude", "naked", "video", "videos", "movie", "movies", "clip", "clips", "pics", "photos", "porn",
    "sex", "girls", "cams", "cam", "tube", "site", "sites", "content", "stream", "streams"]);

// Numbering punctuation right after the numeral ("XXX." / "XXX:"), so
// "results XXX videos" stays explicit (guard CX-19).
// A period must start a capitalised title ("XXX. The angular ...") so
// "XXX. videos." stays explicit (guard CX-19 reopen).
const ROMAN_XXX_AFTER = /^(?:\.\s+[A-Z]|:\s*\S)/;
// Adult wording just before it makes it explicit whatever follows (guard
// CX-19 round 3: "Adult search results. XXX.").
const ADULT_BEFORE = /\b(?:adult|porn\w*|sex\w*|nsfw|nude|naked|erotic\w*|hentai|escort)\b/i;

function explicitXxx(text) {
    for (const m of text.matchAll(/\bxxx\b/gi)) {
        const before = text.slice(Math.max(0, m.index - 60), m.index);
        const after = text.slice(m.index + m[0].length, m.index + m[0].length + 6);
        const nextWord = (text.slice(m.index + m[0].length, m.index + m[0].length + 30).match(/[A-Za-z]+/) || [""])[0];
        if (m[0] === m[0].toUpperCase() && ROMAN_XXX_CONTEXT.test(before) && ROMAN_XXX_AFTER.test(after)
            && !ADULT_NEXT_WORDS.has(nextWord.toLowerCase())
            && !ADULT_BEFORE.test(text.slice(Math.max(0, m.index - 60), m.index))) continue;
        return true;
    }
    return false;
}

export function looksExplicitWebText(value) {
    const text = String(value || "");
    return Boolean(text) && (EXPLICIT_PATTERNS.some(pattern => pattern.test(text)) || explicitXxx(text));
}

export function safeAssistantWebText(value) {
    const text = String(value || "");
    return looksExplicitWebText(text)
        ? "Some web results were withheld by the safety filter."
        : text;
}

export function isSafeWebSource(source) {
    if (!source || typeof source !== "object") return false;
    const text = [
        source.title,
        source.name,
        source.snippet,
        source.content,
        source.text,
        source.description,
    ].map(value => String(value || "")).join(" ");
    return !isBlockedWebUrl(source.url || source.link || source.href || source.source_url)
        && !looksExplicitWebText(text);
}

function isDisplayableWebImageUrl(value) {
    const raw = String(value || "").trim();
    return /^https?:\/\//i.test(raw) || /^\/\//.test(raw);
}

export function isSafeWebImage(image) {
    if (!image || typeof image !== "object") return false;
    const url = image.url || image.src || image.image_url;
    if (!isDisplayableWebImageUrl(url)) return false;

    const text = [
        image.description,
        image.alt,
        image.title,
        image.sourceTitle,
        image.source_title,
        image.sourcePageTitle,
        image.source_page_title,
        url,
        image.sourceUrl,
        image.source_url,
    ].map(value => String(value || "")).join(" ");

    return !isBlockedWebUrl(url)
        && !isBlockedWebUrl(image.sourceUrl || image.source_url || image.source || "")
        && !looksExplicitWebText(text);
}
