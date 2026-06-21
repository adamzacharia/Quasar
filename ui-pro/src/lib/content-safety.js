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
    /\bxxx\b/i,
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

export function looksExplicitWebText(value) {
    const text = String(value || "");
    return Boolean(text) && EXPLICIT_PATTERNS.some(pattern => pattern.test(text));
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

export function isSafeWebImage(image) {
    // General-web image pixels are not verifiable from URL metadata alone.
    void image;
    return false;
}
