// One-click MCP connectors for Settings > MCP servers.
//
// Every entry goes through the same Connect flow as a pasted URL; nothing
// here bypasses the backend's detection. Only servers checked live are
// listed. `verified` says how far the check went:
//   "tool-call"     connected and a real tool call returned a result
//   "tools-listed"  connected and listed tools (no call made)
//   "oauth-discovery"  answered 401 with OAuth metadata, the sign-in server's
//                      settings validated and it offers dynamic client
//                      registration (no account was signed in)
//
// Logos are the provider's own favicon, fetched at view time the same way the
// Research timeline shows source icons; no brand assets are bundled.
//
// Plain JS so ui-pro/tests/*.test.mjs can import it.

export const PRESETS_VERIFIED_ON = "2026-10-01";

export const MCP_PRESETS = [
    {
        id: "deepwiki", name: "DeepWiki", url: "https://mcp.deepwiki.com/mcp", domain: "deepwiki.com",
        auth: "none", verified: "tool-call", blurb: "Ask questions about any public GitHub repo",
    },
    {
        id: "context7", name: "Context7", url: "https://mcp.context7.com/mcp", domain: "context7.com",
        auth: "none", verified: "tool-call", blurb: "Up-to-date library and API docs",
    },
    {
        id: "huggingface", name: "Hugging Face", url: "https://huggingface.co/mcp", domain: "huggingface.co",
        auth: "none", verified: "tool-call", blurb: "Search models, datasets and papers on the Hub",
    },
    {
        id: "mslearn", name: "Microsoft Learn", url: "https://learn.microsoft.com/api/mcp", domain: "learn.microsoft.com",
        auth: "none", verified: "tool-call", blurb: "Search Microsoft and Azure documentation",
    },
    {
        id: "gitmcp", name: "GitHub repo docs", url: "https://gitmcp.io/{owner}/{repo}", domain: "gitmcp.io",
        auth: "none", verified: "tool-call", needsRepo: true, blurb: "Docs and code search for one repo (gitmcp)",
    },
    {
        id: "monocrawl", name: "Monocrawl", url: "https://www.monocrawl.com/mcp/oauth", domain: "monocrawl.com",
        auth: "oauth", verified: "oauth-discovery", blurb: "Web data APIs; sign in with your Monocrawl account",
    },
    {
        id: "notion", name: "Notion", url: "https://mcp.notion.com/mcp", domain: "notion.so",
        auth: "oauth", verified: "oauth-discovery", blurb: "Read and write your Notion pages",
    },
    {
        id: "linear", name: "Linear", url: "https://mcp.linear.app/mcp", domain: "linear.app",
        auth: "oauth", verified: "oauth-discovery", blurb: "Issues and projects in Linear",
    },
];

// Astronomy and space servers from the official MCP Registry. Each one was
// connected and answered a real tool call on ASTRONOMY_PRESETS_VERIFIED_ON
// (moon phase, Kp index, arXiv search, Earthdata collections, a dark-sky
// spot). All are keyless. They add things Quasar does not have built in;
// third-party wrappers of archives Quasar already covers (SIMBAD, ADS, MAST,
// ...) are left out on purpose. `builtinOverlap` names a partial overlap.
export const ASTRONOMY_PRESETS_VERIFIED_ON = "2026-10-02";

export const ASTRONOMY_PRESETS = [
    {
        id: "astronomy-calc", name: "Astronomy calculator", url: "https://astronomy.caseyjhand.com/mcp", domain: "caseyjhand.com",
        auth: "none", verified: "tool-call", blurb: "Sky positions, rise and set times, moon phases, eclipses, what is visible tonight",
    },
    {
        id: "noaa-spaceweather", name: "NOAA space weather", url: "https://noaa-spaceweather.caseyjhand.com/mcp", domain: "caseyjhand.com",
        auth: "none", verified: "tool-call", blurb: "Kp index, solar wind, flares, alerts and aurora forecasts from NOAA SWPC data (third-party server)",
    },
    {
        id: "arxiv-fulltext", name: "arXiv full text", url: "https://arxiv.caseyjhand.com/mcp", domain: "caseyjhand.com",
        auth: "none", verified: "tool-call", builtinOverlap: "arXiv search",
        blurb: "Read the full text of arXiv papers (Quasar already searches arXiv; this adds reading the paper itself)",
    },
    {
        id: "nasa-earthdata", name: "NASA Earthdata", url: "https://cmr.earthdata.nasa.gov/mcp/v1", domain: "earthdata.nasa.gov",
        auth: "none", verified: "tool-call", official: true, blurb: "NASA's own server for Earth science datasets and granules (CMR)",
    },
    {
        id: "star-ninja", name: "Star Ninja", url: "https://stars.2pm.ninja/mcp", domain: "2pm.ninja",
        auth: "none", verified: "tool-call", blurb: "Sky darkness, cloud cover and moonless hours for an observing spot",
    },
];

const ALL_PRESETS = [...MCP_PRESETS, ...ASTRONOMY_PRESETS];

// Where a provider hands out API keys, for servers that answer "needs an API
// key" (checked 2026-10-01 from the provider's own docs).
const KEY_PAGES = {
    "www.monocrawl.com": "https://www.monocrawl.com/dashboard/api/keys",
    "monocrawl.com": "https://www.monocrawl.com/dashboard/api/keys",
};

/** The provider's API key page for a server URL, when known. */
export function apiKeyPageFor(url) {
    try { return KEY_PAGES[new URL(url).hostname.toLowerCase()] || null; } catch { return null; }
}

/** gitmcp.io URL for "owner/repo" or a github.com URL; null when unusable. */
export function gitmcpUrl(input) {
    const raw = String(input || "").trim().replace(/\.git$/, "").replace(/\/+$/, "");
    const m = raw.match(/^(?:https?:\/\/)?(?:www\.)?(?:github\.com\/)?([A-Za-z0-9_.-]+)\/([A-Za-z0-9_.-]+)$/);
    if (!m) return null;
    return `https://gitmcp.io/${m[1]}/${m[2]}`;
}

/** The preset a saved server came from (same URL), for its key link and icon. */
export function presetForUrl(url) {
    const u = String(url || "").replace(/\/+$/, "").toLowerCase();
    return ALL_PRESETS.find((p) => p.url.replace(/\/+$/, "").toLowerCase() === u)
        || (u.startsWith("https://gitmcp.io/") ? MCP_PRESETS.find((p) => p.id === "gitmcp") : undefined);
}

/** Domain to fetch a favicon for (preset domain, else the URL's host). */
export function iconDomain(url) {
    const preset = presetForUrl(url);
    if (preset) return preset.domain;
    try { return new URL(url).hostname.replace(/^(www|mcp|api)\./, ""); } catch { return ""; }
}
