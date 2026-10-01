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
    return MCP_PRESETS.find((p) => p.url.replace(/\/+$/, "").toLowerCase() === u)
        || (u.startsWith("https://gitmcp.io/") ? MCP_PRESETS.find((p) => p.id === "gitmcp") : undefined);
}

/** Domain to fetch a favicon for (preset domain, else the URL's host). */
export function iconDomain(url) {
    const preset = presetForUrl(url);
    if (preset) return preset.domain;
    try { return new URL(url).hostname.replace(/^(www|mcp|api)\./, ""); } catch { return ""; }
}
