// "Paste a server URL" box for Settings > MCP servers.
//
// People paste whatever their MCP server's README shows: a URL, a Claude
// Desktop / Cursor / VS Code JSON block, an `npx ...` command, a
// `claude mcp add ...` line. This turns any of those into one of:
//
//   { kind: "url",   url, name?, headers?, transport?, note?, keyHeader? }   connect it
//   { kind: "local", command, message, alternative? }            hosted site cannot run it
//   { kind: "empty" } | { kind: "invalid", message }
//
// Plain JS (not TS) so ui-pro/tests/*.test.mjs can import it directly.

const LOCAL_ONLY_MESSAGE =
    "This starts a program on your own computer, which the hosted Quasar site cannot do.";

// Local packages that have a hosted server Quasar can connect to instead.
// Only entries whose hosted URL was checked live (see mcp-presets.js date).
export const HOSTED_ALTERNATIVES = [
    { match: /(^|\/)monocrawl-mcp(@|$)|^@monocrawl\//i, name: "Monocrawl", url: "https://www.monocrawl.com/mcp/oauth" },
    { match: /^@upstash\/context7-mcp(@|$)/i, name: "Context7", url: "https://mcp.context7.com/mcp" },
    { match: /(^|\/)(mcp-)?deepwiki(-mcp)?(@|$)/i, name: "DeepWiki", url: "https://mcp.deepwiki.com/mcp" },
];

// Commands that spawn a local program. `npx` and friends are the common ones.
const LAUNCHERS = new Set(["npx", "uvx", "bunx", "pnpx", "node", "python", "python3", "uv", "pipx",
    "docker", "deno", "bun", "pnpm", "yarn", "npm", "go", "cargo", "java"]);

// Packages that only bridge to a REMOTE server: the URL inside is what we want.
const REMOTE_BRIDGES = new Set(["mcp-remote", "add-mcp", "@modelcontextprotocol/mcp-remote", "supergateway"]);

const URL_RE = /https?:\/\/[^\s"'<>`]+/i;

function stripTrailingPunct(url) {
    return url.replace(/[),.;\]]+$/, "");
}

function looksLikeBareHost(text) {
    // "mcp.deepwiki.com/mcp" (no scheme, no spaces, a dot in the host part)
    return /^[a-z0-9-]+(\.[a-z0-9-]+)+(:\d+)?(\/\S*)?$/i.test(text);
}

// Two-label public suffixes common enough to matter for a display name
// (example.co.uk is "Example", not "Co").
const TWO_LEVEL_SUFFIXES = new Set(["co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "edu.au", "co.jp", "ac.jp", "co.nz", "co.in", "com.br", "co.za", "com.cn", "com.mx"]);

/** Display name for a server URL: gitmcp.io/owner/repo -> "repo", www.monocrawl.com -> "Monocrawl". */
export function deriveServerName(url) {
    let u;
    try { u = new URL(url); } catch { return ""; }
    const host = u.hostname.toLowerCase();
    if (host === "gitmcp.io") {
        const parts = u.pathname.split("/").filter(Boolean);
        if (parts.length >= 2) return parts[1];
        if (parts.length === 1) return parts[0];
    }
    const labels = host.split(".").filter((p) => p && !["www", "mcp", "api"].includes(p));
    const suffixLen = labels.length >= 3 && TWO_LEVEL_SUFFIXES.has(labels.slice(-2).join(".")) ? 2 : 1;
    const core = labels.length > suffixLen ? labels[labels.length - suffixLen - 1] : (labels[0] || "server");
    return core.charAt(0).toUpperCase() + core.slice(1);
}

/**
 * A name for `url` that does not silently overwrite a DIFFERENT saved server
 * (saving an existing name updates that server). Same URL = same server, so
 * its name is reused. gitmcp repos fall back to "owner-repo", others get " 2".
 */
export function uniqueServerName(base, url, servers) {
    const taken = new Map((servers || []).map((s) => [String(s.name), String(s.url || "")]));
    const same = (n) => taken.get(n) === url;
    if (!taken.has(base) || same(base)) return base;
    try {
        const u = new URL(url);
        const parts = u.pathname.split("/").filter(Boolean);
        if (u.hostname.toLowerCase() === "gitmcp.io" && parts.length >= 2) {
            const alt = `${parts[0]}-${parts[1]}`;
            if (!taken.has(alt) || same(alt)) return alt;
        }
    } catch { /* fall through */ }
    for (let i = 2; i < 100; i += 1) {
        const candidate = `${base} ${i}`;
        if (!taken.has(candidate) || same(candidate)) return candidate;
    }
    return `${base} ${Date.now()}`;
}

/** Split a shell-ish command line, honoring simple quotes. */
export function splitCommand(line) {
    const out = [];
    let cur = "";
    let quote = null;
    let has = false;
    for (const ch of String(line)) {
        if (quote) {
            if (ch === quote) quote = null; else cur += ch;
            continue;
        }
        if (ch === "\"" || ch === "'") { quote = ch; has = true; continue; }
        if (/\s/.test(ch)) {
            if (has || cur) out.push(cur);
            cur = ""; has = false;
            continue;
        }
        cur += ch; has = true;
    }
    if (has || cur) out.push(cur);
    return out;
}

function headersFromArgs(args) {
    // mcp-remote style: --header "Authorization: Bearer ${TOKEN}"
    const headers = {};
    for (let i = 0; i < args.length; i += 1) {
        const a = args[i];
        if ((a === "--header" || a === "-H") && args[i + 1]) {
            const h = args[i + 1];
            const j = h.indexOf(":");
            if (j > 0) headers[h.slice(0, j).trim()] = h.slice(j + 1).trim();
            i += 1;
        }
    }
    return headers;
}

function packageOf(args) {
    // "-y @upstash/context7-mcp@latest" -> "@upstash/context7-mcp@latest";
    // "uvx --from pkg cmd" / "npx -p pkg cmd" name the package explicitly.
    for (let i = 0; i < args.length; i += 1) {
        const a = args[i];
        if ((a === "--from" || a === "--package" || a === "-p") && args[i + 1]) return args[i + 1];
        if (a.startsWith("--from=") || a.startsWith("--package=")) return a.slice(a.indexOf("=") + 1);
        if (a.startsWith("-")) continue;
        return a;
    }
    return "";
}

/** "@scope/name@1.2" -> "@scope/name", "name@latest" -> "name", "pkg==1.0" -> "pkg". */
export function stripVersion(pkg) {
    const p = String(pkg || "").replace(/(==|>=|~=).*$/, "");
    const at = p.startsWith("@") ? p.indexOf("@", 1) : p.indexOf("@");
    return at > 0 ? p.slice(0, at) : p;
}

function alternativeFor(pkg) {
    if (!pkg) return undefined;
    const hit = HOSTED_ALTERNATIVES.find((alt) => alt.match.test(pkg));
    return hit ? { name: hit.name, url: hit.url } : undefined;
}

export function hasPlaceholder(value) {
    // Kept identical to _PLACEHOLDER_RE in services/mcp_registry.py (both
    // test suites run the same cases).
    return /\$\{[^}]*\}|\$[A-Z_][A-Z0-9_]*|<[^>]+>|\{[^{}]*\}|YOUR_|\.\.\./i.test(String(value || ""));
}

/** "Basic " from "Basic ${TOKEN}" / "Token <key>": the scheme a key header's
 *  placeholder sits behind, so the typed key is sent with that scheme (not
 *  Bearer). Only a single word followed by whitespace counts. */
function schemePrefix(value) {
    const m = String(value || "").match(/^([A-Za-z][A-Za-z0-9_-]*)\s+(?=\S)/);
    if (!m) return undefined;
    return hasPlaceholder(m[1]) ? undefined : `${m[1]} `;
}

/**
 * Real header values, and the name of the first header whose value is only a
 * placeholder ("Bearer ${TOKEN}", "YOUR_API_KEY"). Placeholder values are
 * never submitted; the form asks for the key for that header instead.
 */
function splitHeaders(headers) {
    const out = {};
    const placeholders = [];
    for (const [k, v] of Object.entries(headers || {})) {
        if (!k || typeof v !== "string") continue;
        if (hasPlaceholder(v)) { placeholders.push(k); continue; }
        out[k] = v;
    }
    // The key field fills Authorization if the config has one, else the first.
    const keyHeader = placeholders.find((k) => k.toLowerCase() === "authorization") || placeholders[0];
    const keyPrefix = keyHeader ? schemePrefix(headers[keyHeader]) : undefined;
    return { headers: out, keyHeader, keyPrefix, placeholders };
}

/** The url result, or an invalid one when the URL itself is a template. */
function urlResult(url, name, rawHeaders, extra = {}) {
    if (hasPlaceholder(url.replace(/^https?:\/\//i, ""))) {
        return { kind: "invalid", message: `The server URL still has a placeholder in it (${url}). Replace it with the real address first.` };
    }
    const { headers, keyHeader, keyPrefix, placeholders } = splitHeaders(rawHeaders);
    const out = { kind: "url", url, name: name || deriveServerName(url), headers, ...extra };
    if (keyHeader) {
        out.keyHeader = keyHeader;
        if (keyPrefix) out.keyPrefix = keyPrefix;
        out.placeholderHeaders = placeholders;
        const others = placeholders.filter((k) => k !== keyHeader);
        out.note = [
            `This config expects your key in the ${keyHeader} header. Paste it in the API key field.`,
            // Never silently dropped: every placeholder header is named.
            others.length ? `It also has placeholder values for ${others.join(", ")}; those are not sent, so add real values under Advanced, Request headers.` : "",
            extra.note,
        ].filter(Boolean).join(" ");
    }
    return out;
}

/** A command (array form) from a JSON config or a pasted command line. */
function fromCommand(command, args, nameHint) {
    const pkg = packageOf(args);
    const bareName = stripVersion(pkg);
    if (REMOTE_BRIDGES.has(bareName)) {
        const url = args.map((a) => (a.match(URL_RE) || [])[0]).find(Boolean);
        if (url) return urlResult(stripTrailingPunct(url), nameHint, headersFromArgs(args));
    }
    if (!command) return { kind: "invalid", message: "That does not look like an MCP server URL or config." };
    return {
        kind: "local",
        command: [command, ...args].filter(Boolean).join(" "),
        message: LOCAL_ONLY_MESSAGE,
        alternative: alternativeFor(bareName || pkg),
    };
}

function fromServerEntry(entry, nameHint) {
    if (!entry || typeof entry !== "object") return null;
    const url = entry.url || entry.serverUrl || entry.httpUrl || entry.uri;
    if (typeof url === "string" && /^https?:\/\//i.test(url)) {
        const type = String(entry.type || entry.transport || "").toLowerCase();
        return urlResult(url.trim(), nameHint, entry.headers || entry.requestInit?.headers,
            { transport: type === "sse" ? "http" : "streamable_http" });
    }
    if (typeof entry.command === "string") {
        return fromCommand(entry.command, Array.isArray(entry.args) ? entry.args.map(String) : [], nameHint);
    }
    return null;
}

function fromJson(data) {
    const maps = [data?.mcpServers, data?.servers, data?.mcp?.servers, data?.context_servers];
    for (const map of maps) {
        if (map && typeof map === "object" && !Array.isArray(map)) {
            const names = Object.keys(map);
            const results = names.map((n) => fromServerEntry(map[n], n)).filter(Boolean);
            const firstUrl = results.find((r) => r.kind === "url");
            const pick = firstUrl || results[0];
            if (!pick) break;
            if (results.length > 1) {
                pick.note = [pick.note, `Found ${results.length} servers; connecting "${pick.name || names[0]}". Paste the others one at a time.`]
                    .filter(Boolean).join(" ");
            }
            return pick;
        }
    }
    const single = fromServerEntry(data, typeof data?.name === "string" ? data.name : undefined);
    return single || { kind: "invalid", message: "That JSON has no server URL or command in it." };
}

function fromCliAdd(tokens) {
    // claude mcp add [--transport http] [--header "K: V"] <name> <url>
    const url = tokens.map((t) => (t.match(URL_RE) || [])[0]).find(Boolean);
    if (!url) return null;
    const idx = tokens.findIndex((t) => t.includes(url));
    const valueFlags = new Set(["--transport", "-t", "--scope", "-s", "--header", "-H", "--env", "-e"]);
    const words = new Set(["add", "mcp", "claude", "codex", "gemini", "http", "sse", "stdio"]);
    let name;
    for (let i = idx - 1; i >= 0; i -= 1) {
        const t = tokens[i];
        if (t.startsWith("-") || words.has(t) || valueFlags.has(tokens[i - 1])) continue;
        name = t;
        break;
    }
    const transport = tokens.includes("sse") ? "http" : "streamable_http";
    return urlResult(stripTrailingPunct(url), name, headersFromArgs(tokens), { transport });
}

/** Parse whatever was pasted into the server URL box. */
export function parseMcpInput(text) {
    const raw = String(text ?? "").trim();
    if (!raw) return { kind: "empty" };

    if (raw.startsWith("{") || raw.startsWith("\"mcpServers\"")) {
        let data;
        const body = raw.startsWith("{") ? raw : `{${raw}}`;
        try { data = JSON.parse(body); } catch {
            return { kind: "invalid", message: "That looks like JSON but could not be read. Paste the whole block, including the outer braces." };
        }
        return fromJson(data);
    }

    if (/^https?:\/\//i.test(raw) && !/\s/.test(raw)) {
        // The whole input is the URL: keep it exactly (a trailing "." or ")"
        // can be part of a path or a signed query value).
        try { new URL(raw); } catch {
            return hasPlaceholder(raw.replace(/^https?:\/\//i, ""))
                ? urlResult(raw, undefined, {})  // explains the placeholder
                : { kind: "invalid", message: "That URL could not be read." };
        }
        return urlResult(raw, undefined, {});
    }

    const tokens = splitCommand(raw.replace(/\\\r?\n/g, " "));
    const first = (tokens[0] || "").toLowerCase();
    if (["claude", "codex", "gemini"].includes(first) && tokens.includes("mcp")) {
        const cli = fromCliAdd(tokens);
        if (cli) return cli;
        const dash = tokens.indexOf("--");
        if (dash >= 0) return fromCommand(tokens[dash + 1], tokens.slice(dash + 2));
        return { kind: "invalid", message: "That command does not include a server URL." };
    }
    if (LAUNCHERS.has(first.replace(/\.(exe|cmd)$/, ""))) {
        return fromCommand(tokens[0], tokens.slice(1));
    }

    if (looksLikeBareHost(raw)) {
        return urlResult(`https://${raw}`, undefined, {}, { note: "Added https:// in front." });
    }
    const embedded = raw.match(URL_RE);
    if (embedded) {
        // Pulled out of prose: trailing sentence punctuation is not part of it.
        return urlResult(stripTrailingPunct(embedded[0]), undefined, {}, { note: "Used the URL found in what you pasted." });
    }
    return { kind: "invalid", message: "Paste a server URL that starts with https://, or a config block from the server's docs." };
}
