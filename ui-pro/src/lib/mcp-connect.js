// State helpers for connecting MCP servers from Settings (OAuth popup flow,
// the Connect request, status labels). Plain JS so ui-pro/tests/*.test.mjs
// can import it; MCPServersPanel.tsx keeps only rendering and wiring.
//
// The OAuth callback page is served by the BACKEND (its origin, not the
// frontend's) and posts { type: "quasar-mcp-oauth", server, ok, error } to
// window.opener with the frontend origin as targetOrigin. Some providers set
// Cross-Origin-Opener-Policy, which cuts the popup off from its opener, so
// the message can be lost and `popup.closed` can turn true while the user is
// still signing in. The waiter therefore also polls the server list, and a
// closed popup only ends the wait after a grace period.

import { deriveServerName, parseMcpInput, splitCommand, uniqueServerName } from "./mcp-paste.js";

export const OAUTH_MESSAGE_TYPE = "quasar-mcp-oauth";
export const OAUTH_TIMEOUT_MS = 5 * 60 * 1000;
export const OAUTH_CLOSED_GRACE_MS = 45 * 1000;

export function originOf(url) {
    try { return new URL(url).origin; } catch { return ""; }
}

/** True for a well-formed result message from our own backend's callback page. */
export function isOAuthResultMessage(event, apiBase) {
    if (!event || event.origin !== originOf(apiBase)) return false;
    const d = event.data;
    return !!d && typeof d === "object" && d.type === OAUTH_MESSAGE_TYPE && typeof d.ok === "boolean"
        && (d.server === null || typeof d.server === "string")
        && (d.error === null || d.error === undefined || typeof d.error === "string");
}

/**
 * `?mcp_oauth=connected|signed_in|failed&mcp_server=X` left by the full-page
 * fallback. Only a hint that a sign-in just happened: the panel shows what
 * the server list actually says (see returnBanner).
 */
export function readOAuthReturn(search) {
    const params = new URLSearchParams(search || "");
    const status = params.get("mcp_oauth");
    if (status !== "connected" && status !== "signed_in" && status !== "failed") return null;
    return { status, server: params.get("mcp_server") || "" };
}

/** The same URL without the OAuth return flags (for history.replaceState). */
export function stripOAuthReturn(href) {
    try {
        const u = new URL(href);
        u.searchParams.delete("mcp_oauth");
        u.searchParams.delete("mcp_server");
        return u.pathname + (u.search ? u.search : "") + u.hash;
    } catch {
        return null;
    }
}

/**
 * The banner after a full-page sign-in, from the REAL server list (a query
 * flag alone proves nothing, anyone can open such a link).
 */
export function returnBanner(ret, servers) {
    if (!ret) return null;
    const srv = (servers || []).find((s) => s.name === ret.server);
    if (ret.status === "failed") {
        // A failure is reported as one even when an OLDER connection to the
        // same server is still up (a failed Reconnect, guard CX-09).
        const kept = srv?.status?.state === "connected" ? " The existing connection is unchanged." : "";
        return { type: "error", text: `The sign-in${srv ? ` to ${srv.name}` : ""} did not finish.${kept} Press Sign in to try again.` };
    }
    if (!srv) return null;
    const state = srv.status?.state;
    if (state === "connected") {
        const n = (srv.status.tools || []).length;
        return { type: "success", text: `Connected to ${srv.name}: ${n} tool${n === 1 ? "" : "s"} ready in your chats.` };
    }
    if (srv.oauth?.signed_in) {
        return { type: "warn", text: `Signed in to ${srv.name}, but it did not connect yet. Its error is shown below; press Test to try again.` };
    }
    return { type: "error", text: `The sign-in to ${srv.name} did not finish. Press Sign in to try again.` };
}

/**
 * What to do while waiting for a popup sign-in. `baselineSignedInAt` is the
 * server's oauth.signed_in_at when the wait began: a newer value means the
 * callback stored fresh tokens even if the MCP connection then failed (the
 * case where only `connected` would wait forever).
 * Returns { done: true, ok, error? } or { done: false, popupClosed }.
 */
// Longer than the backend's worst-case connection test after a sign-in
// (2 x 20 s connect timeout + 2 s, services/user_mcp.py), guard CX-27.
export const SIGNED_IN_SETTLE_MS = 45 * 1000;

export function oauthWaitDecision({ startedAt, now, popupClosedAt, serverStatus, oauth, baselineSignedInAt, attempt,
    signedInSeenAt = null, timeoutMs = OAUTH_TIMEOUT_MS, closedGraceMs = OAUTH_CLOSED_GRACE_MS }) {
    // Only THIS sign-in ends the wait: a "connected" status may be the old
    // connection a Reconnect is replacing, and a newer signed_in_at may come
    // from another tab's sign-in. With the attempt id from the connect
    // reply, only tokens from that attempt count (frontend guard CX-20).
    const signedIn = !!(oauth && oauth.signed_in);
    const mine = attempt
        ? signedIn && oauth.signed_in_attempt === attempt
        : signedIn && oauth.signed_in_at != null && oauth.signed_in_at !== baselineSignedInAt;
    if (mine) {
        // The callback stores the tokens and THEN tests the connection (a
        // second or two). Finish on that test's result, not on the tokens
        // alone, or a poll landing in between reports a false "did not
        // connect" (seen live). Cap the settle time in case no result comes.
        // Only a status the pool settled AFTER this sign-in counts: a
        // Reconnect's old "connected" (or a stale error) does not (CX-26).
        const fresh = serverStatus && typeof serverStatus.checked_at === "number"
            && typeof oauth.signed_in_at === "number" && serverStatus.checked_at >= oauth.signed_in_at;
        const state = fresh ? serverStatus.state : undefined;
        if (state === "connected") return { done: true, ok: true };
        if (state === "error" || state === "needs_api_key") {
            return { done: true, ok: true, error: serverStatus.error || "Signed in, but the server did not connect yet." };
        }
        if (signedInSeenAt != null && now - signedInSeenAt > SIGNED_IN_SETTLE_MS) {
            return { done: true, ok: true, error: (fresh && serverStatus.error) || "Signed in, but the server did not connect yet." };
        }
        return { done: false, popupClosed: popupClosedAt != null, signedIn: true };
    }
    if (popupClosedAt != null && now - popupClosedAt > closedGraceMs) {
        return { done: true, ok: false, error: "The sign-in window closed before the sign-in finished. Press Sign in to try again." };
    }
    if (now - startedAt > timeoutMs) {
        return { done: true, ok: false, error: "Sign-in timed out after 5 minutes. Press Sign in to try again." };
    }
    return { done: false, popupClosed: popupClosedAt != null };
}

/**
 * One-shot completion for a wait: the message listener and the poller both
 * call `finish`; only the first call runs `onFinish` (no double banners, no
 * second refresh). `cancel()` (Cancel button, effect cleanup) makes every
 * later `finish` a no-op, so a poll already in flight cannot complete a
 * wait the user abandoned (frontend guard CX-07).
 */
export function createOnce(onFinish) {
    let done = false;
    return {
        get done() { return done; },
        finish(...args) {
            if (done) return false;
            done = true;
            onFinish(...args);
            return true;
        },
        cancel() { done = true; },
    };
}

/** What the Connect response asks the form to do next. */
export function nextStep(body) {
    const status = body && body.status;
    if (status === "needs_auth" && typeof body.authorize_url === "string" && /^https?:\/\//i.test(body.authorize_url)) {
        return { step: "oauth", authorizeUrl: body.authorize_url, server: body.server?.name };
    }
    if (status === "needs_api_key") return { step: "api_key", message: body.message || "This server needs an API key." };
    if (status === "connected") {
        const tools = body.connection?.tools || [];
        return { step: "done", server: body.server?.name, tools };
    }
    if (status === "needs_auth") return { step: "error", message: "The server asked for sign-in but sent no sign-in address.", saved: false };
    return {
        step: "error",
        message: body?.connection?.error || body?.message || "The server did not connect.",
        // Only the backend knows whether it saved anything ("Nothing was
        // changed" replies carry saved: false).
        saved: body?.saved === true,
    };
}

/** Label + tone for a server's status pill. */
export function statusView(status, auth) {
    const state = status && status.state ? status.state
        : status && status.connected ? "connected" : status ? "error" : null;
    switch (state) {
        case "connected": {
            const n = (status.tools || []).length;
            return { tone: "ok", label: `Connected (tools listed) · ${n} tool${n === 1 ? "" : "s"}` };
        }
        case "needs_auth": return { tone: "warn", label: "Needs sign-in" };
        case "needs_api_key": return { tone: "warn", label: "Needs API key" };
        case "connecting": return { tone: "idle", label: "Connecting" };
        case "error": return { tone: "err", label: "Not connected" };
        default: return { tone: "idle", label: auth === "oauth" ? "Signed in, not checked yet" : "Not checked yet" };
    }
}

/** `Bearer <key>`; a key typed with its own scheme ("Basic ...") is sent as typed. */
export function bearerHeader(key) {
    const k = String(key || "").trim();
    if (!k) return null;
    return /^(bearer|basic|token)\s+/i.test(k) ? k : `Bearer ${k}`;
}

/** "KEY=value" (env) or "Name: value" (headers), one per line. */
export function parsePairs(text, sep) {
    const out = {};
    for (const line of String(text || "").split("\n")) {
        const i = line.indexOf(sep);
        if (i <= 0) continue;
        const k = line.slice(0, i).trim();
        const v = line.slice(i + 1).trim();
        if (k) out[k] = v;
    }
    return out;
}

function endpointOfInput(text) {
    const p = parseMcpInput(text);
    if (p.kind !== "url") return "";
    try {
        const u = new URL(p.url);
        return `${u.origin.toLowerCase()}${u.pathname.replace(/\/+$/, "")}${u.search}`;
    } catch { return ""; }
}

/**
 * Whether a typed API key / headers still belong to the pasted server after
 * the URL box changed. They survive only if the box still names exactly the
 * same endpoint (origin, path and query; only a trailing slash or #fragment
 * may differ). Another path or query on the same host can be another tenant
 * (/team-a/mcp vs /team-b/mcp, ?tenant=A vs ?tenant=B), so it drops them, as
 * does a different host or an empty box (frontend guard CX-01).
 */
export function credentialsStillApply(prevInput, nextInput) {
    const before = endpointOfInput(prevInput);
    const after = endpointOfInput(nextInput);
    return !!before && before === after;
}

/**
 * The POST /api/mcp-servers body for a Connect click, or { error }.
 *  - preset: { url, name, transport? } for a gallery / hosted-alternative
 *    click. It NEVER carries the form's headers, key or env: those were typed
 *    for whatever was pasted, not for this server.
 *  - parsed: parseMcpInput(form input).
 *  - form: { advancedOpen, transport, headerText, envText, cmd, cmdArgs,
 *    apiKey, keyHeader, name, nameTyped }
 *  - servers: the saved list, so a derived name never overwrites a different
 *    server (a typed name may: that is how a user edits one).
 */
export function buildConnectBody({ preset = null, parsed, form, servers = [] }) {
    const local = !preset && !!form.advancedOpen && form.transport === "stdio";
    if (local) {
        const cmd = String(form.cmd || "").trim();
        if (!cmd) return { error: "Enter the command to run." };
        return {
            body: {
                name: String(form.name || "").trim() || cmd, transport: "stdio", url: null, command: cmd,
                args: splitCommand(form.cmdArgs || ""), env: parsePairs(form.envText, "="), headers: {},
                replace: true,
            },
        };
    }
    const url = preset ? preset.url : (parsed && parsed.kind === "url" ? parsed.url : "");
    if (!/^https?:\/\//i.test(url)) {
        return { error: parsed && parsed.kind === "invalid" ? parsed.message : "Paste the server URL first. It starts with https://" };
    }
    let headers = {};
    if (!preset) {
        headers = { ...(parsed.headers || {}), ...parsePairs(form.headerText, ":") };
        const key = String(form.apiKey || "").trim();
        const keyHeader = form.keyHeader || "Authorization";
        if (key && !Object.keys(headers).some((k) => k.toLowerCase() === keyHeader.toLowerCase())) {
            headers[keyHeader] = keyHeader.toLowerCase() === "authorization" ? bearerHeader(key) : key;
        }
    }
    const transport = preset ? (preset.transport || "http")
        : (form.advancedOpen && form.transport !== "stdio" ? form.transport : (parsed.transport || "http"));
    const typed = !preset && form.nameTyped && String(form.name || "").trim();
    const base = typed || (preset ? preset.name : (parsed.name || deriveServerName(url))) || deriveServerName(url);
    const name = typed ? base : uniqueServerName(base, url, servers);
    // replace=false: the backend refuses (409) rather than overwrite a
    // different server that took this derived name in another tab since
    // our list was loaded; the panel then picks a fresh name (guard CX-03).
    return { body: { name, transport, url, command: null, args: [], env: {}, headers, replace: !!typed } };
}
