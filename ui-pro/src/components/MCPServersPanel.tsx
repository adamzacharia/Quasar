"use client";

// Settings > MCP servers. Paste a URL (or click a preset), press Connect, and
// the backend works out what the server needs: OAuth servers open the
// provider's sign-in in a popup, API-key servers ask for a key, open servers
// just connect. Transport, headers and local commands sit under "Advanced".
// Request building, the popup wait and status wording live in
// lib/mcp-connect.js (unit-tested); this file renders and wires them.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, CheckCircle, ChevronDown, ExternalLink, KeyRound, Loader2, Lock, LogIn, LogOut, Plus, RefreshCw, Search, Telescope, Trash2, Wrench } from "lucide-react";
import { useAuthStore, authBearerHeaders } from "../lib/auth-store";
import { parseMcpInput, deriveServerName } from "../lib/mcp-paste.js";
import type { McpTransport, ParsedMcpInput } from "../lib/mcp-paste.js";
import {
    ASTRONOMY_PRESETS, ASTRONOMY_PRESETS_VERIFIED_ON, MCP_PRESETS, PRESETS_VERIFIED_ON, apiKeyPageFor, gitmcpUrl, iconDomain,
} from "../lib/mcp-presets.js";
import type { McpPreset } from "../lib/mcp-presets.js";
import {
    REGISTRY_HOME, mergeResults, overlapNote, registryAction, registryBadges, registrySearchParams,
} from "../lib/mcp-registry.js";
import type { RegistryItem, RegistryMode, RegistrySearchResult } from "../lib/mcp-registry.js";
import {
    buildConnectBody, createOnce, credentialsStillApply, isOAuthResultMessage, nextStep, oauthWaitDecision,
    readOAuthReturn, returnBanner, statusView, stripOAuthReturn,
} from "../lib/mcp-connect.js";
import type { McpOAuthInfo, McpServerStatus } from "../lib/mcp-connect.js";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const POLL_MS = 2500;
const POPUP_NAME = "quasar-mcp-oauth";
const POPUP_FEATURES = "popup=yes,width=520,height=720";

interface MCPServer {
    name: string;
    transport?: McpTransport;
    command?: string;
    args?: string[];
    url?: string;
    auth?: "none" | "oauth";
    /** Masked by the API: names only, values are "********". */
    env?: Record<string, string>;
    headers?: Record<string, string>;
    status?: McpServerStatus;
    oauth?: McpOAuthInfo;
}

type FormMsg = { type: "success" | "error" | "warn"; text: string } | null;

interface OAuthWait {
    server: string;
    authorizeUrl: string;
    blocked: boolean;
    startedAt: number;
    closedAt: number | null;
    baselineSignedInAt: number | null;
    /** The sign-in attempt id from the connect reply (ends the wait on THIS sign-in only). */
    attempt: string | null;
}

const VERIFIED_LABEL: Record<McpPreset["verified"], string> = {
    "tool-call": "checked: a tool call returned a result",
    "tools-listed": "checked: connects and lists tools",
    "oauth-discovery": "checked: sign-in settings only, no account tested",
};

const INPUT = "w-full rounded-xl border border-[var(--q-border)] bg-[var(--q-card)] px-3 py-2 text-sm text-[var(--q-text)] placeholder:text-[var(--q-text-faint)] focus:border-[var(--q-border-strong)] focus:outline-none";
const LABEL = "mb-1.5 block text-xs font-medium text-[var(--q-text-muted)]";

const TRANSPORTS: { value: McpTransport; label: string; hint: string }[] = [
    { value: "http", label: "Automatic", hint: "Tries modern streamable HTTP first, then the older SSE transport." },
    { value: "streamable_http", label: "Streamable HTTP", hint: "Modern MCP servers, usually ending in /mcp." },
    { value: "stdio", label: "Local command", hint: "Runs a program on the Quasar server itself." },
];

function Favicon({ domain }: { domain: string }) {
    const [failed, setFailed] = useState(false);
    if (!domain || failed) return <Wrench className="h-4 w-4 text-[var(--q-text-faint)]" />;
    return (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={`https://www.google.com/s2/favicons?domain=${domain}&sz=32`} alt="" width={16} height={16}
            className="h-4 w-4 rounded-sm" onError={() => setFailed(true)} />
    );
}

function StatusPill({ status, auth }: { status?: McpServerStatus; auth?: string }) {
    const view = statusView(status, auth);
    const tone = view.tone === "ok" ? "bg-emerald-500/10 text-emerald-600"
        : view.tone === "warn" ? "bg-amber-500/10 q-warn"
            : view.tone === "err" ? "bg-red-500/10 q-err"
                : "bg-[var(--q-canvas)] text-[var(--q-text-muted)]";
    return <span className={`rounded-md px-2 py-0.5 text-[11px] font-medium ${tone}`}>{view.label}</span>;
}

function Banner({ msg }: { msg: NonNullable<FormMsg> }) {
    const cls = msg.type === "success" ? "bg-emerald-500/10 text-emerald-600" : msg.type === "warn" ? "bg-amber-500/10 q-warn" : "bg-red-500/10 q-err";
    return (
        <div role="status" className={`flex items-start gap-2.5 rounded-xl px-4 py-3 text-sm ${cls}`}>
            {msg.type === "success" ? <CheckCircle className="mt-0.5 h-4 w-4 shrink-0" /> : <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />}
            <span>{msg.text}</span>
        </div>
    );
}

function api(path: string, init: RequestInit = {}) {
    return fetch(`${API_BASE}${path}`, {
        credentials: "include",
        ...init,
        headers: authBearerHeaders(init.body ? { "Content-Type": "application/json" } : undefined),
    });
}

async function readJson(res: Response): Promise<Record<string, unknown>> {
    try { return await res.json(); } catch { return {}; }
}

/** Open the sign-in window NOW, inside the click, so a popup blocker allows
 *  it; the address is filled in once the backend has answered. */
function preOpenPopup(): Window | null {
    const w = window.open("", POPUP_NAME, POPUP_FEATURES);
    if (w) {
        try {
            w.document.title = "Connecting...";
            w.document.body.style.cssText = "font:15px system-ui;padding:40px;color:#555";
            w.document.body.textContent = "Connecting to the sign-in page...";
        } catch { /* not same-origin any more */ }
    }
    return w;
}

export function MCPServersPanel() {
    const { isAuthenticated } = useAuthStore();
    const [activeTab, setActiveTab] = useState<"installed" | "add">("installed");
    const [servers, setServers] = useState<MCPServer[]>([]);
    // Connect stays off until the saved list is known: a derived name is
    // only made unique against servers we have actually seen (guard CX-03).
    const [listLoaded, setListLoaded] = useState(false);
    const [listError, setListError] = useState(false);
    const [loading, setLoading] = useState(false);
    const [busy, setBusy] = useState<string | null>(null);
    const [listMsg, setListMsg] = useState<FormMsg>(null);

    // Connect form
    const [input, setInput] = useState("");
    const [name, setName] = useState("");
    const [nameTouched, setNameTouched] = useState(false);
    const [apiKey, setApiKey] = useState("");
    const [keyPrompt, setKeyPrompt] = useState<{ message: string; keyPage: string | null } | null>(null);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [transport, setTransport] = useState<McpTransport>("http");
    const [headerText, setHeaderText] = useState("");
    const [envText, setEnvText] = useState("");
    const [cmd, setCmd] = useState("npx");
    const [cmdArgs, setCmdArgs] = useState("");
    const [repoFor, setRepoFor] = useState<McpPreset | null>(null);
    const [repo, setRepo] = useState("");
    const [formMsg, setFormMsg] = useState<FormMsg>(null);
    const [connecting, setConnecting] = useState(false);
    const keyInputRef = useRef<HTMLInputElement>(null);
    const urlInputRef = useRef<HTMLTextAreaElement>(null);

    // MCP Registry search
    const [regQuery, setRegQuery] = useState("");
    const [regMode, setRegMode] = useState<RegistryMode | null>(null);
    const [regItems, setRegItems] = useState<RegistryItem[]>([]);
    const [regMeta, setRegMeta] = useState<Omit<RegistrySearchResult, "items"> | null>(null);
    const [regLoading, setRegLoading] = useState(false);
    const [regError, setRegError] = useState<string | null>(null);
    const [regShowAll, setRegShowAll] = useState(false);
    const [regPending, setRegPending] = useState<string | null>(null);
    const regSeq = useRef(0);
    // The spinner belongs to the card whose click started this connect only.
    useEffect(() => { if (!connecting && regPending) setRegPending(null); }, [connecting, regPending]);
    // The form message sits at the top of the tab; a click on a registry
    // result further down must still show it (unless a field takes focus).
    const formMsgRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (formMsg && formMsg.type !== "success") formMsgRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }, [formMsg]);

    // OAuth popup wait
    const [wait, setWait] = useState<OAuthWait | null>(null);
    const popupRef = useRef<Window | null>(null);

    const parsed: ParsedMcpInput = useMemo(() => parseMcpInput(input), [input]);
    const parsedName = parsed.kind === "url" ? (parsed.name || deriveServerName(parsed.url)) : "";
    const keyHeader = parsed.kind === "url" ? parsed.keyHeader : undefined;
    const keyPrefix = parsed.kind === "url" ? parsed.keyPrefix : undefined;
    const showKey = !!keyPrompt || !!keyHeader;
    const localSelected = showAdvanced && transport === "stdio";

    const fetchServers = useCallback(async (): Promise<MCPServer[] | null> => {
        if (!isAuthenticated) return null;
        try {
            const res = await api("/api/mcp-servers");
            if (res.ok) {
                const list = (await res.json()) as MCPServer[];
                setServers(list);
                setListLoaded(true);
                setListError(false);
                return list;
            }
        } catch { /* noop */ }
        setListError(true);
        return null;
    }, [isAuthenticated]);

    /** First load, and the Try again button. Back from a full-page sign-in
     *  (popup was blocked) the `?mcp_oauth=` flag only says a sign-in happened;
     *  the banner reflects the real server list, and the flag is removed only
     *  once that list loaded, so a failed fetch leaves it for the retry
     *  (guards CX-22, CX-24). */
    const loadList = useCallback(async () => {
        setLoading(true);
        const ret = readOAuthReturn(window.location.search);
        try {
            const list = await fetchServers();
            if (!ret) return;
            if (!list) {
                setListMsg({ type: "warn", text: "Could not load your servers to confirm the sign-in. Press Try again." });
                return;
            }
            const clean = stripOAuthReturn(window.location.href);
            if (clean) window.history.replaceState(null, "", clean);
            setListMsg(returnBanner(ret, list));
        } finally {
            setLoading(false);
        }
    }, [fetchServers]);

    useEffect(() => {
        if (isAuthenticated) loadList();
    }, [isAuthenticated, loadList]);

    const resetForm = () => {
        setInput(""); setName(""); setNameTouched(false); setApiKey(""); setKeyPrompt(null);
        setHeaderText(""); setEnvText(""); setCmdArgs(""); setRepoFor(null); setRepo("");
    };

    const finishOAuth = useCallback(async (server: string, ok: boolean, error?: string | null) => {
        setWait(null);
        try { popupRef.current?.close(); } catch { /* cross-origin after COOP */ }
        popupRef.current = null;
        const list = await fetchServers();
        const srv = (list || []).find((s) => s.name === server);
        if (ok) {
            const connected = srv?.status?.state === "connected";
            const tools = srv?.status?.tools || [];
            setListMsg(connected
                ? { type: "success", text: `Connected to ${server}: ${tools.length} tool${tools.length === 1 ? "" : "s"} ready in your chats.` }
                // Signed in, but the MCP server itself did not connect.
                : { type: "warn", text: error || `Signed in to ${server}, but it did not connect yet. Press Test to try again.` });
            setFormMsg(null);
            resetForm();
            setActiveTab("installed");
        } else {
            const text = error || "Sign-in did not finish.";
            setFormMsg({ type: "error", text });
            setListMsg({ type: "error", text: `${server}: ${text}` });
        }
    }, [fetchServers]);

    const openPopup = (server: string, authorizeUrl: string, pre: Window | null, list: MCPServer[] | null,
        attempt: unknown) => {
        let popup: Window | null = null;
        if (pre && !pre.closed) {
            try { pre.location.href = authorizeUrl; popup = pre; } catch { popup = null; }
        }
        if (!popup) popup = window.open(authorizeUrl, POPUP_NAME, POPUP_FEATURES);
        popupRef.current = popup;
        const baseline = (list || servers).find((s) => s.name === server)?.oauth?.signed_in_at ?? null;
        setWait({ server, authorizeUrl, blocked: !popup, startedAt: Date.now(), closedAt: null, baselineSignedInAt: baseline,
            attempt: typeof attempt === "string" ? attempt : null });
    };

    const retryPopup = (w: OAuthWait) => {
        const popup = window.open(w.authorizeUrl, POPUP_NAME, POPUP_FEATURES);
        popupRef.current = popup;
        setWait({ ...w, blocked: !popup, startedAt: Date.now(), closedAt: null });
    };

    const cancelWait = () => {
        setWait(null);
        try { popupRef.current?.close(); } catch { /* noop */ }
        popupRef.current = null;
    };

    // While a sign-in is open: listen for the callback page's message and
    // poll the server list (the message can be lost, see mcp-connect.js).
    // One shared once-guard: whichever finishes first wins, the other no-ops.
    useEffect(() => {
        if (!wait || wait.blocked) return;
        const { server, startedAt, baselineSignedInAt, attempt } = wait;
        const once = createOnce((ok: boolean, error?: string | null) => { finishOAuth(server, ok, error); });
        const onMessage = (event: MessageEvent) => {
            if (!isOAuthResultMessage(event, API_BASE)) return;
            const data = event.data as { server: string | null; ok: boolean; error?: string | null };
            if (data.server && data.server !== server) return;
            once.finish(data.ok, data.error ?? null);
        };
        window.addEventListener("message", onMessage);
        let closedAt: number | null = wait.closedAt;
        let signedInSeenAt: number | null = null;
        const timer = window.setInterval(async () => {
            if (once.done) return;
            if (closedAt == null && popupRef.current?.closed) {
                closedAt = Date.now();
                setWait((w) => (w && w.server === server ? { ...w, closedAt } : w));
            }
            const list = await fetchServers();
            if (once.done || !list) return;
            const srv = list.find((s) => s.name === server);
            const decision = oauthWaitDecision({
                startedAt, now: Date.now(), popupClosedAt: closedAt, serverStatus: srv?.status,
                oauth: srv?.oauth, baselineSignedInAt, attempt, signedInSeenAt,
            });
            if (decision.done) once.finish(decision.ok, decision.error ?? null);
            else if (decision.signedIn && signedInSeenAt == null) signedInSeenAt = Date.now();
        }, POLL_MS);
        return () => {
            // Cancel / unmount / a new wait: a poll still in flight must not
            // complete this one afterwards (guard CX-07).
            once.cancel();
            window.removeEventListener("message", onMessage);
            window.clearInterval(timer);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [wait?.server, wait?.blocked, wait?.startedAt, finishOAuth, fetchServers]);

    const onInputChange = (next: string) => {
        if (!credentialsStillApply(input, next)) {
            // A key typed for one server must never be sent to another.
            setApiKey("");
            setHeaderText("");
        }
        setInput(next);
        setKeyPrompt(null);
        setFormMsg(null);
    };

    const connect = async (preset: { url: string; name: string; transport?: McpTransport } | null, pre: Window | null = null) => {
        if (!isAuthenticated || connecting || !listLoaded) { try { pre?.close(); } catch { /* noop */ } return; }
        setFormMsg(null);
        const built = buildConnectBody({
            preset, parsed, servers,
            form: { advancedOpen: showAdvanced, transport, headerText, envText, cmd, cmdArgs, apiKey, keyHeader, keyPrefix,
                name, nameTyped: nameTouched },
        });
        if (built.error !== undefined) {
            try { pre?.close(); } catch { /* noop */ }
            setFormMsg({ type: "error", text: built.error });
            return;
        }
        let body = built.body;
        setConnecting(true);
        let keepPopup = false;
        try {
            let res = await api("/api/mcp-servers", { method: "POST", body: JSON.stringify(body) });
            let data = await readJson(res);
            if (res.status === 409 && !body.replace) {
                // Another tab saved a different server under the derived name
                // since our list loaded: reload, pick a free name, retry once.
                const fresh = await fetchServers();
                const rebuilt = fresh && buildConnectBody({
                    preset, parsed, servers: fresh,
                    form: { advancedOpen: showAdvanced, transport, headerText, envText, cmd, cmdArgs, apiKey, keyHeader, keyPrefix,
                        name, nameTyped: nameTouched },
                });
                if (rebuilt && rebuilt.body && rebuilt.body.name !== body.name) {
                    body = rebuilt.body;
                    res = await api("/api/mcp-servers", { method: "POST", body: JSON.stringify(body) });
                    data = await readJson(res);
                }
            }
            if (!res.ok) {
                setFormMsg({ type: "error", text: String(data.detail || "Could not connect that server.") });
                return;
            }
            const step = nextStep(data);
            const savedName = String((data.server as { name?: string } | undefined)?.name || body.name);
            if (step.step === "oauth") {
                setKeyPrompt(null);
                // Navigate the sign-in window FIRST, nothing else awaited
                // before it (guard CX-05). The baseline is the list from
                // before this request: the sign-in it starts is newer.
                keepPopup = true;
                openPopup(savedName, step.authorizeUrl, pre, servers, data.attempt);
                fetchServers();
            } else if (step.step === "api_key") {
                setKeyPrompt({ message: step.message, keyPage: body.url ? apiKeyPageFor(body.url) : null });
                setFormMsg({ type: "warn", text: `${step.message} Paste it below and press Connect again.` });
                setTimeout(() => keyInputRef.current?.focus(), 50);
            } else if (step.step === "done") {
                const n = step.tools.length;
                setListMsg({ type: "success", text: `Connected to ${savedName}: ${n} tool${n === 1 ? "" : "s"} ready in your chats.` });
                resetForm();
                await fetchServers();
                setActiveTab("installed");
            } else {
                setFormMsg({ type: step.saved ? "warn" : "error", text: step.saved ? `Saved ${savedName}, but it did not connect: ${step.message}` : step.message });
                if (step.saved) fetchServers();
            }
        } catch {
            setFormMsg({ type: "error", text: "Network error. Check your connection and try again." });
        } finally {
            if (!keepPopup) { try { pre?.close(); } catch { /* noop */ } }
            setConnecting(false);
        }
    };

    const pickPreset = (preset: McpPreset) => {
        setFormMsg(null);
        setKeyPrompt(null);
        if (preset.needsRepo) {
            setRepoFor(preset);
            return;
        }
        setRepoFor(null);
        // Known sign-in servers: open the window inside this click.
        const pre = preset.auth === "oauth" ? preOpenPopup() : null;
        connectPreset({ url: preset.url, name: preset.name, transport: "http" }, pre);
    };

    /** A gallery / hosted-alternative click. The form switches to that server
     *  (so a needs-API-key follow-up targets it, guard CX-19) and any key or
     *  headers typed for something else are dropped first. */
    const connectPreset = (preset: { url: string; name: string; transport?: McpTransport }, pre: Window | null = null) => {
        setApiKey("");
        setHeaderText("");
        setInput(preset.url);
        setName(preset.name);
        setNameTouched(false);
        connect(preset, pre);
    };

    /** Search the registry. A newer search (or Load more) wins over an older
     *  one still in flight. */
    const searchRegistry = async (mode: RegistryMode, more = false) => {
        if (mode.kind === "q" && !mode.q.trim()) return;
        const seq = ++regSeq.current;
        const cursor = more ? regMeta?.next_cursor : null;
        setRegLoading(true);
        setRegError(null);
        if (!more) { setRegMode(mode); setRegItems([]); setRegMeta(null); setRegShowAll(false); }
        try {
            const res = await api(`/api/mcp-registry/search?${registrySearchParams(mode, cursor)}`);
            const data = await readJson(res);
            if (seq !== regSeq.current) return;
            if (!res.ok) {
                setRegError(String(data.detail || "The search did not work. Try again."));
                return;
            }
            const result = data as unknown as RegistrySearchResult;
            const { items, ...meta } = result;
            setRegItems((cur) => (more ? mergeResults(cur, items || []) : (items || [])));
            setRegMeta(meta);
        } catch {
            if (seq === regSeq.current) setRegError("Network error. Check your connection and try again.");
        } finally {
            if (seq === regSeq.current) setRegLoading(false);
        }
    };

    const pickRegistryItem = (item: RegistryItem) => {
        setRepoFor(null);
        const act = registryAction(item);
        if (act.kind === "connect") {
            setRegPending(item.id);
            connectPreset({ url: act.url, name: act.name, transport: "http" });
            return;
        }
        // Needs the user first (a key, or part of the address): fill the box.
        // A remote URL never goes out as a Local command (guard CX-06).
        if (transport === "stdio") setTransport("http");
        setApiKey("");
        setHeaderText("");
        setKeyPrompt(null);
        setName("");
        setNameTouched(false);
        setInput(act.input);
        setFormMsg({ type: "warn", text: act.note });
        setTimeout(() => (act.focusKey ? keyInputRef.current : urlInputRef.current)?.focus(), 50);
    };

    const connectRepo = () => {
        const url = gitmcpUrl(repo);
        if (!url) {
            setFormMsg({ type: "error", text: "Enter the repository as owner/repo, for example astropy/astropy." });
            return;
        }
        connectPreset({ url, name: deriveServerName(url), transport: "http" });
    };

    const action = async (srv: MCPServer, kind: "test" | "reconnect" | "disconnect") => {
        // Sign in / Reconnect on a sign-in server always ends in the popup.
        const pre = kind === "reconnect" && srv.auth === "oauth" ? preOpenPopup() : null;
        setBusy(`${kind}:${srv.name}`);
        setListMsg(null);
        let keepPopup = false;
        try {
            const path = `/api/mcp-servers/${encodeURIComponent(srv.name)}/${kind}`;
            const res = await api(path, { method: "POST" });
            const data = await readJson(res);
            if (!res.ok) {
                setListMsg({ type: "error", text: String(data.detail || "That did not work.") });
            } else if (data.status === "needs_auth" && typeof data.authorize_url === "string") {
                keepPopup = true;
                openPopup(srv.name, data.authorize_url, pre, servers, data.attempt);
            } else if (kind === "disconnect") {
                setListMsg({ type: "success", text: `Signed out of ${srv.name}.${data.revoked ? " The server revoked the old sign-in." : ""}` });
            } else if (data.message) {
                setListMsg({ type: "error", text: String(data.message) });
            }
            await fetchServers();
        } catch {
            setListMsg({ type: "error", text: "Network error." });
        } finally {
            if (!keepPopup) { try { pre?.close(); } catch { /* noop */ } }
        }
        setBusy(null);
    };

    const remove = async (srv: MCPServer) => {
        if (!confirm(`Remove the MCP server "${srv.name}"?${srv.auth === "oauth" ? " This also signs you out of it." : ""}`)) return;
        setBusy(`delete:${srv.name}`);
        setListMsg(null);
        try {
            const res = await api(`/api/mcp-servers/${encodeURIComponent(srv.name)}`, { method: "DELETE" });
            if (res.ok) {
                await fetchServers();
            } else {
                const data = await readJson(res);
                setListMsg({ type: "error", text: `Could not remove ${srv.name}: ${String(data.detail || res.status)}` });
            }
        } catch {
            setListMsg({ type: "error", text: `Could not remove ${srv.name}: network error.` });
        }
        setBusy(null);
    };

    if (!isAuthenticated) {
        return (
            <div className="flex h-full flex-col items-center justify-center gap-4 px-8 py-16 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--q-canvas)]">
                    <Lock className="h-7 w-7 text-[var(--q-text-faint)]" />
                </div>
                <div>
                    <p className="text-sm font-medium text-[var(--q-text)]">Sign in to use MCP servers</p>
                    <p className="mt-1.5 max-w-xs text-xs text-[var(--q-text-muted)]">Connecting external tools through the Model Context Protocol requires an account.</p>
                </div>
            </div>
        );
    }

    const tabClass = (active: boolean) =>
        `px-4 py-2 text-sm font-medium border-b-2 transition-colors ${active ? "border-[var(--q-ink)] text-[var(--q-text)]" : "border-transparent text-[var(--q-text-muted)] hover:text-[var(--q-text)]"}`;
    const isBusy = (kind: string, n: string) => busy === `${kind}:${n}`;
    const keyHeaderName = keyHeader || "Authorization";

    const waitBox = wait && (
        <div className="rounded-xl border border-[var(--q-border)] bg-[var(--q-canvas)] px-4 py-3 text-sm text-[var(--q-text)]" role="status">
            {wait.blocked ? (
                <div className="space-y-2.5">
                    <p>Your browser blocked the sign-in window for <b>{wait.server}</b>.</p>
                    <div className="flex flex-wrap gap-2">
                        <button type="button" className="q-pill-ink h-8 px-3 text-xs" onClick={() => retryPopup(wait)}>
                            <LogIn className="h-3.5 w-3.5" /> Open sign-in window
                        </button>
                        <button type="button" className="q-pill h-8 px-3 text-xs" onClick={() => window.location.assign(wait.authorizeUrl)}>
                            Sign in in this tab instead
                        </button>
                        <button type="button" className="q-pill h-8 px-3 text-xs" onClick={cancelWait}>Cancel</button>
                    </div>
                </div>
            ) : (
                <div className="flex items-center justify-between gap-3">
                    <span className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        {wait.closedAt
                            ? <>Checking whether the sign-in to <b>{wait.server}</b> finished...</>
                            : <>Waiting for you to sign in to <b>{wait.server}</b> in the popup...</>}
                    </span>
                    <button type="button" className="q-pill h-7 px-2.5 text-xs" onClick={cancelWait}>Cancel</button>
                </div>
            )}
        </div>
    );

    return (
        <div className="flex h-full flex-col bg-[var(--q-bg)]">
            <div className="flex shrink-0 border-b border-[var(--q-border)] px-4 pt-2">
                <button onClick={() => setActiveTab("installed")} className={tabClass(activeTab === "installed")}>
                    Your servers ({servers.length})
                </button>
                <button onClick={() => setActiveTab("add")} className={`${tabClass(activeTab === "add")} flex items-center gap-1.5`}>
                    <Plus className="h-4 w-4" /> Add server
                </button>
            </div>

            <div className="flex-1 space-y-5 overflow-y-auto p-6">
                {activeTab === "installed" && (
                    <div className="space-y-3">
                        {listMsg && <Banner msg={listMsg} />}
                        {waitBox}
                        {loading && <div className="flex items-center gap-2 text-sm text-[var(--q-text-muted)]"><Loader2 className="h-4 w-4 animate-spin" /> Loading...</div>}
                        {/* Only before the first successful load: a failed
                            background poll later changes nothing here (CX-25). */}
                        {!loading && listError && !listLoaded && (
                            <div className="flex items-center justify-between gap-3 rounded-xl bg-red-500/10 px-4 py-3 text-sm q-err" role="status">
                                <span>Could not load your MCP servers. Connecting stays off until they load.</span>
                                <button type="button" className="q-pill h-8 shrink-0 px-3 text-xs" onClick={() => { loadList(); }}>
                                    <RefreshCw className="h-3.5 w-3.5" /> Try again
                                </button>
                            </div>
                        )}
                        {!loading && listLoaded && servers.length === 0 && (
                            <div className="py-12 text-center">
                                <Wrench className="mx-auto mb-3 h-10 w-10 text-[var(--q-text-faint)]" strokeWidth={1.5} />
                                <p className="mb-1.5 text-sm text-[var(--q-text)]">No MCP servers yet</p>
                                <p className="mx-auto max-w-xs text-xs text-[var(--q-text-muted)]">Connect a hosted MCP server and its tools become available to Quasar in your chats.</p>
                                <button onClick={() => setActiveTab("add")} className="q-pill mt-4 h-8 px-3.5 text-xs">
                                    <Plus className="h-3.5 w-3.5" /> Add a server
                                </button>
                            </div>
                        )}
                        {!loading && servers.map((srv) => {
                            const status = srv.status;
                            const state = status?.state ?? (status?.connected ? "connected" : status ? "error" : undefined);
                            const oauth = srv.auth === "oauth";
                            const secretNames = [...Object.keys(srv.headers || {}), ...Object.keys(srv.env || {})];
                            return (
                                <div key={srv.name} className="glass-card rounded-2xl p-4">
                                    <div className="flex flex-col gap-3">
                                        <div className="min-w-0">
                                            <div className="flex flex-wrap items-center gap-2">
                                                <Favicon domain={srv.url ? iconDomain(srv.url) : ""} />
                                                <h4 className="truncate text-sm font-medium text-[var(--q-text)]">{srv.name}</h4>
                                                {oauth && <span className="q-tag">Sign-in</span>}
                                                {srv.transport === "stdio" && <span className="q-tag">Local command</span>}
                                                <StatusPill status={status} auth={srv.auth} />
                                            </div>
                                            <p className="mt-2 truncate rounded-lg bg-[var(--q-canvas)] px-2 py-1.5 font-mono text-xs text-[var(--q-text-muted)]">
                                                {srv.transport === "stdio" ? `$ ${srv.command} ${(srv.args || []).join(" ")}` : srv.url}
                                            </p>
                                            {state === "connected" && (status?.tools.length ?? 0) > 0 && (
                                                <p className="mt-2 text-[11px] text-[var(--q-text-muted)]">Tools: {status?.tools.join(", ")}</p>
                                            )}
                                            {state && state !== "connected" && status?.error && (
                                                <p className={`mt-2 text-[11px] ${state === "error" ? "q-err" : "q-warn"}`}>{status.error}</p>
                                            )}
                                            {secretNames.length > 0 && (
                                                <p className="mt-2 text-[10px] text-[var(--q-text-faint)]">Secrets saved (encrypted): {secretNames.join(", ")}</p>
                                            )}
                                        </div>
                                        <div className="flex flex-wrap items-center justify-end gap-1.5">
                                            {oauth && (
                                                <button onClick={() => action(srv, "reconnect")} disabled={!!busy || !!wait} className={state === "needs_auth" ? "q-pill-ink h-8 px-3 text-xs" : "q-pill h-8 px-3 text-xs"}
                                                    title="Open the provider's sign-in again">
                                                    {isBusy("reconnect", srv.name) ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <LogIn className="h-3.5 w-3.5" />}
                                                    {state === "needs_auth" ? "Sign in" : "Reconnect"}
                                                </button>
                                            )}
                                            {oauth && state !== "needs_auth" && (
                                                <button onClick={() => action(srv, "disconnect")} disabled={!!busy} className="q-pill h-8 px-3 text-xs" title="Sign out and revoke the saved sign-in">
                                                    {isBusy("disconnect", srv.name) ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <LogOut className="h-3.5 w-3.5" />}
                                                    Disconnect
                                                </button>
                                            )}
                                            <button onClick={() => action(srv, "test")} disabled={!!busy} className="q-pill h-8 px-3 text-xs" title="Connect now and list its tools (no tool is called)">
                                                {isBusy("test", srv.name) ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                                                Test
                                            </button>
                                            <button onClick={() => remove(srv)} disabled={!!busy} className="rounded-full p-2 text-[var(--q-text-faint)] transition-colors hover:bg-red-500/10 hover:text-red-500" title="Remove server" aria-label={`Remove ${srv.name}`}>
                                                {isBusy("delete", srv.name) ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}

                {activeTab === "add" && (
                    <div className="max-w-2xl space-y-5">
                        {waitBox}
                        {formMsg && <div ref={formMsgRef}><Banner msg={formMsg} /></div>}
                        {listError && !listLoaded && (
                            <Banner msg={{ type: "error", text: "Could not load your MCP servers, so Connect is off. Open Your servers and press Try again." }} />
                        )}

                        <div>
                            <p className={LABEL}>Popular servers</p>
                            <div className="flex flex-wrap gap-1.5">
                                {MCP_PRESETS.map((p) => (
                                    <button key={p.id} type="button" onClick={() => pickPreset(p)} disabled={connecting || !!wait || !listLoaded}
                                        className="q-pill h-8 gap-1.5 px-3 text-xs"
                                        title={`${p.blurb}. ${p.auth === "oauth" ? "Opens a sign-in. " : ""}${PRESETS_VERIFIED_ON}, ${VERIFIED_LABEL[p.verified]}.`}>
                                        <Favicon domain={p.domain} />
                                        {p.name}
                                        {p.auth === "oauth" && <LogIn className="h-3 w-3 text-[var(--q-text-faint)]" />}
                                    </button>
                                ))}
                            </div>
                            {repoFor && (
                                <div className="mt-2.5 flex gap-2">
                                    <input type="text" value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="owner/repo, e.g. astropy/astropy"
                                        onKeyDown={(e) => { if (e.key === "Enter") connectRepo(); }} className={`${INPUT} font-mono`} aria-label="GitHub repository" autoFocus />
                                    <button type="button" onClick={connectRepo} disabled={connecting || !listLoaded} className="q-pill-ink h-9 shrink-0 px-4 text-sm">Connect</button>
                                </div>
                            )}
                            <p className="mt-1.5 text-[11px] text-[var(--q-text-faint)]">
                                Checked on {PRESETS_VERIFIED_ON}. Servers with the sign-in arrow open the provider&apos;s sign-in page; for those only the sign-in settings were checked, not an account. Hover a server for details.
                            </p>
                        </div>

                        <div>
                            <p className={LABEL}>Astronomy and space</p>
                            <div className="flex flex-wrap gap-1.5">
                                {ASTRONOMY_PRESETS.map((p) => (
                                    <button key={p.id} type="button" onClick={() => pickPreset(p)} disabled={connecting || !!wait || !listLoaded}
                                        className="q-pill h-8 gap-1.5 px-3 text-xs"
                                        title={`${p.blurb}. ${p.official ? "Run by NASA itself. " : "Third-party server. "}${ASTRONOMY_PRESETS_VERIFIED_ON}, ${VERIFIED_LABEL[p.verified]}.`}>
                                        <Favicon domain={p.domain} />
                                        {p.name}
                                    </button>
                                ))}
                            </div>
                            <p className="mt-1.5 text-[11px] text-[var(--q-text-faint)]">
                                From the MCP Registry, checked on {ASTRONOMY_PRESETS_VERIFIED_ON}: each one connected and answered a real tool call, no key needed. All are third-party except NASA Earthdata. Wrappers of archives Quasar already has built in (SIMBAD, ADS, MAST, Gaia and others) are left out.
                            </p>
                        </div>

                        <div>
                            <label className={LABEL} htmlFor="mcp-registry-q">Search the MCP Registry</label>
                            <form className="flex gap-2" role="search" onSubmit={(e) => { e.preventDefault(); searchRegistry({ kind: "q", q: regQuery }); }}>
                                <div className="relative flex-1">
                                    <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--q-text-faint)]" />
                                    <input id="mcp-registry-q" type="search" value={regQuery} onChange={(e) => setRegQuery(e.target.value)} maxLength={80}
                                        placeholder="Server name, e.g. arxiv, nasa, github" className={`${INPUT} pl-9`} />
                                </div>
                                <button type="submit" disabled={regLoading || !regQuery.trim()} className="q-pill-ink h-9 shrink-0 px-4 text-sm">Search</button>
                                <button type="button" disabled={regLoading} onClick={() => searchRegistry({ kind: "topic", topic: "astronomy" })}
                                    className={`${regMode?.kind === "topic" ? "q-pill-ink" : "q-pill"} h-9 shrink-0 gap-1.5 px-3 text-sm`}
                                    title="Several astronomy searches merged, with astrology and unrelated matches removed">
                                    <Telescope className="h-4 w-4" /> Astronomy
                                </button>
                            </form>
                            <p className="mt-1.5 text-[11px] text-[var(--q-text-faint)]">
                                The registry matches server names only, and a plain &quot;astro&quot; search is mostly astrology, so use Astronomy for astronomy servers.
                                Listings are published by their authors and not checked by Quasar. Connect only servers you trust: their tools see what you ask in chat.
                                Only hosted servers (ones with a URL) are shown.
                            </p>

                            {regLoading && (
                                <div className="mt-3 flex items-center gap-2 text-sm text-[var(--q-text-muted)]" role="status">
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    {regMode?.kind === "topic" && !regItems.length ? "Searching the registry for astronomy servers (the first search can take up to 30 seconds)..." : "Searching the registry..."}
                                </div>
                            )}
                            {regError && <div className="mt-3"><Banner msg={{ type: "error", text: regError }} /></div>}
                            {!regLoading && !regError && regMeta && regItems.length === 0 && (
                                <p className="mt-3 text-sm text-[var(--q-text-muted)]">
                                    No hosted servers matched{regMeta.next_cursor ? " on this page" : ""}. Try a shorter name{regMode?.kind === "q" ? ", or Astronomy" : ""}.
                                </p>
                            )}
                            {regItems.length > 0 && (
                                <div className="mt-3 space-y-2">
                                    <p className="text-[11px] text-[var(--q-text-muted)]">
                                        {regItems.length} hosted server{regItems.length === 1 ? "" : "s"}
                                        {regMode?.kind === "topic" ? " about astronomy and space; ones that add something Quasar lacks come first" : ""}.
                                        {regMeta?.stale ? " The registry did not answer, so these are from an earlier search." : ""}
                                        {regMeta?.partial ? " Some registry searches failed; the list may be incomplete." : ""}
                                    </p>
                                    {(regShowAll ? regItems : regItems.slice(0, 12)).map((item) => {
                                        const badges = registryBadges(item, servers);
                                        const act = registryAction(item);
                                        const overlap = overlapNote(item);
                                        return (
                                            <div key={item.id} className="rounded-xl border border-[var(--q-border)] bg-[var(--q-card)] p-3">
                                                <div className="flex items-start justify-between gap-3">
                                                    <div className="min-w-0 flex-1">
                                                        <div className="flex flex-wrap items-center gap-1.5">
                                                            <Favicon domain={iconDomain(item.url)} />
                                                            <span className="text-sm font-medium text-[var(--q-text)]">{item.title}</span>
                                                            {badges.map((b) => (
                                                                <span key={b.text} className={`rounded-md px-1.5 py-0.5 text-[10px] font-medium ${
                                                                    b.tone === "ok" ? "bg-emerald-500/10 text-emerald-600"
                                                                        : b.tone === "warn" ? "bg-amber-500/10 q-warn"
                                                                            : "bg-[var(--q-canvas)] text-[var(--q-text-muted)]"}`}>{b.text}</span>
                                                            ))}
                                                        </div>
                                                        {item.description && <p className="mt-1 line-clamp-2 text-xs text-[var(--q-text-muted)]">{item.description}</p>}
                                                        {overlap && <p className="mt-1 text-[11px] q-warn">{overlap}</p>}
                                                        <p className="mt-1 truncate font-mono text-[10px] text-[var(--q-text-faint)]" title={`${item.id}${item.version ? ` v${item.version}` : ""}`}>
                                                            {item.url}
                                                        </p>
                                                        {(item.website || item.repository) && (
                                                            <p className="mt-1 flex gap-3 text-[11px]">
                                                                {item.repository && <a href={item.repository} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 text-[var(--q-text-muted)] underline">Source <ExternalLink className="h-3 w-3" /></a>}
                                                                {item.website && <a href={item.website} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 text-[var(--q-text-muted)] underline">Website <ExternalLink className="h-3 w-3" /></a>}
                                                            </p>
                                                        )}
                                                    </div>
                                                    <button type="button" onClick={() => pickRegistryItem(item)} disabled={connecting || !!wait || !listLoaded}
                                                        className="q-pill h-8 shrink-0 px-3 text-xs" aria-label={`${act.kind === "connect" ? "Connect" : "Set up"} ${item.title}`}>
                                                        {connecting && regPending === item.id ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Connecting...</>
                                                            : act.kind === "connect" ? <><Plus className="h-3.5 w-3.5" /> Connect</> : <><KeyRound className="h-3.5 w-3.5" /> Set up</>}
                                                    </button>
                                                </div>
                                            </div>
                                        );
                                    })}
                                    <div className="flex flex-wrap items-center gap-2">
                                        {!regShowAll && regItems.length > 12 && (
                                            <button type="button" className="q-pill h-8 px-3 text-xs" onClick={() => setRegShowAll(true)}>Show all {regItems.length}</button>
                                        )}
                                        {regMeta?.next_cursor && regMode?.kind === "q" && (
                                            <button type="button" className="q-pill h-8 px-3 text-xs" disabled={regLoading}
                                                onClick={() => { setRegShowAll(true); searchRegistry(regMode, true); }}>Load more from the registry</button>
                                        )}
                                        <a href={REGISTRY_HOME} target="_blank" rel="noopener noreferrer" className="ml-auto inline-flex items-center gap-0.5 text-[11px] text-[var(--q-text-faint)] underline">
                                            {regMeta?.source || "registry.modelcontextprotocol.io"} <ExternalLink className="h-3 w-3" />
                                        </a>
                                    </div>
                                </div>
                            )}
                        </div>

                        {!localSelected && (
                            <div>
                                <label className={LABEL} htmlFor="mcp-url">Paste a server URL</label>
                                <textarea id="mcp-url" ref={urlInputRef} value={input} rows={input.includes("\n") ? 5 : input.length > 90 ? 4 : 1}
                                    onChange={(e) => onInputChange(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !input.includes("\n")) { e.preventDefault(); connect(null); } }}
                                    placeholder="https://mcp.example.com/mcp" className={`${INPUT} resize-none font-mono`} spellCheck={false} />
                                <p className="mt-1.5 text-[11px] text-[var(--q-text-faint)]">A URL, or a config block or command copied from the server&apos;s docs.</p>
                                {parsed.kind === "url" && parsed.note && <p className="mt-1.5 text-[11px] q-warn">{parsed.note}</p>}
                                {parsed.kind === "invalid" && input.trim().length > 3 && <p className="mt-1.5 text-[11px] q-err">{parsed.message}</p>}
                                {parsed.kind === "local" && (
                                    <div className="mt-2 rounded-xl bg-amber-500/10 px-3 py-2.5 text-xs q-warn">
                                        <p>{parsed.message}</p>
                                        {parsed.alternative ? (
                                            <button type="button" className="q-pill-ink mt-2 h-8 px-3 text-xs" disabled={connecting}
                                                onClick={() => { const alt = parsed.alternative!; connectPreset({ url: alt.url, name: alt.name, transport: "http" }); }}>
                                                Connect the hosted {parsed.alternative.name} server instead
                                            </button>
                                        ) : (
                                            <p className="mt-1">Look in the server&apos;s docs for a hosted URL (often ending in /mcp), or run Quasar locally.</p>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}

                        <div>
                            <label className={LABEL} htmlFor="mcp-name">Name (optional)</label>
                            <input id="mcp-name" type="text" value={nameTouched ? name : ""} onChange={(e) => { setName(e.target.value); setNameTouched(true); }}
                                placeholder={parsedName || "Filled in from the address"} className={INPUT} />
                        </div>

                        {showKey && !localSelected && (
                            <div>
                                <label className={LABEL} htmlFor="mcp-key"><KeyRound className="mr-1 inline h-3.5 w-3.5" />API key</label>
                                <input id="mcp-key" ref={keyInputRef} type="password" autoComplete="off" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === "Enter") connect(null); }} placeholder="Paste your key" className={`${INPUT} font-mono`} />
                                <p className="mt-1.5 text-[11px] text-[var(--q-text-faint)]">
                                    {keyPrefix
                                        ? <>Sent as <code>{keyHeaderName}: {keyPrefix.trim()} &lt;key&gt;</code>, as the server&apos;s config asks.</>
                                        : keyHeaderName.toLowerCase() === "authorization"
                                        ? <>Sent as <code>Authorization: Bearer &lt;key&gt;</code> (a key that already starts with a scheme such as <code>Basic</code> is sent as typed).</>
                                        : <>Sent in the <code>{keyHeaderName}</code> header, as the server&apos;s config asks.</>}
                                    {" "}Stored encrypted, never shown again. For a different header, use Advanced.
                                    {keyPrompt?.keyPage && (
                                        <> {" "}<a href={keyPrompt.keyPage} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 underline">Get a key <ExternalLink className="h-3 w-3" /></a></>
                                    )}
                                </p>
                            </div>
                        )}

                        <div className="rounded-xl border border-[var(--q-border)]">
                            <button type="button" onClick={() => setShowAdvanced((v) => !v)} aria-expanded={showAdvanced}
                                className="flex w-full items-center justify-between px-4 py-2.5 text-xs font-medium text-[var(--q-text-muted)] hover:text-[var(--q-text)]">
                                Advanced
                                <ChevronDown className={`h-4 w-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                            </button>
                            {showAdvanced && (
                                <div className="space-y-4 border-t border-[var(--q-border)] px-4 py-4">
                                    <div>
                                        <p className={LABEL}>Transport</p>
                                        <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Transport">
                                            {TRANSPORTS.map((t) => (
                                                <button key={t.value} type="button" role="radio" aria-checked={transport === t.value}
                                                    onClick={() => setTransport(t.value)}
                                                    className={transport === t.value ? "q-pill-ink h-8 px-3 text-xs" : "q-pill h-8 px-3 text-xs"}>
                                                    {t.label}
                                                </button>
                                            ))}
                                        </div>
                                        <p className="mt-1.5 text-[11px] text-[var(--q-text-faint)]">{TRANSPORTS.find((t) => t.value === transport)?.hint}</p>
                                    </div>
                                    {transport === "stdio" ? (
                                        <>
                                            <p className="rounded-lg bg-amber-500/10 px-3 py-2 text-[11px] q-warn">
                                                The hosted Quasar site does not run local commands, for safety. This only works on your own Quasar install with QUASAR_ENABLE_MCP_STDIO turned on.
                                            </p>
                                            <div className="flex gap-3">
                                                <div className="w-1/3">
                                                    <label className={LABEL}>Command</label>
                                                    <input type="text" value={cmd} onChange={(e) => setCmd(e.target.value)} placeholder="npx, uvx..." className={`${INPUT} font-mono`} />
                                                </div>
                                                <div className="flex-1">
                                                    <label className={LABEL}>Arguments (quote values with spaces)</label>
                                                    <input type="text" value={cmdArgs} onChange={(e) => setCmdArgs(e.target.value)} placeholder="-y @modelcontextprotocol/server-everything" className={`${INPUT} font-mono`} />
                                                </div>
                                            </div>
                                            <div>
                                                <label className={LABEL}>Environment variables</label>
                                                <textarea value={envText} onChange={(e) => setEnvText(e.target.value)} placeholder="KEY=value, one per line" rows={3} className={`${INPUT} font-mono leading-relaxed`} />
                                            </div>
                                        </>
                                    ) : (
                                        <div>
                                            <label className={LABEL}>Request headers</label>
                                            <p className="mb-2 text-[11px] text-[var(--q-text-faint)]">One per line, <code>Name: value</code>. Stored encrypted; values are never shown again. Cleared when you paste a different server.</p>
                                            <textarea value={headerText} onChange={(e) => setHeaderText(e.target.value)} placeholder="Authorization: Bearer <token>" rows={2} className={`${INPUT} font-mono leading-relaxed`} />
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        <div className="flex justify-end">
                            <button onClick={() => connect(null)} disabled={connecting || !!wait || !listLoaded || (!localSelected && parsed.kind !== "url")} className="q-pill-ink h-10 px-5 text-sm">
                                {connecting ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle className="h-4 w-4" />}
                                {connecting ? "Connecting..." : "Connect"}
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
