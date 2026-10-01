"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useChatStore } from "../lib/store";
import { useAuthStore } from "../lib/auth-store";
import { useThemeStore } from "../lib/theme-store";
import {
    Plus, MessageSquare, Bookmark, Settings, HelpCircle,
    X, ExternalLink, Github, BookOpen, Search,
    Check, LogOut, User as UserIcon, Trash2,
    Database, RefreshCw, Sparkles, Star, ChevronDown, MoreHorizontal, Sun, Moon
} from "lucide-react";
import { SettingsModal } from "./SettingsModal";
import { ModelDropdown } from "./ModelDropdown";
import { ModelIcon } from "./ModelIcon";
import {
    listDatalabJobs, cancelDatalabJob, listMyTables, deleteMyTable,
    type DatalabJobRecord, type MyTableEntry,
} from "../lib/api";

function timeAgo(date: Date): string {
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return "Just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}

/* ────────────────────────────────────────────
   OVERLAY PANEL — reusable slide-over
   ──────────────────────────────────────────── */
function OverlayPanel({ open, onClose, title, icon: Icon, children }: {
    open: boolean;
    onClose: () => void;
    title: string;
    icon: React.ComponentType<{ className?: string }>;
    children: React.ReactNode;
}) {
    if (!open) return null;
    return (
        <div className="absolute inset-0 z-50 flex flex-col glass-sidebar animate-in fade-in slide-in-from-left-2 duration-200">
            {/* Header */}
            <div className="flex h-[66px] shrink-0 items-center justify-between px-4 border-b border-[var(--q-border)]">
                <div className="flex items-center gap-2">
                    <Icon className="size-4 text-[var(--q-text-muted)]" />
                    <h2 className="text-[15px] font-medium text-[var(--q-text)]">{title}</h2>
                </div>
                <button onClick={onClose} aria-label="Close panel" className="flex size-8 items-center justify-center rounded-full text-[var(--q-text-muted)] hover:bg-[var(--q-glass-control-hover)] hover:text-[var(--q-text)] transition-colors">
                    <X className="size-4" strokeWidth={1.75} />
                </button>
            </div>
            {/* Body */}
            <div className="flex-1 overflow-y-auto">
                {children}
            </div>
        </div>
    );
}

/* ────────────────────────────────────────────
   SETTINGS PANEL CONTENT
   ──────────────────────────────────────────── */
function SettingsContent({ selectedModel, availableModels, onSelectModel }: {
    selectedModel: string;
    availableModels: string[];
    onSelectModel: (m: string) => void;
}) {
    return (
        <div className="p-5 space-y-6">
            {/* Model */}
            <div>
                <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 block">AI Model</label>
                <div className="space-y-1">
                    {availableModels.map((m) => (
                        <button key={m} onClick={() => onSelectModel(m)}
                            className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm transition-colors ${m === selectedModel
                                ? "bg-primary/15 text-primary border border-primary/30"
                                : "text-slate-300 glass-control"
                                }`}>
                            <div className="flex items-center gap-2">
                                <ModelIcon model={m} className="w-4 h-4 shrink-0" />
                                <span className="font-medium">{m}</span>
                            </div>
                            {m === selectedModel && <Check className="w-4 h-4" />}
                        </button>
                    ))}
                </div>
            </div>

            {/* API Status */}
            <div>
                <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 block">Connection</label>
                <div className="glass-control rounded-lg p-3 flex items-center gap-3">
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-accent animate-pulse" />
                    <div>
                        <p className="text-sm text-white font-medium">Backend Connected</p>
                        <p className="text-[10px] text-slate-500">
                            {process.env.NEXT_PUBLIC_API_URL ? new URL(process.env.NEXT_PUBLIC_API_URL).host : "localhost:8000"}
                        </p>
                    </div>
                </div>
            </div>

            {/* Version */}
            <div>
                <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 block">About</label>
                <div className="glass-control rounded-lg p-3 space-y-1.5">
                    <div className="flex justify-between text-xs"><span className="text-slate-500">Version</span><span className="text-slate-300 font-mono">2.0.0</span></div>
                    <div className="flex justify-between text-xs"><span className="text-slate-500">Engine</span><span className="text-slate-300 font-mono">Responses API</span></div>
                    <a href="https://github.com/adamzacharia/Quasar" target="_blank" rel="noopener noreferrer"
                        className="flex items-center justify-between gap-3 text-xs text-slate-300 hover:text-primary transition-colors">
                        <span className="text-slate-500">Repository</span>
                        <span className="inline-flex items-center gap-1.5 font-mono truncate">
                            <Github className="w-3.5 h-3.5 shrink-0" />
                            adamzacharia/Quasar
                        </span>
                    </a>
                    <div className="flex justify-between text-xs"><span className="text-slate-500">Env</span><span className="text-slate-300 font-mono">Development</span></div>
                </div>
            </div>
        </div>
    );
}

/* ────────────────────────────────────────────
   SAVED PAPERS PANEL CONTENT
   ──────────────────────────────────────────── */
function SavedPapersContent() {
    const { savedPapers, removePaper } = useChatStore();

    if (savedPapers.length === 0) {
        return (
            <div className="p-5 space-y-4">
                <p className="text-xs text-slate-500">Papers you bookmark during research sessions will appear here.</p>

                <div className="flex flex-col items-center justify-center py-12 text-center">
                    <div className="w-14 h-14 rounded-2xl glass-control flex items-center justify-center mb-4">
                        <Bookmark className="w-7 h-7 text-slate-600" />
                    </div>
                    <p className="text-sm text-slate-400 font-medium">No saved papers yet</p>
                    <p className="text-xs text-slate-600 mt-1.5 max-w-[200px]">
                        Ask the assistant to find papers for you in the chat, then bookmark the ones you want to save.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="p-4 space-y-2">
            <p className="text-xs text-slate-500 px-1 mb-3">{savedPapers.length} saved paper{savedPapers.length !== 1 ? "s" : ""}</p>
            {savedPapers.map((paper) => {
                const adsUrl = paper.bibcode
                    ? `https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(paper.bibcode)}`
                    : paper.doi ? `https://doi.org/${paper.doi}`
                    : paper.arxivId ? `https://arxiv.org/abs/${paper.arxivId}`
                    : null;
                return (
                    <div key={paper.id} className="glass-control rounded-xl p-3 space-y-1.5 group">
                        <div className="flex items-start justify-between gap-2">
                            {adsUrl ? (
                                <a href={adsUrl} target="_blank" rel="noopener noreferrer"
                                   className="text-xs font-semibold text-white hover:text-primary transition-colors leading-snug flex-1">
                                    {paper.title}
                                </a>
                            ) : (
                                <span className="text-xs font-semibold text-white leading-snug flex-1">{paper.title}</span>
                            )}
                            <button onClick={() => removePaper(paper.id)}
                                    title="Remove from saved"
                                    className="p-1 text-yellow-500 hover:text-red-400 transition-colors shrink-0 opacity-0 group-hover:opacity-100">
                                <X className="w-3.5 h-3.5" />
                            </button>
                        </div>
                        <p className="text-[10px] text-slate-400 line-clamp-1">{paper.authors} · {paper.year}</p>
                        {adsUrl && (
                            <a href={adsUrl} target="_blank" rel="noopener noreferrer"
                               className="inline-flex items-center gap-1 text-[10px] text-slate-500 hover:text-primary transition-colors">
                                <ExternalLink className="w-3 h-3" />Open in ADS
                            </a>
                        )}
                    </div>
                );
            })}
        </div>
    );
}

/* ────────────────────────────────────────────
   DATA LAB PANEL — background jobs + My tables
   ──────────────────────────────────────────── */
function jobStatusChip(status: string): string {
    switch (status) {
        case "running": return "bg-amber-500/15 text-amber-400";
        case "queued":
        case "submitted": return "bg-sky-500/15 text-sky-400";
        case "succeeded": return "bg-emerald-500/15 text-emerald-400";
        case "failed": return "bg-red-500/15 text-red-400";
        case "canceled": return "bg-slate-500/15 text-slate-400";
        default: return "bg-slate-500/15 text-slate-400";
    }
}

/** Job states that still hold server resources (can be cancelled). */
const ACTIVE_JOB_STATUSES = ["queued", "running", "submitted"];

function DatalabJobsContent() {
    const [jobs, setJobs] = useState<DatalabJobRecord[]>([]);
    const [loading, setLoading] = useState(true);

    const refresh = async () => {
        try { setJobs(await listDatalabJobs()); }
        catch { /* backend offline / signed out — keep the last list */ }
        finally { setLoading(false); }
    };

    useEffect(() => {
        refresh();
        const timer = setInterval(refresh, 5000);
        return () => clearInterval(timer);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const cancel = async (jobId: string) => {
        try { await cancelDatalabJob(jobId); } catch { /* record may already be terminal */ }
        refresh();
    };

    if (!loading && jobs.length === 0) {
        return (
            <div className="p-5 space-y-4">
                <p className="text-xs text-slate-500">Background Data Lab queries (async submits, tiled scans) appear here.</p>
                <div className="flex flex-col items-center justify-center py-10 text-center">
                    <div className="w-14 h-14 rounded-2xl glass-control flex items-center justify-center mb-4">
                        <Database className="w-7 h-7 text-slate-600" />
                    </div>
                    <p className="text-sm text-slate-400 font-medium">No jobs yet</p>
                    <p className="text-xs text-slate-600 mt-1.5 max-w-[210px]">
                        Ask for a wide catalog query &ldquo;as a background job&rdquo; and track it here.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="p-4 space-y-2">
            <div className="flex items-center justify-between px-1 mb-2">
                <p className="text-xs text-slate-500">{jobs.length} job{jobs.length !== 1 ? "s" : ""} · refreshes every 5s</p>
                <button onClick={refresh} title="Refresh now"
                    className="p-1 rounded text-slate-500 hover:text-white transition-colors">
                    <RefreshCw className="w-3.5 h-3.5" />
                </button>
            </div>
            {jobs.map((job) => {
                const active = ACTIVE_JOB_STATUSES.includes(job.status);
                return (
                    <div key={job.job_id} className="glass-control rounded-xl p-3 space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-semibold text-white truncate" title={job.job_id}>
                                {job.kind}{job.external ? " · server" : ""}
                            </span>
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 ${jobStatusChip(job.status)}`}>
                                {job.status}
                            </span>
                        </div>
                        <p className="text-[10px] text-slate-500 font-mono truncate" title={job.job_id}>{job.job_id}</p>
                        <div className="flex items-center justify-between gap-2">
                            <p className="text-[10px] text-slate-400">
                                {timeAgo(new Date(job.updated_at * 1000))}
                                {job.result?.rowcount !== undefined ? ` · ${job.result.rowcount} rows` : ""}
                                {job.result?.candidates_found !== undefined ? ` · ${job.result.candidates_found} candidates` : ""}
                            </p>
                            {active && (
                                <button onClick={() => cancel(job.job_id)}
                                    className="text-[10px] text-red-500/80 hover:text-red-400 transition-colors shrink-0">
                                    Cancel
                                </button>
                            )}
                        </div>
                        {job.error && <p className="text-[10px] text-red-400/80 line-clamp-2">{job.error}</p>}
                    </div>
                );
            })}
        </div>
    );
}

function MyTablesContent() {
    const [tables, setTables] = useState<MyTableEntry[]>([]);
    const [loading, setLoading] = useState(true);

    const refresh = async () => {
        try { setTables(await listMyTables()); }
        catch { /* backend offline / signed out */ }
        finally { setLoading(false); }
    };

    useEffect(() => { refresh(); }, []);

    const remove = async (name: string) => {
        await deleteMyTable(name);
        refresh();
    };

    if (!loading && tables.length === 0) {
        return (
            <div className="p-5 space-y-4">
                <p className="text-xs text-slate-500">Durable tables saved from Data Lab results (they outlive the 1-hour result cache).</p>
                <div className="flex flex-col items-center justify-center py-10 text-center">
                    <div className="w-14 h-14 rounded-2xl glass-control flex items-center justify-center mb-4">
                        <Database className="w-7 h-7 text-slate-600" />
                    </div>
                    <p className="text-sm text-slate-400 font-medium">No saved tables yet</p>
                    <p className="text-xs text-slate-600 mt-1.5 max-w-[210px]">
                        After a catalog query, tell the assistant &ldquo;save this as &lt;name&gt;&rdquo;.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="p-4 space-y-2">
            <p className="text-xs text-slate-500 px-1 mb-2">{tables.length} saved table{tables.length !== 1 ? "s" : ""}</p>
            {tables.map((table) => (
                <div key={table.name} className="glass-control rounded-xl p-3 space-y-1.5 group">
                    <div className="flex items-start justify-between gap-2">
                        <span className="text-xs font-semibold text-white font-mono flex-1 truncate" title={table.name}>
                            {table.name}
                        </span>
                        <button onClick={() => remove(table.name)} title="Delete table"
                            className="p-1 text-slate-500 hover:text-red-400 transition-colors shrink-0 opacity-0 group-hover:opacity-100">
                            <Trash2 className="w-3.5 h-3.5" />
                        </button>
                    </div>
                    <p className="text-[10px] text-slate-400">
                        {table.rowcount} rows
                        {table.catalog ? ` · ${table.catalog}${table.table ? `.${table.table}` : ""}` : ""}
                        {" · "}{timeAgo(new Date(table.saved_at * 1000))}
                    </p>
                    {table.description && <p className="text-[10px] text-slate-500 line-clamp-2">{table.description}</p>}
                </div>
            ))}
        </div>
    );
}

function DataLabPanelContent() {
    const [tab, setTab] = useState<"jobs" | "tables">("jobs");
    return (
        <div className="flex flex-col h-full">
            <div className="flex gap-1 px-4 pt-3">
                {(["jobs", "tables"] as const).map((key) => (
                    <button key={key} onClick={() => setTab(key)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${tab === key ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}>
                        {key === "jobs" ? "Jobs" : "My tables"}
                    </button>
                ))}
            </div>
            <div className="flex-1 overflow-y-auto">
                {tab === "jobs" ? <DatalabJobsContent /> : <MyTablesContent />}
            </div>
        </div>
    );
}

/* ────────────────────────────────────────────
   HISTORY HELPERS — date buckets + letter tints
   ──────────────────────────────────────────── */
const HISTORY_GROUPS = ["Today", "Yesterday", "Previous 7 days", "Older"] as const;
type HistoryGroup = typeof HISTORY_GROUPS[number];

function historyGroup(date: Date, now: Date): HistoryGroup {
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const t = date.getTime();
    const day = 24 * 60 * 60 * 1000;
    if (t >= startOfToday) return "Today";
    if (t >= startOfToday - day) return "Yesterday";
    if (t >= startOfToday - 7 * day) return "Previous 7 days";
    return "Older";
}

/** Soft 10% tints for starred-chat letter avatars, stable per conversation id. */
function letterTint(id: string, isDark: boolean): string {
    const tints = isDark
        ? ["bg-[var(--q-accent-soft)] text-primary", "bg-amber-500/10 text-amber-300", "bg-emerald-500/10 text-emerald-300"]
        : ["bg-[var(--q-accent-soft)] text-primary", "bg-amber-500/10 text-amber-600", "bg-emerald-500/10 text-emerald-600"];
    let sum = 0;
    for (let i = 0; i < id.length; i++) sum += id.charCodeAt(i);
    return tints[sum % tints.length];
}

/** Row entrance stagger (30ms), capped so a long history does not trickle in. */
const riseDelay = (i: number) => ({ animationDelay: `${Math.min(i, 14) * 30}ms` });

/* ────────────────────────────────────────────
   MAIN SIDEBAR
   ──────────────────────────────────────────── */
interface SidebarProps {
    collapsed?: boolean;
    onToggle?: () => void;
    /** "drawer" = the phone slide-over (own width, close button, search field). */
    variant?: "panel" | "drawer";
    /** Dismisses the drawer. Also fired after any navigation so the panel gets out of the way. */
    onClose?: () => void;
}

export function Sidebar({ collapsed = false, onToggle, variant = "panel", onClose }: SidebarProps) {
    const {
        conversations, activeConversationId, setActiveConversation,
        createNewConversation, selectedModel, setSelectedModel,
        fetchModels, loadConversations, loadConversationMessages,
        deleteConversation, clearAllConversations,
    } = useChatStore();

    const { user, logout, isAuthenticated, openAuthModal } = useAuthStore();
    const { theme, toggle: toggleTheme } = useThemeStore();
    const isDark = theme === "dark";
    const pathname = usePathname();
    const router = useRouter();

    // Fetch model list from backend on mount
    useEffect(() => { fetchModels(); }, [fetchModels]);

    // Load conversations from server when authenticated (auth rides the cookie)
    useEffect(() => {
        if (isAuthenticated) {
            loadConversations();
        } else {
            clearAllConversations();
        }
    }, [isAuthenticated, loadConversations, clearAllConversations]);

    const [activePanel, setActivePanel] = useState<"papers" | "datalab" | null>(null);
    const [settingsOpen, setSettingsOpen] = useState(false);
    // Set when Settings is opened from a deep link (e.g. "add an API key").
    const [settingsTab, setSettingsTab] = useState<"providerKeys" | undefined>(undefined);
    const [query, setQuery] = useState("");

    const isDrawer = variant === "drawer";
    // The filter field hides behind the header's search icon; the drawer opens
    // with it showing, since there the recents list is the way back in.
    const [searchOpen, setSearchOpen] = useState(isDrawer);
    const [closedGroups, setClosedGroups] = useState<Partial<Record<HistoryGroup, boolean>>>({});

    // Account popover (rail avatar). Fixed-positioned so it escapes the
    // overflow-hidden sidebar wrapper, even when only the rail is showing.
    const accountBtnRef = useRef<HTMLButtonElement>(null);
    const accountPopRef = useRef<HTMLDivElement>(null);
    const [accountPos, setAccountPos] = useState<{ left: number; bottom: number } | null>(null);
    useEffect(() => {
        if (!accountPos) return;
        const onDown = (e: MouseEvent) => {
            const target = e.target as Node;
            if (accountPopRef.current?.contains(target) || accountBtnRef.current?.contains(target)) return;
            setAccountPos(null);
        };
        const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setAccountPos(null); };
        document.addEventListener("mousedown", onDown);
        document.addEventListener("keydown", onKey);
        return () => {
            document.removeEventListener("mousedown", onDown);
            document.removeEventListener("keydown", onKey);
        };
    }, [accountPos]);
    const toggleAccount = () => {
        if (accountPos) { setAccountPos(null); return; }
        const btn = accountBtnRef.current;
        const r = btn?.getBoundingClientRect();
        // Anchor to the rail's right edge so the card clears the rail.
        const railRight = btn?.closest("nav")?.getBoundingClientRect().right ?? r?.right ?? 0;
        if (r) setAccountPos({ left: railRight + 8, bottom: window.innerHeight - r.bottom });
    };

    // The Data Lab entry only appears once the user has something there (a
    // background job or a saved table); most users never do. Re-checked when a
    // chat turn ends (that is when a job gets submitted), and every few
    // seconds while a job is still running so the badge clears on its own.
    const isStreaming = useChatStore((s) => s.isStreaming);
    const [dataLab, setDataLab] = useState({ jobs: 0, running: 0, tables: 0 });
    useEffect(() => {
        if (!isAuthenticated) return;
        let stopped = false;
        const check = async () => {
            try {
                const [jobs, tables] = await Promise.all([listDatalabJobs(), listMyTables()]);
                if (stopped) return;
                setDataLab({
                    jobs: jobs.length,
                    running: jobs.filter((j) => ACTIVE_JOB_STATUSES.includes(j.status)).length,
                    tables: tables.length,
                });
            } catch { /* backend offline: keep the last state */ }
        };
        check();
        const timer = setInterval(check, dataLab.running > 0 ? 5000 : 60000);
        return () => { stopped = true; clearInterval(timer); };
    }, [isAuthenticated, isStreaming, dataLab.running]);
    const showDataLab = isAuthenticated && (dataLab.jobs > 0 || dataLab.tables > 0 || activePanel === "datalab");

    const togglePanel = (panel: "papers" | "datalab") => {
        setActivePanel((prev) => (prev === panel ? null : panel));
    };

    // Rail version: from the collapsed rail there is no panel to slide the
    // overlay over, so open the sidebar with that overlay already showing.
    const openRailPanel = (panel: "papers" | "datalab") => {
        if (collapsed) {
            setActivePanel(panel);
            onToggle?.();
        } else {
            togglePanel(panel);
        }
    };

    const handleChatRail = () => {
        // Switching from another page/overlay reveals chat; clicking the
        // already visible chat panel toggles it closed.
        if (variant === "drawer") {
            if (pathname !== "/" || !activePanel) onClose?.();
        } else if (collapsed || (pathname === "/" && !activePanel)) {
            onToggle?.();
        }
        setActivePanel(null);
        if (pathname !== "/") {
            router.push("/");
        }
    };

    const handleSelectConversation = (convId: string) => {
        setActiveConversation(convId);
        setActivePanel(null);
        // Load messages from server if not already loaded
        if (isAuthenticated) {
            loadConversationMessages(convId);
        }
        onClose?.();
    };

    const handleDeleteConversation = (e: React.MouseEvent, convId: string) => {
        e.stopPropagation();
        if (isAuthenticated) {
            deleteConversation(convId);
        }
    };

    const handleNewChat = () => {
        createNewConversation();
        setActivePanel(null);
        if (pathname !== "/") {
            router.push("/");
        }
        onClose?.();
    };

    const visibleConversations = query.trim()
        ? conversations.filter((c) => c.title.toLowerCase().includes(query.trim().toLowerCase()))
        : conversations;

    // Starred chats get their own section; the rest bucket by last activity.
    const starred = visibleConversations.filter((c) => c.isStarred);
    const now = new Date();
    const buckets = HISTORY_GROUPS
        .map((group) => ({
            group,
            items: visibleConversations.filter((c) =>
                !c.isStarred && historyGroup(new Date(c.updatedAt ?? c.createdAt), now) === group),
        }))
        .filter((g) => g.items.length > 0);
    // First stagger index of each bucket, so rows rise in one continuous wave.
    const grouped = buckets.map((g, gi) => ({
        ...g,
        start: starred.length + buckets.slice(0, gi).reduce((n, prev) => n + prev.items.length, 0),
    }));

    // Get user initials from display name, or fallback to username initials
    const initials = user?.display_name
        ? user.display_name.split(' ').filter(Boolean).map((n: string) => n[0]).join('').substring(0, 2).toUpperCase()
        : user?.username
            ? user.username.substring(0, 2).toUpperCase()
            : "U";

    const avatar = (size: string) => user?.picture_url ? (
        <img src={user.picture_url} alt="" className={`${size} rounded-full object-cover`} referrerPolicy="no-referrer" />
    ) : (
        <span className={`${size} flex items-center justify-center rounded-full border border-[var(--q-border)] bg-[var(--q-card)] text-[12px] font-semibold text-[var(--q-text)]`}>
            {initials}
        </span>
    );

    const railIcon = "size-[18px]";

    /* ── RAIL — always visible on desktop, also inside the phone drawer ── */
    const rail = (
        <nav
            aria-label="Primary"
            className={`flex h-full w-[var(--q-sidebar-rail-width)] shrink-0 flex-col items-center bg-[var(--q-rail)] py-3 ${collapsed ? "border-r border-[var(--q-border)]" : ""}`}
        >
            {variant === "panel" ? (
                <button
                    type="button"
                    onClick={onToggle}
                    className="mb-4 flex size-[42px] items-center justify-center rounded-full border border-[var(--q-border)] bg-[var(--q-card)] transition-colors hover:border-[var(--q-border-strong)]"
                    title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                    aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                    aria-expanded={!collapsed}
                >
                    <img src="/quasar_logo.png" alt="" className="size-[30px] object-contain" />
                </button>
            ) : (
                <Link href="/" onClick={onClose} className="mb-4 flex size-[42px] items-center justify-center rounded-full border border-[var(--q-border)] bg-[var(--q-card)]" title="Quasar" aria-label="Quasar home">
                    <img src="/quasar_logo.png" alt="Quasar" className="size-[30px] object-contain" />
                </Link>
            )}

            <div className="flex flex-col items-center gap-2">
                <button type="button" onClick={handleChatRail} className="q-rail-btn"
                    data-active={pathname === "/" && !activePanel ? "true" : undefined}
                    aria-expanded={!collapsed && pathname === "/" && !activePanel}
                    title="Chat" aria-label="Chat">
                    <MessageSquare className={railIcon} strokeWidth={1.75} />
                </button>
                <Link href="/gallery" onClick={onClose} className="q-rail-btn"
                    data-active={pathname === "/gallery" ? "true" : undefined}
                    title="Recipe Gallery" aria-label="Recipe Gallery">
                    <BookOpen className={railIcon} strokeWidth={1.75} />
                </Link>
                <button type="button" onClick={() => openRailPanel("papers")} className="q-rail-btn"
                    data-active={activePanel === "papers" ? "true" : undefined}
                    title="Saved Papers" aria-label="Saved Papers">
                    <Bookmark className={railIcon} strokeWidth={1.75} />
                </button>
                {/* Data Lab shows only once the user has a job or a saved table. */}
                {showDataLab && (
                    <button type="button" onClick={() => openRailPanel("datalab")} className="q-rail-btn"
                        data-active={activePanel === "datalab" ? "true" : undefined}
                        title={dataLab.running > 0
                            ? `Data Lab: ${dataLab.running} job${dataLab.running === 1 ? "" : "s"} running`
                            : "Data Lab jobs & saved tables"}
                        aria-label="Data Lab">
                        <span className="relative">
                            <Database className={railIcon} strokeWidth={1.75} />
                            {dataLab.running > 0 && (
                                <span className="absolute -right-1 -top-1 flex h-2 w-2" aria-hidden="true">
                                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60 motion-reduce:hidden" />
                                    <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                                </span>
                            )}
                        </span>
                    </button>
                )}
                <Link href="/help" onClick={onClose} className="q-rail-btn"
                    data-active={pathname === "/help" ? "true" : undefined}
                    title="Help and docs" aria-label="Help and docs">
                    <HelpCircle className={railIcon} strokeWidth={1.75} />
                </Link>
            </div>

            <div className="mt-auto flex flex-col items-center gap-2">
                <button type="button" onClick={toggleTheme} className="q-rail-btn"
                    title={isDark ? "Switch to light theme" : "Switch to dark theme"}
                    aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}>
                    {isDark ? <Moon className={railIcon} strokeWidth={1.75} /> : <Sun className={railIcon} strokeWidth={1.75} />}
                </button>
                <button type="button" onClick={() => setSettingsOpen(true)} className="q-rail-btn" title="Settings" aria-label="Settings">
                    <Settings className={railIcon} strokeWidth={1.75} />
                </button>
                {isAuthenticated ? (
                    <button
                        ref={accountBtnRef}
                        type="button"
                        onClick={toggleAccount}
                        className="relative mt-1 rounded-full"
                        title={user?.display_name || user?.username || "Account"}
                        aria-label="Account"
                        aria-haspopup="menu"
                        aria-expanded={!!accountPos}
                    >
                        {avatar("size-9")}
                        <span className="absolute right-0 top-0 size-2.5 rounded-full border-2 border-[var(--q-rail)] bg-emerald-500" aria-hidden="true" />
                    </button>
                ) : (
                    <button type="button" onClick={openAuthModal} className="q-rail-btn mt-1" title="Sign in" aria-label="Sign in">
                        <UserIcon className={railIcon} strokeWidth={1.75} />
                    </button>
                )}
            </div>
        </nav>
    );

    /* ── HISTORY ROW — shared by Starred and the date groups ── */
    const historyRow = (conv: typeof conversations[number], index: number, starredRow: boolean) => {
        const isActive = conv.id === activeConversationId;
        return (
            <div key={conv.id} className="q-rise relative group/item" style={riseDelay(index)}>
                <button onClick={() => handleSelectConversation(conv.id)}
                    title={conv.title}
                    className={`flex w-full items-center gap-2.5 rounded-xl border text-left text-[13px] transition-colors ${starredRow ? "px-2 py-1.5" : "px-2.5 py-1.5"} ${isActive
                        ? "border-[var(--q-border)] bg-[var(--q-card)] text-[var(--q-text)]"
                        : "border-transparent text-[var(--q-text-secondary)] hover:bg-[var(--q-glass-control-hover)] hover:text-[var(--q-text)]"}`}>
                    {starredRow && (
                        <span className={`q-letter ${letterTint(conv.id, isDark)}`} aria-hidden="true">
                            {(conv.title.trim()[0] || "?").toUpperCase()}
                        </span>
                    )}
                    <span className="min-w-0 flex-1 truncate pr-6">{conv.title}</span>
                </button>
                {/* "…" at rest on starred rows; hovering any row swaps in delete. */}
                {starredRow && (
                    <MoreHorizontal aria-hidden="true"
                        className="pointer-events-none absolute right-2.5 top-1/2 size-4 -translate-y-1/2 text-[var(--q-text-faint)] transition-opacity group-hover/item:opacity-0" />
                )}
                <button
                    onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                    onClick={(e) => handleDeleteConversation(e, conv.id)}
                    title="Delete conversation"
                    aria-label="Delete conversation"
                    className={`absolute right-1.5 top-1/2 -translate-y-1/2 rounded-full p-1 text-[var(--q-text-faint)] transition-opacity hover:bg-red-500/10 hover:text-red-500 ${isActive && !starredRow ? "opacity-70" : "opacity-0"} group-hover/item:opacity-100 focus-visible:opacity-100`}
                >
                    <Trash2 className="size-3.5" strokeWidth={1.75} />
                </button>
            </div>
        );
    };

    return (
        <aside className={`relative flex h-full overflow-hidden ${isDrawer
            ? "w-full glass-drawer"
            : collapsed
                ? "w-[var(--q-sidebar-rail-width)] shrink-0"
                : "w-[calc(var(--q-sidebar-rail-width)+var(--q-sidebar-width))] shrink-0"}`}>
            {rail}

            {/* ── PANEL — title, New Chat, Starred, history, model card.
                The rail and chat header toggle the panel. The drawer gets
                its own close button, since on a
                phone there is no chat header behind it to reach. ── */}
            {!collapsed && (
                <div className={`relative flex min-w-0 flex-col bg-[var(--q-bg)] ${isDrawer
                    ? "flex-1"
                    : "w-[var(--q-sidebar-width)] shrink-0 border-r border-[var(--q-border)]"}`}>
                    {/* Header */}
                    <div className="flex h-[66px] shrink-0 items-center gap-1 px-4">
                        <h2 className="flex-1 text-[15px] font-medium text-[var(--q-text)]">Chat</h2>
                        <button
                            type="button"
                            onClick={() => setSearchOpen((v) => !v)}
                            className="flex size-8 items-center justify-center rounded-full text-[var(--q-text-muted)] transition-colors hover:bg-[var(--q-glass-control-hover)] hover:text-[var(--q-text)]"
                            title="Search research"
                            aria-label="Search research"
                            aria-expanded={searchOpen}
                        >
                            <Search className="size-4" strokeWidth={1.75} />
                        </button>
                        {isDrawer && (
                            <button
                                type="button"
                                onClick={onClose}
                                aria-label="Close navigation"
                                className="flex size-8 items-center justify-center rounded-full text-[var(--q-text-muted)] transition-colors hover:bg-[var(--q-glass-control-hover)] hover:text-[var(--q-text)]"
                            >
                                <X className="size-4" strokeWidth={1.75} />
                            </button>
                        )}
                    </div>

                    {/* New Chat */}
                    <div className="shrink-0 px-3">
                        <button onClick={handleNewChat} className="q-pill-ink h-10 w-full px-4 text-[13px]">
                            <Plus className="size-4" strokeWidth={1.75} />
                            <span>New Chat</span>
                            <Sparkles className="size-3.5" strokeWidth={1.75} />
                        </button>
                    </div>

                    {/* Conversation filter */}
                    {(searchOpen || query) && (
                        <div className="shrink-0 px-3 pt-2.5">
                            <label className="flex items-center gap-2 rounded-full border border-[var(--q-border)] bg-[var(--q-card)] px-3 py-1.5 focus-within:border-[var(--q-border-strong)]">
                                <Search className="size-3.5 shrink-0 text-[var(--q-text-faint)]" strokeWidth={1.75} />
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === "Escape" && !isDrawer) { setQuery(""); setSearchOpen(false); } }}
                                    placeholder="Search research…"
                                    aria-label="Search research"
                                    autoFocus={!isDrawer}
                                    className="w-full min-w-0 bg-transparent border-none outline-none text-[13px] text-[var(--q-text)] placeholder:text-[var(--q-text-faint)]"
                                />
                                {query && (
                                    <button type="button" onClick={() => setQuery("")} aria-label="Clear search"
                                        className="shrink-0 text-[var(--q-text-faint)] hover:text-[var(--q-text)]">
                                        <X className="size-3.5" strokeWidth={1.75} />
                                    </button>
                                )}
                            </label>
                        </div>
                    )}

                    {/* Conversation History */}
                    <div className="mt-3 flex-1 overflow-y-auto px-3 pb-3">
                        {visibleConversations.length > 0 ? (
                            <>
                                {starred.length > 0 && (
                                    <section className="mb-2 border-b border-[var(--q-border)] pb-3">
                                        <div className="flex items-center gap-1.5 px-1 py-1.5 text-[13px] text-[var(--q-text-faint)]">
                                            <Star className="size-3.5" strokeWidth={1.75} />
                                            Starred
                                        </div>
                                        <div className="space-y-0.5">
                                            {starred.map((conv, i) => historyRow(conv, i, true))}
                                        </div>
                                    </section>
                                )}
                                {grouped.map(({ group, items, start }) => {
                                    const closed = !!closedGroups[group];
                                    return (
                                        <section key={group} className="mt-1">
                                            <button
                                                type="button"
                                                onClick={() => setClosedGroups((prev) => ({ ...prev, [group]: !prev[group] }))}
                                                aria-expanded={!closed}
                                                className="flex w-full items-center justify-between rounded-lg px-1 py-1.5 text-[13px] text-[var(--q-text-faint)] transition-colors hover:text-[var(--q-text-muted)]"
                                            >
                                                {group}
                                                <ChevronDown className={`size-3.5 transition-transform ${closed ? "-rotate-90" : ""}`} strokeWidth={1.75} />
                                            </button>
                                            {!closed && (
                                                <div className="mb-1 space-y-0.5">
                                                    {items.map((conv, i) => historyRow(conv, start + i, false))}
                                                </div>
                                            )}
                                        </section>
                                    );
                                })}
                            </>
                        ) : query.trim() ? (
                            <div className="px-3 py-8 text-center">
                                <p className="text-[13px] text-[var(--q-text-muted)]">No matches for “{query.trim()}”.</p>
                            </div>
                        ) : (
                            <div className="px-3 py-8 text-center">
                                <p className="text-[13px] text-[var(--q-text-muted)]">No conversations yet.</p>
                                <p className="mt-1 text-[12px] text-[var(--q-text-faint)]">Start a new chat to begin!</p>
                            </div>
                        )}
                    </div>

                    {/* Bottom card — model picker (plus sign-in while signed out) */}
                    <div className="shrink-0 space-y-2 px-3 pb-3">
                        {!isAuthenticated && (
                            <button
                                onClick={openAuthModal}
                                className="q-pill h-9 w-full px-4 text-[13px]"
                            >
                                <UserIcon className="size-4" strokeWidth={1.75} />
                                Sign In / Sign Up
                            </button>
                        )}
                        <ModelDropdown
                            selectedModel={selectedModel}
                            onSelect={setSelectedModel}
                            onAddProviderKey={() => { setSettingsTab("providerKeys"); setSettingsOpen(true); }}
                        />
                    </div>

                    {/* ── OVERLAY PANELS ── */}
                    <OverlayPanel open={activePanel === "papers"} onClose={() => setActivePanel(null)} title="Saved Papers" icon={Bookmark}>
                        <SavedPapersContent />
                    </OverlayPanel>
                    <OverlayPanel open={activePanel === "datalab"} onClose={() => setActivePanel(null)} title="Data Lab" icon={Database}>
                        <DataLabPanelContent />
                    </OverlayPanel>
                </div>
            )}

            {/* Account popover — name, username, Log out */}
            {isAuthenticated && accountPos && (
                <div
                    ref={accountPopRef}
                    role="menu"
                    aria-label="Account"
                    className="fixed z-[70] w-56 rounded-2xl border border-[var(--q-border)] bg-[var(--q-card)] p-1.5"
                    style={{ left: accountPos.left, bottom: accountPos.bottom, boxShadow: "var(--q-popover-shadow)" }}
                >
                    <div className="flex items-center gap-2.5 px-2.5 py-2">
                        {avatar("size-8")}
                        <div className="flex min-w-0 flex-col">
                            <span className="truncate text-[13px] font-medium text-[var(--q-text)]">{user?.display_name || user?.username || "User"}</span>
                            <span className="truncate text-[12px] text-[var(--q-text-muted)]">{user?.username || ""}</span>
                        </div>
                    </div>
                    <div className="my-1 h-px bg-[var(--q-border)]" />
                    <button
                        type="button"
                        role="menuitem"
                        onClick={() => { setAccountPos(null); logout(); }}
                        className="flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2 text-left text-[13px] text-[var(--q-text-secondary)] transition-colors hover:bg-[var(--q-glass-control-hover)] hover:text-[var(--q-text)]"
                    >
                        <LogOut className="size-4 text-[var(--q-text-muted)]" strokeWidth={1.75} />
                        Log out
                    </button>
                </div>
            )}

            {/* Settings float modal — rendered outside sidebar via portal-like pattern */}
            <SettingsModal open={settingsOpen} initialTab={settingsTab} onClose={() => { setSettingsOpen(false); setSettingsTab(undefined); }} />
        </aside>
    );
}
