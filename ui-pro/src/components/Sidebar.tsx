"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useChatStore } from "../lib/store";
import { useAuthStore } from "../lib/auth-store";
import {
    Plus, MessageSquare, History, Bookmark, Settings, HelpCircle,
    X, ExternalLink, Github, BookOpen, Search,
    Telescope, FileText, Zap, Check, LogOut, User as UserIcon, Trash2,
    Database, RefreshCw
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
            <div className="flex items-center justify-between px-5 py-4 border-b border-slate-700/50">
                <div className="flex items-center gap-2.5">
                    <Icon className="w-5 h-5 text-primary" />
                    <h2 className="text-sm font-semibold text-white">{title}</h2>
                </div>
                <button onClick={onClose} className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-700/50 hover:text-white transition-colors">
                    <X className="w-4 h-4" />
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
                const active = ["queued", "running", "submitted"].includes(job.status);
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

    const togglePanel = (panel: "papers" | "datalab") => {
        setActivePanel((prev) => (prev === panel ? null : panel));
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

    // Get user initials from display name, or fallback to username initials
    const initials = user?.display_name
        ? user.display_name.split(' ').filter(Boolean).map((n: string) => n[0]).join('').substring(0, 2).toUpperCase()
        : user?.username
            ? user.username.substring(0, 2).toUpperCase()
            : "U";

    if (collapsed) {
        return (
            <aside className="relative w-[var(--q-sidebar-rail-width)] glass-sidebar border-r border-slate-700/50 flex flex-col items-center h-full shrink-0 overflow-hidden py-3">
                <button
                    type="button"
                    onClick={onToggle}
                    className="mb-5 flex size-11 items-center justify-center rounded-xl transition-colors hover:bg-white/10"
                    title="Expand sidebar"
                    aria-label="Expand sidebar"
                >
                    <img src="/quasar_logo.png" alt="" className="size-8 object-contain" />
                </button>

                <div className="flex flex-col items-center gap-2">
                    <button onClick={handleNewChat} className="sidebar-rail-button text-primary" title="New chat" aria-label="New chat">
                        <Plus className="w-5 h-5" />
                    </button>
                    <button type="button" onClick={onToggle} className="sidebar-rail-button" title="Search and history" aria-label="Search and history">
                        <Search className="w-5 h-5" />
                    </button>
                    <button type="button" onClick={onToggle} className="sidebar-rail-button" title="Recent chats" aria-label="Recent chats">
                        <MessageSquare className="w-5 h-5" />
                    </button>
                    <Link
                        href="/gallery"
                        className={`sidebar-rail-button ${pathname === "/gallery" ? "text-primary bg-primary/10" : ""}`}
                        title="Recipe Gallery"
                        aria-label="Recipe Gallery"
                    >
                        <BookOpen className="w-5 h-5" />
                    </Link>
                </div>

                <div className="mt-auto flex flex-col items-center gap-2">
                    <Link
                        href="/help"
                        className={`sidebar-rail-button ${pathname === "/help" ? "text-primary bg-primary/10" : ""}`}
                        title="Help and docs"
                        aria-label="Help and docs"
                    >
                        <HelpCircle className="w-5 h-5" />
                    </Link>
                    <button type="button" onClick={() => setSettingsOpen(true)} className="sidebar-rail-button" title="Settings" aria-label="Settings">
                        <Settings className="w-5 h-5" />
                    </button>
                    {isAuthenticated ? (
                        <button type="button" onClick={onToggle} className="mt-2 rounded-full" title={user?.display_name || user?.username || "Account"} aria-label="Account">
                            {user?.picture_url ? (
                                <img src={user.picture_url} alt="" className="size-9 rounded-full object-cover shadow-md" referrerPolicy="no-referrer" />
                            ) : (
                                <span className="flex size-9 items-center justify-center rounded-full bg-gradient-to-tr from-[#818cf8] to-[#c77dff] text-on-accent text-xs font-bold shadow-md">
                                    {initials}
                                </span>
                            )}
                        </button>
                    ) : (
                        <button type="button" onClick={openAuthModal} className="sidebar-rail-button mt-2" title="Sign in" aria-label="Sign in">
                            <UserIcon className="w-5 h-5" />
                        </button>
                    )}
                </div>

                <SettingsModal open={settingsOpen} initialTab={settingsTab} onClose={() => { setSettingsOpen(false); setSettingsTab(undefined); }} />
            </aside>
        );
    }
    return (
        <aside className={`relative flex flex-col h-full overflow-hidden ${isDrawer
            ? "w-full glass-drawer"
            : "w-[var(--q-sidebar-width)] glass-sidebar border-r border-slate-700/50 shrink-0"}`}>
            {/* Collapse is handled by the single toggle in the chat header (ChatArea)
                — no duplicate button here. The drawer gets its own close button,
                since on a phone there is no chat header behind it to reach. */}
            {/* Logo */}
            {isDrawer ? (
                <div className="flex items-center gap-[11px] px-4 py-3">
                    <img src="/quasar_logo.png" alt="" className="size-[38px] shrink-0 object-contain" />
                    <Link href="/" onClick={onClose} className="flex flex-1 flex-col min-w-0">
                        <h1 className="text-[15.5px] font-bold tracking-[-0.01em]" style={{ color: "var(--q-text)" }}>QUASAR</h1>
                        <span className="text-[10.5px]" style={{ color: "var(--q-text-secondary)" }}>Research Assistant</span>
                    </Link>
                    <button
                        type="button"
                        onClick={onClose}
                        aria-label="Close navigation"
                        className="glass-control flex size-[32px] shrink-0 items-center justify-center rounded-full"
                        style={{ color: "var(--q-text-secondary)" }}
                    >
                        <X className="size-[13px]" strokeWidth={1.9} />
                    </button>
                </div>
            ) : (
                <Link href="/" className="p-6 flex items-center gap-3">
                    <img src="/quasar_logo.png" alt="Quasar" className="size-[60px] object-contain" />
                    <div className="flex flex-col">
                        <h1 className="text-lg font-bold tracking-tight text-white">QUASAR</h1>
                        <span className="text-xs text-slate-400 font-medium">Research Assistant</span>
                    </div>
                </Link>
            )}

            {/* New Chat */}
            <div className={isDrawer ? "px-3.5 mb-2.5" : "px-4 mb-6"}>
                <button onClick={handleNewChat}
                    className={`w-full flex items-center justify-center gap-2 font-semibold px-4 group ${isDrawer
                        ? "btn-accent py-2.5 rounded-[13px] text-[13.5px] shadow-[0_10px_24px_rgba(147,51,234,0.45)]"
                        : "btn-new-chat py-3 rounded-xl"}`}>
                    <Plus className={`${isDrawer ? "size-[17px]" : "w-5 h-5"} transition-transform group-hover:rotate-90`} />
                    <span>New Chat</span>
                </button>
            </div>

            {/* Conversation filter — drawer only, where the recents list is the
                primary way back into a session. */}
            {isDrawer && (
                <div className="px-3.5 mb-2.5">
                    <label className="glass-control flex items-center gap-2.5 rounded-[11px] px-3 py-2">
                        <Search className="size-[14px] shrink-0" style={{ color: "var(--q-text-muted)" }} />
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Search research…"
                            aria-label="Search research"
                            className="w-full min-w-0 bg-transparent border-none outline-none text-xs placeholder:text-[var(--q-text-muted)]"
                        />
                    </label>
                </div>
            )}

            <div className={`${isDrawer ? "px-3.5 mb-3" : "px-4 mb-4"} space-y-1`}>
                <Link href="/gallery" onClick={onClose}
                    className={`w-full flex items-center gap-3 rounded-xl transition-colors ${isDrawer ? "px-3 py-2.5 text-[12.5px] font-medium" : "px-3 py-2.5 text-sm"} ${pathname === "/gallery" ? "bg-indigo-400/15 text-indigo-300 border border-indigo-400/30" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}>
                    <BookOpen className={isDrawer ? "size-[15px]" : "w-4 h-4"} />
                    Recipe Gallery
                </Link>
            </div>

            {/* Conversation History */}
            <div className="flex-1 overflow-y-auto px-3 space-y-1">
                {visibleConversations.length > 0 ? (
                    <>
                        <div className="px-3 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">Recent Research</div>
                        {visibleConversations.map((conv) => {
                            const isActive = conv.id === activeConversationId;
                            return (
                                <div key={conv.id} className="relative group/item">
                                    <button onClick={() => handleSelectConversation(conv.id)}
                                        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-all text-left ${isActive ? "glass-active text-white" : "text-slate-300 hover:bg-white/10 group"}`}>
                                        {isActive ? <MessageSquare className="w-5 h-5 text-primary shrink-0" /> : <History className="w-5 h-5 text-slate-400 group-hover:text-primary transition-colors shrink-0" />}
                                        <div className="flex flex-col overflow-hidden flex-1">
                                            <span className="text-sm font-medium truncate">{conv.title}</span>
                                            <span className="text-[10px] text-slate-500">{timeAgo(conv.updatedAt)}</span>
                                        </div>
                                    </button>
                                    {/* Delete button — visible on hover, always visible when active */}
                                    <button
                                        onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                                        onClick={(e) => handleDeleteConversation(e, conv.id)}
                                        title="Delete conversation"
                                        className={`absolute right-1 top-1/2 -translate-y-1/2 p-2 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-all ${isActive ? "opacity-70" : "opacity-0"} group-hover/item:opacity-100`}
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </button>
                                </div>
                            );
                        })}
                    </>
                ) : query.trim() ? (
                    <div className="px-3 py-8 text-center">
                        <p className="text-sm text-slate-500">No matches for “{query.trim()}”.</p>
                    </div>
                ) : (
                    <div className="px-3 py-8 text-center">
                        <p className="text-sm text-slate-500">No conversations yet.</p>
                        <p className="text-xs text-slate-600 mt-1">Start a new chat to begin!</p>
                    </div>
                )}
            </div>

            {/* Bottom Controls — compact, so Recent Research keeps its room */}
            <div className="p-3 border-t border-slate-700/50 space-y-2.5">
                {/* Model Dropdown */}
                <ModelDropdown
                    selectedModel={selectedModel}
                    onSelect={setSelectedModel}
                    onAddProviderKey={() => { setSettingsTab("providerKeys"); setSettingsOpen(true); }}
                />

                {/* Compact action row: Saved · Data Lab · Settings · Help */}
                <div className="flex gap-1.5">
                    <button onClick={() => togglePanel("papers")}
                        title="Saved Papers"
                        className={`flex-1 flex items-center justify-center gap-1.5 px-1 py-2 rounded-lg transition-colors text-xs ${activePanel === "papers" ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}>
                        <Bookmark className="w-4 h-4 shrink-0" />Saved
                    </button>
                    <button onClick={() => togglePanel("datalab")}
                        title="Data Lab jobs & saved tables"
                        className={`flex-1 flex items-center justify-center gap-1.5 px-1 py-2 rounded-lg transition-colors text-xs ${activePanel === "datalab" ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}>
                        <Database className="w-4 h-4 shrink-0" />Data Lab
                    </button>
                    <button onClick={() => setSettingsOpen(true)}
                        title="Settings"
                        className="flex-1 flex items-center justify-center gap-1.5 px-1 py-2 rounded-lg transition-colors text-xs text-slate-400 hover:bg-white/10 hover:text-white">
                        <Settings className="w-4 h-4 shrink-0" />Settings
                    </button>
                    <Link href="/help" onClick={onClose}
                        title="Help & Docs"
                        className={`flex-1 flex items-center justify-center gap-1.5 px-1 py-2 rounded-lg transition-colors text-xs ${pathname === "/help" ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}>
                        <HelpCircle className="w-4 h-4 shrink-0" />Help
                    </Link>
                </div>

                {/* User / Sign-in (Log Out folded into this row when signed in) */}
                <div className="pt-2.5 border-t border-slate-700/50">
                    {isAuthenticated ? (
                        <div className="flex items-center gap-3 px-1">
                            {user?.picture_url ? (
                                <img src={user.picture_url} alt={user.display_name || "User"} className="size-8 rounded-full object-cover shadow-md" referrerPolicy="no-referrer" />
                            ) : (
                                <div className="size-8 rounded-full bg-gradient-to-tr from-[#818cf8] to-[#c77dff] flex items-center justify-center text-on-accent text-xs font-bold shadow-md">
                                    {initials}
                                </div>
                            )}
                            <div className="flex flex-col flex-1 overflow-hidden">
                                <span className="text-sm font-semibold text-white truncate">{user?.display_name || user?.username || "User"}</span>
                                <span className="text-[10px] text-slate-400 truncate">{user?.username || ""}</span>
                            </div>
                            <button onClick={logout} title="Log out"
                                className="shrink-0 p-1.5 rounded-lg text-red-500/70 hover:bg-red-500/10 hover:text-red-500 transition-colors">
                                <LogOut className="w-4 h-4" />
                            </button>
                        </div>
                    ) : (
                        <button
                            onClick={openAuthModal}
                            className="glass-control w-full text-[var(--q-text)] font-medium text-sm py-2.5 px-4 rounded-xl transition-all flex items-center justify-center gap-2"
                        >
                            <UserIcon className="w-4 h-4" />
                            Sign In / Sign Up
                        </button>
                    )}
                </div>
            </div>

            {/* ── OVERLAY PANELS ── */}
            <OverlayPanel open={activePanel === "papers"} onClose={() => setActivePanel(null)} title="Saved Papers" icon={Bookmark}>
                <SavedPapersContent />
            </OverlayPanel>
            <OverlayPanel open={activePanel === "datalab"} onClose={() => setActivePanel(null)} title="Data Lab" icon={Database}>
                <DataLabPanelContent />
            </OverlayPanel>

            {/* Settings float modal — rendered outside sidebar via portal-like pattern */}
            <SettingsModal open={settingsOpen} initialTab={settingsTab} onClose={() => { setSettingsOpen(false); setSettingsTab(undefined); }} />
        </aside>
    );
}
