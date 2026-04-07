"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useChatStore } from "../lib/store";
import { useAuthStore } from "../lib/auth-store";
import {
    Plus, MessageSquare, History, Bookmark, Settings, HelpCircle,
    ChevronDown, Bot, X, ExternalLink, Github, BookOpen, Search,
    Telescope, FileText, Zap, Check, LogOut, User as UserIcon, Trash2
} from "lucide-react";
import { SettingsModal } from "./SettingsModal";

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
        <div className="absolute inset-0 z-50 flex flex-col bg-sidebar-dark animate-in fade-in slide-in-from-left-2 duration-200">
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
   MODEL DROPDOWN
   ──────────────────────────────────────────── */
function ModelDropdown({ selectedModel, availableModels, onSelect }: {
    selectedModel: string;
    availableModels: string[];
    onSelect: (model: string) => void;
}) {
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClick(e: MouseEvent) {
            if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
        }
        document.addEventListener("mousedown", handleClick);
        return () => document.removeEventListener("mousedown", handleClick);
    }, []);

    const openaiModels = availableModels.filter(m => m.startsWith("gpt-") || m.startsWith("o1") || m.startsWith("o3") || m.startsWith("o4"));
    const geminiModels = availableModels.filter(m => m.startsWith("gemini-") || m.startsWith("gemma-"));
    const claudeModels = availableModels.filter(m => m.startsWith("claude-"));
    const localModels = availableModels.filter(m => m.startsWith("local/"));

    const renderGroup = (label: string, models: string[]) => models.length === 0 ? null : (
        <>
            <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                {label}
            </div>
            {models.map((model) => (
                <button key={model} onClick={() => { onSelect(model); setOpen(false); }}
                    className={`w-full flex items-center justify-between px-3 py-2 text-sm transition-colors ${model === selectedModel ? "bg-primary/10 text-primary" : "text-slate-300 hover:bg-slate-700/50 hover:text-white"}`}>
                    <span className="font-medium truncate">{model.startsWith("local/") ? model.replace("local/", "") : model}</span>
                    {model === selectedModel && <Check className="w-4 h-4 text-primary shrink-0 ml-1" />}
                </button>
            ))}
        </>
    );

    return (
        <div ref={ref} className="relative">
            <button onClick={() => setOpen(!open)}
                className="flex items-center justify-between w-full px-3 py-2.5 text-xs font-medium text-slate-300 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors">
                <div className="flex items-center gap-2"><Bot className="w-4 h-4" /><span className="truncate">Model: {selectedModel.startsWith("local/") ? selectedModel.replace("local/", "🖥️ ") : selectedModel}</span></div>
                <ChevronDown className={`w-4 h-4 shrink-0 transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
            </button>

            {open && (
                <div className="absolute bottom-full left-0 right-0 mb-1.5 bg-slate-800 border border-slate-600/50 rounded-xl shadow-2xl shadow-black/40 overflow-hidden z-50 animate-in fade-in slide-in-from-bottom-2 duration-150 max-h-72 overflow-y-auto">
                    {renderGroup("OpenAI", openaiModels)}
                    {claudeModels.length > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("Anthropic Claude", claudeModels)}
                    {geminiModels.length > 0 && (openaiModels.length > 0 || claudeModels.length > 0) && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("Google Gemini", geminiModels)}
                    {localModels.length > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("🖥️ Local LLM", localModels)}
                </div>
            )}
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
                                : "text-slate-300 bg-slate-800/50 hover:bg-slate-700/50"
                                }`}>
                            <div className="flex items-center gap-2">
                                <Bot className="w-4 h-4" />
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
                <div className="bg-slate-800/50 rounded-lg p-3 flex items-center gap-3">
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
                <div className="bg-slate-800/50 rounded-lg p-3 space-y-1.5">
                    <div className="flex justify-between text-xs"><span className="text-slate-500">Version</span><span className="text-slate-300 font-mono">2.0.0</span></div>
                    <div className="flex justify-between text-xs"><span className="text-slate-500">Engine</span><span className="text-slate-300 font-mono">Responses API</span></div>
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
                    <div className="w-14 h-14 rounded-2xl bg-slate-800/80 flex items-center justify-center mb-4">
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
                    <div key={paper.id} className="bg-slate-800/50 rounded-xl p-3 space-y-1.5 group hover:bg-slate-800/80 transition-colors">
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
   MAIN SIDEBAR
   ──────────────────────────────────────────── */
export function Sidebar() {
    const {
        conversations, activeConversationId, setActiveConversation,
        createNewConversation, selectedModel, availableModels, setSelectedModel,
        fetchModels, loadConversations, loadConversationMessages,
        deleteConversation, clearAllConversations,
    } = useChatStore();

    const { user, logout, isAuthenticated, openAuthModal, token } = useAuthStore();
    const pathname = usePathname();

    // Fetch model list from backend on mount
    useEffect(() => { fetchModels(); }, [fetchModels]);

    // Load conversations from server when authenticated
    useEffect(() => {
        if (isAuthenticated && token) {
            loadConversations(token);
        } else {
            clearAllConversations();
        }
    }, [isAuthenticated, token, loadConversations, clearAllConversations]);

    const [activePanel, setActivePanel] = useState<"papers" | null>(null);
    const [settingsOpen, setSettingsOpen] = useState(false);

    const togglePanel = (panel: "papers") => {
        setActivePanel((prev) => (prev === panel ? null : panel));
    };

    const handleSelectConversation = (convId: string) => {
        setActiveConversation(convId);
        setActivePanel(null);
        // Load messages from server if not already loaded
        if (token) {
            loadConversationMessages(convId, token);
        }
    };

    const handleDeleteConversation = (e: React.MouseEvent, convId: string) => {
        e.stopPropagation();
        if (token) {
            deleteConversation(convId, token);
        }
    };

    // Get user initials
    const initials = user?.display_name
        ? user.display_name.split(' ').map((n: string) => n[0]).join('').substring(0, 2).toUpperCase()
        : "AZ";

    return (
        <aside className="relative w-[280px] bg-sidebar-dark border-r border-slate-700/50 flex flex-col h-full shrink-0 overflow-hidden">
            {/* Logo */}
            <div className="p-6 flex items-center gap-3">
                <div className="size-10 rounded-xl bg-gradient-to-br from-primary to-accent-purple flex items-center justify-center shadow-lg shadow-primary/20">
                    <span className="text-white font-bold text-2xl">Q</span>
                </div>
                <div className="flex flex-col">
                    <h1 className="text-lg font-bold tracking-tight text-white">QUASAR</h1>
                    <span className="text-xs text-slate-400 font-medium">Research Assistant</span>
                </div>
            </div>

            {/* New Chat */}
            <div className="px-4 mb-6">
                <button onClick={() => { createNewConversation(); setActivePanel(null); }}
                    className="w-full flex items-center justify-center gap-2 bg-primary hover:bg-primary/90 text-primary-dark font-semibold py-3 px-4 rounded-full transition-colors shadow-lg shadow-primary/20 group">
                    <Plus className="w-5 h-5 transition-transform group-hover:rotate-90" />
                    <span>New Chat</span>
                </button>
            </div>

            {/* Conversation History */}
            <div className="flex-1 overflow-y-auto px-3 space-y-1">
                {conversations.length > 0 ? (
                    <>
                        <div className="px-3 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">Recent Research</div>
                        {conversations.map((conv) => {
                            const isActive = conv.id === activeConversationId;
                            return (
                                <div key={conv.id} className="relative group/item">
                                    <button onClick={() => handleSelectConversation(conv.id)}
                                        className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-all text-left ${isActive ? "bg-slate-700/50 text-white border-l-2 border-primary" : "text-slate-300 hover:bg-slate-800/50 group"}`}>
                                        {isActive ? <MessageSquare className="w-5 h-5 text-primary shrink-0" /> : <History className="w-5 h-5 text-slate-400 group-hover:text-primary transition-colors shrink-0" />}
                                        <div className="flex flex-col overflow-hidden flex-1">
                                            <span className="text-sm font-medium truncate">{conv.title}</span>
                                            <span className="text-[10px] text-slate-500">{timeAgo(conv.updatedAt)}</span>
                                        </div>
                                    </button>
                                    {/* Delete button — visible on hover */}
                                    <button
                                        onClick={(e) => handleDeleteConversation(e, conv.id)}
                                        title="Delete conversation"
                                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-all opacity-0 group-hover/item:opacity-100"
                                    >
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </button>
                                </div>
                            );
                        })}
                    </>
                ) : (
                    <div className="px-3 py-8 text-center">
                        <p className="text-sm text-slate-500">No conversations yet.</p>
                        <p className="text-xs text-slate-600 mt-1">Start a new chat to begin!</p>
                    </div>
                )}
            </div>

            {/* Bottom Controls */}
            <div className="p-4 border-t border-slate-700/50 space-y-3">
                {/* Model Dropdown */}
                <ModelDropdown
                    selectedModel={selectedModel}
                    availableModels={availableModels}
                    onSelect={setSelectedModel}
                />

                <div className="pt-2 border-t border-slate-700/50 space-y-1">
                    <button onClick={() => togglePanel("papers")}
                        className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-sm ${activePanel === "papers" ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-slate-800/50 hover:text-white"}`}>
                        <Bookmark className="w-4 h-4" />Saved Papers
                    </button>
                    <button onClick={() => setSettingsOpen(true)}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-sm text-slate-400 hover:bg-slate-800/50 hover:text-white">
                        <Settings className="w-4 h-4" />Settings
                    </button>
                    <Link href="/help"
                        className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-sm ${pathname === "/help" ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-slate-800/50 hover:text-white"}`}>
                        <HelpCircle className="w-4 h-4" />Help & Docs
                    </Link>
                    {isAuthenticated && (
                        <button onClick={logout}
                            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-sm text-red-500/70 hover:bg-red-500/10 hover:text-red-500">
                            <LogOut className="w-4 h-4" />Log Out
                        </button>
                    )}
                </div>

                <div className="pt-3 border-t border-slate-700/50 px-2 pb-2">
                    {isAuthenticated ? (
                        <div className="flex items-center gap-3">
                            <div className="size-8 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-400 flex items-center justify-center text-white text-xs font-bold shadow-md">
                                {initials}
                            </div>
                            <div className="flex flex-col flex-1 overflow-hidden">
                                <span className="text-sm font-semibold text-white truncate">{user?.display_name || "Alex Z."}</span>
                                <span className="text-[10px] text-slate-400 truncate">{user?.username || "Astronomer"}</span>
                            </div>
                        </div>
                    ) : (
                        <button
                            onClick={openAuthModal}
                            className="w-full bg-[#2a2a2a] hover:bg-[#333] border border-slate-700 text-white font-medium text-sm py-2.5 px-4 rounded-xl transition-all shadow-sm flex items-center justify-center gap-2"
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

            {/* Settings float modal — rendered outside sidebar via portal-like pattern */}
            <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
        </aside>
    );
}
