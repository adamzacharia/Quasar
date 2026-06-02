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

const MODEL_PRICING: Record<string, { in: number, out: number }> = {
    // GPT-5 Series
    "gpt-5.4": { in: 15.00, out: 45.00 },
    "gpt-5.4-mini": { in: 0.75, out: 4.50 },
    "gpt-5.4-2026-03-05": { in: 15.00, out: 45.00 },
    "gpt-5": { in: 10.00, out: 30.00 },
    "gpt-5-mini": { in: 0.50, out: 1.50 },
    "gpt-5-nano": { in: 0.10, out: 0.30 },
    
    // GPT-4 Series
    "gpt-4.1": { in: 2.50, out: 10.00 },
    "gpt-4o": { in: 2.50, out: 10.00 },
    "gpt-4o-mini": { in: 0.15, out: 0.60 },
    
    // Anthropic
    "claude-3-7-sonnet": { in: 3.00, out: 15.00 },
    "claude-3-5-sonnet-20241022": { in: 3.00, out: 15.00 },
    "claude-3-5-haiku-20241022": { in: 0.25, out: 1.25 },
    "claude-3-opus-20240229": { in: 15.00, out: 75.00 },
    
    // Google Gemini
    "gemini-2.5-pro": { in: 1.25, out: 5.00 },
    "gemini-1.5-pro": { in: 1.25, out: 5.00 },
    "gemini-1.5-flash": { in: 0.075, out: 0.30 },

    // DeepSeek
    "deepseek-v4-pro": { in: 0.435, out: 0.87 },
    "deepseek-v4-flash": { in: 0.14, out: 0.28 },
};

function getModelCost(model: string) {
    if (model.startsWith("local/")) return { in: 0, out: 0 };
    if (MODEL_PRICING[model]) return MODEL_PRICING[model];
    // Fallbacks
    if (model.includes("deepseek-v4-flash")) return MODEL_PRICING["deepseek-v4-flash"];
    if (model.includes("deepseek-v4-pro") || model.includes("deepseek")) return MODEL_PRICING["deepseek-v4-pro"];
    if (model.includes("gpt-4o-mini")) return MODEL_PRICING["gpt-4o-mini"];
    if (model.includes("gpt-5.4-mini")) return MODEL_PRICING["gpt-5.4-mini"];
    if (model.includes("gpt-5.4")) return MODEL_PRICING["gpt-5.4"];
    if (model.includes("gpt-4.1") || model.includes("gpt-4o")) return MODEL_PRICING["gpt-4o"];
    if (model.includes("sonnet")) return MODEL_PRICING["claude-3-5-sonnet-20241022"];
    if (model.includes("haiku")) return MODEL_PRICING["claude-3-5-haiku-20241022"];
    if (model.includes("opus")) return MODEL_PRICING["claude-3-opus-20240229"];
    if (model.includes("flash")) return MODEL_PRICING["gemini-1.5-flash"];
    if (model.includes("pro") && model.includes("gemini")) return MODEL_PRICING["gemini-1.5-pro"];
    return null;
}

/* ────────────────────────────────────────────
   MODEL ICON COMPONENT
   ──────────────────────────────────────────── */
function ModelIcon({ model, className = "w-4 h-4" }: { model: string; className?: string }) {
    const isDeepSeek = model.toLowerCase().includes("deepseek");
    const isOpenAI = model.startsWith("gpt-") || model.startsWith("o1") || model.startsWith("o3") || model.startsWith("o4");

    if (isDeepSeek) {
        return (
            <svg viewBox="0 0 512 509.64" className={`${className} fill-[#4D6BFE] shrink-0`} xmlns="http://www.w3.org/2000/svg">
                <path fillRule="nonzero" d="M440.898 139.167c-4.001-1.961-5.723 1.776-8.062 3.673-.801.612-1.479 1.407-2.154 2.141-5.848 6.246-12.681 10.349-21.607 9.859-13.048-.734-24.192 3.368-34.04 13.348-2.093-12.307-9.048-19.658-19.635-24.37-5.54-2.449-11.141-4.9-15.02-10.227-2.708-3.795-3.447-8.021-4.801-12.185-.861-2.509-1.725-5.082-4.618-5.512-3.139-.49-4.372 2.142-5.601 4.349-4.925 9.002-6.833 18.921-6.647 28.962.432 22.597 9.972 40.597 28.932 53.397 2.154 1.47 2.707 2.939 2.032 5.082-1.293 4.41-2.832 8.695-4.186 13.105-.862 2.817-2.157 3.429-5.172 2.205-10.402-4.346-19.391-10.778-27.332-18.553-13.481-13.044-25.668-27.434-40.873-38.702a177.614 177.614 0 00-10.834-7.409c-15.512-15.063 2.032-27.434 6.094-28.902 4.247-1.532 1.478-6.797-12.251-6.736-13.727.061-26.285 4.653-42.288 10.777-2.34.92-4.801 1.593-7.326 2.142-14.527-2.756-29.608-3.368-45.367-1.593-29.671 3.305-53.368 17.329-70.788 41.272-20.928 28.785-25.854 61.482-19.821 95.59 6.34 35.943 24.683 65.704 52.876 88.974 29.239 24.123 62.911 35.943 101.32 33.677 23.329-1.346 49.307-4.468 78.607-29.27 7.387 3.673 15.142 5.144 28.008 6.246 9.911.92 19.452-.49 26.839-2.019 11.573-2.449 10.773-13.166 6.586-15.124-33.915-15.797-26.47-9.368-33.24-14.573 17.235-20.39 43.213-41.577 53.369-110.222.8-5.448.121-8.877 0-13.287-.061-2.692.553-3.734 3.632-4.041 8.494-.981 16.742-3.305 24.314-7.471 21.975-12.002 30.84-31.719 32.933-55.355.307-3.612-.061-7.348-3.879-9.245v-.003zM249.4 351.89c-32.872-25.838-48.814-34.352-55.4-33.984-6.155.368-5.048 7.41-3.694 12.002 1.415 4.532 3.264 7.654 5.848 11.634 1.785 2.634 3.017 6.551-1.784 9.493-10.587 6.55-28.993-2.205-29.856-2.635-21.421-12.614-39.334-29.269-51.954-52.047-12.187-21.924-19.267-45.435-20.435-70.542-.308-6.061 1.478-8.207 7.509-9.307 7.94-1.471 16.127-1.778 24.068-.615 33.547 4.9 62.108 19.902 86.054 43.66 13.666 13.531 24.007 29.699 34.658 45.496 11.326 16.778 23.514 32.761 39.026 45.865 5.479 4.592 9.848 8.083 14.035 10.656-12.62 1.407-33.673 1.714-48.075-9.676zm15.899-102.519c.521-2.111 2.421-3.658 4.722-3.658a4.74 4.74 0 011.661.305c.678.246 1.293.614 1.786 1.163.861.859 1.354 2.083 1.354 3.368 0 2.695-2.154 4.837-4.862 4.837a4.748 4.748 0 01-4.738-4.034 5.01 5.01 0 01.077-1.981zm47.208 26.915c-2.606.996-5.2 1.778-7.707 1.88-4.679.244-9.787-1.654-12.556-3.981-4.308-3.612-7.386-5.631-8.679-11.941-.554-2.695-.247-6.858.246-9.246 1.108-5.144-.124-8.451-3.754-11.451-2.954-2.449-6.711-3.122-10.834-3.122-1.539 0-2.954-.673-4.001-1.224-1.724-.856-3.139-3-1.785-5.634.432-.856 2.525-2.939 3.018-3.305 5.6-3.185 12.065-2.144 18.034.244 5.54 2.266 9.727 6.429 15.759 12.307 6.155 7.102 7.263 9.063 10.773 14.39 2.771 4.163 5.294 8.451 7.018 13.348.877 2.561.071 4.74-2.341 6.277-.981.625-2.109 1.044-3.191 1.458z" />
            </svg>
        );
    }

    if (isOpenAI) {
        return (
            <svg viewBox="0 0 16 16" className={`${className} text-[#10a37f] fill-current shrink-0`} xmlns="http://www.w3.org/2000/svg">
                <path d="M14.949 6.547a3.94 3.94 0 0 0-.348-3.273 4.11 4.11 0 0 0-4.4-1.934A4.1 4.1 0 0 0 8.423.2 4.15 4.15 0 0 0 6.305.086a4.1 4.1 0 0 0-1.891.948 4.04 4.04 0 0 0-1.158 1.753 4.1 4.1 0 0 0-1.563.679A4 4 0 0 0 .554 4.72a3.99 3.99 0 0 0 .502 4.731 3.94 3.94 0 0 0 .346 3.274 4.11 4.11 0 0 0 4.402 1.933c.382.425.852.764 1.377.995.526.231 1.095.35 1.67.346 1.78.002 3.358-1.132 3.901-2.804a4.1 4.1 0 0 0 1.563-.68 4 4 0 0 0 1.14-1.253 3.99 3.99 0 0 0-.506-4.716m-6.097 8.406a3.05 3.05 0 0 1-1.945-.694l.096-.054 3.23-1.838a.53.53 0 0 0 .265-.455v-4.49l1.366.778q.02.011.025.035v3.722c-.003 1.653-1.361 2.992-3.037 2.996m-6.53-2.75a2.95 2.95 0 0 1-.36-2.01l.095.057L5.29 12.09a.53.53 0 0 0 .527 0l3.949-2.246v1.555c-.006.27-.04.538-.1.8a3 3 0 0 1-3.002 2.378M.858 3.69a2.95 2.95 0 0 1 1.585-1.261l.003.111.096 3.682a.53.53 0 0 0 .263.454l3.943 2.248V7.37L2.802 5.122a2.99 2.99 0 0 1-.502-1.37q-.004-.03-.004-.061M8 12.24l-3.23-1.84a.53.53 0 0 0-.526 0l-3.95 2.25v-.111q.003-.79.333-1.506a3 3 0 0 1 2.704-1.78l.095.054 3.23 1.837c.16.09.358.09.518 0l3.949-2.25v1.555a3 3 0 0 1-3.128 1.851m6.097-6.035a3 3 0 0 1-.32 1.472l-.096-.054-3.23-1.839a.53.53 0 0 0-.518 0L6.034 8.033v-1.55l3.949-2.246a3 3 0 0 1 3.122 1.854l-.01.037zm-.424-2.515a2.95 2.95 0 0 1 .36 2.011l-.093-.056-3.23-1.838a.53.53 0 0 0-.526 0L6.18 5.753V4.198c.006-.27.04-.539.1-.8a3 3 0 0 1 3.003-2.379m-3.037 2.115v4.49l-1.366-.778A.53.53 0 0 0 6 7.502V3.78a3 3 0 0 1 3.037-2.996 3 3 0 0 1 1.945.694l-.096.054-3.23 1.838a.53.53 0 0 0-.265.455"/>
            </svg>
        );
    }

    return <Bot className={className} />;
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
    const deepseekModels = availableModels.filter(m => m.toLowerCase().includes("deepseek"));
    const geminiModels = availableModels.filter(m => m.startsWith("gemini-") || m.startsWith("gemma-"));
    const claudeModels = availableModels.filter(m => m.startsWith("claude-"));
    const localModels = availableModels.filter(m => m.startsWith("local/"));

    const renderGroup = (label: string, models: string[]) => models.length === 0 ? null : (
        <>
            <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                {label}
            </div>
            {models.map((model) => {
                const cost = getModelCost(model);
                return (
                    <button key={model} onClick={() => { onSelect(model); setOpen(false); }}
                        className={`w-full flex items-center justify-between px-3 py-2 text-sm transition-colors ${model === selectedModel ? "bg-primary/10 text-primary" : "text-slate-300 hover:bg-slate-700/50 hover:text-white"}`}>
                        <div className="flex flex-col items-start truncate overflow-hidden pr-2">
                            <span className="font-medium truncate w-full text-left">{model.startsWith("local/") ? model.replace("local/", "") : model}</span>
                            {cost && (
                                <span className="text-[9px] text-slate-500 font-mono mt-0.5">
                                    In: ${cost.in}/M · Out: ${cost.out}/M
                                </span>
                            )}
                        </div>
                        {model === selectedModel && <Check className="w-4 h-4 text-primary shrink-0" />}
                    </button>
                );
            })}
        </>
    );

    const currentCost = getModelCost(selectedModel);

    return (
        <div ref={ref} className="relative">
            <button onClick={() => setOpen(!open)}
                className="flex flex-col items-start w-full px-3 py-2 text-xs font-medium text-slate-300 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors">
                <div className="flex items-center justify-between w-full">
                    <div className="flex items-center gap-2"><ModelIcon model={selectedModel} className="w-4 h-4 shrink-0" /><span className="truncate">Model: {selectedModel.startsWith("local/") ? selectedModel.replace("local/", "[Local] ") : selectedModel}</span></div>
                    <ChevronDown className={`w-4 h-4 shrink-0 transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
                </div>
                {currentCost && (
                    <div className="flex items-center gap-2 mt-1 ml-6 text-[10px] text-slate-400/80 font-mono">
                        <span>In: ${currentCost.in}/M</span>
                        <span className="text-slate-600">|</span>
                        <span>Out: ${currentCost.out}/M</span>
                    </div>
                )}
            </button>

            {open && (
                <div className="absolute bottom-full left-0 right-0 mb-1.5 bg-slate-800 border border-slate-600/50 rounded-xl shadow-2xl shadow-black/40 overflow-hidden z-50 animate-in fade-in slide-in-from-bottom-2 duration-150 max-h-72 overflow-y-auto custom-scrollbar">
                    {renderGroup("OpenAI", openaiModels)}
                    {deepseekModels.length > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("DeepSeek", deepseekModels)}
                    {claudeModels.length > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("Anthropic Claude", claudeModels)}
                    {geminiModels.length > 0 && (openaiModels.length > 0 || deepseekModels.length > 0 || claudeModels.length > 0) && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("Google Gemini", geminiModels)}
                    {localModels.length > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("Local LLM", localModels)}
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

    // Get user initials from display name, or fallback to username initials
    const initials = user?.display_name
        ? user.display_name.split(' ').filter(Boolean).map((n: string) => n[0]).join('').substring(0, 2).toUpperCase()
        : user?.username
            ? user.username.substring(0, 2).toUpperCase()
            : "U";

    return (
        <aside className="relative w-[280px] bg-sidebar-dark border-r border-slate-700/50 flex flex-col h-full shrink-0 overflow-hidden">
            {/* Logo */}
            <div className="p-6 flex items-center gap-3">
                <img src="/quasar_logo.png" alt="Quasar" className="size-[60px] object-contain" />
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
                            {user?.picture_url ? (
                                <img src={user.picture_url} alt={user.display_name || "User"} className="size-8 rounded-full object-cover shadow-md" referrerPolicy="no-referrer" />
                            ) : (
                                <div className="size-8 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-400 flex items-center justify-center text-white text-xs font-bold shadow-md">
                                    {initials}
                                </div>
                            )}
                            <div className="flex flex-col flex-1 overflow-hidden">
                                <span className="text-sm font-semibold text-white truncate">{user?.display_name || user?.username || "User"}</span>
                                <span className="text-[10px] text-slate-400 truncate">{user?.username || ""}</span>
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
