"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useChatStore } from "../lib/store";
import { useAuthStore } from "../lib/auth-store";
import {
    Plus, MessageSquare, History, Bookmark, Settings, HelpCircle,
    ChevronDown, Bot, X, ExternalLink, Github, BookOpen, Search,
    Telescope, FileText, Zap, Check, LogOut, User as UserIcon, Trash2, Cpu, Waves,
    Database, RefreshCw
} from "lucide-react";
import { SettingsModal } from "./SettingsModal";
import { isTaccModel } from "../lib/models";
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
    
    // Anthropic (current Messages API model IDs)
    "claude-opus-4-8": { in: 5.00, out: 25.00 },
    "claude-sonnet-5": { in: 3.00, out: 15.00 },
    "claude-haiku-4-5": { in: 1.00, out: 5.00 },
    "claude-fable-5": { in: 10.00, out: 50.00 },
    
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
    if (isTaccModel(model)) return null;
    if (MODEL_PRICING[model]) return MODEL_PRICING[model];
    // Fallbacks
    if (model.includes("deepseek-v4-flash")) return MODEL_PRICING["deepseek-v4-flash"];
    if (model.includes("deepseek-v4-pro") || model.includes("deepseek")) return MODEL_PRICING["deepseek-v4-pro"];
    if (model.includes("gpt-4o-mini")) return MODEL_PRICING["gpt-4o-mini"];
    if (model.includes("gpt-5.4-mini")) return MODEL_PRICING["gpt-5.4-mini"];
    if (model.includes("gpt-5.4")) return MODEL_PRICING["gpt-5.4"];
    if (model.includes("gpt-4.1") || model.includes("gpt-4o")) return MODEL_PRICING["gpt-4o"];
    if (model.includes("fable")) return MODEL_PRICING["claude-fable-5"];
    if (model.includes("sonnet")) return MODEL_PRICING["claude-sonnet-5"];
    if (model.includes("haiku")) return MODEL_PRICING["claude-haiku-4-5"];
    if (model.includes("opus")) return MODEL_PRICING["claude-opus-4-8"];
    if (model.includes("flash")) return MODEL_PRICING["gemini-1.5-flash"];
    if (model.includes("pro") && model.includes("gemini")) return MODEL_PRICING["gemini-1.5-pro"];
    return null;
}

function getModelTag(model: string) {
    if (isTaccModel(model)) return "US hosted";
    return null;
}

/* ────────────────────────────────────────────
   MODEL ICON COMPONENT
   ──────────────────────────────────────────── */
function ModelIcon({ model, className = "w-4 h-4" }: { model: string; className?: string }) {
    const isTacc = isTaccModel(model);
    const isDeepSeek = model.toLowerCase().includes("deepseek");
    const isOpenAI = !isTacc && (model.startsWith("gpt-") || model.startsWith("o1") || model.startsWith("o3") || model.startsWith("o4"));

    if (isTacc) {
        return <Cpu className={`${className} text-cyan-300 shrink-0`} />;
    }

    if (isDeepSeek) {
        return (
            <svg viewBox="0 0 512 509.64" className={`${className} fill-[#4D6BFE] shrink-0`} xmlns="http://www.w3.org/2000/svg">
                <path fillRule="nonzero" d="M440.898 139.167c-4.001-1.961-5.723 1.776-8.062 3.673-.801.612-1.479 1.407-2.154 2.141-5.848 6.246-12.681 10.349-21.607 9.859-13.048-.734-24.192 3.368-34.04 13.348-2.093-12.307-9.048-19.658-19.635-24.37-5.54-2.449-11.141-4.9-15.02-10.227-2.708-3.795-3.447-8.021-4.801-12.185-.861-2.509-1.725-5.082-4.618-5.512-3.139-.49-4.372 2.142-5.601 4.349-4.925 9.002-6.833 18.921-6.647 28.962.432 22.597 9.972 40.597 28.932 53.397 2.154 1.47 2.707 2.939 2.032 5.082-1.293 4.41-2.832 8.695-4.186 13.105-.862 2.817-2.157 3.429-5.172 2.205-10.402-4.346-19.391-10.778-27.332-18.553-13.481-13.044-25.668-27.434-40.873-38.702a177.614 177.614 0 00-10.834-7.409c-15.512-15.063 2.032-27.434 6.094-28.902 4.247-1.532 1.478-6.797-12.251-6.736-13.727.061-26.285 4.653-42.288 10.777-2.34.92-4.801 1.593-7.326 2.142-14.527-2.756-29.608-3.368-45.367-1.593-29.671 3.305-53.368 17.329-70.788 41.272-20.928 28.785-25.854 61.482-19.821 95.59 6.34 35.943 24.683 65.704 52.876 88.974 29.239 24.123 62.911 35.943 101.32 33.677 23.329-1.346 49.307-4.468 78.607-29.27 7.387 3.673 15.142 5.144 28.008 6.246 9.911.92 19.452-.49 26.839-2.019 11.573-2.449 10.773-13.166 6.586-15.124-33.915-15.797-26.47-9.368-33.24-14.573 17.235-20.39 43.213-41.577 53.369-110.222.8-5.448.121-8.877 0-13.287-.061-2.692.553-3.734 3.632-4.041 8.494-.981 16.742-3.305 24.314-7.471 21.975-12.002 30.84-31.719 32.933-55.355.307-3.612-.061-7.348-3.879-9.245v-.003zM249.4 351.89c-32.872-25.838-48.814-34.352-55.4-33.984-6.155.368-5.048 7.41-3.694 12.002 1.415 4.532 3.264 7.654 5.848 11.634 1.785 2.634 3.017 6.551-1.784 9.493-10.587 6.55-28.993-2.205-29.856-2.635-21.421-12.614-39.334-29.269-51.954-52.047-12.187-21.924-19.267-45.435-20.435-70.542-.308-6.061 1.478-8.207 7.509-9.307 7.94-1.471 16.127-1.778 24.068-.615 33.547 4.9 62.108 19.902 86.054 43.66 13.666 13.531 24.007 29.699 34.658 45.496 11.326 16.778 23.514 32.761 39.026 45.865 5.479 4.592 9.848 8.083 14.035 10.656-12.62 1.407-33.673 1.714-48.075-9.676zm15.899-102.519c.521-2.111 2.421-3.658 4.722-3.658a4.74 4.74 0 011.661.305c.678.246 1.293.614 1.786 1.163.861.859 1.354 2.083 1.354 3.368 0 2.695-2.154 4.837-4.862 4.837a4.748 4.748 0 01-4.738-4.034 5.01 5.01 0 01.077-1.981zm47.208 26.915c-2.606.996-5.2 1.778-7.707 1.88-4.679.244-9.787-1.654-12.556-3.981-4.308-3.612-7.386-5.631-8.679-11.941-.554-2.695-.247-6.858.246-9.246 1.108-5.144-.124-8.451-3.754-11.451-2.954-2.449-6.711-3.122-10.834-3.122-1.539 0-2.954-.673-4.001-1.224-1.724-.856-3.139-3-1.785-5.634.432-.856 2.525-2.939 3.018-3.305 5.6-3.185 12.065-2.144 18.034.244 5.54 2.266 9.727 6.429 15.759 12.307 6.155 7.102 7.263 9.063 10.773 14.39 2.771 4.163 5.294 8.451 7.018 13.348.877 2.561.071 4.74-2.341 6.277-.981.625-2.109 1.044-3.191 1.458z" />
            </svg>
        );
    }

    if (isOpenAI) {
        return (
            <svg viewBox="-1 -.1 949.1 959.8" className={`${className} text-white fill-current shrink-0`} xmlns="http://www.w3.org/2000/svg">
                <path d="m925.8 456.3c10.4 23.2 17 48 19.7 73.3 2.6 25.3 1.3 50.9-4.1 75.8-5.3 24.9-14.5 48.8-27.3 70.8-8.4 14.7-18.3 28.5-29.7 41.2-11.3 12.6-23.9 24-37.6 34-13.8 10-28.5 18.4-44.1 25.3-15.5 6.8-31.7 12-48.3 15.4-7.8 24.2-19.4 47.1-34.4 67.7-14.9 20.6-33 38.7-53.6 53.6-20.6 15-43.4 26.6-67.6 34.4-24.2 7.9-49.5 11.8-75 11.8-16.9.1-33.9-1.7-50.5-5.1-16.5-3.5-32.7-8.8-48.2-15.7s-30.2-15.5-43.9-25.5c-13.6-10-26.2-21.5-37.4-34.2-25 5.4-50.6 6.7-75.9 4.1-25.3-2.7-50.1-9.3-73.4-19.7-23.2-10.3-44.7-24.3-63.6-41.4s-35-37.1-47.7-59.1c-8.5-14.7-15.5-30.2-20.8-46.3s-8.8-32.7-10.6-49.6c-1.8-16.8-1.7-33.8.1-50.7 1.8-16.8 5.5-33.4 10.8-49.5-17-18.9-31-40.4-41.4-63.6-10.3-23.3-17-48-19.6-73.3-2.7-25.3-1.3-50.9 4-75.8s14.5-48.8 27.3-70.8c8.4-14.7 18.3-28.6 29.6-41.2s24-24 37.7-34 28.5-18.5 44-25.3c15.6-6.9 31.8-12 48.4-15.4 7.8-24.3 19.4-47.1 34.3-67.7 15-20.6 33.1-38.7 53.7-53.7 20.6-14.9 43.4-26.5 67.6-34.4 24.2-7.8 49.5-11.8 75-11.7 16.9-.1 33.9 1.6 50.5 5.1s32.8 8.7 48.3 15.6c15.5 7 30.2 15.5 43.9 25.5 13.7 10.1 26.3 21.5 37.5 34.2 24.9-5.3 50.5-6.6 75.8-4s50 9.3 73.3 19.6c23.2 10.4 44.7 24.3 63.6 41.4 18.9 17 35 36.9 47.7 59 8.5 14.6 15.5 30.1 20.8 46.3 5.3 16.1 8.9 32.7 10.6 49.6 1.8 16.9 1.8 33.9-.1 50.8-1.8 16.9-5.5 33.5-10.8 49.6 17.1 18.9 31 40.3 41.4 63.6zm-333.2 426.9c21.8-9 41.6-22.3 58.3-39s30-36.5 39-58.4c9-21.8 13.7-45.2 13.7-68.8v-223q-.1-.3-.2-.7-.1-.3-.3-.6-.2-.3-.5-.5-.3-.3-.6-.4l-80.7-46.6v269.4c0 2.7-.4 5.5-1.1 8.1-.7 2.7-1.7 5.2-3.1 7.6s-3 4.6-5 6.5a32.1 32.1 0 0 1 -6.5 5l-191.1 110.3c-1.6 1-4.3 2.4-5.7 3.2 7.9 6.7 16.5 12.6 25.5 17.8 9.1 5.2 18.5 9.6 28.3 13.2 9.8 3.5 19.9 6.2 30.1 8 10.3 1.8 20.7 2.7 31.1 2.7 23.6 0 47-4.7 68.8-13.8zm-455.1-151.4c11.9 20.5 27.6 38.3 46.3 52.7 18.8 14.4 40.1 24.9 62.9 31s46.6 7.7 70 4.6 45.9-10.7 66.4-22.5l193.2-111.5.5-.5q.2-.2.3-.6.2-.3.3-.6v-94l-233.2 134.9c-2.4 1.4-4.9 2.4-7.5 3.2-2.7.7-5.4 1-8.2 1-2.7 0-5.4-.3-8.1-1-2.6-.8-5.2-1.8-7.6-3.2l-191.1-110.4c-1.7-1-4.2-2.5-5.6-3.4-1.8 10.3-2.7 20.7-2.7 31.1s1 20.8 2.8 31.1c1.8 10.2 4.6 20.3 8.1 30.1 3.6 9.8 8 19.2 13.2 28.2zm-50.2-417c-11.8 20.5-19.4 43.1-22.5 66.5s-1.5 47.1 4.6 70c6.1 22.8 16.6 44.1 31 62.9 14.4 18.7 32.3 34.4 52.7 46.2l193.1 111.6q.3.1.7.2h.7q.4 0 .7-.2.3-.1.6-.3l81-46.8-233.2-134.6c-2.3-1.4-4.5-3.1-6.5-5a32.1 32.1 0 0 1 -5-6.5c-1.3-2.4-2.4-4.9-3.1-7.6-.7-2.6-1.1-5.3-1-8.1v-227.1c-9.8 3.6-19.3 8-28.3 13.2-9 5.3-17.5 11.3-25.5 18-7.9 6.7-15.3 14.1-22 22.1-6.7 7.9-12.6 16.5-17.8 25.5zm663.3 154.4c2.4 1.4 4.6 3 6.6 5 1.9 1.9 3.6 4.1 5 6.5 1.3 2.4 2.4 5 3.1 7.6.6 2.7 1 5.4.9 8.2v227.1c32.1-11.8 60.1-32.5 80.8-59.7 20.8-27.2 33.3-59.7 36.2-93.7s-3.9-68.2-19.7-98.5-39.9-55.5-69.5-72.5l-193.1-111.6q-.3-.1-.7-.2h-.7q-.3.1-.7.2-.3.1-.6.3l-80.6 46.6 233.2 134.7zm80.5-121h-.1v.1zm-.1-.1c5.8-33.6 1.9-68.2-11.3-99.7-13.1-31.5-35-58.6-63-78.2-28-19.5-61-30.7-95.1-32.2-34.2-1.4-68 6.9-97.6 23.9l-193.1 111.5q-.3.2-.5.5l-.4.6q-.1.3-.2.7-.1.3-.1.7v93.2l233.2-134.7c2.4-1.4 5-2.4 7.6-3.2 2.7-.7 5.4-1 8.1-1 2.8 0 5.5.3 8.2 1 2.6.8 5.1 1.8 7.5 3.2l191.1 110.4c1.7 1 4.2 2.4 5.6 3.3zm-505.3-103.2c0-2.7.4-5.4 1.1-8.1.7-2.6 1.7-5.2 3.1-7.6 1.4-2.3 3-4.5 5-6.5 1.9-1.9 4.1-3.6 6.5-4.9l191.1-110.3c1.8-1.1 4.3-2.5 5.7-3.2-26.2-21.9-58.2-35.9-92.1-40.2-33.9-4.4-68.3 1-99.2 15.5-31 14.5-57.2 37.6-75.5 66.4-18.3 28.9-28 62.3-28 96.5v223q.1.4.2.7.1.3.3.6.2.3.5.6.2.2.6.4l80.7 46.6zm43.8 294.7 103.9 60 103.9-60v-119.9l-103.8-60-103.9 60z" />
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

    const taccModels = availableModels.filter(isTaccModel);
    const openaiModels = availableModels.filter(m => !isTaccModel(m) && (m.startsWith("gpt-") || m.startsWith("o1") || m.startsWith("o3") || m.startsWith("o4")));
    const deepseekModels = availableModels.filter(m => !isTaccModel(m) && m.toLowerCase().includes("deepseek"));
    const geminiModels = availableModels.filter(m => !isTaccModel(m) && (m.startsWith("gemini-") || m.startsWith("gemma-")));
    const claudeModels = availableModels.filter(m => !isTaccModel(m) && m.startsWith("claude-"));
    const localModels = availableModels.filter(m => m.startsWith("local/"));

    const renderGroup = (label: string, models: string[]) => models.length === 0 ? null : (
        <>
            <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                {label}
            </div>
            {models.map((model) => {
                const cost = getModelCost(model);
                const tag = getModelTag(model);
                return (
                    <button key={model} onClick={() => { onSelect(model); setOpen(false); }}
                        className={`w-full flex items-center justify-between px-3 py-2 text-sm transition-colors ${model === selectedModel ? "bg-primary/15 text-primary border-l-2 border-primary" : "text-slate-300 hover:bg-slate-700/70 hover:text-white"}`}>
                        <div className="flex flex-col items-start truncate overflow-hidden pr-2">
                            <span className="font-medium truncate w-full text-left">{model.startsWith("local/") ? model.replace("local/", "") : model}</span>
                            {tag && (
                                <span className="mt-1 inline-flex items-center rounded border border-cyan-400/30 bg-cyan-400/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-cyan-200">
                                    {tag}
                                </span>
                            )}
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
    const currentTag = getModelTag(selectedModel);

    return (
        <div ref={ref} className="relative">
            <button onClick={() => setOpen(!open)}
                className="flex flex-col items-start w-full px-3 py-2 text-xs font-medium text-slate-300 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors border border-slate-700/50">
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
                {currentTag && (
                    <div className="mt-1 ml-6">
                        <span className="inline-flex items-center rounded border border-cyan-400/30 bg-cyan-400/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-cyan-200">
                            {currentTag}
                        </span>
                    </div>
                )}
            </button>

            {open && (
                <div className="absolute bottom-full left-0 right-0 mb-1.5 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl shadow-black/50 overflow-hidden z-50 animate-in fade-in slide-in-from-bottom-2 duration-150 max-h-72 overflow-y-auto custom-scrollbar">
                    {renderGroup("OpenAI", openaiModels)}
                    {deepseekModels.length > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("DeepSeek", deepseekModels)}
                    {taccModels.length > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("TACC Tejas", taccModels)}
                    {claudeModels.length > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                    {renderGroup("Anthropic Claude", claudeModels)}
                    {geminiModels.length > 0 && (openaiModels.length > 0 || deepseekModels.length > 0 || taccModels.length > 0 || claudeModels.length > 0) && <div className="border-t border-slate-700/50 mx-2" />}
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
        createNewConversation, selectedModel, availableModels, setSelectedModel,
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
                        href="/spectral-lines"
                        className={`sidebar-rail-button ${pathname === "/spectral-lines" ? "text-primary bg-primary/10" : ""}`}
                        title="Spectral Line Explorer"
                        aria-label="Spectral Line Explorer"
                    >
                        <Waves className="w-5 h-5" />
                    </Link>
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

                <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
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
                <Link href="/spectral-lines" onClick={onClose}
                    className={`w-full flex items-center gap-3 rounded-xl transition-colors ${isDrawer ? "px-3 py-2.5 text-[12.5px] font-medium" : "px-3 py-2.5 text-sm"} ${pathname === "/spectral-lines" ? "bg-indigo-400/15 text-indigo-300 border border-indigo-400/30" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}>
                    <Waves className={isDrawer ? "size-[15px]" : "w-4 h-4"} />
                    Spectral Line Explorer
                </Link>
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
                    availableModels={availableModels}
                    onSelect={setSelectedModel}
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
            <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
        </aside>
    );
}
