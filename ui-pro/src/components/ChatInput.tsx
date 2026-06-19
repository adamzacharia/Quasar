"use client";

import { useState, useRef, useEffect } from "react";
import { Send, PlusCircle, X, FileText, Image as ImageIcon, Square, ShieldCheck, Globe } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface AttachedFile {
    file: File;
    preview?: string; // base64 data URL for images
    type: "image" | "document";
}

interface ChatInputProps {
    onSend: (message: string, attachments?: AttachedFile[], options?: { groundedSummary?: boolean; webSearch?: boolean }) => void;
    onStop?: () => void;
    isStreaming: boolean;
    initialValue?: string;
}

export function ChatInput({ onSend, onStop, isStreaming, initialValue = "" }: ChatInputProps) {
    const [value, setValue] = useState(initialValue);
    const [attachments, setAttachments] = useState<AttachedFile[]>([]);
    const [hitCount, setHitCount] = useState<number | null>(null);
    const [groundedSummary, setGroundedSummary] = useState(false);
    const [webSearch, setWebSearch] = useState(true);
    const [menuOpen, setMenuOpen] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const imageInputRef = useRef<HTMLInputElement>(null);
    const documentInputRef = useRef<HTMLInputElement>(null);
    const menuRef = useRef<HTMLDivElement>(null);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { if (initialValue) { setValue(initialValue); inputRef.current?.focus(); } }, [initialValue]);

    // ── Fetch hit count on mount (with timeout to avoid slow renders) ──
    useEffect(() => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 4000);
        fetch(`${API_BASE}/api/analytics/hit`, { signal: controller.signal })
            .then(res => res.json())
            .then(data => {
                if (typeof data.hits === "number") setHitCount(data.hits);
            })
            .catch(() => {/* silent fail — timeout or network error */})
            .finally(() => clearTimeout(timeout));
        return () => { controller.abort(); clearTimeout(timeout); };
    }, []);

    useEffect(() => {
        if (!menuOpen) return;

        const handlePointerDown = (event: MouseEvent | TouchEvent) => {
            const target = event.target as Node | null;
            if (target && menuRef.current?.contains(target)) return;
            setMenuOpen(false);
        };

        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") setMenuOpen(false);
        };

        document.addEventListener("mousedown", handlePointerDown);
        document.addEventListener("touchstart", handlePointerDown);
        document.addEventListener("keydown", handleKeyDown);

        return () => {
            document.removeEventListener("mousedown", handlePointerDown);
            document.removeEventListener("touchstart", handlePointerDown);
            document.removeEventListener("keydown", handleKeyDown);
        };
    }, [menuOpen]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, selectedType?: "image" | "document") => {
        const files = Array.from(e.target.files || []);
        files.forEach(file => {
            const type = selectedType ?? (file.type.startsWith("image/") ? "image" : "document");
            if (type === "image") {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    setAttachments(prev => [...prev, { file, preview: ev.target?.result as string, type }]);
                };
                reader.readAsDataURL(file);
            } else {
                setAttachments(prev => [...prev, { file, type }]);
            }
        });
        // Reset so same file can be re-selected
        e.target.value = "";
    };

    const removeAttachment = (index: number) => {
        setAttachments(prev => prev.filter((_, i) => i !== index));
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const hasContent = value.trim() || attachments.length > 0;
        if (hasContent && !isStreaming) {
            onSend(value.trim(), attachments.length > 0 ? attachments : undefined, { groundedSummary, webSearch });
            setValue("");
            setAttachments([]);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as unknown as React.FormEvent);
        }
    };

    return (
        <div className="w-full px-4 md:px-8 pb-3 pt-2 z-20">
            <div className="w-full max-w-[var(--q-chat-input-width)] mx-auto relative">
                <form onSubmit={handleSubmit} className="relative group">
                    <div className="absolute inset-0 bg-primary/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    <div className="relative w-full glass-surface rounded-2xl ring-1 ring-white/10 focus-within:border-primary/50 focus-within:ring-primary/50 transition-all">

                        {/* Attachment previews */}
                        {attachments.length > 0 && (
                            <div className="flex flex-wrap gap-2 px-3 pt-3">
                                {attachments.map((att, i) => (
                                    <div key={i} className="relative group/att glass-control flex items-center gap-2 rounded-xl px-3 py-2 max-w-[200px]">
                                        {att.type === "image" && att.preview ? (
                                            <img src={att.preview} alt={att.file.name} className="w-8 h-8 rounded-lg object-cover shrink-0" />
                                        ) : (
                                            <FileText className="w-5 h-5 text-primary shrink-0" />
                                        )}
                                        <span className="text-xs text-slate-300 truncate max-w-[120px]">{att.file.name}</span>
                                        <button
                                            type="button"
                                            onClick={() => removeAttachment(i)}
                                            className="ml-1 p-0.5 rounded-full glass-control hover:bg-red-500/80 text-slate-400 hover:text-white transition-all opacity-0 group-hover/att:opacity-100"
                                        >
                                            <X className="w-3 h-3" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Input row */}
                        <div className="flex items-center gap-2 p-2">
                            {/* Hidden file inputs */}
                            <input
                                ref={imageInputRef}
                                type="file"
                                multiple
                                accept="image/*"
                                className="hidden"
                                onChange={(event) => handleFileChange(event, "image")}
                            />
                            <input
                                ref={documentInputRef}
                                type="file"
                                multiple
                                accept=".pdf,.txt,.csv,.md,.json,.fits,.fit,.fits.gz,.uvfits,.doc,.docx"
                                className="hidden"
                                onChange={(event) => handleFileChange(event, "document")}
                            />
                            <div ref={menuRef} className="relative shrink-0">
                                <button
                                    type="button"
                                    onClick={() => setMenuOpen(open => !open)}
                                    disabled={isStreaming}
                                    aria-haspopup="menu"
                                    aria-expanded={menuOpen}
                                    className="relative p-2.5 text-slate-400 hover:text-primary hover:bg-white/10 rounded-full transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                    title="Add files or modes"
                                >
                                    <PlusCircle className="w-5 h-5" />
                                    {groundedSummary && (
                                        <span className={`absolute ${webSearch ? "right-3.5" : "right-1.5"} top-1.5 h-2 w-2 rounded-full bg-emerald-400 ring-2 ring-card-dark`} aria-hidden="true" />
                                    )}
                                    {webSearch && (
                                        <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-cyan-400 ring-2 ring-card-dark" aria-hidden="true" />
                                    )}
                                </button>

                                {menuOpen && (
                                    <div
                                        role="menu"
                                        className="glass-popover absolute bottom-full left-0 z-30 mb-2 w-56 max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl p-1.5"
                                    >
                                        <button
                                            type="button"
                                            role="menuitem"
                                            onClick={() => {
                                                setMenuOpen(false);
                                                imageInputRef.current?.click();
                                            }}
                                            className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm text-slate-200 transition-colors hover:bg-white/10"
                                        >
                                            <ImageIcon className="h-4 w-4 text-primary" />
                                            <span>Upload images</span>
                                        </button>
                                        <button
                                            type="button"
                                            role="menuitem"
                                            onClick={() => {
                                                setMenuOpen(false);
                                                documentInputRef.current?.click();
                                            }}
                                            className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm text-slate-200 transition-colors hover:bg-white/10"
                                        >
                                            <FileText className="h-4 w-4 text-primary" />
                                            <span>Upload documents</span>
                                        </button>
                                        <div className="my-1 h-px bg-white/10" />
                                        <div className="group/grounded relative">
                                            <button
                                                type="button"
                                                role="menuitemcheckbox"
                                                aria-checked={groundedSummary}
                                                aria-describedby="grounded-mode-tooltip"
                                                onClick={() => setGroundedSummary(value => !value)}
                                                className="flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2 text-left text-sm text-slate-200 transition-colors hover:bg-white/10 focus-visible:bg-white/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-emerald-400/50"
                                            >
                                                <span className="flex min-w-0 items-center gap-3">
                                                    <ShieldCheck className={`h-4 w-4 ${groundedSummary ? "text-emerald-300" : "text-slate-500"}`} />
                                                    <span>Grounded</span>
                                                </span>
                                                <span className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors ${
                                                    groundedSummary
                                                        ? "border-emerald-400/50 bg-emerald-500/25"
                                                        : "border-slate-700 bg-slate-900"
                                                }`}>
                                                    <span className={`h-3.5 w-3.5 rounded-full transition-transform ${
                                                        groundedSummary
                                                            ? "translate-x-4 bg-emerald-300"
                                                            : "translate-x-1 bg-slate-500"
                                                    }`} />
                                                </span>
                                            </button>
                                            <div
                                                id="grounded-mode-tooltip"
                                                role="tooltip"
                                                className="max-h-0 overflow-hidden px-3 text-[11px] leading-relaxed text-slate-400 opacity-0 transition-all duration-150 group-hover/grounded:mb-1 group-hover/grounded:max-h-28 group-hover/grounded:opacity-100 group-focus-within/grounded:mb-1 group-focus-within/grounded:max-h-28 group-focus-within/grounded:opacity-100"
                                            >
                                                <span className="block font-semibold text-emerald-200">Grounded mode</span>
                                                Summarizes only rows, counts, identifiers, coordinates, links, and explicit errors returned by tools in this run. It avoids outside background knowledge, guesses, and unstated counts.
                                            </div>
                                        </div>
                                        <button
                                            type="button"
                                            role="menuitemcheckbox"
                                            aria-checked={webSearch}
                                            onClick={() => setWebSearch(value => !value)}
                                            className="flex w-full items-center justify-between gap-3 rounded-lg px-3 py-2 text-left text-sm text-slate-200 transition-colors hover:bg-white/10"
                                        >
                                            <span className="flex min-w-0 items-center gap-3">
                                                <Globe className={`h-4 w-4 ${webSearch ? "text-cyan-300" : "text-slate-500"}`} />
                                                <span>Web Search</span>
                                            </span>
                                            <span className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors ${
                                                webSearch
                                                    ? "border-cyan-400/50 bg-cyan-500/25"
                                                    : "border-slate-700 bg-slate-900"
                                            }`}>
                                                <span className={`h-3.5 w-3.5 rounded-full transition-transform ${
                                                    webSearch
                                                        ? "translate-x-4 bg-cyan-300"
                                                        : "translate-x-1 bg-slate-500"
                                                }`} />
                                            </span>
                                        </button>
                                    </div>
                                )}
                            </div>
                            <input
                                ref={inputRef}
                                type="text"
                                value={value}
                                onChange={(e) => setValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={isStreaming ? "QUASAR is thinking..." : "Ask QUASAR about observations, data, or literature..."}
                                className="flex-1 bg-transparent border-none outline-none text-white placeholder-slate-500 focus:ring-0 text-sm"
                                disabled={isStreaming}
                            />
                            {/* Keyboard shortcut hint */}
                            {!isStreaming && value.trim() && (
                                <span className="text-[10px] text-slate-500 font-mono mr-1 select-none hidden sm:inline">⏎</span>
                            )}
                            {isStreaming ? (
                                <button
                                    type="button"
                                    onClick={onStop}
                                    className="p-2.5 bg-red-500/80 hover:bg-red-500 text-white rounded-full transition-all shadow-lg shadow-red-500/20 flex items-center justify-center animate-pulse"
                                    title="Stop generating"
                                >
                                    <Square className="w-4 h-4 fill-current" />
                                </button>
                            ) : (
                                <button
                                    type="submit"
                                    disabled={!value.trim() && attachments.length === 0}
                                    className="p-2.5 bg-primary hover:bg-primary/90 text-white rounded-full transition-all shadow-lg shadow-primary/20 flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                    <Send className="w-5 h-5" />
                                </button>
                            )}
                        </div>
                    </div>
                </form>
                <div className="text-center mt-2">
                    <p className="text-[10px] text-slate-600">
                        QUASAR may produce inaccurate information.
                        <span className="text-slate-700 ml-2">· Accepts images, PDFs, FITS, CSV</span>
                        <span className={`text-slate-700 ml-2 transition-opacity duration-500 ${hitCount !== null ? 'opacity-100' : 'opacity-0'}`}>· Visits: <span className="text-primary/50 font-mono tabular-nums">{hitCount !== null ? hitCount.toLocaleString() : '—'}</span></span>
                    </p>
                </div>
            </div>
        </div>
    );
}
