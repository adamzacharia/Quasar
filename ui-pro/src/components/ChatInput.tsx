"use client";

import { useState, useRef, useEffect } from "react";
import { ArrowUp, ChevronDown, Paperclip, WandSparkles, X, FileText, Image as ImageIcon, Square, ShieldCheck, Globe } from "lucide-react";
import type { WebSearchMode } from "../lib/types";

interface AttachedFile {
    file: File;
    preview?: string; // base64 data URL for images
    type: "image" | "document";
}

interface ChatInputProps {
    onSend: (message: string, attachments?: AttachedFile[], options?: { groundedSummary?: boolean; webSearch?: boolean; webSearchMode?: WebSearchMode }) => void;
    onStop?: () => void;
    isStreaming: boolean;
    initialValue?: string;
    /** "hero" = centered card on the empty landing; "docked" = pinned at the bottom during a chat. */
    variant?: "hero" | "docked";
    /* Composer options + analytics are owned by the parent so they survive the
       hero→docked switch (and the visit counter is fetched only once). */
    grounded: boolean;
    onGroundedChange: (v: boolean) => void;
    /** Web search mode (Phase 2): off | auto | always. Persisted by the parent. */
    webSearchMode: WebSearchMode;
    onWebSearchModeChange: (mode: WebSearchMode) => void;
    hitCount?: number | null;
    /** The selected model generates images: the message is an image prompt. */
    imageModel?: boolean;
}

const WEB_SEARCH_MODE_OPTIONS: { value: WebSearchMode; label: string; hint: string }[] = [
    { value: "off", label: "Off", hint: "Never search the web" },
    { value: "auto", label: "Auto", hint: "Search when the question needs current information" },
    { value: "always", label: "Always", hint: "Search the web on every message" },
];

type OpenMenu = "sources" | "attach" | null;

const ICON = "h-4 w-4 shrink-0";
const MENU_ROW =
    "flex w-full items-center gap-2.5 rounded-xl px-2.5 py-2 text-left text-[13px] text-[var(--q-text)] transition-colors hover:bg-[var(--q-glass-control-hover)] focus-visible:bg-[var(--q-glass-control-hover)] focus-visible:outline-none";

export function ChatInput({
    onSend, onStop, isStreaming, initialValue = "", variant = "docked",
    grounded: groundedSummary, onGroundedChange, webSearchMode, onWebSearchModeChange, hitCount = null,
    imageModel = false,
}: ChatInputProps) {
    const isHero = variant === "hero";
    const webSearch = webSearchMode !== "off";
    const [value, setValue] = useState(initialValue);
    const [attachments, setAttachments] = useState<AttachedFile[]>([]);
    const [openMenu, setOpenMenu] = useState<OpenMenu>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const imageInputRef = useRef<HTMLInputElement>(null);
    const documentInputRef = useRef<HTMLInputElement>(null);
    const sourcesRef = useRef<HTMLDivElement>(null);
    const attachRef = useRef<HTMLDivElement>(null);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { if (initialValue) { setValue(initialValue); inputRef.current?.focus(); } }, [initialValue]);

    useEffect(() => {
        if (!openMenu) return;
        const menuRef = openMenu === "sources" ? sourcesRef : attachRef;

        const handlePointerDown = (event: MouseEvent | TouchEvent) => {
            const target = event.target as Node | null;
            if (target && menuRef.current?.contains(target)) return;
            setOpenMenu(null);
        };

        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") setOpenMenu(null);
        };

        document.addEventListener("mousedown", handlePointerDown);
        document.addEventListener("touchstart", handlePointerDown);
        document.addEventListener("keydown", handleKeyDown);

        return () => {
            document.removeEventListener("mousedown", handlePointerDown);
            document.removeEventListener("touchstart", handlePointerDown);
            document.removeEventListener("keydown", handleKeyDown);
        };
    }, [openMenu]);

    const toggleMenu = (menu: Exclude<OpenMenu, null>) => setOpenMenu(open => (open === menu ? null : menu));

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
            onSend(value.trim(), attachments.length > 0 ? attachments : undefined, { groundedSummary, webSearch, webSearchMode });
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

    const webLabel = WEB_SEARCH_MODE_OPTIONS.find(o => o.value === webSearchMode)?.label ?? "Off";
    const sourcesOpen = openMenu === "sources";
    const attachOpen = openMenu === "attach";

    return (
        <div className={isHero ? "w-full z-20" : "w-full px-4 md:px-8 pb-3 pt-2 z-20"}>
            {/* --q-chat-input-width carries an 8vw inset meant to keep the composer
                off the edges of a wide monitor. On a phone that inset stacks on top
                of the px-4 gutter and squeezes the dock narrower than the content
                above it, so it only applies from md up. */}
            <div className={`w-full mx-auto relative ${isHero ? "max-w-[var(--q-suggestion-grid-width)]" : "max-w-none md:max-w-[var(--q-chat-input-width)]"}`}>
                <form onSubmit={handleSubmit} className="relative">
                    <div className="relative w-full rounded-[22px] border border-[var(--q-border)] bg-[var(--q-card)] shadow-[0_1px_2px_var(--q-shadow-glow),0_8px_24px_-12px_var(--q-shadow-glow)] transition-colors focus-within:border-[var(--q-border-strong)]">

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

                        {/* Attachment previews */}
                        {attachments.length > 0 && (
                            <div className={`flex flex-wrap gap-2 pt-3 ${isHero ? "px-5" : "px-4"}`}>
                                {attachments.map((att, i) => (
                                    <div key={i} className="group/att relative flex max-w-[200px] items-center gap-2 rounded-xl border border-[var(--q-border)] bg-[var(--q-card)] py-1.5 pl-1.5 pr-2">
                                        {att.type === "image" && att.preview ? (
                                            <img src={att.preview} alt={att.file.name} className="h-7 w-7 shrink-0 rounded-lg object-cover" />
                                        ) : (
                                            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[var(--q-canvas)]">
                                                <FileText className="h-3.5 w-3.5 text-[var(--q-text-muted)]" strokeWidth={1.75} />
                                            </span>
                                        )}
                                        <span className="max-w-[120px] truncate text-[12px] text-[var(--q-text-secondary)]">{att.file.name}</span>
                                        <button
                                            type="button"
                                            onClick={() => removeAttachment(i)}
                                            aria-label={`Remove ${att.file.name}`}
                                            className="rounded-full p-0.5 text-[var(--q-text-faint)] opacity-0 transition-all hover:bg-[var(--q-glass-control-hover)] hover:text-[var(--q-text)] focus-visible:opacity-100 group-hover/att:opacity-100"
                                        >
                                            <X className="h-3 w-3" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Row 1: prompt */}
                        <div className={`flex items-center gap-2.5 ${isHero ? "px-5 pt-4 pb-1.5" : "px-4 pt-3.5 pb-1"}`}>
                            <WandSparkles className="h-4 w-4 shrink-0 text-[var(--q-text-faint)]" strokeWidth={1.75} aria-hidden="true" />
                            <input
                                ref={inputRef}
                                type="text"
                                value={value}
                                onChange={(e) => setValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={isStreaming
                                    ? (imageModel ? "Generating image…" : "Quasar is thinking…")
                                    : imageModel
                                      ? "Describe the image to generate, or attach one to edit…"
                                      : "Ask Quasar about observations, data, or literature…"}
                                className="min-w-0 flex-1 border-none bg-transparent text-[14px] text-[var(--q-text)] outline-none placeholder:text-[var(--q-text-faint)] focus:ring-0"
                                disabled={isStreaming}
                            />
                            {/* Keyboard shortcut hint */}
                            {!isStreaming && value.trim() && (
                                <span className="hidden select-none font-mono text-[11px] text-[var(--q-text-faint)] sm:inline">⏎</span>
                            )}
                        </div>

                        {/* Row 2: sources · attach · send */}
                        <div className={`flex items-center gap-2 ${isHero ? "px-4 pb-3.5 pt-2" : "px-3 pb-3 pt-2"}`}>
                            <div ref={sourcesRef} className="relative shrink-0">
                                <button
                                    type="button"
                                    onClick={() => toggleMenu("sources")}
                                    disabled={isStreaming}
                                    aria-haspopup="menu"
                                    aria-expanded={sourcesOpen}
                                    className="q-pill h-8 gap-1.5 px-3 text-[13px]"
                                    title="Sources: grounded mode and web search"
                                >
                                    {groundedSummary && (
                                        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" aria-hidden="true" />
                                    )}
                                    {webSearch && (
                                        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-500" aria-hidden="true" />
                                    )}
                                    <span className="whitespace-nowrap">Web: {webLabel}</span>
                                    <ChevronDown className={`h-3.5 w-3.5 text-[var(--q-text-muted)] transition-transform ${sourcesOpen ? "rotate-180" : ""}`} strokeWidth={1.75} />
                                </button>

                                {sourcesOpen && (
                                    <div
                                        role="menu"
                                        className="glass-popover absolute bottom-full left-0 z-30 mb-2 w-64 max-w-[calc(100vw-2rem)] overflow-hidden rounded-2xl p-1.5"
                                    >
                                        <div className="group/grounded relative">
                                            <button
                                                type="button"
                                                role="menuitemcheckbox"
                                                aria-checked={groundedSummary}
                                                aria-describedby="grounded-mode-tooltip"
                                                onClick={() => onGroundedChange(!groundedSummary)}
                                                className={`${MENU_ROW} justify-between`}
                                            >
                                                <span className="flex min-w-0 items-center gap-2.5">
                                                    <ShieldCheck className={`${ICON} ${groundedSummary ? "text-emerald-300" : "text-[var(--q-text-muted)]"}`} strokeWidth={1.75} />
                                                    <span>Grounded</span>
                                                </span>
                                                <span className={`relative inline-flex h-[18px] w-8 shrink-0 items-center rounded-full border transition-colors ${
                                                    groundedSummary
                                                        ? "border-emerald-500/40 bg-emerald-500/15"
                                                        : "border-[var(--q-border)] bg-[var(--q-canvas)]"
                                                }`}>
                                                    <span className={`h-3 w-3 rounded-full transition-transform ${
                                                        groundedSummary
                                                            ? "translate-x-[15px] bg-emerald-500"
                                                            : "translate-x-[2px] bg-[var(--q-border-strong)]"
                                                    }`} />
                                                </span>
                                            </button>
                                            <div
                                                id="grounded-mode-tooltip"
                                                role="tooltip"
                                                className="max-h-0 overflow-hidden px-2.5 text-[12px] leading-relaxed text-[var(--q-text-muted)] opacity-0 transition-all duration-150 group-hover/grounded:mb-1 group-hover/grounded:max-h-28 group-hover/grounded:opacity-100 group-focus-within/grounded:mb-1 group-focus-within/grounded:max-h-28 group-focus-within/grounded:opacity-100"
                                            >
                                                <span className="block font-medium text-[var(--q-text)]">Grounded mode</span>
                                                Summarizes only rows, counts, identifiers, coordinates, links, and explicit errors returned by tools in this run. It avoids outside background knowledge, guesses, and unstated counts.
                                            </div>
                                        </div>
                                        <div className="mx-2 my-1 h-px bg-[var(--q-border)]" />
                                        {/* Web search mode (Phase 2): a segmented control instead of the
                                            on/off switch. Persisted by ChatArea in localStorage. */}
                                        <div className="rounded-xl px-2.5 py-2" role="group" aria-label="Web search mode" data-testid="web-search-mode">
                                            <span className="flex min-w-0 items-center gap-2.5 text-[13px] text-[var(--q-text)]">
                                                <Globe className={`${ICON} ${webSearch ? "text-cyan-300" : "text-[var(--q-text-muted)]"}`} strokeWidth={1.75} />
                                                <span>Web search</span>
                                            </span>
                                            <div className="mt-2 grid grid-cols-3 gap-0.5 rounded-full border border-[var(--q-border)] bg-[var(--q-canvas)] p-0.5">
                                                {WEB_SEARCH_MODE_OPTIONS.map((opt) => {
                                                    const active = webSearchMode === opt.value;
                                                    return (
                                                        <button
                                                            key={opt.value}
                                                            type="button"
                                                            role="menuitemradio"
                                                            aria-checked={active}
                                                            aria-label={`Web search ${opt.label}`}
                                                            data-mode={opt.value}
                                                            title={opt.hint}
                                                            onClick={() => onWebSearchModeChange(opt.value)}
                                                            className={`rounded-full px-2 py-1 text-[12px] font-medium transition-colors ${
                                                                active
                                                                    ? "bg-[var(--q-card)] text-[var(--q-text)] shadow-[0_1px_2px_var(--q-shadow-glow)] ring-1 ring-[var(--q-border-strong)]"
                                                                    : "text-[var(--q-text-muted)] hover:text-[var(--q-text)]"
                                                            }`}
                                                        >
                                                            {opt.label}
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="ml-auto flex shrink-0 items-center gap-2">
                                <div ref={attachRef} className="relative">
                                    <button
                                        type="button"
                                        onClick={() => toggleMenu("attach")}
                                        disabled={isStreaming}
                                        aria-haspopup="menu"
                                        aria-expanded={attachOpen}
                                        className="q-pill h-8 gap-1.5 px-3 text-[13px]"
                                        title="Attach images or documents"
                                    >
                                        <Paperclip className="h-3.5 w-3.5 text-[var(--q-text-muted)]" strokeWidth={1.75} />
                                        <span>Attach</span>
                                    </button>

                                    {attachOpen && (
                                        <div
                                            role="menu"
                                            className="glass-popover absolute bottom-full right-0 z-30 mb-2 w-52 max-w-[calc(100vw-2rem)] overflow-hidden rounded-2xl p-1.5"
                                        >
                                            <button
                                                type="button"
                                                role="menuitem"
                                                onClick={() => {
                                                    setOpenMenu(null);
                                                    imageInputRef.current?.click();
                                                }}
                                                className={MENU_ROW}
                                            >
                                                <ImageIcon className={`${ICON} text-[var(--q-text-muted)]`} strokeWidth={1.75} />
                                                <span>Upload images</span>
                                            </button>
                                            <button
                                                type="button"
                                                role="menuitem"
                                                onClick={() => {
                                                    setOpenMenu(null);
                                                    documentInputRef.current?.click();
                                                }}
                                                className={MENU_ROW}
                                            >
                                                <FileText className={`${ICON} text-[var(--q-text-muted)]`} strokeWidth={1.75} />
                                                <span>Upload documents</span>
                                            </button>
                                        </div>
                                    )}
                                </div>

                                {isStreaming ? (
                                    <button
                                        type="button"
                                        onClick={onStop}
                                        className="q-err inline-flex h-8 items-center justify-center gap-1.5 rounded-full bg-red-500/10 px-3.5 text-[13px] font-medium transition-colors hover:bg-red-500/15"
                                        title="Stop generating"
                                    >
                                        <Square className="h-3 w-3 fill-current" />
                                        <span>Stop</span>
                                    </button>
                                ) : (
                                    <button
                                        type="submit"
                                        disabled={!value.trim() && attachments.length === 0}
                                        className="q-pill-ink h-8 gap-1.5 px-3.5 text-[13px]"
                                    >
                                        <ArrowUp className="h-3.5 w-3.5" strokeWidth={2} />
                                        <span>Send</span>
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </form>
                {!isHero && (
                    <p className="mt-2 flex flex-wrap justify-center gap-x-1.5 text-center text-[12px] text-[var(--q-text-faint)]">
                        <span>Quasar may produce inaccurate information.</span>
                        <span className="whitespace-nowrap">· Accepts images, PDFs, FITS, CSV</span>
                        <span className={`whitespace-nowrap transition-opacity duration-500 ${hitCount !== null ? "opacity-100" : "opacity-0"}`}>· Visits: <span className="tabular-nums">{hitCount !== null ? hitCount.toLocaleString() : "—"}</span></span>
                    </p>
                )}
            </div>
        </div>
    );
}
