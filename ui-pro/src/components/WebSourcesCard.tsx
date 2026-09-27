"use client";

/* eslint-disable @next/next/no-img-element */
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ExternalLink, Globe, ChevronRight, ChevronUp, ShieldCheck, X } from "lucide-react";
import type { WebSource, WebImage } from "../lib/types";
import { rankWebSources } from "../lib/evidence-quality";
import { isSafeWebImage, isSafeWebSource } from "../lib/content-safety";
import { citationLabel, distinctPathHints, domainOf, sourceCardDomId, splitCitedSources } from "../lib/web-citations.js";

/*
 * Grounded web sources, laid out like ChatGPT / Claude: a compact "N sources"
 * pill above the answer, a short row of cited sources under the finished
 * answer, and the full list in a side panel that the pill, the row and the
 * inline citation chips open. Nothing large is drawn while the answer streams.
 * Pages that share a title show their path, so they can be told apart.
 */

const OPEN_EVENT = "quasar:web-sources-open";

/** Open the sources panel of one message, optionally scrolled to a citation. */
export function openWebSources(messageId: string, citationId?: string) {
    if (typeof window === "undefined") return;
    window.dispatchEvent(new CustomEvent(OPEN_EVENT, { detail: { messageId, citationId } }));
}

function sourceDomain(s: WebSource): string {
    return s.domain || domainOf(s.url) || "";
}

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
const NON_RENDERED = new Set(["SCRIPT", "STYLE", "LINK", "TEMPLATE", "NOSCRIPT"]);

/** Plain-text excerpt: markdown marks and link syntax removed, whitespace collapsed. */
function plainSnippet(text?: string): string {
    return String(text || "")
        .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
        .replace(/[#*_`>|]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function FaviconStack({ sources }: { sources: WebSource[] }) {
    const domains: string[] = [];
    for (const s of sources) {
        const d = sourceDomain(s);
        if (d && !domains.includes(d)) domains.push(d);
    }
    return (
        <span className="flex -space-x-1">
            {domains.slice(0, 4).map((d) => (
                <img
                    key={d}
                    src={getFavicon(`https://${d}`)}
                    alt=""
                    className="h-4 w-4 rounded-full ring-1 ring-black/20"
                    style={{ background: "var(--q-glass-control)" }}
                    onError={(e) => { (e.target as HTMLImageElement).style.visibility = "hidden"; }}
                />
            ))}
        </span>
    );
}

/** Compact "N sources" pill above a grounded answer; opens the sources panel. */
export function WebSourcesStrip({ sources = [], messageId }: { sources?: WebSource[]; messageId: string }) {
    const safe = sources.filter(isSafeWebSource);
    if (safe.length === 0) return null;
    return (
        <button
            type="button"
            data-testid="web-sources-strip"
            onClick={() => openWebSources(messageId)}
            className="glass-control my-2 inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs transition-colors hover:text-primary"
            style={{ color: "var(--q-text-muted)" }}
            title="Show sources"
        >
            <Globe className="h-3.5 w-3.5" />
            <FaviconStack sources={safe} />
            <span style={{ color: "var(--q-text)" }}>{safe.length} source{safe.length === 1 ? "" : "s"}</span>
        </button>
    );
}

/**
 * Row under the finished answer: the first cited sources as small chips (open
 * the page) and a button that opens the full list in the side panel.
 */
export function WebSourcesFooter({ sources = [], messageId }: { sources?: WebSource[]; messageId: string }) {
    const safe = sources.filter(isSafeWebSource);
    if (safe.length === 0) return null;
    const { cited } = splitCitedSources(safe) as { cited: WebSource[]; uncited: WebSource[] };
    const shown = cited.slice(0, 4);
    return (
        <div data-testid="web-sources-card" className="my-3 flex flex-wrap items-center gap-2">
            {shown.map((s) => (
                <a
                    key={s.id || s.url}
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    data-source-id={s.id || undefined}
                    title={s.title || s.url}
                    className="glass-control inline-flex max-w-[16rem] items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-colors hover:text-primary"
                    style={{ color: "var(--q-text)" }}
                >
                    {s.id && (
                        <span className="text-[10px] font-semibold" style={{ color: "var(--q-text-muted)" }}>
                            {citationLabel(s.id)}
                        </span>
                    )}
                    <img
                        src={getFavicon(s.url)}
                        alt=""
                        className="h-3.5 w-3.5 rounded-sm"
                        onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                    />
                    <span className="truncate">{s.title || sourceDomain(s)}</span>
                </a>
            ))}
            <button
                type="button"
                onClick={() => openWebSources(messageId)}
                className="glass-control inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs transition-colors hover:text-primary"
                style={{ color: "var(--q-text-muted)" }}
            >
                <Globe className="h-3.5 w-3.5" />
                <FaviconStack sources={safe} />
                <span style={{ color: "var(--q-text)" }}>
                    {shown.length ? "All sources" : `${safe.length} source${safe.length === 1 ? "" : "s"}`}
                </span>
            </button>
        </div>
    );
}

function PanelSourceCard({ source, messageId, hint = "" }: { source: WebSource; messageId: string; hint?: string }) {
    const quality = source.evidenceQuality;
    const domain = sourceDomain(source);
    // Page excerpts arrive as raw markdown ("### Download PDF"): show plain text.
    const snippet = plainSnippet(source.snippet);
    return (
        <a
            id={source.id ? sourceCardDomId(messageId, source.id) : undefined}
            data-source-id={source.id || undefined}
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-start gap-2.5 rounded-lg p-2.5 transition-colors hover:bg-[var(--q-glass-hover)] data-[highlight=true]:ring-2 data-[highlight=true]:ring-cyan-400/80"
        >
            <span
                className="mt-0.5 inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md px-1 text-[10px] font-semibold"
                style={{ color: "var(--q-text)", border: "1px solid var(--q-glass-border)" }}
            >
                {source.id ? citationLabel(source.id) : "·"}
            </span>
            <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5 text-[11px]" style={{ color: "var(--q-text-muted)" }}>
                    <img
                        src={getFavicon(source.url)}
                        alt=""
                        className="h-3.5 w-3.5 rounded-sm"
                        onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                    />
                    <span className="truncate">
                        {domain}{hint}{source.publishedDate ? ` · ${source.publishedDate}` : ""}
                        {quality?.label ? ` · ${quality.label}` : ""}
                    </span>
                </span>
                <span className="mt-0.5 line-clamp-2 text-sm font-medium group-hover:text-primary" style={{ color: "var(--q-text)" }}>
                    {source.title || domain || "Untitled"}
                </span>
                {snippet && (
                    <span className="mt-0.5 line-clamp-2 text-xs" style={{ color: "var(--q-text-muted)" }}>
                        {snippet}
                    </span>
                )}
            </span>
            <ExternalLink className="mt-1 h-3 w-3 shrink-0 opacity-0 transition-opacity group-hover:opacity-100" style={{ color: "var(--q-text-muted)" }} />
        </a>
    );
}

/**
 * Side panel with every source of one answer: cited first (with their
 * citation numbers), then the other pages consulted. Opened through
 * openWebSources(); Escape, the close button or the backdrop closes it.
 */
export function WebSourcesPanel({ sources = [], messageId }: { sources?: WebSource[]; messageId: string }) {
    const [open, setOpen] = useState(false);
    const [focusId, setFocusId] = useState<string | null>(null);
    const closeRef = useRef<HTMLButtonElement | null>(null);
    const rootRef = useRef<HTMLDivElement | null>(null);
    const panelRef = useRef<HTMLElement | null>(null);
    const returnFocusRef = useRef<HTMLElement | null>(null);

    useEffect(() => {
        const onOpen = (e: Event) => {
            const detail = (e as CustomEvent).detail || {};
            if (detail.messageId !== messageId) return;
            // The chip, pill or footer button that opened the panel gets focus back on close.
            const active = document.activeElement;
            returnFocusRef.current = active instanceof HTMLElement && active !== document.body ? active : null;
            setFocusId(detail.citationId || null);
            setOpen(true);
        };
        window.addEventListener(OPEN_EVENT, onOpen);
        return () => window.removeEventListener(OPEN_EVENT, onOpen);
    }, [messageId]);

    useEffect(() => {
        if (!open) return;
        // aria-modal: the rest of the page is inert while the panel is open, so
        // neither Tab nor a screen reader reaches the obscured chat controls.
        const inerted: Element[] = [];
        for (const el of Array.from(document.body.children)) {
            if (el === rootRef.current || el.hasAttribute("inert") || NON_RENDERED.has(el.tagName)) continue;
            el.setAttribute("inert", "");
            inerted.push(el);
        }
        closeRef.current?.focus();
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                setOpen(false);
                return;
            }
            const panel = panelRef.current;
            if (e.key !== "Tab" || !panel) return;
            // Keep Tab cycling inside the panel.
            const items = Array.from(panel.querySelectorAll<HTMLElement>(FOCUSABLE));
            if (items.length === 0) {
                e.preventDefault();
                return;
            }
            const first = items[0];
            const last = items[items.length - 1];
            const active = document.activeElement;
            const inside = active instanceof Node && panel.contains(active);
            if (e.shiftKey && (active === first || !inside)) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && (active === last || !inside)) {
                e.preventDefault();
                first.focus();
            }
        };
        window.addEventListener("keydown", onKey);
        return () => {
            window.removeEventListener("keydown", onKey);
            for (const el of inerted) el.removeAttribute("inert");
            const back = returnFocusRef.current;
            returnFocusRef.current = null;
            if (back && back.isConnected) back.focus();
        };
    }, [open]);

    useEffect(() => {
        if (!open || !focusId) return;
        const timer = setTimeout(() => {
            const el = document.getElementById(sourceCardDomId(messageId, focusId));
            if (!el) return;
            el.scrollIntoView({ behavior: "smooth", block: "center" });
            el.setAttribute("data-highlight", "true");
            setTimeout(() => el.removeAttribute("data-highlight"), 1800);
        }, 80);
        return () => clearTimeout(timer);
    }, [open, focusId, messageId]);

    const safe = sources.filter(isSafeWebSource);
    if (!open || safe.length === 0 || typeof document === "undefined") return null;
    const { cited, uncited } = splitCitedSources(safe) as { cited: WebSource[]; uncited: WebSource[] };
    const hintList = distinctPathHints(safe) as string[];
    const hints = new Map(safe.map((s, i) => [s, hintList[i]] as const));
    const card = (s: WebSource) => (
        <PanelSourceCard key={s.id || s.url} source={s} messageId={messageId} hint={hints.get(s) || ""} />
    );
    const heading = (text: string, n: number) => (
        <div className="px-2.5 pb-1 text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--q-text-muted)" }}>
            {text} ({n})
        </div>
    );

    return createPortal(
        <div ref={rootRef} className="fixed inset-0 z-50 flex justify-end" role="presentation">
            <div className="absolute inset-0" style={{ background: "var(--q-drawer-scrim)" }} onClick={() => setOpen(false)} />
            <aside
                ref={panelRef}
                role="dialog"
                aria-modal="true"
                aria-label="Sources"
                data-testid="web-sources-panel"
                className="glass-drawer relative flex h-full w-[min(420px,92vw)] flex-col"
                style={{ borderRight: "none", borderLeft: "1px solid var(--q-drawer-border)" }}
            >
                <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: "1px solid var(--q-glass-border)" }}>
                    <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: "var(--q-text)" }}>
                        <Globe className="h-4 w-4" />
                        Sources
                        <span className="font-normal" style={{ color: "var(--q-text-muted)" }}>{safe.length}</span>
                    </div>
                    <button
                        ref={closeRef}
                        type="button"
                        aria-label="Close sources"
                        onClick={() => setOpen(false)}
                        className="rounded-md p-1 transition-colors hover:bg-[var(--q-glass-hover)]"
                        style={{ color: "var(--q-text-muted)" }}
                    >
                        <X className="h-4 w-4" />
                    </button>
                </div>
                <div className="flex-1 space-y-4 overflow-y-auto p-2">
                    {cited.length > 0 && (
                        <section>
                            {heading("Cited in the answer", cited.length)}
                            {cited.map(card)}
                        </section>
                    )}
                    {uncited.length > 0 && (
                        <section>
                            {heading(cited.length ? "More sources" : "Sources consulted", uncited.length)}
                            {uncited.map(card)}
                        </section>
                    )}
                </div>
            </aside>
        </div>,
        document.body,
    );
}


interface WebSourcesCardProps {
    sources?: WebSource[];
    images?: WebImage[];
}

function getFavicon(url: string): string {
    try {
        const domain = new URL(url).hostname;
        return `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
    } catch {
        return "";
    }
}

function getDomain(url: string): string {
    try {
        return new URL(url).hostname.replace("www.", "");
    } catch {
        return url;
    }
}

function getQualityColor(tier?: string): string {
    switch (tier) {
        case "primary":
            return "#7dd3fc";
        case "peer_reviewed":
            return "#86efac";
        case "preprint":
            return "#facc15";
        case "institutional":
            return "#c4b5fd";
        case "reference":
            return "#f9a8d4";
        default:
            return "#94a3b8";
    }
}

export function WebSourcesCard({ sources = [], images = [] }: WebSourcesCardProps) {
    const [showAllSources, setShowAllSources] = useState(false);
    const [failedImages, setFailedImages] = useState<Set<number>>(new Set());

    const rankedSources = rankWebSources(sources.filter(isSafeWebSource)) as WebSource[];
    const visibleSources = showAllSources ? rankedSources : rankedSources.slice(0, 4);
    const validImages = images
        .map((image, originalIndex) => ({ image, originalIndex }))
        .filter(({ image, originalIndex }) => isSafeWebImage(image) && !failedImages.has(originalIndex));

    const handleImageError = (index: number) => {
        setFailedImages(prev => new Set(prev).add(index));
    };

    if (rankedSources.length === 0 && validImages.length === 0) return null;

    return (
        <div className="space-y-4 my-3">
            {/* Source Pills */}
            {rankedSources.length > 0 && (
                <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs uppercase font-medium"
                         style={{ color: 'var(--q-text-muted)' }}>
                        <Globe className="w-3.5 h-3.5" />
                        <span>Evidence-ranked sources</span>
                        <span style={{ color: 'var(--q-text-muted)', opacity: 0.6 }}>({rankedSources.length})</span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                        {visibleSources.map((source, i) => {
                            const quality = source.evidenceQuality;
                            const qualityColor = getQualityColor(quality?.tier);
                            return (
                                <a
                                    key={`${source.url}-${i}`}
                                    href={source.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="glass-control group flex items-start gap-2.5 p-2.5 rounded-lg
                                           transition-all duration-200 cursor-pointer
                                           overflow-hidden"
                                    title={quality?.reason || source.snippet}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.borderColor = 'var(--q-border-strong)';
                                        e.currentTarget.style.background = 'var(--q-glass-control-hover)';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.borderColor = 'var(--q-glass-border)';
                                        e.currentTarget.style.background = 'var(--q-glass-control)';
                                    }}
                                >
                                    <div className="shrink-0 mt-0.5 space-y-1">
                                        <img
                                            src={getFavicon(source.url)}
                                            alt=""
                                            className="w-4 h-4 rounded-sm opacity-70 group-hover:opacity-100 transition-opacity"
                                            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                                        />
                                        {quality && (
                                            <ShieldCheck className="w-4 h-4" style={{ color: qualityColor }} />
                                        )}
                                    </div>
                                    <div className="min-w-0 flex-1">
                                        {quality && (
                                            <div className="text-[10px] font-medium truncate mb-0.5"
                                                 style={{ color: qualityColor }}>
                                                {quality.label}
                                            </div>
                                        )}
                                        <div className="text-xs font-medium truncate group-hover:text-primary transition-colors"
                                             style={{ color: 'var(--q-text)' }}>
                                            {source.title || "Untitled"}
                                        </div>
                                        <div className="text-[10px] truncate mt-0.5"
                                             style={{ color: 'var(--q-text-muted)' }}>
                                            {getDomain(source.url)}
                                        </div>
                                    </div>
                                    <ExternalLink className="w-3 h-3 shrink-0 mt-0.5 group-hover:text-primary transition-colors"
                                                  style={{ color: 'var(--q-text-muted)' }} />
                                </a>
                            );
                        })}
                    </div>
                    {rankedSources.length > 4 && (
                        showAllSources ? (
                            <button
                                onClick={() => setShowAllSources(false)}
                                className="flex items-center gap-1 text-xs hover:text-primary transition-colors mt-1"
                                style={{ color: 'var(--q-text-muted)' }}
                            >
                                <span>Show less</span>
                                <ChevronUp className="w-3 h-3" />
                            </button>
                        ) : (
                            <button
                                onClick={() => setShowAllSources(true)}
                                className="flex items-center gap-1 text-xs hover:text-primary transition-colors mt-1"
                                style={{ color: 'var(--q-text-muted)' }}
                            >
                                <span>View all {rankedSources.length} sources</span>
                                <ChevronRight className="w-3 h-3" />
                            </button>
                        )
                    )}
                </div>
            )}

            {/* Image Grid */}
            {validImages.length > 0 && (
                <div className="space-y-2">
                    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
                        {validImages.map(({ image, originalIndex }) => {
                            const clickUrl = image.sourceUrl || image.url;
                            const clickTitle = image.sourceTitle
                                ? `Open source: ${image.sourceTitle}`
                                : image.description || "View image";
                            return (
                                <a
                                    key={originalIndex}
                                    href={clickUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="glass-control group relative aspect-square rounded-lg overflow-hidden
                                               hover:scale-[1.03]
                                               transition-all duration-200 cursor-pointer"
                                    title={clickTitle}
                                >
                                    <img
                                        src={image.url}
                                        alt={image.description || "Web search result"}
                                        className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity"
                                        loading="lazy"
                                        referrerPolicy="no-referrer"
                                        onError={() => handleImageError(originalIndex)}
                                    />
                                    {/* Hover overlay with description */}
                                    {image.description && (
                                        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent
                                                        p-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                            <p className="text-[10px] text-white/90 line-clamp-2 leading-tight">
                                                {image.description}
                                            </p>
                                        </div>
                                    )}
                                    {/* External link indicator */}
                                    <div className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <div className="bg-black/60 backdrop-blur-sm rounded-full p-1">
                                            <ExternalLink className="w-2.5 h-2.5 text-white" />
                                        </div>
                                    </div>
                                </a>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
