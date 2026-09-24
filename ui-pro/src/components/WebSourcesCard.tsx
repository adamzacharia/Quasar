"use client";

/* eslint-disable @next/next/no-img-element */
import { useState } from "react";
import { ExternalLink, Globe, ChevronRight, ChevronUp, ShieldCheck } from "lucide-react";
import type { WebSource, WebImage } from "../lib/types";
import { rankWebSources } from "../lib/evidence-quality";
import { isSafeWebImage, isSafeWebSource } from "../lib/content-safety";
import { citationLabel, domainOf, sourceCardDomId, splitCitedSources } from "../lib/web-citations.js";

function gridDomId(messageId: string): string {
    return `web-grid-${String(messageId || "m").replace(/[^A-Za-z0-9_-]/g, "")}`;
}

/**
 * Compact "N sources" strip shown above a grounded answer as soon as the web
 * evidence exists (before the answer finishes). Clicking scrolls to the grid.
 */
export function WebSourcesStrip({ sources = [], messageId }: { sources?: WebSource[]; messageId: string }) {
    const safe = sources.filter(isSafeWebSource);
    if (safe.length === 0) return null;
    const domains: string[] = [];
    for (const s of safe) {
        const d = s.domain || domainOf(s.url);
        if (d && !domains.includes(d)) domains.push(d);
    }
    const jump = () => document.getElementById(gridDomId(messageId))?.scrollIntoView({ behavior: "smooth", block: "start" });
    return (
        <button
            type="button"
            data-testid="web-sources-strip"
            onClick={jump}
            className="glass-control my-2 inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs transition-colors hover:text-primary"
            style={{ color: "var(--q-text-muted)" }}
            title={domains.slice(0, 8).join(", ")}
        >
            <Globe className="h-3.5 w-3.5" />
            <span className="flex -space-x-1">
                {domains.slice(0, 5).map((d) => (
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
            <span style={{ color: "var(--q-text)" }}>{safe.length} source{safe.length === 1 ? "" : "s"}</span>
        </button>
    );
}

function EvidenceSourceCard({ source, messageId }: { source: WebSource; messageId: string }) {
    const quality = source.evidenceQuality;
    const domain = source.domain || getDomain(source.url);
    return (
        <a
            id={source.id ? sourceCardDomId(messageId, source.id) : undefined}
            data-source-id={source.id || undefined}
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            title={source.snippet || quality?.reason || source.title}
            className="glass-control group flex items-start gap-2.5 rounded-lg p-2.5 transition-all duration-200 hover:scale-[1.01] data-[highlight=true]:ring-2 data-[highlight=true]:ring-cyan-400/80"
        >
            <span
                className="mt-0.5 inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md px-1 text-[10px] font-semibold"
                style={{ color: "var(--q-text)", border: "1px solid var(--q-glass-border)" }}
            >
                {source.id ? citationLabel(source.id) : "·"}
            </span>
            <span className="min-w-0 flex-1">
                <span className="block truncate text-xs font-medium group-hover:text-primary" style={{ color: "var(--q-text)" }}>
                    {source.title || domain || "Untitled"}
                </span>
                <span className="mt-0.5 flex items-center gap-1 truncate text-[10px]" style={{ color: "var(--q-text-muted)" }}>
                    <img
                        src={getFavicon(source.url)}
                        alt=""
                        className="h-3 w-3 rounded-sm"
                        onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                    />
                    <span className="truncate">
                        {domain}{source.publishedDate ? ` · ${source.publishedDate}` : ""}
                        {quality?.label ? ` · ${quality.label}` : ""}
                    </span>
                </span>
            </span>
            <ExternalLink className="mt-0.5 h-3 w-3 shrink-0" style={{ color: "var(--q-text-muted)" }} />
        </a>
    );
}

/**
 * Sources grid under a grounded answer: sources the answer cites first (with
 * their citation number), the rest under a collapsed "Also consulted". When
 * the answer cites nothing, every source is listed as consulted.
 */
export function WebEvidenceGrid({ sources = [], messageId }: { sources?: WebSource[]; messageId: string }) {
    const safe = sources.filter(isSafeWebSource);
    if (safe.length === 0) return null;
    const { cited, uncited } = splitCitedSources(safe) as { cited: WebSource[]; uncited: WebSource[] };
    const grid = "grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3";
    return (
        <div id={gridDomId(messageId)} data-testid="web-sources-card" className="my-3 space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium uppercase" style={{ color: "var(--q-text-muted)" }}>
                <Globe className="h-3.5 w-3.5" />
                <span>{cited.length ? "Sources cited" : "Sources consulted"}</span>
                <span style={{ opacity: 0.6 }}>({cited.length || uncited.length})</span>
            </div>
            {cited.length > 0 ? (
                <>
                    <div className={grid}>
                        {cited.map((s) => <EvidenceSourceCard key={s.id || s.url} source={s} messageId={messageId} />)}
                    </div>
                    {uncited.length > 0 && (
                        <details className="group/also">
                            <summary className="cursor-pointer select-none text-xs hover:text-primary" style={{ color: "var(--q-text-muted)" }}>
                                Also consulted ({uncited.length})
                            </summary>
                            <div className={`${grid} mt-2`}>
                                {uncited.map((s) => <EvidenceSourceCard key={s.id || s.url} source={s} messageId={messageId} />)}
                            </div>
                        </details>
                    )}
                </>
            ) : (
                <div className={grid}>
                    {uncited.map((s) => <EvidenceSourceCard key={s.id || s.url} source={s} messageId={messageId} />)}
                </div>
            )}
        </div>
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
                                           overflow-hidden hover:scale-[1.02]"
                                    title={quality?.reason || source.snippet}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.borderColor = 'rgba(167, 139, 250, 0.45)';
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
