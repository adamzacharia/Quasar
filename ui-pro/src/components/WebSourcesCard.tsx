"use client";

import { useState } from "react";
import { ExternalLink, Globe, ChevronRight } from "lucide-react";
import type { WebSource, WebImage } from "../lib/types";

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

export function WebSourcesCard({ sources = [], images = [] }: WebSourcesCardProps) {
    const [showAllSources, setShowAllSources] = useState(false);
    const [failedImages, setFailedImages] = useState<Set<number>>(new Set());

    const visibleSources = showAllSources ? sources : sources.slice(0, 4);
    const validImages = images.filter((_, i) => !failedImages.has(i));

    const handleImageError = (index: number) => {
        setFailedImages(prev => new Set(prev).add(index));
    };

    if (sources.length === 0 && validImages.length === 0) return null;

    return (
        <div className="space-y-4 my-3">
            {/* Source Pills */}
            {sources.length > 0 && (
                <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs uppercase tracking-wider font-medium"
                         style={{ color: 'var(--q-text-muted)' }}>
                        <Globe className="w-3.5 h-3.5" />
                        <span>Sources</span>
                        <span style={{ color: 'var(--q-text-muted)', opacity: 0.6 }}>({sources.length})</span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                        {visibleSources.map((source, i) => (
                            <a
                                key={i}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="group flex items-start gap-2.5 p-2.5 rounded-xl
                                           transition-all duration-200 cursor-pointer
                                           overflow-hidden hover:scale-[1.02]"
                                style={{
                                    background: 'var(--q-glass-bg)',
                                    border: '1px solid var(--q-glass-border)',
                                }}
                                title={source.snippet}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.borderColor = 'rgba(244, 113, 181, 0.4)';
                                    e.currentTarget.style.background = 'var(--q-glass-hover)';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.borderColor = 'var(--q-glass-border)';
                                    e.currentTarget.style.background = 'var(--q-glass-bg)';
                                }}
                            >
                                <img
                                    src={getFavicon(source.url)}
                                    alt=""
                                    className="w-4 h-4 rounded-sm mt-0.5 shrink-0 opacity-70 group-hover:opacity-100 transition-opacity"
                                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                                />
                                <div className="min-w-0 flex-1">
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
                        ))}
                    </div>
                    {sources.length > 4 && !showAllSources && (
                        <button
                            onClick={() => setShowAllSources(true)}
                            className="flex items-center gap-1 text-xs hover:text-primary transition-colors mt-1"
                            style={{ color: 'var(--q-text-muted)' }}
                        >
                            <span>View all {sources.length} sources</span>
                            <ChevronRight className="w-3 h-3" />
                        </button>
                    )}
                </div>
            )}

            {/* Image Grid */}
            {validImages.length > 0 && (
                <div className="space-y-2">
                    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
                        {images.map((image, i) => {
                            if (failedImages.has(i)) return null;
                            return (
                                <a
                                    key={i}
                                    href={image.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="group relative aspect-square rounded-xl overflow-hidden
                                               hover:scale-[1.03]
                                               transition-all duration-200 cursor-pointer"
                                    style={{
                                        background: 'var(--q-glass-bg)',
                                        border: '1px solid var(--q-glass-border)',
                                    }}
                                    title={image.description || "View image"}
                                >
                                    <img
                                        src={image.url}
                                        alt={image.description || "Web search result"}
                                        className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity"
                                        loading="lazy"
                                        onError={() => handleImageError(i)}
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
