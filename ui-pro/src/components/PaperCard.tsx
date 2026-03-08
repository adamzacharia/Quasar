"use client";

import { ArrowRight, Quote, ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import { useState } from "react";
import type { Paper } from "@/lib/types";

const TYPE_STYLES: Record<string, string> = {
    journal: "bg-emerald-500/10 text-emerald-400",
    arxiv: "bg-orange-500/10 text-orange-400",
    radio: "bg-primary/10 text-primary",
};

interface PaperCardProps { paper: Paper; }

export function PaperCard({ paper }: PaperCardProps) {
    const [expanded, setExpanded] = useState(false);
    const typeLabel = paper.type === "journal" ? "Journal" : paper.type === "arxiv" ? "ArXiv" : "Radio";

    // Build ADS link from bibcode, fallback to DOI
    const adsUrl = paper.bibcode
        ? `https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(paper.bibcode)}`
        : paper.doi
            ? `https://doi.org/${paper.doi}`
            : paper.arxivId
                ? `https://arxiv.org/abs/${paper.arxivId}`
                : null;

    return (
        <div className="bg-sidebar-dark rounded-2xl border border-slate-700/50 hover:border-emerald-500/50 transition-colors group flex flex-col justify-between h-full">
            {/* Clickable card header — opens ADS link */}
            <a
                href={adsUrl || "#"}
                target="_blank"
                rel="noopener noreferrer"
                className="block p-4 space-y-2"
                onClick={(e) => { if (!adsUrl) e.preventDefault(); }}
            >
                <div className="flex items-start justify-between">
                    <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${TYPE_STYLES[paper.type] || TYPE_STYLES.journal}`}>{typeLabel}</span>
                    <span className="text-xs text-slate-400">{paper.year}</span>
                </div>
                <h4 className="text-sm font-bold text-white group-hover:text-primary transition-colors leading-snug">{paper.title}</h4>
                <p className="text-xs text-slate-400 line-clamp-1">{paper.authors}</p>
            </a>

            {/* Abstract expansion */}
            {paper.abstract && (
                <div className="px-4 pb-2">
                    <button
                        onClick={() => setExpanded(!expanded)}
                        className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-primary transition-colors"
                    >
                        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        {expanded ? "Hide Abstract" : "View Abstract"}
                    </button>
                    {expanded && (
                        <p className="mt-2 text-xs text-slate-300 leading-relaxed border-t border-slate-700/50 pt-2">
                            {paper.abstract}
                        </p>
                    )}
                </div>
            )}

            {/* Footer with citations & external link */}
            <div className="mt-auto px-4 pb-4 pt-3 border-t border-slate-700/50 flex items-center justify-between text-xs">
                <div className="flex items-center gap-1 text-slate-500"><Quote className="w-3.5 h-3.5" /><span>{paper.citationCount} Citations</span></div>
                {adsUrl ? (
                    <a href={adsUrl} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-slate-400 hover:text-primary transition-colors">
                        <span className="text-[10px]">ADS</span>
                        <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                ) : (
                    <ArrowRight className="w-4 h-4 text-slate-400 group-hover:translate-x-1 transition-transform" />
                )}
            </div>
        </div>
    );
}
