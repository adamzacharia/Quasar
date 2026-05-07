"use client";

import { ArrowRight, Quote, ChevronDown, ChevronUp, ExternalLink, Bookmark, Award, TrendingUp, FileText } from "lucide-react";
import { useState } from "react";
import type { Paper } from "../lib/types";
import { useChatStore } from "../lib/store";

const TYPE_STYLES: Record<string, string> = {
    journal: "bg-emerald-500/10 text-emerald-400",
    arxiv: "bg-orange-500/10 text-orange-400",
    radio: "bg-primary/10 text-primary",
};

interface PaperCardProps { paper: Paper; }

export function PaperCard({ paper }: PaperCardProps) {
    const [expanded, setExpanded] = useState(false);
    const { savedPapers, savePaper, removePaper } = useChatStore();
    const isSaved = savedPapers.some(p => p.id === paper.id);
    const typeLabel = paper.type === "journal" ? "Journal" : paper.type === "arxiv" ? "ArXiv" : "Radio";

    // Build ADS link from bibcode, fallback to DOI
    const adsUrl = paper.bibcode
        ? `https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(paper.bibcode)}`
        : paper.doi
            ? `https://doi.org/${paper.doi}`
            : paper.arxivId
                ? `https://arxiv.org/abs/${paper.arxivId}`
                : null;

    const handleToggleSave = (e: React.MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (isSaved) {
            removePaper(paper.id);
        } else {
            savePaper(paper);
        }
    };

    // Enrichment badge helpers
    const hasFunders = paper.funders && paper.funders.length > 0;
    const hasFwci = paper.fwci != null && paper.fwci > 0;

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
                    <div className="flex items-center gap-1.5 flex-wrap">
                        <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${TYPE_STYLES[paper.type] || TYPE_STYLES.journal}`}>{typeLabel}</span>
                        {/* Top 1% badge */}
                        {paper.isTop1Percent && (
                            <span className="px-1.5 py-0.5 rounded-md text-[9px] font-bold uppercase tracking-wider bg-amber-500/15 text-amber-400 flex items-center gap-0.5">
                                <Award className="w-2.5 h-2.5" />Top 1%
                            </span>
                        )}
                        {/* Top 10% badge (only if NOT top 1%) */}
                        {!paper.isTop1Percent && paper.isTop10Percent && (
                            <span className="px-1.5 py-0.5 rounded-md text-[9px] font-bold uppercase tracking-wider bg-blue-500/15 text-blue-400 flex items-center gap-0.5">
                                <TrendingUp className="w-2.5 h-2.5" />Top 10%
                            </span>
                        )}
                        {/* OA PDF badge */}
                        {paper.oaPdfUrl && (
                            <span
                                className="px-1.5 py-0.5 rounded-md text-[9px] font-bold uppercase tracking-wider bg-green-500/15 text-green-400 flex items-center gap-0.5 cursor-pointer hover:bg-green-500/25 transition-colors"
                                onClick={(e) => { e.preventDefault(); e.stopPropagation(); window.open(paper.oaPdfUrl, '_blank'); }}
                                title="Open Access PDF"
                            >
                                <FileText className="w-2.5 h-2.5" />PDF
                            </span>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-400">{paper.year}</span>
                        <button
                            onClick={handleToggleSave}
                            title={isSaved ? "Remove from Saved Papers" : "Save Paper"}
                            className={`p-1 rounded-md transition-all ${
                                isSaved
                                    ? "text-yellow-500 hover:text-yellow-400 bg-yellow-500/10"
                                    : "text-slate-500 hover:text-yellow-500 hover:bg-yellow-500/10 opacity-0 group-hover:opacity-100"
                            }`}
                        >
                            <Bookmark className="w-3.5 h-3.5" fill={isSaved ? "currentColor" : "none"} />
                        </button>
                    </div>
                </div>
                <h4 className="text-sm font-bold text-white group-hover:text-primary transition-colors leading-snug">{paper.title}</h4>
                <p className="text-xs text-slate-400 line-clamp-1">{paper.authors}</p>

                {/* Funder tags */}
                {hasFunders && (
                    <div className="flex items-center gap-1.5 flex-wrap pt-1">
                        <span className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider">Funded by</span>
                        {paper.funders!.slice(0, 3).map((f, i) => (
                            <span key={i} className="px-1.5 py-0.5 rounded text-[9px] bg-violet-500/10 text-violet-400 font-medium">
                                {f.name}
                            </span>
                        ))}
                        {paper.funders!.length > 3 && (
                            <span className="text-[9px] text-slate-500">+{paper.funders!.length - 3} more</span>
                        )}
                    </div>
                )}
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

            {/* Footer with citations, FWCI & external link */}
            <div className="mt-auto px-4 pb-4 pt-3 border-t border-slate-700/50 flex items-center justify-between text-xs">
                <div className="flex items-center gap-3">
                    {paper.bibcode ? (
                        <a href={`https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(paper.bibcode)}/citations`}
                           target="_blank" rel="noopener noreferrer"
                           className="flex items-center gap-1 text-slate-500 hover:text-primary transition-colors">
                            <Quote className="w-3.5 h-3.5" /><span>{paper.citationCount} Citations</span>
                        </a>
                    ) : (
                        <div className="flex items-center gap-1 text-slate-500"><Quote className="w-3.5 h-3.5" /><span>{paper.citationCount} Citations</span></div>
                    )}
                    {/* FWCI score */}
                    {hasFwci && (
                        <span className="text-[10px] text-cyan-400/80" title="Field-Weighted Citation Impact">
                            FWCI {paper.fwci!.toFixed(1)}
                        </span>
                    )}
                </div>
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

