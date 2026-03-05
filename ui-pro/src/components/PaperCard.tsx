"use client";

import { ArrowRight, Quote } from "lucide-react";
import type { Paper } from "@/lib/types";

const TYPE_STYLES: Record<string, string> = {
    journal: "bg-emerald-500/10 text-emerald-400",
    arxiv: "bg-orange-500/10 text-orange-400",
    radio: "bg-primary/10 text-primary",
};

interface PaperCardProps { paper: Paper; }

export function PaperCard({ paper }: PaperCardProps) {
    const typeLabel = paper.type === "journal" ? "Journal" : paper.type === "arxiv" ? "ArXiv" : "Radio";
    return (
        <div className="bg-sidebar-dark p-4 rounded-2xl border border-slate-700/50 hover:border-emerald-500/50 transition-colors group cursor-pointer flex flex-col justify-between h-full">
            <div className="space-y-2">
                <div className="flex items-start justify-between">
                    <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${TYPE_STYLES[paper.type] || TYPE_STYLES.journal}`}>{typeLabel}</span>
                    <span className="text-xs text-slate-400">{paper.year}</span>
                </div>
                <h4 className="text-sm font-bold text-white line-clamp-2 group-hover:text-primary transition-colors">{paper.title}</h4>
                <p className="text-xs text-slate-400 line-clamp-1">{paper.authors}</p>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-700/50 flex items-center justify-between text-xs">
                <div className="flex items-center gap-1 text-slate-500"><Quote className="w-3.5 h-3.5" /><span>{paper.citationCount} Citations</span></div>
                <ArrowRight className="w-4 h-4 text-slate-400 group-hover:translate-x-1 transition-transform" />
            </div>
        </div>
    );
}
