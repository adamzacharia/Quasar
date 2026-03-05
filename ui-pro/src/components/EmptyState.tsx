"use client";

import { Satellite, FileText, HelpCircle, AudioWaveform } from "lucide-react";

interface Suggestion {
    icon: React.ReactNode;
    iconBg: string;
    iconHoverBg: string;
    title: string;
    description: string;
    prompt: string;
}

const SUGGESTIONS: Suggestion[] = [
    { icon: <Satellite className="w-5 h-5" />, iconBg: "bg-primary/10 text-primary", iconHoverBg: "group-hover:bg-primary group-hover:text-primary-dark", title: "Find ALMA observations", description: "Look for Sz65 in Band 6 data archives.", prompt: "@archive Find ALMA observations of Sz65 in Band 6" },
    { icon: <FileText className="w-5 h-5" />, iconBg: "bg-indigo-500/10 text-indigo-400", iconHoverBg: "group-hover:bg-indigo-500 group-hover:text-white", title: "Search for papers", description: "Find latest research on protoplanetary disks.", prompt: "@paper Find recent papers on protoplanetary disks" },
    { icon: <HelpCircle className="w-5 h-5" />, iconBg: "bg-emerald-500/10 text-emerald-400", iconHoverBg: "group-hover:bg-emerald-500 group-hover:text-white", title: "Proprietary Period", description: "What is the current ALMA data policy?", prompt: "What is the ALMA proprietary period?" },
    { icon: <AudioWaveform className="w-5 h-5" />, iconBg: "bg-orange-500/10 text-orange-400", iconHoverBg: "group-hover:bg-orange-500 group-hover:text-white", title: "Line Coverage Check", description: "Check CO(2-1) line coverage for M87.", prompt: "@archive Check CO(2-1) line coverage for M87" },
];

interface EmptyStateProps { onSuggestionClick: (prompt: string) => void; }

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
    return (
        <div className="flex-1 flex flex-col items-center justify-center p-4 md:p-8 w-full max-w-5xl mx-auto">
            <div className="flex flex-col items-center justify-center text-center mb-12">
                <div className="relative mb-6">
                    <div className="absolute inset-0 bg-primary/20 rounded-full blur-3xl opacity-50" />
                    <div className="relative size-24 rounded-3xl bg-gradient-to-br from-primary to-accent-purple flex items-center justify-center logo-glow shadow-2xl">
                        <span className="text-white font-bold text-6xl">Q</span>
                    </div>
                </div>
                <h1 className="text-4xl md:text-5xl font-bold mb-3 tracking-tight text-white">
                    <span className="text-gradient">QUASAR</span> Assistant
                </h1>
                <p className="text-slate-400 text-lg font-light max-w-md mx-auto leading-relaxed">
                    Your AI assistant for radio astronomy research, data analysis, and literature review.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-3xl">
                {SUGGESTIONS.map((s, i) => (
                    <button key={i} onClick={() => onSuggestionClick(s.prompt)} className="glass-card p-5 rounded-3xl text-left group flex items-start gap-4">
                        <div className={`p-2.5 rounded-full transition-colors ${s.iconBg} ${s.iconHoverBg}`}>{s.icon}</div>
                        <div>
                            <h3 className="text-white font-medium text-sm mb-1 group-hover:text-primary transition-colors">{s.title}</h3>
                            <p className="text-slate-400 text-xs">{s.description}</p>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
}
