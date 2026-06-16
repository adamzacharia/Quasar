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
    { icon: <Satellite className="w-5 h-5" />, iconBg: "bg-primary/10 text-primary", iconHoverBg: "group-hover:bg-primary group-hover:text-primary-dark", title: "Find ALMA observations", description: "Look for Sz65 in Band 6 data archives.", prompt: "Find ALMA observations of Sz65 in Band 6" },
    { icon: <FileText className="w-5 h-5" />, iconBg: "bg-indigo-500/10 text-indigo-400", iconHoverBg: "group-hover:bg-indigo-500 group-hover:text-white", title: "Search for papers", description: "Find latest research on protoplanetary disks.", prompt: "Find recent papers on protoplanetary disks" },
    { icon: <HelpCircle className="w-5 h-5" />, iconBg: "bg-emerald-500/10 text-emerald-400", iconHoverBg: "group-hover:bg-emerald-500 group-hover:text-white", title: "Proprietary Period", description: "What is the current ALMA data policy?", prompt: "What is the ALMA proprietary period?" },
    { icon: <AudioWaveform className="w-5 h-5" />, iconBg: "bg-orange-500/10 text-orange-400", iconHoverBg: "group-hover:bg-orange-500 group-hover:text-white", title: "Line Coverage Check", description: "Check CO(2-1) line coverage for M87.", prompt: "Check CO(2-1) line coverage for M87" },
];

interface EmptyStateProps { onSuggestionClick: (prompt: string) => void; }

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
    return (
        <div className="hide-scrollbar flex-1 w-full overflow-y-auto px-3 pt-8 pb-3 md:p-8 md:pb-6">
            <div className="mx-auto flex min-h-full w-full max-w-[var(--q-empty-state-width)] flex-col items-center justify-start md:justify-center">
                <div className="flex flex-col items-center justify-center text-center mb-4 md:mb-6">
                    <div className="relative mb-1">
                        <div className="absolute inset-0 bg-primary/20 rounded-full blur-3xl opacity-50" />
                        <div className="relative size-14 md:size-32 flex items-center justify-center">
                            <img src="/quasar_logo.png" alt="Quasar" className="size-14 md:size-32 object-contain drop-shadow-[0_0_15px_rgba(45,30,209,0.5)]" />
                        </div>
                    </div>
                    <h1 className="text-2xl md:text-4xl font-bold mb-1 md:mb-2 tracking-tight" style={{ color: 'var(--q-text)' }}>
                        <span className="text-gradient">QUASAR</span>
                    </h1>
                    <p className="text-sm md:text-base font-light max-w-[22rem] md:max-w-md mx-auto leading-relaxed" style={{ color: 'var(--q-text-secondary)' }}>
                        Your AI assistant for radio astronomy research, data analysis, and literature review.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 md:gap-4 w-full max-w-[var(--q-suggestion-grid-width)]">
                    {SUGGESTIONS.map((s, i) => (
                        <button key={i} onClick={() => onSuggestionClick(s.prompt)}
                            className="glass-card min-h-[76px] p-3 md:p-5 rounded-2xl md:rounded-3xl text-left group flex items-start gap-3 md:gap-4">
                            <div className={`shrink-0 p-2 md:p-2.5 rounded-full transition-colors ${s.iconBg} ${s.iconHoverBg}`}>{s.icon}</div>
                            <div className="min-w-0">
                                <h3 className="font-medium text-sm mb-0.5 md:mb-1 group-hover:text-primary transition-colors" style={{ color: 'var(--q-text)' }}>{s.title}</h3>
                                <p className="text-xs" style={{ color: 'var(--q-text-secondary)' }}>{s.description}</p>
                            </div>
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}
