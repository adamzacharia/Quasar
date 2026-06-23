"use client";

import { useEffect, useState } from "react";
import { Radio, FileText, HelpCircle, BarChart3 } from "lucide-react";

interface Category {
    key: string;
    icon: React.ReactNode;
    title: string;
    examples: string[];
}

// Aurora empty state — rotating example prompts per the QUASAR design handoff.
const CATEGORIES: Category[] = [
    {
        key: "archive",
        icon: <Radio className="w-[19px] h-[19px]" />,
        title: "Search the archive",
        examples: [
            "Find ALMA observations of Sz65 in Band 6.",
            "VLA archive data for the jet in 3C 273.",
            "ALMA Cycle 9 observations of TW Hya.",
            "Pull Band 7 continuum for HL Tau.",
        ],
    },
    {
        key: "lit",
        icon: <FileText className="w-[19px] h-[19px]" />,
        title: "Search the literature",
        examples: [
            "Recent papers on protoplanetary disk substructure.",
            "Papers citing the EHT M87* results.",
            "2025 reviews of dust trapping in disks.",
            "Latest work on disk wind signatures.",
        ],
    },
    {
        key: "policy",
        icon: <HelpCircle className="w-[19px] h-[19px]" />,
        title: "Policy & guidance",
        examples: [
            "What is the current ALMA proprietary period?",
            "How do I apply for ALMA DDT time?",
            "Band 6 sensitivity for a 1-hour track?",
            "How is ALMA data weighted by default?",
        ],
    },
    {
        key: "spectral",
        icon: <BarChart3 className="w-[19px] h-[19px]" />,
        title: "Spectral coverage",
        examples: [
            "Check CO(2-1) line coverage for M87.",
            "Does Band 7 cover HCO+ at z = 0.03?",
            "Redshifted [CII] 158um at z = 6.1.",
            "Which N2H+ lines fall in Band 3?",
        ],
    },
];

interface EmptyStateProps {
    onSuggestionClick: (prompt: string) => void;
    /** The hero composer (a ChatInput in "hero" variant), placed between the subtitle and the cards. */
    composer?: React.ReactNode;
    /** Visit counter, shown in the footer disclaimer (owned by ChatArea). */
    hitCount?: number | null;
}

export function EmptyState({ onSuggestionClick, composer, hitCount = null }: EmptyStateProps) {
    const [tick, setTick] = useState(0);

    // Cycle the example prompts, unless the user prefers reduced motion.
    useEffect(() => {
        if (typeof window !== "undefined"
            && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
            return;
        }
        const id = setInterval(() => setTick((t) => t + 1), 4400);
        return () => clearInterval(id);
    }, []);

    return (
        <div className="hide-scrollbar flex-1 w-full overflow-y-auto px-3 pt-8 pb-3 md:p-8 md:pb-6">
            <div className="mx-auto flex min-h-full w-full max-w-[var(--q-empty-state-width)] flex-col items-center justify-start md:justify-center">
                {/* Hero */}
                <div className="flex flex-col items-center justify-center text-center mb-6 md:mb-7">
                    <div className="relative mb-3 md:mb-4">
                        <div className="absolute inset-[-22px] rounded-full bg-primary/30 blur-2xl" aria-hidden="true" />
                        <img src="/quasar_logo.png" alt="Quasar" className="relative size-14 md:size-20 object-contain" />
                    </div>
                    <h1 className="text-2xl md:text-[31px] font-semibold tracking-tight" style={{ color: "var(--q-text)" }}>
                        What are you researching today?
                    </h1>
                    <p className="text-sm md:text-[15px] mt-2 max-w-[26rem] md:max-w-lg" style={{ color: "var(--q-text-secondary)" }}>
                        Observations, archives, literature — ask in plain language.
                    </p>
                </div>

                {/* Hero composer (centered) */}
                {composer && (
                    <div className="w-full mb-7 md:mb-8">
                        {composer}
                    </div>
                )}

                {/* Starting points */}
                <div className="w-full max-w-[var(--q-suggestion-grid-width)]">
                    <div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.12em]" style={{ color: "var(--q-text-muted)" }}>
                        Try a starting point
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 md:gap-3.5">
                        {CATEGORIES.map((c) => {
                            const example = c.examples[tick % c.examples.length];
                            return (
                                <button
                                    key={c.key}
                                    onClick={() => onSuggestionClick(example)}
                                    className="glass-card rounded-2xl md:rounded-3xl p-3.5 md:p-[18px] text-left group flex items-start gap-3.5 md:gap-4"
                                >
                                    <div className="shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-primary/15 text-primary transition-colors group-hover:bg-primary/25">
                                        {c.icon}
                                    </div>
                                    <div className="min-w-0">
                                        <h3 className="font-semibold text-sm md:text-[15px] mb-1 group-hover:text-primary transition-colors" style={{ color: "var(--q-text)" }}>
                                            {c.title}
                                        </h3>
                                        <p
                                            key={tick}
                                            className="text-xs md:text-[13px] leading-relaxed animate-qfade min-h-[32px] md:min-h-[39px]"
                                            style={{ color: "var(--q-text-secondary)" }}
                                        >
                                            {example}
                                        </p>
                                    </div>
                                </button>
                            );
                        })}
                    </div>

                    <p className="mt-7 text-center text-[10px]" style={{ color: "var(--q-text-faint)" }}>
                        QUASAR may produce inaccurate information. · Accepts images, PDFs, FITS, CSV
                        {hitCount !== null && (
                            <> · <span className="font-mono tabular-nums" style={{ color: "var(--q-mono-accent)" }}>{hitCount.toLocaleString()}</span> visits</>
                        )}
                    </p>
                </div>
            </div>
        </div>
    );
}
