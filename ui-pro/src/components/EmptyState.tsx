"use client";

import { useEffect, useState } from "react";
import { Radio, FileText, HelpCircle, BarChart3 } from "lucide-react";

interface Category {
    key: string;
    icon: React.ReactNode;
    title: string;
    example: string;
}

// Aurora empty state — example prompts per the QUASAR design handoff.
const ICON_CLASS = "size-[15px] md:size-[19px]";

const CATEGORIES: Category[] = [
    {
        key: "archive",
        icon: <Radio className={ICON_CLASS} />,
        title: "Search the archive",
        example: "Find ALMA observations of Sz65 in Band 6.",
    },
    {
        key: "lit",
        icon: <FileText className={ICON_CLASS} />,
        title: "Search the literature",
        example: "Recent papers on protoplanetary disk substructure.",
    },
    {
        key: "policy",
        icon: <HelpCircle className={ICON_CLASS} />,
        title: "Policy & guidance",
        example: "What is the current ALMA proprietary period?",
    },
    {
        key: "spectral",
        icon: <BarChart3 className={ICON_CLASS} />,
        title: "Spectral coverage",
        example: "Check CO(2-1) line coverage for M87.",
    },
];

const STORAGE_KEY = "quasar_starting_points_day";

/** The local calendar day, e.g. "2026-7-10" — the day boundary the user perceives. */
function todayKey(): string {
    const d = new Date();
    return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
}

interface EmptyStateProps {
    onSuggestionClick: (prompt: string) => void;
    /** The hero composer (a ChatInput in "hero" variant), placed between the subtitle and the cards. */
    composer?: React.ReactNode;
    /** Visit counter, shown in the footer disclaimer (owned by ChatArea). */
    hitCount?: number | null;
    /** False on phones, where the docked composer below prints its own disclaimer. */
    showDisclaimer?: boolean;
}

export function EmptyState({ onSuggestionClick, composer, hitCount = null, showDisclaimer = true }: EmptyStateProps) {
    const [showStartingPoints, setShowStartingPoints] = useState(false);

    // Starting points are shown on the first chat opened each day, then stay hidden.
    useEffect(() => {
        const today = todayKey();
        if (localStorage.getItem(STORAGE_KEY) === today) return;
        localStorage.setItem(STORAGE_KEY, today);
        // localStorage is only available after the component mounts.
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setShowStartingPoints(true);
    }, []);

    return (
        <div className="hide-scrollbar flex-1 w-full overflow-y-auto px-3 py-4 md:p-8 md:pb-6">
            {/* min-h-full (not h-full) keeps justify-center safe: once the content
                outgrows the viewport the box grows instead of centering overflow
                out of scroll reach. */}
            <div className="mx-auto flex min-h-full w-full max-w-[var(--q-empty-state-width)] flex-col items-center">
                {/* Centered content — grows to fill the viewport so the disclaimer can sit at the bottom */}
                <div className="flex w-full flex-1 flex-col items-center justify-center">
                    {/* Hero */}
                    <div className="flex flex-col items-center justify-center text-center mb-6 md:mb-7">
                        <div className="relative mb-3 md:mb-4">
                            <div className="absolute inset-[-18px] md:inset-[-22px] rounded-full bg-primary/30 blur-2xl" aria-hidden="true" />
                            <img src="/quasar_logo.png" alt="Quasar" className="relative size-[62px] md:size-20 object-contain" />
                        </div>
                        <h1 className="text-[23px] md:text-[31px] font-semibold tracking-tight" style={{ color: "var(--q-text)" }}>
                            What are you researching today?
                        </h1>
                        <p className="text-[12.5px] md:text-[15px] mt-1.5 md:mt-2 max-w-[26rem] md:max-w-lg" style={{ color: "var(--q-text-secondary)" }}>
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
                        {showStartingPoints && (
                            <>
                                <div className="mb-2.5 md:mb-3 text-[10px] md:text-[11px] font-semibold uppercase tracking-[0.12em]" style={{ color: "var(--q-text-muted)" }}>
                                    Try a starting point
                                </div>
                                <div className="grid grid-cols-2 gap-[10px] md:gap-3.5">
                                    {CATEGORIES.map((c) => (
                                        <button
                                            key={c.key}
                                            onClick={() => onSuggestionClick(c.example)}
                                            className="glass-card rounded-[16px] md:rounded-3xl p-[13px] md:p-[18px] text-left group flex flex-col gap-2 md:flex-row md:items-start md:gap-4"
                                        >
                                            <div className="shrink-0 size-[30px] md:size-10 rounded-full flex items-center justify-center bg-primary/15 text-primary transition-colors group-hover:bg-primary/25">
                                                {c.icon}
                                            </div>
                                            <div className="min-w-0">
                                                <h3 className="font-semibold text-[12.5px] md:text-[15px] mb-0.5 md:mb-1 group-hover:text-primary transition-colors" style={{ color: "var(--q-text)" }}>
                                                    {c.title}
                                                </h3>
                                                <p className="text-[10.5px] md:text-[13px] leading-[1.45] md:leading-relaxed" style={{ color: "var(--q-text-secondary)" }}>
                                                    {c.example}
                                                </p>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </>
                        )}
                    </div>
                </div>

                {/* Disclaimer — pinned to the bottom of the page below the centered content */}
                {showDisclaimer && (
                    <p className="pt-6 text-center text-[10px]" style={{ color: "var(--q-text-faint)" }}>
                        QUASAR may produce inaccurate information. · Accepts images, PDFs, FITS, CSV
                        {hitCount !== null && (
                            <> · <span className="font-mono tabular-nums" style={{ color: "var(--q-mono-accent)" }}>{hitCount.toLocaleString()}</span> visits</>
                        )}
                    </p>
                )}
            </div>
        </div>
    );
}
