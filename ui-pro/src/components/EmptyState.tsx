"use client";

import { useEffect, useState } from "react";
import { Radio, FileText, HelpCircle, BarChart3 } from "lucide-react";

interface Category {
    key: string;
    icon: React.ReactNode;
    title: string;
    example: string;
}

// Orbita empty state: example prompts per the QUASAR design handoff.
const ICON_CLASS = "size-4";
const ICON_STROKE = 1.75;

const CATEGORIES: Category[] = [
    {
        key: "archive",
        icon: <Radio className={ICON_CLASS} strokeWidth={ICON_STROKE} />,
        title: "Search the archive",
        example: "Find ALMA observations of Sz65 in Band 6.",
    },
    {
        key: "lit",
        icon: <FileText className={ICON_CLASS} strokeWidth={ICON_STROKE} />,
        title: "Search the literature",
        example: "Recent papers on protoplanetary disk substructure.",
    },
    {
        key: "policy",
        icon: <HelpCircle className={ICON_CLASS} strokeWidth={ICON_STROKE} />,
        title: "Policy & guidance",
        example: "What is the current ALMA proprietary period?",
    },
    {
        key: "spectral",
        icon: <BarChart3 className={ICON_CLASS} strokeWidth={ICON_STROKE} />,
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
        <div className="hide-scrollbar flex-1 w-full overflow-y-auto px-4 py-4 md:p-8 md:pb-6">
            {/* min-h-full (not h-full) keeps justify-center safe: once the content
                outgrows the viewport the box grows instead of centering overflow
                out of scroll reach. */}
            <div className="mx-auto flex min-h-full w-full max-w-[var(--q-empty-state-width)] flex-col items-center">
                {/* Centered content — grows to fill the viewport so the disclaimer can sit at the bottom */}
                <div className="flex w-full flex-1 flex-col items-center justify-center">
                    {/* Hero */}
                    <div className="mb-6 flex flex-col items-center justify-center text-center md:mb-7">
                        <div className="mb-4 flex size-11 items-center justify-center rounded-full border border-[var(--q-border)] bg-[var(--q-card)] shadow-[var(--q-glass-shadow)]">
                            <img src="/quasar_logo.png" alt="Quasar" className="size-8 object-contain" />
                        </div>
                        <h1 className="text-[22px] font-medium tracking-tight text-[var(--q-text)] md:text-[26px]">
                            What are you researching today?
                        </h1>
                        <p className="mt-1.5 max-w-[26rem] text-[13px] text-[var(--q-text-muted)] md:max-w-lg md:text-[14px]">
                            Observations, archives, literature. Ask in plain language.
                        </p>
                    </div>

                    {/* Hero composer (centered) */}
                    {composer && (
                        <div className="mb-6 w-full md:mb-7">
                            {composer}
                        </div>
                    )}

                    {/* Starting points */}
                    <div className="w-full max-w-[var(--q-suggestion-grid-width)]">
                        {showStartingPoints && (
                            <>
                                <div className="mb-2.5 px-1 text-[13px] text-[var(--q-text-faint)]">
                                    Try a starting point
                                </div>
                                <div className="grid grid-cols-2 gap-2 md:gap-2.5">
                                    {CATEGORIES.map((c, i) => (
                                        <button
                                            key={c.key}
                                            onClick={() => onSuggestionClick(c.example)}
                                            style={{ animationDelay: `${i * 40}ms` }}
                                            className="q-rise group flex flex-col gap-2 rounded-2xl border border-[var(--q-border)] bg-[var(--q-card)] p-3 text-left shadow-[var(--q-glass-shadow)] transition-colors hover:border-[var(--q-border-strong)] md:flex-row md:items-start md:gap-3 md:p-3.5"
                                        >
                                            <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-[var(--q-canvas)] text-[var(--q-text-muted)] transition-colors group-hover:text-[var(--q-text)]">
                                                {c.icon}
                                            </div>
                                            <div className="min-w-0">
                                                <h3 className="mb-0.5 text-[13px] font-medium text-[var(--q-text)]">
                                                    {c.title}
                                                </h3>
                                                <p className="text-[12px] leading-[1.45] text-[var(--q-text-muted)]">
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
                    <p className="pt-6 text-center text-[12px] text-[var(--q-text-faint)]">
                        Quasar may produce inaccurate information. · Accepts images, PDFs, FITS, CSV
                        {hitCount !== null && (
                            <> · <span className="tabular-nums">{hitCount.toLocaleString()}</span> visits</>
                        )}
                    </p>
                )}
            </div>
        </div>
    );
}
