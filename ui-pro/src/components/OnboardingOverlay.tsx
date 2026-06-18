"use client";

import { useState, useEffect, useCallback } from "react";
import { X, ChevronRight, ChevronLeft, Rocket, MessageSquare, Sparkles, LayoutDashboard, CheckCircle2 } from "lucide-react";

const STORAGE_KEY = "quasar_onboarded";

interface OnboardingStep {
    icon: React.ReactNode;
    title: string;
    description: string;
    highlight?: string; // CSS selector to spotlight (optional)
}

const STEPS: OnboardingStep[] = [
    {
        icon: <Rocket className="w-8 h-8" />,
        title: "Welcome to QUASAR",
        description:
            "Your AI-powered radio astronomy research assistant. QUASAR can search ALMA archives, find and analyze papers, process FITS data, check spectral line coverage, and much more.",
    },
    {
        icon: <MessageSquare className="w-8 h-8" />,
        title: "Ask Anything",
        description:
            "Type any astronomy question in the chat input below. QUASAR understands natural language — try \"Find ALMA observations of M87 in Band 6\" or \"What is the CO(2-1) rest frequency?\"",
    },
    {
        icon: <Sparkles className="w-8 h-8" />,
        title: "Try Example Queries",
        description:
            "Click one of the suggestion cards on the home screen to get started instantly. Each card sends a pre-built query that showcases QUASAR's capabilities.",
    },
    {
        icon: <LayoutDashboard className="w-8 h-8" />,
        title: "Your Research Hub",
        description:
            "The sidebar holds your conversation history, model selector, saved papers, and settings. You can switch AI models, upload personal documents, and connect external tools.",
    },
    {
        icon: <CheckCircle2 className="w-8 h-8" />,
        title: "You're All Set!",
        description:
            "Start by asking a question or clicking an example query. You can restart this tutorial anytime from Settings.",
    },
];

interface OnboardingOverlayProps {
    onComplete: () => void;
}

export function OnboardingOverlay({ onComplete }: OnboardingOverlayProps) {
    const [step, setStep] = useState(0);
    const [visible, setVisible] = useState(false);
    const [exiting, setExiting] = useState(false);

    useEffect(() => {
        // Small delay for mount animation
        const t = setTimeout(() => setVisible(true), 50);
        return () => clearTimeout(t);
    }, []);

    const finish = useCallback(() => {
        setExiting(true);
        setTimeout(() => {
            localStorage.setItem(STORAGE_KEY, "true");
            onComplete();
        }, 300);
    }, [onComplete]);

    const next = () => {
        if (step < STEPS.length - 1) {
            setStep((s) => s + 1);
        } else {
            finish();
        }
    };

    const prev = () => {
        if (step > 0) setStep((s) => s - 1);
    };

    const current = STEPS[step];

    return (
        <div
            className={`fixed inset-0 z-[9999] flex items-center justify-center transition-opacity duration-300
                ${visible && !exiting ? "opacity-100" : "opacity-0 pointer-events-none"}`}
        >
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/60 backdrop-blur-lg" onClick={finish} />

            {/* Card */}
            <div
                className={`glass-surface relative z-10 w-full max-w-md mx-4 rounded-3xl overflow-hidden
                    transition-all duration-300 ease-out
                    ${visible && !exiting ? "scale-100 translate-y-0" : "scale-95 translate-y-4"}`}
            >
                {/* Skip button */}
                <button
                    onClick={finish}
                    className="absolute top-4 right-4 p-2 rounded-xl transition-colors z-10"
                    style={{ color: 'var(--q-text-muted)' }}
                    title="Skip tutorial"
                >
                    <X className="w-4 h-4" />
                </button>

                {/* Content */}
                <div className="px-8 pt-10 pb-8 flex flex-col items-center text-center">
                    {/* Icon */}
                    <div className="w-16 h-16 rounded-2xl bg-primary/15 flex items-center justify-center mb-6 text-primary">
                        {current.icon}
                    </div>

                    {/* Title */}
                    <h2 className="text-xl font-bold mb-3 tracking-tight" style={{ color: 'var(--q-text)' }}>
                        {current.title}
                    </h2>

                    {/* Description */}
                    <p className="text-sm leading-relaxed max-w-sm" style={{ color: 'var(--q-text-secondary)' }}>
                        {current.description}
                    </p>
                </div>

                {/* Footer: dots + nav */}
                <div className="px-8 pb-8 flex items-center justify-between">
                    {/* Step dots */}
                    <div className="flex gap-1.5">
                        {STEPS.map((_, i) => (
                            <button
                                key={i}
                                onClick={() => setStep(i)}
                                className={`h-2 rounded-full transition-all duration-300 ${
                                    i === step
                                        ? "w-6 bg-primary"
                                        : i < step
                                        ? "w-2 bg-primary/40"
                                        : "w-2"
                                }`}
                                style={i > step ? { background: 'var(--q-text-muted)', opacity: 0.4 } : undefined}
                            />
                        ))}
                    </div>

                    {/* Nav buttons */}
                    <div className="flex items-center gap-2">
                        {step > 0 && (
                            <button
                                onClick={prev}
                                className="flex items-center gap-1 px-3 py-2 rounded-xl text-sm font-medium transition-colors"
                                style={{ color: 'var(--q-text-secondary)' }}
                            >
                                <ChevronLeft className="w-4 h-4" />
                                Back
                            </button>
                        )}
                        <button
                            onClick={next}
                            className="flex items-center gap-1.5 px-5 py-2.5 rounded-xl text-sm font-semibold bg-primary hover:bg-primary/90 text-primary-dark transition-colors shadow-lg shadow-primary/20"
                        >
                            {step === STEPS.length - 1 ? "Get Started" : "Next"}
                            {step < STEPS.length - 1 && <ChevronRight className="w-4 h-4" />}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

/**
 * Check if the user has completed onboarding.
 * Call this from page.tsx to decide whether to show the overlay.
 */
export function useShowOnboarding(): [boolean, () => void] {
    const [show, setShow] = useState(false);

    useEffect(() => {
        const done = localStorage.getItem(STORAGE_KEY);
        if (!done) {
            // localStorage is only available after the component mounts.
            // eslint-disable-next-line react-hooks/set-state-in-effect
            setShow(true);
        }
    }, []);

    const dismiss = useCallback(() => {
        setShow(false);
        localStorage.setItem(STORAGE_KEY, "true");
    }, []);

    return [show, dismiss];
}

/**
 * Reset onboarding state so the tutorial shows again.
 * Called from Settings > "Restart Tutorial".
 */
export function resetOnboarding() {
    localStorage.removeItem(STORAGE_KEY);
}
