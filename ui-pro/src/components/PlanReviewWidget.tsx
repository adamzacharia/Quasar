"use client";

import React, { useState, useRef, useEffect } from "react";
import {
    CheckCircle2,
    MessageSquare,
    Send,
    ArrowRight,
    Loader2,
    RefreshCw,
    GitFork,
} from "lucide-react";

// ── Agent type icons (matching TaskExecutionWidget) ────────────────────────

function IconRadioDish({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 11 Q8 2 14 11" />
            <line x1="8" y1="6.5" x2="8" y2="14" />
            <line x1="5" y1="14" x2="11" y2="14" />
            <circle cx="8" cy="6.5" r="1" fill="currentColor" stroke="none" />
            <path d="M5.5 4 Q8 1.5 10.5 4" strokeWidth="1" opacity="0.6" />
        </svg>
    );
}

function IconPaperStar({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="2" width="9" height="12" rx="1.5" />
            <line x1="4.5" y1="6" x2="8.5" y2="6" />
            <line x1="4.5" y1="8.5" x2="7" y2="8.5" />
            <path d="M12.5 5 l0.6 1.3 1.4 0.2 -1 1 0.2 1.4 -1.2-0.7 -1.2 0.7 0.2-1.4 -1-1 1.4-0.2z" fill="currentColor" stroke="none" />
        </svg>
    );
}

function IconOrbitRings({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="8" cy="8" r="1.5" fill="currentColor" stroke="none" />
            <ellipse cx="8" cy="8" rx="5" ry="2.5" transform="rotate(-30 8 8)" />
            <ellipse cx="8" cy="8" rx="5" ry="2.5" transform="rotate(60 8 8)" />
            <circle cx="11.3" cy="6.5" r="0.9" fill="currentColor" stroke="none" />
        </svg>
    );
}

function IconSpectrum({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            <line x1="1" y1="8" x2="3" y2="8" />
            <path d="M3 8 Q4.5 3 5.5 8 Q6.5 13 8 8 Q9.5 3 10.5 8 Q11.5 13 13 8" />
            <line x1="13" y1="8" x2="15" y2="8" />
            <line x1="8" y1="3" x2="8" y2="5" strokeWidth="1.8" opacity="0.7" />
        </svg>
    );
}

function IconGlobeNetwork({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="8" cy="8" r="6" />
            <path d="M2.5 8 Q8 5 13.5 8" />
            <path d="M2.5 8 Q8 11 13.5 8" />
            <path d="M8 2 Q11 8 8 14" />
            <path d="M8 2 Q5 8 8 14" />
        </svg>
    );
}

function IconConstellation({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="3" cy="4" r="1.2" fill="currentColor" stroke="none" />
            <circle cx="13" cy="4" r="1.2" fill="currentColor" stroke="none" />
            <circle cx="8" cy="13" r="1.5" fill="currentColor" stroke="none" />
            <line x1="3" y1="4" x2="8" y2="13" />
            <line x1="13" y1="4" x2="8" y2="13" />
            <line x1="3" y1="4" x2="13" y2="4" />
        </svg>
    );
}

function IconStar({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            <path d="M8 2 l1.5 3.5 3.7 0.5 -2.7 2.6 0.6 3.7 -3.1-1.7 -3.1 1.7 0.6-3.7 -2.7-2.6 3.7-0.5z" />
        </svg>
    );
}

function IconCompute({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="3" width="12" height="9" rx="1.5" />
            <line x1="5" y1="14" x2="11" y2="14" />
            <line x1="8" y1="12" x2="8" y2="14" />
            <path d="M5 6.5 l1.5 2 3-3.5" strokeWidth="1.6" />
        </svg>
    );
}

type AgentType = "archive" | "literature" | "analysis" | "viz" | "web" | "compute" | "synthesis" | "general";

const AGENT_CONFIG: Record<AgentType, { Icon: (props: { className?: string }) => React.ReactElement; label: string }> = {
    archive:    { Icon: IconRadioDish,     label: "Archive Search" },
    literature: { Icon: IconPaperStar,     label: "Literature Review" },
    analysis:   { Icon: IconOrbitRings,    label: "Data Analysis" },
    viz:        { Icon: IconSpectrum,      label: "Visualization" },
    web:        { Icon: IconGlobeNetwork,  label: "Web Search" },
    compute:    { Icon: IconCompute,       label: "Computation" },
    synthesis:  { Icon: IconConstellation, label: "Synthesis" },
    general:    { Icon: IconStar,          label: "General Task" },
};

function AgentIcon({ agentType, className = "" }: { agentType: string; className?: string }) {
    const cfg = AGENT_CONFIG[agentType as AgentType] ?? AGENT_CONFIG.general;
    return <cfg.Icon className={className} />;
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface PlanReviewData {
    conversationId: string;
    title: string;
    subtasks: {
        id: string;
        description: string;
        agentType: string;
        dependsOn: string[];
    }[];
    reasoning: string;
    iteration: number;
    maxIterations: number;
    query: string;
}

interface PlanReviewWidgetProps {
    plan: PlanReviewData;
    onApprove: () => void;
    onFeedback: (feedback: string) => void;
    isSubmitting: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────

export function PlanReviewWidget({ plan, onApprove, onFeedback, isSubmitting }: PlanReviewWidgetProps) {
    const [showFeedback, setShowFeedback] = useState(false);
    const [feedbackText, setFeedbackText] = useState("");
    const inputRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        if (showFeedback && inputRef.current) {
            inputRef.current.focus();
        }
    }, [showFeedback]);

    const handleSubmitFeedback = () => {
        if (feedbackText.trim()) {
            onFeedback(feedbackText.trim());
            setFeedbackText("");
            setShowFeedback(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmitFeedback();
        }
    };

    // Count parallel rounds (groups of tasks with no dependencies on each other)
    const independentTasks = plan.subtasks.filter(t => t.dependsOn.length === 0).length;

    return (
        <div
            className="rounded-2xl overflow-hidden max-w-2xl mx-auto my-3 backdrop-blur-sm animate-in fade-in slide-in-from-bottom-2 duration-300"
            style={{ background: "var(--q-glass-bg)", border: "1px solid var(--q-glass-border)" }}
        >
            {/* ── Header ─────────────────────────────────────────────── */}
            <div
                className="flex items-center justify-between px-4 py-3"
                style={{ borderBottom: "1px solid var(--q-border)" }}
            >
                <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 flex items-center justify-center bg-primary/20 rounded-lg">
                        <IconConstellation className="w-5 h-5 text-primary" />
                    </div>
                    <div className="flex flex-col">
                        <span className="text-sm font-semibold" style={{ color: "var(--q-text)" }}>
                            Execution Plan Review
                        </span>
                        <span className="text-[10px] font-mono" style={{ color: "var(--q-text-muted)" }}>
                            {plan.subtasks.length} tasks • {independentTasks} can run in parallel
                            {plan.iteration > 1 && ` • Revision ${plan.iteration}`}
                        </span>
                    </div>
                </div>
                {plan.iteration > 1 && (
                    <span
                        className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                        style={{
                            background: "var(--q-primary-muted, rgba(236,72,153,0.15))",
                            color: "var(--q-primary, #ec4899)",
                        }}
                    >
                        <RefreshCw className="w-3 h-3 inline mr-1" />
                        Rev {plan.iteration}/{plan.maxIterations}
                    </span>
                )}
            </div>

            {/* ── Task List ──────────────────────────────────────────── */}
            <div className="px-3 py-2 space-y-1 max-h-[320px] overflow-y-auto custom-scrollbar">
                {plan.subtasks.map((task, idx) => {
                    const agentCfg = AGENT_CONFIG[task.agentType as AgentType] ?? AGENT_CONFIG.general;
                    const hasDeps = task.dependsOn.length > 0;

                    return (
                        <div
                            key={task.id}
                            className="flex items-start gap-2.5 px-3 py-2 rounded-lg transition-colors hover:bg-white/5"
                        >
                            {/* Step number */}
                            <span
                                className="text-[10px] font-mono w-5 h-5 flex items-center justify-center rounded-full shrink-0 mt-0.5"
                                style={{
                                    background: "var(--q-border)",
                                    color: "var(--q-text-muted)",
                                }}
                            >
                                {idx + 1}
                            </span>

                            {/* Agent icon */}
                            <AgentIcon
                                agentType={task.agentType}
                                className="w-4 h-4 shrink-0 mt-0.5 text-primary/70"
                            />

                            {/* Task description + metadata */}
                            <div className="flex-1 min-w-0">
                                <span className="text-xs leading-relaxed" style={{ color: "var(--q-text-secondary)" }}>
                                    {task.description}
                                </span>
                                <div className="flex items-center gap-2 mt-0.5">
                                    <span
                                        className="text-[9px] font-mono px-1.5 py-0.5 rounded"
                                        style={{
                                            background: "var(--q-border)",
                                            color: "var(--q-text-muted)",
                                        }}
                                    >
                                        {agentCfg.label}
                                    </span>
                                    {hasDeps && (
                                        <span className="flex items-center gap-0.5 text-[9px] font-mono" style={{ color: "var(--q-text-muted)" }}>
                                            <GitFork className="w-3 h-3" />
                                            depends on {task.dependsOn.join(", ")}
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* ── Feedback Input ──────────────────────────────────────── */}
            {showFeedback && (
                <div
                    className="px-4 py-3 animate-in fade-in slide-in-from-top-1 duration-200"
                    style={{ borderTop: "1px solid var(--q-border)" }}
                >
                    <label className="text-[10px] font-semibold uppercase tracking-wider mb-1.5 block" style={{ color: "var(--q-text-muted)" }}>
                        Describe your changes
                    </label>
                    <div className="flex gap-2">
                        <textarea
                            ref={inputRef}
                            value={feedbackText}
                            onChange={(e) => setFeedbackText(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="e.g., &quot;Step 2 should search Band 7 instead&quot; or &quot;Add a literature search for recent M87 papers&quot;"
                            rows={2}
                            className="flex-1 text-xs px-3 py-2 rounded-lg resize-none focus:outline-none focus:ring-1 focus:ring-primary/50"
                            style={{
                                background: "var(--q-input-bg, rgba(255,255,255,0.05))",
                                border: "1px solid var(--q-border)",
                                color: "var(--q-text)",
                            }}
                            disabled={isSubmitting}
                        />
                        <button
                            onClick={handleSubmitFeedback}
                            disabled={!feedbackText.trim() || isSubmitting}
                            className="self-end px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200 flex items-center gap-1.5 disabled:opacity-40"
                            style={{
                                background: "var(--q-primary, #ec4899)",
                                color: "#fff",
                            }}
                        >
                            {isSubmitting ? (
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                                <Send className="w-3.5 h-3.5" />
                            )}
                            Revise
                        </button>
                    </div>
                </div>
            )}

            {/* ── Action Buttons ──────────────────────────────────────── */}
            <div
                className="flex items-center gap-2 px-4 py-3"
                style={{ borderTop: "1px solid var(--q-border)" }}
            >
                <button
                    onClick={onApprove}
                    disabled={isSubmitting}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-200 disabled:opacity-40 hover:brightness-110"
                    style={{
                        background: "linear-gradient(135deg, #10b981, #059669)",
                        color: "#fff",
                        boxShadow: "0 2px 8px rgba(16,185,129,0.25)",
                    }}
                >
                    {isSubmitting ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                        <CheckCircle2 className="w-3.5 h-3.5" />
                    )}
                    Approve & Execute
                    <ArrowRight className="w-3 h-3 ml-0.5" />
                </button>

                <button
                    onClick={() => setShowFeedback(!showFeedback)}
                    disabled={isSubmitting}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium transition-all duration-200 disabled:opacity-40"
                    style={{
                        background: "var(--q-border)",
                        color: "var(--q-text-secondary)",
                    }}
                >
                    <MessageSquare className="w-3.5 h-3.5" />
                    {showFeedback ? "Cancel" : "Suggest Changes"}
                </button>

                {plan.iteration > 1 && (
                    <span className="ml-auto text-[10px] font-mono" style={{ color: "var(--q-text-muted)" }}>
                        {plan.maxIterations - plan.iteration} revisions remaining
                    </span>
                )}
            </div>
        </div>
    );
}
