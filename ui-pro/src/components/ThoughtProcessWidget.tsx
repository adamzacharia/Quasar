"use client";

import { ChevronDown, CheckCircle2, Brain, Loader2 } from "lucide-react";
import { useState, useEffect, type ReactNode } from "react";

export interface ThoughtStep {
    text: string;
    status: "completed" | "running" | "error";
    isCode?: boolean;
}

interface ThoughtProcessWidgetProps {
    title?: string;
    status: "running" | "completed" | "error";
    steps: ThoughtStep[];
    forceCollapsed?: boolean;
    children?: ReactNode;
    startTime?: Date | string | number;
    duration?: number;
}

export function ThoughtProcessWidget({
    title = "Thinking",
    status,
    steps,
    forceCollapsed,
    children,
    startTime,
    duration
}: ThoughtProcessWidgetProps) {
    const [seconds, setSeconds] = useState(() => {
        if (status === "running" && startTime) {
            const elapsed = Math.round((Date.now() - new Date(startTime).getTime()) / 1000);
            return Math.max(0, elapsed);
        }
        return 0;
    });

    useEffect(() => {
        if (status !== "running") return;

        // Update immediately on mount/status change to align with startTime
        if (startTime) {
            const elapsed = Math.round((Date.now() - new Date(startTime).getTime()) / 1000);
            // Synchronize the local timer with the server-provided start timestamp.
            // eslint-disable-next-line react-hooks/set-state-in-effect
            setSeconds(Math.max(0, elapsed));
        }

        const interval = setInterval(() => {
            if (startTime) {
                const elapsed = Math.round((Date.now() - new Date(startTime).getTime()) / 1000);
                setSeconds(Math.max(0, elapsed));
            } else {
                setSeconds(s => s + 1);
            }
        }, 1000);

        return () => clearInterval(interval);
    }, [status, startTime]);

    const isOpen = forceCollapsed ? false : (status === "running" || steps.length <= 2);
    const isRunning = status === "running";
    const displayTitle = isRunning ? title : (title === "Thinking" ? "Thought" : title);
    const displaySeconds = duration !== undefined ? duration : seconds;

    return (
        <div className="glass-surface max-w-2xl my-3 rounded-xl overflow-hidden transition-all duration-300"
             style={{
                 borderColor: isRunning ? 'rgba(244, 113, 181, 0.2)' : 'var(--q-glass-border)',
             }}>
            <details className="group" open={isOpen}>
                <summary className="flex items-center justify-between px-4 py-3 cursor-pointer list-none select-none hover:bg-[var(--q-glass-hover)] transition-colors">
                    <div className="flex items-center gap-3">
                        {/* Icon */}
                        <div className={`w-7 h-7 flex items-center justify-center rounded-lg transition-colors ${
                            isRunning
                                ? "bg-primary/20"
                                : "bg-emerald-500/15"
                        }`}>
                            {isRunning ? (
                                <Loader2 className="w-4 h-4 text-primary animate-spin" />
                            ) : (
                                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            )}
                        </div>
                        {/* Title + timer */}
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-medium" style={{ color: 'var(--q-text)' }}>
                                {displayTitle}
                            </span>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-md" style={{
                                color: 'var(--q-text-muted)',
                                background: 'var(--q-glass-bg)',
                            }}>
                                {`${displaySeconds}s`}
                            </span>
                        </div>
                    </div>
                    <ChevronDown
                        className="w-4 h-4 group-open:rotate-180 transition-transform duration-200"
                        style={{ color: 'var(--q-text-muted)' }}
                    />
                </summary>

                <div className="px-4 pb-3 pt-2 space-y-1.5" style={{ borderTop: '1px solid var(--q-border)' }}>
                    {steps.map((step, i) => (
                        <div key={i} className="flex items-start gap-2.5 text-xs py-1">
                            <div className="mt-0.5 shrink-0">
                                {step.status === "completed" ? (
                                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                                ) : step.status === "error" ? (
                                    <div className="w-3.5 h-3.5 rounded-full bg-red-500/20 border border-red-500 flex items-center justify-center">
                                        <span className="text-red-500 font-bold text-[8px]">!</span>
                                    </div>
                                ) : (
                                    <div className="w-3.5 h-3.5 flex items-center justify-center">
                                        <div className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
                                    </div>
                                )}
                            </div>
                            <div className={`overflow-x-auto w-full leading-relaxed ${
                                step.status === "running"
                                    ? "text-primary font-medium"
                                    : ""
                            }`} style={{
                                color: step.status === "running" ? undefined : 'var(--q-text-secondary)',
                            }}>
                                {step.isCode ? (
                                    <pre className="p-2 rounded-md text-[10px] font-mono whitespace-pre-wrap" style={{
                                        background: 'var(--q-code-bg)',
                                        color: 'var(--q-text-muted)',
                                    }}>{step.text}</pre>
                                ) : (
                                    step.text
                                )}
                            </div>
                        </div>
                    ))}
                    {children}
                </div>
            </details>
        </div>
    );
}
