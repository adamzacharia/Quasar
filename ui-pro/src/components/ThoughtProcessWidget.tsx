"use client";

import { ChevronDown, Brain, CheckCircle2 } from "lucide-react";
import { useState, useEffect } from "react";

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
}

export function ThoughtProcessWidget({ title = "Thought Process", status, steps, forceCollapsed }: ThoughtProcessWidgetProps) {
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        if (status !== "running") return;
        const interval = setInterval(() => setSeconds(s => s + 1), 1000);
        return () => clearInterval(interval);
    }, [status]);

    const isOpen = forceCollapsed ? false : (status === "running" || steps.length <= 2);

    return (
        <div className="bg-[#1d2e4a]/50 border border-slate-700/50 rounded-2xl overflow-hidden max-w-2xl my-3">
            <details className="group" open={isOpen}>
                <summary className="flex items-center justify-between p-4 cursor-pointer list-none hover:bg-white/5 transition-colors">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 flex items-center justify-center bg-primary/20 rounded-lg">
                            <Brain className={`w-5 h-5 ${status === 'running' ? 'text-primary animate-pulse' : 'text-primary'}`} />
                        </div>
                        <div className="flex flex-col">
                            <span className="text-sm font-semibold text-slate-100">{title}</span>
                            <span className="text-[10px] text-slate-400 font-mono">
                                {status === "running" ? `Thinking for ${seconds}s...` : `Completed in ${seconds}s`}
                            </span>
                        </div>
                    </div>
                    <ChevronDown className="w-5 h-5 text-slate-400 group-open:rotate-180 transition-transform" />
                </summary>
                <div className="px-4 pb-4 space-y-3 border-t border-slate-700/30 pt-3 mt-1">
                    {steps.map((step, i) => (
                        <div key={i} className="flex items-start gap-3 text-xs w-full">
                            <div className="mt-0.5 shrink-0">
                                {step.status === "completed" ? (
                                    <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                                ) : step.status === "error" ? (
                                    <div className="w-4 h-4 rounded-full bg-red-500/20 border border-red-500 flex items-center justify-center">
                                        <span className="text-red-500 font-bold text-[10px]">!</span>
                                    </div>
                                ) : (
                                    <div className="w-4 h-4 flex items-center justify-center">
                                        <div className="w-1.5 h-1.5 bg-primary rounded-full animate-ping" />
                                    </div>
                                )}
                            </div>
                            <div className={`overflow-x-auto w-full ${step.status === "running" ? "text-primary font-medium italic" : "text-slate-300"}`}>
                                {step.isCode ? (
                                    <pre className="p-2 bg-slate-900/50 rounded-md text-[10px] text-slate-400 font-mono whitespace-pre-wrap">{step.text}</pre>
                                ) : (
                                    step.text
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </details>
        </div>
    );
}
