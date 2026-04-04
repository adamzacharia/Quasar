"use client";

import React, { useState, useEffect } from "react";
import { ChevronDown, CheckCircle2, GitFork, Loader2, AlertCircle } from "lucide-react";
import type { TaskGroup, TaskItem, TaskChecklist } from "../lib/types";

// ── Astrophysics SVG Icons (styled after the brain favicon) ─────────────────
// Each icon is a 16×16 SVG, stroke-based, using the primary pink colour.

function IconRadioDish({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            {/* Dish bowl */}
            <path d="M2 11 Q8 2 14 11" />
            {/* Support arm */}
            <line x1="8" y1="6.5" x2="8" y2="14" />
            {/* Base */}
            <line x1="5" y1="14" x2="11" y2="14" />
            {/* Signal dot at focus */}
            <circle cx="8" cy="6.5" r="1" fill="currentColor" stroke="none" />
            {/* Signal arcs */}
            <path d="M5.5 4 Q8 1.5 10.5 4" strokeWidth="1" opacity="0.6" />
        </svg>
    );
}

function IconPaperStar({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            {/* Document */}
            <rect x="2" y="2" width="9" height="12" rx="1.5" />
            {/* Lines */}
            <line x1="4.5" y1="6" x2="8.5" y2="6" />
            <line x1="4.5" y1="8.5" x2="7" y2="8.5" />
            {/* Star badge */}
            <path d="M12.5 5 l0.6 1.3 1.4 0.2 -1 1 0.2 1.4 -1.2-0.7 -1.2 0.7 0.2-1.4 -1-1 1.4-0.2z" fill="currentColor" stroke="none" />
        </svg>
    );
}

function IconOrbitRings({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            {/* Central body */}
            <circle cx="8" cy="8" r="1.5" fill="currentColor" stroke="none" />
            {/* Inner orbit */}
            <ellipse cx="8" cy="8" rx="5" ry="2.5" transform="rotate(-30 8 8)" />
            {/* Outer orbit */}
            <ellipse cx="8" cy="8" rx="5" ry="2.5" transform="rotate(60 8 8)" />
            {/* Orbiting dot */}
            <circle cx="11.3" cy="6.5" r="0.9" fill="currentColor" stroke="none" />
        </svg>
    );
}

function IconSpectrum({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            {/* Waveform baseline */}
            <line x1="1" y1="8" x2="3" y2="8" />
            <path d="M3 8 Q4.5 3 5.5 8 Q6.5 13 8 8 Q9.5 3 10.5 8 Q11.5 13 13 8" />
            <line x1="13" y1="8" x2="15" y2="8" />
            {/* Emission line */}
            <line x1="8" y1="3" x2="8" y2="5" strokeWidth="1.8" opacity="0.7" />
        </svg>
    );
}

function IconGlobeNetwork({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="8" cy="8" r="6" />
            {/* Latitude */}
            <path d="M2.5 8 Q8 5 13.5 8" />
            <path d="M2.5 8 Q8 11 13.5 8" />
            {/* Meridian */}
            <path d="M8 2 Q11 8 8 14" />
            <path d="M8 2 Q5 8 8 14" />
        </svg>
    );
}

function IconConstellation({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            {/* Stars */}
            <circle cx="3" cy="4" r="1.2" fill="currentColor" stroke="none" />
            <circle cx="13" cy="4" r="1.2" fill="currentColor" stroke="none" />
            <circle cx="8" cy="13" r="1.5" fill="currentColor" stroke="none" />
            {/* Lines connecting them */}
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

// ── Agent type → {Icon component, label} ────────────────────────────────────
type AgentType = "archive" | "literature" | "analysis" | "viz" | "web" | "synthesis" | "general";

const AGENT_CONFIG: Record<AgentType, { Icon: (props: { className?: string }) => React.ReactElement; label: string }> = {
    archive:    { Icon: IconRadioDish,     label: "Archive Search" },
    literature: { Icon: IconPaperStar,     label: "Literature Review" },
    analysis:   { Icon: IconOrbitRings,    label: "Data Analysis" },
    viz:        { Icon: IconSpectrum,      label: "Visualization" },
    web:        { Icon: IconGlobeNetwork,  label: "Web Search" },
    synthesis:  { Icon: IconConstellation, label: "Result Synthesis" },
    general:    { Icon: IconStar,          label: "General Task" },
};

function AgentIcon({ agentType, className = "" }: { agentType: string; className?: string }) {
    const cfg = AGENT_CONFIG[agentType as AgentType] ?? AGENT_CONFIG.general;
    return <cfg.Icon className={className} />;
}

// ── Status indicator ─────────────────────────────────────────────────────────
function StatusIcon({ status }: { status: string }) {
    switch (status) {
        case "completed":
            return <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />;
        case "running":
            return <Loader2 className="w-4 h-4 text-primary animate-spin shrink-0" />;
        case "error":
            return <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />;
        default:
            return <div className="w-4 h-4 rounded-full border border-slate-600 shrink-0" />;
    }
}

// ── Single Task Item Row ─────────────────────────────────────────────────────
function TaskItemRow({ task }: { task: TaskItem }) {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className="group">
            <button
                onClick={() => task.detail && setExpanded(!expanded)}
                className={`flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg transition-colors
                    ${task.status === "running" ? "bg-primary/5" : "hover:bg-white/5"}
                    ${task.detail ? "cursor-pointer" : "cursor-default"}`}
            >
                <StatusIcon status={task.status} />
                {/* Agent type icon */}
                <AgentIcon
                    agentType={task.agentType}
                    className={`w-4 h-4 shrink-0 ${
                        task.status === "running"   ? "text-primary" :
                        task.status === "completed" ? "text-emerald-500/70" :
                        task.status === "error"     ? "text-red-400" :
                                                      "text-slate-500"
                    }`}
                />
                <span className={`text-xs flex-1 ${
                    task.status === "running"   ? "text-primary font-medium" :
                    task.status === "completed" ? "text-slate-300" :
                    task.status === "error"     ? "text-red-400" :
                                                  "text-slate-400"
                }`}>
                    {task.description}
                </span>
                {task.detail && (
                    <ChevronDown className={`w-3.5 h-3.5 text-slate-500 transition-transform ${expanded ? "rotate-180" : ""}`} />
                )}
            </button>
            {expanded && task.detail && (
                <div className="ml-[3.25rem] px-3 pb-2">
                    <p className="text-[10px] text-slate-500 font-mono leading-relaxed bg-slate-900/40 rounded-md px-2.5 py-1.5">
                        {task.detail}
                    </p>
                </div>
            )}
        </div>
    );
}

// ── Parallel Task Group ──────────────────────────────────────────────────────
interface TaskGroupWidgetProps {
    group: TaskGroup;
    tasks: Map<string, TaskItem>;
}

function TaskGroupWidget({ group, tasks }: TaskGroupWidgetProps) {
    const groupTasks = group.taskIds
        .map(id => tasks.get(id))
        .filter((t): t is TaskItem => !!t);

    const allCompleted = groupTasks.every(t => t.status === "completed");
    const anyRunning   = groupTasks.some(t => t.status === "running");

    return (
        <div className="space-y-0.5">
            {/* Group header */}
            <div className="flex items-center gap-2 px-3 py-1.5">
                <GitFork className={`w-3.5 h-3.5 ${anyRunning ? "text-primary" : allCompleted ? "text-emerald-500" : "text-slate-500"}`} />
                <span className={`text-[11px] font-semibold uppercase tracking-wider ${
                    anyRunning ? "text-primary" : allCompleted ? "text-emerald-500/80" : "text-slate-500"
                }`}>
                    {group.title}
                </span>
            </div>
            {/* Task items indented under group */}
            <div className="ml-2 border-l border-slate-700/50 pl-2 space-y-0.5">
                {groupTasks.map(task => (
                    <TaskItemRow key={task.id} task={task} />
                ))}
            </div>
        </div>
    );
}

// ── Flat Checklist View (shown first as the plan overview) ───────────────────
function TaskChecklistWidget({ checklist }: { checklist: TaskChecklist }) {
    const completed = checklist.tasks.filter(t => t.status === "completed").length;
    const total     = checklist.tasks.length;

    return (
        <div className="space-y-1">
            <div className="flex items-center justify-between px-3 py-1.5">
                <div className="flex items-center gap-2">
                    <IconConstellation className="w-3.5 h-3.5 text-primary opacity-80" />
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                        Execution Plan
                    </span>
                </div>
                <span className="text-[10px] font-mono text-slate-500">
                    {completed}/{total} completed
                </span>
            </div>
            <div className="space-y-0.5">
                {checklist.tasks.map(task => (
                    <div key={task.id} className="flex items-center gap-2.5 px-3 py-1.5">
                        <StatusIcon status={task.status} />
                        <AgentIcon
                            agentType={task.agentType}
                            className={`w-3.5 h-3.5 shrink-0 ${
                                task.status === "running"   ? "text-primary" :
                                task.status === "completed" ? "text-emerald-500/60" :
                                                              "text-slate-600"
                            }`}
                        />
                        <span className={`text-xs ${
                            task.status === "completed" ? "text-slate-400 line-through" :
                            task.status === "running"   ? "text-primary font-medium" :
                            task.status === "error"     ? "text-red-400" :
                                                          "text-slate-500"
                        }`}>
                            {task.description}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ── Main Widget ──────────────────────────────────────────────────────────────
export interface TaskExecutionState {
    groups:    TaskGroup[];
    tasks:     Map<string, TaskItem>;
    checklist: TaskChecklist | null;
    isActive:  boolean;
}

interface TaskExecutionWidgetProps {
    state: TaskExecutionState;
}

export function TaskExecutionWidget({ state }: TaskExecutionWidgetProps) {
    const [seconds, setSeconds] = useState(0);
    const [isCollapsed, setIsCollapsed] = useState(false);

    useEffect(() => {
        if (!state.isActive) return;
        const interval = setInterval(() => setSeconds(s => s + 1), 1000);
        return () => clearInterval(interval);
    }, [state.isActive]);

    // Auto-collapse when execution completes
    useEffect(() => {
        if (!state.isActive && state.tasks.size > 0) {
            setIsCollapsed(true);
        }
    }, [state.isActive, state.tasks.size]);

    if (!state.isActive && state.groups.length === 0 && !state.checklist) return null;

    const totalTasks     = state.tasks.size;
    const completedTasks = Array.from(state.tasks.values()).filter(t => t.status === "completed").length;
    const hasErrors      = Array.from(state.tasks.values()).some(t => t.status === "error");

    return (
        <div className="bg-slate-900/40 border border-slate-700/30 rounded-xl overflow-hidden w-full my-1 backdrop-blur-sm">
            {/* Header — clickable to expand/collapse when done */}
            <button
                onClick={() => !state.isActive && setIsCollapsed(!isCollapsed)}
                className={`flex items-center justify-between px-4 py-3 w-full text-left ${
                    !state.isActive ? "cursor-pointer hover:bg-white/[0.02] transition-colors" : ""
                } ${!isCollapsed ? "border-b border-slate-700/30" : ""}`}
            >
                <div className="flex items-center gap-2.5">
                    {/* Pink brain-style icon container */}
                    <div className="w-8 h-8 flex items-center justify-center bg-primary/20 rounded-lg">
                        <IconConstellation className={`w-5 h-5 text-primary ${state.isActive ? "animate-pulse" : ""}`} />
                    </div>
                    <div className="flex flex-col">
                        <span className="text-sm font-semibold text-slate-100">
                            Multi-Agent Workforce
                        </span>
                        <span className="text-[10px] font-mono text-slate-400">
                            {state.isActive
                                ? `Running for ${seconds}s…`
                                : hasErrors
                                    ? `Completed with errors in ${seconds}s`
                                    : `Completed in ${seconds}s`}
                        </span>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {/* Progress pill */}
                    {totalTasks > 0 && (
                        <div className="flex items-center gap-2">
                            <div className="h-1.5 w-20 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-primary rounded-full transition-all duration-500"
                                    style={{ width: `${(completedTasks / totalTasks) * 100}%` }}
                                />
                            </div>
                            <span className="text-[10px] font-mono text-slate-400">
                                {completedTasks}/{totalTasks}
                            </span>
                        </div>
                    )}
                    {/* Expand/collapse chevron — only when done */}
                    {!state.isActive && (
                        <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform duration-200 ${isCollapsed ? "" : "rotate-180"}`} />
                    )}
                </div>
            </button>

            {/* Content — collapsible when done */}
            {!isCollapsed && (
                <div className="px-2 py-2 space-y-1 max-h-[400px] overflow-y-auto custom-scrollbar">
                    {/* Full checklist plan first */}
                    {state.checklist && (
                        <TaskChecklistWidget checklist={state.checklist} />
                    )}
                    {/* Divider between plan and live groups */}
                    {state.checklist && state.groups.length > 0 && (
                        <div className="border-t border-slate-700/30 mx-3 my-1" />
                    )}
                    {/* Live parallel execution groups */}
                    {state.groups.map(group => (
                        <TaskGroupWidget key={group.groupId} group={group} tasks={state.tasks} />
                    ))}
                </div>
            )}
        </div>
    );
}

// ── Helper ────────────────────────────────────────────────────────────────────
export function createEmptyTaskExecutionState(): TaskExecutionState {
    return { groups: [], tasks: new Map(), checklist: null, isActive: false };
}
