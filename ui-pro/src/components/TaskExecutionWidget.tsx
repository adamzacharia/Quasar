"use client";

import { useState, useEffect } from "react";
import { ChevronDown, CheckCircle2, GitFork, Loader2, AlertCircle } from "lucide-react";
import type { TaskGroup, TaskItem, TaskChecklist } from "../lib/types";

// ── Agent type icon mapping ──────────────────────────────────────────────
const AGENT_ICONS: Record<string, string> = {
    archive: "🗄️",
    literature: "📚",
    analysis: "🔬",
    viz: "📊",
    web: "🌐",
    synthesis: "📋",
    general: "⚡",
};

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

// ── Single Task Item Row ─────────────────────────────────────────────────
function TaskItemRow({ task }: { task: TaskItem }) {
    const [expanded, setExpanded] = useState(false);
    const icon = task.icon || AGENT_ICONS[task.agentType] || "⚡";

    return (
        <div className="group">
            <button
                onClick={() => task.detail && setExpanded(!expanded)}
                className={`flex items-center gap-2.5 w-full text-left px-3 py-2 rounded-lg transition-colors
                    ${task.status === "running" ? "bg-primary/5" : "hover:bg-white/5"}
                    ${task.detail ? "cursor-pointer" : "cursor-default"}`}
            >
                <StatusIcon status={task.status} />
                <span className="text-base select-none">{icon}</span>
                <span className={`text-xs flex-1 ${
                    task.status === "running" ? "text-primary font-medium" :
                    task.status === "completed" ? "text-slate-300" :
                    task.status === "error" ? "text-red-400" :
                    "text-slate-400"
                }`}>
                    {task.description}
                </span>
                {task.detail && (
                    <ChevronDown className={`w-3.5 h-3.5 text-slate-500 transition-transform ${expanded ? "rotate-180" : ""}`} />
                )}
            </button>
            {expanded && task.detail && (
                <div className="ml-[2.75rem] px-3 pb-2">
                    <p className="text-[10px] text-slate-500 font-mono leading-relaxed bg-slate-900/40 rounded-md px-2.5 py-1.5">
                        {task.detail}
                    </p>
                </div>
            )}
        </div>
    );
}

// ── Parallel Group ───────────────────────────────────────────────────────
interface TaskGroupWidgetProps {
    group: TaskGroup;
    tasks: Map<string, TaskItem>;
}

function TaskGroupWidget({ group, tasks }: TaskGroupWidgetProps) {
    const groupTasks = group.taskIds
        .map(id => tasks.get(id))
        .filter((t): t is TaskItem => !!t);

    const allCompleted = groupTasks.every(t => t.status === "completed");
    const anyRunning = groupTasks.some(t => t.status === "running");

    return (
        <div className="space-y-0.5">
            {/* Group header */}
            <div className="flex items-center gap-2 px-3 py-1.5">
                <GitFork className={`w-4 h-4 ${anyRunning ? "text-primary" : allCompleted ? "text-emerald-500" : "text-slate-500"}`} />
                <span className={`text-[11px] font-semibold uppercase tracking-wider ${
                    anyRunning ? "text-primary" : allCompleted ? "text-emerald-500/80" : "text-slate-500"
                }`}>
                    {group.title}
                </span>
            </div>
            {/* Task items */}
            <div className="ml-2 border-l border-slate-700/50 pl-2 space-y-0.5">
                {groupTasks.map(task => (
                    <TaskItemRow key={task.id} task={task} />
                ))}
            </div>
        </div>
    );
}

// ── Checklist View ───────────────────────────────────────────────────────
function TaskChecklistWidget({ checklist }: { checklist: TaskChecklist }) {
    const completed = checklist.tasks.filter(t => t.status === "completed").length;
    const total = checklist.tasks.length;

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between px-3 py-1.5">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    📋 Execution Plan
                </span>
                <span className="text-[10px] font-mono text-slate-500">
                    {completed}/{total} completed
                </span>
            </div>
            <div className="space-y-0.5">
                {checklist.tasks.map(task => (
                    <div key={task.id} className="flex items-center gap-2.5 px-3 py-1.5">
                        <StatusIcon status={task.status} />
                        <span className={`text-xs ${
                            task.status === "completed" ? "text-slate-400 line-through" :
                            task.status === "running" ? "text-primary font-medium" :
                            task.status === "error" ? "text-red-400" :
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

// ── Main Widget ──────────────────────────────────────────────────────────
export interface TaskExecutionState {
    groups: TaskGroup[];
    tasks: Map<string, TaskItem>;
    checklist: TaskChecklist | null;
    isActive: boolean;
}

interface TaskExecutionWidgetProps {
    state: TaskExecutionState;
}

export function TaskExecutionWidget({ state }: TaskExecutionWidgetProps) {
    const [seconds, setSeconds] = useState(0);

    useEffect(() => {
        if (!state.isActive) return;
        const interval = setInterval(() => setSeconds(s => s + 1), 1000);
        return () => clearInterval(interval);
    }, [state.isActive]);

    if (!state.isActive && state.groups.length === 0 && !state.checklist) return null;

    const totalTasks = state.tasks.size;
    const completedTasks = Array.from(state.tasks.values()).filter(t => t.status === "completed").length;
    const hasErrors = Array.from(state.tasks.values()).some(t => t.status === "error");

    return (
        <div className="bg-[#1a2a44]/60 border border-slate-700/40 rounded-2xl overflow-hidden max-w-2xl my-3 backdrop-blur-sm">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/30">
                <div className="flex items-center gap-2.5">
                    <div className={`w-2 h-2 rounded-full ${
                        state.isActive ? "bg-primary animate-pulse" :
                        hasErrors ? "bg-red-500" :
                        "bg-emerald-500"
                    }`} />
                    <span className="text-sm font-semibold text-slate-100">
                        Multi-Agent Workforce
                    </span>
                </div>
                <div className="flex items-center gap-3">
                    {totalTasks > 0 && (
                        <span className="text-[10px] font-mono text-slate-400">
                            {completedTasks}/{totalTasks} tasks
                        </span>
                    )}
                    <span className="text-[10px] font-mono text-slate-500">
                        {state.isActive ? `${seconds}s` : `${seconds}s total`}
                    </span>
                </div>
            </div>

            {/* Content */}
            <div className="px-2 py-2 space-y-1 max-h-[400px] overflow-y-auto custom-scrollbar">
                {/* Checklist first (shows full plan) */}
                {state.checklist && (
                    <TaskChecklistWidget checklist={state.checklist} />
                )}

                {/* Then parallel groups */}
                {state.groups.map(group => (
                    <TaskGroupWidget key={group.groupId} group={group} tasks={state.tasks} />
                ))}
            </div>
        </div>
    );
}

// ── Helper: Create initial empty state ───────────────────────────────────
export function createEmptyTaskExecutionState(): TaskExecutionState {
    return {
        groups: [],
        tasks: new Map(),
        checklist: null,
        isActive: false,
    };
}
