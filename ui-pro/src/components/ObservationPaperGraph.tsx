"use client";

import { Database, FileText, GitBranch, RadioTower, Target, Waypoints } from "lucide-react";

type GraphNode = {
    id: string;
    type: "archive" | "project" | "target" | "paper" | "provenance" | string;
    label: string;
    detail?: string;
    count?: number;
    url?: string;
};

type GraphEdge = {
    from: string;
    to: string;
    label?: string;
    strength?: number;
};

type ResearchGraph = {
    nodes: GraphNode[];
    edges: GraphEdge[];
    summary: {
        archives: number;
        projects: number;
        identifiers?: number;
        observations: number;
        papers: number;
        linkedPapers: number;
    };
};

interface ObservationPaperGraphProps {
    graph: ResearchGraph;
}

const COLUMN_BY_TYPE: Record<string, number> = {
    archive: 0,
    provenance: 0,
    identifier: 1,
    project: 1,
    target: 2,
    paper: 3,
};

const COLUMN_LABELS = ["Archives", "Identifiers", "Targets", "Papers"];
const COLUMN_X = [10, 36, 62, 88];

function nodeColor(type: string): string {
    switch (type) {
        case "archive":
            return "#38bdf8";
        case "project":
            return "#a78bfa";
        case "identifier":
            return "#a78bfa";
        case "target":
            return "#f472b6";
        case "paper":
            return "#34d399";
        case "provenance":
            return "#f59e0b";
        default:
            return "#94a3b8";
    }
}

function nodeIcon(type: string) {
    const cls = "w-3.5 h-3.5";
    switch (type) {
        case "archive":
            return <Database className={cls} />;
        case "project":
            return <RadioTower className={cls} />;
        case "identifier":
            return <GitBranch className={cls} />;
        case "target":
            return <Target className={cls} />;
        case "paper":
            return <FileText className={cls} />;
        default:
            return <GitBranch className={cls} />;
    }
}

function truncateLabel(value: string, max = 38): string {
    const suffix = "...";
    return value.length > max ? `${value.slice(0, Math.max(0, max - suffix.length))}${suffix}` : value;
}

export function ObservationPaperGraph({ graph }: ObservationPaperGraphProps) {
    const columns = [[], [], [], []] as GraphNode[][];
    for (const node of graph.nodes) {
        const col = COLUMN_BY_TYPE[node.type] ?? 0;
        columns[col].push(node);
    }

    const maxColumnSize = Math.max(...columns.map((nodes) => nodes.length), 1);
    const graphHeight = Math.max(330, 96 + maxColumnSize * 78);
    const positions = new Map<string, { x: number; y: number }>();
    columns.forEach((nodes, col) => {
        const step = 100 / (nodes.length + 1);
        nodes.forEach((node, index) => {
            positions.set(node.id, {
                x: COLUMN_X[col],
                y: Math.max(12, Math.min(88, step * (index + 1))),
            });
        });
    });

    const identifierCount = graph.summary.identifiers || graph.summary.projects;
    const linkedPct = graph.summary.papers > 0
        ? Math.round((graph.summary.linkedPapers / graph.summary.papers) * 100)
        : 0;

    return (
        <div className="pl-11">
            <section
                className="rounded-lg overflow-hidden"
                style={{ background: "var(--q-card)", border: "1px solid var(--q-border)" }}
            >
                <div className="px-4 py-3 border-b flex flex-wrap items-center justify-between gap-3"
                     style={{ borderColor: "var(--q-border)", background: "var(--q-surface)" }}>
                    <div className="flex items-center gap-2 min-w-0">
                        <Waypoints className="w-4 h-4 text-cyan-300 shrink-0" />
                        <div className="min-w-0">
                            <h3 className="text-sm font-semibold text-slate-100 truncate">Observation-Paper Graph</h3>
                            <p className="text-[11px] text-slate-400 truncate">
                                Archive records, observing programs, targets, and literature provenance
                            </p>
                        </div>
                    </div>
                    <div className="grid grid-cols-4 gap-2 text-center">
                        <div>
                            <div className="text-sm font-semibold text-slate-100">{graph.summary.observations}</div>
                            <div className="text-[9px] uppercase text-slate-500">Obs</div>
                        </div>
                        <div>
                            <div className="text-sm font-semibold text-slate-100">{identifierCount}</div>
                            <div className="text-[9px] uppercase text-slate-500">IDs</div>
                        </div>
                        <div>
                            <div className="text-sm font-semibold text-slate-100">{graph.summary.papers}</div>
                            <div className="text-[9px] uppercase text-slate-500">Papers</div>
                        </div>
                        <div>
                            <div className="text-sm font-semibold text-slate-100">{linkedPct}%</div>
                            <div className="text-[9px] uppercase text-slate-500">Linked</div>
                        </div>
                    </div>
                </div>

                <div className="max-h-[520px] overflow-auto custom-scrollbar">
                    <div className="relative min-w-[620px]" style={{ height: `${graphHeight}px` }}>
                        <div className="absolute inset-x-0 top-0 grid grid-cols-4 px-5 py-2 text-[10px] uppercase text-slate-500 font-medium">
                            {COLUMN_LABELS.map((label) => (
                                <div key={label} className="text-center">{label}</div>
                            ))}
                        </div>

                        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                            {graph.edges.map((edge, index) => {
                                const from = positions.get(edge.from);
                                const to = positions.get(edge.to);
                                if (!from || !to) return null;
                                const strokeWidth = Math.min(1.4, 0.45 + (edge.strength || 1) * 0.08);
                                return (
                                    <path
                                        key={`${edge.from}-${edge.to}-${index}`}
                                        d={`M ${from.x} ${from.y} C ${(from.x + to.x) / 2} ${from.y}, ${(from.x + to.x) / 2} ${to.y}, ${to.x} ${to.y}`}
                                        fill="none"
                                        stroke="var(--q-observation-graph-edge)"
                                        strokeWidth={strokeWidth}
                                        vectorEffect="non-scaling-stroke"
                                    />
                                );
                            })}
                        </svg>

                        {graph.nodes.map((node) => {
                            const pos = positions.get(node.id);
                            if (!pos) return null;
                            const color = nodeColor(node.type);
                            const content = (
                                <div
                                    className="w-[128px] rounded-lg px-2.5 py-2 shadow-sm transition-all hover:scale-[1.02]"
                                    style={{
                                        background: "var(--q-observation-graph-node-bg)",
                                        border: `1px solid ${color}66`,
                                        boxShadow: `0 0 0 1px ${color}18`,
                                    }}
                                    title={node.detail || node.label}
                                >
                                    <div className="flex items-center gap-1.5 min-w-0" style={{ color }}>
                                        {nodeIcon(node.type)}
                                        <span className="text-[10px] font-semibold uppercase truncate">{node.type}</span>
                                    </div>
                                    <div
                                        className="text-xs font-semibold leading-snug mt-1 line-clamp-2"
                                        style={{ color: "var(--q-observation-graph-node-text)" }}
                                    >
                                        {truncateLabel(node.label)}
                                    </div>
                                    {node.detail && (
                                        <div
                                            className="text-[10px] truncate mt-1"
                                            style={{ color: "var(--q-observation-graph-node-detail)" }}
                                        >
                                            {node.detail}
                                        </div>
                                    )}
                                </div>
                            );

                            return (
                                <div
                                    key={node.id}
                                    className="absolute -translate-x-1/2 -translate-y-1/2"
                                    style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
                                >
                                    {node.url ? (
                                        <a href={node.url} target="_blank" rel="noopener noreferrer">
                                            {content}
                                        </a>
                                    ) : content}
                                </div>
                            );
                        })}
                    </div>
                </div>
            </section>
        </div>
    );
}
