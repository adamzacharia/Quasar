"use client";

import { Download, ExternalLink, Link2 } from "lucide-react";
import type { DataTableResult } from "../lib/types";

interface DataTableCardProps { data: DataTableResult; }

export function DataTableCard({ data }: DataTableCardProps) {
    const columns = data.columns ?? [];
    const rows = data.rows ?? [];

    // Download CSV helper
    const handleDownloadCSV = () => {
        const header = columns.join(",");
        const body = rows.map(row =>
            columns.map(col => `"${String(row[col] ?? "").replace(/"/g, '""')}"`).join(",")
        ).join("\n");
        const blob = new Blob([header + "\n" + body], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${data.sourceName || "alma"}_results.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="bg-sidebar-dark border border-slate-700/50 rounded-2xl overflow-hidden shadow-sm">

            {/* ── Metrics bar ── */}
            {data.metrics && data.metrics.length > 0 && (
                <div className="flex flex-wrap divide-x divide-slate-700/50 border-b border-slate-700/50 bg-surface-dark">
                    {data.metrics.map((metric, i) => (
                        <div key={i} className="px-5 py-3 flex flex-col items-center justify-center min-w-[80px]">
                            <span className={`text-xl font-bold tabular-nums ${metric.color || "text-white"}`}>
                                {metric.value}
                            </span>
                            <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wide whitespace-nowrap">
                                {metric.label}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {/* ── Table ── */}
            {columns.length > 0 && rows.length > 0 ? (
                <div className="overflow-x-auto overflow-y-auto max-h-[420px] custom-scrollbar">
                    <table className="w-full text-left border-collapse min-w-max">
                        <thead className="sticky top-0 z-10">
                            <tr className="bg-surface-dark border-b border-slate-700/60">
                                {columns.map((col, i) => (
                                    <th
                                        key={i}
                                        className="py-2.5 px-4 text-[11px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap"
                                    >
                                        {col}
                                    </th>
                                ))}
                                {/* Extra column header for per-row links */}
                                {data.hasRowLinks && (
                                    <th className="py-2.5 px-3 text-[11px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap">
                                        View
                                    </th>
                                )}
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row, ri) => (
                                <tr
                                    key={ri}
                                    className="border-b border-slate-700/20 hover:bg-slate-800/40 transition-colors"
                                >
                                    {columns.map((col, ci) => {
                                        const val = row[col];
                                        const isFirst = ci === 0;
                                        const display = val === null || val === undefined || val === "nan" || val === "None" || val === ""
                                            ? "—"
                                            : String(val);
                                        return (
                                            <td
                                                key={ci}
                                                title={display !== "—" ? display : undefined}
                                                className={`py-2 px-4 text-xs font-mono max-w-[220px] truncate ${
                                                    isFirst ? "text-primary font-semibold" : "text-slate-300"
                                                }`}
                                            >
                                                {display}
                                            </td>
                                        );
                                    })}
                                    {/* Per-row archive link button */}
                                    {data.hasRowLinks && (
                                        <td className="py-2 px-3">
                                            {row["_link"] ? (
                                                <a
                                                    href={String(row["_link"])}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    title="Open this observation in ALMA Archive"
                                                    className="inline-flex items-center gap-1 px-2 py-1 text-[10px] font-semibold text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/10 rounded-md transition-colors whitespace-nowrap"
                                                >
                                                    <Link2 className="w-3 h-3" />ALMA ↗
                                                </a>
                                            ) : (
                                                <span className="text-slate-600 text-xs">—</span>
                                            )}
                                        </td>
                                    )}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div className="p-6 text-slate-400 text-sm text-center">No data to display</div>
            )}

            {/* ── Footer ── */}
            <div className="flex items-center justify-between gap-3 px-4 py-3 bg-surface-dark border-t border-slate-700/50 flex-wrap">
                <span className="text-xs text-slate-500">
                    {rows.length} row{rows.length !== 1 ? "s" : ""} · {columns.length} col{columns.length !== 1 ? "s" : ""}
                </span>
                <div className="flex gap-2 flex-wrap justify-end">
                    {data.archiveLink && (
                        <a
                            href={data.archiveLink}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/10 rounded-lg transition-colors"
                        >
                            <ExternalLink className="w-3.5 h-3.5" />All on ALMA Archive
                        </a>
                    )}
                    <button
                        onClick={handleDownloadCSV}
                        className="flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-700 rounded-lg transition-colors"
                    >
                        <Download className="w-3.5 h-3.5" />Download CSV
                    </button>
                </div>
            </div>
        </div>
    );
}
