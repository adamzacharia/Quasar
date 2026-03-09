"use client";

import { ChevronLeft, ChevronRight, FileText, Download, BarChart3 } from "lucide-react";
import type { DataTableResult } from "../lib/types";

interface DataTableCardProps { data: DataTableResult; }

export function DataTableCard({ data }: DataTableCardProps) {
    return (
        <div className="bg-sidebar-dark border border-slate-700/50 rounded-2xl overflow-hidden shadow-sm">
            <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-slate-700/50 border-b border-slate-700/50 bg-surface-dark">
                {data.metrics.map((metric, i) => (
                    <div key={i} className="p-4 flex flex-col items-center justify-center">
                        <span className={`text-2xl font-bold ${metric.color || "text-white"}`}>{metric.value}</span>
                        <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">{metric.label}</span>
                    </div>
                ))}
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="text-xs text-slate-400 bg-surface-dark border-b border-slate-700/50">
                            {data.columns.map((col, i) => (<th key={i} className="py-3 px-4 font-semibold">{col}</th>))}
                        </tr>
                    </thead>
                    <tbody className="text-sm text-slate-300 font-mono">
                        {data.rows.map((row, i) => (
                            <tr key={i} className="border-b border-slate-700/30 hover:bg-slate-800/30 transition-colors">
                                {data.columns.map((col, j) => (
                                    <td key={j} className={`py-3 px-4 ${col === "Project" ? "text-primary" : ""} ${col.includes("Freq") || col.includes("Res") ? "text-right" : ""} ${col === "PI Name" ? "font-display text-xs" : ""}`}>
                                        {row[col]}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="flex items-center justify-end gap-3 p-3 bg-surface-dark border-t border-slate-700/50">
                <button className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-300 hover:bg-slate-700 rounded-lg transition-colors">
                    <Download className="w-4 h-4" />Download CSV
                </button>
                <button className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-primary hover:bg-primary/90 rounded-lg shadow-md shadow-primary/20 transition-colors">
                    <BarChart3 className="w-4 h-4" />View Plots
                </button>
            </div>
        </div>
    );
}
