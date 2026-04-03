"use client";

import { useState } from "react";
import { Download, ExternalLink, Link2, X, Telescope } from "lucide-react";
import type { DataTableResult } from "../lib/types";

interface DataTableCardProps { data: DataTableResult; }

/* ── Lightbox overlay for zoomed sky preview ── */
function PreviewLightbox({
    src, target, ra, dec, onClose,
}: { src: string; target?: string; ra?: string; dec?: string; onClose: () => void }) {
    // Build a larger version of the HiPS URL (400px, wider FOV)
    const largeSrc = src
        .replace("width=120", "width=400")
        .replace("height=120", "height=400")
        .replace("fov=0.033", "fov=0.1");

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={onClose}
        >
            <div
                className="relative bg-surface-dark border border-slate-600/60 rounded-2xl p-3 shadow-2xl max-w-[480px]"
                onClick={e => e.stopPropagation()}
            >
                <button
                    onClick={onClose}
                    className="absolute -top-2 -right-2 p-1 bg-slate-800 border border-slate-600 rounded-full hover:bg-red-500/20 transition-colors z-10"
                >
                    <X className="w-4 h-4 text-slate-300" />
                </button>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                    src={largeSrc}
                    alt={target ? `Sky view of ${target}` : "Sky preview"}
                    className="rounded-xl w-[400px] h-[400px] object-cover bg-slate-900"
                    loading="eager"
                />
                <div className="mt-2 text-center">
                    {target && (
                        <p className="text-sm font-semibold text-slate-200">{target}</p>
                    )}
                    {ra && dec && (
                        <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                            RA {ra}° — Dec {dec}°
                        </p>
                    )}
                    <p className="text-[9px] text-slate-600 mt-1">
                        DSS2 Color · CDS HiPS · ~2&apos; FoV
                    </p>
                </div>
            </div>
        </div>
    );
}

/* ── Inline thumbnail component ── */
function SkyThumbnail({
    src, target, ra, dec, onClick,
}: { src: string; target?: string; ra?: string; dec?: string; onClick: () => void }) {
    const [loaded, setLoaded] = useState(false);
    const [errored, setErrored] = useState(false);

    if (errored) {
        return (
            <div className="w-10 h-10 rounded bg-slate-800 flex items-center justify-center" title="Preview unavailable">
                <Telescope className="w-4 h-4 text-slate-600" />
            </div>
        );
    }

    return (
        <button
            onClick={onClick}
            className="relative w-10 h-10 rounded overflow-hidden border border-slate-600/40 hover:border-primary/60 
                       transition-all hover:scale-110 hover:shadow-lg hover:shadow-primary/10 cursor-pointer group flex-shrink-0"
            title={target ? `View sky around ${target}` : "View sky preview"}
        >
            {!loaded && (
                <div className="absolute inset-0 bg-slate-800 animate-pulse rounded" />
            )}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
                src={src}
                alt={target ? `Sky near ${target}` : "Sky thumbnail"}
                className={`w-full h-full object-cover transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
                loading="lazy"
                onLoad={() => setLoaded(true)}
                onError={() => setErrored(true)}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
        </button>
    );
}

export function DataTableCard({ data }: DataTableCardProps) {
    const columns = data.columns ?? [];
    const rows = data.rows ?? [];
    const [lightbox, setLightbox] = useState<{
        src: string; target?: string; ra?: string; dec?: string;
    } | null>(null);

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
        <>
            {/* Lightbox modal */}
            {lightbox && (
                <PreviewLightbox
                    src={lightbox.src}
                    target={lightbox.target}
                    ra={lightbox.ra}
                    dec={lightbox.dec}
                    onClose={() => setLightbox(null)}
                />
            )}

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
                                    {/* Preview column header */}
                                    {data.hasPreview && (
                                        <th className="py-2.5 px-3 text-[11px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap w-14">
                                            <Telescope className="w-3.5 h-3.5 inline-block" />
                                        </th>
                                    )}
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
                                        {/* Preview thumbnail cell */}
                                        {data.hasPreview && (
                                            <td className="py-1.5 px-3">
                                                {row["_preview"] ? (
                                                    <SkyThumbnail
                                                        src={String(row["_preview"])}
                                                        target={String(row[columns[1]] ?? row[columns[0]] ?? "")}
                                                        ra={(() => {
                                                            const url = String(row["_preview"]);
                                                            const m = url.match(/ra=([\d.]+)/);
                                                            return m ? m[1] : undefined;
                                                        })()}
                                                        dec={(() => {
                                                            const url = String(row["_preview"]);
                                                            const m = url.match(/dec=(-?[\d.]+)/);
                                                            return m ? m[1] : undefined;
                                                        })()}
                                                        onClick={() =>
                                                            setLightbox({
                                                                src: String(row["_preview"]),
                                                                target: String(row[columns[1]] ?? row[columns[0]] ?? ""),
                                                                ra: (() => {
                                                                    const url = String(row["_preview"]);
                                                                    const m = url.match(/ra=([\d.]+)/);
                                                                    return m ? m[1] : undefined;
                                                                })(),
                                                                dec: (() => {
                                                                    const url = String(row["_preview"]);
                                                                    const m = url.match(/dec=(-?[\d.]+)/);
                                                                    return m ? m[1] : undefined;
                                                                })(),
                                                            })
                                                        }
                                                    />
                                                ) : (
                                                    <div className="w-10 h-10 rounded bg-slate-800/50 flex items-center justify-center">
                                                        <Telescope className="w-4 h-4 text-slate-700" />
                                                    </div>
                                                )}
                                            </td>
                                        )}
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
                        {data.hasPreview && " · DSS2 previews"}
                    </span>
                    <div className="flex gap-2 flex-wrap justify-end">
                        <button
                            onClick={handleDownloadCSV}
                            className="flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-700 rounded-lg transition-colors"
                        >
                            <Download className="w-3.5 h-3.5" />Download CSV
                        </button>
                    </div>
                </div>
            </div>
        </>
    );
}
