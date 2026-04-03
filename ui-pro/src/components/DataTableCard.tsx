"use client";

import { useState } from "react";
import { Download, Link2, X, Telescope, BarChart3 } from "lucide-react";
import type { DataTableResult } from "../lib/types";

interface DataTableCardProps { data: DataTableResult; }

/* ── Donut chart via CSS conic-gradient ── */
const CHART_COLORS = [
    "#6366f1", "#8b5cf6", "#10b981", "#f59e0b",
    "#ef4444", "#06b6d4", "#ec4899", "#84cc16",
];

function MiniDonut({
    data, title,
}: { data: Record<string, number>; title: string }) {
    const entries = Object.entries(data);
    const total = entries.reduce((s, [, v]) => s + v, 0);
    if (total === 0) return null;

    // Build conic-gradient segments
    let cumDeg = 0;
    const segments = entries.map(([, v], i) => {
        const start = cumDeg;
        cumDeg += (v / total) * 360;
        return `${CHART_COLORS[i % CHART_COLORS.length]} ${start}deg ${cumDeg}deg`;
    });

    return (
        <div className="flex flex-col items-center gap-1.5 min-w-[100px]">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                {title}
            </span>
            <div
                className="w-14 h-14 rounded-full"
                style={{
                    background: `conic-gradient(${segments.join(", ")})`,
                    mask: "radial-gradient(closest-side, transparent 55%, black 56%)",
                    WebkitMask: "radial-gradient(closest-side, transparent 55%, black 56%)",
                }}
            />
            <div className="flex flex-wrap justify-center gap-x-2 gap-y-0.5 max-w-[160px]">
                {entries.map(([k, v], i) => (
                    <span key={k} className="text-[9px] text-slate-400 flex items-center gap-1 whitespace-nowrap">
                        <span
                            className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                            style={{ background: CHART_COLORS[i % CHART_COLORS.length] }}
                        />
                        {k.length > 12 ? k.slice(0, 11) + "…" : k}
                        <span className="text-slate-500">({v})</span>
                    </span>
                ))}
            </div>
        </div>
    );
}

/* ── Lightbox overlay for zoomed sky preview ── */
function PreviewLightbox({
    src, target, ra, dec, onClose,
}: { src: string; target?: string; ra?: string; dec?: string; onClose: () => void }) {
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
    src, target, onClick,
}: { src: string; target?: string; onClick: () => void }) {
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

    const hasDemographics = data.demographics && (
        data.demographics.bands || data.demographics.projects ||
        data.demographics.telescopes || data.demographics.instruments
    );

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

                {/* ── Demographics Charts ── */}
                {hasDemographics && (
                    <div className="border-b border-slate-700/50 bg-slate-900/40 px-4 py-3">
                        <div className="flex items-center gap-1.5 mb-2">
                            <BarChart3 className="w-3.5 h-3.5 text-slate-500" />
                            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                                Demographics
                            </span>
                        </div>
                        <div className="flex flex-wrap items-start justify-center gap-6">
                            {data.demographics?.bands && Object.keys(data.demographics.bands).length > 1 && (
                                <MiniDonut data={data.demographics.bands} title="Bands" />
                            )}
                            {data.demographics?.telescopes && Object.keys(data.demographics.telescopes).length > 1 && (
                                <MiniDonut data={data.demographics.telescopes} title="Telescopes" />
                            )}
                            {data.demographics?.instruments && Object.keys(data.demographics.instruments).length > 1 && (
                                <MiniDonut data={data.demographics.instruments} title="Instruments" />
                            )}
                            {data.demographics?.projects && Object.keys(data.demographics.projects).length > 1 && (
                                <MiniDonut data={data.demographics.projects} title="Projects" />
                            )}
                        </div>
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
                                                        onClick={() => {
                                                            const url = String(row["_preview"]);
                                                            const raMatch = url.match(/ra=([\d.]+)/);
                                                            const decMatch = url.match(/dec=(-?[\d.]+)/);
                                                            setLightbox({
                                                                src: url,
                                                                target: String(row[columns[1]] ?? row[columns[0]] ?? ""),
                                                                ra: raMatch ? raMatch[1] : undefined,
                                                                dec: decMatch ? decMatch[1] : undefined,
                                                            });
                                                        }}
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
                        {data.fitsEstimate && data.fitsEstimate > 0 && ` · ~${data.fitsEstimate} FITS files`}
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
