"use client";

import { useState, useRef, useEffect } from "react";
import { Download, Link2, X, Telescope, BarChart3, Map } from "lucide-react";
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

/* ── Mini vertical bar chart (timeline) ── */
function MiniTimeline({
    data, title,
}: { data: Record<string, number>; title: string }) {
    const entries = Object.entries(data).sort(([a], [b]) => Number(a) - Number(b));
    if (entries.length === 0) return null;
    const maxVal = Math.max(...entries.map(([, v]) => v));
    if (maxVal === 0) return null;

    return (
        <div className="flex flex-col items-center gap-1.5 min-w-[120px]">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                {title}
            </span>
            <div className="flex items-end gap-[2px] h-12">
                {entries.map(([year, count]) => {
                    const heightPct = (count / maxVal) * 100;
                    return (
                        <div
                            key={year}
                            className="group relative flex flex-col items-center"
                        >
                            <div
                                className="w-[7px] rounded-t-[2px] bg-gradient-to-t from-indigo-600 to-indigo-400 transition-all hover:from-indigo-500 hover:to-indigo-300 cursor-default"
                                style={{ height: `${Math.max(heightPct, 4)}%`, minHeight: 2 }}
                                title={`${year}: ${count} obs`}
                            />
                            {/* Tooltip on hover */}
                            <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-slate-800 border border-slate-600 rounded px-1 py-0.5 text-[8px] text-slate-300 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                                {year}: {count}
                            </div>
                        </div>
                    );
                })}
            </div>
            <div className="flex justify-between w-full px-0.5">
                <span className="text-[8px] text-slate-500">{entries[0][0]}</span>
                {entries.length > 1 && (
                    <span className="text-[8px] text-slate-500">{entries[entries.length - 1][0]}</span>
                )}
            </div>
        </div>
    );
}

/* ── Mini horizontal bar chart (histogram) ── */
function MiniHistogram({
    data, title, unit,
}: { data: Record<string, number>; title: string; unit?: string }) {
    const entries = Object.entries(data);
    if (entries.length === 0) return null;
    const maxVal = Math.max(...entries.map(([, v]) => v));
    if (maxVal === 0) return null;

    return (
        <div className="flex flex-col items-center gap-1.5 min-w-[110px] max-w-[160px]">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                {title}
            </span>
            <div className="w-full space-y-[3px]">
                {entries.map(([label, count]) => {
                    const widthPct = (count / maxVal) * 100;
                    return (
                        <div key={label} className="flex items-center gap-1.5 group">
                            <span className="text-[8px] text-slate-500 w-[38px] text-right flex-shrink-0 font-mono">
                                {label}
                            </span>
                            <div className="flex-1 h-[7px] bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full rounded-full bg-gradient-to-r from-cyan-600 to-cyan-400 transition-all"
                                    style={{ width: `${Math.max(widthPct, 3)}%` }}
                                />
                            </div>
                            <span className="text-[8px] text-slate-500 w-[22px] text-left flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                                {count}
                            </span>
                        </div>
                    );
                })}
            </div>
            {unit && (
                <span className="text-[8px] text-slate-600">{unit}</span>
            )}
        </div>
    );
}

/* ── Aitoff‑Hammer projection helpers ── */
function aitoffProject(
    raDeg: number, decDeg: number,
    cx: number, cy: number, scaleX: number, scaleY: number,
): [number, number] {
    // Convert RA from [0,360) → longitude [-π, π]  (180° at center)
    let lam = ((raDeg - 180) * Math.PI) / 180;
    if (lam > Math.PI) lam -= 2 * Math.PI;
    if (lam < -Math.PI) lam += 2 * Math.PI;
    const phi = (decDeg * Math.PI) / 180;

    const alpha = Math.acos(Math.cos(phi) * Math.cos(lam / 2));
    const sinc = alpha === 0 ? 1 : Math.sin(alpha) / alpha;

    const x = (2 * Math.cos(phi) * Math.sin(lam / 2)) / sinc;
    const y = Math.sin(phi) / sinc;

    return [cx + x * scaleX, cy - y * scaleY];
}

function drawSkyMap(
    canvas: HTMLCanvasElement,
    coords: { ra: number; dec: number }[],
    opts: { large?: boolean; time?: number } = {},
) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    const large = opts.large ?? false;
    const t = opts.time ?? 0;

    const cx = W / 2;
    const cy = H / 2;
    const scaleX = W * 0.44;
    const scaleY = H * 0.44;

    // ── Background gradient ──
    const bg = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H));
    bg.addColorStop(0, "#0c1929");
    bg.addColorStop(0.5, "#0a1220");
    bg.addColorStop(1, "#060d17");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // ── Background star field (static faint dots) ──
    if (large) {
        ctx.fillStyle = "rgba(180, 200, 230, 0.12)";
        // Deterministic pseudo-random stars based on canvas size
        for (let i = 0; i < 120; i++) {
            const sx = ((i * 7919 + 31) % W);
            const sy = ((i * 6271 + 17) % H);
            const sr = ((i * 3) % 3 === 0) ? 0.8 : 0.4;
            ctx.beginPath();
            ctx.arc(sx, sy, sr, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // ── Aitoff boundary ellipse ──
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, scaleX * 2.02, scaleY * 1.02, 0, 0, Math.PI * 2);
    const boundGrad = ctx.createRadialGradient(cx, cy, scaleY * 0.2, cx, cy, scaleX * 2);
    boundGrad.addColorStop(0, "rgba(30, 58, 95, 0.15)");
    boundGrad.addColorStop(1, "rgba(15, 30, 50, 0.05)");
    ctx.fillStyle = boundGrad;
    ctx.fill();
    ctx.strokeStyle = "rgba(100, 140, 200, 0.15)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    // ── Coordinate grid (denser) ──
    const decStep = large ? 15 : 30;
    const raStep = large ? 30 : 60;

    // Minor grid lines
    ctx.strokeStyle = "rgba(100, 140, 200, 0.06)";
    ctx.lineWidth = 0.4;
    for (let dec = -75; dec <= 75; dec += decStep) {
        ctx.beginPath();
        let first = true;
        for (let ra = 0; ra <= 360; ra += 2) {
            const [px, py] = aitoffProject(ra, dec, cx, cy, scaleX, scaleY);
            if (first) { ctx.moveTo(px, py); first = false; }
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }
    for (let ra = 0; ra < 360; ra += raStep) {
        ctx.beginPath();
        let first = true;
        for (let dec = -90; dec <= 90; dec += 2) {
            const [px, py] = aitoffProject(ra, dec, cx, cy, scaleX, scaleY);
            if (first) { ctx.moveTo(px, py); first = false; }
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    // Major grid lines (every 30° dec, 90° RA)
    ctx.strokeStyle = "rgba(100, 140, 200, 0.12)";
    ctx.lineWidth = 0.6;
    for (const dec of [-60, -30, 30, 60]) {
        ctx.beginPath();
        let first = true;
        for (let ra = 0; ra <= 360; ra += 2) {
            const [px, py] = aitoffProject(ra, dec, cx, cy, scaleX, scaleY);
            if (first) { ctx.moveTo(px, py); first = false; }
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    // Equator (RA axis) — brighter
    ctx.strokeStyle = "rgba(100, 140, 200, 0.22)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    let firstEq = true;
    for (let ra = 0; ra <= 360; ra += 2) {
        const [px, py] = aitoffProject(ra, 0, cx, cy, scaleX, scaleY);
        if (firstEq) { ctx.moveTo(px, py); firstEq = false; }
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // ── Galactic plane hint (soft band) ──
    ctx.save();
    ctx.globalAlpha = 0.04;
    ctx.strokeStyle = "#c4a55a";
    ctx.lineWidth = large ? 14 : 6;
    ctx.beginPath();
    let firstGal = true;
    for (let l = 0; l <= 360; l += 3) {
        const lRad = (l * Math.PI) / 180;
        const ra = (l + 192.85) % 360;
        const dec = 27.13 * Math.sin(lRad - 0.57) - 5;
        const [px, py] = aitoffProject(ra, dec, cx, cy, scaleX, scaleY);
        if (firstGal) { ctx.moveTo(px, py); firstGal = false; }
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();

    // ── Data points (with twinkling) ──
    const baseR = large ? 3.5 : 2;
    const baseGlow = large ? 12 : 6;

    coords.forEach(({ ra, dec }, idx) => {
        const [px, py] = aitoffProject(ra, dec, cx, cy, scaleX, scaleY);

        // Each point has a unique phase offset for independent twinkling
        const phase = idx * 2.39996 + ra * 0.01 + dec * 0.01; // golden-ratio-ish spread
        const twinkle = 0.6 + 0.4 * Math.sin(t * 0.003 + phase);
        const glowTwinkle = 0.5 + 0.5 * Math.sin(t * 0.002 + phase * 1.3);

        const pointR = baseR * (0.85 + 0.15 * twinkle);
        const glowR = baseGlow * (0.8 + 0.3 * glowTwinkle);

        // Outer glow
        const glow = ctx.createRadialGradient(px, py, 0, px, py, glowR);
        glow.addColorStop(0, `rgba(56, 200, 220, ${0.35 * twinkle})`);
        glow.addColorStop(0.4, `rgba(56, 200, 220, ${0.1 * twinkle})`);
        glow.addColorStop(1, "rgba(56, 200, 220, 0)");
        ctx.fillStyle = glow;
        ctx.fillRect(px - glowR, py - glowR, glowR * 2, glowR * 2);

        // Core dot
        ctx.beginPath();
        ctx.arc(px, py, pointR, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(56, 216, 220, ${0.7 + 0.3 * twinkle})`;
        ctx.fill();

        // Hot center
        ctx.beginPath();
        ctx.arc(px, py, pointR * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(220, 255, 255, ${0.6 + 0.4 * twinkle})`;
        ctx.fill();
    });

    // ── RA / Dec labels ──
    if (large) {
        ctx.font = "11px 'Inter', sans-serif";
        ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        for (let ra = 0; ra < 360; ra += 30) {
            const [px, py] = aitoffProject(ra, 0, cx, cy, scaleX, scaleY);
            const h = Math.round(ra / 15);
            ctx.fillText(`${h}h`, px, py + 4);
        }
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (let dec = -75; dec <= 75; dec += 15) {
            if (dec === 0) continue;
            const [px, py] = aitoffProject(180, dec, cx, cy, scaleX, scaleY);
            ctx.fillText(`${dec > 0 ? "+" : ""}${dec}°`, px - 4, py);
        }
    }
}

/* ── Mini sky map thumbnail (animated) ── */
function MiniSkyMap({
    coords, onClick,
}: { coords: { ra: number; dec: number }[]; onClick: () => void }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animRef = useRef<number>(0);

    useEffect(() => {
        let running = true;
        const animate = (time: number) => {
            if (!running || !canvasRef.current) return;
            drawSkyMap(canvasRef.current, coords, { time });
            animRef.current = requestAnimationFrame(animate);
        };
        animRef.current = requestAnimationFrame(animate);
        return () => { running = false; cancelAnimationFrame(animRef.current); };
    }, [coords]);

    return (
        <div className="flex flex-col items-center gap-1.5 min-w-[100px]">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                Sky Map
            </span>
            <button
                onClick={onClick}
                className="relative rounded-lg overflow-hidden border border-slate-600/40
                           hover:border-cyan-500/50 transition-all hover:scale-105
                           hover:shadow-lg hover:shadow-cyan-500/10 cursor-pointer group"
                title="Click to expand sky map"
            >
                <canvas
                    ref={canvasRef}
                    width={160}
                    height={90}
                    className="block"
                    style={{ width: 160, height: 90 }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent
                                opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="absolute bottom-1 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Map className="w-3 h-3 text-cyan-400/60" />
                </div>
            </button>
            <span className="text-[9px] text-slate-500">{coords.length} positions</span>
        </div>
    );
}

/* ── Sky map lightbox (expanded) ── */
function SkyMapLightbox({
    coords, sourceName, onClose,
}: { coords: { ra: number; dec: number }[]; sourceName?: string; onClose: () => void }) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animRef = useRef<number>(0);

    useEffect(() => {
        let running = true;
        const animate = (time: number) => {
            if (!running || !canvasRef.current) return;
            drawSkyMap(canvasRef.current, coords, { large: true, time });
            animRef.current = requestAnimationFrame(animate);
        };
        animRef.current = requestAnimationFrame(animate);
        return () => { running = false; cancelAnimationFrame(animRef.current); };
    }, [coords]);

    // Compute RA/Dec range for footer
    const raMin = Math.min(...coords.map(c => c.ra));
    const raMax = Math.max(...coords.map(c => c.ra));
    const decMin = Math.min(...coords.map(c => c.dec));
    const decMax = Math.max(...coords.map(c => c.dec));

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={onClose}
        >
            <div
                className="relative bg-[#0a1220] border border-slate-600/60 rounded-2xl p-4 shadow-2xl max-w-[680px] w-full mx-4"
                onClick={e => e.stopPropagation()}
            >
                <button
                    onClick={onClose}
                    className="absolute -top-2 -right-2 p-1 bg-slate-800 border border-slate-600 rounded-full hover:bg-red-500/20 transition-colors z-10"
                >
                    <X className="w-4 h-4 text-slate-300" />
                </button>

                <canvas
                    ref={canvasRef}
                    width={1200}
                    height={600}
                    className="w-full rounded-xl"
                    style={{ maxHeight: 400 }}
                />

                <div className="mt-3 text-center space-y-1">
                    <p className="text-sm font-semibold text-slate-200 flex items-center justify-center gap-2">
                        <Map className="w-4 h-4 text-cyan-400" />
                        {sourceName ? `${sourceName} — Sky Distribution` : "Sky Distribution"}
                    </p>
                    <p className="text-[10px] text-slate-500 font-mono">
                        {coords.length} observations · RA {raMin.toFixed(2)}°–{raMax.toFixed(2)}° · Dec {decMin.toFixed(2)}°–{decMax.toFixed(2)}°
                    </p>
                    <p className="text-[9px] text-slate-600">
                        Aitoff equal-area projection · Equatorial J2000
                    </p>
                </div>
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
    const [skyMapOpen, setSkyMapOpen] = useState(false);

    const skyCoords = data.demographics?.skyCoords;
    const hasDemographics = data.demographics && (
        data.demographics.bands || data.demographics.projects ||
        data.demographics.telescopes || data.demographics.instruments ||
        data.demographics.observationYears || data.demographics.scienceCategories ||
        data.demographics.resolutionBins ||
        (skyCoords && skyCoords.length > 0)
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
            {/* Lightbox modals */}
            {lightbox && (
                <PreviewLightbox
                    src={lightbox.src}
                    target={lightbox.target}
                    ra={lightbox.ra}
                    dec={lightbox.dec}
                    onClose={() => setLightbox(null)}
                />
            )}
            {skyMapOpen && skyCoords && skyCoords.length > 0 && (
                <SkyMapLightbox
                    coords={skyCoords}
                    sourceName={data.sourceName}
                    onClose={() => setSkyMapOpen(false)}
                />
            )}

            <div className="rounded-2xl overflow-hidden shadow-sm" style={{ background: 'var(--q-card)', border: '1px solid var(--q-border)' }}>

                {/* ── Metrics bar ── */}
                {data.metrics && data.metrics.length > 0 && (
                    <div className="flex flex-wrap border-b" style={{ background: 'var(--q-surface)', borderColor: 'var(--q-border)' }}>
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
                    <div className="border-b px-4 py-3" style={{ borderColor: 'var(--q-border)', background: 'var(--q-glass-bg)' }}>
                        <div className="flex items-center gap-1.5 mb-2">
                            <BarChart3 className="w-3.5 h-3.5 text-slate-500" />
                            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                                Demographics
                            </span>
                        </div>
                        <div className="flex flex-wrap items-start justify-center gap-6">
                            {skyCoords && skyCoords.length > 0 && (
                                <MiniSkyMap
                                    coords={skyCoords}
                                    onClick={() => setSkyMapOpen(true)}
                                />
                            )}
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
                            {data.demographics?.scienceCategories && Object.keys(data.demographics.scienceCategories).length > 1 && (
                                <MiniDonut data={data.demographics.scienceCategories} title="Science" />
                            )}
                            {data.demographics?.observationYears && Object.keys(data.demographics.observationYears).length > 1 && (
                                <MiniTimeline data={data.demographics.observationYears} title="Timeline" />
                            )}
                            {data.demographics?.resolutionBins && Object.keys(data.demographics.resolutionBins).length > 1 && (
                                <MiniHistogram data={data.demographics.resolutionBins} title="Resolution" unit="arcsec" />
                            )}
                        </div>
                    </div>
                )}

                {/* ── Table ── */}
                {columns.length > 0 && rows.length > 0 ? (
                    <div className="overflow-x-auto overflow-y-auto max-h-[420px] custom-scrollbar">
                        <table className="w-full text-left border-collapse min-w-max">
                            <thead className="sticky top-0 z-10">
                                <tr style={{ background: 'var(--q-surface)', borderBottom: '1px solid var(--q-border)' }}>
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
                                        className="transition-colors"
                                        style={{ borderBottom: '1px solid var(--q-border)' }}
                                        onMouseEnter={(e) => e.currentTarget.style.background = 'var(--q-glass-hover)'}
                                        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
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
                                                        title={`Open in ${data.sourceName || "Archive"}`}
                                                        className="inline-flex items-center gap-1 px-2 py-1 text-[10px] font-semibold text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/10 rounded-md transition-colors whitespace-nowrap"
                                                    >
                                                        <Link2 className="w-3 h-3" />{(data.sourceName || "Archive").split(" / ")[0].split(" ")[0]} ↗
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
                <div className="flex items-center justify-between gap-3 px-4 py-3 border-t flex-wrap" style={{ background: 'var(--q-surface)', borderColor: 'var(--q-border)' }}>
                    <span className="text-xs text-slate-500">
                        {rows.length} row{rows.length !== 1 ? "s" : ""} · {columns.length} col{columns.length !== 1 ? "s" : ""}
                        {data.hasPreview && " · Sky previews"}
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
