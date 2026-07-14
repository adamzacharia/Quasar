"use client";

import { useState, useRef, useEffect } from "react";
import { Download, Eye, ExternalLink, Link2, Loader2, X, Telescope, BarChart3, Map, AlertTriangle, Activity, Layers, FileCode, Radio, Copy, Check } from "lucide-react";
import type { DataTableResult } from "../lib/types";
import { createWorkbenchSession } from "../lib/api";
import { useAuthStore, authBearerHeaders } from "../lib/auth-store";
import { AladinSkyView, type StcsFootprint } from "./AladinSkyView";
import { buildCrossMatchPrompt, dispatchPrefillPrompt } from "../lib/prompt-dispatch";

interface DataTableCardProps { data: DataTableResult; }

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function optionalCellText(value: unknown): string | undefined {
    const text = String(value ?? "").trim();
    if (!text || text === "nan" || text === "None" || text === "-") return undefined;
    return text;
}

function QA2StatusBadge({ status }: { status: string }) {
    const label = status && status !== "—" ? status : "Unknown";
    const normalized = label.replace(/[\s_-]+/g, "").toLowerCase();

    const styleByStatus: Record<string, { dot: string; text: string }> = {
        pass: { dot: "bg-emerald-400 shadow-emerald-400/30", text: "text-emerald-200" },
        semipass: { dot: "bg-amber-300 shadow-amber-300/30", text: "text-amber-200" },
        unknown: { dot: "bg-slate-500 shadow-slate-500/20", text: "text-slate-400" },
    };
    const style = styleByStatus[normalized];

    if (!style) {
        return <span className="text-slate-300">{label}</span>;
    }

    return (
        <span className={`inline-flex items-center gap-1.5 font-sans text-xs font-semibold ${style.text}`}>
            <span className={`h-2 w-2 shrink-0 rounded-full shadow-[0_0_8px_currentColor] ${style.dot}`} aria-hidden="true" />
            {label}
        </span>
    );
}

interface FitsPreviewResult {
    imageDataUrl: string;
    downloadUrl: string;
    filename: string;
    metadata: {
        object?: string;
        unit?: string;
        shape?: number[];
        beamMajorArcsec?: number | null;
        beamMinorArcsec?: number | null;
        beamPaDeg?: number | null;
        restFreqGhz?: number | null;
        pixelScaleArcsec?: { x?: number | null; y?: number | null };
        spectralAxisLabel?: string | null;
        isCube?: boolean;
        rawShape?: number[];
        cubeShape?: number[] | null;
        channelCount?: number | null;
        rms?: number | null;
        rmsUnit?: string | null;
        min?: number;
        max?: number;
        sizeBytes?: number;
    };
    workbench?: {
        kind: "image" | "cube";
        primaryImageLabel?: string;
        channelMapsDataUrl?: string | null;
        pvSliceDataUrl?: string | null;
        spectrum?: {
            x: number[];
            y: number[];
            xLabel: string;
            yLabel: string;
            extraction?: string;
        } | null;
        rms?: number | null;
        rmsUnit?: string | null;
        contourLevels?: number[];
        lineOverlays?: {
            label: string;
            value: number;
            unit: string;
            source: string;
            inRange?: boolean;
        }[];
        exports?: { label: string; command: string }[];
        evidence?: {
            dataAccess?: string;
            headersInspected?: string[];
            assumptions?: string[];
            confidence?: string;
        };
    };
    suggestedActions?: string[];
}

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
type SkyMapTab = "interactive" | "allSky";

function finiteSkyRange(coords: { ra: number; dec: number }[]) {
    const valid = coords.filter((coord) => Number.isFinite(coord.ra) && Number.isFinite(coord.dec));
    if (valid.length === 0) return null;
    return {
        raMin: Math.min(...valid.map((coord) => coord.ra)),
        raMax: Math.max(...valid.map((coord) => coord.ra)),
        decMin: Math.min(...valid.map((coord) => coord.dec)),
        decMax: Math.max(...valid.map((coord) => coord.dec)),
    };
}

type SkySelection = { ra: number; dec: number; i?: number; label?: string };

/** Right-hand inspect panel for a clicked sky marker (click-to-inspect). */
function SourceInspectPanel({
    selection, row, onShowInTable, onCrossMatch, onClose,
}: {
    selection: SkySelection;
    row?: Record<string, string | number>;
    onShowInTable?: () => void;
    onCrossMatch?: () => void;
    onClose: () => void;
}) {
    const fields = row
        ? Object.entries(row).filter(([key, value]) =>
            !key.startsWith("_") && String(value ?? "").trim() !== "")
        : [];
    const preview = row ? String(row["_preview"] ?? "") : "";
    const crossMatchPrompt = buildCrossMatchPrompt({
        ra: selection.ra,
        dec: selection.dec,
        label: selection.label,
    });
    const handleCrossMatch = () => {
        if (!crossMatchPrompt) return;
        // Prefill the composer (no auto-send), then close the sky-map modal so
        // the prefilled prompt is visible and focused for review/edit. (T7.3)
        dispatchPrefillPrompt(crossMatchPrompt);
        onCrossMatch?.();
    };
    // Coordinate-stamped launchpads into the tool surface: each prefills a
    // prompt (never auto-sends, same T7.3 pattern as cross-match) so a clicked
    // marker becomes a starting point for imaging/analysis, not a dead end.
    const raText = selection.ra.toFixed(5);
    const decText = selection.dec.toFixed(5);
    const atText = `at RA ${raText}, Dec ${decText}${selection.label ? ` (${selection.label})` : ""}`;
    const quickActions: { label: string; title: string; prompt: string }[] = [
        {
            label: "Cutout",
            title: "Fetch a multiwavelength image cutout here",
            prompt: `Show me an optical cutout ${atText} with a 0.1 deg field of view.`,
        },
        {
            label: "Radio contours",
            title: "Overlay VLASS radio contours on optical imagery here",
            prompt: `Overlay VLASS radio contours on a DSS2 optical image ${atText}.`,
        },
        {
            label: "Measure",
            title: "Detect and measure sources in a cutout here",
            prompt: `Detect sources and measure their photometry in a 0.1 deg optical cutout ${atText}.`,
        },
        {
            label: "ZTF alerts",
            title: "Search ZTF/ALeRCE transients near this position",
            prompt: `Search ZTF alerts within 60 arcsec ${atText}.`,
        },
        {
            label: "VLASS epochs",
            title: "Compare VLASS radio epochs for variability here",
            prompt: `Compare the VLASS epochs ${atText} and check for radio variability.`,
        },
        {
            label: "Finding chart",
            title: "Generate an observing finding chart here",
            prompt: `Generate a finding chart ${atText} with a 5 arcmin field.`,
        },
    ];
    const runQuickAction = (prompt: string) => {
        dispatchPrefillPrompt(prompt);
        onCrossMatch?.();
    };
    return (
        <div className="absolute inset-y-0 right-0 z-20 flex w-64 flex-col border-l border-slate-700/50 bg-slate-950/90 backdrop-blur">
            <div className="flex items-start justify-between gap-2 border-b border-slate-700/50 px-3 py-2.5">
                <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-slate-100">
                        {selection.label || "Selected source"}
                    </p>
                    <p className="font-mono text-[10px] text-slate-400">
                        RA {selection.ra.toFixed(4)}° · Dec {selection.dec.toFixed(4)}°
                    </p>
                </div>
                <button
                    onClick={onClose}
                    className="rounded p-0.5 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
                    aria-label="Close source details"
                >
                    <X className="h-3.5 w-3.5" />
                </button>
            </div>
            <div className="custom-scrollbar min-h-0 flex-1 overflow-y-auto px-3 py-2">
                {preview && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                        src={preview}
                        alt="Sky thumbnail"
                        className="mb-2 w-full rounded-md border border-slate-700/50"
                        loading="lazy"
                    />
                )}
                {fields.length > 0 ? (
                    <dl className="space-y-1.5">
                        {fields.map(([key, value]) => (
                            <div key={key}>
                                <dt className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">{key}</dt>
                                <dd className="break-words font-mono text-[11px] text-slate-200">{String(value)}</dd>
                            </div>
                        ))}
                    </dl>
                ) : (
                    <p className="text-[11px] text-slate-500">
                        No table row is linked to this marker.
                    </p>
                )}
            </div>
            {(onShowInTable || crossMatchPrompt) && (
                <div className="space-y-1.5 border-t border-slate-700/50 p-2">
                    {crossMatchPrompt && (
                        <button
                            onClick={handleCrossMatch}
                            title="Prefill a chat prompt to cross-match this position against SIMBAD"
                            className="flex w-full items-center justify-center gap-1.5 rounded-md border border-amber-400/30 px-2 py-1.5 text-[11px] font-semibold text-amber-300 transition-colors hover:bg-amber-400/10"
                        >
                            <Layers className="h-3.5 w-3.5" />
                            Cross-match here
                        </button>
                    )}
                    {onShowInTable && (
                        <button
                            onClick={onShowInTable}
                            className="w-full rounded-md border border-cyan-500/30 px-2 py-1.5 text-[11px] font-semibold text-cyan-300 transition-colors hover:bg-cyan-500/10"
                        >
                            Show in table
                        </button>
                    )}
                    <div className="grid grid-cols-2 gap-1.5 pt-0.5">
                        {quickActions.map((action) => (
                            <button
                                key={action.label}
                                onClick={() => runQuickAction(action.prompt)}
                                title={action.title}
                                className="rounded-md border border-slate-700/70 px-1.5 py-1 text-[10px] font-medium text-slate-300 transition-colors hover:bg-slate-800 hover:text-cyan-200"
                            >
                                {action.label}
                            </button>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

function SkyMapLightbox({
    coords, sourceName, rows, footprints, onShowInTable, onClose,
}: {
    coords: SkySelection[];
    sourceName?: string;
    rows?: Record<string, string | number>[];
    footprints?: StcsFootprint[];
    onShowInTable?: (rowIndex: number) => void;
    onClose: () => void;
}) {
    const [activeTab, setActiveTab] = useState<SkyMapTab>("interactive");
    const [interactiveFallbackReason, setInteractiveFallbackReason] = useState<string | null>(null);
    const [selection, setSelection] = useState<SkySelection | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animRef = useRef<number>(0);

    useEffect(() => {
        if (activeTab !== "allSky") return;

        let running = true;
        const animate = (time: number) => {
            if (!running || !canvasRef.current) return;
            drawSkyMap(canvasRef.current, coords, { large: true, time });
            animRef.current = requestAnimationFrame(animate);
        };
        animRef.current = requestAnimationFrame(animate);
        return () => { running = false; cancelAnimationFrame(animRef.current); };
    }, [activeTab, coords]);

    const range = finiteSkyRange(coords);
    const tabs: { id: SkyMapTab; label: string }[] = [
        { id: "interactive", label: "Interactive" },
        { id: "allSky", label: "All-sky" },
    ];

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-lg"
            onClick={onClose}
        >
            <div
                className="glass-surface relative rounded-2xl p-4 w-[min(720px,calc(100vw-32px))] max-h-[calc(100vh-32px)]"
                onClick={e => e.stopPropagation()}
            >
                <button
                    onClick={onClose}
                    className="absolute -top-2 -right-2 p-1 bg-slate-800 border border-slate-600 rounded-full hover:bg-red-500/20 transition-colors z-10"
                    aria-label="Close sky map"
                >
                    <X className="w-4 h-4 text-slate-300" />
                </button>

                <div className="mb-3 flex flex-wrap items-center justify-between gap-3 pr-6">
                    <p className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                        <Map className="w-4 h-4 text-cyan-400" />
                        {sourceName ? `${sourceName} — Sky Distribution` : "Sky Distribution"}
                    </p>
                    <div className="inline-flex rounded-lg border border-cyan-400/25 bg-slate-900/40 p-1">
                        {tabs.map((tab) => {
                            const disabled = tab.id === "interactive" && Boolean(interactiveFallbackReason);
                            return (
                                <button
                                    key={tab.id}
                                    type="button"
                                    disabled={disabled}
                                    title={disabled ? interactiveFallbackReason || undefined : undefined}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`rounded-md px-3 py-1 text-[11px] font-semibold transition-colors ${activeTab === tab.id
                                        ? "bg-sky-400 text-slate-950 shadow-sm shadow-sky-500/40"
                                        : "text-slate-200 hover:bg-sky-400/10 hover:text-white"} ${disabled ? "cursor-not-allowed opacity-40 hover:bg-transparent hover:text-slate-200" : ""}`}
                                >
                                    {tab.label}
                                </button>
                            );
                        })}
                    </div>
                </div>

                {activeTab === "interactive" ? (
                    <div className="relative h-[60vh] min-h-[300px] max-h-[420px] rounded-xl overflow-hidden border border-slate-700/50 bg-slate-950/80">
                        <AladinSkyView
                            coords={coords}
                            sourceName={sourceName}
                            footprints={footprints}
                            onFallback={(reason) => {
                                setInteractiveFallbackReason(reason);
                                setActiveTab("allSky");
                            }}
                            onSourceSelect={setSelection}
                        />
                        {selection && (
                            <SourceInspectPanel
                                selection={selection}
                                row={selection.i !== undefined && rows ? rows[selection.i] : undefined}
                                onShowInTable={
                                    selection.i !== undefined && onShowInTable
                                        ? () => onShowInTable(selection.i as number)
                                        : undefined
                                }
                                onCrossMatch={onClose}
                                onClose={() => setSelection(null)}
                            />
                        )}
                    </div>
                ) : (
                    <canvas
                        ref={canvasRef}
                        width={1200}
                        height={600}
                        className="w-full rounded-xl"
                        style={{ maxHeight: 400 }}
                    />
                )}

                <div className="mt-3 text-center space-y-1">
                    <p className="text-[10px] text-slate-500 font-mono">
                        {coords.length} observations {range
                            ? `· RA ${range.raMin.toFixed(2)}°–${range.raMax.toFixed(2)}° · Dec ${range.decMin.toFixed(2)}°–${range.decMax.toFixed(2)}°`
                            : "· RA/Dec range unavailable"}
                    </p>
                    <p className="text-[9px] text-slate-600">
                        {activeTab === "interactive"
                            ? "Aladin Lite · CDS Strasbourg"
                            : "Aitoff equal-area projection · Equatorial J2000"}
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
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-lg"
            onClick={onClose}
        >
            <div
                className="glass-surface relative rounded-2xl p-3 max-w-[480px]"
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
function SpectrumPlot({
    spectrum,
    lineOverlays,
}: {
    spectrum: NonNullable<NonNullable<FitsPreviewResult["workbench"]>["spectrum"]>;
    lineOverlays?: NonNullable<FitsPreviewResult["workbench"]>["lineOverlays"];
}) {
    const pointCount = Math.min(spectrum.x.length, spectrum.y.length);
    if (pointCount < 2) return null;

    const pairs = spectrum.x.slice(0, pointCount).map((x, i) => ({ x, y: spectrum.y[i] }))
        .filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (pairs.length < 2) return null;

    const xMin = Math.min(...pairs.map(p => p.x));
    const xMax = Math.max(...pairs.map(p => p.x));
    const yMin = Math.min(...pairs.map(p => p.y));
    const yMax = Math.max(...pairs.map(p => p.y));
    const yPad = Math.max((yMax - yMin) * 0.08, Math.abs(yMax || 1) * 0.02, 1e-12);
    const plotW = 560;
    const plotH = 190;
    const padL = 44;
    const padR = 12;
    const padT = 16;
    const padB = 34;
    const innerW = plotW - padL - padR;
    const innerH = plotH - padT - padB;
    const sx = (x: number) => padL + ((x - xMin) / Math.max(xMax - xMin, 1e-12)) * innerW;
    const sy = (y: number) => padT + (1 - ((y - (yMin - yPad)) / Math.max((yMax + yPad) - (yMin - yPad), 1e-12))) * innerH;
    const d = pairs.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    const overlayUnit = spectrum.xLabel.match(/\(([^)]+)\)/)?.[1] || "";
    const usableOverlays = (lineOverlays || []).filter(line =>
        line.inRange !== false && (!overlayUnit || line.unit.toLowerCase() === overlayUnit.toLowerCase())
    );

    return (
        <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <div className="flex items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                    <Activity className="w-3.5 h-3.5 text-cyan-300" />
                    <span className="text-xs font-semibold text-slate-200">Spectrum</span>
                </div>
                {spectrum.extraction && (
                    <span className="text-[10px] text-slate-500 font-mono truncate">{spectrum.extraction}</span>
                )}
            </div>
            <svg viewBox={`0 0 ${plotW} ${plotH}`} className="w-full h-auto max-h-[220px]">
                <rect x={padL} y={padT} width={innerW} height={innerH} rx="4" fill="#020617" stroke="rgba(148,163,184,0.18)" />
                <line x1={padL} y1={sy(0)} x2={plotW - padR} y2={sy(0)} stroke="rgba(148,163,184,0.16)" strokeDasharray="4 4" />
                {usableOverlays.map(line => {
                    const xPos = sx(line.value);
                    return (
                        <g key={`${line.label}-${line.value}`}>
                            <line x1={xPos} y1={padT} x2={xPos} y2={padT + innerH} stroke="rgba(250,204,21,0.55)" strokeDasharray="3 3" />
                            <text x={xPos + 4} y={padT + 12} fill="#fde68a" fontSize="9">{line.label}</text>
                        </g>
                    );
                })}
                <path d={d} fill="none" stroke="#22d3ee" strokeWidth="1.5" />
                <text x={padL} y={plotH - 10} fill="#94a3b8" fontSize="10">{xMin.toPrecision(5)}</text>
                <text x={plotW - padR} y={plotH - 10} fill="#94a3b8" fontSize="10" textAnchor="end">{xMax.toPrecision(5)}</text>
                <text x={plotW / 2} y={plotH - 8} fill="#cbd5e1" fontSize="10" textAnchor="middle">{spectrum.xLabel}</text>
                <text x="12" y={plotH / 2} fill="#cbd5e1" fontSize="10" transform={`rotate(-90 12 ${plotH / 2})`} textAnchor="middle">{spectrum.yLabel}</text>
            </svg>
        </div>
    );
}

function CopyCommandButton({ label, command }: { label: string; command: string }) {
    const [copied, setCopied] = useState(false);
    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(command);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1200);
        } catch {
            setCopied(false);
        }
    };

    return (
        <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md border border-slate-700 text-[10px] font-semibold text-slate-200 hover:bg-slate-800 transition-colors"
            title={command}
        >
            {copied ? <Check className="w-3 h-3 text-emerald-300" /> : <Copy className="w-3 h-3 text-slate-400" />}
            {label}
        </button>
    );
}

function FitsPreviewLightbox({
    preview,
    row,
    onClose,
}: {
    preview: FitsPreviewResult;
    row: Record<string, string | number>;
    onClose: () => void;
}) {
    const metadata = preview.metadata || {};
    const workbench = preview.workbench;
    const isCube = Boolean(metadata.isCube || workbench?.kind === "cube");
    const fmt = (value: number | null | undefined, digits = 3) =>
        typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
    const fmtSci = (value: number | null | undefined, digits = 3) =>
        typeof value === "number" && Number.isFinite(value) ? value.toExponential(digits) : "-";
    const sizeMb = typeof metadata.sizeBytes === "number" ? metadata.sizeBytes / (1024 * 1024) : null;

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-lg"
            onClick={onClose}
        >
            <div
                className="glass-surface relative rounded-2xl w-[min(1180px,calc(100vw-32px))] max-h-[calc(100vh-32px)] overflow-y-auto"
                onClick={e => e.stopPropagation()}
            >
                <button
                    onClick={onClose}
                    className="absolute top-3 right-3 p-1.5 bg-slate-900/90 border border-slate-700 rounded-full hover:bg-red-500/20 transition-colors z-10"
                    aria-label="Close FITS preview"
                >
                    <X className="w-4 h-4 text-slate-300" />
                </button>

                <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px]">
                    <div className="p-4 border-b lg:border-b-0 lg:border-r border-slate-800">
                        <div className="mb-3 pr-10">
                            <p className="text-xs text-slate-500 uppercase tracking-wider">Cube/Product Workbench</p>
                            <h3 className="text-sm font-semibold text-slate-100 truncate" title={preview.filename}>
                                {preview.filename}
                            </h3>
                            <div className="mt-2 flex flex-wrap items-center gap-2">
                                <span className="inline-flex items-center gap-1 rounded-md border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[10px] font-semibold text-cyan-200">
                                    {isCube ? <Layers className="w-3 h-3" /> : <Radio className="w-3 h-3" />}
                                    {isCube ? "Spectral cube" : "Image product"}
                                </span>
                                {metadata.channelCount && (
                                    <span className="rounded-md border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300">
                                        {metadata.channelCount} channels
                                    </span>
                                )}
                                {workbench?.rms && (
                                    <span className="rounded-md border border-slate-700 px-2 py-0.5 text-[10px] text-slate-300">
                                        RMS {fmtSci(workbench.rms)} {workbench.rmsUnit || ""}
                                    </span>
                                )}
                            </div>
                        </div>
                        {workbench?.primaryImageLabel && (
                            <p className="mb-2 text-[11px] text-slate-500">{workbench.primaryImageLabel}</p>
                        )}
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                            src={preview.imageDataUrl}
                            alt={`Rendered FITS preview for ${preview.filename}`}
                            className="w-full rounded-xl bg-slate-950 border border-slate-800"
                        />
                        {workbench?.spectrum && (
                            <div className="mt-3">
                                <SpectrumPlot spectrum={workbench.spectrum} lineOverlays={workbench.lineOverlays} />
                            </div>
                        )}
                        {workbench?.channelMapsDataUrl && (
                            <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                                <div className="mb-2 flex items-center gap-2">
                                    <Layers className="w-3.5 h-3.5 text-cyan-300" />
                                    <span className="text-xs font-semibold text-slate-200">Channel Maps</span>
                                </div>
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                    src={workbench.channelMapsDataUrl}
                                    alt={`Channel map grid for ${preview.filename}`}
                                    className="w-full rounded-md bg-slate-950"
                                />
                            </div>
                        )}
                        {workbench?.pvSliceDataUrl && (
                            <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                                <div className="mb-2 flex items-center gap-2">
                                    <Activity className="w-3.5 h-3.5 text-cyan-300" />
                                    <span className="text-xs font-semibold text-slate-200">PV Slice</span>
                                </div>
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                    src={workbench.pvSliceDataUrl}
                                    alt={`PV slice for ${preview.filename}`}
                                    className="w-full rounded-md bg-slate-950"
                                />
                            </div>
                        )}
                    </div>

                    <div className="p-4 space-y-4">
                        <div>
                            <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Product</p>
                            <dl className="space-y-1.5 text-xs">
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Type</dt>
                                    <dd className="text-slate-200 text-right">{String(row.Product || "-")}</dd>
                                </div>
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Proposal</dt>
                                    <dd className="text-slate-200 text-right font-mono">{String(row["Proposal ID"] || "-")}</dd>
                                </div>
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Object</dt>
                                    <dd className="text-slate-200 text-right">{metadata.object || String(row.Target || "-")}</dd>
                                </div>
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Shape</dt>
                                    <dd className="text-slate-200 text-right font-mono">{metadata.shape?.join(" x ") || "-"}</dd>
                                </div>
                                {metadata.rawShape && metadata.rawShape.length > 0 && (
                                    <div className="flex justify-between gap-3">
                                        <dt className="text-slate-500">Raw shape</dt>
                                        <dd className="text-slate-200 text-right font-mono">{metadata.rawShape.join(" x ")}</dd>
                                    </div>
                                )}
                                {metadata.cubeShape && metadata.cubeShape.length > 0 && (
                                    <div className="flex justify-between gap-3">
                                        <dt className="text-slate-500">Cube shape</dt>
                                        <dd className="text-slate-200 text-right font-mono">{metadata.cubeShape.join(" x ")}</dd>
                                    </div>
                                )}
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Unit</dt>
                                    <dd className="text-slate-200 text-right">{metadata.unit || "-"}</dd>
                                </div>
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Beam</dt>
                                    <dd className="text-slate-200 text-right font-mono">
                                        {fmt(metadata.beamMajorArcsec)} x {fmt(metadata.beamMinorArcsec)} arcsec
                                    </dd>
                                </div>
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Beam PA</dt>
                                    <dd className="text-slate-200 text-right font-mono">{fmt(metadata.beamPaDeg, 1)} deg</dd>
                                </div>
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Rest freq</dt>
                                    <dd className="text-slate-200 text-right font-mono">{fmt(metadata.restFreqGhz, 2)} GHz</dd>
                                </div>
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">Pixel scale</dt>
                                    <dd className="text-slate-200 text-right font-mono">
                                        {fmt(metadata.pixelScaleArcsec?.x)} x {fmt(metadata.pixelScaleArcsec?.y)} arcsec
                                    </dd>
                                </div>
                                <div className="flex justify-between gap-3">
                                    <dt className="text-slate-500">RMS</dt>
                                    <dd className="text-slate-200 text-right font-mono">{fmtSci(metadata.rms)} {metadata.rmsUnit || ""}</dd>
                                </div>
                                {sizeMb !== null && (
                                    <div className="flex justify-between gap-3">
                                        <dt className="text-slate-500">Preview fetch</dt>
                                        <dd className="text-slate-200 text-right font-mono">{fmt(sizeMb, 2)} MB</dd>
                                    </div>
                                )}
                            </dl>
                        </div>

                        <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                            <p className="text-xs font-semibold text-slate-200 mb-1">What do you want to do with it?</p>
                            <p className="text-xs text-slate-400 leading-relaxed">
                                Ask Quasar to measure peak flux/RMS, compare spectral windows, explain the header, or prepare a CASA/Python analysis.
                            </p>
                            <div className="mt-2 space-y-1">
                                {(preview.suggestedActions || []).map(action => (
                                    <div key={action} className="text-[11px] text-slate-300 border border-slate-800 rounded px-2 py-1">
                                        {action}
                                    </div>
                                ))}
                            </div>
                        </div>

                        {workbench?.lineOverlays && workbench.lineOverlays.length > 0 && (
                            <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                                <div className="flex items-center gap-2 mb-2">
                                    <Radio className="w-3.5 h-3.5 text-amber-300" />
                                    <p className="text-xs font-semibold text-slate-200">Line Overlays</p>
                                </div>
                                <div className="space-y-1">
                                    {workbench.lineOverlays.map(line => (
                                        <div key={`${line.label}-${line.value}`} className="flex items-center justify-between gap-2 text-[11px]">
                                            <span className="text-slate-300">{line.label}</span>
                                            <span className={`font-mono ${line.inRange === false ? "text-slate-500" : "text-amber-200"}`}>
                                                {fmt(line.value, 5)} {line.unit}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {workbench?.contourLevels && workbench.contourLevels.length > 0 && (
                            <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                                <p className="text-xs font-semibold text-slate-200 mb-1">Contours</p>
                                <p className="text-[11px] text-slate-400 font-mono">
                                    {workbench.contourLevels.map(level => fmtSci(level, 2)).join(", ")} {workbench.rmsUnit || metadata.unit || ""}
                                </p>
                            </div>
                        )}

                        {workbench?.exports && workbench.exports.length > 0 && (
                            <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                                <div className="flex items-center gap-2 mb-2">
                                    <FileCode className="w-3.5 h-3.5 text-cyan-300" />
                                    <p className="text-xs font-semibold text-slate-200">Export Commands</p>
                                </div>
                                <div className="flex flex-wrap gap-1.5">
                                    {workbench.exports.map(item => (
                                        <CopyCommandButton key={item.label} label={item.label} command={item.command} />
                                    ))}
                                </div>
                            </div>
                        )}

                        {workbench?.evidence && (
                            <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                                <div className="flex items-center justify-between gap-2 mb-2">
                                    <p className="text-xs font-semibold text-slate-200">Evidence</p>
                                    <span className="rounded border border-slate-700 px-1.5 py-0.5 text-[10px] text-slate-400">
                                        {workbench.evidence.confidence || "medium"} confidence
                                    </span>
                                </div>
                                {workbench.evidence.dataAccess && (
                                    <p className="text-[11px] text-slate-400 leading-relaxed">{workbench.evidence.dataAccess}</p>
                                )}
                                {workbench.evidence.headersInspected && workbench.evidence.headersInspected.length > 0 && (
                                    <p className="mt-2 text-[10px] text-slate-500 font-mono">
                                        Headers: {workbench.evidence.headersInspected.join(", ")}
                                    </p>
                                )}
                                {workbench.evidence.assumptions && workbench.evidence.assumptions.length > 0 && (
                                    <ul className="mt-2 space-y-1 text-[10px] text-slate-500">
                                        {workbench.evidence.assumptions.slice(0, 3).map(item => (
                                            <li key={item}>{item}</li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        )}

                        <div className="flex flex-col gap-2">
                            <a
                                href={preview.downloadUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-white text-black text-xs font-semibold hover:bg-slate-200 transition-colors"
                            >
                                <Download className="w-3.5 h-3.5" />Download FITS
                            </a>
                            <a
                                href={preview.downloadUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-slate-700 text-slate-200 text-xs font-semibold hover:bg-slate-800 transition-colors"
                            >
                                <Link2 className="w-3.5 h-3.5" />Open ALMA link
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

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
    const isAlmaProducts = data.tableKind === "alma_products";
    const { isAuthenticated, openAuthModal } = useAuthStore();
    const [lightbox, setLightbox] = useState<{
        src: string; target?: string; ra?: string; dec?: string;
    } | null>(null);
    const [skyMapOpen, setSkyMapOpen] = useState(false);
    const [highlightRow, setHighlightRow] = useState<number | null>(null);
    const rowRefs = useRef<Record<number, HTMLTableRowElement | null>>({});
    const highlightTimerRef = useRef<number>(0);

    // "Show in table" from the sky-map inspect panel: close the lightbox,
    // scroll the matching row into view, and flash-highlight it.
    const handleShowInTable = (rowIndex: number) => {
        setSkyMapOpen(false);
        setHighlightRow(rowIndex);
        window.setTimeout(() => {
            rowRefs.current[rowIndex]?.scrollIntoView({ behavior: "smooth", block: "center" });
        }, 50);
        window.clearTimeout(highlightTimerRef.current);
        highlightTimerRef.current = window.setTimeout(() => setHighlightRow(null), 2500);
    };
    const [fitsPreview, setFitsPreview] = useState<{
        preview: FitsPreviewResult;
        row: Record<string, string | number>;
    } | null>(null);
    const [previewLoadingRow, setPreviewLoadingRow] = useState<number | null>(null);
    const [workbenchLoadingRow, setWorkbenchLoadingRow] = useState<number | null>(null);
    const [previewError, setPreviewError] = useState<string | null>(null);

    const skyCoords = data.demographics?.skyCoords;
    const skyFootprints = data.demographics?.skyFootprints;
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

    const handlePreviewFits = async (row: Record<string, string | number>, rowIndex: number) => {
        const url = String(row["_link"] || "");
        if (!url) return;
        setPreviewError(null);
        setPreviewLoadingRow(rowIndex);
        try {
            const response = await fetch(`${API_BASE}/api/fits/preview`, { credentials: "include",
                method: "POST",
                headers: authBearerHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({
                    url,
                    filename: String(row.File || ""),
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.detail || "Could not render FITS preview.");
            }
            setFitsPreview({ preview: payload as FitsPreviewResult, row });
        } catch (error) {
            setPreviewError(error instanceof Error ? error.message : "Could not render FITS preview.");
        } finally {
            setPreviewLoadingRow(null);
        }
    };

    const handleOpenWorkbench = async (row: Record<string, string | number>, rowIndex: number) => {
        const url = String(row["_link"] || "");
        if (!url) return;
        if (!isAuthenticated) {
            setPreviewError("Sign in to open a persistent FITS workbench session.");
            openAuthModal();
            return;
        }
        setPreviewError(null);
        setWorkbenchLoadingRow(rowIndex);
        try {
            const session = await createWorkbenchSession({
                source_url: url,
                filename: optionalCellText(row.File),
                project_code: optionalCellText(row["Proposal ID"]),
                mous_uid: optionalCellText(row["MOUS ID"]),
            });
            window.open(`/workbench/${encodeURIComponent(session.session_id)}`, "_blank", "noopener,noreferrer");
        } catch (error) {
            setPreviewError(error instanceof Error ? error.message : "Could not open FITS workbench.");
        } finally {
            setWorkbenchLoadingRow(null);
        }
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
                    rows={rows}
                    footprints={skyFootprints}
                    onShowInTable={handleShowInTable}
                    onClose={() => setSkyMapOpen(false)}
                />
            )}
            {fitsPreview && (
                <FitsPreviewLightbox
                    preview={fitsPreview.preview}
                    row={fitsPreview.row}
                    onClose={() => setFitsPreview(null)}
                />
            )}

            <div className="glass-surface rounded-2xl overflow-hidden">

                {data.partial && (
                    <div className="flex flex-wrap items-center gap-2 border-b px-4 py-2 text-xs text-amber-200 bg-amber-500/10" style={{ borderColor: 'var(--q-border)' }}>
                        <AlertTriangle className="w-4 h-4 shrink-0 text-amber-300" />
                        <span className="font-semibold">Partial results</span>
                        {data.warnings && data.warnings.length > 0 && (
                            <span className="text-amber-100/80 truncate">
                                {data.warnings.slice(0, 2).join(" | ")}
                            </span>
                        )}
                    </div>
                )}

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
                    <div className="border-b px-4 py-3" style={{ borderColor: 'var(--q-border)', background: 'var(--q-glass-control)' }}>
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
                {previewError && (
                    <div className="px-4 py-2 border-b text-xs text-red-300 bg-red-950/30" style={{ borderColor: 'var(--q-border)' }}>
                        {previewError}
                    </div>
                )}

                {columns.length > 0 && rows.length > 0 ? (
                    <div className="overflow-x-auto overflow-y-auto max-h-[var(--q-table-max-height)] custom-scrollbar">
                        <table className="w-full text-left border-collapse min-w-max">
                            <thead className="sticky top-0 z-10">
                                <tr style={{ background: 'var(--q-surface)', borderBottom: '1px solid var(--q-border)' }}>
                                    {/* Preview column header */}
                                    {data.hasPreview && (
                                        <th className="py-2.5 px-3 text-[11px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap w-14">
                                            <Telescope className="w-3.5 h-3.5 inline-block" />
                                        </th>
                                    )}
                                    {isAlmaProducts && (
                                        <th className="py-2.5 px-3 text-[11px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap">
                                            Actions
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
                                {rows.map((row, ri) => {
                                    const productFilename = String(row.File || "").toLowerCase();
                                    const productKind = String(row.Product || "").toLowerCase();
                                    const canOpenInWorkbench = (
                                        /\.fits(?:\.gz)?$/.test(productFilename)
                                        || productKind.includes("fits")
                                        || productKind.includes("spectral cube")
                                    );
                                    const isHighlighted = highlightRow === ri;
                                    return (
                                    <tr
                                        key={ri}
                                        ref={(el) => { rowRefs.current[ri] = el; }}
                                        className="transition-colors"
                                        style={{
                                            borderBottom: '1px solid var(--q-border)',
                                            background: isHighlighted ? 'rgba(34, 211, 238, 0.14)' : undefined,
                                        }}
                                        onMouseEnter={(e) => { if (highlightRow !== ri) e.currentTarget.style.background = 'var(--q-glass-hover)'; }}
                                        onMouseLeave={(e) => { if (highlightRow !== ri) e.currentTarget.style.background = 'transparent'; }}
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
                                        {isAlmaProducts && (
                                            <td className="py-2 px-3">
                                                {row["_link"] ? (
                                                    <div className="flex items-center gap-1.5 whitespace-nowrap">
                                                        {canOpenInWorkbench && (
                                                            <>
                                                                <button
                                                                    type="button"
                                                                    onClick={() => handlePreviewFits(row, ri)}
                                                                    disabled={previewLoadingRow === ri || workbenchLoadingRow === ri}
                                                                    className="inline-flex items-center gap-1 px-2 py-1 text-[10px] font-semibold text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/10 rounded-md transition-colors disabled:opacity-60"
                                                                >
                                                                    {previewLoadingRow === ri ? (
                                                                        <Loader2 className="w-3 h-3 animate-spin" />
                                                                    ) : (
                                                                        <Eye className="w-3 h-3" />
                                                                    )}
                                                                    Preview
                                                                </button>
                                                                <button
                                                                    type="button"
                                                                    onClick={() => handleOpenWorkbench(row, ri)}
                                                                    disabled={workbenchLoadingRow === ri || previewLoadingRow === ri}
                                                                    className="inline-flex items-center gap-1 px-2 py-1 text-[10px] font-semibold text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/10 rounded-md transition-colors disabled:opacity-60"
                                                                >
                                                                    {workbenchLoadingRow === ri ? (
                                                                        <Loader2 className="w-3 h-3 animate-spin" />
                                                                    ) : (
                                                                        <ExternalLink className="w-3 h-3" />
                                                                    )}
                                                                    Workbench
                                                                </button>
                                                            </>
                                                        )}
                                                        <a
                                                            href={String(row["_link"])}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            title="Download directly in your browser. Progress is shown by your browser's download manager."
                                                            className="inline-flex items-center gap-1 px-2 py-1 text-[10px] font-semibold text-slate-200 border border-slate-600/50 hover:bg-slate-700 rounded-md transition-colors"
                                                        >
                                                            <Download className="w-3 h-3" />Download
                                                        </a>
                                                    </div>
                                                ) : (
                                                    <span className="text-slate-600 text-xs">-</span>
                                                )}
                                            </td>
                                        )}
                                        {columns.map((col, ci) => {
                                            const val = row[col];
                                            const isFirst = ci === 0;
                                            const display = val === null || val === undefined || val === "nan" || val === "None" || val === ""
                                                ? "—"
                                                : String(val);
                                            const isQa2 = col === "QA2";
                                            return (
                                                <td
                                                    key={ci}
                                                    title={display !== "—" ? display : undefined}
                                                    className={`py-2 px-4 text-xs ${isQa2 ? "max-w-[140px]" : "font-mono max-w-[220px] truncate"} ${
                                                        isFirst ? "text-primary font-semibold" : "text-slate-300"
                                                    }`}
                                                >
                                                    {isQa2 ? <QA2StatusBadge status={display} /> : display}
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
                                    );
                                })}
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
                        {data.fitsEstimate && data.fitsEstimate > 0 && (
                            data.tableKind === "alma_products"
                                ? ` · ${data.fitsEstimate} FITS files`
                                : ` · ~${data.fitsEstimate} FITS files`
                        )}
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
