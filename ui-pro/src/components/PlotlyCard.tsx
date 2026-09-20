/* eslint-disable @next/next/no-img-element */
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Download, Loader2, RotateCcw, Waves } from "lucide-react";
import type { PlotlyModule } from "plotly.js-basic-dist-min";
import type { PlotlyFigureSpec, PlotlyCardMeta } from "../lib/types";
import { ImageLightbox } from "./ImageLightbox";
import { QueryProvenance } from "./QueryProvenance";
import { IllustrativeBadge } from "./IllustrativeBadge";
import { previewBadgeFacts } from "../lib/export-decision";
import { useThemeStore } from "../lib/theme-store";
import {
    asRecord,
    lightExportData,
    lightExportLayout,
    themedData,
    themedLayout,
} from "../lib/plotly-theme";
import { createRenderChain } from "../lib/render-chain";

type PlotlyCardProps = {
    spec?: PlotlyFigureSpec;
    title?: string;
    pngFallback?: string;
    meta?: PlotlyCardMeta;
    /** The exact request behind this figure (Feature 1), when known. */
    request?: import("../lib/api").ToolRequest;
};

function isValidSpec(spec: PlotlyFigureSpec | undefined): spec is PlotlyFigureSpec & { data: unknown[] } {
    return !!spec && typeof spec === "object" && Array.isArray(spec.data) && spec.data.length > 0;
}

// Theme skins live in lib/plotly-theme.js (RE-C1): the on-screen layout now
// follows the ACTIVE app theme — the old hardcoded dark glass skin rendered
// slate-300 text at ~1.3:1 on the light theme's near-white card. The light
// "publication" export skin (solid white, dark text) stays the PNG path.

/** Plotly attaches the rendered figure to its root node; the minimal module
 *  declaration doesn't model that, so widen locally where we read it back. */
type PlotlyGraphDiv = HTMLElement & {
    data?: unknown[];
    layout?: Record<string, unknown>;
};

/** Plotly.toImage also accepts a figure spec in place of a graph div, which
 *  lets us export the light re-skin without a visible re-theme flash. */
type PlotlyToImageFromFigure = (
    figure: { data: unknown[]; layout: Record<string, unknown>; config?: Record<string, unknown> },
    options: { format: "png"; scale: number; width?: number; height?: number },
) => Promise<string>;

/** Render the light publication re-skin off-DOM and return a PNG data URL. */
async function lightPngDataUrl(Plotly: PlotlyModule, el: PlotlyGraphDiv): Promise<string> {
    const figure = {
        // Clone the data like the layout: toImage's figure-spec branch hands
        // traces to a hidden newPlot whose cleaning mutates them in place —
        // live references would let that leak into the on-screen figure.
        data: JSON.parse(JSON.stringify(
            lightExportData(Array.isArray(el.data) ? el.data : []),
        )) as unknown[],
        layout: lightExportLayout(el.layout),
        config: { displaylogo: false },
    };
    const toImage = Plotly.toImage as unknown as PlotlyToImageFromFigure;
    return toImage(figure, {
        format: "png",
        scale: 2,
        width: el.offsetWidth || 960,
        height: el.offsetHeight || 480,
    });
}

function triggerDownload(dataUrl: string, filename: string) {
    const anchor = document.createElement("a");
    anchor.href = dataUrl;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
}

function fileSlug(title?: string): string {
    const slug = (title || "quasar_plot").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
    return slug || "quasar_plot";
}

type PeriodFoldMeta = {
    time: number[];
    value: number[];
    bestPeriod: number;
    valueKind: string;
};

/** Raw fold arrays embedded by the backend (layout.meta.kind === "period_fold")
 *  so the phase diagram can be re-folded client-side at any trial period. */
function periodFoldMeta(spec?: PlotlyFigureSpec): PeriodFoldMeta | null {
    const meta = asRecord(asRecord(spec?.layout).meta);
    if (meta.kind !== "period_fold") return null;
    const time = Array.isArray(meta.time_days) ? (meta.time_days as unknown[]).map(Number) : [];
    const value = Array.isArray(meta.value) ? (meta.value as unknown[]).map(Number) : [];
    const bestPeriod = Number(meta.best_period_d);
    if (!time.length || time.length !== value.length || !Number.isFinite(bestPeriod) || bestPeriod <= 0) {
        return null;
    }
    return { time, value, bestPeriod, valueKind: String(meta.value_kind || "") };
}

/** Common Lomb-Scargle alias periods worth one click: harmonics and the
 *  1-day sampling beats that ground-based cadences produce. */
function aliasPeriods(period: number): { label: string; value: number }[] {
    const aliases: { label: string; value: number }[] = [
        { label: "2P", value: period * 2 },
        { label: "P/2", value: period / 2 },
    ];
    for (const [label, sign] of [["1d beat −", -1], ["1d beat +", 1]] as const) {
        const beatFreq = 1 / period + sign;
        if (beatFreq > 0) {
            const beat = 1 / beatFreq;
            if (Number.isFinite(beat) && beat > 0) aliases.push({ label, value: beat });
        }
    }
    return aliases;
}

/**
 * Interactive Plotly figure card for "plotly" SSE events. Loads the
 * lightweight basic bundle client-side only (dynamic import in useEffect,
 * never in the SSR bundle) and falls back to the PNG snapshot as a regular
 * image card if the bundle fails to load or the spec is invalid.
 */
export function PlotlyCard({ spec, title, pngFallback, meta, request }: PlotlyCardProps) {
    const plotRef = useRef<HTMLDivElement>(null);
    const plotlyRef = useRef<PlotlyModule | null>(null);
    const [status, setStatus] = useState<"loading" | "ready" | "fallback">(
        () => (isValidSpec(spec) ? "loading" : "fallback")
    );
    const [downloading, setDownloading] = useState(false);
    const [lightboxOpen, setLightboxOpen] = useState(false);
    // Active app theme (data-theme on <html>, kept in sync by the zustand
    // store) — the plot re-skins live when the user toggles it.
    const theme = useThemeStore((s) => s.theme);

    // Escalate the badge when the meta says this figure came from a truncated
    // preview (f2-CX-11). Today's Data Lab plot tools draw from the FULL stored
    // frame (result_store) and stamp any source truncation on the figure itself,
    // so no producer sets these fields yet — this is the client half, live the
    // moment a producer stamps preview_derived/shown_rows/total_rows on meta.
    const badgeFacts = previewBadgeFacts(meta);

    const foldMeta = useMemo(() => periodFoldMeta(spec), [spec]);
    const [foldPeriod, setFoldPeriod] = useState<number | null>(null);
    const activePeriod = foldPeriod ?? foldMeta?.bestPeriod ?? null;

    // Re-fold the phase trace at a trial period entirely client-side:
    // phase = ((t - t0) mod P) / P, plotted twice for continuity.
    const applyFold = useCallback((period: number) => {
        const Plotly = plotlyRef.current;
        const el = plotRef.current;
        if (!Plotly || !el || !foldMeta || !Number.isFinite(period) || period <= 0) return;
        const t0 = Math.min(...foldMeta.time);
        const phase = foldMeta.time.map((t) => ((t - t0) / period) % 1.0);
        const x = [...phase, ...phase.map((p) => p + 1.0)];
        const y = [...foldMeta.value, ...foldMeta.value];
        const traces = Array.isArray(spec?.data) ? (spec!.data as Record<string, unknown>[]) : [];
        let foldIndex = traces.findIndex((trace) => asRecord(trace).xaxis === "x2");
        if (foldIndex < 0) foldIndex = traces.length > 1 ? 1 : 0;
        void Plotly.restyle(el, { x: [x], y: [y] }, [foldIndex]);
        setFoldPeriod(period);
    }, [foldMeta, spec]);

    // Latest fold state, readable from the async plot effect without joining
    // its dependency list: a theme toggle re-plots from the pristine spec
    // (which bakes the BEST-period fold), so any custom fold is re-applied.
    const foldPeriodRef = useRef<number | null>(foldPeriod);
    const applyFoldRef = useRef(applyFold);
    useEffect(() => { foldPeriodRef.current = foldPeriod; }, [foldPeriod]);
    useEffect(() => { applyFoldRef.current = applyFold; }, [applyFold]);

    const layoutHeight = typeof spec?.layout?.height === "number" ? (spec.layout.height as number) : 420;

    // CX-07 (task-3c99b37-115): renders into the shared plot element are
    // SERIALIZED through this per-card chain. Effect cleanup cannot cancel
    // an already-running Plotly.newPlot, so each effect run (a) waits for
    // the previous render to settle before starting its own newPlot — an
    // old theme render can never finish late and overwrite a newer one —
    // and (b) re-checks cancelled/isStale after every await, skipping its
    // follow-up mutations (fold restore, setStatus) once superseded.
    const renderChainRef = useRef(createRenderChain());

    useEffect(() => {
        if (!isValidSpec(spec)) {
            setStatus("fallback");
            return;
        }
        const el = plotRef.current;
        if (!el) return;
        let cancelled = false;

        void renderChainRef.current.enqueue(async (isStale) => {
            try {
                if (cancelled || isStale()) return;
                const mod = await import("plotly.js-basic-dist-min");
                const Plotly = mod.default ?? (mod as unknown as PlotlyModule);
                if (cancelled || isStale()) return;
                plotlyRef.current = Plotly;
                // The built-in camera snapshots the transparent dark layout
                // (toImageButtonOptions cannot override layout), so swap it for
                // an identical-looking button that exports the light re-skin.
                const camera = (Plotly as unknown as { Icons?: { camera?: unknown } }).Icons?.camera;
                const config: Record<string, unknown> = camera
                    ? {
                        responsive: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ["lasso2d", "toImage"],
                        modeBarButtonsToAdd: [{
                            name: "toImageLight",
                            title: "Download plot as a png",
                            icon: camera,
                            click: (gd: PlotlyGraphDiv) => {
                                void lightPngDataUrl(Plotly, gd)
                                    .then((dataUrl) => triggerDownload(dataUrl, `${fileSlug(title)}.png`))
                                    .catch(() => {
                                        // Light export failed (e.g. canvas size cap):
                                        // serve the server-rendered PNG instead of a dead click.
                                        if (pngFallback) triggerDownload(pngFallback, `${fileSlug(title)}.png`);
                                    });
                            },
                        }],
                    }
                    : {
                        // Icons missing from the bundle: keep the stock camera
                        // (dark export) rather than losing the button entirely.
                        responsive: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ["lasso2d"],
                        toImageButtonOptions: { format: "png", filename: fileSlug(title), scale: 2 },
                    };
                await Plotly.newPlot(el, themedData(spec.data, theme), themedLayout(spec.layout, theme), config);
                if (cancelled || isStale()) return;
                // A re-plot (theme toggle) resets the phase trace to the baked
                // best-period fold; restore any custom fold the user had set.
                const pendingFold = foldPeriodRef.current;
                if (pendingFold !== null) applyFoldRef.current(pendingFold);
                setStatus("ready");
            } catch {
                if (!cancelled && !isStale()) setStatus("fallback");
            }
        });

        return () => {
            cancelled = true;
            if (plotlyRef.current) {
                try { plotlyRef.current.purge(el); } catch { /* already gone */ }
            }
        };
    }, [spec, title, pngFallback, theme]);

    const downloadPng = async () => {
        const Plotly = plotlyRef.current;
        const el = plotRef.current as PlotlyGraphDiv | null;
        if (!Plotly || !el || downloading) return;
        setDownloading(true);
        try {
            const dataUrl = await lightPngDataUrl(Plotly, el);
            triggerDownload(dataUrl, `${fileSlug(title)}.png`);
        } catch {
            // Light export failed (e.g. canvas size cap): serve the
            // server-rendered PNG instead of a dead click.
            if (pngFallback) triggerDownload(pngFallback, `${fileSlug(title)}.png`);
        } finally {
            setDownloading(false);
        }
    };

    // ── Fallback: static PNG image card (or a quiet note if none given) ──
    if (status === "fallback") {
        if (!pngFallback) {
            // A render failure must not also cost the card its provenance (CX-16)
            // — the query still ran, and is arguably MORE useful when the plot
            // didn't draw.
            return (
                <div className="pl-11">
                    <div className="mt-4 rounded-xl border border-slate-700/50 bg-slate-900/60 px-4 py-3 text-xs text-slate-400">
                        Interactive plot could not be rendered{title ? `: ${title}` : "."}
                        <IllustrativeBadge className="mt-2" {...badgeFacts} />
                        <QueryProvenance request={request} />
                    </div>
                </div>
            );
        }
        return (
            <div className="pl-11">
                <div className="mt-4 rounded-xl border border-indigo-500/30 bg-slate-900/60 overflow-hidden shadow-xl shadow-indigo-500/5">
                    <img
                        src={pngFallback}
                        alt={title || "Plot"}
                        className="w-full max-h-[600px] object-contain bg-black cursor-zoom-in hover:brightness-110 transition"
                        loading="lazy"
                        onClick={() => setLightboxOpen(true)}
                    />
                    <div className="px-4 py-2.5 border-t border-slate-700/50 flex items-center justify-between gap-3">
                        <span className="min-w-0 truncate text-xs text-slate-400">{title || "Plot"}</span>
                        <div className="flex shrink-0 items-center gap-1.5">
                            {meta?.line_explorer_url && (
                                <a
                                    href={meta.line_explorer_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1.5 rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-1 text-[11px] font-semibold text-indigo-200 transition-colors hover:bg-indigo-500/15"
                                    title="Open this spectrum in the Spectral Line Explorer"
                                >
                                    <Waves className="h-3.5 w-3.5" />
                                    Line Explorer
                                </a>
                            )}
                            <a
                                href={pngFallback}
                                download
                                className="inline-flex items-center gap-1.5 rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-1 text-[11px] font-semibold text-indigo-200 transition-colors hover:bg-indigo-500/15"
                            >
                                <Download className="h-3.5 w-3.5" />
                                PNG
                            </a>
                        </div>
                        <IllustrativeBadge className="mt-2" {...badgeFacts} />
                        <QueryProvenance request={request} />
                    </div>
                </div>
                {lightboxOpen && (
                    <ImageLightbox src={pngFallback} caption={title} onClose={() => setLightboxOpen(false)} />
                )}
            </div>
        );
    }

    // ── Interactive plot (with loading overlay until Plotly is ready) ──
    return (
        <div className="pl-11">
            <div className="mt-4 rounded-xl border border-indigo-500/30 bg-slate-900/60 overflow-hidden shadow-xl shadow-indigo-500/5">
                <div className="relative">
                    <div ref={plotRef} className="w-full" style={{ height: layoutHeight }} />
                    {status === "loading" && (
                        <div className="absolute inset-0 flex items-center justify-center gap-2 bg-slate-950/60 text-xs font-medium text-slate-300 backdrop-blur-sm">
                            <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-300" />
                            Loading interactive plot…
                        </div>
                    )}
                </div>
                {foldMeta && activePeriod !== null && (
                    <div className="px-4 py-2 border-t border-slate-700/50 flex flex-wrap items-center gap-2.5">
                        <span className="text-[11px] font-semibold text-indigo-200">
                            Fold P = {activePeriod.toPrecision(6)} d
                        </span>
                        <input
                            aria-label="Trial fold period"
                            type="range"
                            min={Math.log10(foldMeta.bestPeriod / 1.5)}
                            max={Math.log10(foldMeta.bestPeriod * 1.5)}
                            step={0.0002}
                            value={Math.log10(activePeriod)}
                            onChange={(event) => applyFold(10 ** Number(event.target.value))}
                            disabled={status !== "ready"}
                            className="h-1 w-40 accent-indigo-400 disabled:opacity-40"
                            title="Slide to re-fold at a nearby trial period"
                        />
                        {aliasPeriods(foldMeta.bestPeriod).map((alias) => (
                            <button
                                key={alias.label}
                                type="button"
                                onClick={() => applyFold(alias.value)}
                                disabled={status !== "ready"}
                                className="rounded-md border border-slate-700/70 bg-slate-950/60 px-1.5 py-0.5 text-[10px] font-medium text-slate-300 transition-colors hover:bg-slate-800 hover:text-indigo-200 disabled:opacity-40"
                                title={`Re-fold at the ${alias.label} alias (${alias.value.toPrecision(5)} d)`}
                            >
                                {alias.label}
                            </button>
                        ))}
                        <button
                            type="button"
                            onClick={() => applyFold(foldMeta.bestPeriod)}
                            disabled={status !== "ready" || activePeriod === foldMeta.bestPeriod}
                            className="inline-flex items-center gap-1 rounded-md border border-slate-700/70 bg-slate-950/60 px-1.5 py-0.5 text-[10px] font-medium text-slate-300 transition-colors hover:bg-slate-800 hover:text-indigo-200 disabled:opacity-40"
                            title="Back to the Lomb-Scargle best period"
                        >
                            <RotateCcw className="h-3 w-3" />
                            best
                        </button>
                    </div>
                )}
                <div className="px-4 py-2.5 border-t border-slate-700/50 flex items-center justify-between gap-3">
                    <span className="min-w-0 truncate text-xs text-slate-400">{title || "Interactive plot"}</span>
                    <div className="flex shrink-0 items-center gap-1.5">
                        {meta?.line_explorer_url && (
                            <a
                                href={meta.line_explorer_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-1 text-[11px] font-semibold text-indigo-200 transition-colors hover:bg-indigo-500/15"
                                title="Open this spectrum in the Spectral Line Explorer"
                            >
                                <Waves className="h-3.5 w-3.5" />
                                Line Explorer
                            </a>
                        )}
                        <button
                            type="button"
                            onClick={downloadPng}
                            disabled={status !== "ready" || downloading}
                            className="inline-flex items-center gap-1.5 rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-1 text-[11px] font-semibold text-indigo-200 transition-colors hover:bg-indigo-500/15 disabled:opacity-40"
                            title="Download as PNG"
                        >
                            {downloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                            PNG
                        </button>
                    </div>
                </div>
                <IllustrativeBadge className="mt-2" {...badgeFacts} />
                <QueryProvenance request={request} />
            </div>
        </div>
    );
}
