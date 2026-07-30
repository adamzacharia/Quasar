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

function asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

const GRID_COLOR = "rgba(148, 163, 184, 0.12)";   // slate-400 @ low alpha
const LINE_COLOR = "rgba(148, 163, 184, 0.28)";
const FONT_COLOR = "#cbd5e1";                      // slate-300

/** Re-skin an incoming Plotly layout to match the dark glassmorphism theme. */
function darkenLayout(layout: Record<string, unknown> | undefined): Record<string, unknown> {
    const base: Record<string, unknown> = { ...(layout || {}) };

    // Cover xaxis/yaxis plus subplot axes (xaxis2, yaxis3, ...)
    const axisKeys = Object.keys(base).filter((key) => /^[xy]axis\d*$/.test(key));
    for (const required of ["xaxis", "yaxis"]) {
        if (!axisKeys.includes(required)) axisKeys.push(required);
    }
    for (const key of axisKeys) {
        base[key] = {
            ...asRecord(base[key]),
            gridcolor: GRID_COLOR,
            zerolinecolor: LINE_COLOR,
            linecolor: LINE_COLOR,
            tickcolor: LINE_COLOR,
        };
    }

    return {
        ...base,
        autosize: true,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { ...asRecord(base.font), color: FONT_COLOR },
        legend: {
            ...asRecord(base.legend),
            bgcolor: "rgba(15, 23, 42, 0.4)",
            bordercolor: GRID_COLOR,
        },
        hoverlabel: {
            ...asRecord(base.hoverlabel),
            bgcolor: "#1e293b",
            bordercolor: LINE_COLOR,
            font: { color: "#e2e8f0" },
        },
        margin: base.margin ?? { t: 48, r: 24, b: 48, l: 60 },
    };
}

// ── Light "publication" export theme ──
// The dark glass layout above is transparent-on-dark, so a PNG snapshot of it
// is light-gray text on transparency — invisible once pasted into a white doc.
// Both export paths (modebar camera + Download button) therefore render a
// re-skinned CLONE of the live figure instead of snapshotting the DOM.
const EXPORT_TEXT_COLOR = "#1e293b";                    // slate-800
const EXPORT_GRID_COLOR = "rgba(100, 116, 139, 0.35)";  // slate-500, visible on white
const EXPORT_AXIS_COLOR = "rgba(51, 65, 85, 0.9)";      // slate-700

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

function withFontColor(font: unknown, color: string): Record<string, unknown> {
    return { ...asRecord(font), color };
}

function rgbChannels(color: unknown): number[] | null {
    if (typeof color !== "string") return null;
    const value = color.trim().toLowerCase();
    if (value === "white") return [255, 255, 255];
    const hex = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/);
    if (hex) {
        return hex[1].length === 3
            ? [...hex[1]].map((c) => parseInt(c + c, 16))
            : [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16));
    }
    const rgb = value.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
    if (rgb) return rgb.slice(1, 4).map(Number);
    return null;
}

/** True for colors that are illegible on a white background (white and very
 *  light grays, i.e. every RGB channel >= 200). */
function isNearWhite(color: unknown): boolean {
    const channels = rgbChannels(color);
    return channels !== null && channels.every((ch) => ch >= 200);
}

/** Wider net for text and guide lines: the dark-theme grays (slate-400 line
 *  labels, slate marker shapes) sit well under the near-white bar yet are
 *  still ~2:1 contrast on white. Trace palettes stay on the stricter check
 *  so deliberately colored data is never touched. */
function isLowContrastOnWhite(color: unknown): boolean {
    const channels = rgbChannels(color);
    return channels !== null && channels.every((ch) => ch >= 140);
}

/** Deep-clone the live (dark) layout and re-skin it for a standalone PNG:
 *  solid white background, dark slate text, clearly visible grid and axes. */
function lightExportLayout(layout: Record<string, unknown> | undefined): Record<string, unknown> {
    const base = JSON.parse(JSON.stringify(layout ?? {})) as Record<string, unknown>;
    // A dark template would resurrect light-on-dark defaults under our overrides.
    delete base.template;

    const axisKeys = Object.keys(base).filter((key) => /^[xy]axis\d*$/.test(key));
    for (const required of ["xaxis", "yaxis"]) {
        if (!axisKeys.includes(required)) axisKeys.push(required);
    }
    for (const key of axisKeys) {
        const axis = asRecord(base[key]);
        const axisTitle = asRecord(axis.title);
        base[key] = {
            ...axis,
            gridcolor: EXPORT_GRID_COLOR,
            zerolinecolor: EXPORT_AXIS_COLOR,
            linecolor: EXPORT_AXIS_COLOR,
            tickcolor: EXPORT_AXIS_COLOR,
            tickfont: withFontColor(axis.tickfont, EXPORT_TEXT_COLOR),
            title: { ...axisTitle, font: withFontColor(axisTitle.font, EXPORT_TEXT_COLOR) },
        };
    }

    // Annotations keep deliberately colored text; anything light enough to
    // wash out on white (incl. the slate-400 spectral-line labels) is darkened
    // (colorless ones inherit the global font, which is dark below).
    if (Array.isArray(base.annotations)) {
        base.annotations = base.annotations.map((entry) => {
            const annotation = asRecord(entry);
            const font = asRecord(annotation.font);
            return isLowContrastOnWhite(font.color)
                ? { ...annotation, font: { ...font, color: EXPORT_TEXT_COLOR } }
                : annotation;
        });
    }

    // Guide shapes (spectral-line markers etc.) use slate strokes tuned for
    // the dark theme — re-ink light ones so they survive on white.
    if (Array.isArray(base.shapes)) {
        base.shapes = base.shapes.map((entry) => {
            const shape = asRecord(entry);
            const line = asRecord(shape.line);
            return isLowContrastOnWhite(line.color)
                ? { ...shape, line: { ...line, color: EXPORT_AXIS_COLOR } }
                : shape;
        });
    }

    const title = typeof base.title === "string" ? { text: base.title } : asRecord(base.title);
    return {
        ...base,
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: withFontColor(base.font, EXPORT_TEXT_COLOR),
        title: { ...title, font: withFontColor(title.font, EXPORT_TEXT_COLOR) },
        legend: {
            ...asRecord(base.legend),
            bgcolor: "rgba(255, 255, 255, 0.9)",
            bordercolor: EXPORT_GRID_COLOR,
            font: withFontColor(asRecord(base.legend).font, EXPORT_TEXT_COLOR),
        },
    };
}

/** Re-map only trivially unsafe trace colors (white/near-white lines or
 *  markers → dark slate). Full palette remapping is deliberately out of
 *  scope — the white background + dark text/axes carry the fix. */
function lightExportData(data: unknown[]): unknown[] {
    return data.map((trace) => {
        const record = asRecord(trace);
        const line = asRecord(record.line);
        const marker = asRecord(record.marker);
        const remapLine = isNearWhite(line.color);
        const remapMarker = isNearWhite(marker.color);
        if (!remapLine && !remapMarker) return trace;
        const out: Record<string, unknown> = { ...record };
        if (remapLine) out.line = { ...line, color: EXPORT_TEXT_COLOR };
        if (remapMarker) out.marker = { ...marker, color: EXPORT_TEXT_COLOR };
        return out;
    });
}

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

    const layoutHeight = typeof spec?.layout?.height === "number" ? (spec.layout.height as number) : 420;

    useEffect(() => {
        if (!isValidSpec(spec)) {
            setStatus("fallback");
            return;
        }
        const el = plotRef.current;
        if (!el) return;
        let cancelled = false;

        (async () => {
            try {
                const mod = await import("plotly.js-basic-dist-min");
                const Plotly = mod.default ?? (mod as unknown as PlotlyModule);
                if (cancelled) return;
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
                await Plotly.newPlot(el, spec.data, darkenLayout(spec.layout), config);
                if (cancelled) return;
                setStatus("ready");
            } catch {
                if (!cancelled) setStatus("fallback");
            }
        })();

        return () => {
            cancelled = true;
            if (plotlyRef.current) {
                try { plotlyRef.current.purge(el); } catch { /* already gone */ }
            }
        };
    }, [spec, title, pngFallback]);

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
