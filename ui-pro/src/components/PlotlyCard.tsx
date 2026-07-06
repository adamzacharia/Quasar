/* eslint-disable @next/next/no-img-element */
"use client";

import { useEffect, useRef, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import type { PlotlyModule } from "plotly.js-basic-dist-min";
import type { PlotlyFigureSpec } from "../lib/types";
import { ImageLightbox } from "./ImageLightbox";

type PlotlyCardProps = {
    spec?: PlotlyFigureSpec;
    title?: string;
    pngFallback?: string;
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

function fileSlug(title?: string): string {
    const slug = (title || "quasar_plot").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
    return slug || "quasar_plot";
}

/**
 * Interactive Plotly figure card for "plotly" SSE events. Loads the
 * lightweight basic bundle client-side only (dynamic import in useEffect,
 * never in the SSR bundle) and falls back to the PNG snapshot as a regular
 * image card if the bundle fails to load or the spec is invalid.
 */
export function PlotlyCard({ spec, title, pngFallback }: PlotlyCardProps) {
    const plotRef = useRef<HTMLDivElement>(null);
    const plotlyRef = useRef<PlotlyModule | null>(null);
    const [status, setStatus] = useState<"loading" | "ready" | "fallback">(
        () => (isValidSpec(spec) ? "loading" : "fallback")
    );
    const [downloading, setDownloading] = useState(false);
    const [lightboxOpen, setLightboxOpen] = useState(false);

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
                await Plotly.newPlot(el, spec.data, darkenLayout(spec.layout), {
                    responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ["lasso2d"],
                    toImageButtonOptions: { format: "png", filename: fileSlug(title), scale: 2 },
                });
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
    }, [spec, title]);

    const downloadPng = async () => {
        const Plotly = plotlyRef.current;
        const el = plotRef.current;
        if (!Plotly || !el || downloading) return;
        setDownloading(true);
        try {
            const dataUrl = await Plotly.toImage(el, { format: "png", scale: 2 });
            const anchor = document.createElement("a");
            anchor.href = dataUrl;
            anchor.download = `${fileSlug(title)}.png`;
            document.body.appendChild(anchor);
            anchor.click();
            document.body.removeChild(anchor);
        } catch {
            // Export is best-effort; the modebar camera button remains available.
        } finally {
            setDownloading(false);
        }
    };

    // ── Fallback: static PNG image card (or a quiet note if none given) ──
    if (status === "fallback") {
        if (!pngFallback) {
            return (
                <div className="pl-11">
                    <div className="mt-4 rounded-xl border border-slate-700/50 bg-slate-900/60 px-4 py-3 text-xs text-slate-400">
                        Interactive plot could not be rendered{title ? `: ${title}` : "."}
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
                        <a
                            href={pngFallback}
                            download
                            className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-1 text-[11px] font-semibold text-indigo-200 transition-colors hover:bg-indigo-500/15"
                        >
                            <Download className="h-3.5 w-3.5" />
                            PNG
                        </a>
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
                <div className="px-4 py-2.5 border-t border-slate-700/50 flex items-center justify-between gap-3">
                    <span className="min-w-0 truncate text-xs text-slate-400">{title || "Interactive plot"}</span>
                    <button
                        type="button"
                        onClick={downloadPng}
                        disabled={status !== "ready" || downloading}
                        className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-1 text-[11px] font-semibold text-indigo-200 transition-colors hover:bg-indigo-500/15 disabled:opacity-40"
                        title="Download as PNG"
                    >
                        {downloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                        PNG
                    </button>
                </div>
            </div>
        </div>
    );
}
