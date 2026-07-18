/* eslint-disable @next/next/no-img-element */
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { ImageLightbox } from "./ImageLightbox";
import type { BlinkFrame } from "../lib/types";

type BlinkCardProps = {
    frames: BlinkFrame[];
    caption?: string;
    /** Static summary image (e.g. the epoch panel) shown under the player. */
    summaryUrl?: string;
    /** Registration/coverage caveats from the epoch service (CX-18): an
     *  unregistered frame with no on-card note reads as a real transient. */
    warnings?: string[];
};

const RATES = [
    { label: "0.5×", ms: 1400 },
    { label: "1×", ms: 700 },
    { label: "2×", ms: 350 },
];

/**
 * Multi-frame blink comparator: ordered, WCS-matched frames (survey epochs,
 * ZTF stamp triplets, DSS1 vs DSS2) played in place so the eye catches what
 * moved or brightened — the canonical human transient-vetting move that a
 * static side-by-side panel defeats.
 */
export function BlinkCard({ frames, caption, summaryUrl, warnings }: BlinkCardProps) {
    const [index, setIndex] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [rateIdx, setRateIdx] = useState(1);
    const [lightboxOpen, setLightboxOpen] = useState(false);
    const preloaded = useRef(false);

    // Resolve server-relative frame URLs to the backend origin. onImage already
    // does this for live SSE, but a reloaded conversation rehydrates raw
    // metadata from the store, so BlinkCard must resolve defensively too (CX-03).
    const safeFrames = useMemo(() => {
        const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        return (frames || [])
            .filter((frame) => frame && frame.url)
            .map((frame) => ({
                ...frame,
                url:
                    frame.url.startsWith("http") || frame.url.startsWith("data:")
                        ? frame.url
                        : `${apiBase}${frame.url}`,
            }));
    }, [frames]);
    const count = safeFrames.length;
    const current = safeFrames[Math.min(index, Math.max(0, count - 1))];

    // Preload every frame once so blinking never flashes an empty image.
    useEffect(() => {
        if (preloaded.current || count === 0 || typeof window === "undefined") return;
        preloaded.current = true;
        for (const frame of safeFrames) {
            const img = new window.Image();
            img.src = frame.url;
        }
    }, [safeFrames, count]);

    useEffect(() => {
        if (!playing || count < 2) return;
        const timer = window.setInterval(
            () => setIndex((value) => (value + 1) % count),
            RATES[rateIdx].ms
        );
        return () => window.clearInterval(timer);
    }, [playing, rateIdx, count]);

    const step = useCallback(
        (delta: number) => {
            setPlaying(false);
            setIndex((value) => (value + delta + count) % count);
        },
        [count]
    );

    // Space toggles play, arrows step — the DS9-blink muscle memory.
    const onKeyDown = useCallback(
        (event: React.KeyboardEvent) => {
            if (event.key === " ") {
                event.preventDefault();
                setPlaying((value) => !value);
            } else if (event.key === "ArrowRight") {
                event.preventDefault();
                step(1);
            } else if (event.key === "ArrowLeft") {
                event.preventDefault();
                step(-1);
            }
        },
        [step]
    );

    if (count === 0) return null;

    return (
        <div className="pl-11">
            <div
                className="mt-4 rounded-xl border border-cyan-500/30 bg-slate-900/60 overflow-hidden shadow-xl shadow-cyan-500/5 outline-none"
                tabIndex={0}
                onKeyDown={onKeyDown}
                aria-label="Blink comparator — space to play/pause, arrow keys to step"
            >
                <div className="relative bg-black">
                    <img
                        src={current.url}
                        alt={current.label || `Frame ${index + 1}`}
                        className="w-full max-h-[540px] object-contain bg-black cursor-zoom-in"
                        onClick={() => setLightboxOpen(true)}
                    />
                    <span className="absolute left-3 top-3 rounded-md border border-slate-600/70 bg-slate-950/85 px-2 py-1 text-[11px] font-semibold text-cyan-200 shadow-lg backdrop-blur">
                        {current.label || `Frame ${index + 1}/${count}`}
                    </span>
                </div>
                <div className="px-4 py-2.5 border-t border-slate-700/50 flex flex-wrap items-center justify-between gap-3">
                    <span className="min-w-0 truncate text-xs text-slate-400">{caption}</span>
                    <div className="flex shrink-0 items-center gap-2">
                        <div className="inline-flex overflow-hidden rounded-md border border-slate-700/70 bg-slate-950/60">
                            <button
                                type="button"
                                aria-label="Previous frame"
                                onClick={() => step(-1)}
                                className="p-1.5 text-slate-300 transition-colors hover:bg-slate-800 hover:text-cyan-200"
                            >
                                <ChevronLeft className="h-3.5 w-3.5" />
                            </button>
                            <button
                                type="button"
                                aria-label={playing ? "Pause blinking" : "Blink frames"}
                                onClick={() => setPlaying((value) => !value)}
                                className={`border-x border-slate-700/70 p-1.5 transition-colors hover:bg-slate-800 ${
                                    playing ? "text-cyan-300" : "text-slate-300 hover:text-cyan-200"
                                }`}
                            >
                                {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                            </button>
                            <button
                                type="button"
                                aria-label="Next frame"
                                onClick={() => step(1)}
                                className="p-1.5 text-slate-300 transition-colors hover:bg-slate-800 hover:text-cyan-200"
                            >
                                <ChevronRight className="h-3.5 w-3.5" />
                            </button>
                        </div>
                        <button
                            type="button"
                            onClick={() => setRateIdx((value) => (value + 1) % RATES.length)}
                            className="rounded-md border border-slate-700/70 bg-slate-950/60 px-2 py-1 text-[11px] font-medium text-slate-200 transition-colors hover:bg-slate-800"
                            title="Blink rate"
                        >
                            {RATES[rateIdx].label}
                        </button>
                        <div className="flex items-center gap-1">
                            {safeFrames.map((frame, k) => (
                                <button
                                    key={`${frame.url}-${k}`}
                                    type="button"
                                    aria-label={frame.label || `Frame ${k + 1}`}
                                    title={frame.label || `Frame ${k + 1}`}
                                    onClick={() => {
                                        setPlaying(false);
                                        setIndex(k);
                                    }}
                                    className={`h-2 w-2 rounded-full transition-colors ${
                                        k === index ? "bg-cyan-300" : "bg-slate-600 hover:bg-slate-400"
                                    }`}
                                />
                            ))}
                        </div>
                    </div>
                </div>
            </div>
            {warnings && warnings.length > 0 && (
                <div className="mt-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11px] leading-relaxed text-amber-200">
                    {warnings.map((w, i) => (
                        <div key={i}>⚠ {w}</div>
                    ))}
                </div>
            )}
            {summaryUrl && (
                <details className="mt-2">
                    <summary className="cursor-pointer text-[11px] font-medium text-slate-400 hover:text-slate-200">
                        Show all epochs side by side
                    </summary>
                    <img
                        src={summaryUrl}
                        alt="All frames side by side"
                        className="mt-2 w-full rounded-xl border border-slate-700/50 bg-black"
                        loading="lazy"
                    />
                </details>
            )}
            {lightboxOpen && (
                <ImageLightbox
                    src={current.url}
                    caption={current.label || caption}
                    onClose={() => setLightboxOpen(false)}
                />
            )}
        </div>
    );
}
