/* eslint-disable @next/next/no-img-element */
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Maximize2, Minus, Plus, RefreshCw, X } from "lucide-react";
import { AladinSkyView, SURVEYS } from "./AladinSkyView";
import type { HipsImageMeta } from "../lib/types";
import { clampHipsFov, hips2fitsUrl, normalizeHipsSurveyId } from "../lib/hips-imagery";

type HipsImageCardProps = {
    imageUrl: string;
    caption?: string;
    imageMeta?: HipsImageMeta;
};

function FullSizeLink({ href }: { href: string }) {
    return (
        <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1"
        >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
            Full size
        </a>
    );
}

function PlainImageCard({ imageUrl, caption }: { imageUrl: string; caption?: string }) {
    return (
        <div className="pl-11">
            <div className="mt-4 rounded-xl border border-cyan-500/30 bg-slate-900/60 overflow-hidden shadow-xl shadow-cyan-500/5">
                <img
                    src={imageUrl}
                    alt={caption || "Rendered FITS image"}
                    className="w-full max-h-[600px] object-contain bg-black"
                    loading="lazy"
                />
                {caption && (
                    <div className="px-4 py-2.5 border-t border-slate-700/50 flex items-center justify-between">
                        <span className="text-xs text-slate-400">{caption}</span>
                        <FullSizeLink href={imageUrl} />
                    </div>
                )}
            </div>
        </div>
    );
}

function cacheBusted(url: string) {
    try {
        const parsed = new URL(url);
        parsed.searchParams.set("retry", String(Date.now()));
        return parsed.toString();
    } catch {
        const separator = url.includes("?") ? "&" : "?";
        return `${url}${separator}retry=${Date.now()}`;
    }
}

export function HipsImageCard({ imageUrl, caption = "", imageMeta }: HipsImageCardProps) {
    const kind = imageMeta?.kind || "";
    const ra = Number(imageMeta?.ra);
    const dec = Number(imageMeta?.dec);
    const hasCoords = Number.isFinite(ra) && Number.isFinite(dec);
    const baseSurvey = normalizeHipsSurveyId(imageMeta?.survey) || SURVEYS[0]?.id || "P/DSS2/color";
    const baseFov = clampHipsFov(imageMeta?.fovDeg ?? 0.25);
    const canSwitchSurvey = kind === "hips" && hasCoords;
    const canInteract = (kind === "hips" || kind === "hips_panel") && hasCoords;

    const [displayedUrl, setDisplayedUrl] = useState(imageUrl);
    const [selectedSurvey, setSelectedSurvey] = useState(baseSurvey);
    const [fovDeg, setFovDeg] = useState(baseFov);
    const [loading, setLoading] = useState(false);
    const [failed, setFailed] = useState(false);
    const [interactiveOpen, setInteractiveOpen] = useState(false);


    useEffect(() => {
        if (!interactiveOpen) return;
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") setInteractiveOpen(false);
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [interactiveOpen]);

    const surveyOptions = useMemo(() => {
        if (!selectedSurvey || SURVEYS.some((option) => option.id === selectedSurvey)) return SURVEYS;
        return [{ id: selectedSurvey, label: selectedSurvey }, ...SURVEYS];
    }, [selectedSurvey]);

    const applyClientImage = useCallback((survey: string, nextFov: number, retry = false) => {
        const nextUrl = hips2fitsUrl({ ra, dec, fovDeg: nextFov, survey });
        if (!nextUrl) {
            setFailed(true);
            setLoading(false);
            return;
        }
        setFailed(false);
        setLoading(true);
        setDisplayedUrl(retry ? cacheBusted(nextUrl) : nextUrl);
    }, [ra, dec]);

    if (!canSwitchSurvey && !canInteract) {
        return <PlainImageCard imageUrl={imageUrl} caption={caption} />;
    }

    return (
        <div className="pl-11">
            <div className="mt-4 rounded-xl border border-cyan-500/30 bg-slate-900/60 overflow-hidden shadow-xl shadow-cyan-500/5">
                <div className="relative bg-black">
                    <img
                        src={displayedUrl}
                        alt={caption || "Rendered FITS image"}
                        className="w-full max-h-[600px] object-contain bg-black"
                        loading="lazy"
                        onLoad={() => setLoading(false)}
                        onError={() => {
                            setLoading(false);
                            setFailed(true);
                        }}
                    />
                    {loading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/45 text-xs font-medium text-slate-200 backdrop-blur-sm">
                            Loading...
                        </div>
                    )}
                </div>
                <div className="px-4 py-2.5 border-t border-slate-700/50 flex flex-wrap items-center justify-between gap-3">
                    <div className="min-w-0">
                        {caption && <span className="block truncate text-xs text-slate-400">{caption}</span>}
                        {failed && (
                            <div className="mt-1 flex items-center gap-2 text-[11px] text-amber-300">
                                <span>no coverage / failed to load</span>
                                {canSwitchSurvey && (
                                    <button
                                        type="button"
                                        onClick={() => applyClientImage(selectedSurvey, fovDeg, true)}
                                        className="inline-flex items-center gap-1 rounded-md border border-amber-300/30 px-1.5 py-0.5 text-[11px] text-amber-200 transition-colors hover:bg-amber-300/10"
                                    >
                                        <RefreshCw className="h-3 w-3" />
                                        Retry
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="flex flex-wrap items-center justify-end gap-2">
                        {canSwitchSurvey && (
                            <>
                                <select
                                    aria-label="Sky survey"
                                    value={selectedSurvey}
                                    onChange={(event) => {
                                        const survey = event.target.value;
                                        setSelectedSurvey(survey);
                                        applyClientImage(survey, fovDeg);
                                    }}
                                    className="max-w-[190px] rounded-md border border-slate-700/70 bg-slate-950/80 px-2 py-1 text-[11px] font-medium text-slate-100 outline-none transition-colors focus:border-cyan-400"
                                >
                                    {surveyOptions.map((option) => (
                                        <option key={option.id} value={option.id}>{option.label}</option>
                                    ))}
                                </select>
                                <div className="inline-flex overflow-hidden rounded-md border border-slate-700/70 bg-slate-950/60">
                                    <button
                                        type="button"
                                        aria-label="Zoom out"
                                        title="Zoom out"
                                        onClick={() => {
                                            const nextFov = clampHipsFov(fovDeg * 2);
                                            setFovDeg(nextFov);
                                            applyClientImage(selectedSurvey, nextFov);
                                        }}
                                        className="p-1.5 text-slate-300 transition-colors hover:bg-slate-800 hover:text-cyan-200"
                                    >
                                        <Minus className="h-3.5 w-3.5" />
                                    </button>
                                    <button
                                        type="button"
                                        aria-label="Zoom in"
                                        title="Zoom in"
                                        onClick={() => {
                                            const nextFov = clampHipsFov(fovDeg / 2);
                                            setFovDeg(nextFov);
                                            applyClientImage(selectedSurvey, nextFov);
                                        }}
                                        className="border-l border-slate-700/70 p-1.5 text-slate-300 transition-colors hover:bg-slate-800 hover:text-cyan-200"
                                    >
                                        <Plus className="h-3.5 w-3.5" />
                                    </button>
                                </div>
                            </>
                        )}
                        {canInteract && (
                            <button
                                type="button"
                                onClick={() => setInteractiveOpen(true)}
                                className="inline-flex items-center gap-1.5 rounded-md border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[11px] font-semibold text-cyan-200 transition-colors hover:bg-cyan-500/15"
                            >
                                <Maximize2 className="h-3.5 w-3.5" />
                                Interactive
                            </button>
                        )}
                        <FullSizeLink href={displayedUrl} />
                    </div>
                </div>
            </div>

            {interactiveOpen && canInteract && (
                <div
                    role="dialog"
                    aria-modal="true"
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-lg"
                    onClick={() => setInteractiveOpen(false)}
                >
                    <div
                        className="glass-surface relative rounded-2xl p-4 w-[min(900px,calc(100vw-32px))] h-[min(720px,calc(100vh-32px))]"
                        onClick={(event) => event.stopPropagation()}
                    >
                        <button
                            type="button"
                            onClick={() => setInteractiveOpen(false)}
                            className="absolute -top-2 -right-2 z-10 rounded-full border border-slate-600 bg-slate-800 p-1 transition-colors hover:bg-red-500/20"
                            aria-label="Close interactive image"
                        >
                            <X className="h-4 w-4 text-slate-300" />
                        </button>
                        <div className="h-full overflow-hidden rounded-xl border border-slate-700/50 bg-slate-950/80">
                            <AladinSkyView
                                coords={[{ ra, dec }]}
                                sourceName={caption || "HiPS image"}
                                survey={selectedSurvey}
                                onFallback={() => setInteractiveOpen(false)}
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
