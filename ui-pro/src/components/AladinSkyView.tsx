"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { loadAladin } from "../lib/aladin-loader";
import { skyViewGeometry } from "../lib/sky-geometry";

type SkyCoord = { ra: number; dec: number };

type AladinSkyViewProps = {
    coords: SkyCoord[];
    sourceName?: string;
    survey?: string;
    className?: string;
    onFallback?: (reason: string) => void;
};

type AladinInstance = {
    addCatalog?: (catalog: unknown) => void;
    gotoRaDec?: (ra: number, dec: number) => void;
    setImageSurvey?: (survey: string) => void;
    remove?: () => void;
};

type AladinCatalog = {
    addSources?: (sources: unknown[]) => void;
};

export const DEFAULT_SURVEY = "P/DSS2/color";
export const SURVEYS = [
    { label: "DSS2 color", id: DEFAULT_SURVEY },
    { label: "SDSS9", id: "P/SDSS9/color" },
    { label: "AllWISE", id: "P/allWISE/color" },
    { label: "GALEX", id: "P/GALEXGR6/AIS/color" },
    { label: "RASS X-ray", id: "P/RASS" },
    { label: "VLASS 3 GHz", id: "NRAO/P/VLASS-Quicklook-MedianStack" },
];

function fallbackMessage(geometry: ReturnType<typeof skyViewGeometry>) {
    if (!geometry.ok) return "No valid sky positions are available for the interactive view.";
    if (geometry.allSky) return "The positions span too much of the sky for the interactive view.";
    return null;
}

function messageFromError(error: unknown) {
    return error instanceof Error && error.message ? error.message : "Aladin Lite could not start.";
}

export function AladinSkyView({
    coords,
    sourceName,
    survey,
    className = "",
    onFallback,
}: AladinSkyViewProps) {
    const geometry = useMemo(() => skyViewGeometry(coords, { maxMarkers: 2000 }), [coords]);
    const reason = fallbackMessage(geometry);
    const [selectedSurvey, setSelectedSurvey] = useState(() => survey || DEFAULT_SURVEY);
    const [loading, setLoading] = useState(true);
    const containerRef = useRef<HTMLDivElement>(null);
    const aladinRef = useRef<AladinInstance | null>(null);
    const selectedSurveyRef = useRef(selectedSurvey);
    const onFallbackRef = useRef(onFallback);

    useEffect(() => {
        onFallbackRef.current = onFallback;
    }, [onFallback]);

    useEffect(() => {
        if (reason) {
            onFallbackRef.current?.(reason);
        }
    }, [reason]);

    useEffect(() => {
        selectedSurveyRef.current = selectedSurvey;
        aladinRef.current?.setImageSurvey?.(selectedSurvey);
    }, [selectedSurvey]);

    useEffect(() => {
        if (reason) return;

        const container = containerRef.current;
        if (!container) return;
        const stableContainer = container;

        let cancelled = false;

        async function startAladin() {
            try {
                const A = await loadAladin();
                if (cancelled) return;

                stableContainer.replaceChildren();
                const instance = A.aladin(stableContainer, {
                    survey: selectedSurveyRef.current,
                    cooFrame: "ICRSd",
                    fov: geometry.fovDeg,
                    showFullscreenControl: false,
                    showLayersControl: false,
                    showShareControl: false,
                    showFrame: false,
                    showCooGrid: false,
                    showProjectionControl: false,
                }) as AladinInstance;

                if (cancelled) {
                    instance.remove?.();
                    stableContainer.replaceChildren();
                    return;
                }

                aladinRef.current = instance;
                instance.gotoRaDec?.(geometry.raCenter, geometry.decCenter);

                const catalog = A.catalog({
                    name: sourceName || "results",
                    sourceSize: 12,
                    color: "#22d3ee",
                    shape: "circle",
                }) as AladinCatalog;
                instance.addCatalog?.(catalog);
                catalog.addSources?.(geometry.markers.map((coord) => A.source(coord.ra, coord.dec)));

                setLoading(false);
            } catch (error) {
                if (!cancelled) {
                    onFallbackRef.current?.(messageFromError(error));
                }
            }
        }

        void startAladin();

        return () => {
            cancelled = true;
            const instance = aladinRef.current;
            aladinRef.current = null;
            instance?.remove?.();
            stableContainer.replaceChildren();
        };
    }, [geometry, reason, sourceName]);

    if (reason) return null;

    return (
        <div className={`relative h-full w-full bg-slate-950 ${className}`}>
            <div ref={containerRef} className="h-full w-full" />
            {loading && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-950/85 text-xs font-medium text-slate-300">
                    Loading sky imagery...
                </div>
            )}
            <select
                aria-label="Sky survey"
                value={selectedSurvey}
                onChange={(event) => setSelectedSurvey(event.target.value)}
                className="absolute right-3 top-3 z-20 max-w-[190px] rounded-md border border-slate-600/70 bg-slate-950/90 px-2 py-1 text-[11px] font-medium text-slate-100 shadow-lg outline-none backdrop-blur focus:border-cyan-400"
            >
                {SURVEYS.map((option) => (
                    <option key={option.id} value={option.id}>
                        {option.label}
                    </option>
                ))}
            </select>
            {geometry.truncated && (
                <div className="absolute right-3 top-12 z-20 rounded-md border border-slate-600/70 bg-slate-950/85 px-2 py-1 text-[10px] font-medium text-slate-200 shadow-lg backdrop-blur">
                    showing first 2000 of {geometry.count} positions
                </div>
            )}
        </div>
    );
}