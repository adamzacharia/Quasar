"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { loadAladin } from "../lib/aladin-loader";
import { skyViewGeometry } from "../lib/sky-geometry";

type SkyCoord = { ra: number; dec: number; i?: number; label?: string };

export type SkySourceData = { ra: number; dec: number; i?: number; label?: string };

export type MocOverlay = {
    id: string;
    name?: string;
    color?: string;
    order?: number;
    mocJson: Record<string, number[]>;
};

export type StcsFootprint = {
    /** STC-S region string (obscore s_region), e.g. "POLYGON ICRS ...". */
    stcs: string;
    i?: number;
    label?: string;
};

type AladinSkyViewProps = {
    coords: SkyCoord[];
    sourceName?: string;
    survey?: string;
    className?: string;
    /** Survey footprint overlays (MOC coverage maps) drawn on the view. */
    mocs?: MocOverlay[];
    /** Per-observation STC-S footprints (obscore s_region) drawn as polygons. */
    footprints?: StcsFootprint[];
    onFallback?: (reason: string) => void;
    /** Called with the marker's data payload when a source is clicked
     *  (click-to-inspect); called with null when empty sky is clicked. */
    onSourceSelect?: (data: SkySourceData | null) => void;
};

type AladinInstance = {
    addCatalog?: (catalog: unknown) => void;
    addMOC?: (moc: unknown) => void;
    addOverlay?: (overlay: unknown) => void;
    gotoRaDec?: (ra: number, dec: number) => void;
    setImageSurvey?: (survey: string) => void;
    on?: (event: string, callback: (payload: unknown) => void) => void;
    remove?: () => void;
};

type AladinCatalog = {
    addSources?: (sources: unknown[]) => void;
};

type AladinOverlay = {
    addFootprints?: (footprints: unknown) => void;
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
    mocs,
    footprints,
    onFallback,
    onSourceSelect,
}: AladinSkyViewProps) {
    const geometry = useMemo(() => skyViewGeometry(coords, { maxMarkers: 2000 }), [coords]);
    const reason = fallbackMessage(geometry);
    const [selectedSurvey, setSelectedSurvey] = useState(() => survey || DEFAULT_SURVEY);
    const [loading, setLoading] = useState(true);
    const containerRef = useRef<HTMLDivElement>(null);
    const aladinRef = useRef<AladinInstance | null>(null);
    const selectedSurveyRef = useRef(selectedSurvey);
    const onFallbackRef = useRef(onFallback);
    const onSourceSelectRef = useRef(onSourceSelect);

    useEffect(() => {
        onFallbackRef.current = onFallback;
    }, [onFallback]);

    useEffect(() => {
        onSourceSelectRef.current = onSourceSelect;
    }, [onSourceSelect]);

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
                    hoverColor: "#67e8f9",
                    selectionColor: "#fbbf24",
                }) as AladinCatalog;
                instance.addCatalog?.(catalog);
                // Each source carries its data payload (row index + label) so a
                // click can open the matching table row (click-to-inspect).
                catalog.addSources?.(
                    geometry.markers.map((coord) =>
                        A.source(coord.ra, coord.dec, {
                            ra: coord.ra,
                            dec: coord.dec,
                            ...(coord.i !== undefined ? { i: coord.i } : {}),
                            ...(coord.label !== undefined ? { label: coord.label } : {}),
                        })
                    )
                );
                instance.on?.("objectClicked", (payload: unknown) => {
                    const source = payload as { data?: SkySourceData } | null;
                    onSourceSelectRef.current?.(source && source.data ? source.data : null);
                });

                // Survey footprint overlays (MOC coverage maps).
                for (const m of mocs || []) {
                    try {
                        const moc = A.MOCFromJSON(m.mocJson, {
                            name: m.name || m.id,
                            color: m.color || "#22d3ee",
                            opacity: 0.35,
                            fill: true,
                            perimeter: true,
                            lineWidth: 1.5,
                        });
                        instance.addMOC?.(moc);
                    } catch {
                        // Skip a malformed MOC rather than failing the whole view.
                    }
                }

                // Per-observation footprints (obscore s_region STC-S polygons).
                // All footprints share one graphic overlay layer; each region is
                // parsed independently so one malformed STC-S string is skipped
                // rather than dropping the whole layer. (T7.2)
                if (footprints && footprints.length > 0) {
                    try {
                        const overlay = A.graphicOverlay({
                            name: "Observation footprints",
                            color: "#fbbf24",
                            lineWidth: 1.2,
                        }) as AladinOverlay;
                        instance.addOverlay?.(overlay);
                        for (const fp of footprints) {
                            try {
                                overlay.addFootprints?.(A.footprintsFromSTCS(fp.stcs));
                            } catch {
                                // Skip a region Aladin cannot parse.
                            }
                        }
                    } catch {
                        // Older Aladin builds may lack graphicOverlay/footprintsFromSTCS.
                    }
                }

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
    }, [geometry, reason, sourceName, mocs, footprints]);

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
                className="absolute right-3 top-3 z-20 max-w-[190px] rounded-md border border-sky-400/50 bg-slate-950/90 px-2 py-1 text-[11px] font-medium text-white shadow-lg outline-none backdrop-blur focus:border-sky-300"
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
            {mocs && mocs.length > 0 && (
                <div className="absolute bottom-3 left-3 z-20 flex flex-wrap gap-1.5">
                    {mocs.map((m) => (
                        <span
                            key={m.id}
                            title={m.id}
                            className="inline-flex items-center gap-1.5 rounded-md border border-slate-600/70 bg-slate-950/85 px-2 py-1 text-[10px] font-medium text-slate-200 shadow-lg backdrop-blur"
                        >
                            <span
                                className="inline-block h-2 w-2 rounded-[2px]"
                                style={{ background: m.color || "#22d3ee" }}
                            />
                            {m.name || m.id}
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
}