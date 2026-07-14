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
    /** Override the auto-computed field of view (deg) — lets cards hand the
     *  zoom the user was already looking at to the interactive view. */
    fovDeg?: number;
    /** Survey footprint overlays (MOC coverage maps) drawn on the view. */
    mocs?: MocOverlay[];
    /** Per-observation STC-S footprints (obscore s_region) drawn as polygons. */
    footprints?: StcsFootprint[];
    /** Show the overlay-survey blend controls (second layer + opacity). */
    showBlendControls?: boolean;
    /** Show the SIMBAD / Gaia progressive-catalog toggles. */
    showCatalogToggles?: boolean;
    onFallback?: (reason: string) => void;
    /** Called with the marker's data payload when a source is clicked
     *  (click-to-inspect); called with null when empty sky is clicked. */
    onSourceSelect?: (data: SkySourceData | null) => void;
};

type AladinImageLayer = {
    setOpacity?: (opacity: number) => void;
};

type AladinInstance = {
    addCatalog?: (catalog: unknown) => void;
    addMOC?: (moc: unknown) => void;
    addOverlay?: (overlay: unknown) => void;
    gotoRaDec?: (ra: number, dec: number) => void;
    setImageSurvey?: (survey: string) => void;
    setOverlayImageLayer?: (survey: unknown) => AladinImageLayer | void;
    getOverlayImageLayer?: () => AladinImageLayer | null;
    removeImageLayer?: (layer: string) => void;
    on?: (event: string, callback: (payload: unknown) => void) => void;
    remove?: () => void;
};

type AladinCatalog = {
    addSources?: (sources: unknown[]) => void;
    show?: () => void;
    hide?: () => void;
};

type AladinOverlay = {
    addFootprints?: (footprints: unknown) => void;
};

/** CDS progressive catalogs (HiPS cat service): render at any zoom with
 *  automatic density thinning — no query round-trip. */
const CATALOG_HIPS = [
    {
        key: "simbad",
        label: "SIMBAD",
        color: "#a78bfa",
        url: "https://axel.u-strasbg.fr/HiPSCatService/SIMBAD",
    },
    {
        key: "gaia",
        label: "Gaia DR3",
        color: "#34d399",
        url: "https://axel.u-strasbg.fr/HiPSCatService/I/355/gaiadr3",
    },
] as const;

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
    fovDeg,
    mocs,
    footprints,
    showBlendControls = true,
    showCatalogToggles = true,
    onFallback,
    onSourceSelect,
}: AladinSkyViewProps) {
    const geometry = useMemo(() => skyViewGeometry(coords, { maxMarkers: 2000 }), [coords]);
    const reason = fallbackMessage(geometry);
    const [selectedSurvey, setSelectedSurvey] = useState(() => survey || DEFAULT_SURVEY);
    const [overlaySurvey, setOverlaySurvey] = useState("");
    const [overlayOpacity, setOverlayOpacity] = useState(0.5);
    const [blinking, setBlinking] = useState(false);
    const [activeCatalogs, setActiveCatalogs] = useState<Record<string, boolean>>({});
    const [unsupportedCatalogs, setUnsupportedCatalogs] = useState<Record<string, boolean>>({});
    const [loading, setLoading] = useState(true);
    const containerRef = useRef<HTMLDivElement>(null);
    const aladinRef = useRef<AladinInstance | null>(null);
    const aladinApiRef = useRef<Record<string, unknown> | null>(null);
    const overlayLayerRef = useRef<AladinImageLayer | null>(null);
    const catalogRefs = useRef<Record<string, AladinCatalog | null>>({});
    const selectedSurveyRef = useRef(selectedSurvey);
    const overlaySurveyRef = useRef(overlaySurvey);
    const overlayOpacityRef = useRef(overlayOpacity);
    const activeCatalogsRef = useRef(activeCatalogs);
    const onFallbackRef = useRef(onFallback);
    const onSourceSelectRef = useRef(onSourceSelect);

    useEffect(() => {
        overlaySurveyRef.current = overlaySurvey;
    }, [overlaySurvey]);
    useEffect(() => {
        overlayOpacityRef.current = overlayOpacity;
    }, [overlayOpacity]);
    useEffect(() => {
        activeCatalogsRef.current = activeCatalogs;
    }, [activeCatalogs]);

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
        // If the base is switched to the current overlay survey, drop the now
        // redundant overlay so the same layer isn't stacked on itself (CX-05).
        if (overlaySurvey && overlaySurvey === selectedSurvey) {
            setOverlaySurvey("");
            setBlinking(false);
        }
    }, [selectedSurvey, overlaySurvey]);

    // Second image layer for multiwavelength blending (Aladin v3 overlay layer).
    useEffect(() => {
        const instance = aladinRef.current;
        if (!instance) return;
        try {
            if (!overlaySurvey) {
                overlayLayerRef.current = null;
                instance.removeImageLayer?.("overlay");
                return;
            }
            const layer = instance.setOverlayImageLayer?.(overlaySurvey);
            const live = (layer as AladinImageLayer) || instance.getOverlayImageLayer?.() || null;
            overlayLayerRef.current = live;
            live?.setOpacity?.(overlayOpacity);
        } catch {
            // Older Aladin builds without overlay layers: control is a no-op.
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [overlaySurvey]);

    useEffect(() => {
        if (!blinking) {
            overlayLayerRef.current?.setOpacity?.(overlaySurvey ? overlayOpacity : 0);
            return;
        }
        let visible = true;
        const timer = window.setInterval(() => {
            visible = !visible;
            overlayLayerRef.current?.setOpacity?.(visible ? overlayOpacity : 0);
        }, 700);
        return () => window.clearInterval(timer);
    }, [blinking, overlayOpacity, overlaySurvey]);

    // Progressive HiPS catalogs (SIMBAD / Gaia DR3): density-thinned at any
    // zoom, rendered natively by Aladin — toggling hides/shows the layer.
    useEffect(() => {
        const instance = aladinRef.current;
        const A = aladinApiRef.current as {
            catalogHiPS?: (url: string, options?: Record<string, unknown>) => unknown;
        } | null;
        if (!instance || !A) return;
        for (const cat of CATALOG_HIPS) {
            const wanted = !!activeCatalogs[cat.key];
            const existing = catalogRefs.current[cat.key];
            if (wanted && !existing) {
                // Mark the toggle unsupported via STATE (not just a ref) so the
                // button actually re-renders disabled instead of staying "active"
                // with nothing drawn (CX-12).
                if (!A.catalogHiPS) {
                    catalogRefs.current[cat.key] = null;
                    setUnsupportedCatalogs((current) =>
                        current[cat.key] ? current : { ...current, [cat.key]: true });
                    continue;
                }
                try {
                    const hipsCatalog = A.catalogHiPS(cat.url, {
                        name: cat.label,
                        color: cat.color,
                        sourceSize: 8,
                        shape: "square",
                    }) as AladinCatalog;
                    instance.addCatalog?.(hipsCatalog);
                    catalogRefs.current[cat.key] = hipsCatalog;
                } catch {
                    catalogRefs.current[cat.key] = null;
                    setUnsupportedCatalogs((current) =>
                        current[cat.key] ? current : { ...current, [cat.key]: true });
                }
            } else if (existing) {
                try {
                    if (wanted) existing.show?.();
                    else existing.hide?.();
                } catch {
                    // keep the view alive even if hide/show is unsupported
                }
            }
        }
    }, [activeCatalogs]);

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
                    fov: fovDeg && fovDeg > 0 ? fovDeg : geometry.fovDeg,
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
                aladinApiRef.current = A as unknown as Record<string, unknown>;
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

                // Re-apply the blend layer + progressive catalogs after a
                // teardown-rebuild cycle (the React state survives, the
                // Aladin instance does not).
                try {
                    if (overlaySurveyRef.current) {
                        const layer = instance.setOverlayImageLayer?.(overlaySurveyRef.current);
                        const live = (layer as AladinImageLayer) || instance.getOverlayImageLayer?.() || null;
                        overlayLayerRef.current = live;
                        live?.setOpacity?.(overlayOpacityRef.current);
                    }
                    const apiWithCatalogs = A as unknown as {
                        catalogHiPS?: (url: string, options?: Record<string, unknown>) => unknown;
                    };
                    for (const cat of CATALOG_HIPS) {
                        if (!activeCatalogsRef.current[cat.key]) continue;
                        // A catalog preselected before init on a build without
                        // catalogHiPS must still be flagged unsupported (CX-04).
                        if (!apiWithCatalogs.catalogHiPS) {
                            catalogRefs.current[cat.key] = null;
                            setUnsupportedCatalogs((current) =>
                                current[cat.key] ? current : { ...current, [cat.key]: true });
                            continue;
                        }
                        const hipsCatalog = apiWithCatalogs.catalogHiPS(cat.url, {
                            name: cat.label,
                            color: cat.color,
                            sourceSize: 8,
                            shape: "square",
                        }) as AladinCatalog;
                        instance.addCatalog?.(hipsCatalog);
                        catalogRefs.current[cat.key] = hipsCatalog;
                    }
                } catch {
                    // Blend/catalog extras must never break the base view.
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
            overlayLayerRef.current = null;
            catalogRefs.current = {};
            instance?.remove?.();
            stableContainer.replaceChildren();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [geometry, reason, sourceName, mocs, footprints, fovDeg]);

    if (reason) return null;

    return (
        <div className={`relative h-full w-full bg-slate-950 ${className}`}>
            <div ref={containerRef} className="h-full w-full" />
            {loading && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-950/85 text-xs font-medium text-slate-300">
                    Loading sky imagery...
                </div>
            )}
            <div className="absolute right-3 top-3 z-20 flex flex-col items-end gap-1.5">
                <select
                    aria-label="Sky survey"
                    value={selectedSurvey}
                    onChange={(event) => setSelectedSurvey(event.target.value)}
                    className="max-w-[190px] rounded-md border border-sky-400/50 bg-slate-950/90 px-2 py-1 text-[11px] font-medium text-white shadow-lg outline-none backdrop-blur focus:border-sky-300"
                >
                    {SURVEYS.map((option) => (
                        <option key={option.id} value={option.id}>
                            {option.label}
                        </option>
                    ))}
                </select>
                {showBlendControls && (
                    <select
                        aria-label="Blend survey overlay"
                        value={overlaySurvey}
                        onChange={(event) => {
                            setOverlaySurvey(event.target.value);
                            if (!event.target.value) setBlinking(false);
                        }}
                        className="max-w-[190px] rounded-md border border-slate-600/70 bg-slate-950/90 px-2 py-1 text-[11px] font-medium text-slate-200 shadow-lg outline-none backdrop-blur focus:border-sky-300"
                    >
                        <option value="">No overlay</option>
                        {SURVEYS.filter((option) => option.id !== selectedSurvey).map((option) => (
                            <option key={option.id} value={option.id}>
                                + {option.label}
                            </option>
                        ))}
                    </select>
                )}
                {showBlendControls && overlaySurvey && (
                    <div className="flex items-center gap-1.5 rounded-md border border-slate-600/70 bg-slate-950/90 px-2 py-1 shadow-lg backdrop-blur">
                        <input
                            aria-label="Overlay opacity"
                            type="range"
                            min={0}
                            max={1}
                            step={0.05}
                            value={overlayOpacity}
                            onChange={(event) => {
                                const value = Number(event.target.value);
                                setOverlayOpacity(value);
                                if (!blinking) overlayLayerRef.current?.setOpacity?.(value);
                            }}
                            className="h-1 w-20 accent-sky-400"
                        />
                        <button
                            type="button"
                            onClick={() => setBlinking((value) => !value)}
                            className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                                blinking
                                    ? "bg-sky-400/90 text-slate-950"
                                    : "bg-slate-800/90 text-slate-200 hover:bg-slate-700"
                            }`}
                            title="Blink the overlay on/off to compare surveys"
                        >
                            Blink
                        </button>
                    </div>
                )}
                {showCatalogToggles && (
                    <div className="flex gap-1.5">
                        {CATALOG_HIPS.map((cat) => {
                            const active = !!activeCatalogs[cat.key];
                            const unsupported = !!unsupportedCatalogs[cat.key];
                            return (
                                <button
                                    key={cat.key}
                                    type="button"
                                    disabled={unsupported}
                                    onClick={() =>
                                        setActiveCatalogs((current) => ({
                                            ...current,
                                            [cat.key]: !current[cat.key],
                                        }))
                                    }
                                    className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[10px] font-medium shadow-lg backdrop-blur ${
                                        active
                                            ? "border-sky-400/60 bg-slate-900/95 text-white"
                                            : "border-slate-600/70 bg-slate-950/85 text-slate-300 hover:text-white"
                                    } ${unsupported ? "opacity-40" : ""}`}
                                    title={`Toggle the ${cat.label} progressive catalog overlay`}
                                >
                                    <span
                                        className="inline-block h-2 w-2 rounded-[2px]"
                                        style={{ background: active ? cat.color : "#475569" }}
                                    />
                                    {cat.label}
                                </button>
                            );
                        })}
                    </div>
                )}
            </div>
            {geometry.truncated && (
                <div className="absolute left-3 top-3 z-20 rounded-md border border-slate-600/70 bg-slate-950/85 px-2 py-1 text-[10px] font-medium text-slate-200 shadow-lg backdrop-blur">
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