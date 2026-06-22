"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
    AlertTriangle,
    ArrowLeft,
    ChevronDown,
    ChevronRight,
    CircleStop,
    Download,
    FlaskConical,
    Loader2,
    Play,
    Radio,
    Search,
    Telescope,
    Waves,
} from "lucide-react";
import { AuthModal } from "@/components/AuthModal";
import { Sidebar } from "@/components/Sidebar";
import {
    SpectralFrequencyRuler,
    SpectralLineTable,
    lineSelectionKey,
} from "@/components/SpectralLineTable";
import { useAuthStore } from "@/lib/auth-store";
import { useChatStore } from "@/lib/store";
import {
    cancelSpectralLineJob,
    downloadSpectralLineExport,
    getSpectralLineJob,
    getSpectralLineMetadata,
    isUnauthorizedApiError,
    searchSpectralSpecies,
    startSpectralLineJob,
    type SpectralCoverageProject,
    type SpectralLineJob,
    type SpectralLineMetadata,
    type SpectralLineRecord,
    type SpectralSpecies,
} from "@/lib/api";

type Mode = "basic" | "advanced";
type ResultTab = "lines" | "coverage" | "confusion" | "exports";

const TERMINAL = new Set(["needs_input", "succeeded", "partial", "failed", "canceled", "orphaned"]);

function numeric(value: unknown, digits = 4): string {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <label className="block">
            <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-wide text-slate-500">{label}</span>
            {children}
        </label>
    );
}

const inputClass = "h-9 w-full rounded-md border border-slate-700 bg-slate-950 px-3 text-xs text-slate-100 outline-none transition focus:border-cyan-500";

function ProjectRow({ project }: { project: SpectralCoverageProject }) {
    const [open, setOpen] = useState(false);
    return (
        <div className="rounded-lg border border-slate-800 bg-slate-950">
            <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-start gap-3 px-4 py-3 text-left">
                {open ? <ChevronDown className="mt-0.5 h-4 w-4 text-cyan-300" /> : <ChevronRight className="mt-0.5 h-4 w-4 text-slate-500" />}
                <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-sm font-semibold text-slate-100">{project.proposal_id}</span>
                        {project.all_lines_full && <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-[9px] font-semibold text-emerald-300">ALL FULL</span>}
                        {!project.all_lines_full && project.covers_all_lines && <span className="rounded bg-amber-500/10 px-2 py-0.5 text-[9px] font-semibold text-amber-300">ALL COVERED</span>}
                    </div>
                    <div className="mt-1 truncate text-xs text-slate-400">{project.obs_title || project.target_name || "ALMA project"}</div>
                    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-slate-500">
                        <span>{project.covered_line_count}/{project.requested_line_count} lines</span>
                        <span>edge {numeric(project.minimum_edge_margin_mhz, 2)} MHz</span>
                        <span>sep {numeric(project.angular_separation_arcsec, 2)}″</span>
                        <span>beam {numeric(project.best_angular_resolution_arcsec, 3)}″</span>
                        <span>{numeric(project.total_exposure_seconds, 0)} s</span>
                    </div>
                </div>
                {project.archive_url && (
                    <a href={project.archive_url} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()} className="text-[10px] font-semibold text-cyan-300 hover:text-cyan-100">
                        Archive
                    </a>
                )}
            </button>
            {open && (
                <div className="space-y-2 border-t border-slate-800 p-3">
                    {project.observations.map((observation, index) => {
                        const matches = Array.isArray(observation.matching_lines) ? observation.matching_lines as Array<Record<string, unknown>> : [];
                        return (
                            <div key={String(observation.observation_id || index)} className="rounded border border-slate-800 bg-slate-900 p-3">
                                <div className="flex flex-wrap justify-between gap-2">
                                    <div>
                                        <div className="font-mono text-[11px] text-slate-200">{String(observation.member_ous_uid || observation.observation_id || "Observation")}</div>
                                        <div className="mt-1 text-[10px] text-slate-500">Array {String(observation.antenna_arrays || "—")} · exposure {numeric(observation.t_exptime, 0)} s</div>
                                    </div>
                                    {observation.archive_url ? <a href={String(observation.archive_url)} target="_blank" rel="noreferrer" className="text-[10px] text-cyan-300">Open MOUS</a> : null}
                                </div>
                                <div className="mt-3 space-y-2">
                                    {matches.map((match, matchIndex) => {
                                        const line = match.line as Record<string, unknown>;
                                        const spws = Array.isArray(match.spws) ? match.spws as Array<Record<string, unknown>> : [];
                                        return (
                                            <div key={matchIndex} className="rounded bg-slate-950 px-3 py-2 text-[10px]">
                                                <div className="font-semibold text-slate-300">
                                                    {String(line?.species || "Line")} {String(line?.transition || "")} · {numeric(line?.observed_frequency_ghz, 6)} GHz
                                                </div>
                                                {spws.map((spw, spwIndex) => (
                                                    <div key={spwIndex} className="mt-1 font-mono text-slate-500">
                                                        {numeric(spw.minimum_ghz, 6)}–{numeric(spw.maximum_ghz, 6)} GHz · {String(spw.classification)} · margin {numeric(spw.edge_margin_mhz, 2)} MHz
                                                    </div>
                                                ))}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

function SpectralLineExplorer() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const sidebarOpen = useChatStore((state) => state.sidebarOpen);
    const token = useAuthStore((state) => state.token);
    const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
    const isInitialized = useAuthStore((state) => state.isInitialized);
    const openAuthModal = useAuthStore((state) => state.openAuthModal);

    const [mode, setMode] = useState<Mode>((searchParams.get("mode") as Mode) || "basic");
    const [metadata, setMetadata] = useState<SpectralLineMetadata | null>(null);
    const [speciesQuery, setSpeciesQuery] = useState(searchParams.get("species_name") || "CO");
    const [speciesMatches, setSpeciesMatches] = useState<SpectralSpecies[]>([]);
    const [selectedSpecies, setSelectedSpecies] = useState<SpectralSpecies[]>([]);
    const [transition, setTransition] = useState(searchParams.get("transition") || "2-1");
    const [band, setBand] = useState(searchParams.get("band") || "6");
    const [windowMin, setWindowMin] = useState(searchParams.get("min") || "");
    const [windowMax, setWindowMax] = useState(searchParams.get("max") || "");
    const [windowUnit, setWindowUnit] = useState(searchParams.get("unit") || "GHz");
    const [additionalWindows, setAdditionalWindows] = useState("");
    const [frame, setFrame] = useState(searchParams.get("frame") || "rest");
    const [target, setTarget] = useState(searchParams.get("target") || "");
    const [redshift, setRedshift] = useState(searchParams.get("z") || "auto");
    const [radialVelocity, setRadialVelocity] = useState("");
    const [velocityConvention, setVelocityConvention] = useState("radio");
    const [version, setVersion] = useState("v3.0");
    const [catalogs, setCatalogs] = useState<string[]>(["JPL", "CDMS", "SLAIM", "LovasNIST"]);
    const [excludedCategories, setExcludedCategories] = useState<string[]>(["atmospheric", "potential", "probable"]);
    const [energyMin, setEnergyMin] = useState("");
    const [energyMax, setEnergyMax] = useState("");
    const [intensityLimit, setIntensityLimit] = useState("");
    const [intensityType, setIntensityType] = useState("CDMS/JPL (log)");
    const [uncertainty, setUncertainty] = useState("");
    const [onlyObserved, setOnlyObserved] = useState(false);
    const [onlyRecommended, setOnlyRecommended] = useState(true);
    const [outputMode, setOutputMode] = useState<"merged" | "raw">("merged");
    const [sortOrder, setSortOrder] = useState("frequency_asc");
    const [job, setJob] = useState<SpectralLineJob | null>(null);
    const [page, setPage] = useState(1);
    const [selectedLineIds, setSelectedLineIds] = useState<string[]>([]);
    const [resultTab, setResultTab] = useState<ResultTab>("lines");
    const [error, setError] = useState("");
    const autorunHandled = useRef("");
    const [rawComparison, setRawComparison] = useState<{ line: SpectralLineRecord; rows: SpectralLineRecord[] } | null>(null);

    const handleSpectralError = useCallback((reason: unknown) => {
        if (isUnauthorizedApiError(reason)) {
            setError("Your session expired — please sign in again.");
            openAuthModal();
            return;
        }
        setError(reason instanceof Error ? reason.message : String(reason));
    }, [openAuthModal]);

    useEffect(() => {
        if (!isInitialized) return;
        if (!isAuthenticated) {
            openAuthModal();
            return;
        }
        getSpectralLineMetadata(token).then(setMetadata).catch(handleSpectralError);
    }, [handleSpectralError, isAuthenticated, isInitialized, openAuthModal, token]);

    useEffect(() => {
        if (!token || !speciesQuery.trim()) return;
        const timer = window.setTimeout(() => {
            searchSpectralSpecies(speciesQuery, 12, token)
                .then((result) => setSpeciesMatches(result.species))
                .catch((reason) => {
                    setSpeciesMatches([]);
                    if (isUnauthorizedApiError(reason)) handleSpectralError(reason);
                });
        }, 250);
        return () => window.clearTimeout(timer);
    }, [handleSpectralError, speciesQuery, token]);

    useEffect(() => {
        const speciesId = searchParams.get("species");
        if (!speciesId || selectedSpecies.length || !token) return;
        searchSpectralSpecies(speciesId, 100, token).then((result) => {
            const selected = result.species.find((item) => String(item.species_id) === speciesId);
            if (selected) {
                setSelectedSpecies([selected]);
                setSpeciesQuery(selected.formula);
            }
        }).catch((reason) => {
            if (isUnauthorizedApiError(reason)) handleSpectralError(reason);
        });
    }, [handleSpectralError, searchParams, selectedSpecies.length, token]);

    const buildQuery = useCallback(() => {
        const primaryWindows = windowMin && windowMax
            ? [{ minimum: Number(windowMin), maximum: Number(windowMax), unit: windowUnit }]
            : [];
        const parsedAdditionalWindows = mode === "advanced"
            ? additionalWindows
                .split(/\r?\n/)
                .map((line) => line.trim())
                .filter(Boolean)
                .map((line) => {
                    const [minimum, maximum, unit = windowUnit] = line.split(/[,\s]+/);
                    return { minimum: Number(minimum), maximum: Number(maximum), unit };
                })
                .filter((item) => Number.isFinite(item.minimum) && Number.isFinite(item.maximum))
            : [];
        const windows = [...primaryWindows, ...parsedAdditionalWindows].slice(0, 30);
        const explicitRedshift = radialVelocity
            ? undefined
            : redshift !== "auto" && redshift !== ""
                ? Number(redshift)
                : undefined;
        return {
            windows,
            alma_bands: windows.length ? [] : [Number(band)],
            species_ids: selectedSpecies.map((item) => item.species_id),
            species_names: selectedSpecies.length ? [] : speciesQuery.trim() ? [speciesQuery.trim()] : [],
            transition: transition.trim() || undefined,
            frame,
            redshift: explicitRedshift,
            radial_velocity_kms: radialVelocity ? Number(radialVelocity) : undefined,
            velocity_convention: velocityConvention,
            version,
            line_lists: catalogs,
            exclude_categories: excludedCategories,
            only_astronomically_observed: onlyObserved,
            only_nrao_recommended: onlyRecommended,
            energy_min: energyMin ? Number(energyMin) : undefined,
            energy_max: energyMax ? Number(energyMax) : undefined,
            energy_type: "eu_k",
            intensity_lower_limit: intensityLimit ? Number(intensityLimit) : undefined,
            intensity_type: intensityLimit ? intensityType : undefined,
            maximum_frequency_uncertainty_mhz: uncertainty ? Number(uncertainty) : undefined,
            output_mode: outputMode,
            sort_order: sortOrder,
            page_size: 100,
        };
    }, [additionalWindows, band, catalogs, energyMax, energyMin, excludedCategories, frame, intensityLimit, intensityType, mode, onlyObserved, onlyRecommended, outputMode, radialVelocity, redshift, selectedSpecies, sortOrder, speciesQuery, transition, uncertainty, velocityConvention, version, windowMax, windowMin, windowUnit]);

    const syncUrl = useCallback((autorun?: string) => {
        const params = new URLSearchParams();
        params.set("mode", mode);
        if (selectedSpecies[0]) params.set("species", String(selectedSpecies[0].species_id));
        else if (speciesQuery) params.set("species_name", speciesQuery);
        if (transition) params.set("transition", transition);
        if (band) params.set("band", band);
        if (windowMin) params.set("min", windowMin);
        if (windowMax) params.set("max", windowMax);
        if (windowUnit) params.set("unit", windowUnit);
        params.set("frame", frame);
        if (target) params.set("target", target);
        params.set("z", redshift || "auto");
        if (autorun) params.set("autorun", autorun);
        router.replace(`/spectral-lines?${params}`, { scroll: false });
    }, [band, frame, mode, redshift, router, selectedSpecies, speciesQuery, target, transition, windowMax, windowMin, windowUnit]);

    const startJob = useCallback(async (operation: SpectralLineJob["operation"], selectedLine?: SpectralLineRecord) => {
        if (!token) {
            openAuthModal();
            return;
        }
        setError("");
        const query = buildQuery();
        let payload: Record<string, unknown> = { mode, query };
        let nextTab: ResultTab = "lines";
        if (operation === "alma_coverage") {
            if (!target.trim()) throw new Error("Target name is required for ALMA coverage");
            payload = {
                ...payload,
                target_name: target.trim(),
                redshift: !radialVelocity && redshift !== "auto" && redshift !== "" ? Number(redshift) : undefined,
                selected_line_ids: selectedLineIds,
                radius_arcsec: 60,
                coverage_mode: "any",
            };
            nextTab = "coverage";
        } else if (operation === "confusion") {
            if (!selectedLine) throw new Error("Select a line for confusion analysis");
            payload = {
                ...payload,
                selected_line: selectedLine,
                redshift: !radialVelocity && redshift !== "auto" && redshift !== "" ? Number(redshift) : undefined,
                window_mhz: 10,
            };
            nextTab = "confusion";
        }
        const started = await startSpectralLineJob(operation, payload, token);
        setJob(started);
        setResultTab(nextTab);
        setPage(1);
        syncUrl(operation === "alma_coverage" ? "coverage" : "search");
    }, [buildQuery, mode, openAuthModal, radialVelocity, redshift, selectedLineIds, syncUrl, target, token]);

    useEffect(() => {
        if (!job || TERMINAL.has(job.status) || !token) return;
        const pollingJobId = job.job_id;
        const timer = window.setInterval(() => {
            getSpectralLineJob(pollingJobId, page, 100, token)
                .then((nextJob) => setJob((current) => current?.job_id === pollingJobId ? nextJob : current))
                .catch(handleSpectralError);
        }, 1500);
        return () => window.clearInterval(timer);
    }, [handleSpectralError, job, page, token]);

    const jobId = job?.job_id;
    const jobStatus = job?.status;
    const jobOperation = job?.operation;
    const requestedDataset =
        jobOperation === "alma_coverage" && resultTab === "lines"
            ? "lines"
            : jobOperation === "alma_coverage" && resultTab === "coverage"
                ? "coverage"
                : jobOperation === "confusion" && resultTab === "confusion"
                    ? "candidates"
                    : jobOperation === "catalog_search" && resultTab === "lines"
                        ? "lines"
                        : undefined;
    useEffect(() => {
        if (!jobId || !jobStatus || !TERMINAL.has(jobStatus) || !token) return;
        getSpectralLineJob(jobId, page, 100, token, requestedDataset)
            .then((nextJob) => setJob((current) => current?.job_id === jobId ? nextJob : current))
            .catch((reason) => {
                if (isUnauthorizedApiError(reason)) handleSpectralError(reason);
            });
    }, [handleSpectralError, jobId, jobStatus, page, requestedDataset, token]);

    useEffect(() => {
        const autorun = searchParams.get("autorun");
        const autorunKey = searchParams.toString();
        if (!autorun || autorunHandled.current === autorunKey || !metadata || !token) return;
        autorunHandled.current = autorunKey;
        const timer = window.setTimeout(() => {
            startJob(autorun === "coverage" ? "alma_coverage" : "catalog_search").catch(handleSpectralError);
        }, 350);
        return () => window.clearTimeout(timer);
    }, [handleSpectralError, metadata, searchParams, startJob, token]);

    const rows = (job?.rows || []) as SpectralLineRecord[];
    const projects = (job?.rows || []) as SpectralCoverageProject[];
    const progress = Math.max(0, Math.min(100, Number(job?.progress || 0)));

    const toggleLine = (line: SpectralLineRecord) => {
        const key = lineSelectionKey(line);
        const identifiers = [key, String(line.line_id || "")].filter(Boolean);
        setSelectedLineIds((current) => {
            const active = identifiers.some((identifier) => current.includes(identifier));
            return active
                ? current.filter((identifier) => !identifiers.includes(identifier))
                : [...current, ...identifiers];
        });
    };

    const download = async (dataset: "lines" | "raw_lines" | "coverage" | "candidates", format: "csv" | "tsv" | "json" | "casa") => {
        if (!job || !token) return;
        try {
            const result = await downloadSpectralLineExport(job.job_id, dataset, format, token);
            const url = URL.createObjectURL(result.blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = result.filename;
            link.click();
            URL.revokeObjectURL(url);
        } catch (reason) {
            handleSpectralError(reason);
        }
    };

    const cancelJob = async () => {
        if (!job || !token) return;
        try {
            setJob(await cancelSpectralLineJob(job.job_id, token));
        } catch (reason) {
            handleSpectralError(reason);
        }
    };

    const compareRawEntries = async (line: SpectralLineRecord) => {
        if (!job || !token) {
            setRawComparison({ line, rows: [] });
            return;
        }
        try {
            const rawJob = await getSpectralLineJob(job.job_id, 1, 500, token, "raw_lines");
            const rawIds = new Set((line.raw_line_ids || []).map(String));
            const matches = (rawJob.rows || []).filter((candidate) => {
                const raw = candidate as SpectralLineRecord;
                if (raw.line_id !== undefined && rawIds.has(String(raw.line_id))) return true;
                return raw.species === line.species
                    && raw.transition === line.transition
                    && Math.abs(Number(raw.frequency_ghz) - Number(line.frequency_ghz)) <= 0.000001;
            }) as SpectralLineRecord[];
            setRawComparison({ line, rows: matches });
        } catch (reason) {
            handleSpectralError(reason);
        }
    };

    if (!isInitialized) {
        return <div className="flex h-full w-full items-center justify-center bg-slate-950"><Loader2 className="h-6 w-6 animate-spin text-cyan-300" /></div>;
    }

    if (!isAuthenticated) {
        return (
            <div className="flex h-full w-full items-center justify-center bg-slate-950 text-slate-400">
                Sign in to use Spectral Line Explorer.
                <AuthModal />
            </div>
        );
    }

    return (
        <div className="flex h-full w-full overflow-hidden">
            {sidebarOpen && <Sidebar />}
            <main className="flex min-w-0 flex-1 flex-col overflow-hidden bg-[#0b0d12]">
                <header className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                    <div>
                        <div className="flex items-center gap-2">
                            <Waves className="h-5 w-5 text-cyan-300" />
                            <h1 className="text-lg font-semibold text-slate-100">Spectral Line Explorer</h1>
                        </div>
                        <p className="mt-1 text-xs text-slate-500">Splatalogue catalog evidence and exact ALMA spectral-window coverage</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="text-right text-[10px] text-slate-500">
                            <div>{metadata?.band_registry.version || "ALMA band registry"}</div>
                            <div>Results expire after one hour</div>
                        </div>
                        <button
                            type="button"
                            onClick={() => router.push("/")}
                            className="inline-flex items-center gap-2 rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-cyan-500/50 hover:bg-slate-800 hover:text-cyan-200"
                        >
                            <ArrowLeft className="h-4 w-4" />
                            Back to chat
                        </button>
                    </div>
                </header>

                <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden xl:grid-cols-[330px_minmax(0,1fr)]">
                    <aside className="overflow-y-auto border-r border-slate-800 bg-slate-950/70 p-4">
                        <div className="mb-4 grid grid-cols-2 rounded-lg bg-slate-900 p-1">
                            {(["basic", "advanced"] as Mode[]).map((item) => (
                                <button key={item} type="button" onClick={() => setMode(item)} className={`rounded-md px-3 py-2 text-xs font-semibold capitalize ${mode === item ? "bg-cyan-400 text-slate-950" : "text-slate-400"}`}>
                                    {item}
                                </button>
                            ))}
                        </div>

                        <div className="space-y-4">
                            <Field label="Species">
                                <div className="relative">
                                    <input value={speciesQuery} onChange={(event) => { setSpeciesQuery(event.target.value); if (mode === "basic") setSelectedSpecies([]); }} className={inputClass} placeholder="CO, CH3OH, methanol…" />
                                    {speciesMatches.length > 0 && speciesQuery && (mode === "advanced" || !selectedSpecies.length) && (
                                        <div className="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-md border border-slate-700 bg-slate-950 shadow-xl">
                                            {speciesMatches.map((item) => (
                                                <button key={item.species_id} type="button" onClick={() => { setSelectedSpecies(mode === "advanced" ? [...selectedSpecies.filter((selected) => selected.species_id !== item.species_id), item].slice(0, 50) : [item]); setSpeciesQuery(mode === "advanced" ? "" : item.formula); setSpeciesMatches([]); }} className="block w-full border-b border-slate-800 px-3 py-2 text-left hover:bg-slate-900">
                                                    <div className="text-xs text-slate-200">{item.formula}</div>
                                                    <div className="text-[10px] text-slate-500">{item.chemical_name} · {item.status}</div>
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                                {selectedSpecies.length > 0 && <div className="mt-2 flex flex-wrap gap-1">{selectedSpecies.map((item) => <button type="button" onClick={() => setSelectedSpecies((current) => current.filter((selected) => selected.species_id !== item.species_id))} key={item.species_id} className="rounded bg-cyan-500/10 px-2 py-1 text-[10px] text-cyan-200">{item.formula} ×</button>)}</div>}
                            </Field>

                            <Field label="Transition">
                                <input value={transition} onChange={(event) => setTransition(event.target.value)} className={inputClass} placeholder="2-1" />
                            </Field>

                            <Field label="ALMA band">
                                <select value={band} onChange={(event) => setBand(event.target.value)} className={inputClass}>
                                    {metadata?.band_registry.bands.map((item) => <option key={item.band} value={item.band}>Band {item.band}: {item.minimum_ghz}–{item.maximum_ghz} GHz</option>)}
                                </select>
                            </Field>

                            <div>
                                <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500">Custom window (overrides band)</div>
                                <div className="grid grid-cols-[1fr_1fr_76px] gap-2">
                                    <input value={windowMin} onChange={(event) => setWindowMin(event.target.value)} className={inputClass} placeholder="Min" />
                                    <input value={windowMax} onChange={(event) => setWindowMax(event.target.value)} className={inputClass} placeholder="Max" />
                                    <select value={windowUnit} onChange={(event) => setWindowUnit(event.target.value)} className={inputClass}>
                                        {(metadata?.units || ["GHz"]).map((unit) => <option key={unit}>{unit}</option>)}
                                    </select>
                                </div>
                                {mode === "advanced" && (
                                    <textarea
                                        value={additionalWindows}
                                        onChange={(event) => setAdditionalWindows(event.target.value)}
                                        className="mt-2 min-h-20 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-[11px] text-slate-100 outline-none focus:border-cyan-500"
                                        placeholder={"Additional windows, one per line:\n84 86 GHz\n1.1 1.3 mm"}
                                    />
                                )}
                            </div>

                            <div className="grid grid-cols-2 gap-2">
                                <Field label="Frame">
                                    <select value={frame} onChange={(event) => setFrame(event.target.value)} className={inputClass}>
                                        <option value="rest">Rest</option>
                                        <option value="observed">Observed</option>
                                    </select>
                                </Field>
                                <Field label="Redshift">
                                    <input value={redshift} onChange={(event) => setRedshift(event.target.value)} className={inputClass} placeholder="auto" />
                                </Field>
                            </div>
                            {mode === "advanced" && (
                                <div className="grid grid-cols-2 gap-2">
                                    <Field label="Radial velocity km/s">
                                        <input value={radialVelocity} onChange={(event) => setRadialVelocity(event.target.value)} className={inputClass} placeholder="Overrides z" />
                                    </Field>
                                    <Field label="Velocity convention">
                                        <select value={velocityConvention} onChange={(event) => setVelocityConvention(event.target.value)} className={inputClass}>
                                            <option value="radio">Radio</option>
                                            <option value="optical">Optical</option>
                                            <option value="relativistic">Relativistic</option>
                                        </select>
                                    </Field>
                                </div>
                            )}

                            <Field label="Target for coverage">
                                <input value={target} onChange={(event) => setTarget(event.target.value)} className={inputClass} placeholder="M87" />
                            </Field>

                            {mode === "advanced" && (
                                <div className="space-y-4 border-t border-slate-800 pt-4">
                                    <Field label="Splatalogue data version">
                                        <select value={version} onChange={(event) => setVersion(event.target.value)} className={inputClass}>
                                            {(metadata?.versions || ["v3.0"]).map((item) => <option key={item}>{item}</option>)}
                                        </select>
                                    </Field>
                                    <Field label="Catalogs">
                                        <div className="grid grid-cols-2 gap-2">
                                            {(metadata?.catalogs || []).map((catalog) => (
                                                <label key={catalog} className="flex items-center gap-2 text-[11px] text-slate-400">
                                                    <input type="checkbox" checked={catalogs.includes(catalog)} onChange={() => setCatalogs((current) => current.includes(catalog) ? current.filter((item) => item !== catalog) : [...current, catalog])} />
                                                    {catalog}
                                                </label>
                                            ))}
                                        </div>
                                    </Field>
                                    <Field label="Exclude species categories">
                                        <div className="grid grid-cols-2 gap-2">
                                            {["atmospheric", "potential", "probable", "known"].map((category) => (
                                                <label key={category} className="flex items-center gap-2 text-[11px] capitalize text-slate-400">
                                                    <input
                                                        type="checkbox"
                                                        checked={excludedCategories.includes(category)}
                                                        onChange={() => setExcludedCategories((current) => current.includes(category) ? current.filter((item) => item !== category) : [...current, category])}
                                                    />
                                                    {category}
                                                </label>
                                            ))}
                                        </div>
                                    </Field>
                                    <div className="grid grid-cols-2 gap-2">
                                        <Field label="Eu min (K)"><input value={energyMin} onChange={(event) => setEnergyMin(event.target.value)} className={inputClass} /></Field>
                                        <Field label="Eu max (K)"><input value={energyMax} onChange={(event) => setEnergyMax(event.target.value)} className={inputClass} /></Field>
                                    </div>
                                    <div className="grid grid-cols-[1fr_150px] gap-2">
                                        <Field label="Intensity lower limit"><input value={intensityLimit} onChange={(event) => setIntensityLimit(event.target.value)} className={inputClass} /></Field>
                                        <Field label="Intensity type">
                                            <select value={intensityType} onChange={(event) => setIntensityType(event.target.value)} className={inputClass}>
                                                <option>CDMS/JPL (log)</option>
                                                <option>Sij-mu2</option>
                                                <option>Aij (log)</option>
                                            </select>
                                        </Field>
                                    </div>
                                    <Field label="Max frequency error (MHz)">
                                        <input value={uncertainty} onChange={(event) => setUncertainty(event.target.value)} className={inputClass} placeholder="50" />
                                    </Field>
                                    <label className="flex items-center gap-2 text-xs text-slate-400"><input type="checkbox" checked={onlyObserved} onChange={(event) => setOnlyObserved(event.target.checked)} />Astronomically observed only</label>
                                    <label className="flex items-center gap-2 text-xs text-slate-400"><input type="checkbox" checked={onlyRecommended} onChange={(event) => setOnlyRecommended(event.target.checked)} />NRAO recommended only</label>
                                    <Field label="Catalog view">
                                        <select value={outputMode} onChange={(event) => setOutputMode(event.target.value as "merged" | "raw")} className={inputClass}>
                                            <option value="merged">Merged transitions</option>
                                            <option value="raw">Raw catalog rows</option>
                                        </select>
                                    </Field>
                                    <Field label="Sort order">
                                        <select value={sortOrder} onChange={(event) => setSortOrder(event.target.value)} className={inputClass}>
                                            <option value="frequency_asc">Rest frequency ascending</option>
                                            <option value="frequency_desc">Rest frequency descending</option>
                                        </select>
                                    </Field>
                                </div>
                            )}

                            <button type="button" onClick={() => startJob("catalog_search").catch(handleSpectralError)} className="flex h-10 w-full items-center justify-center gap-2 rounded-md bg-cyan-400 text-xs font-bold text-slate-950 hover:bg-cyan-300">
                                <Search className="h-4 w-4" /> Search lines
                            </button>
                            <button type="button" onClick={() => startJob("alma_coverage").catch(handleSpectralError)} className="flex h-10 w-full items-center justify-center gap-2 rounded-md border border-cyan-500/50 bg-cyan-500/10 text-xs font-bold text-cyan-200 hover:bg-cyan-500/15">
                                <Telescope className="h-4 w-4" /> Check exact ALMA coverage
                            </button>
                        </div>
                    </aside>

                    <section className="min-w-0 overflow-y-auto p-5">
                        {error && <div className="mb-4 flex gap-2 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-200"><AlertTriangle className="h-4 w-4 shrink-0" />{error}</div>}

                        {job && (
                            <div className="mb-4 rounded-lg border border-slate-800 bg-slate-950 p-4">
                                <div className="flex items-center justify-between gap-3">
                                    <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
                                        {!TERMINAL.has(job.status) ? <Loader2 className="h-4 w-4 animate-spin text-cyan-300" /> : <Play className="h-4 w-4 text-slate-500" />}
                                        {job.phase}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="font-mono text-[10px] text-slate-500">{job.status} · {progress}%</span>
                                        {!TERMINAL.has(job.status) && <button type="button" onClick={cancelJob} className="rounded border border-amber-500/40 p-1.5 text-amber-300"><CircleStop className="h-3.5 w-3.5" /></button>}
                                    </div>
                                </div>
                                <div className="mt-3 h-1 overflow-hidden rounded bg-slate-900"><div className="h-full bg-cyan-400" style={{ width: `${progress}%` }} /></div>
                                {job.warnings?.map((warning) => <div key={warning} className="mt-2 text-[10px] text-amber-300">{warning}</div>)}
                                {job.error && <div className="mt-2 text-xs text-red-300">{job.error}</div>}
                                {job.status === "needs_input" && <div className="mt-2 text-xs text-amber-200">SIMBAD/NED redshift values conflict or are missing. Enter an explicit redshift and rerun.</div>}
                            </div>
                        )}

                        <div className="mb-4 flex flex-wrap gap-1 border-b border-slate-800">
                            {([
                                ["lines", Waves],
                                ["coverage", Telescope],
                                ["confusion", Radio],
                                ["exports", Download],
                            ] as const).map(([tab, Icon]) => (
                                <button key={tab} type="button" onClick={() => setResultTab(tab)} className={`flex items-center gap-2 border-b-2 px-4 py-2 text-xs font-semibold capitalize ${resultTab === tab ? "border-cyan-400 text-cyan-200" : "border-transparent text-slate-500"}`}>
                                    <Icon className="h-3.5 w-3.5" />{tab}
                                </button>
                            ))}
                        </div>

                        {resultTab === "lines" && (
                            <div className="space-y-4">
                                <SpectralFrequencyRuler lines={rows} />
                                <div className="flex items-center justify-between text-xs text-slate-500">
                                    <span>{job?.pagination?.total_rows || 0} transitions · {selectedLineIds.length ? `${selectedLineIds.length} selection identifiers` : "select lines for coverage"}</span>
                                    <span>Page {job?.pagination?.page || 0} / {job?.pagination?.total_pages || 0}</span>
                                </div>
                                <SpectralLineTable lines={rows} selectedIds={selectedLineIds} onToggleSelect={toggleLine} onAnalyzeConfusion={(line) => startJob("confusion", line).catch(handleSpectralError)} onCompareRaw={compareRawEntries} />
                                {(job?.pagination?.total_pages || 0) > 1 && (
                                    <div className="flex justify-end gap-2">
                                        <button type="button" disabled={page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))} className="rounded border border-slate-700 px-3 py-2 text-xs text-slate-300 disabled:opacity-40">Previous</button>
                                        <button type="button" disabled={page >= (job?.pagination?.total_pages || 0)} onClick={() => setPage((value) => value + 1)} className="rounded border border-slate-700 px-3 py-2 text-xs text-slate-300 disabled:opacity-40">Next</button>
                                    </div>
                                )}
                                {rawComparison && (
                                    <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/5 p-4">
                                        <div className="flex justify-between">
                                            <div className="text-xs font-semibold text-cyan-100">Raw catalog entries</div>
                                            <button type="button" onClick={() => setRawComparison(null)} className="text-[10px] text-slate-500">Close</button>
                                        </div>
                                        <div className="mt-2 font-mono text-[11px] text-slate-300">Selected catalogs: {rawComparison.line.catalogs?.join(", ") || "—"}</div>
                                        <div className="mt-1 text-[10px] text-slate-500">{rawComparison.line.frequency_selection_reason}</div>
                                        <div className="mt-3">
                                            <SpectralLineTable lines={rawComparison.rows} compact />
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {resultTab === "coverage" && (
                            <div className="space-y-3">
                                {projects.map((project) => <ProjectRow key={project.proposal_id} project={project} />)}
                                {!projects.length && <div className="rounded-lg border border-slate-800 bg-slate-950 py-16 text-center text-sm text-slate-500">Run exact ALMA coverage to see project-level results and matching SPWs.</div>}
                            </div>
                        )}

                        {resultTab === "confusion" && (
                            <div className="space-y-4">
                                <div className="flex gap-2 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-xs leading-5 text-amber-100">
                                    <FlaskConical className="h-4 w-4 shrink-0" />
                                    The score is a catalog-evidence heuristic, not a flux, abundance, detectability, or LTE prediction.
                                </div>
                                <SpectralLineTable lines={rows} compact />
                            </div>
                        )}

                        {resultTab === "exports" && (
                            <div className="grid gap-3 md:grid-cols-2">
                                {(["csv", "tsv", "json", "casa"] as const).map((format) => (
                                    <button key={format} type="button" disabled={!job || !["succeeded", "partial"].includes(job.status)} onClick={() => download(job?.operation === "alma_coverage" ? "coverage" : job?.operation === "confusion" ? "candidates" : "lines", format)} className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 p-4 text-left disabled:opacity-40">
                                        <div><div className="text-sm font-semibold uppercase text-slate-200">{format}</div><div className="mt-1 text-[10px] text-slate-500">{format === "casa" ? "Line list plus unit-aware Python snippet" : "Complete result dataset"}</div></div>
                                        <Download className="h-4 w-4 text-cyan-300" />
                                    </button>
                                ))}
                            </div>
                        )}
                    </section>
                </div>
            </main>
            <AuthModal />
        </div>
    );
}

export default function SpectralLinesPage() {
    return (
        <Suspense fallback={<div className="flex h-full w-full items-center justify-center bg-slate-950"><Loader2 className="h-6 w-6 animate-spin text-cyan-300" /></div>}>
            <SpectralLineExplorer />
        </Suspense>
    );
}
