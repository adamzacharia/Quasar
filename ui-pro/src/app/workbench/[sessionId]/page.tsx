"use client";

import { useCallback, useEffect, useMemo, useState, type PointerEvent } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
    Activity,
    AlertTriangle,
    ArrowLeft,
    Check,
    CircleStop,
    Clipboard,
    Download,
    ExternalLink,
    FileCode,
    Layers,
    Loader2,
    Radio,
    RotateCw,
    Search,
    Settings2,
    SlidersHorizontal,
    Telescope,
    Waves,
} from "lucide-react";
import { AuthModal } from "@/components/AuthModal";
import { Sidebar } from "@/components/Sidebar";
import { SpectralLineTable, lineSelectionKey } from "@/components/SpectralLineTable";
import { useAuthStore } from "@/lib/auth-store";
import { useChatStore } from "@/lib/store";
import {
    cancelWorkbenchJob,
    getWorkbenchExports,
    getWorkbenchJob,
    getWorkbenchLineOverlays,
    getWorkbenchLinePresets,
    getWorkbenchMetadata,
    startWorkbenchJob,
    type WorkbenchExports,
    type WorkbenchJob,
    type WorkbenchLineOverlays,
    type WorkbenchLinePreset,
    type WorkbenchMetadata,
    type WorkbenchPrepareResult,
    type WorkbenchPvSlicePlan,
    type WorkbenchRenderPlan,
    type WorkbenchSpectrumPlan,
    type SpectralLineRecord,
} from "@/lib/api";

type RenderMode = "image" | "channel" | "moment" | "pv";
type ExportFormat = "casa" | "carta" | "ds9" | "python" | "notebook" | "csv" | "figure_png";
type InteractionMode = "aperture" | "rms" | "pv";
type ImagePoint = { x: number; y: number };
type ViewBox = { left: number; top: number; width: number; height: number };

const EXPORT_FORMATS: ExportFormat[] = ["casa", "carta", "ds9", "python", "notebook", "csv", "figure_png"];

function asText(value: unknown, fallback = "-"): string {
    if (value === null || value === undefined || value === "") return fallback;
    return String(value);
}

function asNumber(value: unknown): number | null {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function metadataNumber(metadata: Record<string, unknown>, keys: string[]): number | null {
    for (const key of keys) {
        const value = asNumber(metadata[key]);
        if (value !== null) return value;
    }
    return null;
}

function metadataArray(metadata: Record<string, unknown>, keys: string[]): number[] {
    for (const key of keys) {
        const value = metadata[key];
        if (Array.isArray(value)) return value.map(Number).filter(Number.isFinite);
    }
    return [];
}

function formatNumber(value: unknown, digits = 4): string {
    const parsed = asNumber(value);
    if (parsed === null) return "-";
    if (Math.abs(parsed) >= 1000 || (Math.abs(parsed) > 0 && Math.abs(parsed) < 0.001)) {
        return parsed.toExponential(3);
    }
    return parsed.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatBytes(value: unknown): string {
    const parsed = asNumber(value);
    if (parsed === null) return "-";
    if (parsed >= 1024 * 1024 * 1024) return `${(parsed / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    if (parsed >= 1024 * 1024) return `${(parsed / (1024 * 1024)).toFixed(1)} MB`;
    if (parsed >= 1024) return `${(parsed / 1024).toFixed(1)} KB`;
    return `${parsed} B`;
}

function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1200);
    };

    return (
        <button
            type="button"
            onClick={handleCopy}
            className="inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900 px-2.5 text-[11px] font-semibold text-slate-300 transition-colors hover:border-cyan-500/50 hover:text-cyan-200"
        >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-300" /> : <Clipboard className="h-3.5 w-3.5" />}
            {copied ? "Copied" : label}
        </button>
    );
}

function DownloadTextButton({ filename, text }: { filename: string; text: string }) {
    const handleDownload = () => {
        let blob: Blob;
        if (text.startsWith("data:")) {
            const [meta, payload] = text.split(",", 2);
            const mimeType = meta.match(/^data:([^;]+)/)?.[1] || "application/octet-stream";
            const binary = atob(payload || "");
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
            blob = new Blob([bytes], { type: mimeType });
        } else {
            blob = new Blob([text], { type: "text/plain;charset=utf-8" });
        }
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
    };

    return (
        <button
            type="button"
            onClick={handleDownload}
            className="inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900 px-2.5 text-[11px] font-semibold text-slate-300 transition-colors hover:border-emerald-500/50 hover:text-emerald-200"
        >
            <Download className="h-3.5 w-3.5" />
            Download
        </button>
    );
}

function Stat({ label, value }: { label: string; value: string }) {
    return (
        <div className="min-w-0 border-r border-slate-800 px-4 py-3 last:border-r-0">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">{label}</div>
            <div className="mt-1 truncate font-mono text-sm text-slate-100" title={value}>{value}</div>
        </div>
    );
}

const TERMINAL_JOB_STATUSES = new Set(["succeeded", "failed", "canceled", "orphaned"]);

function isTerminalJob(job: WorkbenchJob | null | undefined): boolean {
    return Boolean(job && TERMINAL_JOB_STATUSES.has(String(job.status)));
}

function WorkbenchJobPanel({ job, onCancel, canceling }: { job: WorkbenchJob | null; onCancel: () => void; canceling: boolean }) {
    if (!job) return null;
    const progress = Math.max(0, Math.min(100, Math.round(Number(job.progress) || 0)));
    const metrics = asRecord(job.metrics);
    const bytesDone = asNumber(metrics.bytes_done);
    const bytesTotal = asNumber(metrics.bytes_total);
    const byteLabel = bytesDone !== null
        ? bytesTotal
            ? `${(bytesDone / (1024 * 1024)).toFixed(1)} / ${(bytesTotal / (1024 * 1024)).toFixed(1)} MB`
            : `${(bytesDone / (1024 * 1024)).toFixed(1)} MB`
        : "";
    const terminal = isTerminalJob(job);

    return (
        <section className="rounded-lg border border-slate-800 bg-slate-950">
            <div className="flex items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
                <div className="flex min-w-0 items-center gap-2 text-sm font-semibold text-slate-100">
                    <Activity className={`h-4 w-4 shrink-0 ${terminal ? "text-slate-500" : "text-cyan-300"}`} />
                    <span className="truncate">{job.operation.replace("_", " ")}</span>
                </div>
                <span className="font-mono text-[11px] text-slate-500">{job.status}</span>
            </div>
            <div className="space-y-3 p-4">
                <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="truncate text-slate-400">{job.phase || "queued"}</span>
                    <span className="font-mono text-slate-300">{progress}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-slate-900">
                    <div
                        className={`h-full rounded-full ${job.status === "failed" ? "bg-red-400" : job.status === "canceled" ? "bg-amber-400" : "bg-cyan-400"}`}
                        style={{ width: `${progress}%` }}
                    />
                </div>
                {byteLabel && <div className="font-mono text-[11px] text-slate-500">{byteLabel}</div>}
                {job.error && <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs leading-5 text-amber-100">{job.error}</div>}
                {!terminal && (
                    <button
                        type="button"
                        onClick={onCancel}
                        disabled={canceling}
                        className="inline-flex h-8 items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 text-[11px] font-semibold text-amber-200 transition-colors hover:bg-amber-500/15 disabled:opacity-60"
                    >
                        {canceling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CircleStop className="h-3.5 w-3.5" />}
                        Cancel
                    </button>
                )}
            </div>
        </section>
    );
}

function SpectrumMiniPlot({ series, lineData }: { series: WorkbenchSpectrumPlan["series"]; lineData?: WorkbenchLineOverlays | null }) {
    const yValues = series.y.map(Number).filter(Number.isFinite);
    if (yValues.length < 2) {
        return (
            <div className="rounded-md border border-slate-800 bg-slate-900 px-3 py-8 text-center text-xs text-slate-500">
                Spectrum values will appear after the FITS product is prepared.
            </div>
        );
    }
    const minY = Math.min(...yValues);
    const maxY = Math.max(...yValues);
    const span = maxY - minY || 1;
    const rawXValues = series.x.map(Number);
    const hasNumericX = rawXValues.length === yValues.length && rawXValues.every(Number.isFinite);
    const xValues = hasNumericX ? rawXValues : yValues.map((_, index) => index);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    const xSpan = maxX - minX || 1;
    const points = yValues.map((value, index) => {
        const x = ((xValues[index] - minX) / xSpan) * 100;
        const y = 42 - ((value - minY) / span) * 34;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    const lineMarkers = (lineData?.lines ?? [])
        .map((line) => {
            const observed = asNumber(line.observed_frequency_ghz);
            const label = asText(line.species || line.name || line.transition, "Line");
            return observed === null ? null : { observed, label };
        })
        .filter((line): line is { observed: number; label: string } => Boolean(line))
        .filter((line) => line.observed >= minX && line.observed <= maxX)
        .slice(0, 8);

    return (
        <div className="rounded-md border border-slate-800 bg-slate-900 p-3">
            <svg viewBox="0 0 100 48" className="h-32 w-full" preserveAspectRatio="none" role="img" aria-label="Extracted spectrum">
                <line x1="0" y1="42" x2="100" y2="42" stroke="#334155" strokeWidth="0.6" />
                <line x1="0" y1="6" x2="0" y2="42" stroke="#334155" strokeWidth="0.6" />
                {lineMarkers.map((line) => {
                    const x = ((line.observed - minX) / xSpan) * 100;
                    return (
                        <g key={`${line.label}-${line.observed}`}>
                            <line x1={x} y1="6" x2={x} y2="42" stroke="#fbbf24" strokeWidth="0.8" strokeDasharray="2 1" vectorEffect="non-scaling-stroke" />
                            <text x={Math.min(96, x + 0.8)} y="9" fill="#fde68a" fontSize="3" className="font-mono">
                                {line.label.slice(0, 12)}
                            </text>
                        </g>
                    );
                })}
                <polyline points={points} fill="none" stroke="#67e8f9" strokeWidth="1.4" vectorEffect="non-scaling-stroke" />
            </svg>
            <div className="mt-2 flex justify-between gap-3 text-[10px] text-slate-500">
                <span className="truncate">{series.x_label}</span>
                <span className="truncate">{series.y_label}</span>
            </div>
        </div>
    );
}

function stateValue(state: Record<string, unknown> | undefined, key: string, fallback: unknown): unknown {
    return state && state[key] !== undefined ? state[key] : fallback;
}

function imageFitBox(containerWidth: number, containerHeight: number, imageWidth: number | null, imageHeight: number | null): ViewBox | null {
    if (!imageWidth || !imageHeight || containerWidth <= 0 || containerHeight <= 0) return null;
    const imageAspect = imageWidth / imageHeight;
    const containerAspect = containerWidth / containerHeight;
    let width = containerWidth;
    let height = containerHeight;
    if (containerAspect > imageAspect) {
        width = containerHeight * imageAspect;
    } else {
        height = containerWidth / imageAspect;
    }
    return {
        left: (containerWidth - width) / 2,
        top: (containerHeight - height) / 2,
        width,
        height,
    };
}

function clampImagePoint(point: ImagePoint, imageWidth: number | null, imageHeight: number | null): ImagePoint {
    const maxX = Math.max(0, (imageWidth ?? 1) - 1);
    const maxY = Math.max(0, (imageHeight ?? 1) - 1);
    return {
        x: Math.max(0, Math.min(maxX, point.x)),
        y: Math.max(0, Math.min(maxY, point.y)),
    };
}

export default function WorkbenchPage() {
    const params = useParams<{ sessionId: string }>();
    const sessionId = String(params.sessionId || "");
    const sidebarOpen = useChatStore((s) => s.sidebarOpen);
    const { token, isAuthenticated, isInitialized, openAuthModal } = useAuthStore();

    const [mounted, setMounted] = useState(false);
    const [metadata, setMetadata] = useState<WorkbenchMetadata | null>(null);
    const [exportsData, setExportsData] = useState<WorkbenchExports | null>(null);
    const [jobs, setJobs] = useState<WorkbenchJob[]>([]);
    const [activeJobId, setActiveJobId] = useState<string | null>(null);
    const [prepareResult, setPrepareResult] = useState<WorkbenchPrepareResult | null>(null);
    const [lineData, setLineData] = useState<WorkbenchLineOverlays | null>(null);
    const [selectedLineIds, setSelectedLineIds] = useState<string[]>([]);
    const [linePresets, setLinePresets] = useState<WorkbenchLinePreset[]>([]);
    const [renderPlan, setRenderPlan] = useState<WorkbenchRenderPlan | null>(null);
    const [spectrumPlan, setSpectrumPlan] = useState<WorkbenchSpectrumPlan | null>(null);
    const [pvSlicePlan, setPvSlicePlan] = useState<WorkbenchPvSlicePlan | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [actionLoading, setActionLoading] = useState<string | null>(null);
    const [cancelingJob, setCancelingJob] = useState(false);
    const [mode, setMode] = useState<RenderMode>("image");
    const [channel, setChannel] = useState(0);
    const [moment, setMoment] = useState(0);
    const [colormap, setColormap] = useState("inferno");
    const [stretch, setStretch] = useState("asinh");
    const [contourSigmaText, setContourSigmaText] = useState("3,5,10");
    const [rmsX1, setRmsX1] = useState("");
    const [rmsY1, setRmsY1] = useState("");
    const [rmsX2, setRmsX2] = useState("");
    const [rmsY2, setRmsY2] = useState("");
    const [redshift, setRedshift] = useState(0);
    const [tolerance, setTolerance] = useState(0.01);
    const [frequency, setFrequency] = useState("");
    const [linePresetKey, setLinePresetKey] = useState("");
    const [apertureX, setApertureX] = useState("");
    const [apertureY, setApertureY] = useState("");
    const [apertureRadius, setApertureRadius] = useState(3);
    const [pvX1, setPvX1] = useState("");
    const [pvY1, setPvY1] = useState("");
    const [pvX2, setPvX2] = useState("");
    const [pvY2, setPvY2] = useState("");
    const [pvWidth, setPvWidth] = useState(3);
    const [interactionMode, setInteractionMode] = useState<InteractionMode>("aperture");
    const [dragStart, setDragStart] = useState<ImagePoint | null>(null);
    const [dragPreview, setDragPreview] = useState<ImagePoint | null>(null);

    const metadataRecord = metadata?.metadata ?? {};
    const state = renderPlan?.state ?? metadata?.state;
    const shape = metadataArray(metadataRecord, ["shape", "raw_shape", "cube_shape"]);
    const imageWidth = shape.length ? shape.at(-1) ?? null : null;
    const imageHeight = shape.length >= 2 ? shape.at(-2) ?? null : null;
    const restFrequency = metadataNumber(metadataRecord, ["rest_freq_ghz", "restFreqGhz"]);
    const channelCount = metadataNumber(metadataRecord, ["channel_count", "channelCount"]);
    const beamMajor = metadataNumber(metadataRecord, ["beam_major_arcsec", "beamMajorArcsec"]);
    const beamMinor = metadataNumber(metadataRecord, ["beam_minor_arcsec", "beamMinorArcsec"]);
    const sensitivity = metadataNumber(metadataRecord, ["sensitivity_mjy_beam", "rms"]);
    const evidence = metadata?.evidence ?? {};
    const cacheRecord = prepareResult?.cache ?? metadata?.cache ?? {};
    const cacheStatus = asText(cacheRecord.status, "not_prepared");
    const cachedBytes = asNumber(cacheRecord.cached_bytes);
    const previewRecord = asRecord(cacheRecord.preview);
    const previewStatus = asText(previewRecord.status, "not_prepared");
    const previewBytes = asNumber(previewRecord.bytes);
    const previewShape = Array.isArray(previewRecord.shape) ? previewRecord.shape.map(String).join(" x ") : "";
    const userCache = asRecord(cacheRecord.user_cache);
    const userCachedBytes = asNumber(userCache.cached_bytes);
    const userCacheLimitBytes = asNumber(userCache.limit_bytes);
    const headerBytesFetched = asNumber(metadataRecord.header_bytes_fetched ?? evidence.header_bytes_fetched);
    const headerRangeLimitBytes = asNumber(metadataRecord.header_range_limit_bytes ?? evidence.header_range_limit_bytes);
    const headerStrategy = metadataRecord.range_header_read === true
        ? `range ${formatBytes(headerBytesFetched)}${headerRangeLimitBytes ? ` / ${formatBytes(headerRangeLimitBytes)} cap` : ""}`
        : metadataRecord.local_header_read === true
            ? "local FITS header"
            : asText(evidence.header_strategy, "metadata service");
    const activeJob = useMemo(() => {
        const selected = activeJobId ? jobs.find((job) => job.job_id === activeJobId) ?? null : null;
        if (selected && !isTerminalJob(selected)) return selected;
        return jobs.find((job) => !isTerminalJob(job)) ?? selected;
    }, [activeJobId, jobs]);
    const runningJob = activeJob && !isTerminalJob(activeJob) ? activeJob : null;
    const isBusy = Boolean(actionLoading || runningJob);
    const aperturePoint = useMemo<ImagePoint | null>(() => {
        const x = asNumber(apertureX);
        const y = asNumber(apertureY);
        return x === null || y === null ? null : clampImagePoint({ x, y }, imageWidth, imageHeight);
    }, [apertureX, apertureY, imageHeight, imageWidth]);
    const rmsRegion = useMemo(() => {
        const x1 = asNumber(rmsX1);
        const y1 = asNumber(rmsY1);
        const x2 = asNumber(rmsX2);
        const y2 = asNumber(rmsY2);
        if (x1 === null || y1 === null || x2 === null || y2 === null) return null;
        const start = clampImagePoint({ x: x1, y: y1 }, imageWidth, imageHeight);
        const end = clampImagePoint({ x: x2, y: y2 }, imageWidth, imageHeight);
        return {
            x1: Math.min(start.x, end.x),
            y1: Math.min(start.y, end.y),
            x2: Math.max(start.x, end.x),
            y2: Math.max(start.y, end.y),
        };
    }, [imageHeight, imageWidth, rmsX1, rmsX2, rmsY1, rmsY2]);
    const pvLine = useMemo(() => {
        const x1 = asNumber(pvX1);
        const y1 = asNumber(pvY1);
        const x2 = asNumber(pvX2);
        const y2 = asNumber(pvY2);
        if (x1 === null || y1 === null || x2 === null || y2 === null) return null;
        return {
            start: clampImagePoint({ x: x1, y: y1 }, imageWidth, imageHeight),
            end: clampImagePoint({ x: x2, y: y2 }, imageWidth, imageHeight),
        };
    }, [imageHeight, imageWidth, pvX1, pvX2, pvY1, pvY2]);
    const pointToPercent = useCallback((point: ImagePoint | null): ImagePoint | null => {
        if (!point || !imageWidth || !imageHeight) return null;
        const xMax = Math.max(1, imageWidth - 1);
        const yMax = Math.max(1, imageHeight - 1);
        return {
            x: (point.x / xMax) * 100,
            y: (1 - point.y / yMax) * 100,
        };
    }, [imageHeight, imageWidth]);
    const aperturePercent = pointToPercent(aperturePoint);
    const pvLinePercent = pvLine ? {
        start: pointToPercent(pvLine.start),
        end: pointToPercent(pvLine.end),
    } : null;
    const rmsRegionPercent = rmsRegion ? {
        left: (rmsRegion.x1 / Math.max(1, (imageWidth ?? 1) - 1)) * 100,
        top: (1 - rmsRegion.y2 / Math.max(1, (imageHeight ?? 1) - 1)) * 100,
        width: ((rmsRegion.x2 - rmsRegion.x1) / Math.max(1, (imageWidth ?? 1) - 1)) * 100,
        height: ((rmsRegion.y2 - rmsRegion.y1) / Math.max(1, (imageHeight ?? 1) - 1)) * 100,
    } : null;
    const dragStartPercent = pointToPercent(dragStart);
    const dragPreviewPercent = pointToPercent(dragPreview);

    const renderSummary = useMemo(() => {
        const activeMode = asText(stateValue(state, "mode", mode));
        const activeMoment = stateValue(state, "moment", moment);
        const activeChannel = stateValue(state, "channel", channel);
        if (activeMode === "channel") return `Channel ${formatNumber(activeChannel, 0)}`;
        if (activeMode === "moment") return `Moment ${formatNumber(activeMoment, 0)}`;
        if (activeMode === "pv") return "PV slice";
        return "Image plane";
    }, [channel, mode, moment, state]);

    const applyJobResult = useCallback((job: WorkbenchJob) => {
        const result = asRecord(job.result);
        if (!Object.keys(result).length) return;
        if (job.operation === "prepare") {
            const nextPrepare = result as unknown as WorkbenchPrepareResult;
            setPrepareResult(nextPrepare);
            setMetadata((current) => current ? {
                ...current,
                status: nextPrepare.status === "cached" ? "cached" : current.status,
                cache: nextPrepare.cache,
                evidence: nextPrepare.evidence,
            } : current);
        } else if (job.operation === "render") {
            setRenderPlan(result as unknown as WorkbenchRenderPlan);
        } else if (job.operation === "spectrum") {
            setSpectrumPlan(result as unknown as WorkbenchSpectrumPlan);
        } else if (job.operation === "pv_slice") {
            setPvSlicePlan(result as unknown as WorkbenchPvSlicePlan);
        } else if (job.operation === "line_overlays") {
            setLineData(result as unknown as WorkbenchLineOverlays);
        } else if (job.operation === "export") {
            setExportsData(result as unknown as WorkbenchExports);
        }
    }, []);

    const startOperationJob = useCallback(async (operation: string, payload: Record<string, unknown>, loadingKey: string) => {
        if (!token) return;
        setActionLoading(loadingKey);
        setError(null);
        try {
            const response = await startWorkbenchJob(sessionId, { operation, payload }, token);
            setJobs(response.jobs?.length ? response.jobs : [response.job]);
            setActiveJobId(response.job.job_id);
        } catch (jobError) {
            setActionLoading(null);
            setError(jobError instanceof Error ? jobError.message : "Could not start workbench job.");
        }
    }, [sessionId, token]);

    const loadWorkbench = useCallback(async () => {
        if (!token || !sessionId) {
            setLoading(false);
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const nextMetadata = await getWorkbenchMetadata(sessionId, token);
            const nextMetadataRecord = nextMetadata.metadata ?? {};
            const nextRestFrequency = metadataNumber(nextMetadataRecord, ["rest_freq_ghz", "restFreqGhz"]);
            const nextShape = metadataArray(nextMetadataRecord, ["shape", "raw_shape", "cube_shape"]);
            const nextWidth = nextShape.at(-1) ?? 100;
            const nextHeight = nextShape.at(-2) ?? 100;
            const nextSpectrumState = asRecord(nextMetadata.state?.spectrum);
            const nextPvState = asRecord(nextMetadata.state?.pv_slice);
            const nextRmsState = asRecord(nextMetadata.state?.rms_region);
            const nextPvPath = Array.isArray(nextPvState.path) ? nextPvState.path.map(asRecord) : [];
            const nextPvStart = nextPvPath[0] ?? { x: nextWidth * 0.25, y: nextHeight * 0.5 };
            const nextPvEnd = nextPvPath.at(-1) ?? { x: nextWidth * 0.75, y: nextHeight * 0.5 };
            setMetadata(nextMetadata);
            setJobs(nextMetadata.jobs ?? []);
            setPrepareResult(null);
            setMode(asText(nextMetadata.state?.mode, "image") as RenderMode);
            setChannel(asNumber(nextMetadata.state?.channel) ?? 0);
            setMoment(asNumber(nextMetadata.state?.moment) ?? 0);
            setColormap(asText(nextMetadata.state?.colormap, "inferno"));
            setStretch(asText(nextMetadata.state?.stretch, "asinh"));
            setContourSigmaText(Array.isArray(nextMetadata.state?.contour_sigma) ? nextMetadata.state.contour_sigma.join(",") : "3,5,10");
            setRmsX1(nextRmsState.x1 !== undefined && nextRmsState.x1 !== null ? String(nextRmsState.x1) : "0");
            setRmsY1(nextRmsState.y1 !== undefined && nextRmsState.y1 !== null ? String(nextRmsState.y1) : "0");
            setRmsX2(nextRmsState.x2 !== undefined && nextRmsState.x2 !== null ? String(nextRmsState.x2) : String(Math.round(nextWidth * 0.25)));
            setRmsY2(nextRmsState.y2 !== undefined && nextRmsState.y2 !== null ? String(nextRmsState.y2) : String(Math.round(nextHeight * 0.25)));
            setRedshift(asNumber(nextMetadata.state?.redshift) ?? 0);
            setFrequency(nextRestFrequency ? String(nextRestFrequency) : "");
            setApertureX(nextSpectrumState.x_pixel !== undefined && nextSpectrumState.x_pixel !== null ? String(nextSpectrumState.x_pixel) : String(Math.round(nextWidth / 2)));
            setApertureY(nextSpectrumState.y_pixel !== undefined && nextSpectrumState.y_pixel !== null ? String(nextSpectrumState.y_pixel) : String(Math.round(nextHeight / 2)));
            setApertureRadius(asNumber(nextSpectrumState.aperture_radius_pixels) ?? 3);
            setPvX1(String(asNumber(nextPvStart.x) ?? Math.round(nextWidth * 0.25)));
            setPvY1(String(asNumber(nextPvStart.y) ?? Math.round(nextHeight * 0.5)));
            setPvX2(String(asNumber(nextPvEnd.x) ?? Math.round(nextWidth * 0.75)));
            setPvY2(String(asNumber(nextPvEnd.y) ?? Math.round(nextHeight * 0.5)));
            setPvWidth(asNumber(nextPvState.width_pixels) ?? 3);
            const [nextExports, nextPresets] = await Promise.all([
                getWorkbenchExports(sessionId, EXPORT_FORMATS, token),
                getWorkbenchLinePresets(),
            ]);
            setExportsData(nextExports);
            setLinePresets(nextPresets.presets);
        } catch (loadError) {
            setError(loadError instanceof Error ? loadError.message : "Could not load workbench session.");
        } finally {
            setLoading(false);
        }
    }, [sessionId, token]);

    useEffect(() => {
        setMounted(true);
    }, []);

    useEffect(() => {
        if (!mounted || !isInitialized) return;
        if (!isAuthenticated) {
            setLoading(false);
            openAuthModal();
            return;
        }
        void loadWorkbench();
    }, [isAuthenticated, isInitialized, loadWorkbench, mounted, openAuthModal]);

    useEffect(() => {
        if (!token || !sessionId || !activeJobId) return;
        let stopped = false;
        let timer: number | null = null;

        const pollJob = async () => {
            try {
                const response = await getWorkbenchJob(sessionId, activeJobId, token);
                if (stopped) return;
                const nextJob = response.job;
                setJobs(response.jobs?.length ? response.jobs : [nextJob]);
                if (isTerminalJob(nextJob)) {
                    applyJobResult(nextJob);
                    setActionLoading(null);
                    setCancelingJob(false);
                    try {
                        const [nextMetadata, nextExports] = await Promise.all([
                            getWorkbenchMetadata(sessionId, token),
                            getWorkbenchExports(sessionId, EXPORT_FORMATS, token),
                        ]);
                        if (!stopped) {
                            setMetadata(nextMetadata);
                            setJobs(nextMetadata.jobs ?? response.jobs ?? [nextJob]);
                            setExportsData(nextExports);
                        }
                    } catch (refreshError) {
                        if (!stopped) setError(refreshError instanceof Error ? refreshError.message : "Could not refresh completed job state.");
                    }
                    return;
                }
                timer = window.setTimeout(pollJob, 1200);
            } catch (jobError) {
                if (!stopped) {
                    setActionLoading(null);
                    setCancelingJob(false);
                    setError(jobError instanceof Error ? jobError.message : "Could not read workbench job status.");
                }
            }
        };

        void pollJob();
        return () => {
            stopped = true;
            if (timer) window.clearTimeout(timer);
        };
    }, [activeJobId, applyJobResult, sessionId, token]);

    const pointFromPointer = useCallback((event: PointerEvent<HTMLDivElement>): ImagePoint | null => {
        const rect = event.currentTarget.getBoundingClientRect();
        const fit = imageFitBox(rect.width, rect.height, imageWidth, imageHeight);
        if (!fit || !imageWidth || !imageHeight) return null;
        const localX = event.clientX - rect.left - fit.left;
        const localY = event.clientY - rect.top - fit.top;
        if (localX < 0 || localY < 0 || localX > fit.width || localY > fit.height) return null;
        return clampImagePoint({
            x: (localX / fit.width) * Math.max(1, imageWidth - 1),
            y: (1 - localY / fit.height) * Math.max(1, imageHeight - 1),
        }, imageWidth, imageHeight);
    }, [imageHeight, imageWidth]);

    const commitImageSelection = useCallback((start: ImagePoint, end: ImagePoint) => {
        const cleanStart = clampImagePoint(start, imageWidth, imageHeight);
        const cleanEnd = clampImagePoint(end, imageWidth, imageHeight);
        if (interactionMode === "aperture") {
            setApertureX(String(Math.round(cleanEnd.x)));
            setApertureY(String(Math.round(cleanEnd.y)));
            return;
        }
        if (interactionMode === "rms") {
            setRmsX1(String(Math.round(Math.min(cleanStart.x, cleanEnd.x))));
            setRmsY1(String(Math.round(Math.min(cleanStart.y, cleanEnd.y))));
            setRmsX2(String(Math.round(Math.max(cleanStart.x, cleanEnd.x))));
            setRmsY2(String(Math.round(Math.max(cleanStart.y, cleanEnd.y))));
            return;
        }
        setPvX1(String(Math.round(cleanStart.x)));
        setPvY1(String(Math.round(cleanStart.y)));
        setPvX2(String(Math.round(cleanEnd.x)));
        setPvY2(String(Math.round(cleanEnd.y)));
    }, [imageHeight, imageWidth, interactionMode]);

    const handleImagePointerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
        if (!renderPlan?.image?.data_url || isBusy) return;
        const point = pointFromPointer(event);
        if (!point) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        setDragStart(point);
        setDragPreview(point);
        if (interactionMode === "aperture") {
            commitImageSelection(point, point);
        }
    }, [commitImageSelection, interactionMode, isBusy, pointFromPointer, renderPlan?.image?.data_url]);

    const handleImagePointerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
        if (!dragStart) return;
        const point = pointFromPointer(event);
        if (point) setDragPreview(point);
    }, [dragStart, pointFromPointer]);

    const handleImagePointerUp = useCallback((event: PointerEvent<HTMLDivElement>) => {
        if (!dragStart) return;
        const point = pointFromPointer(event) ?? dragPreview ?? dragStart;
        commitImageSelection(dragStart, point);
        setDragStart(null);
        setDragPreview(null);
    }, [commitImageSelection, dragPreview, dragStart, pointFromPointer]);

    const applyRenderPlan = async () => {
        await startOperationJob("render", {
            mode,
            channel: mode === "channel" ? channel : null,
            moment: mode === "moment" ? moment : null,
            colormap,
            stretch,
            contour_sigma: contourSigmaText
                .split(",")
                .map((value) => Number(value.trim()))
                .filter(Number.isFinite),
            rms_region: rmsX1 && rmsY1 && rmsX2 && rmsY2
                ? { x1: Number(rmsX1), y1: Number(rmsY1), x2: Number(rmsX2), y2: Number(rmsY2) }
                : null,
        }, "render");
    };

    const prepareProduct = async () => {
        await startOperationJob("prepare", {}, "prepare");
    };

    const extractSpectrum = async () => {
        await startOperationJob("spectrum", {
            x_pixel: apertureX ? Number(apertureX) : null,
            y_pixel: apertureY ? Number(apertureY) : null,
            aperture_radius_pixels: apertureRadius,
            max_points: 512,
        }, "spectrum");
    };

    async function planPvSlice() {
        await startOperationJob("pv_slice", {
            path: [
                { x: Number(pvX1 || 0), y: Number(pvY1 || 0) },
                { x: Number(pvX2 || imageWidth || 0), y: Number(pvY2 || imageHeight || 0) },
            ],
            width_pixels: pvWidth,
            max_points: 512,
        }, "pv");
    }

    const findLines = async () => {
        if (!token) return;
        setActionLoading("lines");
        setError(null);
        try {
            const nextLines = await getWorkbenchLineOverlays(sessionId, {
                observed_frequency_ghz: frequency ? Number(frequency) : undefined,
                line_preset_key: linePresetKey || undefined,
                redshift,
                tolerance_ghz: tolerance,
                top_n: 10,
            }, token);
            setLineData(nextLines);
            if (nextLines.presets?.length) setLinePresets(nextLines.presets);
            if (nextLines.query_observed_frequency_ghz) setFrequency(String(nextLines.query_observed_frequency_ghz));
        } catch (lineError) {
            setError(lineError instanceof Error ? lineError.message : "Could not query spectral-line overlays.");
        } finally {
            setActionLoading(null);
        }
    };

    const toggleWorkbenchLine = (line: SpectralLineRecord) => {
        const key = lineSelectionKey(line);
        setSelectedLineIds((current) =>
            current.includes(key)
                ? current.filter((item) => item !== key)
                : [...current, key]
        );
    };

    const exportSelectedLines = () => {
        const lines = (lineData?.lines || []) as SpectralLineRecord[];
        const selected = lines.filter((line) => selectedLineIds.includes(lineSelectionKey(line)));
        const payload = JSON.stringify(selected.length ? selected : lines, null, 2);
        const url = URL.createObjectURL(new Blob([payload], { type: "application/json" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = `${metadata?.filename || "cube"}-spectral-lines.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    const cancelActiveJob = async () => {
        if (!token || !runningJob) return;
        setCancelingJob(true);
        setError(null);
        try {
            const response = await cancelWorkbenchJob(sessionId, runningJob.job_id, token);
            setJobs(response.jobs?.length ? response.jobs : [response.job]);
            setActiveJobId(response.job.job_id);
        } catch (cancelError) {
            setError(cancelError instanceof Error ? cancelError.message : "Could not cancel workbench job.");
        } finally {
            setCancelingJob(false);
        }
    };

    if (!mounted) return null;

    if (!isAuthenticated) {
        return (
            <div className="flex h-screen w-full items-center justify-center bg-[#090d16] px-6 text-slate-200">
                <div className="w-full max-w-md rounded-lg border border-slate-800 bg-slate-950 p-6 shadow-xl">
                    <div className="mb-4 flex items-center gap-3">
                        <Telescope className="h-5 w-5 text-cyan-300" />
                        <h1 className="text-lg font-semibold text-white">Workbench access</h1>
                    </div>
                    <p className="text-sm leading-6 text-slate-400">Sign in to open this ALMA product workbench session.</p>
                    <button
                        type="button"
                        onClick={openAuthModal}
                        className="mt-5 inline-flex h-10 items-center gap-2 rounded-md bg-cyan-500 px-4 text-sm font-semibold text-slate-950 transition-colors hover:bg-cyan-400"
                    >
                        Sign in
                    </button>
                </div>
                <AuthModal />
            </div>
        );
    }

    return (
        <div className="flex h-screen w-full overflow-hidden bg-[#090d16] text-slate-200">
            <div className={`${sidebarOpen ? "w-[var(--q-sidebar-width)]" : "w-0"} shrink-0 overflow-hidden transition-all duration-300`}>
                <Sidebar />
            </div>

            <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
                <header className="border-b border-slate-800 bg-slate-950/95 px-5 py-3">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex min-w-0 items-center gap-3">
                            <Link href="/" className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-800 text-slate-400 transition-colors hover:border-slate-700 hover:text-white" title="Back to chat">
                                <ArrowLeft className="h-4 w-4" />
                            </Link>
                            <div className="min-w-0">
                                <div className="flex items-center gap-2">
                                    <Telescope className="h-4 w-4 text-cyan-300" />
                                    <h1 className="truncate text-base font-semibold text-white">{metadata?.filename || "Cube/Product Workbench"}</h1>
                                </div>
                                <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                                    <span>{metadata?.archive || "Archive"}</span>
                                    <span className="text-slate-700">/</span>
                                    <span>{metadata?.project_code || "No project code"}</span>
                                    <span className="text-slate-700">/</span>
                                    <span className="max-w-[360px] truncate font-mono">{metadata?.mous_uid || sessionId}</span>
                                </div>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className={`hidden rounded-md border px-2 py-1 font-mono text-[11px] md:block ${
                                cacheStatus === "cached"
                                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
                                    : "border-slate-800 bg-slate-900 text-slate-500"
                            }`}>
                                cache {cacheStatus}
                            </div>
                            <button
                                type="button"
                                onClick={prepareProduct}
                                disabled={isBusy}
                                className="inline-flex h-9 items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 text-xs font-semibold text-emerald-200 transition-colors hover:bg-emerald-500/15 disabled:opacity-60"
                            >
                                {actionLoading === "prepare" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                                Prepare
                            </button>
                            {metadata?.source_url && (
                                <a
                                    href={metadata.source_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-700 bg-slate-900 px-3 text-xs font-semibold text-slate-300 transition-colors hover:border-cyan-500/50 hover:text-cyan-200"
                                >
                                    <Download className="h-3.5 w-3.5" />
                                    FITS
                                </a>
                            )}
                            <button
                                type="button"
                                onClick={loadWorkbench}
                                disabled={loading}
                                className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-700 bg-slate-900 px-3 text-xs font-semibold text-slate-300 transition-colors hover:border-cyan-500/50 hover:text-cyan-200 disabled:opacity-60"
                            >
                                <RotateCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
                                Refresh
                            </button>
                        </div>
                    </div>
                </header>

                {error && (
                    <div className="flex items-center gap-2 border-b border-red-900/60 bg-red-950/40 px-5 py-2 text-xs text-red-200">
                        <AlertTriangle className="h-4 w-4 shrink-0" />
                        {error}
                    </div>
                )}

                {loading ? (
                    <div className="flex flex-1 items-center justify-center">
                        <div className="flex items-center gap-3 text-sm text-slate-400">
                            <Loader2 className="h-5 w-5 animate-spin text-cyan-300" />
                            Loading workbench
                        </div>
                    </div>
                ) : (
                    <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden xl:grid-cols-[280px_minmax(0,1fr)_360px]">
                        <aside className="min-h-0 overflow-y-auto border-r border-slate-800 bg-slate-950/70 p-4">
                            <section className="space-y-3">
                                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                    <Settings2 className="h-3.5 w-3.5" />
                                    Render
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    {(["image", "channel", "moment", "pv"] as RenderMode[]).map((item) => (
                                        <button
                                            key={item}
                                            type="button"
                                            onClick={() => setMode(item)}
                                            className={`h-9 rounded-md border px-2 text-xs font-semibold capitalize transition-colors ${
                                                mode === item
                                                    ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-100"
                                                    : "border-slate-800 bg-slate-900 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                                            }`}
                                        >
                                            {item}
                                        </button>
                                    ))}
                                </div>
                                <label className="block text-xs text-slate-400">
                                    Channel
                                    <input
                                        type="number"
                                        min={0}
                                        max={channelCount ?? undefined}
                                        value={channel}
                                        onChange={(event) => setChannel(Number(event.target.value))}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    />
                                </label>
                                <label className="block text-xs text-slate-400">
                                    Moment
                                    <select
                                        value={moment}
                                        onChange={(event) => setMoment(Number(event.target.value))}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    >
                                        <option value={0}>0</option>
                                        <option value={1}>1</option>
                                        <option value={2}>2</option>
                                    </select>
                                </label>
                                <label className="block text-xs text-slate-400">
                                    Colormap
                                    <select
                                        value={colormap}
                                        onChange={(event) => setColormap(event.target.value)}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    >
                                        <option value="inferno">Inferno</option>
                                        <option value="viridis">Viridis</option>
                                        <option value="magma">Magma</option>
                                        <option value="cividis">Cividis</option>
                                    </select>
                                </label>
                                <label className="block text-xs text-slate-400">
                                    Stretch
                                    <select
                                        value={stretch}
                                        onChange={(event) => setStretch(event.target.value)}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    >
                                        <option value="linear">Linear</option>
                                        <option value="asinh">Asinh</option>
                                        <option value="log">Log</option>
                                        <option value="sqrt">Sqrt</option>
                                    </select>
                                </label>
                                <label className="block text-xs text-slate-400">
                                    Contours sigma
                                    <input
                                        type="text"
                                        value={contourSigmaText}
                                        onChange={(event) => setContourSigmaText(event.target.value)}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    />
                                </label>
                                <div className="space-y-2 rounded-md border border-slate-800 bg-slate-900/60 p-3">
                                    <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">RMS Region</div>
                                    <div className="grid grid-cols-2 gap-2">
                                        <label className="block text-xs text-slate-400">
                                            X1
                                            <input
                                                type="number"
                                                value={rmsX1}
                                                onChange={(event) => setRmsX1(event.target.value)}
                                                className="mt-1 h-8 w-full rounded-md border border-slate-800 bg-slate-950 px-2 font-mono text-xs text-slate-100 outline-none focus:border-cyan-500"
                                            />
                                        </label>
                                        <label className="block text-xs text-slate-400">
                                            Y1
                                            <input
                                                type="number"
                                                value={rmsY1}
                                                onChange={(event) => setRmsY1(event.target.value)}
                                                className="mt-1 h-8 w-full rounded-md border border-slate-800 bg-slate-950 px-2 font-mono text-xs text-slate-100 outline-none focus:border-cyan-500"
                                            />
                                        </label>
                                        <label className="block text-xs text-slate-400">
                                            X2
                                            <input
                                                type="number"
                                                value={rmsX2}
                                                onChange={(event) => setRmsX2(event.target.value)}
                                                className="mt-1 h-8 w-full rounded-md border border-slate-800 bg-slate-950 px-2 font-mono text-xs text-slate-100 outline-none focus:border-cyan-500"
                                            />
                                        </label>
                                        <label className="block text-xs text-slate-400">
                                            Y2
                                            <input
                                                type="number"
                                                value={rmsY2}
                                                onChange={(event) => setRmsY2(event.target.value)}
                                                className="mt-1 h-8 w-full rounded-md border border-slate-800 bg-slate-950 px-2 font-mono text-xs text-slate-100 outline-none focus:border-cyan-500"
                                            />
                                        </label>
                                    </div>
                                </div>
                                <button
                                    type="button"
                                    onClick={applyRenderPlan}
                                    disabled={isBusy}
                                    className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md bg-cyan-500 text-sm font-semibold text-slate-950 transition-colors hover:bg-cyan-400 disabled:opacity-60"
                                >
                                    {actionLoading === "render" ? <Loader2 className="h-4 w-4 animate-spin" /> : <SlidersHorizontal className="h-4 w-4" />}
                                    Apply
                                </button>
                            </section>

                            <section className="mt-7 space-y-3 border-t border-slate-800 pt-5">
                                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                    <Waves className="h-3.5 w-3.5" />
                                    Spectrum
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    <label className="block text-xs text-slate-400">
                                        X pixel
                                        <input
                                            type="number"
                                            value={apertureX}
                                            onChange={(event) => setApertureX(event.target.value)}
                                            className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Y pixel
                                        <input
                                            type="number"
                                            value={apertureY}
                                            onChange={(event) => setApertureY(event.target.value)}
                                            className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                        />
                                    </label>
                                </div>
                                <label className="block text-xs text-slate-400">
                                    Radius pixels
                                    <input
                                        type="number"
                                        min={0.5}
                                        step={0.5}
                                        value={apertureRadius}
                                        onChange={(event) => setApertureRadius(Number(event.target.value))}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    />
                                </label>
                                <button
                                    type="button"
                                    onClick={extractSpectrum}
                                    disabled={isBusy}
                                    className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-cyan-500/50 bg-cyan-500/10 text-sm font-semibold text-cyan-200 transition-colors hover:bg-cyan-500/15 disabled:opacity-60"
                                >
                                    {actionLoading === "spectrum" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Waves className="h-4 w-4" />}
                                    Extract Spectrum
                                </button>
                            </section>

                            <section className="mt-7 space-y-3 border-t border-slate-800 pt-5">
                                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                    <Layers className="h-3.5 w-3.5" />
                                    PV Slice
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    <label className="block text-xs text-slate-400">
                                        X1
                                        <input
                                            type="number"
                                            value={pvX1}
                                            onChange={(event) => setPvX1(event.target.value)}
                                            className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Y1
                                        <input
                                            type="number"
                                            value={pvY1}
                                            onChange={(event) => setPvY1(event.target.value)}
                                            className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        X2
                                        <input
                                            type="number"
                                            value={pvX2}
                                            onChange={(event) => setPvX2(event.target.value)}
                                            className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                        />
                                    </label>
                                    <label className="block text-xs text-slate-400">
                                        Y2
                                        <input
                                            type="number"
                                            value={pvY2}
                                            onChange={(event) => setPvY2(event.target.value)}
                                            className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                        />
                                    </label>
                                </div>
                                <label className="block text-xs text-slate-400">
                                    Width pixels
                                    <input
                                        type="number"
                                        min={0.5}
                                        step={0.5}
                                        value={pvWidth}
                                        onChange={(event) => setPvWidth(Number(event.target.value))}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    />
                                </label>
                                <button
                                    type="button"
                                    onClick={planPvSlice}
                                    disabled={isBusy}
                                    className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-amber-500/50 bg-amber-500/10 text-sm font-semibold text-amber-200 transition-colors hover:bg-amber-500/15 disabled:opacity-60"
                                >
                                    {actionLoading === "pv" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Layers className="h-4 w-4" />}
                                    Plan PV Slice
                                </button>
                            </section>

                            <section className="mt-7 space-y-3 border-t border-slate-800 pt-5">
                                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                    <Radio className="h-3.5 w-3.5" />
                                    Lines
                                </div>
                                <label className="block text-xs text-slate-400">
                                    Common line
                                    <select
                                        value={linePresetKey}
                                        onChange={(event) => setLinePresetKey(event.target.value)}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    >
                                        <option value="">Header/manual frequency</option>
                                        {linePresets.map((preset) => (
                                            <option key={preset.key} value={preset.key}>
                                                {preset.label} ({formatNumber(preset.rest_frequency_ghz, 6)} GHz)
                                            </option>
                                        ))}
                                    </select>
                                </label>
                                <label className="block text-xs text-slate-400">
                                    Observed GHz
                                    <input
                                        type="number"
                                        step="0.0001"
                                        value={frequency}
                                        onChange={(event) => setFrequency(event.target.value)}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    />
                                </label>
                                <label className="block text-xs text-slate-400">
                                    Redshift
                                    <input
                                        type="number"
                                        min={0}
                                        step="0.001"
                                        value={redshift}
                                        onChange={(event) => setRedshift(Number(event.target.value))}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    />
                                </label>
                                <label className="block text-xs text-slate-400">
                                    Tolerance GHz
                                    <input
                                        type="number"
                                        min={0}
                                        step="0.001"
                                        value={tolerance}
                                        onChange={(event) => setTolerance(Number(event.target.value))}
                                        className="mt-1 h-9 w-full rounded-md border border-slate-800 bg-slate-900 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-500"
                                    />
                                </label>
                                <button
                                    type="button"
                                    onClick={findLines}
                                    disabled={isBusy}
                                    className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-emerald-500/50 bg-emerald-500/10 text-sm font-semibold text-emerald-200 transition-colors hover:bg-emerald-500/15 disabled:opacity-60"
                                >
                                    {actionLoading === "lines" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                                    Find Lines
                                </button>
                            </section>
                        </aside>

                        <section className="min-h-0 overflow-y-auto bg-[#0b111d]">
                            <div className="grid border-b border-slate-800 bg-slate-950/50 sm:grid-cols-2 lg:grid-cols-5">
                                <Stat label="Status" value={metadata?.status || "-"} />
                                <Stat label="Shape" value={shape.length ? shape.join(" x ") : "-"} />
                                <Stat label="Rest GHz" value={formatNumber(restFrequency, 6)} />
                                <Stat label="Beam" value={beamMajor && beamMinor ? `${formatNumber(beamMajor, 3)} x ${formatNumber(beamMinor, 3)} arcsec` : "-"} />
                                <Stat label="RMS" value={sensitivity ? `${formatNumber(sensitivity, 4)} mJy/beam` : "-"} />
                            </div>

                            <div className="p-5">
                                <div className="min-h-[420px] rounded-lg border border-slate-800 bg-slate-950">
                                    <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                                        <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                                            {mode === "channel" ? <Layers className="h-4 w-4 text-cyan-300" /> : <Waves className="h-4 w-4 text-cyan-300" />}
                                            {renderSummary}
                                        </div>
                                        <div className="rounded-md border border-slate-800 px-2 py-1 font-mono text-[11px] text-slate-500">
                                            {colormap} / {stretch}
                                        </div>
                                    </div>
                                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-4 py-2">
                                        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                                            <SlidersHorizontal className="h-3.5 w-3.5" />
                                            Image Select
                                        </div>
                                        <div className="grid grid-cols-3 gap-1">
                                            {(["aperture", "rms", "pv"] as InteractionMode[]).map((item) => (
                                                <button
                                                    key={item}
                                                    type="button"
                                                    onClick={() => setInteractionMode(item)}
                                                    className={`h-7 rounded-md border px-2 text-[11px] font-semibold uppercase transition-colors ${
                                                        interactionMode === item
                                                            ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-100"
                                                            : "border-slate-800 bg-slate-900 text-slate-500 hover:border-slate-700 hover:text-slate-300"
                                                    }`}
                                                >
                                                    {item}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="grid min-h-[360px] place-items-center px-5">
                                        <div className="w-full max-w-3xl">
                                            <div
                                                className={`relative h-[520px] min-h-72 overflow-hidden rounded-md border border-slate-800 bg-slate-950 touch-none select-none ${
                                                    renderPlan?.image?.data_url && !isBusy ? "cursor-crosshair" : ""
                                                }`}
                                                onPointerDown={handleImagePointerDown}
                                                onPointerMove={handleImagePointerMove}
                                                onPointerUp={handleImagePointerUp}
                                                onPointerCancel={() => {
                                                    setDragStart(null);
                                                    setDragPreview(null);
                                                }}
                                            >
                                                {renderPlan?.image?.data_url ? (
                                                    // eslint-disable-next-line @next/next/no-img-element
                                                    <img
                                                        src={renderPlan.image.data_url}
                                                        alt={renderPlan.image.label || "Rendered FITS plane"}
                                                        className="pointer-events-none absolute inset-0 h-full w-full object-contain"
                                                    />
                                                ) : (
                                                    <div className="relative h-full bg-[linear-gradient(135deg,#111827_0%,#060a12_55%,#14212d_100%)]">
                                                        <div className="absolute inset-x-8 top-1/2 h-px bg-cyan-300/20" />
                                                        <div className="absolute inset-y-8 left-1/2 w-px bg-cyan-300/20" />
                                                        <div className="absolute left-[18%] top-[22%] h-20 w-28 rounded-full border border-cyan-300/25 bg-cyan-300/5 blur-[1px]" />
                                                        <div className="absolute right-[18%] bottom-[20%] h-24 w-32 rounded-full border border-amber-300/20 bg-amber-300/5 blur-[1px]" />
                                                    </div>
                                                )}
                                                <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                                                    {rmsRegionPercent && (
                                                        <rect
                                                            x={rmsRegionPercent.left}
                                                            y={rmsRegionPercent.top}
                                                            width={rmsRegionPercent.width}
                                                            height={rmsRegionPercent.height}
                                                            fill="rgba(251, 191, 36, 0.08)"
                                                            stroke="#fbbf24"
                                                            strokeWidth="0.35"
                                                            strokeDasharray="1 0.8"
                                                        />
                                                    )}
                                                    {pvLinePercent?.start && pvLinePercent.end && (
                                                        <line
                                                            x1={pvLinePercent.start.x}
                                                            y1={pvLinePercent.start.y}
                                                            x2={pvLinePercent.end.x}
                                                            y2={pvLinePercent.end.y}
                                                            stroke="#38bdf8"
                                                            strokeWidth="0.45"
                                                            strokeLinecap="round"
                                                        />
                                                    )}
                                                    {aperturePercent && (
                                                        <g>
                                                            <circle cx={aperturePercent.x} cy={aperturePercent.y} r="1.25" fill="none" stroke="#34d399" strokeWidth="0.35" />
                                                            <line x1={aperturePercent.x - 1.9} y1={aperturePercent.y} x2={aperturePercent.x + 1.9} y2={aperturePercent.y} stroke="#34d399" strokeWidth="0.25" />
                                                            <line x1={aperturePercent.x} y1={aperturePercent.y - 1.9} x2={aperturePercent.x} y2={aperturePercent.y + 1.9} stroke="#34d399" strokeWidth="0.25" />
                                                        </g>
                                                    )}
                                                    {dragStartPercent && dragPreviewPercent && interactionMode === "rms" && (
                                                        <rect
                                                            x={Math.min(dragStartPercent.x, dragPreviewPercent.x)}
                                                            y={Math.min(dragStartPercent.y, dragPreviewPercent.y)}
                                                            width={Math.abs(dragPreviewPercent.x - dragStartPercent.x)}
                                                            height={Math.abs(dragPreviewPercent.y - dragStartPercent.y)}
                                                            fill="rgba(251, 191, 36, 0.12)"
                                                            stroke="#fde68a"
                                                            strokeWidth="0.35"
                                                        />
                                                    )}
                                                    {dragStartPercent && dragPreviewPercent && interactionMode === "pv" && (
                                                        <line
                                                            x1={dragStartPercent.x}
                                                            y1={dragStartPercent.y}
                                                            x2={dragPreviewPercent.x}
                                                            y2={dragPreviewPercent.y}
                                                            stroke="#7dd3fc"
                                                            strokeWidth="0.5"
                                                            strokeLinecap="round"
                                                            strokeDasharray="1 0.8"
                                                        />
                                                    )}
                                                </svg>
                                                <div className="absolute bottom-4 left-4 rounded border border-slate-700 bg-slate-950/90 px-2 py-1 font-mono text-[11px] text-slate-400">
                                                    {renderPlan?.image?.wcs_status === "present" ? "RA/Dec WCS" : shape.length ? `${shape.at(-2)} x ${shape.at(-1)}` : "image plane"}
                                                </div>
                                                <div className="absolute bottom-4 right-4 rounded border border-slate-700 bg-slate-950/90 px-2 py-1 font-mono text-[11px] text-slate-400">
                                                    {interactionMode === "aperture" ? "click aperture" : interactionMode === "rms" ? "drag RMS box" : "drag PV line"}
                                                </div>
                                            </div>
                                            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
                                                <span>Render state: {renderPlan?.status || "metadata loaded"}</span>
                                                <span className="flex flex-wrap items-center gap-2">
                                                    {renderPlan?.stats?.rms !== undefined && renderPlan.stats.rms !== null && (
                                                        <span>RMS {formatNumber(renderPlan.stats.rms, 4)} {renderPlan.stats.unit || ""} ({renderPlan.stats.rms_method || "auto"})</span>
                                                    )}
                                                    {renderPlan?.image?.analysis_product && (
                                                        <span>Analysis {renderPlan.image.analysis_product}</span>
                                                    )}
                                                    {renderPlan?.contours?.levels && renderPlan.contours.levels.length > 0 && (
                                                        <span>Contours {renderPlan.contours.levels.map((level) => formatNumber(level, 3)).join(", ")}</span>
                                                    )}
                                                    {renderPlan?.image?.line_labels && renderPlan.image.line_labels.length > 0 && (
                                                        <span>
                                                            Lines {renderPlan.image.line_labels.map((line) => asText(line.label, "Line")).join(", ")}
                                                        </span>
                                                    )}
                                                    {channelCount !== null && <span>{channelCount} channels</span>}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {(spectrumPlan || pvSlicePlan) && (
                                    <div className="mt-5 grid gap-5 lg:grid-cols-2">
                                        {spectrumPlan && (
                                            <section className="rounded-lg border border-slate-800 bg-slate-950">
                                                <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                                                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                                                        <Waves className="h-4 w-4 text-cyan-300" />
                                                        Spectrum Extraction
                                                    </div>
                                                    <span className="text-[11px] text-slate-500">{spectrumPlan.status}</span>
                                                </div>
                                                <div className="grid gap-3 p-4 text-xs text-slate-400 sm:grid-cols-2">
                                                    <div>
                                                        <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Aperture</div>
                                                        <div className="mt-1 font-mono text-slate-200">
                                                            x={formatNumber(spectrumPlan.extraction.x_pixel, 1)}, y={formatNumber(spectrumPlan.extraction.y_pixel, 1)}, r={formatNumber(spectrumPlan.extraction.aperture_radius_pixels, 1)} px
                                                        </div>
                                                    </div>
                                                    <div>
                                                        <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Spectral Axis</div>
                                                        <div className="mt-1 font-mono text-slate-200">{spectrumPlan.spectral_axis.label}</div>
                                                    </div>
                                                    <div>
                                                        <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Channels</div>
                                                        <div className="mt-1 font-mono text-slate-200">{spectrumPlan.spectral_axis.channel_count}</div>
                                                    </div>
                                                    <div>
                                                        <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Samples</div>
                                                        <div className="mt-1 font-mono text-slate-200">{spectrumPlan.series.x.length} planned</div>
                                                    </div>
                                                    {spectrumPlan.series.analysis_product && (
                                                        <div>
                                                            <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Analysis</div>
                                                            <div className="mt-1 font-mono text-slate-200">{spectrumPlan.series.analysis_product}</div>
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="px-4 pb-4">
                                                    <SpectrumMiniPlot series={spectrumPlan.series} lineData={lineData} />
                                                </div>
                                            </section>
                                        )}

                                        {pvSlicePlan && (
                                            <section className="rounded-lg border border-slate-800 bg-slate-950">
                                                <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                                                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                                                        <Layers className="h-4 w-4 text-amber-300" />
                                                        PV Slice
                                                    </div>
                                                    <span className="text-[11px] text-slate-500">{pvSlicePlan.status}</span>
                                                </div>
                                                <div className="grid gap-3 p-4 text-xs text-slate-400 sm:grid-cols-2">
                                                    <div>
                                                        <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Path</div>
                                                        <div className="mt-1 font-mono text-slate-200">
                                                            {pvSlicePlan.path.map((point) => `(${formatNumber(point.x, 1)}, ${formatNumber(point.y, 1)})`).join(" -> ")}
                                                        </div>
                                                    </div>
                                                    <div>
                                                        <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Width</div>
                                                        <div className="mt-1 font-mono text-slate-200">{formatNumber(pvSlicePlan.width_pixels, 1)} px</div>
                                                    </div>
                                                    <div>
                                                        <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Spatial Samples</div>
                                                        <div className="mt-1 font-mono text-slate-200">{formatNumber(asRecord(pvSlicePlan.spatial_axis).samples, 0)}</div>
                                                    </div>
                                                    <div>
                                                        <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Spectral Axis</div>
                                                        <div className="mt-1 font-mono text-slate-200">{asText(asRecord(pvSlicePlan.spectral_axis).label)}</div>
                                                    </div>
                                                    {pvSlicePlan.image.analysis_product && (
                                                        <div>
                                                            <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Analysis</div>
                                                            <div className="mt-1 font-mono text-slate-200">{pvSlicePlan.image.analysis_product}</div>
                                                        </div>
                                                    )}
                                                </div>
                                                {pvSlicePlan.image.data_url && (
                                                    <div className="px-4 pb-4">
                                                        {/* eslint-disable-next-line @next/next/no-img-element */}
                                                        <img
                                                            src={pvSlicePlan.image.data_url}
                                                            alt="PV slice"
                                                            className="w-full rounded-md border border-slate-800 bg-slate-900"
                                                        />
                                                    </div>
                                                )}
                                            </section>
                                        )}
                                    </div>
                                )}

                                <div className="mt-5 grid gap-5 lg:grid-cols-2">
                                    <section className="rounded-lg border border-slate-800 bg-slate-950">
                                        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                                            <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                                                <Radio className="h-4 w-4 text-emerald-300" />
                                                Line Overlays
                                            </div>
                                            {lineData && (
                                                <div className="flex items-center gap-2 text-[11px] text-slate-500">
                                                    {lineData.backend && (
                                                        <span
                                                            className="rounded border border-slate-800 bg-slate-900 px-1.5 py-0.5 uppercase tracking-wide"
                                                            title={lineData.query_note || lineData.query_error || undefined}
                                                        >
                                                            {lineData.backend === "slap" ? "SLAP fallback" : lineData.backend}
                                                        </span>
                                                    )}
                                                    <span>{lineData.n_matches} matches</span>
                                                    <button
                                                        type="button"
                                                        onClick={exportSelectedLines}
                                                        className="rounded border border-slate-800 bg-slate-900 px-2 py-1 text-[10px] text-cyan-200"
                                                    >
                                                        Export {selectedLineIds.length ? "selected" : "all"}
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                        <div className="max-h-[420px] overflow-y-auto p-3">
                                            {lineData?.lines.length ? (
                                                <SpectralLineTable
                                                    lines={(lineData.lines as SpectralLineRecord[]).map((line) => ({
                                                        ...line,
                                                        frequency_ghz: Number(line.frequency_ghz ?? line.rest_frequency_ghz),
                                                    }))}
                                                    selectedIds={selectedLineIds}
                                                    onToggleSelect={toggleWorkbenchLine}
                                                    compact
                                                />
                                            ) : (
                                                <div className="px-4 py-8 text-sm text-slate-500">No line overlays loaded.</div>
                                            )}
                                        </div>
                                    </section>

                                    <section className="rounded-lg border border-slate-800 bg-slate-950">
                                        <div className="flex items-center gap-2 border-b border-slate-800 px-4 py-3 text-sm font-semibold text-slate-100">
                                            <FileCode className="h-4 w-4 text-cyan-300" />
                                            Export
                                        </div>
                                        <div className="divide-y divide-slate-900">
                                            {exportsData && Object.entries(exportsData.exports).length > 0 ? (
                                                Object.entries(exportsData.exports).map(([format, item]) => (
                                                    <div key={format} className="p-4">
                                                        <div className="mb-2 flex items-center justify-between gap-2">
                                                                <div>
                                                                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">{format}</div>
                                                                    <div className="font-mono text-[11px] text-slate-600">{item.filename}</div>
                                                                </div>
                                                                <div className="flex items-center gap-2">
                                                                    <CopyButton text={item.content} />
                                                                    <DownloadTextButton filename={item.filename} text={item.content} />
                                                                </div>
                                                            </div>
                                                        <pre className="max-h-28 overflow-auto rounded-md border border-slate-800 bg-slate-900 p-3 text-[11px] leading-5 text-slate-300">
                                                            {item.content}
                                                        </pre>
                                                    </div>
                                                ))
                                            ) : (
                                                <div className="px-4 py-8 text-sm text-slate-500">No export plans available.</div>
                                            )}
                                        </div>
                                    </section>
                                </div>
                            </div>
                        </section>

                        <aside className="min-h-0 overflow-y-auto border-l border-slate-800 bg-slate-950/70 p-4">
                            {activeJob && (
                                <div className="mb-5">
                                    <WorkbenchJobPanel job={activeJob} onCancel={cancelActiveJob} canceling={cancelingJob} />
                                </div>
                            )}

                            <section className="rounded-lg border border-slate-800 bg-slate-950">
                                <div className="flex items-center gap-2 border-b border-slate-800 px-4 py-3 text-sm font-semibold text-slate-100">
                                    <ExternalLink className="h-4 w-4 text-cyan-300" />
                                    Product
                                </div>
                                <dl className="divide-y divide-slate-900 text-xs">
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">Filename</dt>
                                        <dd className="truncate font-mono text-slate-200" title={metadata?.filename}>{metadata?.filename || "-"}</dd>
                                    </div>
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">Archive</dt>
                                        <dd className="text-slate-200">{metadata?.archive || "-"}</dd>
                                    </div>
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">Project</dt>
                                        <dd className="truncate font-mono text-slate-200">{metadata?.project_code || "-"}</dd>
                                    </div>
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">Object</dt>
                                        <dd className="truncate font-mono text-slate-200">{asText(metadataRecord.object || metadataRecord.OBJECT)}</dd>
                                    </div>
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">Unit</dt>
                                        <dd className="font-mono text-slate-200">{asText(metadataRecord.unit || metadataRecord.bunit || metadataRecord.BUNIT)}</dd>
                                    </div>
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">Header</dt>
                                        <dd className="font-mono text-slate-200">{headerStrategy}</dd>
                                    </div>
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">Cache</dt>
                                        <dd className="font-mono text-slate-200">
                                            {cacheStatus}{cachedBytes ? ` / ${(cachedBytes / (1024 * 1024)).toFixed(1)} MB` : ""}
                                        </dd>
                                    </div>
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">Preview</dt>
                                        <dd className="font-mono text-slate-200">
                                            {previewStatus}
                                            {previewBytes ? ` / ${formatBytes(previewBytes)}` : ""}
                                            {previewShape ? ` / ${previewShape}` : ""}
                                        </dd>
                                    </div>
                                    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 px-4 py-3">
                                        <dt className="text-slate-500">User Cache</dt>
                                        <dd className="font-mono text-slate-200">
                                            {userCachedBytes !== null && userCacheLimitBytes !== null
                                                ? `${(userCachedBytes / (1024 * 1024)).toFixed(1)} / ${(userCacheLimitBytes / (1024 * 1024)).toFixed(0)} MB`
                                                : "-"}
                                        </dd>
                                    </div>
                                </dl>
                            </section>

                            <section className="mt-5 rounded-lg border border-slate-800 bg-slate-950">
                                <div className="flex items-center gap-2 border-b border-slate-800 px-4 py-3 text-sm font-semibold text-slate-100">
                                    <AlertTriangle className="h-4 w-4 text-amber-300" />
                                    Evidence
                                </div>
                                <div className="space-y-4 p-4 text-xs text-slate-400">
                                    <div>
                                        <div className="mb-1 font-semibold uppercase tracking-wide text-slate-500">Confidence</div>
                                        <div className="text-slate-200">{asText(evidence.confidence, "unknown")}</div>
                                    </div>
                                    <div>
                                        <div className="mb-1 font-semibold uppercase tracking-wide text-slate-500">WCS</div>
                                        <div className="text-slate-200">{asText(evidence.wcs_status, "unknown")}</div>
                                    </div>
                                    <div>
                                        <div className="mb-1 font-semibold uppercase tracking-wide text-slate-500">Spectral Axis</div>
                                        <div className="text-slate-200">{asText(evidence.spectral_axis_status, "unknown")}</div>
                                    </div>
                                    {Array.isArray(evidence.operations) && evidence.operations.length > 0 && (
                                        <div>
                                            <div className="mb-1 font-semibold uppercase tracking-wide text-slate-500">Operations</div>
                                            <div className="flex flex-wrap gap-1.5">
                                                {evidence.operations.slice(0, 8).map((item) => (
                                                    <span key={String(item)} className="rounded-md border border-slate-800 bg-slate-900 px-2 py-1 font-mono text-[10px] text-slate-300">
                                                        {String(item)}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {Array.isArray(evidence.headers_inspected) && evidence.headers_inspected.length > 0 && (
                                        <div>
                                            <div className="mb-1 font-semibold uppercase tracking-wide text-slate-500">Headers</div>
                                            <div className="max-h-28 overflow-y-auto rounded-md border border-slate-800 bg-slate-900 p-2 font-mono text-[11px] leading-5 text-slate-300">
                                                {evidence.headers_inspected.slice(0, 40).join(", ")}
                                            </div>
                                        </div>
                                    )}
                                    {Array.isArray(evidence.assumptions) && evidence.assumptions.length > 0 && (
                                        <div>
                                            <div className="mb-1 font-semibold uppercase tracking-wide text-slate-500">Assumptions</div>
                                            <ul className="space-y-1.5">
                                                {evidence.assumptions.slice(0, 4).map((item) => (
                                                    <li key={String(item)} className="leading-5">{String(item)}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {Array.isArray(evidence.warnings) && evidence.warnings.length > 0 && (
                                        <div>
                                            <div className="mb-1 font-semibold uppercase tracking-wide text-slate-500">Warnings</div>
                                            <ul className="space-y-1.5 text-amber-200">
                                                {evidence.warnings.slice(0, 4).map((item) => (
                                                    <li key={String(item)} className="leading-5">{String(item)}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            </section>
                        </aside>
                    </div>
                )}
            </main>
            <AuthModal />
        </div>
    );
}
