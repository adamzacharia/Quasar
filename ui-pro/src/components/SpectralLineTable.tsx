"use client";

import { useMemo } from "react";
import { AlertTriangle, Check, Clipboard, GitCompareArrows, Radio, Telescope } from "lucide-react";
import type { SpectralLineRecord } from "@/lib/api";

function number(value: unknown, digits = 6): string {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return "—";
    return parsed.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function text(value: unknown): string {
    if (value === null || value === undefined || value === "") return "—";
    return String(value);
}

function lineKey(line: SpectralLineRecord, index: number): string {
    return String(line.line_id || line.raw_line_ids?.join("-") || `${line.species}-${line.transition}-${line.frequency_ghz}-${index}`);
}

export function SpectralFrequencyRuler({ lines }: { lines: SpectralLineRecord[] }) {
    const frequencies = lines
        .map((line) => Number(line.observed_frequency_ghz ?? line.frequency_ghz))
        .filter(Number.isFinite);
    if (!frequencies.length) return null;
    const minimum = Math.min(...frequencies);
    const maximum = Math.max(...frequencies);
    const span = maximum - minimum || Math.max(minimum * 1e-6, 0.001);

    return (
        <div className="rounded-lg border border-slate-800 bg-slate-950 px-4 py-3">
            <div className="mb-2 flex items-center justify-between text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                <span>Observed-frequency ruler</span>
                <span className="font-mono normal-case">{number(minimum)} – {number(maximum)} GHz</span>
            </div>
            <div className="relative h-10 rounded bg-slate-900">
                <div className="absolute left-2 right-2 top-5 h-px bg-slate-700" />
                {lines.slice(0, 120).map((line, index) => {
                    const frequency = Number(line.observed_frequency_ghz ?? line.frequency_ghz);
                    if (!Number.isFinite(frequency)) return null;
                    const left = 2 + ((frequency - minimum) / span) * 96;
                    return (
                        <div
                            key={lineKey(line, index)}
                            className="group absolute top-2 h-6 w-px bg-cyan-400"
                            style={{ left: `${left}%` }}
                            title={`${text(line.species || line.formula)} ${text(line.transition)} — ${frequency.toFixed(6)} GHz`}
                        >
                            <span className="absolute left-1 top-0 hidden whitespace-nowrap rounded bg-slate-800 px-1.5 py-1 text-[9px] text-slate-200 shadow group-hover:block">
                                {text(line.species || line.formula)} {text(line.transition)}
                            </span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

export function SpectralLineTable({
    lines,
    selectedIds = [],
    onToggleSelect,
    onAnalyzeConfusion,
    onCompareRaw,
    compact = false,
}: {
    lines: SpectralLineRecord[];
    selectedIds?: string[];
    onToggleSelect?: (line: SpectralLineRecord) => void;
    onAnalyzeConfusion?: (line: SpectralLineRecord) => void;
    onCompareRaw?: (line: SpectralLineRecord) => void;
    compact?: boolean;
}) {
    const selected = useMemo(() => new Set(selectedIds), [selectedIds]);

    const copy = async (value: unknown) => {
        if (value !== null && value !== undefined) {
            await navigator.clipboard.writeText(String(value));
        }
    };

    return (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="min-w-full divide-y divide-slate-800 text-left text-xs">
                <thead className="bg-slate-950 text-[10px] uppercase tracking-wide text-slate-500">
                    <tr>
                        {onToggleSelect && <th className="w-10 px-3 py-2">Use</th>}
                        <th className="px-3 py-2">Species / transition</th>
                        <th className="px-3 py-2">Rest GHz</th>
                        <th className="px-3 py-2">Observed GHz</th>
                        {!compact && <th className="px-3 py-2">Predicted / measured</th>}
                        <th className="px-3 py-2">Evidence</th>
                        {!compact && <th className="px-3 py-2">Energy / strength</th>}
                        <th className="px-3 py-2">Actions</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 bg-slate-900/60">
                    {lines.map((line, index) => {
                        const key = lineKey(line, index);
                        const isSelected = selected.has(key);
                        const catalogs = line.catalogs?.join(", ") || "Splatalogue";
                        return (
                            <tr key={key} className={isSelected ? "bg-cyan-500/5" : "hover:bg-white/[0.03]"}>
                                {onToggleSelect && (
                                    <td className="px-3 py-3 align-top">
                                        <button
                                            type="button"
                                            onClick={() => onToggleSelect(line)}
                                            className={`flex h-5 w-5 items-center justify-center rounded border ${isSelected ? "border-cyan-400 bg-cyan-400 text-slate-950" : "border-slate-600 text-transparent"}`}
                                            aria-label={isSelected ? "Remove line from selection" : "Select line"}
                                        >
                                            <Check className="h-3.5 w-3.5" />
                                        </button>
                                    </td>
                                )}
                                <td className="max-w-[260px] px-3 py-3 align-top">
                                    <div className="font-semibold text-slate-100">{text(line.species || line.formula)}</div>
                                    <div className="mt-1 font-mono text-[11px] text-cyan-200">{text(line.transition || line.resolved_quantum_numbers)}</div>
                                    {line.unresolved_quantum_numbers && (
                                        <div className="mt-1 text-[10px] text-slate-500">HFS/QN: {line.unresolved_quantum_numbers}</div>
                                    )}
                                </td>
                                <td className="px-3 py-3 align-top font-mono text-slate-200">
                                    <button type="button" className="hover:text-cyan-200" onClick={() => copy(line.frequency_ghz)}>
                                        {number(line.frequency_ghz)}
                                    </button>
                                    <div className="mt-1 text-[10px] text-slate-500">± {number(line.frequency_uncertainty_mhz, 4)} MHz</div>
                                </td>
                                <td className="px-3 py-3 align-top font-mono text-slate-200">
                                    <button type="button" className="hover:text-cyan-200" onClick={() => copy(line.observed_frequency_ghz)}>
                                        {number(line.observed_frequency_ghz)}
                                    </button>
                                    {line.alma_bands?.length ? (
                                        <div className="mt-1 text-[10px] text-slate-500">Band {line.alma_bands.join("/")}</div>
                                    ) : null}
                                    {line.alma_band_edge_warning && (
                                        <div className="mt-1 flex max-w-[180px] gap-1 text-[10px] text-amber-300" title={line.alma_band_edge_warning}>
                                            <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" /> Receiver edge
                                        </div>
                                    )}
                                </td>
                                {!compact && (
                                    <td className="px-3 py-3 align-top font-mono text-[11px]">
                                        <div className="text-slate-300">P {number(line.predicted_frequency_ghz)}</div>
                                        <div className="mt-1 text-slate-300">M {number(line.measured_frequency_ghz)}</div>
                                        <div className="mt-1 max-w-[190px] font-sans text-[10px] text-slate-500">{text(line.frequency_selection_reason)}</div>
                                    </td>
                                )}
                                <td className="px-3 py-3 align-top">
                                    <div className="text-slate-300">{catalogs}</div>
                                    <div className="mt-1 flex flex-wrap gap-1">
                                        {line.nrao_recommended && <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[9px] text-emerald-300">NRAO</span>}
                                        {line.astronomically_observed && <span className="rounded bg-violet-500/10 px-1.5 py-0.5 text-[9px] text-violet-300">Observed</span>}
                                        {line.score !== undefined && <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] text-amber-200">{number(line.score, 1)} {line.classification}</span>}
                                    </div>
                                </td>
                                {!compact && (
                                    <td className="px-3 py-3 align-top font-mono text-[10px] text-slate-400">
                                        <div>E<sub>u</sub> {number(line.upper_energy_k, 3)} K</div>
                                        <div className="mt-1">Sμ² {number(line.sijmu2, 4)}</div>
                                        <div className="mt-1">log A {number(line.aij_log, 4)}</div>
                                    </td>
                                )}
                                <td className="px-3 py-3 align-top">
                                    <div className="flex gap-1">
                                        <button type="button" onClick={() => copy(line.frequency_ghz)} className="rounded border border-slate-700 p-1.5 text-slate-400 hover:text-white" title="Copy rest frequency">
                                            <Clipboard className="h-3.5 w-3.5" />
                                        </button>
                                        {onAnalyzeConfusion && (
                                            <button type="button" onClick={() => onAnalyzeConfusion(line)} className="rounded border border-slate-700 p-1.5 text-slate-400 hover:text-amber-200" title="Analyze nearby contaminants">
                                                <Radio className="h-3.5 w-3.5" />
                                            </button>
                                        )}
                                        {onCompareRaw && (
                                            <button type="button" onClick={() => onCompareRaw(line)} className="rounded border border-slate-700 p-1.5 text-slate-400 hover:text-cyan-200" title="Compare raw catalog entries">
                                                <GitCompareArrows className="h-3.5 w-3.5" />
                                            </button>
                                        )}
                                        {onToggleSelect && (
                                            <button type="button" onClick={() => onToggleSelect(line)} className="rounded border border-slate-700 p-1.5 text-slate-400 hover:text-cyan-200" title="Use for ALMA coverage">
                                                <Telescope className="h-3.5 w-3.5" />
                                            </button>
                                        )}
                                    </div>
                                </td>
                            </tr>
                        );
                    })}
                    {!lines.length && (
                        <tr>
                            <td colSpan={8} className="px-4 py-10 text-center text-sm text-slate-500">
                                No spectral lines on this page.
                            </td>
                        </tr>
                    )}
                </tbody>
            </table>
        </div>
    );
}

export function lineSelectionKey(line: SpectralLineRecord, index = 0): string {
    return lineKey(line, index);
}

