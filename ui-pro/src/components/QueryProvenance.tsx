"use client";

import { Check, ChevronDown, ChevronRight, Copy, Terminal, X } from "lucide-react";
import { useState } from "react";
import type { ToolRequest, ToolTraceCall } from "../lib/api";
// Pure selection logic lives in lib/provenance.js so the node test harness can
// exercise the SSE/reload consumption contract without a React renderer (CX-24).
import { requestsFrom as requestsFromLib } from "../lib/provenance";

/** Feature 1 — the raw-query provenance surface.
 *
 * A collapsible, monospace, copy-to-clipboard block showing the EXACT request a
 * tool made. The payload is built and secret-redacted server-side
 * (core/provenance.py); this component only renders it.
 */

const KIND_LABEL: Record<string, string> = {
    adql: "ADQL",
    http: "HTTP",
    ads: "ADS",
    params: "Params",
    args: "Args",
};

const KIND_STYLE: Record<string, string> = {
    adql: "bg-primary/10 text-primary",
    http: "bg-sky-500/10 text-sky-400",
    ads: "bg-emerald-500/10 text-emerald-400",
    params: "bg-amber-500/10 text-amber-400",
    args: "bg-slate-500/10 text-slate-400",
};

/** Default ON locally; set NEXT_PUBLIC_SHOW_QUERY_PROVENANCE=0 to hide. */
export function queryProvenanceEnabled(): boolean {
    return process.env.NEXT_PUBLIC_SHOW_QUERY_PROVENANCE !== "0";
}

function requestsFrom(
    request?: ToolRequest,
    calls?: ToolTraceCall[],
    toolName?: string,
): { name: string; request: ToolRequest }[] {
    return requestsFromLib(request, calls, toolName) as { name: string; request: ToolRequest }[];
}

/** The request's structured params as renderable [key, line] pairs: scalars
 *  verbatim, objects/arrays compact JSON. Empty → no block. Real requests
 *  carry context beyond `text` here (MOC request_pattern + per-ID orders,
 *  VLASS soda_url, SIA endpoint) that was previously invisible and never
 *  copied. (A1 CX-05 sliver) */
function paramEntries(request: ToolRequest): [string, string][] {
    const params = request.params;
    if (!params || typeof params !== "object") return [];
    return Object.entries(params).map(([key, value]) => {
        let text: string;
        if (value !== null && typeof value === "object") {
            try {
                text = JSON.stringify(value);
            } catch {
                text = String(value);
            }
        } else {
            text = String(value);
        }
        return [key, text];
    });
}

/** Copy payload = the request text plus its params lines, so a paste
 *  reproduces everything the block shows. (A1 CX-05 sliver) */
function copyPayload(request: ToolRequest): string {
    const params = paramEntries(request);
    if (params.length === 0) return request.text;
    return `${request.text}\n${params.map(([k, v]) => `${k}: ${v}`).join("\n")}`;
}

interface QueryProvenanceProps {
    /** A single card's own request (preferred — unambiguous). */
    request?: ToolRequest;
    /** Or the turn's whole trace, optionally narrowed to one tool. */
    calls?: ToolTraceCall[];
    toolName?: string;
    label?: string;
    className?: string;
}

export function QueryProvenance({ request, calls, toolName, label, className }: QueryProvenanceProps) {
    const [open, setOpen] = useState(false);
    const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
    const [copyError, setCopyError] = useState<number | null>(null);

    if (!queryProvenanceEnabled()) return null;
    const entries = requestsFrom(request, calls, toolName);
    if (entries.length === 0) return null;

    const execCommandCopy = (text: string): boolean => {
        // The textarea must come off the DOM on EVERY path — an execCommand
        // throw used to leave it attached (CX-40).
        let ta: HTMLTextAreaElement | null = null;
        try {
            ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "fixed";
            ta.style.opacity = "0";
            document.body.appendChild(ta);
            ta.select();
            return document.execCommand("copy");
        } catch {
            return false;
        } finally {
            if (ta && ta.parentNode) ta.parentNode.removeChild(ta);
        }
    };

    const copy = async (text: string, index: number) => {
        // Only claim success once a write actually lands. The Clipboard API
        // REJECTS (not just "absent") on a non-secure origin or denied
        // permission — fall through to execCommand in BOTH cases (CX-21), and
        // show the error state if nothing copied rather than a false checkmark.
        let ok = false;
        try {
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
                ok = true;
            }
        } catch {
            ok = false;  // permission denied / insecure origin → try the fallback
        }
        if (!ok) ok = execCommandCopy(text);
        if (ok) {
            setCopiedIndex(index);
            setTimeout(() => setCopiedIndex(null), 2000);
        } else {
            setCopyError(index);
            setTimeout(() => setCopyError(null), 2500);
        }
    };

    return (
        <div className={`mt-3 ${className || ""}`}>
            <button
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
                {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                <Terminal className="w-3.5 h-3.5" />
                <span>{label || (open ? "Hide query" : "Show query")}</span>
                {!open && entries.length > 1 && (
                    <span className="text-muted-foreground/70">({entries.length})</span>
                )}
            </button>

            {open && (
                <div className="mt-2 space-y-2">
                    {entries.map((entry, index) => (
                        <div
                            key={`${entry.name}-${index}`}
                            className="rounded-lg border border-border bg-muted/40 overflow-hidden"
                        >
                            <div className="flex items-center justify-between gap-2 px-3 py-1.5 border-b border-border/60">
                                <div className="flex items-center gap-2 min-w-0">
                                    <span
                                        className={`px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${
                                            KIND_STYLE[entry.request.kind] || KIND_STYLE.args
                                        }`}
                                    >
                                        {KIND_LABEL[entry.request.kind] || entry.request.kind}
                                    </span>
                                    {entry.name && (
                                        <span className="text-[11px] text-muted-foreground truncate font-mono">
                                            {entry.name}
                                        </span>
                                    )}
                                </div>
                                <button
                                    onClick={() => copy(copyPayload(entry.request), index)}
                                    title={copyError === index ? "Copy failed — select and copy manually" : "Copy request"}
                                    className="p-1 text-muted-foreground hover:text-foreground transition-colors shrink-0"
                                >
                                    {copiedIndex === index ? (
                                        <Check className="w-3.5 h-3.5" />
                                    ) : copyError === index ? (
                                        <X className="w-3.5 h-3.5 text-red-400" />
                                    ) : (
                                        <Copy className="w-3.5 h-3.5" />
                                    )}
                                </button>
                            </div>
                            <pre className="px-3 py-2 text-[11px] font-mono whitespace-pre-wrap break-words overflow-x-auto max-h-64 text-foreground/90">
                                {entry.request.text}
                            </pre>
                            {paramEntries(entry.request).length > 0 && (
                                <div className="px-3 py-2 border-t border-border/60 space-y-0.5 overflow-x-auto max-h-40">
                                    {paramEntries(entry.request).map(([key, value]) => (
                                        <div key={key} className="text-[11px] font-mono whitespace-pre-wrap break-words">
                                            <span className="text-muted-foreground">{key}:</span>{" "}
                                            <span className="text-foreground/90">{value}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                            {entry.request.snippet && (
                                <pre className="px-3 py-2 text-[11px] font-mono whitespace-pre-wrap break-words border-t border-border/60 text-muted-foreground">
                                    {entry.request.snippet}
                                </pre>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
