"use client";

/* eslint-disable @next/next/no-img-element */
import {
    AlertCircle, BookOpen, ChartLine, CheckCircle2, ChevronDown, Database, Download, FileText, GitMerge,
    Globe, Image as ImageIcon, Loader2, Orbit, Plug, Search, Sparkles,
} from "lucide-react";
import { useEffect, useId, useState, type ReactNode } from "react";
import type { ThoughtStep } from "./ThoughtProcessWidget";
import type { WebDecision, WebSource } from "../lib/types";
import { buildResearchTimeline, REASONING_VISIBLE } from "../lib/research-timeline.js";

interface ResearchTimelineProps {
    status: "running" | "completed";
    steps: ThoughtStep[];
    /** Collapse once the answer has started (the user can still reopen it). */
    forceCollapsed?: boolean;
    startTime?: Date | string | number;
    duration?: number;
    webDecision?: WebDecision;
    webSources?: WebSource[];
    paperCount?: number;
    /** The turn is still streaming. Defaults to `status === "running"`; it
     *  stays true after the answer starts, so a mid-answer tool still reads
     *  as running when the panel is reopened. */
    streaming?: boolean;
    /** Streamed model reasoning, shown under the timeline. */
    children?: ReactNode;
}

type Item = ReturnType<typeof buildResearchTimeline>["sections"][number]["items"][number];

function faviconUrl(domain?: string): string {
    return domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=32` : "";
}

function Favicon({ domain, className = "" }: { domain?: string; className?: string }) {
    const [failed, setFailed] = useState(false);
    if (!domain || failed) return <Globe className={`h-3.5 w-3.5 shrink-0 ${className}`} aria-hidden="true" />;
    return (
        <img
            src={faviconUrl(domain)}
            alt=""
            aria-hidden="true"
            referrerPolicy="no-referrer"
            onError={() => setFailed(true)}
            className={`h-3.5 w-3.5 shrink-0 rounded-sm ${className}`}
        />
    );
}

function ItemIcon({ item }: { item: Item }) {
    const cls = "h-3.5 w-3.5 shrink-0";
    if (item.status === "running") return <Loader2 className={`${cls} animate-spin text-primary`} aria-hidden="true" />;
    if (item.status === "error") return <AlertCircle className={`${cls} q-err`} aria-hidden="true" />;
    switch (item.icon) {
        case "favicon": return <Favicon domain={item.domain} />;
        case "search": return <Search className={cls} aria-hidden="true" />;
        case "alert": return <AlertCircle className={`${cls} q-warn`} aria-hidden="true" />;
        case "page": return <FileText className={cls} aria-hidden="true" />;
        case "book": return <BookOpen className={cls} aria-hidden="true" />;
        case "download": return <Download className={cls} aria-hidden="true" />;
        case "image": return <ImageIcon className={cls} aria-hidden="true" />;
        case "chart": return <ChartLine className={cls} aria-hidden="true" />;
        case "merge": return <GitMerge className={cls} aria-hidden="true" />;
        case "database": return <Database className={cls} aria-hidden="true" />;
        default: return <Sparkles className={cls} aria-hidden="true" />;
    }
}

function formatMs(ms: number): string {
    return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)}s`;
}

/**
 * A call to one of the user's MCP servers: violet chip with the server name
 * as a badge, then the outcome (green check + duration, or red "Failed" with
 * the server's error on hover). Without this a failed MCP call looked exactly
 * like a successful one.
 */
function McpChip({ item, index }: { item: Item; index: number }) {
    const mcp = item.mcp;
    const running = item.status === "running";
    const error = item.status === "error";
    const outcome = running ? "running" : error ? `failed${mcp?.error ? `: ${mcp.error}` : ""}` : "succeeded";
    const title = `${mcp?.server ?? ""} MCP server · ${item.text} · ${outcome}`;
    return (
        <span
            className={`q-item-in inline-flex max-w-full items-center gap-1.5 rounded-lg px-2 py-1 text-[11px] leading-5 ${error ? "q-err" : "q-mcp-chip"}`}
            style={{
                animationDelay: `${Math.min(index, 8) * 55}ms`,
                ...(error ? { background: "rgba(248, 113, 113, 0.08)", border: "1px solid rgba(248, 113, 113, 0.3)" } : {}),
            }}
            title={title}
        >
            {running
                ? <Loader2 className="q-mcp-icon h-3.5 w-3.5 shrink-0 animate-spin" aria-hidden="true" />
                : <Plug className={`h-3.5 w-3.5 shrink-0 ${error ? "" : "q-mcp-icon"}`} aria-hidden="true" />}
            <span className={`shrink-0 rounded px-1.5 text-[10px] font-semibold leading-4 ${error ? "" : "q-mcp-badge"}`}
                style={error ? { background: "rgba(248, 113, 113, 0.15)" } : undefined}>
                {mcp?.server}
            </span>
            <span className={`truncate ${running ? "q-shimmer" : ""}`}>{item.text}</span>
            {running && typeof item.elapsedSeconds === "number" && item.elapsedSeconds >= 15 && (
                <span className="shrink-0 font-mono" style={{ color: "var(--q-text-faint)" }}>{item.elapsedSeconds}s</span>
            )}
            {!running && !error && (
                <span className="q-ok inline-flex shrink-0 items-center gap-0.5">
                    <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                    {typeof mcp?.ms === "number" && <span className="font-mono text-[10px]">{formatMs(mcp.ms)}</span>}
                </span>
            )}
            {error && (
                <span className="inline-flex min-w-0 shrink items-center gap-0.5 font-medium">
                    <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    <span className="truncate">{mcp?.error ? `Failed: ${mcp.error}` : "Failed"}</span>
                </span>
            )}
            {!error && <span className="sr-only">{outcome}</span>}
        </span>
    );
}

function Chip({ item, index, mono }: { item: Item; index: number; mono: boolean }) {
    if (item.mcp) return <McpChip item={item} index={index} />;
    const running = item.status === "running";
    const error = item.status === "error";
    const body = (
        <>
            <ItemIcon item={item} />
            <span className={`truncate ${running ? "q-shimmer" : ""}`}>{item.text}</span>
            {running && typeof item.elapsedSeconds === "number" && item.elapsedSeconds >= 15 && (
                <span className="shrink-0 font-mono" style={{ color: "var(--q-text-faint)" }}>{item.elapsedSeconds}s</span>
            )}
        </>
    );
    const cls = `q-item-in inline-flex max-w-full items-center gap-1.5 rounded-lg px-2.5 py-1 text-[11px] leading-5 ${mono ? "font-mono" : ""} ${error ? "q-err" : ""}`;
    const style = {
        animationDelay: `${Math.min(index, 8) * 55}ms`,
        background: error ? "rgba(248, 113, 113, 0.08)" : "var(--q-glass-control)",
        border: `1px solid ${error ? "rgba(248, 113, 113, 0.22)" : "var(--q-glass-border)"}`,
        color: error ? undefined : "var(--q-text-secondary)",
    };
    if (item.url) {
        return (
            <a href={item.url} target="_blank" rel="noopener noreferrer" title={item.url}
                className={`${cls} transition-colors hover:border-primary/40`} style={style}>
                {body}
            </a>
        );
    }
    return <span className={cls} style={style} title={item.text}>{body}</span>;
}

/**
 * Live research view for an assistant turn: a vertical timeline of what the
 * backend is doing ("Searching the web" query chips, "Querying" archive
 * chips, "Reading" source chips, "Wrapping up"), with the running phase
 * shimmering. Collapses into a one-line summary once the answer starts.
 */
export function ResearchTimeline({
    status, steps, forceCollapsed, startTime, duration,
    webDecision, webSources, paperCount, streaming, children,
}: ResearchTimelineProps) {
    const running = status === "running";
    const timeline = buildResearchTimeline({ steps, running: streaming ?? running, webDecision, webSources, paperCount });
    const bodyId = useId();
    const [showAllThoughts, setShowAllThoughts] = useState(false);

    // A manual toggle only holds for the phase it was made in: when the run
    // starts, finishes or the answer arrives, the default applies again
    // (open while working, closed after).
    const phase = `${running}-${Boolean(forceCollapsed)}`;
    const [toggle, setToggle] = useState<{ phase: string; open: boolean } | null>(null);
    const defaultOpen = !forceCollapsed && (running || timeline.sections.length <= 1);
    const open = toggle && toggle.phase === phase ? toggle.open : defaultOpen;

    // Only a running turn ticks; a finished one keeps its last value (or the
    // persisted duration), never "now minus an old timestamp".
    const [seconds, setSeconds] = useState(0);
    useEffect(() => {
        if (!running) return;
        const start = startTime ? new Date(startTime).getTime() : Date.now();
        const tick = () => setSeconds(Math.max(0, Math.round((Date.now() - start) / 1000)));
        tick();
        const interval = setInterval(tick, 1000);
        return () => clearInterval(interval);
    }, [running, startTime]);
    const displaySeconds = !running && duration !== undefined ? duration : seconds;

    const summary = timeline.sourceCount > 0
        ? `${timeline.sourceCount} source${timeline.sourceCount === 1 ? "" : "s"}`
        : timeline.stepCount > 0
            ? `${timeline.stepCount} step${timeline.stepCount === 1 ? "" : "s"}`
            : "";
    const sections = timeline.sections;
    const showPlaceholder = running && sections.length === 0;
    // A plain reply (only plumbing steps, no reasoning text) did no research:
    // an empty "Research" header would suggest otherwise.
    if (!running && sections.length === 0 && !children) return null;

    return (
        <div
            className="q-research glass-surface my-3 w-full max-w-2xl overflow-hidden rounded-2xl transition-[border-color] duration-500"
            data-running={running ? "true" : "false"}
            style={{ borderColor: running ? "var(--q-border-strong)" : "var(--q-glass-border)" }}
        >
            <button
                type="button"
                onClick={() => setToggle({ phase, open: !open })}
                aria-expanded={open}
                aria-controls={bodyId}
                className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors hover:bg-[var(--q-glass-hover)]"
            >
                <span className="flex min-w-0 items-center gap-2.5">
                    <Orbit
                        className={`h-[18px] w-[18px] shrink-0 ${running ? "q-orbit-spin text-primary" : ""}`}
                        style={running ? undefined : { color: "var(--q-text-muted)" }}
                        aria-hidden="true"
                    />
                    <span className="text-sm font-medium" style={{ color: "var(--q-text)" }}>
                        {running ? "Researching" : "Research"}
                    </span>
                    {/* A reloaded turn has no live timer; "0s" would be wrong. */}
                    {(running || displaySeconds > 0) && (
                        <span className="rounded-md px-1.5 py-0.5 font-mono text-[10px] tabular-nums"
                            style={{ color: "var(--q-text-muted)", background: "var(--q-glass-bg)" }}>
                            {displaySeconds}s
                        </span>
                    )}
                    {!open && running && timeline.current && (
                        <span className="q-shimmer truncate text-xs">{timeline.current}</span>
                    )}
                </span>
                <span className="flex shrink-0 items-center gap-2">
                    {timeline.favicons.length > 0 && (
                        <span className="flex -space-x-1" aria-hidden="true">
                            {timeline.favicons.map((d) => (
                                <span key={d} className="flex h-4 w-4 items-center justify-center rounded-full"
                                    style={{ background: "var(--q-bg)", boxShadow: "0 0 0 1.5px var(--q-bg)" }}>
                                    <Favicon domain={d} className="rounded-full" />
                                </span>
                            ))}
                        </span>
                    )}
                    {summary && <span className="text-xs" style={{ color: "var(--q-text-muted)" }}>{summary}</span>}
                    <ChevronDown
                        className={`h-4 w-4 transition-transform duration-300 ${open ? "rotate-180" : ""}`}
                        style={{ color: "var(--q-text-muted)" }}
                        aria-hidden="true"
                    />
                </span>
            </button>

            {/* Screen readers hear each new phase once, not the whole list. */}
            <span className="sr-only" aria-live="polite">{running ? timeline.current || "" : ""}</span>

            {/* inert: collapsed chips/links must not take focus or be read out. */}
            <div id={bodyId} className="q-collapse" data-open={open ? "true" : "false"} inert={!open}>
                <div>
                    <div className="px-4 pb-4 pt-1">
                        <ol className="relative">
                            {showPlaceholder && (
                                <li className="relative pl-6">
                                    <span className="q-dot q-dot-live absolute left-0 top-[6px] h-2 w-2 rounded-full text-primary" aria-hidden="true" />
                                    <span className="q-shimmer text-[13px]">Understanding the question</span>
                                </li>
                            )}
                            {sections.map((sec, i) => {
                                const live = sec.status === "running";
                                const last = i === sections.length - 1;
                                const mono = sec.kind === "searching" || sec.kind === "reading";
                                return (
                                    <li key={sec.kind} className={`q-item-in relative pl-6 ${last ? "" : "pb-4"}`}>
                                        {!last && (
                                            <span className="q-line-grow absolute bottom-0 left-[3.5px] top-[18px] w-px"
                                                style={{ background: "var(--q-border)" }} aria-hidden="true" />
                                        )}
                                        <span
                                            className={`q-dot absolute left-0 top-[6px] h-2 w-2 rounded-full transition-colors duration-500 ${live ? "q-dot-live text-primary" : sec.status === "error" ? "q-err" : ""}`}
                                            style={live || sec.status === "error" ? undefined : { color: "var(--q-text-faint)" }}
                                            aria-hidden="true"
                                        />
                                        <div className={`text-[13px] ${live ? "q-shimmer" : ""}`}
                                            style={live ? undefined : { color: "var(--q-text-muted)" }}>
                                            {sec.title}
                                        </div>
                                        {sec.kind === "reasoning" ? (
                                            <div className="mt-1 space-y-1">
                                                {sec.hiddenCount ? (
                                                    <button type="button" onClick={() => setShowAllThoughts((v) => !v)}
                                                        className="text-[11px] underline-offset-2 hover:underline"
                                                        style={{ color: "var(--q-text-faint)" }}>
                                                        {showAllThoughts
                                                            ? "Show fewer thoughts"
                                                            : `Show ${sec.hiddenCount} earlier thought${sec.hiddenCount === 1 ? "" : "s"}`}
                                                    </button>
                                                ) : null}
                                                {(showAllThoughts ? sec.items : sec.items.slice(-REASONING_VISIBLE)).map((item, j) => (
                                                    <p key={`${j}-${item.text}`} title={item.text}
                                                        className={`q-item-in text-xs leading-relaxed ${showAllThoughts ? "" : "line-clamp-2"}`}
                                                        style={{ color: "var(--q-text-secondary)" }}>
                                                        {item.text}
                                                    </p>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="mt-1.5 flex flex-wrap gap-1.5">
                                                {sec.items.map((item, j) => (
                                                    <Chip key={`${item.text}-${item.url || ""}`} item={item} index={j} mono={mono} />
                                                ))}
                                                {sec.overflow ? (
                                                    <span className="q-item-in inline-flex items-center rounded-lg px-2.5 py-1 font-mono text-[11px] leading-5"
                                                        style={{ background: "var(--q-glass-control)", border: "1px solid var(--q-glass-border)", color: "var(--q-text-muted)" }}>
                                                        +{sec.overflow} more
                                                    </span>
                                                ) : null}
                                            </div>
                                        )}
                                    </li>
                                );
                            })}
                        </ol>
                        {children ? <div className="mt-3">{children}</div> : null}
                    </div>
                </div>
            </div>
        </div>
    );
}
