"use client";

/* eslint-disable @next/next/no-img-element */
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { ExternalLink } from "lucide-react";
import type { WebSource } from "../lib/types";
import { citationLabel, domainOf, sourceCardDomId } from "../lib/web-citations.js";

interface WebCitationChipProps {
    /** Evidence id, e.g. "W3". */
    citationId: string;
    /** The source this id points at (undefined while sources are still loading). */
    source?: WebSource;
    /** Owning assistant message id: scopes the scroll target in the sources grid. */
    messageId: string;
}

function favicon(url: string): string {
    const domain = domainOf(url);
    return domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=32` : "";
}

/**
 * Inline citation chip for web evidence ("[W3]" in the answer renders as a
 * small "3"). Hover or keyboard focus shows a card with the source's favicon,
 * title, domain, date and a three-line excerpt; clicking scrolls to and
 * highlights the source in the grid below the answer. The card's "Open" link
 * opens the page in a new tab.
 */
export function WebCitationChip({ citationId, source, messageId }: WebCitationChipProps) {
    const [open, setOpen] = useState(false);
    const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const cardId = useId();
    const label = citationLabel(citationId);

    const cancelClose = () => {
        if (closeTimer.current) clearTimeout(closeTimer.current);
        closeTimer.current = null;
    };
    const scheduleClose = () => {
        cancelClose();
        closeTimer.current = setTimeout(() => setOpen(false), 180);
    };
    useEffect(() => () => cancelClose(), []);

    const jumpToSource = useCallback(() => {
        const el = document.getElementById(sourceCardDomId(messageId, citationId));
        if (!el) return;
        // The grid collapses "Also consulted" sources; open it when the target is inside.
        const details = el.closest("details");
        if (details && !details.open) details.open = true;
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        el.setAttribute("data-highlight", "true");
        setTimeout(() => el.removeAttribute("data-highlight"), 1800);
    }, [messageId, citationId]);

    const quality = source?.evidenceQuality;
    const title = source?.title || source?.domain || citationId;
    const domain = source?.domain || (source?.url ? domainOf(source.url) : "");

    return (
        <span
            className="relative inline-block align-baseline"
            onMouseEnter={() => { cancelClose(); setOpen(true); }}
            onMouseLeave={scheduleClose}
            // Focus anywhere inside (the chip or the popup's Open link) keeps the
            // card open; leaving the whole chip closes it (guard CX-23).
            onFocus={() => { cancelClose(); setOpen(true); }}
            onBlur={(e) => {
                const next = e.relatedTarget as Node | null;
                if (!next || !e.currentTarget.contains(next)) scheduleClose();
            }}
            onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
        >
            <button
                type="button"
                data-web-cite={citationId}
                aria-describedby={open ? cardId : undefined}
                aria-label={source ? `Source ${label}: ${title}` : `Source ${label}`}
                onClick={(e) => { e.preventDefault(); jumpToSource(); }}
                className="mx-[1px] inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-[1px] items-center justify-center rounded-md px-1 text-[10px] font-semibold leading-none transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-cyan-400/70"
                style={{
                    color: "var(--q-text)",
                    background: "var(--q-glass-control)",
                    border: "1px solid var(--q-glass-border)",
                }}
            >
                {label}
            </button>
            {open && source && (
                <span
                    id={cardId}
                    role="tooltip"
                    onMouseEnter={cancelClose}
                    onMouseLeave={scheduleClose}
                    className="glass-popover absolute left-1/2 top-full z-40 mt-1.5 block w-72 max-w-[calc(100vw-2rem)] -translate-x-1/2 rounded-xl p-3 text-left shadow-xl"
                >
                    <span className="flex items-center gap-2">
                        {source.url && (
                            <img
                                src={favicon(source.url)}
                                alt=""
                                className="h-4 w-4 shrink-0 rounded-sm"
                                onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                            />
                        )}
                        <span className="min-w-0 flex-1 truncate text-[11px]" style={{ color: "var(--q-text-muted)" }}>
                            {domain}{source.publishedDate ? ` · ${source.publishedDate}` : ""}
                        </span>
                        <span className="shrink-0 rounded px-1 text-[10px] font-semibold" style={{ color: "var(--q-text-muted)", border: "1px solid var(--q-glass-border)" }}>
                            {label}
                        </span>
                    </span>
                    <span className="mt-1.5 block text-xs font-semibold leading-snug" style={{ color: "var(--q-text)" }}>
                        {title}
                    </span>
                    {source.snippet && (
                        <span className="mt-1 block text-[11px] leading-relaxed line-clamp-3" style={{ color: "var(--q-text-muted)" }}>
                            {source.snippet}
                        </span>
                    )}
                    <span className="mt-2 flex items-center justify-between gap-2">
                        <span className="truncate text-[10px]" style={{ color: "var(--q-text-muted)" }}>
                            {quality?.label || "Web source"}
                        </span>
                        <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-medium hover:text-primary"
                            style={{ color: "var(--q-text)", border: "1px solid var(--q-glass-border)" }}
                        >
                            Open <ExternalLink className="h-3 w-3" />
                        </a>
                    </span>
                </span>
            )}
        </span>
    );
}
