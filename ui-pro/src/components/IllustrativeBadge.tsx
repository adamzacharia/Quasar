"use client";

import { AlertTriangle, Info } from "lucide-react";

/**
 * Persistent "this figure is not the evidence" marker (Feature 2).
 *
 * A rendered figure is a lossy view of its data — axes get clipped, points
 * overplot, colours quantise — and that stays true even when the underlying
 * table is complete. So this badge is unconditional on every plot and image
 * card rather than only on ones derived from truncated results: the citable
 * artifact is always the source table or the exported CSV, never the picture.
 *
 * When the card's meta says the figure was built from a TRUNCATED PREVIEW of
 * its result (`previewDerived`), the badge escalates to an amber warning with
 * the row counts — a preview-derived figure is not merely lossy, it is drawn
 * from incomplete data and must never read as the full dataset (f2-CX-11).
 */
export function IllustrativeBadge({
    className = "",
    previewDerived = false,
    shownRows,
    totalRows,
}: {
    className?: string;
    /** True when the figure was rendered from a truncated preview rather than
     *  the full stored result. */
    previewDerived?: boolean;
    /** Rows the figure was actually built from, when known. */
    shownRows?: number;
    /** True row count of the full result, when known. */
    totalRows?: number;
}) {
    if (previewDerived) {
        const counts = shownRows !== undefined && totalRows !== undefined
            ? ` (${shownRows.toLocaleString()} of ${totalRows.toLocaleString()} rows)`
            : "";
        return (
            <div
                className={`flex items-center gap-1.5 text-[11px] text-amber-300 ${className}`}
                title="This figure was rendered from a truncated preview of its result, not the full dataset. Re-run the query or export the full CSV before drawing conclusions from it."
            >
                <AlertTriangle className="h-3 w-3 shrink-0" />
                <span>Derived from a preview{counts} — not the full dataset.</span>
            </div>
        );
    }
    return (
        <div
            className={`flex items-center gap-1.5 text-[11px] text-slate-400 ${className}`}
            title="Figures are for orientation only. Cite the source table or the exported CSV — not the rendered plot."
        >
            <Info className="h-3 w-3 shrink-0 opacity-70" />
            <span>Illustrative — cite the source table/CSV, not the plot.</span>
        </div>
    );
}
