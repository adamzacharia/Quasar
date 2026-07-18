/**
 * Full-result export decision logic (Feature 2).
 *
 * The Feature-2 principle: truncated/partial/preview data must NEVER masquerade
 * as complete. These are the pure decisions behind DataTableCard's download
 * button — what to offer, how to label it, and how to read the export route's
 * response — kept as plain JS beside eval-mode.js so `node --test
 * tests/*.test.mjs` can exercise every branch without a React harness.
 */

/** @typedef {"complete"|"truncated"|"unknown"} Completeness */

const fmt = (n) => Number(n).toLocaleString("en-US");

/**
 * Tri-state completeness of a data card (f2-CX-03).
 *
 * `truncated` is a Feature-2 field; cards replayed from history that predate it
 * carry neither `truncated` nor `totalRows`. Defaulting those to "complete"
 * (the old `data.truncated ?? false`) let a historical 10k-row preview export
 * as if it were the whole result — so the absence of the flag is its own state,
 * and "unknown" is never labeled complete.
 *
 * @param {boolean|null|undefined} truncated
 * @returns {Completeness}
 */
export function completenessOf(truncated) {
    if (truncated === true) return "truncated";
    if (truncated === false) return "complete";
    return "unknown";
}

/**
 * Everything the download button needs decided in one place.
 *
 * @param {{
 *   totalRows?: number|null,
 *   truncated?: boolean|null,
 *   resultId?: string|null,
 *   shownRowCount: number,
 *   strictExport?: boolean,
 *   upstreamPartial?: boolean,
 *   upstreamTotal?: number|null,
 * }} card
 * @returns {{
 *   completeness: Completeness,
 *   canExportFull: boolean,
 *   localAllowed: boolean,
 *   disabled: boolean,
 *   buttonLabel: string,
 *   tooltip: string,
 *   footerText: string,
 *   footerTone: "default"|"warning",
 * }}
 */
export function exportPlan(card) {
    const plan = basePlan(card);
    const { upstreamPartial = false, upstreamTotal = null } = card;
    // Upstream-partial (f2-CX-21): the STORED frame is itself a truncated
    // slice of the remote result (e.g. SIA capped rows before storage), which
    // stored-vs-displayed math can never see. The server export still streams
    // every stored row — but nothing may read as "the complete dataset".
    if (!upstreamPartial) return plan;
    const stored = card.totalRows ?? card.shownRowCount ?? null;
    const kept = upstreamTotal && stored !== null
        ? ` (kept ${fmt(stored)} of ${fmt(upstreamTotal)})`
        : "";
    return {
        ...plan,
        buttonLabel: plan.completeness === "complete" && plan.canExportFull
            ? plan.buttonLabel.replace(/^Download CSV/, "Export stored rows")
            : plan.buttonLabel,
        tooltip: `${plan.tooltip} Note: the upstream service truncated this result before it was stored${kept} — the export contains every stored row, but not every matching row.`,
        footerText: `${plan.footerText} · upstream-truncated result${kept}`,
        footerTone: /** @type {const} */ ("warning"),
    };
}

/** The upstream-agnostic plan (see exportPlan for the f2-CX-21 overlay). */
function basePlan({ totalRows, truncated, resultId, shownRowCount, strictExport = false }) {
    const shown = Number(shownRowCount) || 0;
    const completeness = completenessOf(truncated);
    const canExportFull = Boolean(resultId);
    // A missing totalRows must not silently become "the total": for "unknown"
    // the true count is exactly what we don't know (f2-CX-03), and a TRUNCATED
    // card without a total must never fabricate "N of N" (f2-CX-17). Only a
    // "complete" card may backfill from shown — there they are the same number.
    const total = completeness === "complete" ? (totalRows ?? shown) : (totalRows ?? null);

    // ── Server export available: the primary, full-fidelity path ──
    if (canExportFull) {
        return {
            completeness,
            canExportFull,
            localAllowed: false,
            disabled: false,
            buttonLabel:
                completeness === "complete" ? `Download CSV (${fmt(total)} rows)`
                : completeness === "truncated" ? (total === null ? "Export full CSV" : `Export full CSV (${fmt(total)} rows)`)
                : "Export full CSV",
            tooltip: "Streams every row and every column from the server.",
            ...footerFor(completeness, shown, total),
        };
    }

    // ── No server export: local download of the DISPLAYED table, or nothing ──
    if (completeness === "truncated") {
        // We cannot produce complete data; offering the preview would be the
        // silent-partial bug Feature 2 exists to prevent.
        return {
            completeness,
            canExportFull,
            localAllowed: false,
            disabled: true,
            buttonLabel: total === null
                ? `Preview only (${fmt(shown)} rows shown — total unknown)`
                : `Preview only (${fmt(shown)} of ${fmt(total)} rows)`,
            tooltip: `${total === null
                ? `Preview only (${fmt(shown)} rows shown — total unknown)`
                : `Preview only (${fmt(shown)} of ${fmt(total)} rows)`} — full export unavailable. Re-run the query to export it.`,
            ...footerFor(completeness, shown, total),
        };
    }

    const localAllowed = !strictExport;
    if (completeness === "unknown") {
        // Pre-Feature-2 replayed card: the shown rows may themselves be a
        // truncated preview, so the label must never read as complete (f2-CX-03).
        return {
            completeness,
            canExportFull,
            localAllowed,
            disabled: !localAllowed,
            buttonLabel: `Download shown rows (${fmt(shown)}) — completeness unknown`,
            tooltip: localAllowed
                ? "This card predates completeness tracking, so the shown rows may be a truncated preview. Re-run the query for a guaranteed-complete export."
                : "This card predates completeness tracking and local downloads are disabled on this deployment (strict export). Re-run the query for a guaranteed-complete export.",
            ...footerFor(completeness, shown, total),
        };
    }

    // completeness === "complete": the row SET is whole, but the local file is
    // still the DISPLAY rendering — selected columns, _fmt-clipped values — so
    // it is labeled as the displayed table, never as "complete" data (f2-CX-04).
    return {
        completeness,
        canExportFull,
        localAllowed,
        disabled: !localAllowed,
        buttonLabel: `Download displayed table (CSV, ${fmt(shown)} rows)`,
        tooltip: localAllowed
            ? "Downloads the table as displayed: display columns only, display-formatted values (floats rounded, long strings clipped). Re-run the query if you need full-fidelity values."
            : "Local downloads are disabled on this deployment (strict export) and no server export is available for this result. Re-run the query.",
        ...footerFor(completeness, shown, total),
    };
}

/**
 * Footer disclosure for the card (f2-CX-08): a truncated table must carry the
 * literal "Preview only" wording, and an unknown one must say so.
 * @param {Completeness} completeness
 * @param {number} shown
 * @param {number|null} total
 */
function footerFor(completeness, shown, total) {
    if (completeness === "truncated") {
        return {
            // A truncated card with no recorded total must say so rather than
            // fabricate an "N of N" disclosure (f2-CX-17).
            footerText: total === null
                ? `Preview only — showing ${fmt(shown)} rows (total unknown)`
                : `Preview only — showing ${fmt(shown)} of ${fmt(total)} rows`,
            footerTone: /** @type {const} */ ("warning"),
        };
    }
    if (completeness === "unknown") {
        return {
            footerText: `${fmt(shown)} rows shown — completeness unknown`,
            footerTone: /** @type {const} */ ("warning"),
        };
    }
    return {
        footerText: `${fmt(shown)} row${shown === 1 ? "" : "s"}`,
        footerTone: /** @type {const} */ ("default"),
    };
}

/**
 * Read the export route's verdict on what it actually delivered (f2-CX-05).
 *
 * The route emits `X-Quasar-Truncated: 1` when a configured export ceiling
 * (QUASAR_RESULT_EXPORT_MAX_ROWS) clipped the stream, with the delivered and
 * true row counts in `X-Quasar-Rowcount` / `X-Quasar-Total-Rows`. A clipped
 * file is still partial data: it downloads, but with a persistent warning and
 * a PARTIAL_ filename prefix — never silently as full.
 *
 * @param {{
 *   ok: boolean,
 *   status: number,
 *   detail?: string|null,
 *   truncatedHeader?: string|null,
 *   rowcountHeader?: string|null,
 *   totalRowsHeader?: string|null,
 * }} response
 * @returns {{ kind: "full"|"partial"|"error", warning?: string, errorMessage?: string, filenamePrefix?: string }}
 */
export function interpretExportResponse({ ok, status, detail, truncatedHeader, rowcountHeader, totalRowsHeader }) {
    if (!ok) {
        // Fail loud. The caller must NEVER fall back to the preview rows here —
        // that would hand the user a truncated file they'd believe was complete.
        return {
            kind: "error",
            errorMessage: detail || (status === 410
                ? "This result has expired. Re-run the query to export it."
                : `Full export failed (HTTP ${status}).`),
        };
    }
    const flag = String(truncatedHeader ?? "").trim().toLowerCase();
    if (flag && flag !== "0" && flag !== "false") {
        const delivered = parseCount(rowcountHeader);
        const total = parseCount(totalRowsHeader);
        return {
            kind: "partial",
            filenamePrefix: "PARTIAL_",
            warning: `Server export ceiling applied — downloaded ${delivered === null ? "some" : fmt(delivered)} of ${total === null ? "the total" : fmt(total)} rows (partial export).`,
        };
    }
    return { kind: "full" };
}

/** @param {string|null|undefined} header */
function parseCount(header) {
    const n = Number.parseInt(String(header ?? ""), 10);
    return Number.isFinite(n) && n >= 0 ? n : null;
}

/**
 * Quote one CSV cell — headers AND values go through the same escaping, so a
 * column name containing a comma/quote/newline can no longer shear the header
 * row (f2-CX-06). Always-quoted, matching the old value behavior.
 * @param {unknown} value
 */
export function escapeCsvCell(value) {
    return `"${String(value ?? "").replace(/"/g, '""')}"`;
}

/**
 * The local (displayed-table) CSV text: display columns, display values.
 * @param {string[]} columns
 * @param {Record<string, unknown>[]} rows
 */
export function localCsvText(columns, rows) {
    return [
        columns.map(escapeCsvCell).join(","),
        ...rows.map((row) => columns.map((col) => escapeCsvCell(row[col])).join(",")),
    ].join("\n");
}

/**
 * Preview-derivation facts for the illustrative badge (f2-CX-11).
 *
 * Reads the structured fields a producer may stamp on a plot/image card's
 * `meta` — `preview_derived`/`truncated` plus `shown_rows`(`rowcount`)/
 * `total_rows`. Absent fields mean "not known to be preview-derived", which
 * keeps the base badge (a deliberate always-on product decision) unchanged.
 *
 * @param {Record<string, unknown>|null|undefined} meta
 * @returns {{ previewDerived: boolean, shownRows?: number, totalRows?: number }}
 */
export function previewBadgeFacts(meta) {
    const m = meta && typeof meta === "object" ? /** @type {Record<string, unknown>} */ (meta) : {};
    const previewDerived = m.preview_derived === true || m.previewDerived === true || m.truncated === true;
    const shownRows = typeof m.shown_rows === "number" ? m.shown_rows
        : typeof m.rowcount === "number" ? m.rowcount : undefined;
    const totalRows = typeof m.total_rows === "number" ? m.total_rows
        : typeof m.totalRows === "number" ? m.totalRows : undefined;
    return { previewDerived, shownRows, totalRows };
}
