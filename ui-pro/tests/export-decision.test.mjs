import test from "node:test";
import assert from "node:assert/strict";

import {
    completenessOf,
    escapeCsvCell,
    exportPlan,
    interpretExportResponse,
    localCsvText,
    previewBadgeFacts,
} from "../src/lib/export-decision.js";

// ── Tri-state completeness (CX-03) ───────────────────────────────

test("truncated flags map to a tri-state, absence is its own state", () => {
    assert.equal(completenessOf(false), "complete");
    assert.equal(completenessOf(true), "truncated");
    assert.equal(completenessOf(undefined), "unknown");
    assert.equal(completenessOf(null), "unknown");
});

test("truncated card WITH an export id: server export, Preview only footer (CX-08)", () => {
    const plan = exportPlan({
        totalRows: 250000, truncated: true, resultId: "dlr_abc",
        shownRowCount: 10000,
    });
    assert.equal(plan.completeness, "truncated");
    assert.equal(plan.canExportFull, true);
    assert.equal(plan.disabled, false);
    assert.equal(plan.buttonLabel, "Export full CSV (250,000 rows)");
    // The literal disclosure the plan requires, with both counts kept.
    assert.equal(plan.footerText, "Preview only — showing 10,000 of 250,000 rows");
    assert.equal(plan.footerTone, "warning");
});

test("truncated card WITHOUT an export id: disabled, never downloadable", () => {
    const plan = exportPlan({
        totalRows: 50000, truncated: true, resultId: undefined,
        shownRowCount: 10000,
    });
    assert.equal(plan.disabled, true);
    assert.equal(plan.localAllowed, false);
    assert.match(plan.buttonLabel, /^Preview only \(10,000 of 50,000 rows\)$/);
    assert.match(plan.tooltip, /Re-run the query/);
    assert.equal(plan.footerTone, "warning");
});

test("complete card WITHOUT an id: local download labeled as the DISPLAYED table (CX-04)", () => {
    const plan = exportPlan({
        totalRows: 42, truncated: false, resultId: undefined,
        shownRowCount: 42,
    });
    assert.equal(plan.completeness, "complete");
    assert.equal(plan.localAllowed, true);
    assert.equal(plan.disabled, false);
    assert.equal(plan.buttonLabel, "Download displayed table (CSV, 42 rows)");
    // The word "complete" must never describe the _fmt-clipped fallback file.
    assert.doesNotMatch(plan.buttonLabel.toLowerCase(), /complete/);
    assert.match(plan.tooltip, /display columns only/i);
    assert.match(plan.tooltip, /display-formatted/i);
    assert.equal(plan.footerText, "42 rows");
    assert.equal(plan.footerTone, "default");
});

test("complete card WITH an id: server download is the primary path", () => {
    const plan = exportPlan({
        totalRows: 42, truncated: false, resultId: "dlr_x",
        shownRowCount: 42,
    });
    assert.equal(plan.buttonLabel, "Download CSV (42 rows)");
    assert.match(plan.tooltip, /every row and every column/);
    assert.equal(plan.footerTone, "default");
});

test("legacy replayed card (no truncated, no totalRows) is UNKNOWN, never complete (CX-03)", () => {
    // Persisted pre-Feature-2 dataTable records rehydrate with neither field;
    // the old `?? false` default let them export a truncated preview silently.
    const plan = exportPlan({
        totalRows: undefined, truncated: undefined, resultId: undefined,
        shownRowCount: 10000,
    });
    assert.equal(plan.completeness, "unknown");
    assert.equal(plan.localAllowed, true);
    assert.equal(plan.disabled, false);
    assert.equal(plan.buttonLabel, "Download shown rows (10,000) — completeness unknown");
    // The old label for this state was "Download CSV (complete, N rows)".
    assert.doesNotMatch(plan.buttonLabel, /Download CSV \(complete/);
    assert.match(plan.tooltip, /predates completeness tracking/);
    assert.match(plan.tooltip, /Re-run the query/);
    assert.equal(plan.footerText, "10,000 rows shown — completeness unknown");
    assert.equal(plan.footerTone, "warning");
});

test("unknown legacy card must not fabricate a total from the shown rows", () => {
    const plan = exportPlan({ truncated: undefined, shownRowCount: 500 });
    assert.doesNotMatch(plan.buttonLabel, /500 of 500/);
    assert.doesNotMatch(plan.footerText, /of 500/);
});

// ── STRICT_EXPORT variants ───────────────────────────────────────

test("strict export disables the local path for a complete card without an id", () => {
    const plan = exportPlan({
        totalRows: 42, truncated: false, resultId: undefined,
        shownRowCount: 42, strictExport: true,
    });
    assert.equal(plan.localAllowed, false);
    assert.equal(plan.disabled, true);
    assert.match(plan.tooltip, /strict export/);
});

test("strict export disables the local path for an UNKNOWN card too (CX-03)", () => {
    const plan = exportPlan({
        truncated: undefined, resultId: undefined,
        shownRowCount: 10000, strictExport: true,
    });
    assert.equal(plan.localAllowed, false);
    assert.equal(plan.disabled, true);
    assert.match(plan.tooltip, /strict export/);
    assert.match(plan.tooltip, /predates completeness tracking/);
});

test("strict export does NOT touch the server path", () => {
    const plan = exportPlan({
        totalRows: 42, truncated: false, resultId: "dlr_x",
        shownRowCount: 42, strictExport: true,
    });
    assert.equal(plan.disabled, false);
    assert.equal(plan.buttonLabel, "Download CSV (42 rows)");
});

// ── Export response interpretation (CX-05, expiry) ───────────────

test("410 is a loud error with the expiry message — never a fallback file", () => {
    const verdict = interpretExportResponse({ ok: false, status: 410, detail: "" });
    assert.equal(verdict.kind, "error");
    assert.match(verdict.errorMessage, /expired/);
    // Nothing downloadable comes back from an error verdict.
    assert.equal(verdict.filenamePrefix, undefined);
    assert.equal(verdict.warning, undefined);
});

test("410 with a server detail keeps the server's wording", () => {
    const verdict = interpretExportResponse({
        ok: false, status: 410,
        detail: "This result is no longer available — results are kept for a limited time and this one has expired. Re-run the query to export it.",
    });
    assert.equal(verdict.kind, "error");
    assert.match(verdict.errorMessage, /no longer available/);
});

test("500 is a loud error naming the status", () => {
    const verdict = interpretExportResponse({ ok: false, status: 500, detail: null });
    assert.equal(verdict.kind, "error");
    assert.match(verdict.errorMessage, /HTTP 500/);
});

test("X-Quasar-Truncated → partial: warning with counts and a PARTIAL_ prefix (CX-05)", () => {
    const verdict = interpretExportResponse({
        ok: true, status: 200,
        truncatedHeader: "1", rowcountHeader: "100000", totalRowsHeader: "250000",
    });
    assert.equal(verdict.kind, "partial");
    assert.equal(verdict.filenamePrefix, "PARTIAL_");
    assert.equal(
        verdict.warning,
        "Server export ceiling applied — downloaded 100,000 of 250,000 rows (partial export).",
    );
});

test("truncated header set but counts unreadable still warns", () => {
    const verdict = interpretExportResponse({
        ok: true, status: 200, truncatedHeader: "1",
        rowcountHeader: null, totalRowsHeader: "not-a-number",
    });
    assert.equal(verdict.kind, "partial");
    assert.equal(verdict.filenamePrefix, "PARTIAL_");
    assert.match(verdict.warning, /partial export/);
});

test("no truncation header → full, no warning, no prefix", () => {
    const verdict = interpretExportResponse({
        ok: true, status: 200,
        truncatedHeader: null, rowcountHeader: "42", totalRowsHeader: "42",
    });
    assert.deepEqual(verdict, { kind: "full" });
});

test("explicit 0/false truncation headers count as full", () => {
    assert.equal(interpretExportResponse({ ok: true, status: 200, truncatedHeader: "0" }).kind, "full");
    assert.equal(interpretExportResponse({ ok: true, status: 200, truncatedHeader: "false" }).kind, "full");
});

// ── CSV escaping (CX-06) ─────────────────────────────────────────

test("escapeCsvCell quotes commas, quotes, and newlines", () => {
    assert.equal(escapeCsvCell("plain"), '"plain"');
    assert.equal(escapeCsvCell('say "hi", ok?'), '"say ""hi"", ok?"');
    assert.equal(escapeCsvCell("two\nlines"), '"two\nlines"');
    assert.equal(escapeCsvCell(null), '""');
    assert.equal(escapeCsvCell(3.14), '"3.14"');
});

test("localCsvText escapes HEADERS with the same rule as values (CX-06)", () => {
    const csv = localCsvText(
        ['Flux ["Jy"]', "RA, Dec"],
        [{ 'Flux ["Jy"]': 1.5, "RA, Dec": "10.5, -30.2" }],
    );
    const [header, row] = csv.split("\n");
    assert.equal(header, '"Flux [""Jy""]","RA, Dec"');
    assert.equal(row, '"1.5","10.5, -30.2"');
});

test("localCsvText with zero rows still emits the header row", () => {
    assert.equal(localCsvText(["a", "b"], []), '"a","b"');
});

// ── Illustrative badge escalation (CX-11) ────────────────────────

test("no meta / plain meta: base badge only, never preview-derived", () => {
    assert.deepEqual(previewBadgeFacts(undefined),
        { previewDerived: false, shownRows: undefined, totalRows: undefined });
    assert.deepEqual(previewBadgeFacts({ line_explorer_url: "/x" }).previewDerived, false);
});

test("stamped meta escalates the badge with its counts", () => {
    const facts = previewBadgeFacts({ preview_derived: true, shown_rows: 10000, total_rows: 250000 });
    assert.deepEqual(facts, { previewDerived: true, shownRows: 10000, totalRows: 250000 });
    // truncated:true and camelCase spellings count too.
    assert.equal(previewBadgeFacts({ truncated: true }).previewDerived, true);
    assert.equal(previewBadgeFacts({ previewDerived: true }).previewDerived, true);
});

test("non-boolean truthy stamps do not escalate — the flag must be literal true", () => {
    assert.equal(previewBadgeFacts({ preview_derived: "yes" }).previewDerived, false);
    assert.equal(previewBadgeFacts({ truncated: 1 }).previewDerived, false);
});

// ── f2-CX-17: truncated with UNKNOWN total must never fabricate "N of N" ─────

test("truncated card with NO recorded total and an id: no fabricated N of N (CX-17)", () => {
    const plan = exportPlan({ truncated: true, resultId: "dlr_abc", shownRowCount: 10 });
    assert.equal(plan.completeness, "truncated");
    assert.equal(plan.buttonLabel, "Export full CSV");
    assert.equal(plan.footerText, "Preview only — showing 10 rows (total unknown)");
    assert.equal(plan.footerTone, "warning");
    assert.ok(!plan.footerText.includes("10 of 10"), "must not fabricate N of N");
});

test("truncated card with NO recorded total and NO id: disclosure says total unknown (CX-17)", () => {
    const plan = exportPlan({ truncated: true, resultId: undefined, shownRowCount: 10 });
    assert.equal(plan.disabled, true);
    assert.equal(plan.buttonLabel, "Preview only (10 rows shown — total unknown)");
    assert.ok(plan.tooltip.includes("total unknown"));
    assert.ok(!plan.buttonLabel.includes("10 of 10"), "must not fabricate N of N");
});

// ── f2-CX-21: upstream-truncated results never read as complete ──────────────

test("upstream-partial complete card WITH id: 'Export stored rows', amber footer (CX-21)", () => {
    const plan = exportPlan({
        totalRows: 1000, truncated: false, resultId: "dlr_abc",
        shownRowCount: 1000, upstreamPartial: true, upstreamTotal: 3000,
    });
    assert.equal(plan.buttonLabel, "Export stored rows (1,000 rows)");
    assert.ok(plan.tooltip.includes("upstream service truncated"));
    assert.ok(plan.footerText.includes("upstream-truncated result (kept 1,000 of 3,000)"));
    assert.equal(plan.footerTone, "warning");
});

test("upstream-partial without a known remote total still warns (CX-21)", () => {
    const plan = exportPlan({
        totalRows: 1000, truncated: false, resultId: "dlr_abc",
        shownRowCount: 1000, upstreamPartial: true,
    });
    assert.ok(plan.footerText.includes("upstream-truncated result"));
    assert.equal(plan.footerTone, "warning");
});

test("upstream-partial + display-truncated keeps the Preview only disclosure (CX-21)", () => {
    const plan = exportPlan({
        totalRows: 20000, truncated: true, resultId: "dlr_abc",
        shownRowCount: 10000, upstreamPartial: true, upstreamTotal: 50000,
    });
    assert.ok(plan.footerText.startsWith("Preview only — showing 10,000 of 20,000 rows"));
    assert.ok(plan.footerText.includes("upstream-truncated result"));
    assert.equal(plan.footerTone, "warning");
});

test("no upstream flag: plans are byte-identical to before (CX-21 no-regression)", () => {
    const a = exportPlan({ totalRows: 42, truncated: false, resultId: "dlr_x", shownRowCount: 42 });
    assert.equal(a.buttonLabel, "Download CSV (42 rows)");
    assert.equal(a.footerTone, "default");
});
