import test from "node:test";
import assert from "node:assert/strict";

import {
    DARK_AXIS_COLOR,
    DARK_FONT_COLOR,
    DARK_GRID_COLOR,
    DARK_PAPER_BG,
    DARK_PLOT_BG,
    LIGHT_AXIS_COLOR,
    LIGHT_FONT_COLOR,
    LIGHT_GRID_COLOR,
    LIGHT_PAPER_BG,
    LIGHT_PLOT_BG,
    lightExportData,
    lightExportLayout,
    themedData,
    themedLayout,
} from "../src/lib/plotly-theme.js";

// ── WCAG 2.x contrast arithmetic (test-side, independent of the module) ──

/** Parse "#rgb", "#rrggbb", or "rgb(a)(r, g, b[, a])" into [r, g, b, a]. */
function parseColor(color) {
    const value = String(color).trim().toLowerCase();
    const hex = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/);
    if (hex) {
        const ch = hex[1].length === 3
            ? [...hex[1]].map((c) => parseInt(c + c, 16))
            : [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16));
        return [...ch, 1];
    }
    const rgba = value.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)$/);
    if (rgba) return [Number(rgba[1]), Number(rgba[2]), Number(rgba[3]), rgba[4] === undefined ? 1 : Number(rgba[4])];
    throw new Error(`unparseable color: ${color}`);
}

/** Alpha-composite fg over an opaque bg → opaque [r, g, b]. */
function composite(fg, bg) {
    const [fr, fg_, fb, fa] = parseColor(fg);
    const [br, bg_, bb] = parseColor(bg);
    return [fr * fa + br * (1 - fa), fg_ * fa + bg_ * (1 - fa), fb * fa + bb * (1 - fa)];
}

function relativeLuminance([r, g, b]) {
    const lin = [r, g, b].map((ch) => {
        const s = ch / 255;
        return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
}

/** WCAG contrast ratio; translucent fg is composited over the bg first. */
function contrast(fg, bg) {
    const l1 = relativeLuminance(composite(fg, bg));
    const l2 = relativeLuminance(parseColor(bg).slice(0, 3));
    const [hi, lo] = l1 >= l2 ? [l1, l2] : [l2, l1];
    return (hi + 0.05) / (lo + 0.05);
}

const alphaOf = (color) => parseColor(color)[3];
const isOpaque = (color) => alphaOf(color) === 1;

// ── RE-C1 regression sentinel: the reported bug, numerically ────────────
// The old on-screen skin drew slate-300 text over the light theme's
// near-white card (#f1f5f9 via globals.css) — the "very hard to read"
// NOIRLab finding. Keep the arithmetic that motivated the fix.

test("RE-C1 sentinel: the old dark-skin text on the light card was ~1.3:1", () => {
    const broken = contrast("#cbd5e1", "#f1f5f9");
    assert.ok(broken < 2, `expected the bug to be <2:1, got ${broken.toFixed(2)}`);
});

// ── Text contrast against BOTH fills of each skin ───────────────────────

test("light skin: text meets the claimed ~13.4:1 — enforce >= 13.0:1 (CX-08)", () => {
    // The module header claims 13.4:1 on the card / 14.9:1 on white; assert
    // a 13.0 floor so any regression below the claimed class is caught
    // (>= 12 would let a 12:1 palette pass while the claim still said 13.4).
    for (const bg of [LIGHT_PAPER_BG, LIGHT_PLOT_BG]) {
        const ratio = contrast(LIGHT_FONT_COLOR, bg);
        assert.ok(ratio >= 13.0, `${LIGHT_FONT_COLOR} on ${bg} is ${ratio.toFixed(2)}:1 (claimed 13.4:1)`);
    }
});

test("dark skin: axis/tick text meets AA (4.5:1) on paper and plot fills", () => {
    for (const bg of [DARK_PAPER_BG, DARK_PLOT_BG]) {
        const ratio = contrast(DARK_FONT_COLOR, bg);
        assert.ok(ratio >= 4.5, `${DARK_FONT_COLOR} on ${bg} is ${ratio.toFixed(2)}:1`);
    }
});

// ── Grid and axis visibility (the "gridlines invisible" half of C1) ─────

test("grid alpha is >= 25% in both skins (was 12% — invisible)", () => {
    assert.ok(alphaOf(LIGHT_GRID_COLOR) >= 0.25, `light grid alpha ${alphaOf(LIGHT_GRID_COLOR)}`);
    assert.ok(alphaOf(DARK_GRID_COLOR) >= 0.25, `dark grid alpha ${alphaOf(DARK_GRID_COLOR)}`);
});

test("dark skin uses the exact stated alphas: grid 30%, axis 60% (CX-08)", () => {
    // The constants' comments state 30% grid / 60% axis — pin them exactly
    // so a silent alpha drift can't undercut the documented contrast claims.
    assert.equal(alphaOf(DARK_GRID_COLOR), 0.3, `dark grid alpha ${alphaOf(DARK_GRID_COLOR)}`);
    assert.equal(alphaOf(DARK_AXIS_COLOR), 0.6, `dark axis alpha ${alphaOf(DARK_AXIS_COLOR)}`);
});

test("gridlines composite to a visible tint over each plot fill", () => {
    // Gridlines are deliberately subtle (well under the 3:1 UI bar), but must
    // clear the ~1.24:1 the old 12%-alpha grid managed.
    const light = contrast(LIGHT_GRID_COLOR, LIGHT_PLOT_BG);
    const dark = contrast(DARK_GRID_COLOR, DARK_PLOT_BG);
    assert.ok(light >= 1.5, `light grid ${light.toFixed(2)}:1`);
    assert.ok(dark >= 1.5, `dark grid ${dark.toFixed(2)}:1`);
});

test("axis lines / zerolines meet the 3:1 non-text bar over each plot fill", () => {
    const light = contrast(LIGHT_AXIS_COLOR, LIGHT_PLOT_BG);
    const dark = contrast(DARK_AXIS_COLOR, DARK_PLOT_BG);
    assert.ok(light >= 3, `light axis ${light.toFixed(2)}:1`);
    assert.ok(dark >= 3, `dark axis ${dark.toFixed(2)}:1`);
});

// ── Opaque backgrounds matched to the theme's card surface ──────────────

test("both skins use OPAQUE backgrounds (never transparent over an unknown ground)", () => {
    for (const theme of ["light", "dark"]) {
        const layout = themedLayout({}, theme);
        for (const key of ["paper_bgcolor", "plot_bgcolor"]) {
            assert.ok(isOpaque(layout[key]), `${theme} ${key} = ${layout[key]} is not opaque`);
        }
    }
    assert.equal(themedLayout({}, "light").paper_bgcolor, LIGHT_PAPER_BG);
    assert.equal(themedLayout({}, "dark").paper_bgcolor, DARK_PAPER_BG);
});

// ── themedLayout: per-theme skinning of a realistic incoming layout ──────

const INCOMING = {
    title: "Gaia DR3 CMD",
    template: {
        layout: { paper_bgcolor: "#111", font: { color: "#e2e8f0" }, colorway: ["#38bdf8", "#f472b6"] },
        // A dark template's white marker DEFAULT would render white-on-white
        // on the light plot (CX-09 round 2); the symbol is intentional styling.
        data: { scatter: [{ marker: { symbol: "diamond", color: "#ffffff" } }] },
    },
    xaxis: { title: { text: "BP-RP" }, gridcolor: "#000" },
    yaxis2: { title: "G" },
    annotations: [
        { text: "Hα", font: { color: "#94a3b8" } },        // slate-400: washes out on white
        { text: "flux peak", font: { color: "#dc2626" } }, // deliberate red: keep
    ],
    shapes: [
        { line: { color: "#94a3b8" } },
        { line: { color: "#0ea5e9" } },
    ],
    meta: { kind: "period_fold", best_period_d: 1.5 },
};

test("light skin re-inks every axis (incl. subplots), text, and the legend", () => {
    const layout = themedLayout(INCOMING, "light");
    for (const key of ["xaxis", "yaxis", "yaxis2"]) {
        assert.equal(layout[key].gridcolor, LIGHT_GRID_COLOR, key);
        assert.equal(layout[key].tickfont.color, LIGHT_FONT_COLOR, key);
        assert.equal(layout[key].title.font.color, LIGHT_FONT_COLOR, key);
    }
    // CX-04: the string-form axis title {title: "G"} must survive the font
    // merge as {text: "G"}, not collapse to {font: ...} alone.
    assert.equal(layout.yaxis2.title.text, "G");
    assert.equal(layout.xaxis.title.text, "BP-RP");
    assert.equal(layout.font.color, LIGHT_FONT_COLOR);
    assert.equal(layout.title.text, "Gaia DR3 CMD");
    assert.equal(layout.title.font.color, LIGHT_FONT_COLOR);
    assert.equal(layout.legend.font.color, LIGHT_FONT_COLOR);
    assert.equal(layout.hoverlabel.font.color, LIGHT_FONT_COLOR);
    assert.equal(layout.autosize, true);
});

test("light skin keeps the template but overrides its theme keys (CX-09)", () => {
    const layout = themedLayout(INCOMING, "light");
    // Intentional template styling (colorway, marker symbol) survives...
    assert.deepEqual(layout.template.layout.colorway, ["#38bdf8", "#f472b6"]);
    assert.equal(layout.template.data.scatter[0].marker.symbol, "diamond");
    // ...the white marker DEFAULT is re-inked like an explicit trace
    // (round 2: template.data defaults apply to traces omitting the prop)...
    assert.equal(layout.template.data.scatter[0].marker.color, LIGHT_FONT_COLOR);
    // ...while the dark template layout keys are overridden so they cannot
    // resurrect light-on-dark defaults underneath our explicit overrides.
    assert.equal(layout.template.layout.paper_bgcolor, LIGHT_PAPER_BG);
    assert.equal(layout.template.layout.plot_bgcolor, LIGHT_PLOT_BG);
    assert.equal(layout.template.layout.font.color, LIGHT_FONT_COLOR);
    // No template in → no template invented.
    assert.equal(themedLayout({}, "light").template, undefined);
});

test("light skin darkens washed-out annotations/shapes but keeps deliberate colors", () => {
    const layout = themedLayout(INCOMING, "light");
    assert.equal(layout.annotations[0].font.color, LIGHT_FONT_COLOR);
    assert.equal(layout.annotations[1].font.color, "#dc2626");
    assert.equal(layout.shapes[0].line.color, LIGHT_AXIS_COLOR);
    assert.equal(layout.shapes[1].line.color, "#0ea5e9");
});

test("dark skin re-inks axes with the raised-alpha palette and keeps the template", () => {
    const layout = themedLayout(INCOMING, "dark");
    for (const key of ["xaxis", "yaxis", "yaxis2"]) {
        assert.equal(layout[key].gridcolor, DARK_GRID_COLOR, key);
        assert.equal(layout[key].linecolor, DARK_AXIS_COLOR, key);
    }
    // CX-04 parity: the string-form title survives on the dark skin too.
    assert.equal(layout.yaxis2.title.text, "G");
    assert.equal(layout.font.color, DARK_FONT_COLOR);
    // Dark parity with the pre-C1 behavior: incoming template untouched.
    assert.deepEqual(layout.template, INCOMING.template);
    assert.equal(layout.autosize, true);
});

test("dark skin re-inks explicit tickfonts, axis-title fonts, and the legend font (CX-05)", () => {
    // A spec skinned for light (or hand-inked dark) would otherwise keep
    // dark ink on the dark ground — only layout.font was themed before.
    const layout = themedLayout({
        xaxis: {
            tickfont: { color: "#0f172a", size: 10 },
            title: { text: "MJD", font: { color: "#111827" } },
        },
        yaxis2: { tickfont: { color: "#1e293b" } },
        legend: { font: { color: "#0b1220" } },
    }, "dark");
    assert.equal(layout.xaxis.tickfont.color, DARK_FONT_COLOR);
    assert.equal(layout.xaxis.tickfont.size, 10, "non-color font props survive");
    assert.equal(layout.xaxis.title.font.color, DARK_FONT_COLOR);
    assert.equal(layout.xaxis.title.text, "MJD");
    assert.equal(layout.yaxis2.tickfont.color, DARK_FONT_COLOR);
    assert.equal(layout.legend.font.color, DARK_FONT_COLOR);
});

test("themedLayout never mutates the incoming spec layout (theme toggles re-derive)", () => {
    const pristine = JSON.parse(JSON.stringify(INCOMING));
    themedLayout(INCOMING, "light");
    themedLayout(INCOMING, "dark");
    assert.deepEqual(INCOMING, pristine);
});

test("period-fold meta rides through both skins", () => {
    assert.deepEqual(themedLayout(INCOMING, "light").meta, INCOMING.meta);
    assert.deepEqual(themedLayout(INCOMING, "dark").meta, INCOMING.meta);
});

// ── themedData: near-white traces only, and only on light ───────────────

test("light data re-inks near-white lines/markers; colored traces untouched", () => {
    const data = [
        { name: "guide", line: { color: "white" } },
        { name: "stars", marker: { color: "#e2e8f0" } },
        { name: "sample", marker: { color: "#38bdf8" } },
    ];
    const themed = themedData(data, "light");
    assert.equal(themed[0].line.color, LIGHT_FONT_COLOR);
    assert.equal(themed[1].marker.color, LIGHT_FONT_COLOR);
    assert.equal(themed[2], data[2], "colored trace must be the same reference");
});

test("light data re-inks near-white scalar textfont ink (CX-06)", () => {
    const data = [
        { mode: "text", textfont: { color: "#f8fafc", size: 11 } },
        { mode: "text", textfont: { color: "#dc2626" } }, // deliberate red: keep
    ];
    const themed = themedData(data, "light");
    assert.equal(themed[0].textfont.color, LIGHT_FONT_COLOR);
    assert.equal(themed[0].textfont.size, 11, "other textfont props survive");
    assert.equal(themed[1], data[1], "colored text trace must be the same reference");
});

test("light data re-inks only near-white entries of ARRAY colors (CX-06)", () => {
    const data = [
        { marker: { color: ["#ffffff", "#38bdf8", "white"], size: 6 } },
        { line: { color: ["#e2e8f0", "#dc2626"] } },
        { marker: { color: [0.2, 0.8] } }, // numeric colorscale values
    ];
    const themed = themedData(data, "light");
    assert.deepEqual(themed[0].marker.color, [LIGHT_FONT_COLOR, "#38bdf8", LIGHT_FONT_COLOR]);
    assert.equal(themed[0].marker.size, 6, "other marker props survive");
    assert.deepEqual(themed[1].line.color, [LIGHT_FONT_COLOR, "#dc2626"]);
    assert.equal(themed[2], data[2], "numeric color arrays pass through by reference");
});

test("light data re-ink preserves alpha; fully-transparent colors untouched (CX-06 round 2)", () => {
    const data = [
        // Invisible by intent, not unreadable — must NOT become opaque dark.
        { marker: { color: "rgba(255, 255, 255, 0)" } },
        // Translucent near-white: re-ink dark WITH the original alpha.
        { marker: { color: "rgba(255, 255, 255, 0.9)" } },
        { mode: "text", textfont: { color: "rgba(255, 255, 255, 0.5)" } },
    ];
    const themed = themedData(data, "light");
    assert.equal(themed[0], data[0], "transparent trace must be the same reference");
    assert.equal(themed[1].marker.color, "rgba(30, 41, 59, 0.9)");
    assert.equal(themed[2].textfont.color, "rgba(30, 41, 59, 0.5)");
});

test("array-valued colors keep per-entry alpha semantics too (CX-06 round 2)", () => {
    const data = [{
        marker: {
            color: [
                "rgba(255, 255, 255, 0)",   // transparent: untouched
                "rgba(255, 255, 255, 0.9)", // translucent white: dark @ 0.9
                "#ffffff",                  // opaque white: flat dark
                "#dc2626",                  // deliberate red: untouched
            ],
        },
    }];
    const themed = themedData(data, "light");
    assert.deepEqual(themed[0].marker.color, [
        "rgba(255, 255, 255, 0)",
        "rgba(30, 41, 59, 0.9)",
        LIGHT_FONT_COLOR,
        "#dc2626",
    ]);
});

test("dark data passes through by reference (no clone, no remap)", () => {
    const data = [{ line: { color: "white" } }];
    assert.equal(themedData(data, "dark"), data);
});

// ── PNG export skin: unchanged contract, idempotent over the light skin ──

test("export layout stays the white publication skin", () => {
    const layout = lightExportLayout(INCOMING);
    assert.equal(layout.paper_bgcolor, "#ffffff");
    assert.equal(layout.plot_bgcolor, "#ffffff");
    assert.equal(layout.font.color, LIGHT_FONT_COLOR);
    // CX-09: the template survives with white export backgrounds and dark ink.
    assert.deepEqual(layout.template.layout.colorway, ["#38bdf8", "#f472b6"]);
    assert.equal(layout.template.layout.paper_bgcolor, "#ffffff");
    assert.equal(layout.template.layout.plot_bgcolor, "#ffffff");
    assert.equal(layout.template.layout.font.color, LIGHT_FONT_COLOR);
    const ratio = contrast(layout.font.color, layout.paper_bgcolor);
    assert.ok(ratio >= 4.5, `export text ${ratio.toFixed(2)}:1`);
});

test("exporting FROM the light on-screen skin does not double-apply", () => {
    // In light mode the live el.layout is already light-skinned; the export
    // path re-skins it again and must land on the same white publication look.
    const once = lightExportLayout(INCOMING);
    const twice = lightExportLayout(themedLayout(INCOMING, "light"));
    assert.equal(twice.paper_bgcolor, "#ffffff");
    assert.equal(twice.plot_bgcolor, "#ffffff");
    assert.equal(twice.font.color, once.font.color);
    assert.equal(twice.xaxis.gridcolor, once.xaxis.gridcolor);
    assert.equal(twice.annotations[1].font.color, "#dc2626");
});

test("exporting FROM the dark on-screen skin still yields legible white PNGs", () => {
    // The dark on-screen skin carries the incoming dark template untouched;
    // the export must still come out light (CX-09: overridden, not deleted).
    const exported = lightExportLayout(themedLayout(INCOMING, "dark"));
    assert.equal(exported.paper_bgcolor, "#ffffff");
    assert.equal(exported.font.color, LIGHT_FONT_COLOR);
    // The dark template's theme keys must not ride into the export...
    assert.equal(exported.template.layout.paper_bgcolor, "#ffffff");
    assert.equal(exported.template.layout.plot_bgcolor, "#ffffff");
    assert.equal(exported.template.layout.font.color, LIGHT_FONT_COLOR);
    // ...but its colorway and intentional trace defaults do — with the
    // white marker default re-inked dark (CX-09 round 2).
    assert.deepEqual(exported.template.layout.colorway, ["#38bdf8", "#f472b6"]);
    assert.equal(exported.template.data.scatter[0].marker.symbol, "diamond");
    assert.equal(exported.template.data.scatter[0].marker.color, LIGHT_FONT_COLOR);
});

test("export data remap is idempotent over already-remapped traces", () => {
    const data = [{ marker: { color: "#ffffff" } }];
    const once = lightExportData(data);
    const twice = lightExportData(once);
    assert.equal(twice[0].marker.color, LIGHT_FONT_COLOR);
});

test("CX-10: transparent layout elements stay hidden; translucent ones keep alpha", () => {
    const layout = {
        annotations: [
            { text: "hidden", font: { color: "rgba(255,255,255,0)" } },
            { text: "faint", font: { color: "rgba(255, 255, 255, 0.5)" } },
            { text: "solid", font: { color: "#ffffff" } },
        ],
        shapes: [
            { line: { color: "rgba(255,255,255,0)" } },
            { line: { color: "rgba(203, 213, 225, 0.4)" } },
        ],
    };
    const out = themedLayout(layout, "light");
    assert.equal(out.annotations[0].font.color, "rgba(255,255,255,0)"); // deliberate hiding survives
    assert.equal(out.annotations[1].font.color, "rgba(30, 41, 59, 0.5)"); // re-inked, alpha kept
    assert.equal(out.annotations[2].font.color, LIGHT_FONT_COLOR); // opaque → constant
    assert.equal(out.shapes[0].line.color, "rgba(255,255,255,0)");
    assert.equal(out.shapes[1].line.color, "rgba(51, 65, 85, 0.4)"); // min(0.4, 0.9)
});
