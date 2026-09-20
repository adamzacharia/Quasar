/**
 * Theme-aware Plotly re-skins (RE-C1).
 *
 * A NOIRLab evaluator found every plot "very hard to read — too little
 * contrast": the on-screen card hardcoded the dark glass skin (slate-300
 * text, 12%-alpha grid, transparent background) while the app defaults to
 * the LIGHT theme, whose CSS remaps the card surface to near-white #f1f5f9 —
 * slate-300 on near-white is ~1.3:1 (AA needs 4.5:1) and the grid vanished.
 *
 * These pure functions pick the skin for the ACTIVE theme (data-theme on
 * <html>, mirrored by lib/theme-store.ts) and keep the standalone-PNG export
 * skin. Kept as plain JS beside export-decision.js so
 * `node --test tests/*.test.mjs` can assert the contrast arithmetic without
 * a React harness.
 */

/** @typedef {"dark"|"light"} PlotTheme */

// ── Dark skin (data-theme="dark") ─────────────────────────────────
// Opaque slate-900 ground (never transparent over an unknown backdrop).
// #cbd5e1 on #0f172a ≈ 12.0:1.
export const DARK_FONT_COLOR = "#cbd5e1"; // slate-300
export const DARK_GRID_COLOR = "rgba(148, 163, 184, 0.3)"; // slate-400 @ 30% (was 12% — invisible)
export const DARK_AXIS_COLOR = "rgba(148, 163, 184, 0.6)"; // axis lines/ticks/zeroline (≥3:1 composited)
export const DARK_PAPER_BG = "#0f172a"; // slate-900, the dark card surface family
export const DARK_PLOT_BG = "#0f172a";

// ── Light skin (data-theme="light" — the app DEFAULT) ─────────────
// globals.css remaps the card's bg-slate-900/60 to #f1f5f9 in light mode;
// the paper matches that surface and the plot area is white.
// #1e293b on #f1f5f9 ≈ 13.4:1; on #ffffff ≈ 14.9:1.
export const LIGHT_FONT_COLOR = "#1e293b"; // slate-800
export const LIGHT_GRID_COLOR = "rgba(100, 116, 139, 0.35)"; // slate-500 @ 35%
export const LIGHT_AXIS_COLOR = "rgba(51, 65, 85, 0.9)"; // slate-700
export const LIGHT_PAPER_BG = "#f1f5f9"; // matches the light card surface (globals.css:474)
export const LIGHT_PLOT_BG = "#ffffff";

// ── Standalone-PNG export (theme-independent: pasted into white docs) ──
export const EXPORT_PAPER_BG = "#ffffff";

/**
 * @param {unknown} value
 * @returns {Record<string, unknown>}
 */
export function asRecord(value) {
    return value && typeof value === "object" && !Array.isArray(value)
        ? /** @type {Record<string, unknown>} */ (value)
        : {};
}

/**
 * @param {unknown} font
 * @param {string} color
 * @returns {Record<string, unknown>}
 */
function withFontColor(font, color) {
    return { ...asRecord(font), color };
}

/** Plotly accepts the string shorthand `{title: "G"}`; normalize it to
 *  `{text: "G"}` so merging a font into the title can never erase the
 *  string (CX-04 — `asRecord("G")` is `{}`).
 *  @param {unknown} title
 *  @returns {Record<string, unknown>} */
function asTitleRecord(title) {
    return typeof title === "string" ? { text: title } : asRecord(title);
}

/** Parse "white", "#rgb", "#rrggbb", or "rgb(a)(r, g, b[, a])" into
 *  [r, g, b, a] (alpha defaults to 1). Alpha is kept so re-inking can
 *  preserve deliberate translucency (CX-06 round 2).
 *  @param {unknown} color
 *  @returns {number[] | null}
 */
function rgbaChannels(color) {
    if (typeof color !== "string") return null;
    const value = color.trim().toLowerCase();
    if (value === "white") return [255, 255, 255, 1];
    const hex = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/);
    if (hex) {
        const channels = hex[1].length === 3
            ? [...hex[1]].map((c) => parseInt(c + c, 16))
            : [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16));
        return [...channels, 1];
    }
    const rgb = value.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)$/);
    if (rgb) {
        return [
            Number(rgb[1]), Number(rgb[2]), Number(rgb[3]),
            rgb[4] === undefined ? 1 : Number(rgb[4]),
        ];
    }
    return null;
}

/**
 * @param {unknown} color
 * @returns {number[] | null}
 */
function rgbChannels(color) {
    const rgba = rgbaChannels(color);
    return rgba === null ? null : rgba.slice(0, 3);
}

/** True for colors that are illegible on a white background (white and very
 *  light grays, i.e. every RGB channel >= 200).
 *  @param {unknown} color
 *  @returns {boolean} */
export function isNearWhite(color) {
    const channels = rgbChannels(color);
    return channels !== null && channels.every((ch) => ch >= 200);
}

/** Wider net for text and guide lines: the dark-theme grays (slate-400 line
 *  labels, slate marker shapes) sit well under the near-white bar yet are
 *  still ~2:1 contrast on white. Trace palettes stay on the stricter check
 *  so deliberately colored data is never touched.
 *  @param {unknown} color
 *  @returns {boolean} */
export function isLowContrastOnWhite(color) {
    const channels = rgbChannels(color);
    return channels !== null && channels.every((ch) => ch >= 140);
}

/** Re-ink a low-contrast LAYOUT color (annotation font, guide-shape stroke)
 *  to `targetColor`, preserving the original alpha: a fully transparent
 *  entry is deliberate HIDING, not wash-out, and stays untouched — the old
 *  opaque replacement was un-hiding rgba(255,255,255,0) annotations/shapes
 *  (guard CX-10). Returns null when the color should be left alone.
 *  @param {unknown} color @param {string} targetColor
 *  @returns {string | null} */
function reinkLayoutColor(color, targetColor) {
    if (!isLowContrastOnWhite(color)) return null;
    const src = rgbaChannels(color);
    const srcAlpha = src ? src[3] : 1;
    if (srcAlpha === 0) return null;
    if (srcAlpha >= 1) return targetColor;
    const tgt = rgbaChannels(targetColor);
    if (!tgt) return targetColor;
    return `rgba(${tgt[0]}, ${tgt[1]}, ${tgt[2]}, ${Math.min(srcAlpha, tgt[3])})`;
}

/** Deep-clone a layout (they arrive as SSE JSON, so always JSON-cloneable).
 *  Prevents Plotly's in-place mutation of one skin from leaking into the
 *  next re-skin when the theme toggles.
 *  @param {Record<string, unknown> | undefined} layout
 *  @returns {Record<string, unknown>} */
function cloneLayout(layout) {
    return /** @type {Record<string, unknown>} */ (JSON.parse(JSON.stringify(layout ?? {})));
}

/** xaxis/yaxis plus subplot axes (xaxis2, yaxis3, ...), always including the
 *  two primaries even when the incoming layout omits them.
 *  @param {Record<string, unknown>} base
 *  @returns {string[]} */
function axisKeysOf(base) {
    const keys = Object.keys(base).filter((key) => /^[xy]axis\d*$/.test(key));
    for (const required of ["xaxis", "yaxis"]) {
        if (!keys.includes(required)) keys.push(required);
    }
    return keys;
}

/** Dark on-screen skin: the glassmorphism look, now on an opaque slate-900
 *  ground with a visible (30%-alpha) grid.
 *  @param {Record<string, unknown> | undefined} layout
 *  @returns {Record<string, unknown>} */
function darkOnScreenLayout(layout) {
    const base = cloneLayout(layout);

    for (const key of axisKeysOf(base)) {
        const axis = asRecord(base[key]);
        const axisTitle = asTitleRecord(axis.title);
        base[key] = {
            ...axis,
            gridcolor: DARK_GRID_COLOR,
            zerolinecolor: DARK_AXIS_COLOR,
            linecolor: DARK_AXIS_COLOR,
            tickcolor: DARK_AXIS_COLOR,
            // CX-05: explicit per-axis fonts would otherwise keep light-theme
            // (dark-ink) colors on the dark ground — re-ink them like the
            // light skin does, not just the global layout.font.
            tickfont: withFontColor(axis.tickfont, DARK_FONT_COLOR),
            title: { ...axisTitle, font: withFontColor(axisTitle.font, DARK_FONT_COLOR) },
        };
    }

    return {
        ...base,
        autosize: true,
        paper_bgcolor: DARK_PAPER_BG,
        plot_bgcolor: DARK_PLOT_BG,
        font: { ...asRecord(base.font), color: DARK_FONT_COLOR },
        legend: {
            ...asRecord(base.legend),
            bgcolor: "rgba(15, 23, 42, 0.7)",
            bordercolor: DARK_GRID_COLOR,
            // CX-05: an explicit legend font color must follow the theme too.
            font: withFontColor(asRecord(base.legend).font, DARK_FONT_COLOR),
        },
        hoverlabel: {
            ...asRecord(base.hoverlabel),
            bgcolor: "#1e293b",
            bordercolor: DARK_AXIS_COLOR,
            font: { color: "#e2e8f0" },
        },
        margin: base.margin ?? { t: 48, r: 24, b: 48, l: 60 },
    };
}

/** Light overrides for a template's `layout` block (CX-09): only the
 *  theme-relevant keys — backgrounds, fonts, grid/axis colors — are
 *  replaced; everything else (colorway, annotationdefaults, ...) rides
 *  through so intentional template styling survives the re-skin.
 *  @param {Record<string, unknown>} templateLayout
 *  @param {{ paper: string, plot: string }} bg
 *  @returns {Record<string, unknown>} */
function lightTemplateLayout(templateLayout, bg) {
    /** @type {Record<string, unknown>} */
    const out = {
        ...templateLayout,
        paper_bgcolor: bg.paper,
        plot_bgcolor: bg.plot,
        font: withFontColor(templateLayout.font, LIGHT_FONT_COLOR),
    };
    // Template axis defaults apply even to subplot axes the layout omits,
    // so their colors must be re-inked here, not just in axisKeysOf(base).
    for (const key of Object.keys(templateLayout).filter((k) => /^[xy]axis\d*$/.test(k))) {
        const axis = asRecord(templateLayout[key]);
        out[key] = {
            ...axis,
            gridcolor: LIGHT_GRID_COLOR,
            zerolinecolor: LIGHT_AXIS_COLOR,
            linecolor: LIGHT_AXIS_COLOR,
            tickcolor: LIGHT_AXIS_COLOR,
            tickfont: withFontColor(axis.tickfont, LIGHT_FONT_COLOR),
        };
    }
    return out;
}

/** Shared light re-skin: dark slate text, clearly visible grid and axes,
 *  low-contrast annotations/guide-shapes re-inked. The paper/plot fills
 *  differ between the on-screen card (#f1f5f9 surface) and the standalone
 *  PNG export (pure white).
 *  @param {Record<string, unknown> | undefined} layout
 *  @param {{ paper: string, plot: string }} bg
 *  @returns {Record<string, unknown>} */
function lightSkinLayout(layout, bg) {
    const base = cloneLayout(layout);
    // CX-09: keep the incoming template — its colorways and trace/marker
    // defaults are intentional styling — but override the theme-relevant
    // template.layout keys (backgrounds, fonts, grid/axis colors) so a dark
    // template cannot resurrect light-on-dark defaults under our overrides.
    if (base.template !== undefined) {
        const template = asRecord(base.template);
        /** @type {Record<string, unknown>} */
        const next = {
            ...template,
            layout: lightTemplateLayout(asRecord(template.layout), bg),
        };
        // CX-09 round 2: template.data trace DEFAULTS (e.g. a dark
        // template's white marker/text ink) apply to traces that omit those
        // props, so they must go through the same near-white re-ink as
        // explicit traces — symbols, sizes, and colored defaults survive.
        if (template.data !== undefined) {
            const data = asRecord(template.data);
            next.data = Object.fromEntries(
                Object.entries(data).map(([traceType, defaults]) => [
                    traceType,
                    Array.isArray(defaults) ? lightExportData(defaults) : defaults,
                ]),
            );
        }
        base.template = next;
    }

    for (const key of axisKeysOf(base)) {
        const axis = asRecord(base[key]);
        const axisTitle = asTitleRecord(axis.title);
        base[key] = {
            ...axis,
            gridcolor: LIGHT_GRID_COLOR,
            zerolinecolor: LIGHT_AXIS_COLOR,
            linecolor: LIGHT_AXIS_COLOR,
            tickcolor: LIGHT_AXIS_COLOR,
            tickfont: withFontColor(axis.tickfont, LIGHT_FONT_COLOR),
            title: { ...axisTitle, font: withFontColor(axisTitle.font, LIGHT_FONT_COLOR) },
        };
    }

    // Annotations keep deliberately colored text; anything light enough to
    // wash out on white (incl. the slate-400 spectral-line labels) is darkened
    // (colorless ones inherit the global font, which is dark below).
    if (Array.isArray(base.annotations)) {
        base.annotations = base.annotations.map((entry) => {
            const annotation = asRecord(entry);
            const font = asRecord(annotation.font);
            const reinked = reinkLayoutColor(font.color, LIGHT_FONT_COLOR);
            return reinked !== null
                ? { ...annotation, font: { ...font, color: reinked } }
                : annotation;
        });
    }

    // Guide shapes (spectral-line markers etc.) use slate strokes tuned for
    // the dark theme — re-ink light ones so they survive on a light ground.
    if (Array.isArray(base.shapes)) {
        base.shapes = base.shapes.map((entry) => {
            const shape = asRecord(entry);
            const line = asRecord(shape.line);
            const reinked = reinkLayoutColor(line.color, LIGHT_AXIS_COLOR);
            return reinked !== null
                ? { ...shape, line: { ...line, color: reinked } }
                : shape;
        });
    }

    const title = asTitleRecord(base.title);
    return {
        ...base,
        paper_bgcolor: bg.paper,
        plot_bgcolor: bg.plot,
        font: withFontColor(base.font, LIGHT_FONT_COLOR),
        title: { ...title, font: withFontColor(title.font, LIGHT_FONT_COLOR) },
        legend: {
            ...asRecord(base.legend),
            bgcolor: "rgba(255, 255, 255, 0.9)",
            bordercolor: LIGHT_GRID_COLOR,
            font: withFontColor(asRecord(base.legend).font, LIGHT_FONT_COLOR),
        },
    };
}

/** Light on-screen skin: the light re-skin plus the on-screen-only bits
 *  (autosize, default margins, a light hover label).
 *  @param {Record<string, unknown> | undefined} layout
 *  @returns {Record<string, unknown>} */
function lightOnScreenLayout(layout) {
    const skinned = lightSkinLayout(layout, { paper: LIGHT_PAPER_BG, plot: LIGHT_PLOT_BG });
    return {
        ...skinned,
        autosize: true,
        hoverlabel: {
            ...asRecord(skinned.hoverlabel),
            bgcolor: "#ffffff",
            bordercolor: LIGHT_AXIS_COLOR,
            font: { color: LIGHT_FONT_COLOR },
        },
        margin: skinned.margin ?? { t: 48, r: 24, b: 48, l: 60 },
    };
}

/**
 * The on-screen layout for the ACTIVE theme. Always returns a fresh deep
 * clone with an OPAQUE background matched to the card surface of that theme.
 *
 * @param {Record<string, unknown> | undefined} layout
 * @param {PlotTheme} theme
 * @returns {Record<string, unknown>}
 */
export function themedLayout(layout, theme) {
    return theme === "light" ? lightOnScreenLayout(layout) : darkOnScreenLayout(layout);
}

/** The dark re-ink for one scalar color, alpha-aware (CX-06 round 2):
 *  - fully-transparent colors (alpha == 0) are left untouched — they are
 *    invisible by intent, not unreadable, and re-inking them opaque would
 *    surface deliberately hidden marks;
 *  - near-white colors with alpha > 0 re-ink to the dark slate WITH their
 *    original alpha, so a deliberately faint guide stays faint (just dark);
 *  - everything else (colored, numeric, unparseable) is untouched.
 *  Returns undefined when nothing changes.
 *  @param {unknown} color
 *  @returns {string | undefined} */
function reinkNearWhiteScalar(color) {
    const rgba = rgbaChannels(color);
    if (rgba === null) return undefined;
    const [r, g, b, a] = rgba;
    if (a === 0) return undefined;
    if (!(r >= 200 && g >= 200 && b >= 200)) return undefined;
    // LIGHT_FONT_COLOR (#1e293b) is rgb(30, 41, 59).
    return a >= 1 ? LIGHT_FONT_COLOR : `rgba(30, 41, 59, ${a})`;
}

/** Near-white scalar color → the dark re-ink; ARRAY-valued colors re-ink
 *  only their near-white entries, leaving every other entry (colored
 *  strings, numeric colorscale values, fully-transparent entries)
 *  untouched (CX-06). Returns undefined when nothing changes, so untouched
 *  traces keep their identity.
 *  @param {unknown} color
 *  @returns {unknown} */
function reinkNearWhite(color) {
    if (Array.isArray(color)) {
        let changed = false;
        const mapped = color.map((entry) => {
            const next = reinkNearWhiteScalar(entry);
            if (next === undefined) return entry;
            changed = true;
            return next;
        });
        return changed ? mapped : undefined;
    }
    return reinkNearWhiteScalar(color);
}

/** Re-map only trivially unsafe trace colors (white/near-white lines,
 *  markers, or text ink → dark slate; scalar or array-valued) for a light
 *  ground. Full palette remapping is deliberately out of scope — the light
 *  background + dark text/axes carry the fix.
 *  @param {unknown[]} data
 *  @returns {unknown[]} */
export function lightExportData(data) {
    return data.map((trace) => {
        const record = asRecord(trace);
        const line = asRecord(record.line);
        const marker = asRecord(record.marker);
        const textfont = asRecord(record.textfont);
        const lineColor = reinkNearWhite(line.color);
        const markerColor = reinkNearWhite(marker.color);
        const textColor = reinkNearWhite(textfont.color); // CX-06: text ink
        if (lineColor === undefined && markerColor === undefined && textColor === undefined) {
            return trace;
        }
        /** @type {Record<string, unknown>} */
        const out = { ...record };
        if (lineColor !== undefined) out.line = { ...line, color: lineColor };
        if (markerColor !== undefined) out.marker = { ...marker, color: markerColor };
        if (textColor !== undefined) out.textfont = { ...textfont, color: textColor };
        return out;
    });
}

/**
 * The on-screen traces for the ACTIVE theme: dark keeps the incoming
 * palette untouched; light re-inks near-white lines/markers that would
 * vanish on the white plot area.
 *
 * @param {unknown[]} data
 * @param {PlotTheme} theme
 * @returns {unknown[]}
 */
export function themedData(data, theme) {
    return theme === "light" ? lightExportData(data) : data;
}

/** The standalone-PNG export skin: solid white background, dark slate text,
 *  clearly visible grid and axes — a PNG of either on-screen skin pasted
 *  into a white document stays legible. Idempotent over an already-light
 *  layout, so exporting from a light on-screen figure never double-applies.
 *  @param {Record<string, unknown> | undefined} layout
 *  @returns {Record<string, unknown>} */
export function lightExportLayout(layout) {
    return lightSkinLayout(layout, { paper: EXPORT_PAPER_BG, plot: EXPORT_PAPER_BG });
}
