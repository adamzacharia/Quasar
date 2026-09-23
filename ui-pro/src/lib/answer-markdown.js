/**
 * Render-time hygiene for assistant answer markdown (ChatMessage.tsx).
 *
 * The UI shows the STREAMED tokens; backend post-processing (guards, the
 * answer verifier, prose hygiene) only reaches the persisted copy and the
 * `final_text` event. These pure functions make the streamed view right too:
 *
 *  - `<br>` literals in table cells / prose (UI benchmark 2026-09-22, D02);
 *  - orphan image placeholders `![CMD 1]` left after URL stripping (L15);
 *  - internal snake_case tool identifiers in prose (D06, D07, D09, D12, D18);
 *  - bracket-delimited LaTeX `[ \sigma = ... ]` that the model emits instead
 *    of `$$...$$` (D01, D05) -> real math delimiters for remark-math.
 *
 * Fenced and inline code are never touched: every transform runs on the
 * text with code swapped for placeholders (guard CX-26), and fences follow
 * CommonMark (``` or ~~~ of any length >= 3, closed by the same character
 * with at least the opening length).
 */

const FENCE_OPEN_RE = /^ {0,3}(`{3,}|~{3,})/;
const FENCE_CLOSE_RE = /^ {0,3}(`{3,}|~{3,})\s*$/;
const INLINE_CODE_RE = /(?<!`)(`+)(?!`)(?:.+?)(?<!`)\1(?!`)/g;

/** Known tool-name prefixes -> service label. Mirrors core/prose_hygiene.py. */
const PREFIX_LABELS = [
    ["datalab_", "Data Lab"],
    ["sparcl_", "SPARCL"],
    ["hips_", "HiPS"],
    ["vlass_", "VLASS"],
    ["ztf_", "ZTF"],
    ["ned_", "NED"],
    ["gaia_", "Gaia"],
    ["vo_", "VO"],
    ["web_", "web"],
    ["monitor_", "watchlist"],
    ["svo_", "SVO"],
    ["alma_", "ALMA"],
    ["mast_", "MAST"],
    ["ads_", "ADS"],
];

const SPECIAL_LABELS = {
    query_alma_science_archive: "the ALMA science-archive query",
    search_by_target: "the ALMA target search",
    search_by_position: "the ALMA position search",
    search_by_frequency: "the ALMA frequency search",
    advanced_search: "the ALMA advanced search",
    search_alma_with_keywords: "the ALMA keyword search",
    triage_alma_data_products: "the ALMA data-product triage",
    list_alma_files: "the ALMA file listing",
    match_cross_archive_sources: "the cross-archive source match",
    match_perseus_protostars_alma_jwst: "the Perseus protostar ALMA/JWST match",
    overlay_archive_images: "the archive image overlay",
    find_alma_line_coverage: "the ALMA line-coverage check",
    search_mast: "the MAST search",
    get_mast_products: "the MAST product listing",
    search_papers: "the literature search",
    resolve_target: "the target-name resolution",
    browse_schema: "the schema lookup",
    datalab_sql_query: "the Data Lab SQL query",
    datalab_select_catalog_rows: "the Data Lab row selection",
    datalab_cone_count: "the Data Lab cone count",
    datalab_q3c_crossmatch: "the Data Lab cross-match",
    datalab_density_aggregate: "the Data Lab density aggregate",
    datalab_density_vetting: "the Data Lab density vetting",
    datalab_color_magnitude_diagram: "the colour-magnitude diagram",
    datalab_color_color_diagram: "the colour-colour diagram",
    datalab_sky_density_map: "the sky density map",
    datalab_catalog_scatter: "the catalogue scatter plot",
    datalab_image_cutout: "the Data Lab image cutout",
    datalab_color_image: "the Data Lab colour image",
    datalab_cutout_grid: "the Data Lab cutout grid",
    datalab_star_lightcurve: "the Data Lab light curve",
    datalab_period_fold: "the period fold",
    datalab_sed_plot: "the SED plot",
    datalab_lss_wedge: "the large-scale-structure wedge plot",
    datalab_describe_table: "the Data Lab table description",
    datalab_list_catalogs: "the Data Lab catalogue listing",
    datalab_job_status: "the Data Lab job status check",
    datalab_job_results: "the Data Lab job results",
    datalab_confirm_sky_area: "the sky-area confirmation",
    plot_sky_map: "the sky map",
    hips_cutout: "the HiPS cutout",
    web_search: "the web search",
    cross_match_source: "the multi-archive source cross-match",
    xmatch_user_list: "the CDS cross-match",
};

const ACRONYMS = new Set(["sql", "adql", "tap", "sia", "cmd", "sed", "rgb", "hips", "qa2", "mous", "ads", "mast",
    "ned", "vo", "url", "id", "fits", "pv", "moc", "ztf", "gw", "gcn", "jwst", "hst", "alma", "co", "lss", "hr",
    "svo", "irsa", "eso", "cadc", "vlass", "uv", "casa", "sparcl", "q3c", "healpix"]);

/**
 * Generic tool-identifier shape (the streamed view has no tool registry; the
 * backend `final_text` pass humanises exact registered names). Narrow on
 * purpose (guard CX-27): a TOOL-SERVICE prefix (datalab_, sparcl_, hips_ ...)
 * or a tool VERB with at least two more words (search_by_target,
 * get_mast_products). Column / catalogue names such as image_id, survey_id,
 * velocity_frame, gaia_dr3 or ls_dr9 never match.
 */
const TOOL_IDENT_RE = /(?<![\w.`])(?:functions\.)?((?:(?:datalab|sparcl|hips|vlass|ztf|monitor|svo|alma|mast|ads)_[a-z0-9]+(?:_[a-z0-9]+)*)|(?:(?:search|query|match|triage|list|download|get|plot|browse|resolve|overlay|find|xmatch|cross|render|compute|extract|detect|measure|calculate|generate|identify|inspect|convert|lookup|summarize|evaluate|reproduce|export|create)_[a-z0-9]+_[a-z0-9]+(?:_[a-z0-9]+)*))(\((?:[^()\n]|\([^()\n]*\))*\))?(?![\w])/g;

export function humanizeToolName(name) {
    const n = String(name || "").trim().replace(/^`|`$/g, "");
    if (!n) return "";
    if (SPECIAL_LABELS[n]) return SPECIAL_LABELS[n];
    let label = "";
    let rest = n;
    for (const [prefix, service] of PREFIX_LABELS) {
        if (n.startsWith(prefix)) {
            label = service;
            rest = n.slice(prefix.length);
            break;
        }
    }
    const words = rest.split("_").filter(Boolean).map((w) => (ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase() : w));
    const body = words.join(" ");
    return label ? `the ${label} ${body} step` : `the ${body} step`;
}

function splitInline(text) {
    const out = [];
    let pos = 0;
    for (const m of text.matchAll(INLINE_CODE_RE)) {
        if (m.index > pos) out.push([false, text.slice(pos, m.index)]);
        out.push([true, m[0]]);
        pos = m.index + m[0].length;
    }
    if (pos < text.length) out.push([false, text.slice(pos)]);
    return out;
}

/** [[isCode, segment]] preserving every character (CommonMark fences + inline spans). */
export function splitCode(text) {
    const out = [];
    const src = String(text || "");
    const lines = src.match(/[^\n]*\n|[^\n]+$/g) || [];
    let prose = "";
    let code = "";
    let fence = null;
    for (const line of lines) {
        if (fence === null) {
            const m = FENCE_OPEN_RE.exec(line);
            if (m) {
                if (prose) { out.push(...splitInline(prose)); prose = ""; }
                fence = m[1];
                code = line;
                continue;
            }
            prose += line;
        } else {
            code += line;
            const c = FENCE_CLOSE_RE.exec(line.replace(/\n$/, ""));
            if (c && c[1][0] === fence[0] && c[1].length >= fence.length) {
                out.push([true, code]);
                code = "";
                fence = null;
            }
        }
    }
    if (fence !== null && code) out.push([true, code]);
    if (prose) out.push(...splitInline(prose));
    return out.filter(([, seg]) => seg);
}

/** Run `fn` on the text with every code segment replaced by a placeholder. */
function outsideCode(text, fn) {
    const code = [];
    const masked = splitCode(text).map(([isCode, seg]) => {
        if (!isCode) return seg;
        code.push(seg);
        return `\u0000CODE${code.length - 1}\u0000`;
    }).join("");
    return fn(masked).replace(/\u0000CODE(\d+)\u0000/g, (_, i) => code[Number(i)]);
}

/** `<br>` -> hard line break outside tables, a separator inside a table row. */
export function stripBrTags(text) {
    const t = String(text || "");
    if (!/<br/i.test(t)) return t;
    return t.split("\n").map((line) => (
        line.trimStart().startsWith("|") ? line.replace(/<br\s*\/?>/gi, " · ") : line.replace(/<br\s*\/?>/gi, "  \n")
    )).join("\n");
}

/** Drop `![alt]` tokens that have no `(url)` part. */
export function stripImagePlaceholders(text) {
    return String(text || "").replace(/!\[([^\]\n]*)\](?!\()/g, "");
}

/**
 * `[ \sigma = \frac{...} ]` on its own line (or `\[ ... \]`) -> `$$ ... $$`;
 * `\( ... \)` inline -> `$ ... $`. Only when the bracket body contains a
 * LaTeX command, so citation-like `[1]` or `[Source: ...]` are untouched.
 */
export function normalizeMathDelimiters(text) {
    let t = String(text || "");
    // \[ ... \] and \( ... \)
    t = t.replace(/\\\[([\s\S]*?)\\\]/g, (_, body) => `$$${body}$$`);
    t = t.replace(/\\\(([\s\S]*?)\\\)/g, (_, body) => `$${body}$`);
    // Bare "[ \frac... ]" lines (gpt-oss habit): whole-line bracket with a TeX command inside.
    t = t.replace(/^[ \t]*\[\s*((?:[^\[\]\n]|\[[^\[\]\n]*\])*?\\[a-zA-Z]+(?:[^\[\]\n]|\[[^\[\]\n]*\])*?)\s*\][ \t]*$/gm, (_, body) => `$$${body}$$`);
    return t;
}

/**
 * Replace internal tool identifiers in prose with plain language. Code is
 * left alone. `knownNames` (optional) adds exact names beyond the generic shape.
 */
export function humanizeToolIdentifiers(text, knownNames = []) {
    const known = new Set((knownNames || []).map((n) => String(n)));
    const INLINE_TOOL_RE = /^`\s*(?:functions\.)?([a-z]+_[a-z0-9_]+)\s*(?:\([^()\n]*\))?\s*`$/;
    return splitCode(text).map(([isCode, seg]) => {
        if (isCode) {
            // A code span holding ONLY a tool identifier is a tool name, not code.
            const m = INLINE_TOOL_RE.exec(seg);
            if (m && (known.has(m[1]) || new RegExp(TOOL_IDENT_RE.source).test(m[1]))) return humanizeToolName(m[1]);
            return seg;
        }
        let s = seg.replace(TOOL_IDENT_RE, (m, name) => humanizeToolName(name));
        if (known.size) {
            const alt = [...known].sort((a, b) => b.length - a.length).map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
            if (alt) s = s.replace(new RegExp(`(?<![\\w.])(?:${alt})(?:\\((?:[^()\\n]|\\([^()\\n]*\\))*\\))?(?![\\w])`, "g"), (m) => humanizeToolName(m.replace(/\(.*$/s, "")));
        }
        s = s.replace(/\b(the [^.\n]{2,60}?(?:step|query|search|lookup|diagram|map|plot|cutout|match|check|fold|listing|vetting|aggregate|count|selection|description|image|grid|panel|composite|overlay|triage|download|resolution|curve|wedge))\s+(?:tool|function|call)\b/g, "$1");
        s = s.replace(/\b(?:the|a|an)\s+(the\s+)/gi, "$1").replace(/\bThe the\b/g, "The");
        return s;
    }).join("");
}

/** Full render-time preparation for an assistant answer. Code is never changed. */
export function prepareAnswerMarkdown(text, options = {}) {
    let t = String(text || "");
    if (!t) return t;
    t = outsideCode(t, (prose) => {
        let p = prose.replace(/!\[([^\]]*)\]\([^)]+\)/g, "");   // banned inline images
        p = stripImagePlaceholders(p);
        p = stripBrTags(p);
        return normalizeMathDelimiters(p);
    });
    if (options.humanizeTools !== false) t = humanizeToolIdentifiers(t, options.toolNames || []);
    return t;
}
