/**
 * Same-page prompt handoff into the ChatArea composer.
 *
 * The `/?prompt=` URL param only prefills the composer on mount (cross-page
 * navigation via the gallery / SLE chips). This window event lets an in-page
 * card — e.g. the sky-map "Cross-match here" button — prefill the composer
 * WITHOUT navigating: ChatArea listens and calls setInputValue, and ChatInput
 * re-syncs + focuses on the changed value. No auto-send, matching `/?prompt=`
 * so the user reviews (and can edit the catalog/radius) before running the
 * tool. Plain JS so `node --test` can import the pure helpers directly. (T7.3)
 */

export const PREFILL_PROMPT_EVENT = "quasar:prefill-prompt";

/**
 * Dispatch a prompt to prefill the chat composer on the same page (no send).
 * No-ops server-side (no window). Empty/blank prompts are ignored.
 * @param {string} prompt
 */
export function dispatchPrefillPrompt(prompt) {
    if (typeof window === "undefined") return;
    const text = String(prompt || "").trim();
    if (!text) return;
    window.dispatchEvent(new CustomEvent(PREFILL_PROMPT_EVENT, { detail: { prompt: text } }));
}

function formatCoord(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return "";
    // Sky-map positions are already ~4-decimal; trim trailing zeros for a clean prompt.
    return String(Number(num.toFixed(6)));
}

/**
 * Build a natural-language `xmatch_user_list` prompt for one sky source. The
 * args are spelled out so the agent runs the CDS X-Match against SIMBAD rather
 * than guessing; the user can edit the catalog/radius before sending.
 * @param {{ra?: number|string, dec?: number|string, label?: string}} [source]
 * @returns {string} the prompt, or "" when coordinates are invalid.
 */
export function buildCrossMatchPrompt({ ra, dec, label } = {}) {
    const raStr = formatCoord(ra);
    const decStr = formatCoord(dec);
    if (raStr === "" || decStr === "") return "";
    const who = label && String(label).trim() ? ` (${String(label).trim()})` : "";
    return (
        `Cross-match this source${who} against SIMBAD to identify it. ` +
        `Use xmatch_user_list with objects=[{"ra": ${raStr}, "dec": ${decStr}}], ` +
        `catalog="simbad", radius_arcsec=5, and report the best match.`
    );
}
