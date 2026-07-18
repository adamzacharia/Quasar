// Which live phase should the answer-buffer spinner name?
//
// "Generating answer" used to show for EVERYTHING that happens after the
// first answer text arrives — including follow-up tool calls that can take
// minutes when an external archive is slow (live 2026-07-18: a CADC search
// ran 477 s behind that generic spinner). The backend already streams
// archive-aware step labels ("Querying ALMA by target (…)", status
// running/completed); this picks the one still running so the spinner can
// tell the user what it is actually waiting on.
//
// Plain JS (not TS) so ui-pro/tests/*.test.mjs can import it directly,
// same pattern as eval-mode.js.

/**
 * Latest still-running thinking step, if any.
 *
 * @param {Array<{text: string, status: string}>|undefined|null} steps
 * @returns {string|undefined} the step label to display, or undefined to
 *   fall back to the generic "Generating answer".
 */
export function latestRunningPhase(steps) {
    if (!Array.isArray(steps)) return undefined;
    for (let i = steps.length - 1; i >= 0; i--) {
        const step = steps[i];
        if (!step || step.status !== "running") continue;
        const text = typeof step.text === "string" ? step.text.trim() : "";
        // Internal liveness/data events are not user-facing phases.
        if (!text || text.startsWith("__")) continue;
        return text;
    }
    return undefined;
}
