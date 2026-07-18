import test from "node:test";
import assert from "node:assert/strict";

import { latestRunningPhase } from "../src/lib/active-phase.js";

const step = (text, status = "running") => ({ text, status });

test("no steps → undefined (generic 'Generating answer' fallback)", () => {
    assert.equal(latestRunningPhase(undefined), undefined);
    assert.equal(latestRunningPhase(null), undefined);
    assert.equal(latestRunningPhase([]), undefined);
});

test("picks the LATEST still-running step", () => {
    const steps = [
        step("Connecting to QUASAR engine", "completed"),
        step("Querying ALMA by target", "running"),
        step("Searching the web for updated information", "running"),
    ];
    assert.equal(
        latestRunningPhase(steps),
        "Searching the web for updated information",
    );
});

test("all steps completed → undefined", () => {
    const steps = [
        step("Querying ALMA by target", "completed"),
        step("Generating response", "completed"),
    ];
    assert.equal(latestRunningPhase(steps), undefined);
});

test("a running step that later completes stops being the phase", () => {
    // store.addThinkingStep flips a step in place when the same text arrives
    // with status "completed" — the phase must fall back to the previous
    // still-running step.
    const steps = [
        step("Querying ALMA by target", "running"),
        step("Searching the web", "completed"),
    ];
    assert.equal(latestRunningPhase(steps), "Querying ALMA by target");
});

test("internal liveness pseudo-steps are never surfaced", () => {
    const steps = [
        step("Querying ALMA by target", "running"),
        step("__tool_heartbeat__search_cadc::Searching CADC", "running"),
        step("__data_ready__{}", "running"),
    ];
    assert.equal(latestRunningPhase(steps), "Querying ALMA by target");
});

test("blank or malformed steps are skipped, not crashed on", () => {
    const steps = [
        step("Querying ALMA by target", "running"),
        { status: "running" },
        null,
        step("   ", "running"),
    ];
    assert.equal(latestRunningPhase(steps), "Querying ALMA by target");
});
