import test from "node:test";
import assert from "node:assert/strict";

import { createRenderChain } from "../src/lib/render-chain.js";

// ── CX-07 (task-3c99b37-115): serialized Plotly render ordering ──────────
// PlotlyCard cleanup cannot cancel an already-running newPlot; the chain
// must therefore guarantee (1) strict start order — a newer render never
// starts under a still-running old one, so the old one cannot finish late
// and overwrite it — and (2) staleness — a superseded render skips its
// follow-up mutations, or is skipped entirely if it never started.

function deferred() {
    /** @type {(value?: unknown) => void} */
    let resolve = () => {};
    const promise = new Promise((r) => { resolve = r; });
    return { promise, resolve };
}

test("a newer render waits for the running one to settle before starting", async () => {
    const chain = createRenderChain();
    const started = deferred();
    const gate = deferred();
    const order = [];

    const first = chain.enqueue(async () => {
        order.push("old:start");
        started.resolve();
        await gate.promise; // the old newPlot still in flight
        order.push("old:end");
    });
    await started.promise;
    const second = chain.enqueue(() => { order.push("new:start"); });

    // The newer render must NOT have started while the old one is pending.
    await new Promise((r) => setTimeout(r, 10));
    assert.deepEqual(order, ["old:start"]);

    gate.resolve();
    await first;
    await second;
    assert.deepEqual(order, ["old:start", "old:end", "new:start"],
        "the old render settles fully BEFORE the new one starts — it can never overwrite it");
});

test("enqueueing a newer render marks the running one stale at its next check", async () => {
    const chain = createRenderChain();
    const started = deferred();
    const gate = deferred();
    const probes = [];

    const first = chain.enqueue(async (isStale) => {
        probes.push(["old before await", isStale()]);
        started.resolve();
        await gate.promise;
        probes.push(["old after await", isStale()]);
    });
    await started.promise;
    const second = chain.enqueue((isStale) => {
        probes.push(["new", isStale()]);
    });

    gate.resolve();
    await first;
    await second;
    assert.deepEqual(probes, [
        ["old before await", false],
        ["old after await", true], // a newer render exists → skip follow-ups
        ["new", false],
    ]);
});

test("a superseded render that never started is skipped entirely", async () => {
    const chain = createRenderChain();
    const started = deferred();
    const gate = deferred();
    let ranSecond = false;
    let ranThird = false;

    const first = chain.enqueue(async () => {
        started.resolve();
        await gate.promise;
    });
    await started.promise;
    const second = chain.enqueue(() => { ranSecond = true; });
    const third = chain.enqueue(() => { ranThird = true; });

    gate.resolve();
    await first;
    await second;
    await third;
    assert.equal(ranSecond, false, "superseded-before-start render must be skipped");
    assert.equal(ranThird, true, "the LATEST render always runs");
});

test("a throwing render rejects its own promise but never wedges the chain", async () => {
    const chain = createRenderChain();
    const first = chain.enqueue(() => { throw new Error("plotly exploded"); });
    await assert.rejects(first, /plotly exploded/);

    let ran = false;
    await chain.enqueue(() => { ran = true; });
    assert.equal(ran, true, "the next render still runs after a failure");
});
