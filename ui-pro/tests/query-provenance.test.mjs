import test from "node:test";
import assert from "node:assert/strict";

import { copyPayload, requestsFrom } from "../src/lib/provenance.js";

/* Feature 1 / CX-24 — the request-selection contract QueryProvenance renders.
 * These drive the same narrowing the SSE consumer and the reload path rely on,
 * including a persisted rich-meta round-trip shaped like sse.py's output. */

const adqlReq = { kind: "adql", text: "SELECT ra, dec FROM gaia_dr3.gaia_source" };
const httpReq = { kind: "http", url: "https://x/sia", text: "https://x/sia?ra=1&dec=2" };

test("card-level request wins outright over the trace", () => {
    const trace = [{ name: "other_tool", request: httpReq }];
    const entries = requestsFrom(adqlReq, trace, "datalab_cone_search");
    assert.equal(entries.length, 1);
    assert.equal(entries[0].name, "datalab_cone_search");
    assert.equal(entries[0].request, adqlReq);
});

test("trace fallback narrows to the named tool", () => {
    const trace = [
        { name: "datalab_sql_query", request: adqlReq },
        { name: "hips_cutout", request: httpReq },
    ];
    const entries = requestsFrom(undefined, trace, "hips_cutout");
    assert.equal(entries.length, 1);
    assert.equal(entries[0].name, "hips_cutout");
    assert.equal(entries[0].request.kind, "http");
});

test("without a toolName the whole trace renders in order", () => {
    const trace = [
        { name: "a", request: adqlReq },
        { name: "b", request: httpReq },
    ];
    const entries = requestsFrom(undefined, trace, undefined);
    assert.deepEqual(entries.map((e) => e.name), ["a", "b"]);
});

test("entries without request.text are dropped (text IS the copy payload)", () => {
    const trace = [
        { name: "no_request" },
        { name: "empty_text", request: { kind: "args", text: "" } },
        { name: "good", request: adqlReq },
    ];
    const entries = requestsFrom(undefined, trace, undefined);
    assert.deepEqual(entries.map((e) => e.name), ["good"]);
});

test("card request without text falls through to the trace", () => {
    const trace = [{ name: "good", request: httpReq }];
    const entries = requestsFrom({ kind: "adql", text: "" }, trace, undefined);
    assert.equal(entries.length, 1);
    assert.equal(entries[0].name, "good");
});

test("empty inputs render nothing rather than throwing", () => {
    assert.deepEqual(requestsFrom(undefined, undefined, undefined), []);
    assert.deepEqual(requestsFrom(undefined, [], "x"), []);
    assert.deepEqual(requestsFrom(null, [null, {}], null), []);
});

test("persisted rich-meta round-trip: the slim reload trace still renders", () => {
    // Shape mirrors core/provenance.py::persistable_trace output persisted in
    // messages.metadata.toolTrace — name/ok/request(+sql), NO arguments/output.
    const persisted = [
        {
            name: "datalab_cone_search",
            ok: true,
            rowcount: 1234,
            sql: "SELECT ...",
            request: { kind: "adql", text: "SELECT ra, dec FROM nsc_dr2.object" },
        },
        { name: "helper_without_request", ok: true },
    ];
    const entries = requestsFrom(undefined, persisted, undefined);
    assert.equal(entries.length, 1);
    assert.equal(copyPayload(entries[0]), "SELECT ra, dec FROM nsc_dr2.object");
});

test("copyPayload is the literal request text and never undefined", () => {
    assert.equal(copyPayload({ request: httpReq }), "https://x/sia?ra=1&dec=2");
    assert.equal(copyPayload({}), "");
    assert.equal(copyPayload(undefined), "");
});
