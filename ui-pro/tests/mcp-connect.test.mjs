import test from "node:test";
import assert from "node:assert/strict";

import { parseMcpInput, deriveServerName, splitCommand, stripVersion, uniqueServerName } from "../src/lib/mcp-paste.js";
import { MCP_PRESETS, PRESETS_VERIFIED_ON, gitmcpUrl, presetForUrl, iconDomain, apiKeyPageFor } from "../src/lib/mcp-presets.js";
import {
    isOAuthResultMessage, readOAuthReturn, stripOAuthReturn, oauthWaitDecision, nextStep, statusView,
    bearerHeader, OAUTH_MESSAGE_TYPE, buildConnectBody, credentialsStillApply, createOnce, returnBanner, parsePairs,
} from "../src/lib/mcp-connect.js";

const FORM = { advancedOpen: false, transport: "http", headerText: "", envText: "", cmd: "npx", cmdArgs: "",
    apiKey: "", keyHeader: undefined, name: "", nameTyped: false };

// ── paste parser: real snippets from server READMEs ─────────────────────────

test("plain URLs", () => {
    assert.deepEqual(parseMcpInput("  https://mcp.deepwiki.com/mcp  "),
        { kind: "url", url: "https://mcp.deepwiki.com/mcp", name: "Deepwiki", headers: {} });
    const bare = parseMcpInput("mcp.context7.com/mcp");
    assert.equal(bare.kind, "url");
    assert.equal(bare.url, "https://mcp.context7.com/mcp");
    assert.equal(parseMcpInput("").kind, "empty");
    assert.equal(parseMcpInput("hello there").kind, "invalid");
});

test("a whole-input URL is kept exactly, punctuation included (CX-10)", () => {
    assert.equal(parseMcpInput("https://example.com/mcp?sig=abc.").url, "https://example.com/mcp?sig=abc.");
    assert.equal(parseMcpInput("https://example.com/mcp)").url, "https://example.com/mcp)");
    // ...but a URL pulled out of prose drops the sentence's full stop.
    assert.equal(parseMcpInput("Connect to https://mcp.deepwiki.com/mcp.").url, "https://mcp.deepwiki.com/mcp");
});

test("names come from the host, and from the repo for gitmcp", () => {
    assert.equal(deriveServerName("https://www.monocrawl.com/mcp/oauth"), "Monocrawl");
    assert.equal(deriveServerName("https://huggingface.co/mcp"), "Huggingface");
    assert.equal(deriveServerName("https://gitmcp.io/astropy/astropy"), "astropy");
    assert.equal(deriveServerName("https://mcp.example.co.uk/mcp"), "Example"); // CX-03
    assert.equal(deriveServerName("https://tools.other.co.uk/mcp"), "Other");
    assert.equal(deriveServerName("not a url"), "");
});

test("derived names never overwrite a different saved server (CX-03)", () => {
    const servers = [{ name: "docs", url: "https://gitmcp.io/foo/docs" }, { name: "Example", url: "https://a.example.co.uk/mcp" }];
    assert.equal(uniqueServerName("docs", "https://gitmcp.io/bar/docs", servers), "bar-docs");
    assert.equal(uniqueServerName("docs", "https://gitmcp.io/foo/docs", servers), "docs"); // same server: reuse
    assert.equal(uniqueServerName("Example", "https://b.example.co.uk/mcp", servers), "Example 2");
    assert.equal(uniqueServerName("Fresh", "https://fresh.example/mcp", servers), "Fresh");
});

test("Claude Desktop / Cursor JSON: placeholder header values are never sent (CX-12)", () => {
    const r = parseMcpInput(`{
      "mcpServers": {
        "context7": { "url": "https://mcp.context7.com/mcp", "headers": { "CONTEXT7_API_KEY": "YOUR_API_KEY", "X-Client": "quasar" } }
      }
    }`);
    assert.equal(r.kind, "url");
    assert.equal(r.url, "https://mcp.context7.com/mcp");
    assert.equal(r.name, "context7");
    assert.deepEqual(r.headers, { "X-Client": "quasar" });
    assert.equal(r.keyHeader, "CONTEXT7_API_KEY");
    assert.match(r.note, /CONTEXT7_API_KEY header/);
});

test("a URL that is still a template is not offered for connection (CX-12)", () => {
    const r = parseMcpInput(`{"mcpServers": {"x": {"url": "https://\${HOST}/mcp"}}}`);
    assert.equal(r.kind, "invalid");
    assert.match(r.message, /placeholder/);
    assert.equal(parseMcpInput("https://<your-host>/mcp").kind, "invalid");
});

test("VS Code servers block and Windsurf serverUrl", () => {
    const vs = parseMcpInput(`{"servers": {"github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/"}}}`);
    assert.equal(vs.kind, "url");
    assert.equal(vs.name, "github");
    assert.equal(vs.transport, "streamable_http");
    const ws = parseMcpInput(`{"mcpServers": {"deepwiki": {"serverUrl": "https://mcp.deepwiki.com/sse", "type": "sse"}}}`);
    assert.equal(ws.url, "https://mcp.deepwiki.com/sse");
    assert.equal(ws.transport, "http");
});

test("a fragment without outer braces still parses", () => {
    const r = parseMcpInput(`"mcpServers": {"hf": {"url": "https://huggingface.co/mcp"}}`);
    assert.equal(r.kind, "url");
    assert.equal(r.name, "hf");
});

test("mcp-remote bridges are unwrapped to the remote URL; placeholder key asked for", () => {
    const r = parseMcpInput(`{"mcpServers": {"linear": {"command": "npx", "args": ["-y", "mcp-remote", "https://mcp.linear.app/sse"]}}}`);
    assert.equal(r.kind, "url");
    assert.equal(r.url, "https://mcp.linear.app/sse");
    assert.equal(r.name, "linear");
    const h = parseMcpInput(`npx -y mcp-remote@latest https://example.com/mcp --header "Authorization: Bearer \${TOKEN}"`);
    assert.equal(h.kind, "url");
    assert.deepEqual(h.headers, {});
    assert.equal(h.keyHeader, "Authorization");
    const real = parseMcpInput(`npx mcp-remote https://example.com/mcp --header "X-Team: astro"`);
    assert.deepEqual(real.headers, { "X-Team": "astro" });
});

test("npx add-mcp <url> pulls out the URL", () => {
    const r = parseMcpInput("npx add-mcp https://www.monocrawl.com/mcp/oauth");
    assert.deepEqual([r.kind, r.url, r.name], ["url", "https://www.monocrawl.com/mcp/oauth", "Monocrawl"]);
});

test("local commands explain the limit and offer hosted equivalents", () => {
    const c7 = parseMcpInput("npx -y @upstash/context7-mcp@latest");
    assert.equal(c7.kind, "local");
    assert.match(c7.message, /hosted Quasar site cannot/);
    assert.deepEqual(c7.alternative, { name: "Context7", url: "https://mcp.context7.com/mcp" });
    const mono = parseMcpInput(`{"mcpServers": {"monocrawl": {"command": "npx", "args": ["-y", "monocrawl-mcp"]}}}`);
    assert.equal(mono.kind, "local");
    assert.equal(mono.alternative.url, "https://www.monocrawl.com/mcp/oauth");
    const uvx = parseMcpInput("uvx --from manna-mcp==0.9.0 manna --stdio");
    assert.equal(uvx.kind, "local");
    assert.equal(uvx.alternative, undefined);
    const docker = parseMcpInput("docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server");
    assert.equal(docker.kind, "local");
});

test("--from / -p name the package for hosted-equivalent lookup (CX-11)", () => {
    assert.equal(parseMcpInput("uvx --from @upstash/context7-mcp context7").alternative?.name, "Context7");
    assert.equal(parseMcpInput("npx -p @upstash/context7-mcp context7-mcp").alternative?.name, "Context7");
    assert.equal(parseMcpInput("uvx --from=@upstash/context7-mcp context7").alternative?.name, "Context7");
});

test("claude mcp add lines", () => {
    const r = parseMcpInput("claude mcp add --transport http deepwiki https://mcp.deepwiki.com/mcp");
    assert.deepEqual([r.kind, r.url, r.name, r.transport], ["url", "https://mcp.deepwiki.com/mcp", "deepwiki", "streamable_http"]);
    const sse = parseMcpInput("claude mcp add --transport sse linear https://mcp.linear.app/sse");
    assert.equal(sse.transport, "http");
    const local = parseMcpInput("claude mcp add context7 -- npx -y @upstash/context7-mcp");
    assert.equal(local.kind, "local");
    assert.equal(local.alternative.name, "Context7");
});

test("multiple servers: the first URL server is used and the note says so", () => {
    const r = parseMcpInput(JSON.stringify({ mcpServers: {
        local: { command: "npx", args: ["-y", "some-local-server"] },
        hf: { url: "https://huggingface.co/mcp" },
        mslearn: { url: "https://learn.microsoft.com/api/mcp" },
    } }));
    assert.equal(r.name, "hf");
    assert.match(r.note, /Found 3 servers/);
});

test("broken JSON gets a readable message", () => {
    const r = parseMcpInput(`{"mcpServers": {`);
    assert.equal(r.kind, "invalid");
    assert.match(r.message, /could not be read/);
});

test("command splitting and version stripping", () => {
    assert.deepEqual(splitCommand(`npx -y "a b" 'c'`), ["npx", "-y", "a b", "c"]);
    assert.equal(stripVersion("@scope/name@1.2.3"), "@scope/name");
    assert.equal(stripVersion("mcp-remote@latest"), "mcp-remote");
    assert.equal(stripVersion("manna-mcp==0.9.0"), "manna-mcp");
});

// ── the Connect request (credential isolation, transport, names) ────────────

test("a preset click never carries the form's key, headers or env (CX-01)", () => {
    const parsed = parseMcpInput("https://www.monocrawl.com/mcp");
    const form = { ...FORM, apiKey: "mn_secret_key", headerText: "X-Api-Key: other", envText: "A=b" };
    const { body } = buildConnectBody({ preset: { url: "https://mcp.deepwiki.com/mcp", name: "DeepWiki" }, parsed, form });
    assert.deepEqual(body.headers, {});
    assert.deepEqual(body.env, {});
    assert.equal(body.url, "https://mcp.deepwiki.com/mcp");
    assert.equal(body.name, "DeepWiki");
});

test("the form's key goes to the pasted server, as Bearer or into the config's header", () => {
    const parsed = parseMcpInput("https://www.monocrawl.com/mcp");
    const a = buildConnectBody({ parsed, form: { ...FORM, apiKey: "mn_abc" } }).body;
    assert.deepEqual(a.headers, { Authorization: "Bearer mn_abc" });
    const c7 = parseMcpInput(`{"mcpServers": {"c7": {"url": "https://mcp.context7.com/mcp", "headers": {"CONTEXT7_API_KEY": "YOUR_API_KEY"}}}}`);
    const b = buildConnectBody({ parsed: c7, form: { ...FORM, apiKey: "ctx7_123", keyHeader: c7.keyHeader } }).body;
    assert.deepEqual(b.headers, { CONTEXT7_API_KEY: "ctx7_123" });
    const typed = buildConnectBody({ parsed, form: { ...FORM, apiKey: "k", headerText: "Authorization: Bearer typed" } }).body;
    assert.equal(typed.headers.Authorization, "Bearer typed", "an explicit header wins over the key field");
});

test("credentials only survive edits that keep the same endpoint (CX-01)", () => {
    assert.equal(credentialsStillApply("https://www.monocrawl.com/mcp", "https://www.monocrawl.com/mcp/"), true);
    assert.equal(credentialsStillApply("https://h.example/mcp?tenant=A", "https://h.example/mcp?tenant=A#x"), true);
    assert.equal(credentialsStillApply("https://h.example/mcp?tenant=A", "https://h.example/mcp?tenant=B"), false);
    assert.equal(credentialsStillApply("https://www.monocrawl.com/mcp", "https://www.monocrawl.com/mcp?x=1"), false);
    assert.equal(credentialsStillApply("https://h.example/team-a/mcp", "https://h.example/team-b/mcp"), false);
    assert.equal(credentialsStillApply("https://www.monocrawl.com/mcp", "https://www.monocrawl.com/mcp/oauth"), false);
    assert.equal(credentialsStillApply("https://www.monocrawl.com/mcp", "https://evil.example/mcp"), false);
    assert.equal(credentialsStillApply("https://www.monocrawl.com/mcp", ""), false);
    assert.equal(credentialsStillApply("", "https://www.monocrawl.com/mcp"), false);
});

test("a preset's key follow-up goes to the preset (CX-19)", () => {
    // connectPreset puts the preset URL in the box; the follow-up Connect is
    // then a form submit for that URL with the newly typed key.
    const parsed = parseMcpInput("https://www.monocrawl.com/mcp");
    const { body } = buildConnectBody({ parsed, form: { ...FORM, apiKey: "mn_new" } });
    assert.equal(body.url, "https://www.monocrawl.com/mcp");
    assert.deepEqual(body.headers, { Authorization: "Bearer mn_new" });
});

test("every placeholder header is named, none is sent (CX-21)", () => {
    const r = parseMcpInput(JSON.stringify({ mcpServers: { x: { url: "https://x.example/mcp",
        headers: { "X-Workspace": "${WORKSPACE}", Authorization: "Bearer ${TOKEN}", "X-Real": "1" } } } }));
    assert.equal(r.keyHeader, "Authorization");
    assert.deepEqual(r.placeholderHeaders, ["X-Workspace", "Authorization"]);
    assert.deepEqual(r.headers, { "X-Real": "1" });
    assert.match(r.note, /X-Workspace/);
    const { body } = buildConnectBody({ parsed: r, form: { ...FORM, apiKey: "t", keyHeader: r.keyHeader,
        headerText: "X-Workspace: astro" } });
    assert.deepEqual(body.headers, { "X-Real": "1", "X-Workspace": "astro", Authorization: "Bearer t" });
});

test("Local command only applies while Advanced is open (CX-08), quoting kept (CX-14)", () => {
    const parsed = parseMcpInput("https://mcp.deepwiki.com/mcp");
    const closed = buildConnectBody({ parsed, form: { ...FORM, transport: "stdio", advancedOpen: false } }).body;
    assert.equal(closed.transport, "http");
    assert.equal(closed.url, "https://mcp.deepwiki.com/mcp");
    assert.equal(closed.command, null);
    const open = buildConnectBody({ parsed, form: { ...FORM, transport: "stdio", advancedOpen: true,
        cmd: "npx", cmdArgs: `-y pkg --filter "foo bar"`, envText: "K=v" } }).body;
    assert.deepEqual([open.transport, open.url, open.command], ["stdio", null, "npx"]);
    assert.deepEqual(open.args, ["-y", "pkg", "--filter", "foo bar"]);
    assert.deepEqual(open.env, { K: "v" });
    assert.match(buildConnectBody({ parsed, form: { ...FORM, transport: "stdio", advancedOpen: true, cmd: " " } }).error, /command/);
});

test("names: typed names are kept (that is how you edit), derived ones are made unique", () => {
    const servers = [{ name: "docs", url: "https://gitmcp.io/foo/docs" }];
    const parsed = parseMcpInput("https://gitmcp.io/bar/docs");
    assert.equal(buildConnectBody({ parsed, form: FORM, servers }).body.name, "bar-docs");
    assert.equal(buildConnectBody({ parsed, form: { ...FORM, name: "docs", nameTyped: true }, servers }).body.name, "docs");
    assert.equal(buildConnectBody({ preset: { url: "https://gitmcp.io/bar/docs", name: "docs" }, parsed, form: FORM, servers }).body.name, "bar-docs");
});

test("derived names ask the backend not to replace a different server (CX-03, cross-tab)", () => {
    const parsed = parseMcpInput("https://gitmcp.io/bar/docs");
    assert.equal(buildConnectBody({ parsed, form: FORM }).body.replace, false);
    assert.equal(buildConnectBody({ preset: { url: "https://mcp.deepwiki.com/mcp", name: "DeepWiki" }, parsed, form: FORM }).body.replace, false);
    assert.equal(buildConnectBody({ parsed, form: { ...FORM, name: "docs", nameTyped: true } }).body.replace, true);
});

test("no URL, no request", () => {
    assert.match(buildConnectBody({ parsed: parseMcpInput(""), form: FORM }).error, /Paste the server URL/);
    assert.match(buildConnectBody({ parsed: parseMcpInput("https://<host>/mcp"), form: FORM }).error, /placeholder/);
    assert.deepEqual(parsePairs("A: 1\nbad\nB:2", ":"), { A: "1", B: "2" });
});

// ── presets ─────────────────────────────────────────────────────────────────

test("presets are https, unique, dated and honest about verification", () => {
    assert.match(PRESETS_VERIFIED_ON, /^\d{4}-\d{2}-\d{2}$/);
    const ids = new Set();
    for (const p of MCP_PRESETS) {
        assert.ok(!ids.has(p.id)); ids.add(p.id);
        assert.match(p.url, /^https:\/\//);
        assert.ok(["tool-call", "tools-listed", "oauth-discovery"].includes(p.verified), p.id);
        assert.ok(["none", "oauth"].includes(p.auth));
        assert.ok(p.domain && !p.domain.includes("/"));
    }
    for (const id of ["deepwiki", "context7", "huggingface", "mslearn", "gitmcp", "monocrawl"]) assert.ok(ids.has(id), id);
});

test("gitmcp URLs from owner/repo or a GitHub link", () => {
    assert.equal(gitmcpUrl("astropy/astropy"), "https://gitmcp.io/astropy/astropy");
    assert.equal(gitmcpUrl("https://github.com/astropy/photutils.git"), "https://gitmcp.io/astropy/photutils");
    assert.equal(gitmcpUrl("just-a-name"), null);
    assert.equal(gitmcpUrl("a/b/c"), null);
    assert.equal(presetForUrl("https://gitmcp.io/x/y").id, "gitmcp");
    assert.equal(presetForUrl("https://mcp.deepwiki.com/mcp/").id, "deepwiki");
    assert.equal(iconDomain("https://mcp.example.org/mcp"), "example.org");
    assert.equal(apiKeyPageFor("https://www.monocrawl.com/mcp"), "https://www.monocrawl.com/dashboard/api/keys");
    assert.equal(apiKeyPageFor("https://unknown.example/mcp"), null);
});

// ── OAuth popup flow ────────────────────────────────────────────────────────

test("only our backend's well-formed message is accepted (CX-13)", () => {
    const api = "https://quasar-oi14.onrender.com";
    const data = { type: OAUTH_MESSAGE_TYPE, server: "Monocrawl", ok: true, error: null };
    assert.equal(isOAuthResultMessage({ origin: api, data }, api), true);
    assert.equal(isOAuthResultMessage({ origin: "https://evil.example", data }, api), false);
    assert.equal(isOAuthResultMessage({ origin: api, data: { ...data, type: "other" } }, api), false);
    assert.equal(isOAuthResultMessage({ origin: api, data: { ...data, ok: "yes" } }, api), false);
    assert.equal(isOAuthResultMessage({ origin: api, data: { ...data, error: {} } }, api), false);
    assert.equal(isOAuthResultMessage({ origin: api, data: { ...data, ok: false, error: "denied" } }, api), true);
    assert.equal(isOAuthResultMessage({ origin: api, data: "string" }, api), false);
    assert.equal(isOAuthResultMessage({ origin: "http://localhost:8000", data }, "http://localhost:8000"), true);
});

test("full-page fallback flags are read and stripped", () => {
    assert.deepEqual(readOAuthReturn("?mcp_oauth=connected&mcp_server=Monocrawl"), { status: "connected", server: "Monocrawl" });
    assert.deepEqual(readOAuthReturn("?mcp_oauth=signed_in&mcp_server=X"), { status: "signed_in", server: "X" });
    assert.deepEqual(readOAuthReturn("?mcp_oauth=failed"), { status: "failed", server: "" });
    assert.equal(readOAuthReturn("?mcp_oauth=weird"), null);
    assert.equal(readOAuthReturn(""), null);
    assert.equal(stripOAuthReturn("https://www.quasarassistant.com/?mcp_oauth=connected&mcp_server=x&keep=1#h"), "/?keep=1#h");
    assert.equal(stripOAuthReturn("http://localhost:3001/?mcp_oauth=failed"), "/");
});

test("the return banner reflects the real server list, not the flag (CX-09)", () => {
    const flag = { status: "connected", server: "X" };
    assert.equal(returnBanner(flag, []), null, "a forged link for an unknown server shows nothing");
    assert.equal(returnBanner(flag, [{ name: "X", status: { state: "needs_auth", tools: [], connected: false } }]).type, "error");
    assert.equal(returnBanner(flag, [{ name: "X", status: { state: "connected", tools: ["a"], connected: true } }]).type, "success");
    assert.equal(returnBanner(flag, [{ name: "X", oauth: { signed_in: true }, status: { state: "error", tools: [], connected: false } }]).type, "warn");
    // A failed Reconnect while the OLD connection is still up is still a failure.
    const failed = returnBanner({ status: "failed", server: "X" }, [{ name: "X", status: { state: "connected", tools: ["a"], connected: true } }]);
    assert.equal(failed.type, "error");
    assert.match(failed.text, /did not finish\. The existing connection is unchanged/);
});

test("with an attempt id, only THIS sign-in ends the wait (CX-20, another tab)", () => {
    const t0 = 1_000_000;
    const other = oauthWaitDecision({ startedAt: t0, now: t0 + 3000, serverStatus: { state: "connected", tools: [], checked_at: 1e10 },
        oauth: { signed_in: true, signed_in_at: 999, signed_in_attempt: "other-tab" }, baselineSignedInAt: 1, attempt: "mine" });
    assert.equal(other.done, false);
    const mine = oauthWaitDecision({ startedAt: t0, now: t0 + 3000, serverStatus: { state: "connected", tools: [], checked_at: 1e10 },
        oauth: { signed_in: true, signed_in_at: 999, signed_in_attempt: "mine" }, baselineSignedInAt: 1, attempt: "mine" });
    assert.deepEqual(mine, { done: true, ok: true });
});

test("wait decision: a new sign-in wins, closed popup gets a grace period, then a timeout", () => {
    const t0 = 1_000_000;
    const fresh = { signed_in: true, signed_in_at: 42 };
    assert.deepEqual(oauthWaitDecision({ startedAt: t0, now: t0 + 1000, serverStatus: { state: "connected", checked_at: 1e10 },
        oauth: fresh, baselineSignedInAt: null }), { done: true, ok: true });
    assert.deepEqual(oauthWaitDecision({ startedAt: t0, now: t0 + 1000, popupClosedAt: t0 + 500 }), { done: false, popupClosed: true });
    const late = oauthWaitDecision({ startedAt: t0, now: t0 + 60_000, popupClosedAt: t0 + 1000 });
    assert.equal(late.done, true); assert.equal(late.ok, false); assert.match(late.error, /closed/);
    const timeout = oauthWaitDecision({ startedAt: t0, now: t0 + 301_000, serverStatus: { state: "needs_auth" } });
    assert.equal(timeout.ok, false); assert.match(timeout.error, /timed out/);
    assert.equal(oauthWaitDecision({ startedAt: t0, now: t0 + 60_000, popupClosedAt: t0, serverStatus: { state: "connected", checked_at: 1e10 },
        oauth: fresh, baselineSignedInAt: 7 }).ok, true);
});

test("Reconnect: the OLD connection's 'connected' status does not end the wait (CX-20)", () => {
    const t0 = 1_000_000;
    const old = { signed_in: true, signed_in_at: 100 };
    const d = oauthWaitDecision({ startedAt: t0, now: t0 + 3000, serverStatus: { state: "connected", tools: ["a"] },
        oauth: old, baselineSignedInAt: 100 });
    assert.equal(d.done, false);
});

test("a sign-in that stored tokens but did not connect still ends the wait (CX-06)", () => {
    const t0 = 1_000_000;
    const status = { state: "error", error: "tools/list failed", tools: [], connected: false, checked_at: 1e10 };
    const fresh = oauthWaitDecision({ startedAt: t0, now: t0 + 5000, serverStatus: status,
        oauth: { signed_in: true, signed_in_at: 1790000100 }, baselineSignedInAt: 1790000000 });
    assert.deepEqual(fresh, { done: true, ok: true, error: "tools/list failed" });
    const same = oauthWaitDecision({ startedAt: t0, now: t0 + 5000, serverStatus: status,
        oauth: { signed_in: true, signed_in_at: 1790000000 }, baselineSignedInAt: 1790000000 });
    assert.equal(same.done, false, "an old sign-in (e.g. before Reconnect) does not count");
    const first = oauthWaitDecision({ startedAt: t0, now: t0 + 5000, serverStatus: status,
        oauth: { signed_in: true, signed_in_at: 5 }, baselineSignedInAt: null });
    assert.equal(first.ok, true);
});

test("tokens seen before the callback's connection test: wait for its result, capped", () => {
    const t0 = 1_000_000;
    const oauth = { signed_in: true, signed_in_at: 5, signed_in_attempt: "a1" };
    const early = oauthWaitDecision({ startedAt: t0, now: t0 + 2000, serverStatus: undefined, oauth, attempt: "a1" });
    assert.deepEqual(early, { done: false, popupClosed: false, signedIn: true });
    const capped = oauthWaitDecision({ startedAt: t0, now: t0 + 50_000, serverStatus: undefined, oauth, attempt: "a1",
        signedInSeenAt: t0 + 2000 });
    assert.equal(capped.done, true); assert.equal(capped.ok, true); assert.match(capped.error, /did not connect/);
    const result = oauthWaitDecision({ startedAt: t0, now: t0 + 4000, serverStatus: { state: "connected", tools: [], checked_at: 1e10 },
        oauth, attempt: "a1", signedInSeenAt: t0 + 2000 });
    assert.deepEqual(result, { done: true, ok: true });
});

test("a status settled before this sign-in is not its connection result (CX-26)", () => {
    const t0 = 1_000_000;
    const oauth = { signed_in: true, signed_in_at: 2000, signed_in_attempt: "a1" };
    const stale = oauthWaitDecision({ startedAt: t0, now: t0 + 3000, attempt: "a1", oauth,
        serverStatus: { state: "connected", tools: ["x"], checked_at: 1500 } });
    assert.equal(stale.done, false, "the Reconnect's OLD connection does not count");
    const staleErr = oauthWaitDecision({ startedAt: t0, now: t0 + 3000, attempt: "a1", oauth,
        serverStatus: { state: "error", error: "old", tools: [], checked_at: 1999 } });
    assert.equal(staleErr.done, false);
    const fresh = oauthWaitDecision({ startedAt: t0, now: t0 + 3000, attempt: "a1", oauth,
        serverStatus: { state: "connected", tools: ["x"], checked_at: 2001 } });
    assert.deepEqual(fresh, { done: true, ok: true });
    // The settle cap outlasts the backend's 42 s worst-case connection test (CX-27).
    const at40 = oauthWaitDecision({ startedAt: t0, now: t0 + 42_000, attempt: "a1", oauth, signedInSeenAt: t0 });
    assert.equal(at40.done, false);
});

test("message and poll share one finish (CX-07)", () => {
    const calls = [];
    const once = createOnce((ok, err) => calls.push([ok, err]));
    assert.equal(once.finish(true, null), true);
    assert.equal(once.finish(false, "late poll"), false);
    assert.equal(once.done, true);
    assert.deepEqual(calls, [[true, null]]);
    // Cancel (or effect cleanup) before any result: a poll that resolves
    // afterwards completes nothing (CX-07).
    const after = [];
    const cancelled = createOnce((...a) => after.push(a));
    cancelled.cancel();
    assert.equal(cancelled.finish(true, null), false);
    assert.deepEqual(after, []);
});

test("connect responses map to the next form step; saved comes only from the backend (CX-02)", () => {
    assert.deepEqual(nextStep({ status: "needs_auth", authorize_url: "https://a.example/authorize?x=1", server: { name: "A" } }),
        { step: "oauth", authorizeUrl: "https://a.example/authorize?x=1", server: "A" });
    assert.equal(nextStep({ status: "needs_auth", authorize_url: "javascript:alert(1)" }).step, "error");
    assert.equal(nextStep({ status: "needs_api_key", message: "m" }).step, "api_key");
    assert.deepEqual(nextStep({ status: "connected", server: { name: "d" }, connection: { tools: ["a"] } }), { step: "done", server: "d", tools: ["a"] });
    assert.deepEqual(nextStep({ status: "error", saved: true, server: { name: "x" }, connection: { error: "boom" } }),
        { step: "error", message: "boom", saved: true });
    const notSaved = nextStep({ status: "error", saved: false, server: { name: "x" }, message: "Could not reach the server. Nothing was changed." });
    assert.equal(notSaved.saved, false, "a server echo in the reply is not proof of a save");
    assert.equal(nextStep({ status: "error", server: { name: "x" } }).saved, false);
});

test("status pills are honest about what was checked", () => {
    assert.equal(statusView({ state: "connected", tools: ["a", "b"] }).label, "Connected (tools listed) · 2 tools");
    assert.equal(statusView({ connected: true, tools: ["a"] }).label, "Connected (tools listed) · 1 tool");
    assert.equal(statusView({ state: "needs_auth" }).tone, "warn");
    assert.equal(statusView({ state: "needs_api_key" }).label, "Needs API key");
    assert.equal(statusView({ connected: false, error: "x" }).label, "Not connected");
    assert.equal(statusView(undefined, "none").label, "Not checked yet");
});

test("bearer header helper: Bearer by default, an explicit scheme as typed (CX-17 copy)", () => {
    assert.equal(bearerHeader(" mn_abc "), "Bearer mn_abc");
    assert.equal(bearerHeader("Bearer mn_abc"), "Bearer mn_abc");
    assert.equal(bearerHeader("Basic abc"), "Basic abc");
    assert.equal(bearerHeader(""), null);
});
