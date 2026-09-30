import test from "node:test";
import assert from "node:assert/strict";

import {
    buildGroups,
    connectedLabels,
    contextChip,
    flattenRows,
    fuzzyMatch,
    modelLabel,
    nextRecentModels,
    priceLine,
    scoreModel,
    suggestedLockedProvider,
} from "../src/lib/model-catalog.js";

const model = (provider, id, extra = {}) => ({
    provider,
    id,
    displayName: extra.displayName || id,
    capabilities: {},
    source: "live",
    ...extra,
});

const catalog = (provider, status, models, extra = {}) => ({
    provider,
    status,
    models,
    stale: false,
    unlockCount: 0,
    ...extra,
});

const OPUS = model("anthropic", "claude-opus-4-8", {
    displayName: "Claude Opus 4.8",
    inputPricePerM: 5,
    outputPricePerM: 25,
    contextWindow: 200000,
});
const SONNET = model("anthropic", "claude-sonnet-5", { displayName: "Claude Sonnet 5" });
const GPT41 = model("openai", "gpt-4.1", { inputPricePerM: 2, outputPricePerM: 8 });
const GPT_MINI = model("openai", "gpt-5.4-mini");
const DEEPSEEK = model("deepseek", "deepseek-v4-pro");
const TACC = model("tacc", "gpt-oss-120b");

/* ── fuzzy scorer ───────────────────────────────────────────────────────── */

test("scorer ranks prefix above word-start above subsequence", () => {
    const prefix = fuzzyMatch("gpt", "gpt-4.1").score;
    const wordStart = fuzzyMatch("mini", "gpt-5.4-mini").score;
    const subsequence = fuzzyMatch("gp1", "gpt-4.1").score;

    assert.ok(prefix > wordStart, "a prefix hit should outrank a word-start hit");
    assert.ok(wordStart > subsequence, "a word-start hit should outrank a subsequence hit");
    assert.ok(subsequence > 0, "a subsequence hit should still match");
});

test("scorer ranks an exact id above a mere prefix", () => {
    assert.ok(fuzzyMatch("gpt-4.1", "gpt-4.1").score > fuzzyMatch("gpt", "gpt-4.1").score);
});

test("scorer prefers a word-start occurrence over one mid-token", () => {
    // "pro" appears mid-token in "deepseek-reasoner-pro"? Use an explicit pair.
    const midToken = fuzzyMatch("eep", "deepseek-v4-pro");
    const atWordStart = fuzzyMatch("v4", "deepseek-v4-pro");
    assert.ok(atWordStart.score > midToken.score);
});

test("scorer is case-insensitive and reports match ranges", () => {
    const upper = fuzzyMatch("OPUS", "claude-opus-4-8");
    const lower = fuzzyMatch("opus", "claude-opus-4-8");
    assert.equal(upper.score, lower.score);
    assert.deepEqual(upper.ranges, [[7, 11]], "should highlight the matched run");
});

test("scorer returns no match when a character is missing", () => {
    assert.equal(fuzzyMatch("zzz", "claude-opus-4-8").score, 0);
    assert.deepEqual(fuzzyMatch("zzz", "claude-opus-4-8").ranges, []);
});

test("empty query never matches", () => {
    assert.equal(fuzzyMatch("", "gpt-4.1").score, 0);
    assert.equal(scoreModel("   ", GPT41, "OpenAI").score, 0);
});

test("alias table hits a model whose id does not contain the typed term", () => {
    // "r1" is not a substring of "deepseek-reasoner"; the alias table bridges it.
    const reasoner = model("deepseek", "deepseek-reasoner");
    assert.equal(fuzzyMatch("r1", reasoner.id).score, 0, "no literal or subsequence hit");
    assert.ok(scoreModel("r1", reasoner, "DeepSeek").score > 0, "alias should match");
});

test("alias hit outranks an incidental subsequence hit", () => {
    const aliased = scoreModel("v4", DEEPSEEK, "DeepSeek").score;
    const incidental = scoreModel("v4", model("openai", "gpt-4-vision-x"), "OpenAI").score;
    assert.ok(aliased > incidental);
});

test("a provider-name match scores below a direct id match", () => {
    const byId = scoreModel("gpt", GPT41, "OpenAI").score;
    const byProvider = scoreModel("openai", GPT41, "OpenAI").score;
    assert.ok(byProvider > 0, "provider name should still match its models");
    assert.ok(byId > byProvider, "an id hit is stronger evidence than a provider hit");
});

test("a model whose display name equals its id still gets highlight ranges", () => {
    // The row renders the display name, so it must carry ranges even when the
    // two strings are identical — otherwise nothing highlights at all.
    const scored = scoreModel("mini", GPT_MINI, "OpenAI");
    assert.ok(scored.score > 0);
    assert.deepEqual(scored.nameRanges, [[8, 12]], "'gpt-5.4-mini' highlights its 'mini'");
});

test("display name matches are reported separately from id matches", () => {
    const scored = scoreModel("Opus", OPUS, "Anthropic Claude");
    assert.ok(scored.score > 0);
    assert.deepEqual(scored.nameRanges, [[7, 11]], "highlights inside 'Claude Opus 4.8'");
    assert.deepEqual(scored.idRanges, [[7, 11]], "highlights inside 'claude-opus-4-8'");
});

/* ── price + chips ──────────────────────────────────────────────────────── */

test("price line keeps the established In/Out format", () => {
    assert.equal(priceLine(GPT41), "In: $2/M · Out: $8/M");
});

test("a model with no known price renders 'Pricing unknown' rather than a guess", () => {
    assert.equal(priceLine(GPT_MINI), "Pricing unknown");
});

test("included-quota TACC models keep showing no price line", () => {
    assert.equal(priceLine(TACC), null);
});

test("context window renders as a compact chip", () => {
    assert.equal(contextChip(OPUS), "200K ctx");
    assert.equal(contextChip(model("google", "gemini-2.5-pro", { contextWindow: 1000000 })), "1M ctx");
    assert.equal(contextChip(GPT41), null, "no window means no chip");
});

test("local models drop their routing prefix in the row label", () => {
    assert.equal(modelLabel(model("local", "local/llama-3", { displayName: "local/llama-3" })), "llama-3");
});

/* ── grouping ───────────────────────────────────────────────────────────── */

test("not-connected providers contribute no selectable models", () => {
    const { groups, locked } = buildGroups("", [
        catalog("openai", "included_quota", [GPT41]),
        catalog("anthropic", "not_connected", [], { unlockCount: 3 }),
    ]);
    assert.deepEqual(groups.map(g => g.key), ["openai"]);
    assert.deepEqual(locked, [{ provider: "anthropic", label: "Anthropic", unlockCount: 3 }]);
});

test("empty groups are hidden while searching", () => {
    const { groups } = buildGroups("opus", [
        catalog("openai", "included_quota", [GPT41]),
        catalog("anthropic", "connected", [OPUS, SONNET]),
    ]);
    assert.deepEqual(groups.map(g => g.key), ["anthropic"], "OpenAI has no 'opus' match");
    assert.deepEqual(groups[0].rows.map(r => r.model.id), ["claude-opus-4-8"]);
});

test("connected providers are flagged BYOK, included-quota ones are not", () => {
    const { groups } = buildGroups("", [
        catalog("openai", "included_quota", [GPT41]),
        catalog("anthropic", "connected", [OPUS]),
    ]);
    assert.equal(groups.find(g => g.key === "openai").byok, false);
    assert.equal(groups.find(g => g.key === "anthropic").byok, true);
});

test("results stay grouped and ordered by score within a provider", () => {
    const { groups } = buildGroups("pro", [
        catalog("deepseek", "connected", [
            model("deepseek", "deepseek-chat-preview"),
            DEEPSEEK,
        ]),
    ]);
    assert.equal(groups.length, 1);
    assert.equal(groups[0].rows[0].model.id, "deepseek-v4-pro", "alias hit sorts first");
});

test("a stale catalog is flagged on its group", () => {
    const { groups } = buildGroups("", [
        catalog("anthropic", "connected", [OPUS], { stale: true }),
    ]);
    assert.equal(groups[0].stale, true);
});

test("flattened rows follow render order for keyboard navigation", () => {
    const { groups } = buildGroups("", [
        catalog("openai", "included_quota", [GPT41, GPT_MINI]),
        catalog("anthropic", "connected", [OPUS]),
    ]);
    assert.deepEqual(
        flattenRows(groups).map(r => r.model.id),
        ["gpt-5.4-mini", "gpt-4.1", "claude-opus-4-8"],
    );
});

/* ── the spec's component scenario, exercised against the real logic ────── */

test("typing 'opus' surfaces Anthropic models when Anthropic is connected", () => {
    const providers = [
        catalog("openai", "included_quota", [GPT41]),
        catalog("anthropic", "connected", [OPUS, SONNET]),
    ];
    const { groups, locked } = buildGroups("opus", providers);

    assert.deepEqual(groups.map(g => g.label), ["Anthropic Claude"]);
    assert.deepEqual(groups[0].rows.map(r => r.model.id), ["claude-opus-4-8"]);
    assert.equal(locked.length, 0, "nothing to unlock when Anthropic is connected");
    assert.equal(suggestedLockedProvider("opus", providers), null, "no key hint needed");
});

test("typing 'opus' shows the add-key hint when Anthropic is not connected", () => {
    const providers = [
        catalog("openai", "included_quota", [GPT41]),
        catalog("anthropic", "not_connected", [], { unlockCount: 3 }),
    ];
    const { groups, locked } = buildGroups("opus", providers);

    assert.equal(flattenRows(groups).length, 0, "a locked provider offers no models");
    assert.deepEqual(locked.map(l => l.label), ["Anthropic"]);
    assert.deepEqual(suggestedLockedProvider("opus", providers), {
        provider: "anthropic",
        label: "Anthropic",
        unlockCount: 3,
    });
});

test("the empty state names the providers that are actually connected", () => {
    const providers = [
        catalog("openai", "included_quota", [GPT41]),
        catalog("deepseek", "included_quota", [DEEPSEEK]),
        catalog("anthropic", "not_connected", [], { unlockCount: 3 }),
    ];
    assert.deepEqual(connectedLabels(providers), ["OpenAI", "DeepSeek"]);
});

test("a provider name that is not connected still offers its key link", () => {
    const providers = [catalog("google", "not_connected", [], { unlockCount: 2 })];
    assert.equal(suggestedLockedProvider("gemini", providers).provider, "google");
    assert.equal(suggestedLockedProvider("gpt", providers), null, "unrelated query, no hint");
});

/* ── recently used ──────────────────────────────────────────────────────── */

test("large catalogs start with four latest models and searches reach every match", () => {
    const ids = ["gpt-4.1", "gpt-5.4-mini", "gpt-5.6-sol", "gpt-6-luna", "gpt-6-sol", "gpt-6-astra", "gpt-6.1-sol"];
    const providers = [catalog("openai", "included_quota", ids.map(id => model("openai", id)))];
    const initial = buildGroups("", providers);
    assert.deepEqual(flattenRows(initial.groups).map(row => row.model.id), ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]);
    assert.equal(initial.groups[0].hiddenCount, 3);
    assert.equal(flattenRows(buildGroups("   ", providers).groups).length, 4);
    const searching = buildGroups("gpt", providers);
    assert.equal(flattenRows(searching.groups).length, ids.length);
    assert.equal(searching.groups[0].rows[0].model.id, "gpt-6.1-sol");
    assert.equal(searching.groups[0].hiddenCount, 0);
    assert.equal(flattenRows(buildGroups("", providers, { expanded: true }).groups).length, ids.length);
    assert.equal(flattenRows(buildGroups("gpt-4.1", providers).groups)[0].model.id, "gpt-4.1", "exact relevance wins over generation");
});

test("selected older model stays within the initial four without hiding latest search matches", () => {
    const ids = ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-4.1"];
    const providers = [catalog("openai", "connected", ids.map(id => model("openai", id)))];
    const options = { selectedModel: "gpt-4.1" };
    const initial = flattenRows(buildGroups("", providers, options).groups);
    assert.equal(initial.length, 4);
    assert.equal(initial[0].model.id, "gpt-6.1-sol");
    assert.equal(initial[3].model.id, "gpt-4.1");
    assert.equal(flattenRows(buildGroups("gpt", providers, options).groups).length, 5);
    assert.equal(flattenRows(buildGroups("", providers, { selectedModel: "missing" }).groups).length, 4);
});

test("search sorts numeric generations and uses provider release times within a generation", () => {
    const providers = [catalog("openai", "connected", [
        model("openai", "gpt-5.9"),
        model("openai", "gpt-5.10"),
        model("openai", "gpt-7-sol", { createdAt: "2026-09-01T00:00:00Z" }),
        model("openai", "gpt-7-luna", { createdAt: "2026-09-02T00:00:00Z" }),
    ])];
    assert.deepEqual(flattenRows(buildGroups("gpt", providers).groups).map(row => row.model.id), ["gpt-7-luna", "gpt-7-sol", "gpt-5.10", "gpt-5.9"]);
});

test("stable aliases stay above newer dated snapshots on broad searches", () => {
    const providers = [catalog("openai", "connected", [
        model("openai", "gpt-6-sol-2026-09-20", { createdAt: "2026-09-20T00:00:00Z" }),
        model("openai", "gpt-6-sol", { createdAt: "2026-09-01T00:00:00Z" }),
    ])];
    assert.deepEqual(flattenRows(buildGroups("gpt", providers).groups).map(row => row.model.id), ["gpt-6-sol", "gpt-6-sol-2026-09-20"]);
    assert.equal(flattenRows(buildGroups("gpt-6-sol-2026-09-20", providers).groups)[0].model.id, "gpt-6-sol-2026-09-20");
});

test("compact groups preserve provider locks, recent shortcuts and keyboard render order", () => {
    const ids = Array.from({ length: 8 }, (_, index) => `gpt-6.${index}-sol`);
    const providers = [
        catalog("openai", "connected", ids.map(id => model("openai", id))),
        catalog("google", "not_connected", [], { unlockCount: 8 }),
    ];
    const { groups, locked } = buildGroups("", providers, { recent: [ids[0]] });
    assert.equal(groups[0].key, "recent");
    assert.equal(groups[1].rows.length, 4);
    assert.deepEqual(flattenRows(groups).map(row => row.model.id), groups.flatMap(group => group.rows.map(row => row.model.id)));
    assert.equal(locked[0].provider, "google");
    assert.equal(flattenRows(buildGroups("google", providers).groups).length, 0);
});

test("recent models are most-recent-first, de-duplicated and capped at 3", () => {
    let recent = [];
    recent = nextRecentModels(recent, "gpt-4.1");
    recent = nextRecentModels(recent, "claude-opus-4-8");
    recent = nextRecentModels(recent, "gpt-4.1");
    assert.deepEqual(recent, ["gpt-4.1", "claude-opus-4-8"]);

    recent = nextRecentModels(recent, "deepseek-v4-pro");
    recent = nextRecentModels(recent, "gpt-oss-120b");
    assert.deepEqual(recent, ["gpt-oss-120b", "deepseek-v4-pro", "gpt-4.1"], "oldest drops off");
});

test("the Recent group leads the unfiltered list and is dropped while searching", () => {
    const providers = [
        catalog("openai", "included_quota", [GPT41]),
        catalog("anthropic", "connected", [OPUS]),
    ];
    const withRecent = buildGroups("", providers, { recent: ["claude-opus-4-8"] });
    assert.equal(withRecent.groups[0].key, "recent");
    assert.deepEqual(withRecent.groups[0].rows.map(r => r.model.id), ["claude-opus-4-8"]);

    const searching = buildGroups("gpt", providers, { recent: ["claude-opus-4-8"] });
    assert.ok(!searching.groups.some(g => g.key === "recent"), "Recent is an unfiltered shortcut");
});

test("a recent model that is no longer available is skipped", () => {
    const providers = [catalog("openai", "included_quota", [GPT41])];
    const { groups } = buildGroups("", providers, { recent: ["claude-opus-4-8", "gpt-4.1"] });
    assert.deepEqual(groups[0].rows.map(r => r.model.id), ["gpt-4.1"]);
});

test("cross-provider duplicate IDs cannot be selected from search, expanded or recent rows", () => {
    const providers = [
        catalog("openai", "connected", [model("openai", "gpt-oss-120b"), GPT41]),
        catalog("tacc", "included_quota", [model("tacc", "gpt-oss-120b"), model("tacc", "Qwen3-32B")]),
    ];
    const initial = buildGroups("", providers, { recent: ["gpt-oss-120b"] });
    const ids = flattenRows(initial.groups).map(row => row.model.id);
    assert.deepEqual(ids, ["gpt-4.1", "Qwen3-32B"]);
    assert.ok(!initial.groups.some(group => group.key === "recent"));
    assert.equal(flattenRows(buildGroups("gpt-oss-120b", providers).groups).length, 0);
    assert.ok(!flattenRows(buildGroups("", providers, { expanded: true }).groups).some(row => row.model.id === "gpt-oss-120b"));
});
