import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";
import { create } from "zustand";

// Execute the actual TS helpers/store with the existing compiler dependency.
// This works on the project's Node versions without relying on native TS loading.
function loadTs(file, dependencies = {}, globals = {}) {
    const source = fs.readFileSync(new URL(file, import.meta.url), "utf8");
    const { outputText } = ts.transpileModule(source, {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
    });
    const module = { exports: {} };
    vm.runInNewContext(outputText, {
        module, exports: module.exports,
        require: name => {
            if (!(name in dependencies)) throw new Error(`Unexpected import: ${name}`);
            return dependencies[name];
        },
        ...globals,
    });
    return module.exports;
}

const pricing = loadTs("../src/lib/pricing.ts");
const models = loadTs("../src/lib/models.ts");

test("generated pricing executes exact rates and provider-specific suffix rules", () => {
    assert.equal(pricing.modelPrice("openai", "gpt-6.1-sol").inputPricePerM, 2);
    assert.equal(pricing.modelPrice("openai", "gpt-6.1-sol").outputPricePerM, 10);
    assert.equal(pricing.modelPrice("openai", "gpt-5.7"), undefined);
    assert.equal(pricing.modelPrice("openai", "gpt-5.4-mini-ultra"), undefined);
    assert.equal(pricing.modelPrice("openai", "gpt-6-sol-2026-09-01").inputPricePerM, 2);
    assert.equal(pricing.modelPrice("google", "gemini-2.5-flash.1").inputPricePerM, 0.3);
    assert.equal(pricing.modelPrice("google", "gpt-6.1-sol"), undefined);
});

test("a failed catalog request produces a complete, priced OpenAI offline fallback", async () => {
    const hook = loadTs("../src/lib/useAvailableModels.ts", {
        react: { useEffect() {} },
        zustand: { create },
        "./auth-store": { authBearerHeaders: () => ({}) },
        "./models": models,
        "./pricing": pricing,
    }, {
        process: { env: {} },
        fetch: async () => { throw new Error("offline fixture"); },
    });
    await hook.useCatalogStore.getState().load();
    const state = hook.useCatalogStore.getState();
    const openai = state.data.providers.find(entry => entry.provider === "openai");
    assert.equal(state.error, "Could not load the model catalog.");
    assert.equal(openai.stale, true);
    assert.equal(openai.models.length, 25);
    assert.equal(openai.models[0].id, "gpt-6.1-sol");
    for (const model of openai.models) {
        const expected = pricing.modelPrice("openai", model.id);
        assert.equal(model.provider, "openai");
        assert.equal(model.inputPricePerM, expected.inputPricePerM);
        assert.equal(model.outputPricePerM, expected.outputPricePerM);
    }
});
