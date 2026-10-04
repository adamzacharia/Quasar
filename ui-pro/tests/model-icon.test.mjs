import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import { Bot } from "lucide-react";
import * as lucide from "lucide-react";
import * as catalogLogic from "../src/lib/model-catalog.js";

function loadTs(file, dependencies = {}) {
    const source = fs.readFileSync(new URL(file, import.meta.url), "utf8");
    const { outputText } = ts.transpileModule(source, {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.ReactJSX },
    });
    const module = { exports: {} };
    vm.runInNewContext(outputText, {
        module, exports: module.exports,
        require: name => {
            if (!(name in dependencies)) throw new Error(`Unexpected import: ${name}`);
            return dependencies[name];
        },
    });
    return module.exports;
}

const models = loadTs("../src/lib/models.ts");
const { ModelIcon } = loadTs("../src/components/ModelIcon.tsx", {
    "react/jsx-runtime": jsxRuntime,
    "lucide-react": { Bot },
    "../lib/models": models,
});
const render = props => renderToStaticMarkup(ModelIcon(props));

test("Claude families render the Anthropic mark instead of the robot", () => {
    for (const model of ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5", "claude-fable-5"]) {
        const markup = render({ model });
        assert.match(markup, /data-provider-logo="anthropic"/);
        assert.match(markup, /viewBox="0 0 24 24"/);
        assert.match(markup, /aria-hidden="true"/);
        assert.doesNotMatch(markup, /lucide-bot/);
    }
});

test("every known TACC model uses the transparent wordmark, including GPT and DeepSeek names", () => {
    for (const model of [...models.TACC_MODELS, "tacc/custom-model"]) {
        const markup = render({ model });
        assert.match(markup, /data-provider-logo="tacc"/);
        assert.match(markup, /tacc-logo\.png/);
        assert.match(markup, /width:auto;aspect-ratio:800 \/ 314/);
        assert.doesNotMatch(markup, /data-provider-logo="(?:openai|deepseek)"/);
    }
});

test("catalog provider metadata overrides ambiguous or custom model ids", () => {
    assert.match(render({ model: "custom-deployment", provider: "tacc" }), /data-provider-logo="tacc"/);
    assert.match(render({ model: "deepseek-v4-pro", provider: "tacc" }), /data-provider-logo="tacc"/);
    assert.match(render({ model: "gpt-oss-120b", provider: "openai" }), /data-provider-logo="openai"/);
    assert.match(render({ model: "custom-claude", provider: "anthropic" }), /data-provider-logo="anthropic"/);
    assert.doesNotMatch(render({ model: "claude-opus-4-8", provider: "local" }), /data-provider-logo=/);
});

test("OpenAI, DeepSeek and unknown-model fallbacks retain their icons", () => {
    assert.match(render({ model: "gpt-6.1-sol" }), /data-provider-logo="openai"/);
    assert.match(render({ model: "deepseek-v4-pro" }), /data-provider-logo="deepseek"/);
    assert.match(render({ model: "local/custom-model" }), /lucide-bot/);
    assert.match(render({ model: "local/custom-model" }), /aria-hidden="true"/);
    assert.match(render({ model: "GPT-FOO" }), /lucide-bot/, "legacy OpenAI ID matching remains case-sensitive");
});

let dropdownCatalog = [];
const { ModelDropdown } = loadTs("../src/components/ModelDropdown.tsx", {
    "react/jsx-runtime": jsxRuntime,
    react: React,
    "lucide-react": lucide,
    "./ModelIcon": { ModelIcon },
    "../lib/models": models,
    "../lib/model-catalog": catalogLogic,
    "../lib/useAvailableModels": { useAvailableModels: () => ({ providers: dropdownCatalog }) },
});

test("the actual dropdown propagates custom TACC metadata and keeps hosted tag/cost semantics", () => {
    const id = "custom-tacc-deployment";
    dropdownCatalog = [{ provider: "tacc", status: "included_quota", models: [
        { provider: "tacc", id, displayName: id, source: "static", capabilities: {}, inputPricePerM: 2, outputPricePerM: 8 },
    ] }];
    const markup = renderToStaticMarkup(React.createElement(ModelDropdown, { selectedModel: id, onSelect() {} }));
    assert.match(markup, /data-provider-logo="tacc"/);
    assert.match(markup, /US hosted/);
    assert.doesNotMatch(markup, /In: \$/);
    assert.doesNotMatch(markup, /Out: \$/);
    assert.match(markup, /title="custom-tacc-deployment"/);
});

test("the dropdown does not claim a provider or price for cross-provider duplicate ids", () => {
    const id = "gpt-oss-120b";
    dropdownCatalog = ["openai", "tacc"].map(provider => ({ provider, status: "included_quota", models: [
        { provider, id, displayName: id, source: "static", capabilities: {}, inputPricePerM: 2, outputPricePerM: 8 },
    ] }));
    const markup = renderToStaticMarkup(React.createElement(ModelDropdown, { selectedModel: id, onSelect() {} }));
    assert.match(markup, /lucide-bot/);
    assert.doesNotMatch(markup, /data-provider-logo=/);
    assert.doesNotMatch(markup, /US hosted|In: \$|Out: \$/);
});

test("Gemini, Gemma and Nano Banana models render the Gemini mark", () => {
    for (const model of ["gemini-3.8-flash", "gemma-4-31b-it", "nano-banana-pro-preview"]) {
        const markup = render({ model });
        assert.match(markup, /data-provider-logo="google"/);
        assert.doesNotMatch(markup, /lucide-bot/);
    }
    assert.match(render({ model: "custom-id", provider: "google" }), /data-provider-logo="google"/);
    // A TACC-served Gemma keeps the TACC wordmark.
    assert.match(render({ model: "gemma-4-31B-it" }), /data-provider-logo="tacc"/);
});

test("image-generation models get an Image tag in the dropdown; chat models do not", () => {
    dropdownCatalog = [{ provider: "google", status: "connected", models: [
        { provider: "google", id: "gemini-3.1-flash-image", displayName: "Nano Banana 2", source: "live",
          capabilities: { imageGeneration: true } },
        { provider: "google", id: "gemini-3.8-flash", displayName: "Gemini 3.8 Flash", source: "live", capabilities: {} },
    ] }];
    // The list only renders while open: a copy whose first useState(false) (the
    // `open` flag) starts true renders the rows statically.
    let forceOpen = true;
    const ReactOpen = { ...React, useState: init => {
        if (forceOpen && init === false) { forceOpen = false; return React.useState(true); }
        return React.useState(init);
    } };
    const { ModelDropdown: OpenDropdown } = loadTs("../src/components/ModelDropdown.tsx", {
        "react/jsx-runtime": jsxRuntime, react: ReactOpen, "lucide-react": lucide, "./ModelIcon": { ModelIcon },
        "../lib/models": models, "../lib/model-catalog": catalogLogic,
        "../lib/useAvailableModels": { useAvailableModels: () => ({ providers: dropdownCatalog }) },
    });
    const markup = renderToStaticMarkup(React.createElement(OpenDropdown, { selectedModel: "gemini-3.8-flash", onSelect() {} }));
    assert.match(markup, /Nano Banana 2/);
    assert.equal((markup.match(/data-model-tag="image"/g) || []).length, 1);
    assert.match(markup, /data-provider-logo="google"/);
});

test("isImageGenerationModelId mirrors the backend detector", () => {
    for (const id of ["gemini-3.1-flash-image", "gemini-3-pro-image-preview", "nano-banana-pro-preview",
        "gpt-image-2", "chatgpt-image-latest"]) {
        assert.equal(catalogLogic.isImageGenerationModelId(id), true, id);
    }
    for (const id of ["gemini-3.8-flash", "gemini-3.1-flash-lite", "gpt-5.4", "", null]) {
        assert.equal(catalogLogic.isImageGenerationModelId(id), false, String(id));
    }
});
