export const TACC_MODELS = [
    "gpt-oss-120b",
    "DeepSeek-V3.2",
    "Qwen3-32B",
    "gemma-4-31B-it",
    "MiniMax-M2.7",
];

export const OPENAI_MODELS = [
    "gpt-6.1-sol",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.3-codex",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
];

export const DEEPSEEK_MODELS = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
];

export const DEFAULT_AVAILABLE_MODELS = [
    ...OPENAI_MODELS,
    ...DEEPSEEK_MODELS,
    ...TACC_MODELS,
];

export function isTaccModel(model: string): boolean {
    return TACC_MODELS.includes(model) || model.startsWith("tacc/");
}

export function mergeAvailableModels(models: unknown): string[] {
    const merged = new Set(DEFAULT_AVAILABLE_MODELS);

    if (Array.isArray(models)) {
        for (const model of models) {
            if (typeof model === "string" && model.trim()) {
                merged.add(model);
            }
        }
    }

    return Array.from(merged);
}
