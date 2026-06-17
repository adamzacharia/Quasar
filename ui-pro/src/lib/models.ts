export const TACC_MODELS = [
    "gpt-oss-120b",
    "Qwen3-32B",
    "gemma-4-31B-it",
];

export const DEFAULT_AVAILABLE_MODELS = [
    "gpt-5.4-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
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
