export const TACC_MODELS = [
    "gpt-oss-120b",
    "Llama-4-Maverick-17B-128E-Instruct",
    "gemma-4-31B-it",
    "MiniMax-M2.7",
    "Qwen3-32B",
    "Meta-Llama-3.2-1B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3.3-70B-Instruct",
    "Mistral-Large-3-675B-Instruct-2512",
    "E5-Mistral-7B-Instruct",
];

export const DEFAULT_AVAILABLE_MODELS = [
    ...TACC_MODELS,
];

export function isTaccModel(model: string): boolean {
    return TACC_MODELS.includes(model) || model.startsWith("tacc/");
}
