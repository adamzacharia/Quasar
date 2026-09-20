/**
 * Types for model-catalog.js.
 *
 * The implementation is plain JS so `node --test` can import it directly (the
 * repo convention for src/lib logic modules); this declaration file is what
 * the TypeScript side sees, so callers stay fully typed with no `any`.
 *
 * ModelInfo / ProviderCatalog / AvailableModels mirror the wire shape of
 * GET /api/models/available — services/provider_models.py.
 */

export type ProviderId =
    | "openai"
    | "deepseek"
    | "anthropic"
    | "google"
    | "tacc"
    | "local";

export type ProviderStatus = "connected" | "included_quota" | "not_connected";

export interface ModelCapabilities {
    vision?: boolean;
    tools?: boolean;
    reasoning?: boolean;
}

export interface ModelInfo {
    provider: ProviderId;
    id: string;
    displayName: string;
    contextWindow?: number;
    maxOutput?: number;
    inputPricePerM?: number;
    outputPricePerM?: number;
    capabilities: ModelCapabilities;
    createdAt?: string;
    source: "live" | "static";
}

export interface ProviderCatalog {
    provider: ProviderId;
    status: ProviderStatus;
    models: ModelInfo[];
    stale: boolean;
    fetchedAt?: string;
    /** Curated fallback count — what a key would unlock, not an entitlement claim. */
    unlockCount: number;
}

export interface AvailableModels {
    providers: ProviderCatalog[];
    defaultModel: string;
}

/** Inclusive start, exclusive end — a run of matched characters. */
export type MatchRange = [number, number];

export interface FuzzyMatch {
    score: number;
    ranges: MatchRange[];
}

export interface ScoredModel {
    score: number;
    idRanges: MatchRange[];
    nameRanges: MatchRange[];
}

export interface ModelRow extends ScoredModel {
    model: ModelInfo;
}

export interface ModelGroup {
    /** Provider id, or "recent" for the recently-used shortcut group. */
    key: string;
    provider: string;
    label: string;
    byok: boolean;
    stale: boolean;
    rows: ModelRow[];
}

export interface LockedProvider {
    provider: ProviderId;
    label: string;
    unlockCount: number;
}

export interface BuiltGroups {
    groups: ModelGroup[];
    locked: LockedProvider[];
}

export interface FlatRow {
    groupKey: string;
    model: ModelInfo;
}

export declare const PROVIDER_ORDER: readonly ProviderId[];
export declare const PROVIDER_LABELS: Readonly<Record<string, string>>;
export declare const PROVIDER_SHORT_LABELS: Readonly<Record<string, string>>;
export declare const MODEL_ALIASES: Readonly<Record<string, readonly string[]>>;
export declare const RECENT_MODELS_KEY: string;
export declare const RECENT_LIMIT: number;

export declare function fuzzyMatch(query: string, text: string): FuzzyMatch;
export declare function scoreModel(
    query: string,
    model: ModelInfo,
    providerLabel?: string,
): ScoredModel;
export declare function priceLine(model: ModelInfo | null | undefined): string | null;
export declare function contextChip(model: ModelInfo | null | undefined): string | null;
export declare function modelLabel(model: ModelInfo | null | undefined): string;
export declare function isSelectable(catalog: ProviderCatalog | null | undefined): boolean;
export declare function buildGroups(
    query: string,
    providers: ProviderCatalog[],
    options?: { recent?: string[] },
): BuiltGroups;
export declare function flattenRows(groups: ModelGroup[]): FlatRow[];
export declare function connectedLabels(providers: ProviderCatalog[]): string[];
export declare function suggestedLockedProvider(
    query: string,
    providers: ProviderCatalog[],
): LockedProvider | null;
export declare function loadRecentModels(): string[];
export declare function nextRecentModels(current: string[], modelId: string): string[];
export declare function persistRecentModels(ids: string[]): void;
