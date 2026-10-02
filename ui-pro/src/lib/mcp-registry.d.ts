export interface RegistryHeader {
    name: string;
    required: boolean;
    secret: boolean;
    description: string;
    value_template: string | null;
}

export interface RegistryItem {
    id: string;
    title: string;
    description: string;
    version: string;
    url: string;
    transport: "streamable-http" | "sse";
    headers: RegistryHeader[];
    needs_key: boolean;
    templated: boolean;
    website: string | null;
    repository: string | null;
    overlaps: string[];
    updated_at: string | null;
}

export interface RegistrySearchResult {
    items: RegistryItem[];
    next_cursor: string | null;
    scanned: number;
    source: string;
    fetched_at: string;
    stale: boolean;
    partial?: boolean;
}

export type RegistryMode = { kind: "q"; q: string } | { kind: "topic"; topic: "astronomy" };

export type RegistryAction =
    | { kind: "connect"; url: string; name: string }
    | { kind: "fill"; input: string; note: string; focusKey: boolean };

export interface RegistryBadge { text: string; tone: "ok" | "info" | "warn" | "muted" }

export const REGISTRY_HOME: string;
export function sameEndpoint(a: string | undefined, b: string | undefined): boolean;
export function registryName(item: RegistryItem): string;
export function registryAction(item: RegistryItem): RegistryAction;
export function headerNeeds(item: RegistryItem): { key: boolean; other: boolean };
export function registryBadges(item: RegistryItem, servers?: { url?: string }[]): RegistryBadge[];
export function overlapNote(item: RegistryItem): string;
export function registrySearchParams(mode: RegistryMode, cursor?: string | null): string;
export function mergeResults(current: RegistryItem[], next: RegistryItem[]): RegistryItem[];
