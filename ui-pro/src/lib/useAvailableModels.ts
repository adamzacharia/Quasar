/**
 * Live provider-model catalog for the model selector.
 *
 * Backed by GET /api/models/available. The cache is a small zustand store
 * rather than component state so the Provider Keys panel can invalidate it
 * after a key is saved, rotated or removed and have the sidebar dropdown
 * update without a reload — zustand is already the app's state library.
 */
import { useEffect } from "react";
import { create } from "zustand";

import { authBearerHeaders } from "./auth-store";
import { DEFAULT_AVAILABLE_MODELS, isTaccModel } from "./models";
import type {
    AvailableModels,
    ModelInfo,
    ProviderCatalog,
    ProviderId,
} from "./model-catalog";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Offline shape of the previously hardcoded list, used only when the catalog
 * request fails outright — the dropdown must never render empty.
 */
function fallbackCatalog(): AvailableModels {
    const byProvider = new Map<ProviderId, ModelInfo[]>();
    for (const id of DEFAULT_AVAILABLE_MODELS) {
        const provider: ProviderId = isTaccModel(id)
            ? "tacc"
            : id.toLowerCase().includes("deepseek")
              ? "deepseek"
              : "openai";
        const models = byProvider.get(provider) ?? [];
        models.push({
            provider,
            id,
            displayName: id,
            capabilities: {},
            source: "static",
        });
        byProvider.set(provider, models);
    }
    const providers: ProviderCatalog[] = [...byProvider.entries()].map(
        ([provider, models]) => ({
            provider,
            status: "included_quota",
            models,
            stale: true,
            unlockCount: 0,
        }),
    );
    return { providers, defaultModel: "gpt-oss-120b" };
}

interface CatalogState {
    data: AvailableModels | null;
    loading: boolean;
    /** Null while healthy; a user-facing message when the fetch failed. */
    error: string | null;
    loaded: boolean;
    load: (force?: boolean) => Promise<void>;
    /** Force a provider-side refetch from the provider's own API. */
    refreshProvider: (provider: string) => Promise<ProviderCatalog>;
}

let inFlight: Promise<void> | null = null;

export const useCatalogStore = create<CatalogState>((set, get) => ({
    data: null,
    loading: false,
    error: null,
    loaded: false,

    load: async (force = false) => {
        if (!force && (get().loaded || get().loading)) {
            if (inFlight) await inFlight;
            return;
        }
        if (inFlight && !force) {
            await inFlight;
            return;
        }
        set({ loading: true });
        const request = (async () => {
            try {
                const res = await fetch(`${API_BASE}/api/models/available`, {
                    credentials: "include",
                    headers: authBearerHeaders(),
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = (await res.json()) as AvailableModels;
                set({ data, loading: false, error: null, loaded: true });
            } catch {
                // Keep whatever we already had; only seed the fallback once.
                set((state) => ({
                    data: state.data ?? fallbackCatalog(),
                    loading: false,
                    loaded: true,
                    error: "Could not load the model catalog.",
                }));
            } finally {
                inFlight = null;
            }
        })();
        inFlight = request;
        await request;
    },

    refreshProvider: async (provider: string) => {
        const res = await fetch(
            `${API_BASE}/api/providers/${encodeURIComponent(provider)}/refresh-models`,
            {
                method: "POST",
                credentials: "include",
                headers: authBearerHeaders(),
            },
        );
        if (!res.ok) {
            let detail = `Could not refresh ${provider} models.`;
            try {
                const body = (await res.json()) as { detail?: string };
                if (body && typeof body.detail === "string") detail = body.detail;
            } catch {
                /* non-JSON error body */
            }
            throw new Error(detail);
        }
        const catalog = (await res.json()) as ProviderCatalog;
        set((state) => {
            if (!state.data) return state;
            const providers = state.data.providers.map((entry) =>
                entry.provider === catalog.provider ? catalog : entry,
            );
            return { data: { ...state.data, providers } };
        });
        return catalog;
    },
}));

/** Invalidate after a provider key is saved, rotated or removed. */
export function invalidateAvailableModels(): Promise<void> {
    return useCatalogStore.getState().load(true);
}

export interface UseAvailableModels {
    providers: ProviderCatalog[];
    defaultModel: string;
    loading: boolean;
    error: string | null;
    reload: () => Promise<void>;
    refreshProvider: (provider: string) => Promise<ProviderCatalog>;
}

/**
 * The selector's data source. Fetches once per session and then serves from
 * the shared store; call `reload()` (or `invalidateAvailableModels()`) to
 * refetch after a provider key changes.
 */
export function useAvailableModels(): UseAvailableModels {
    const data = useCatalogStore((state) => state.data);
    const loading = useCatalogStore((state) => state.loading);
    const error = useCatalogStore((state) => state.error);
    const load = useCatalogStore((state) => state.load);
    const refreshProvider = useCatalogStore((state) => state.refreshProvider);

    useEffect(() => {
        void load();
    }, [load]);

    return {
        providers: data?.providers ?? [],
        defaultModel: data?.defaultModel ?? "gpt-oss-120b",
        loading,
        error,
        reload: () => load(true),
        refreshProvider,
    };
}
