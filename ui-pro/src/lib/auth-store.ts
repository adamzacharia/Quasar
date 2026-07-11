"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { bumpAuthGeneration, currentAuthGeneration } from "./auth-generation";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface User {
    id: string;
    username: string;
    display_name: string;
    // Optional: /api/auth/me does not return these, so a cookie-only bootstrap
    // (cleared localStorage) legitimately lacks them. Never default them.
    auth_provider?: "local" | "google";
    picture_url?: string;
    is_admin?: boolean;
}

interface AuthStore {
    user: User | null;
    isAuthenticated: boolean;
    /** False until the httpOnly-cookie session has been verified via /api/auth/me. */
    isInitialized: boolean;
    isAuthModalOpen: boolean;
    setAuth: (user: User) => void;
    /** Local sign-out + best-effort server cookie clear. */
    logout: () => void;
    /** Local-only sign-out (no server call) — for 401s discovered mid-session. */
    clearAuth: () => void;
    setInitialized: (initialized: boolean) => void;
    openAuthModal: () => void;
    closeAuthModal: () => void;
}

// Per-user app state (conversations, workbench, etc.) must not leak across an
// account change on the same browser. Registered by the chat store so this
// store can scrub it on sign-out without an import cycle.
let sessionScrubbers: Array<() => void> = [];
export function registerSessionScrubber(fn: () => void): void {
    sessionScrubbers.push(fn);
}
function scrubSession(): void {
    for (const fn of sessionScrubbers) {
        try {
            fn();
        } catch {
            /* a scrubber must never block sign-out */
        }
    }
}

export const useAuthStore = create<AuthStore>()(
    persist(
        (set) => ({
            user: null,
            isAuthenticated: false,
            isInitialized: false,
            isAuthModalOpen: false,
            setAuth: (user) => {
                bumpAuthGeneration();
                set({
                    user,
                    isAuthenticated: true,
                    isInitialized: true,
                    isAuthModalOpen: false,
                });
            },
            logout: () => {
                bumpAuthGeneration();
                scrubSession();
                // Best-effort server-side cookie clear; local state clears regardless.
                // `keepalive` lets it complete even if the user navigates away.
                fetch(`${API_BASE}/api/auth/logout`, {
                    method: "POST",
                    credentials: "include",
                    keepalive: true,
                }).catch(() => undefined);
                // isInitialized must stay true: the terminal branches of an
                // in-flight verifyAuth() short-circuit on the generation bump,
                // so without this the neutral gate could hang forever. Open the
                // auth modal atomically so the signed-out UI is immediate.
                set({
                    user: null,
                    isAuthenticated: false,
                    isInitialized: true,
                    isAuthModalOpen: true,
                });
            },
            clearAuth: () => {
                bumpAuthGeneration();
                scrubSession();
                // Local-only sign-out (e.g. a 401 discovered mid-session): drop
                // the identity, keep isInitialized, prompt re-auth.
                set({
                    user: null,
                    isAuthenticated: false,
                    isInitialized: true,
                    isAuthModalOpen: true,
                });
            },
            setInitialized: (initialized) => set({ isInitialized: initialized }),
            openAuthModal: () => set({ isAuthModalOpen: true }),
            closeAuthModal: () => set({ isAuthModalOpen: false }),
        }),
        {
            name: "quasar-auth",
            // v1 strips the legacy persisted JWT (S6: tokens never touch
            // localStorage again — auth lives in the httpOnly cookie).
            version: 1,
            migrate: (persisted: unknown) => {
                const state = (persisted ?? {}) as Record<string, unknown>;
                delete state.token;
                return state as never;
            },
            partialize: (state) => ({ user: state.user }),
            onRehydrateStorage: () => () => {
                // Server-authoritative: the persisted user is display data only.
                // isAuthenticated/isInitialized stay false until /api/auth/me
                // confirms the cookie; the UI shows a neutral gate meanwhile.
                void verifyAuth();
            },
        }
    )
);

/**
 * Verify the session against the httpOnly auth cookie (server-authoritative).
 *
 * 200 → merge the confirmed identity over the persisted user (fields /me does
 *        not return — auth_provider, picture_url — are preserved, never
 *        defaulted) and mark authenticated + initialized atomically.
 * 401 → no/expired cookie (e.g. a legacy pre-cookie login): clear user state
 *        and mark initialized so the UI can offer sign-in.
 * Network error → mark initialized but keep state (offline must not log the
 *        user out); API calls will surface their own errors.
 *
 * Every terminal branch sets all four auth fields (user / isAuthenticated /
 * isInitialized / isAuthModalOpen) atomically, per the plan-gate ruling, so no
 * intermediate half-decided state is ever observable.
 */
export async function verifyAuth(): Promise<void> {
    const generation = currentAuthGeneration();
    try {
        const res = await fetch(`${API_BASE}/api/auth/me`, {
            credentials: "include",
            cache: "no-store",
        });
        if (generation !== currentAuthGeneration()) return; // superseded by login/logout
        if (res.ok) {
            const data = await res.json().catch(() => null);
            const u = data?.user;
            if (generation !== currentAuthGeneration()) return;
            if (u && u.id) {
                useAuthStore.setState((s) => ({
                    user: {
                        ...(s.user ?? {}),
                        id: String(u.id),
                        username: u.username ?? s.user?.username ?? "",
                        display_name: u.display_name ?? s.user?.display_name ?? "",
                        is_admin: Boolean(u.is_admin),
                    },
                    isAuthenticated: true,
                    isInitialized: true,
                    isAuthModalOpen: false,
                }));
            } else {
                // Malformed success — treat as signed out. Leave the modal to
                // the page-level gate (cold-start home vs. an in-app page).
                useAuthStore.setState({
                    user: null,
                    isAuthenticated: false,
                    isInitialized: true,
                });
            }
        } else if (res.status === 401) {
            // No/expired cookie. The page-level gate decides whether to prompt
            // (home opens the modal; deep pages show their own sign-in gate).
            useAuthStore.setState({
                user: null,
                isAuthenticated: false,
                isInitialized: true,
            });
        } else {
            useAuthStore.setState({ isInitialized: true });
        }
    } catch {
        if (generation !== currentAuthGeneration()) return;
        useAuthStore.setState({ isInitialized: true });
    }
}
