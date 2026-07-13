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
    /**
     * INTERIM Bearer revert (2026-07-13): the JWT is persisted and sent as an
     * Authorization header again because the production topology is cross-site
     * (quasarassistant.com → quasar-oi14.onrender.com) and browsers that block
     * third-party cookies silently drop the S6 httpOnly auth cookie, logging
     * users out on their first request. Remove this field (and the Bearer
     * plumbing that reads it) once the API moves to api.quasarassistant.com
     * and the cookie becomes first-party.
     */
    token: string | null;
    isAuthenticated: boolean;
    /** False until the httpOnly-cookie session has been verified via /api/auth/me. */
    isInitialized: boolean;
    isAuthModalOpen: boolean;
    setAuth: (user: User, token?: string | null) => void;
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
const sessionScrubbers: Array<() => void> = [];
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
            token: null,
            isAuthenticated: false,
            isInitialized: false,
            isAuthModalOpen: false,
            setAuth: (user, token) => {
                bumpAuthGeneration();
                set({
                    user,
                    token: token ?? null,
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
                    token: null,
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
                    token: null,
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
            // v1 stripped the legacy persisted JWT (S6 cookie migration).
            // v2 persists it again (INTERIM Bearer revert — see the `token`
            // field comment): third-party-cookie blocking breaks the
            // cross-site cookie in production, so auth rides the
            // Authorization header until the API is same-site.
            version: 2,
            migrate: (persisted: unknown) => {
                // v0 states still carry their pre-S6 token — keep it (if the
                // JWT secret rotated it will fail /me and clear cleanly).
                // v1 states simply have no token; nothing to transform.
                const state = (persisted ?? {}) as Record<string, unknown>;
                return state as never;
            },
            partialize: (state) => ({ user: state.user, token: state.token }),
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
    // onRehydrateStorage fires this DURING the create() call above, so nothing
    // here may touch useAuthStore synchronously (TDZ crash — blank app). Yield
    // one microtask first: module evaluation finishes, the store binding
    // exists, and only then do we read the token.
    await Promise.resolve();
    try {
        const res = await fetch(`${API_BASE}/api/auth/me`, {
            credentials: "include",
            cache: "no-store",
            // INTERIM Bearer revert: browsers that block third-party cookies
            // never present the httpOnly cookie cross-site, so the persisted
            // token is the credential that actually authenticates.
            headers: authBearerHeaders(),
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
                    token: null,
                    isAuthenticated: false,
                    isInitialized: true,
                });
            }
        } else if (res.status === 401) {
            // No/expired credential. The page-level gate decides whether to
            // prompt (home opens the modal; deep pages show their own gate).
            useAuthStore.setState({
                user: null,
                token: null,
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

/**
 * INTERIM Bearer revert (2026-07-13) — read the persisted JWT.
 *
 * The S6 httpOnly cookie is dropped by third-party-cookie blocking in the
 * cross-site production topology (quasarassistant.com → onrender.com), so
 * every backend call attaches the token as an Authorization header again.
 * The backend has always accepted Bearer-or-cookie, so no server change is
 * needed. Delete this helper (and its call sites) when the API moves to
 * api.quasarassistant.com and the cookie becomes first-party.
 */
export function getStoredToken(): string | null {
    return useAuthStore.getState().token;
}

// ── Cross-tab session sync (INTERIM Bearer revert) ───────────────────────────
// The S6 cookie was browser-wide state: signing out in one tab made every
// other tab 401 on its next request. A Bearer token lives per-tab in zustand
// memory, so mirror auth changes through the persisted key instead:
//  - a NEW-bundle tab (v2 write) signs out / token gone → sign out here too;
//  - a NEW-bundle tab signs in with a different token → adopt that session;
//  - a STALE pre-revert tab (v1 write, token stripped by its partialize)
//    clobbers the key during the deploy-overlap window → re-persist our live
//    state so the fresh token survives until stale tabs age out.
// Guarded on token difference so two v2 tabs can never ping-pong writes.
if (typeof window !== "undefined") {
    window.addEventListener("storage", (event) => {
        if (event.key !== "quasar-auth") return;
        try {
            const current = useAuthStore.getState();
            const incoming = event.newValue ? JSON.parse(event.newValue) : null;
            const version = incoming?.version;
            const state = (incoming?.state ?? {}) as {
                token?: string | null;
                user?: User | null;
            };
            if (incoming && typeof version === "number" && version < 2) {
                // Stale-bundle clobber: heal by re-persisting (any setState
                // makes the persist middleware rewrite the key at v2).
                if (current.token) useAuthStore.setState({});
                return;
            }
            const incomingToken = state.token ?? null;
            if (!incomingToken && current.isAuthenticated) {
                current.clearAuth();
            } else if (
                incomingToken &&
                incomingToken !== current.token &&
                state.user &&
                state.user.id
            ) {
                bumpAuthGeneration();
                useAuthStore.setState({
                    user: state.user,
                    token: incomingToken,
                    isAuthenticated: true,
                    isInitialized: true,
                    isAuthModalOpen: false,
                });
            }
        } catch {
            /* a malformed storage write must never break the app */
        }
    });
}

/** Merge an Authorization: Bearer header (when signed in) into `extra`. */
export function authBearerHeaders(
    extra: Record<string, string> = {},
): Record<string, string> {
    const token = getStoredToken();
    return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}
