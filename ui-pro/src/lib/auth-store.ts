"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface User {
    id: string;
    username: string;
    display_name: string;
    auth_provider: "local" | "google";
}

interface AuthStore {
    user: User | null;
    token: string | null;
    isAuthenticated: boolean;
    isInitialized: boolean;
    isAuthModalOpen: boolean;
    setAuth: (user: User, token: string) => void;
    logout: () => void;
    setInitialized: (initialized: boolean) => void;
    openAuthModal: () => void;
    closeAuthModal: () => void;
}

export const useAuthStore = create<AuthStore>()(
    persist(
        (set) => ({
            user: null,
            token: null,
            isAuthenticated: false,
            isInitialized: false,
            isAuthModalOpen: false,
            setAuth: (user, token) =>
                set({ user, token, isAuthenticated: true, isAuthModalOpen: false }),
            logout: () => set({ user: null, token: null, isAuthenticated: false }),
            setInitialized: (initialized) => set({ isInitialized: initialized }),
            openAuthModal: () => set({ isAuthModalOpen: true }),
            closeAuthModal: () => set({ isAuthModalOpen: false }),
        }),
        {
            name: "quasar-auth",
            partialize: (state) => ({ token: state.token, user: state.user }), // Persist token + user
        }
    )
);
