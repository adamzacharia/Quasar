// lib/theme-store.ts
// Zustand store for dark/light theme management.
// Persists choice in localStorage and applies data-theme attribute to <html>.

import { create } from "zustand";

type Theme = "dark" | "light";

interface ThemeStore {
    theme: Theme;
    toggle: () => void;
    setTheme: (t: Theme) => void;
}

function getInitialTheme(): Theme {
    if (typeof window === "undefined") return "dark";
    const saved = localStorage.getItem("quasar_theme");
    if (saved === "light" || saved === "dark") return saved;
    // Respect system preference
    if (window.matchMedia?.("(prefers-color-scheme: light)").matches) return "light";
    return "dark";
}

function applyTheme(theme: Theme) {
    if (typeof document === "undefined") return;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("quasar_theme", theme);
}

export const useThemeStore = create<ThemeStore>((set) => ({
    theme: getInitialTheme(),
    toggle: () =>
        set((state) => {
            const next = state.theme === "dark" ? "light" : "dark";
            applyTheme(next);
            return { theme: next };
        }),
    setTheme: (t: Theme) => {
        applyTheme(t);
        set({ theme: t });
    },
}));
