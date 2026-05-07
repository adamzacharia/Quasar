"use client";

import { useEffect } from "react";
import { useThemeStore } from "../lib/theme-store";

/**
 * Client component that syncs the Zustand theme store with the DOM
 * on hydration. The inline <script> in layout.tsx handles the initial
 * flash-free theme set; this component keeps the store in sync.
 */
export function ThemeInitializer() {
    const setTheme = useThemeStore((s) => s.setTheme);

    useEffect(() => {
        const saved = localStorage.getItem("quasar_theme");
        if (saved === "light" || saved === "dark") {
            setTheme(saved);
        }
    }, [setTheme]);

    return null;
}
