"use client";

import { Menu, Moon, Sun } from "lucide-react";
import { useChatStore } from "../lib/store";
import { useThemeStore } from "../lib/theme-store";

/**
 * The phone-only top bar on the empty home screen: drawer trigger, wordmark,
 * theme toggle. Once a conversation starts, ChatArea swaps back to the regular
 * chat header (title + star).
 */
export function MobileHomeBar() {
    const toggleSidebar = useChatStore((s) => s.toggleSidebar);
    const theme = useThemeStore((s) => s.theme);
    const toggleTheme = useThemeStore((s) => s.toggle);

    const isDark = theme === "dark";

    return (
        <header className="shrink-0 flex items-center justify-between px-4 pt-2.5 pb-1">
            {/* Sizes are pinned in px, not rem: the root font-size drops to 14px
                below the md breakpoint, which would shrink rem-based controls
                (size-10 -> 35px) below the design's touch targets. */}
            <button
                type="button"
                onClick={toggleSidebar}
                aria-label="Open navigation"
                className="glass-control flex size-[40px] items-center justify-center rounded-full"
                style={{ color: "var(--q-text-secondary)" }}
            >
                <Menu className="size-[19px]" strokeWidth={1.8} />
            </button>

            <div className="flex items-center gap-2">
                <img src="/quasar_logo.png" alt="" className="size-[26px] object-contain" />
                <span className="text-[15px] font-bold tracking-[0.01em]" style={{ color: "var(--q-text)" }}>
                    QUASAR
                </span>
            </div>

            <button
                type="button"
                onClick={toggleTheme}
                aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
                className="glass-control flex size-[40px] items-center justify-center rounded-full"
            >
                {isDark ? (
                    <Sun className="size-[17px] text-[#f3d27a]" strokeWidth={1.8} />
                ) : (
                    <Moon className="size-[16px] text-[#8b5cf6]" strokeWidth={1.8} />
                )}
            </button>
        </header>
    );
}
