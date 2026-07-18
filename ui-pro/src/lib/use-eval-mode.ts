"use client";

/**
 * Whether eval mode is actually in force (Feature 4).
 *
 * Eval mode is admin tooling for building the benchmark label set, so the
 * stored per-user flag is necessary but NOT sufficient: the build must enable
 * it and the signed-in user must be an admin.
 *
 * This lives in one place on purpose. The flag is persisted in localStorage,
 * which is per-BROWSER, not per-user — so checking `is_admin` only where the
 * Settings toggle renders would let a value left behind by an admin session
 * gate the next user on a shared machine, and a stale value would survive into
 * a production build that meant to compile eval mode out entirely. Both are the
 * exact opposite of "admin-only".
 */

import { useAuthStore } from "./auth-store";
import { useChatStore } from "./store";

/** Compiled out of any build without the flag — NEXT_PUBLIC_* is inlined. */
export const EVAL_MODE_ENABLED = process.env.NEXT_PUBLIC_ENABLE_EVAL_MODE === "1";

export function useEvalModeActive(): boolean {
    const stored = useChatStore((s) => s.evalMode);
    const isAdmin = useAuthStore((s) => Boolean(s.user?.is_admin));
    return EVAL_MODE_ENABLED && isAdmin && stored;
}
