"use client";

// Monotonic auth "generation". Every state-changing auth event (login, logout,
// local sign-out) bumps it. Async work that resolves later — a bootstrap
// /api/auth/me, or a conversation-list fetch — captures the generation before
// it starts and discards its result if the generation changed meanwhile, so a
// slow response from account A can never resurrect or clobber account B.
//
// It lives in this leaf module (imported by both the auth store and the chat
// store) purely to avoid an import cycle between those two stores.
let authGeneration = 0;

export function bumpAuthGeneration(): number {
    authGeneration += 1;
    return authGeneration;
}

export function currentAuthGeneration(): number {
    return authGeneration;
}
