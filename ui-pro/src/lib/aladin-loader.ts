/* eslint-disable @typescript-eslint/no-explicit-any */

const ALADIN_SCRIPT_ID = "aladin-lite-v3-3-8-2";
const ALADIN_SCRIPT_ORIGIN = "https://aladin.cds.unistra.fr";
const ALADIN_SCRIPT_SRC = "https://aladin.cds.unistra.fr/AladinLite/api/v3/3.8.2/aladin.js";
// The script is fetched from an external CDN whose cold-cache latency is
// occasionally >20s (observed live: the viewer "hung" then worked hours later
// once the browser had cached it). Allow a more generous cold-load window and
// retry once on a hard network error before giving up.
const ALADIN_LOAD_TIMEOUT_MS = 35_000;
const ALADIN_MAX_SCRIPT_ATTEMPTS = 2;

function preconnectAladinOrigin() {
    if (typeof document === "undefined") return;
    if (document.querySelector(`link[data-aladin-preconnect]`)) return;
    for (const rel of ["preconnect", "dns-prefetch"]) {
        const link = document.createElement("link");
        link.rel = rel;
        link.href = ALADIN_SCRIPT_ORIGIN;
        link.crossOrigin = "anonymous";
        link.setAttribute("data-aladin-preconnect", "1");
        document.head.appendChild(link);
    }
}

function clearFailedLoad() {
    if (typeof window !== "undefined") {
        window.__aladinLoader = undefined;
        window.A = undefined;
    }

    if (typeof document === "undefined") return;
    const script = document.getElementById(ALADIN_SCRIPT_ID);
    script?.parentNode?.removeChild(script);
}

function toErrorMessage(error: unknown, fallback: string) {
    return error instanceof Error && error.message ? error.message : fallback;
}

async function initializeAladin(A: any): Promise<any> {
    if (!A) {
        throw new Error("Aladin Lite loaded but did not expose the global A object.");
    }

    const init = A.init;
    if (init && typeof init.then === "function") {
        await init;
        return A;
    }

    if (typeof init === "function") {
        const result = init();
        if (result && typeof result.then === "function") {
            await result;
        }
    }

    return A;
}

export function loadAladin(): Promise<any> {
    if (typeof window === "undefined" || typeof document === "undefined") {
        return Promise.reject(new Error("Aladin Lite can only be loaded in a browser."));
    }

    if (window.__aladinLoader) {
        return window.__aladinLoader;
    }

    window.__aladinLoader = new Promise((resolve, reject) => {
        let settled = false;
        let timeoutId = 0;

        const fail = (message: string) => {
            if (settled) return;
            settled = true;
            window.clearTimeout(timeoutId);
            clearFailedLoad();
            reject(new Error(message));
        };

        const complete = async () => {
            if (settled) return;
            try {
                const A = await initializeAladin(window.A);
                if (settled) return;
                settled = true;
                window.clearTimeout(timeoutId);
                resolve(A);
            } catch (error) {
                fail(`Aladin Lite could not initialize: ${toErrorMessage(error, "WebGL2 may be unavailable.")}`);
            }
        };

        timeoutId = window.setTimeout(() => {
            fail("Aladin Lite load timed out after 20 seconds.");
        }, ALADIN_LOAD_TIMEOUT_MS);

        if (window.A) {
            void complete();
            return;
        }

        preconnectAladinOrigin();

        const injectScript = (attempt: number) => {
            const existing = document.getElementById(ALADIN_SCRIPT_ID);
            existing?.parentNode?.removeChild(existing);
            const script = document.createElement("script");
            script.id = ALADIN_SCRIPT_ID;
            // Cache-bust on retry so a failed fetch is not served from a poisoned cache entry.
            script.src = attempt > 1 ? `${ALADIN_SCRIPT_SRC}?retry=${attempt}` : ALADIN_SCRIPT_SRC;
            script.async = true;
            script.defer = true;
            script.onload = () => {
                void complete();
            };
            script.onerror = () => {
                if (settled) return;
                if (attempt < ALADIN_MAX_SCRIPT_ATTEMPTS) {
                    window.setTimeout(() => {
                        if (!settled) injectScript(attempt + 1);
                    }, 1500 * attempt);
                } else {
                    fail("Aladin Lite could not be downloaded (CDN unreachable after retries).");
                }
            };
            document.head.appendChild(script);
        };

        injectScript(1);
    });

    return window.__aladinLoader;
}

declare global {
    interface Window {
        A?: any;
        __aladinLoader?: Promise<any>;
    }
}