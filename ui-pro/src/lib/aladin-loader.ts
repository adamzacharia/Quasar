/* eslint-disable @typescript-eslint/no-explicit-any */

const ALADIN_SCRIPT_ID = "aladin-lite-v3-3-8-2";
const ALADIN_SCRIPT_SRC = "https://aladin.cds.unistra.fr/AladinLite/api/v3/3.8.2/aladin.js";
const ALADIN_LOAD_TIMEOUT_MS = 20_000;

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

        let script = document.getElementById(ALADIN_SCRIPT_ID) as HTMLScriptElement | null;
        if (!script) {
            script = document.createElement("script");
            script.id = ALADIN_SCRIPT_ID;
            script.src = ALADIN_SCRIPT_SRC;
            script.async = true;
            script.defer = true;
            document.head.appendChild(script);
        }

        script.onload = () => {
            void complete();
        };
        script.onerror = () => {
            fail("Aladin Lite could not be downloaded.");
        };
    });

    return window.__aladinLoader;
}

declare global {
    interface Window {
        A?: any;
        __aladinLoader?: Promise<any>;
    }
}