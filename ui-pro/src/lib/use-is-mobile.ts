"use client";

import { useEffect, useState } from "react";

/** The 768px seam shared by Tailwind's `md:` prefix and the globals.css breakpoints. */
const MOBILE_QUERY = "(max-width: 767px)";

/**
 * True on phone-width viewports.
 *
 * The initial value is read synchronously from matchMedia so the first client
 * paint already has the right layout. That makes the hook unsafe to render on
 * the server — mount it only under a client-only gate (see app/page.tsx, which
 * returns null until `mounted`).
 */
export function useIsMobile(): boolean {
    const [isMobile, setIsMobile] = useState(
        () => typeof window !== "undefined" && window.matchMedia(MOBILE_QUERY).matches,
    );

    useEffect(() => {
        const mq = window.matchMedia(MOBILE_QUERY);
        const onChange = (event: MediaQueryListEvent) => setIsMobile(event.matches);
        mq.addEventListener("change", onChange);
        return () => mq.removeEventListener("change", onChange);
    }, []);

    return isMobile;
}
