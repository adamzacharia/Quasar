/**
 * Quasar Custom SVG Icons
 * 
 * Stroke-based 16×16 SVGs matching the project's primary pink/accent design language.
 * Inspired by the astrophysics icon set in TaskExecutionWidget.
 * Use these instead of generic Lucide icons for brand-critical placements.
 */

import React from "react";

/**
 * Neural network / brain icon — abstract constellation-style neurons with connections.
 * Used for Thinking/Reasoning UI states.
 */
export function IconNeuralNet({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            {/* Central node */}
            <circle cx="8" cy="8" r="1.6" fill="currentColor" stroke="none" />
            {/* Satellite nodes */}
            <circle cx="3" cy="4" r="1" fill="currentColor" stroke="none" />
            <circle cx="13" cy="4" r="1" fill="currentColor" stroke="none" />
            <circle cx="4" cy="13" r="1" fill="currentColor" stroke="none" />
            <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
            {/* Connections */}
            <line x1="8" y1="8" x2="3" y2="4" opacity="0.6" />
            <line x1="8" y1="8" x2="13" y2="4" opacity="0.6" />
            <line x1="8" y1="8" x2="4" y2="13" opacity="0.6" />
            <line x1="8" y1="8" x2="12" y2="12" opacity="0.6" />
            {/* Cross-connections */}
            <line x1="3" y1="4" x2="4" y2="13" strokeWidth="0.8" opacity="0.3" />
            <line x1="13" y1="4" x2="12" y2="12" strokeWidth="0.8" opacity="0.3" />
            {/* Pulse ring */}
            <circle cx="8" cy="8" r="4" strokeWidth="0.6" strokeDasharray="2 2" opacity="0.35" />
        </svg>
    );
}

/**
 * Open book icon — astrophysics journal / documentation style.
 * Clean book with a star accent for the scientific context.
 */
export function IconOpenBook({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            {/* Spine */}
            <line x1="8" y1="3" x2="8" y2="14" />
            {/* Left page */}
            <path d="M8 3 Q5 2 2 3 L2 13 Q5 12 8 14" />
            {/* Right page */}
            <path d="M8 3 Q11 2 14 3 L14 13 Q11 12 8 14" />
            {/* Left text lines */}
            <line x1="4" y1="6" x2="6.5" y2="6" strokeWidth="1" opacity="0.5" />
            <line x1="4" y1="8" x2="6" y2="8" strokeWidth="1" opacity="0.5" />
            <line x1="4" y1="10" x2="6.5" y2="10" strokeWidth="1" opacity="0.5" />
            {/* Right-page star marker */}
            <path d="M11 7 l0.4 0.9 1 0.15 -0.7 0.7 0.15 1 -0.85-0.5 -0.85 0.5 0.15-1 -0.7-0.7 1-0.15z" fill="currentColor" stroke="none" opacity="0.7" />
        </svg>
    );
}

/**
 * Web/network globe — clean globe with signal arcs.
 * Used for web search and external resource indicators.
 */
export function IconWebGlobe({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
            {/* Globe */}
            <circle cx="8" cy="8" r="6" />
            {/* Latitude lines */}
            <path d="M2.5 8 Q8 5.5 13.5 8" />
            <path d="M2.5 8 Q8 10.5 13.5 8" />
            {/* Meridian */}
            <path d="M8 2 Q11 8 8 14" />
            <path d="M8 2 Q5 8 8 14" />
            {/* Signal arcs in top-right */}
            <path d="M11 3 Q12.5 4 12 5.5" strokeWidth="1" opacity="0.5" />
            <path d="M12.5 2 Q14.5 3.5 13.5 5.5" strokeWidth="0.8" opacity="0.35" />
        </svg>
    );
}

/**
 * Link/connection icon — two chain links with a spark.
 * Used for external links and connected resources.
 */
export function IconChainLink({ className = "" }: { className?: string }) {
    return (
        <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            {/* Left link */}
            <path d="M6.5 9.5 L4 12 Q2 14 3 15 L4 14 Q2.5 12.5 4.5 10.5 L7 8" />
            <path d="M4 7 L7 4 Q9 2 10 3" />
            {/* Right link */}
            <path d="M9.5 6.5 L12 4 Q14 2 13 1 L12 2 Q13.5 3.5 11.5 5.5 L9 8" />
            {/* Central connection */}
            <line x1="6" y1="10" x2="10" y2="6" />
            {/* Spark dot */}
            <circle cx="8" cy="8" r="0.8" fill="currentColor" stroke="none" opacity="0.7" />
        </svg>
    );
}
