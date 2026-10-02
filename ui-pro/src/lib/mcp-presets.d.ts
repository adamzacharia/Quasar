export interface McpPreset {
    id: string;
    name: string;
    url: string;
    domain: string;
    auth: "none" | "oauth";
    verified: "tool-call" | "tools-listed" | "oauth-discovery";
    blurb: string;
    needsRepo?: boolean;
    /** The publisher is the data provider itself (e.g. NASA). */
    official?: boolean;
    /** A built-in Quasar tool that already covers part of this server. */
    builtinOverlap?: string;
}

export const PRESETS_VERIFIED_ON: string;
export const MCP_PRESETS: McpPreset[];
export const ASTRONOMY_PRESETS_VERIFIED_ON: string;
export const ASTRONOMY_PRESETS: McpPreset[];
export function apiKeyPageFor(url: string): string | null;
export function gitmcpUrl(input: string): string | null;
export function presetForUrl(url: string): McpPreset | undefined;
export function iconDomain(url: string): string;
