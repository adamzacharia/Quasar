export interface McpPreset {
    id: string;
    name: string;
    url: string;
    domain: string;
    auth: "none" | "oauth";
    verified: "tool-call" | "tools-listed" | "oauth-discovery";
    blurb: string;
    needsRepo?: boolean;
}

export const PRESETS_VERIFIED_ON: string;
export const MCP_PRESETS: McpPreset[];
export function apiKeyPageFor(url: string): string | null;
export function gitmcpUrl(input: string): string | null;
export function presetForUrl(url: string): McpPreset | undefined;
export function iconDomain(url: string): string;
