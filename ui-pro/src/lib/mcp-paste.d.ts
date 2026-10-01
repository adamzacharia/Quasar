export type McpTransport = "streamable_http" | "http" | "stdio";

export interface ParsedMcpUrl {
    kind: "url";
    url: string;
    name?: string;
    headers?: Record<string, string>;
    transport?: McpTransport;
    note?: string;
    /** A header whose pasted value was only a placeholder: ask for the key for it. */
    keyHeader?: string;
    /** Every header whose pasted value was a placeholder (none of them is sent). */
    placeholderHeaders?: string[];
}

export interface ParsedMcpLocal {
    kind: "local";
    command: string;
    message: string;
    alternative?: { name: string; url: string };
}

export type ParsedMcpInput =
    | ParsedMcpUrl
    | ParsedMcpLocal
    | { kind: "empty" }
    | { kind: "invalid"; message: string };

export const HOSTED_ALTERNATIVES: { match: RegExp; name: string; url: string }[];
export function deriveServerName(url: string): string;
export function uniqueServerName(base: string, url: string, servers: { name: string; url?: string }[]): string;
export function hasPlaceholder(value: string): boolean;
export function splitCommand(line: string): string[];
export function stripVersion(pkg: string): string;
export function parseMcpInput(text: string): ParsedMcpInput;
