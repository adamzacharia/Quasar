import type { McpTransport, ParsedMcpInput } from "./mcp-paste.js";

export const OAUTH_MESSAGE_TYPE: "quasar-mcp-oauth";
export const OAUTH_TIMEOUT_MS: number;
export const OAUTH_CLOSED_GRACE_MS: number;

export interface McpServerStatus {
    connected: boolean;
    state?: "connected" | "needs_auth" | "needs_api_key" | "error" | "connecting";
    tools: string[];
    error?: string | null;
    transport?: string | null;
    /** Epoch seconds when the pool settled this connection attempt. */
    checked_at?: number | null;
}

export interface McpOAuthInfo {
    signed_in: boolean;
    signed_in_at?: number | null;
    signed_in_attempt?: string | null;
}

export interface McpServerLike {
    name: string;
    url?: string;
    status?: McpServerStatus;
    oauth?: McpOAuthInfo;
}

export type ConnectStep =
    | { step: "oauth"; authorizeUrl: string; server?: string }
    | { step: "api_key"; message: string }
    | { step: "done"; server?: string; tools: string[] }
    | { step: "error"; message: string; saved?: boolean };

export interface ConnectForm {
    advancedOpen: boolean;
    transport: McpTransport;
    headerText: string;
    envText: string;
    cmd: string;
    cmdArgs: string;
    apiKey: string;
    keyHeader?: string;
    name: string;
    nameTyped: boolean;
}

export interface ConnectBody {
    name: string;
    transport: McpTransport;
    url: string | null;
    command: string | null;
    args: string[];
    env: Record<string, string>;
    headers: Record<string, string>;
    /** false: the backend answers 409 instead of updating a different server with this name. */
    replace: boolean;
}

export function originOf(url: string): string;
export function isOAuthResultMessage(event: { origin: string; data: unknown } | null | undefined, apiBase: string): boolean;
export function readOAuthReturn(search: string): { status: "connected" | "signed_in" | "failed"; server: string } | null;
export function stripOAuthReturn(href: string): string | null;
export function returnBanner(ret: { status: string; server: string } | null, servers: McpServerLike[]):
    { type: "success" | "warn" | "error"; text: string } | null;
export function oauthWaitDecision(args: {
    startedAt: number;
    now: number;
    popupClosedAt?: number | null;
    serverStatus?: McpServerStatus | null;
    oauth?: McpOAuthInfo | null;
    baselineSignedInAt?: number | null;
    attempt?: string | null;
    signedInSeenAt?: number | null;
    timeoutMs?: number;
    closedGraceMs?: number;
}): { done: true; ok: boolean; error?: string } | { done: false; popupClosed: boolean; signedIn?: boolean };
export const SIGNED_IN_SETTLE_MS: number;
export function createOnce<A extends unknown[]>(onFinish: (...args: A) => void): { readonly done: boolean; finish: (...args: A) => boolean; cancel: () => void };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function nextStep(body: any): ConnectStep;
export function statusView(status: McpServerStatus | null | undefined, auth?: string): { tone: "ok" | "warn" | "err" | "idle"; label: string };
export function bearerHeader(key: string): string | null;
export function parsePairs(text: string, sep: "=" | ":"): Record<string, string>;
export function credentialsStillApply(prevInput: string, nextInput: string): boolean;
export function buildConnectBody(args: {
    preset?: { url: string; name: string; transport?: McpTransport } | null;
    parsed: ParsedMcpInput;
    form: ConnectForm;
    servers?: { name: string; url?: string }[];
}): { body: ConnectBody; error?: undefined } | { error: string; body?: undefined };
