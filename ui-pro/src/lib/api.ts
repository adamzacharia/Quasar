import { DEFAULT_AVAILABLE_MODELS, mergeAvailableModels } from "./models";
import { splitProviderChunk } from "./feedback-report";
import { getStoredToken } from "./auth-store";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * INTERIM Bearer revert (2026-07-13): every helper below defaults to the
 * persisted JWT when the caller passes no explicit token, because the S6
 * httpOnly cookie is dropped by third-party-cookie blocking in the cross-site
 * production topology (quasarassistant.com → onrender.com). The explicit
 * `token` parameters are kept so legacy callers keep working. Remove the
 * getStoredToken() fallbacks once the API is same-site
 * (api.quasarassistant.com) and auth can ride the cookie alone again.
 */
function bearerOnlyHeaders(token?: string | null): Record<string, string> {
    const t = token || getStoredToken();
    return t ? { Authorization: `Bearer ${t}` } : {};
}

export interface ChatRequest {
    message: string;
    conversation_id?: string;
    model?: string;
    attachments?: File[];
    token?: string;  // auth token for personal RAG
    grounded_summary?: boolean;
    web_search?: boolean;
}

export interface ChatRunMeta {
    run_id: string;
    trace_id: string;
    model: string;
    provider: string;
    conversation_id: string;
    inactivity_timeout_seconds?: number;
    turn_timeout_seconds?: number;
    /** Stable id of this turn's text block (Feature 4). The assistant message
     *  is created before any SSE arrives, so run_meta is where it learns its
     *  identity. Also persisted in rich_meta.runMeta, so replay agrees. */
    text_block_id?: string;
    /** Set when the turn ended in an infrastructure error — such turns are
     *  exempt from the eval-mode rating gate. */
    status?: string;
    errorCode?: string;
}

/** The exact request one tool call made. `text` is the copyable literal for
 *  every kind — the UI never needs per-archive rendering. Built (and secret-
 *  redacted) server-side in core/provenance.py. */
export interface ToolRequest {
    kind: "adql" | "http" | "ads" | "params" | "args";
    text: string;
    service?: string;
    endpoint?: string;
    url?: string;
    method?: string;
    q?: string;
    rows?: number;
    snippet?: string;
    params?: Record<string, unknown>;
    args?: Record<string, unknown>;
}

/** One entry of the `tool_trace` event / persisted `rich_meta.toolTrace`. */
export interface ToolTraceCall {
    name: string;
    ok?: boolean;
    sql?: string;
    rowcount?: number;
    request?: ToolRequest;
    arguments?: Record<string, unknown>;
}

export interface StreamCallbacks {
    onToken: (token: string) => void;
    onThought?: (thought: string) => void;
    onToolCall?: (toolName: string, input: string, request?: ToolRequest) => void;
    onToolTrace?: (calls: ToolTraceCall[]) => void;
    onData?: (data: Record<string, unknown>) => void;
    /** `blockId` is the card's stable identity (Feature 4) — the store uses it
     *  as the message id so a reload rehydrates the same rating. */
    onPapers?: (papers: Record<string, unknown>[], request?: ToolRequest, blockId?: string) => void;
    onNotebook?: (notebook: Record<string, unknown>) => void;
    onImage?: (image: { url: string; caption: string; meta?: unknown; request?: ToolRequest; blockId?: string }) => void;
    onPlotly?: (plot: { spec?: { data?: unknown[]; layout?: Record<string, unknown> } | null; title?: string; png_fallback?: string; meta?: Record<string, unknown>; request?: ToolRequest; blockId?: string }) => void;
    onStatus?: (step: string, state: string) => void;
    onTaskGroup?: (group: Record<string, unknown>) => void;
    onTaskUpdate?: (update: Record<string, unknown>) => void;
    onTaskList?: (list: Record<string, unknown>) => void;
    onPlanReview?: (plan: {
        conversationId: string;
        title: string;
        subtasks: { id: string; description: string; agentType: string; dependsOn: string[] }[];
        reasoning: string;
        iteration: number;
        maxIterations: number;
        query: string;
    }) => void;
    onWebSources?: (data: {
        sources: { title: string; url: string; snippet: string; evidenceQuality?: Record<string, unknown>; evidence_quality?: Record<string, unknown> }[];
        images: { url: string; description: string; sourceUrl?: string; sourceTitle?: string }[];
        query: string;
        provider?: string;
        image_provider?: string;
        search_type?: string;
    }) => void;
    onConversationMeta?: (meta: { conversation_id: string }) => void;
    onRunMeta?: (meta: ChatRunMeta) => void;
    onUsage?: (usage: {
        inputTokens: number;
        outputTokens: number;
        totalTokens: number;
        durationMs?: number;
        /** Estimated USD for this turn. null when nothing in the turn could be
         *  priced (TACC / self-hosted) — that is "unknown", not "free", so do
         *  not render null as $0.00. */
        costUsd?: number | null;
        /** Tokens excluded from costUsd because their model has no known price. */
        unpricedTokens?: number;
        costIsEstimate?: boolean;
    }) => void;
    onDownloadProgress?: (data: { filename: string; downloaded_bytes: number; total_bytes: number | null; speed_kbps: number; percent: number | null; eta_seconds?: number | null; phase?: string }) => void;
    onComplete: (fullResponse: string) => void;
    /** UI-03: the stream ended WITHOUT [DONE] or an error event (backend
     *  crash, proxy cut). The partial text is passed so the caller can keep
     *  it but must mark the turn degraded, never render it as complete. */
    onIncomplete?: (partialResponse: string) => void;
    onError: (error: string, status?: number) => void;
}

async function getErrorMessage(response: Response): Promise<string> {
    const fallback = `API error: ${response.status}`;
    const contentType = response.headers.get("content-type") || "";

    try {
        if (contentType.includes("application/json")) {
            const payload = await response.json();
            return payload.detail || payload.message || payload.error || fallback;
        }
        const text = (await response.text()).trim();
        return text || fallback;
    } catch {
        return fallback;
    }
}

export class ApiError extends Error {
    status: number;

    constructor(message: string, status: number) {
        super(message);
        this.name = "ApiError";
        this.status = status;
    }
}

async function createApiError(response: Response): Promise<ApiError> {
    return new ApiError(await getErrorMessage(response), response.status);
}

export function isUnauthorizedApiError(error: unknown): boolean {
    if (error instanceof ApiError && error.status === 401) return true;
    const message = error instanceof Error ? error.message : String(error);
    return /invalid token|unauthorized/i.test(message);
}

export async function sendChatMessage(request: ChatRequest, callbacks: StreamCallbacks, signal?: AbortSignal): Promise<void> {
    try {
        let response: Response;
        // INTERIM Bearer revert: fall back to the persisted token so the chat
        // request authenticates even when the browser drops the cross-site
        // auth cookie (the production failure mode).
        const authToken = request.token || getStoredToken();

        if (request.attachments && request.attachments.length > 0) {
            // Multipart upload for attachments
            const form = new FormData();
            form.append("message", request.message);
            if (request.conversation_id) form.append("conversation_id", request.conversation_id);
            if (request.model) form.append("model", request.model);
            if (request.grounded_summary) form.append("grounded_summary", "true");
            form.append("web_search", request.web_search !== false ? "true" : "false");
            request.attachments.forEach(f => form.append("files", f));
            const headers: Record<string, string> = bearerOnlyHeaders(authToken);
            response = await fetch(`${API_BASE}/api/chat/upload`, { credentials: "include", method: "POST", headers, body: form, signal });
        } else {
            const headers: Record<string, string> = { "Content-Type": "application/json", ...bearerOnlyHeaders(authToken) };
            response = await fetch(`${API_BASE}/api/chat`, { credentials: "include",
                method: "POST",
                headers,
                body: JSON.stringify({
                    message: request.message,
                    conversation_id: request.conversation_id,
                    model: request.model,
                    grounded_summary: Boolean(request.grounded_summary),
                    web_search: request.web_search !== false,
                }),
                signal,
            });
        }

        if (!response.ok) {
            callbacks.onError(await getErrorMessage(response), response.status);
            return;
        }
        const reader = response.body?.getReader();
        if (!reader) { callbacks.onError("No response body"); return; }
        const decoder = new TextDecoder();
        let fullText = "";
        let buffer = "";
        let provider = "";
        let lastMeaningfulEventAt = Date.now();
        const streamStartedAt = Date.now();
        let inactivityLimitMs = 110_000;
        let totalLimitMs = 380_000;
        let watchdogError = "";

        // Total visual-pacing budget per message. The drip used to run
        // unbounded INSIDE the read loop: hidden-tab timer throttling stretched
        // the 8ms waits to 1s+ (after 5 min, ~1/min), so a fully-arrived answer
        // took many minutes to "render" and the total-runtime watchdog then
        // printed "exceeded the maximum chat runtime" over a finished, persisted
        // answer (live P15: backend done in 253s, error shown at 1,149s).
        let dripBudgetMs = 5_000;

        const emitToken = async (content: string) => {
            fullText += content;
            if (provider === "tacc" && content.length > 80 && dripBudgetMs > 0 && !document.hidden) {
                const pieces = splitProviderChunk(provider, content);
                for (const piece of pieces) {
                    callbacks.onToken(piece);
                    // The drip pauses the read loop, so it must also feed the
                    // inactivity watchdog: in a throttled background tab the 8ms
                    // waits stretch to 1s+ and the watchdog would otherwise cancel
                    // a stream whose payload has already fully arrived.
                    lastMeaningfulEventAt = Date.now();
                    if (dripBudgetMs <= 0 || document.hidden) {
                        // Budget exhausted (or tab hidden) mid-chunk: flush the
                        // rest instantly instead of pacing it.
                        continue;
                    }
                    const waitStarted = Date.now();
                    await new Promise(resolve => setTimeout(resolve, 8));
                    dripBudgetMs -= Date.now() - waitStarted;
                }
                return;
            }
            callbacks.onToken(content);
        };

        const watchdog = window.setInterval(() => {
            const now = Date.now();
            if (now - lastMeaningfulEventAt > inactivityLimitMs) {
                watchdogError = "The model stopped sending progress. The request was ended so the chat would not remain stuck.";
            } else if (
                now - streamStartedAt > totalLimitMs &&
                now - lastMeaningfulEventAt > 30_000 &&
                !document.hidden
            ) {
                // Only declare a runtime overrun when the stream has ALSO gone
                // quiet: while events still flow the backend is alive and owns
                // its own turn deadline. Skipping hidden tabs avoids cancelling
                // work the user simply isn't watching (throttled timers made
                // this check fire minutes late anyway).
                watchdogError = "The model exceeded the maximum chat runtime.";
            }
            if (watchdogError) {
                reader.cancel(watchdogError).catch(() => undefined);
            }
        }, 5_000);

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6);
                        if (data === "[DONE]") { callbacks.onComplete(fullText); return; }
                        try {
                            const parsed = JSON.parse(data);
                            lastMeaningfulEventAt = Date.now();
                            if (parsed.type === "token") {
                                await emitToken(parsed.content);
                            } else if (parsed.type === "run_meta") {
                                provider = parsed.provider || "";
                                inactivityLimitMs = Math.max(
                                    110_000,
                                    Number(parsed.inactivity_timeout_seconds || 90) * 1000 + 20_000,
                                );
                                totalLimitMs = Math.max(
                                    380_000,
                                    Number(parsed.turn_timeout_seconds || 240) * 1000 + 20_000,
                                );
                                callbacks.onRunMeta?.(parsed);
                        } else if (parsed.type === "thought" && callbacks.onThought) {
                            callbacks.onThought(parsed.content);
                        } else if (parsed.type === "status" && callbacks.onStatus) {
                            callbacks.onStatus(parsed.step, parsed.state);
                        } else if (parsed.type === "tool_call") {
                            // Show as a thinking step, not a separate message bubble
                            if (callbacks.onStatus) {
                                callbacks.onStatus(parsed.displayName || parsed.name, parsed.status || "completed");
                            }
                            // NOTE: deliberately does NOT call onToolCall — that handler
                            // spawns a separate message bubble per tool call, which this
                            // UI intentionally renders as a thinking step instead. The
                            // event's `request` reaches cards via the `data` payload and
                            // the turn's `tool_trace`.
                        } else if (parsed.type === "tool_trace" && callbacks.onToolTrace) {
                            // The backend has always emitted this; nothing consumed it
                            // until the query-provenance surface (Feature 1).
                            callbacks.onToolTrace(Array.isArray(parsed.calls) ? parsed.calls : []);
                        } else if (parsed.type === "data" && callbacks.onData) {
                            callbacks.onData(parsed);
                        } else if (parsed.type === "papers" && callbacks.onPapers) {
                            callbacks.onPapers(parsed.papers, parsed.request || undefined, parsed.blockId || undefined);
                        } else if (parsed.type === "notebook" && callbacks.onNotebook) {
                            callbacks.onNotebook(parsed);
                        } else if (parsed.type === "image" && callbacks.onImage) {
                            callbacks.onImage(parsed);
                        } else if (parsed.type === "plotly" && callbacks.onPlotly) {
                            callbacks.onPlotly(parsed);
                        } else if (parsed.type === "task_group" && callbacks.onTaskGroup) {
                            callbacks.onTaskGroup(parsed);
                        } else if (parsed.type === "task_update" && callbacks.onTaskUpdate) {
                            callbacks.onTaskUpdate(parsed);
                        } else if (parsed.type === "task_list" && callbacks.onTaskList) {
                            callbacks.onTaskList(parsed);
                        } else if (parsed.type === "plan_review" && callbacks.onPlanReview) {
                            callbacks.onPlanReview(parsed);
                        } else if (parsed.type === "conversation_meta" && callbacks.onConversationMeta) {
                            callbacks.onConversationMeta(parsed);
                        } else if (parsed.type === "usage" && callbacks.onUsage) {
                            callbacks.onUsage(parsed);
                        } else if (parsed.type === "web_sources" && callbacks.onWebSources) {
                            callbacks.onWebSources(parsed);
                        } else if (parsed.type === "download_progress" && callbacks.onDownloadProgress) {
                            callbacks.onDownloadProgress(parsed);
                        } else if (parsed.type === "error") {
                            callbacks.onError(parsed.content);
                            return;
                        }
                        } catch {
                            // Non-JSON data, treat as token
                            lastMeaningfulEventAt = Date.now();
                            await emitToken(data);
                        }
                    }
                }
            }
        } finally {
            window.clearInterval(watchdog);
        }
        if (watchdogError) {
            // Self-heal: if real answer text already arrived, complete with it
            // (the backend persisted the message anyway) instead of replacing a
            // finished answer with an error bubble (live P15).
            if (fullText.trim()) {
                callbacks.onComplete(fullText);
            } else {
                callbacks.onError(watchdogError);
            }
            return;
        }
        // UI-03: reaching here means the reader drained WITHOUT [DONE] or an
        // explicit error event — the [DONE] sentinel exists precisely to
        // distinguish clean termination, so its absence is a truncated stream
        // (backend crash, proxy timeout), never a completion. Historical
        // pattern: "truncated data masquerading as complete".
        if (callbacks.onIncomplete) {
            callbacks.onIncomplete(fullText);
        } else {
            callbacks.onError("The response stream ended unexpectedly — the answer may be incomplete.");
        }
    } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return; // user cancelled
        callbacks.onError(error instanceof Error ? error.message : "Request failed.");
    }
}

export async function submitPlanFeedback(
    conversationId: string,
    approve: boolean,
    feedback: string = "",
    token?: string,
    runId?: string,
): Promise<{ status: string; action: string }> {
    const headers: Record<string, string> = { "Content-Type": "application/json", ...bearerOnlyHeaders(token) };

    const res = await fetch(`${API_BASE}/api/plan-feedback`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify({
            conversation_id: conversationId,
            approve,
            feedback,
            // UIAPI-08: address THIS run's queue — two concurrent runs in one
            // conversation each have their own registration server-side.
            run_id: runId || "",
        }),
    });

    if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Plan feedback failed: ${errText}`);
    }
    return res.json();
}

export interface UsageCostBreakdown {
    cost_usd: number;
    by_provider_usd: Record<string, number>;
    total_tokens: number;
    /** Tokens excluded from cost_usd because their model has no known price
     *  (TACC / self-hosted). Surface alongside cost_usd — on its own, cost_usd
     *  looks like the whole story when it may cover only part of the tokens. */
    unpriced_tokens: number;
}

export interface UsageSummary {
    platform: Record<string, { used_tokens: number; limit_tokens?: number | null; unlimited: boolean; exhausted: boolean }>;
    byok: Record<string, { used_tokens: number; limit_tokens?: number | null; unlimited: boolean; exhausted: boolean }>;
    is_admin: boolean;
    is_quota_exempt?: boolean;
    platform_quota_window_days?: number;
    daily?: {
        used_tokens: number;
        limit_tokens?: number | null;
        unlimited: boolean;
        exhausted: boolean;
        remaining_tokens?: number | null;
        window_hours?: number;
    };
    cost?: {
        platform_today: UsageCostBreakdown;
        platform_week: UsageCostBreakdown;
        byok_today: UsageCostBreakdown;
        byok_week: UsageCostBreakdown;
        is_estimate: boolean;
        pricing_last_verified: string;
    };
}

/** Per-user tokens + estimated cost (today / this week) and cap headroom.
 *  Canonical endpoint; `/api/usage-quota` returns the same payload. */
export async function getUsageSummary(token?: string | null): Promise<UsageSummary> {
    const res = await fetch(`${API_BASE}/api/usage/summary`, {
        credentials: "include",
        headers: bearerOnlyHeaders(token),
    });
    if (!res.ok) throw new Error(await getErrorMessage(res));
    return res.json();
}

export async function getModels(): Promise<string[]> {
    try {
        const res = await fetch(`${API_BASE}/api/models`, { credentials: "include" });
        const data = await res.json();
        return mergeAvailableModels(data.models);
    }
    catch { return DEFAULT_AVAILABLE_MODELS; }
}

export type SpectralLineJobStatus =
    | "queued"
    | "running"
    | "needs_input"
    | "succeeded"
    | "partial"
    | "failed"
    | "canceled"
    | "orphaned";

export interface SpectralSpecies {
    species_id: number;
    tag: string;
    formula: string;
    chemical_name: string;
    molecular_mass?: number | null;
    status: "known" | "probable" | "potential" | "atmospheric" | "unknown";
    label: string;
}

export interface SpectralLineRecord {
    species_id?: string | number;
    line_id?: string | number;
    raw_line_ids?: Array<string | number>;
    formula?: string;
    species?: string;
    chemical_name?: string;
    species_status?: string;
    transition?: string;
    resolved_quantum_numbers?: string;
    unresolved_quantum_numbers?: string;
    catalogs?: string[];
    frequency_ghz?: number;
    predicted_frequency_ghz?: number | null;
    predicted_frequency_uncertainty_mhz?: number | null;
    measured_frequency_ghz?: number | null;
    measured_frequency_uncertainty_mhz?: number | null;
    observed_frequency_ghz?: number | null;
    frequency_uncertainty_mhz?: number | null;
    frequency_selection_reason?: string;
    nrao_recommended?: boolean | null;
    astronomically_observed?: boolean | null;
    lower_energy_k?: number | null;
    upper_energy_k?: number | null;
    log_intensity?: number | null;
    lovas_astronomical_intensity?: number | null;
    sijmu2?: number | null;
    sij?: number | null;
    aij_log?: number | null;
    alma_bands?: number[];
    alma_preferred_band?: number | null;
    alma_band_edge_warning?: string | null;
    score?: number;
    classification?: string;
    velocity_offset_kms?: number | null;
    score_components?: Record<string, number>;
    [key: string]: unknown;
}

export interface SpectralCoverageProject {
    proposal_id: string;
    target_name?: string;
    pi_name?: string;
    obs_title?: string;
    bands?: string[];
    covers_all_lines: boolean;
    all_lines_full: boolean;
    covered_line_count: number;
    requested_line_count: number;
    minimum_edge_margin_mhz?: number | null;
    angular_separation_arcsec?: number | null;
    best_angular_resolution_arcsec?: number | null;
    total_exposure_seconds?: number;
    archive_url?: string;
    observations: Array<Record<string, unknown>>;
}

export interface SpectralLineMetadata {
    enabled: boolean;
    band_registry: {
        version: string;
        source: string;
        bands: Array<{ band: number; minimum_ghz: number; maximum_ghz: number }>;
    };
    units: string[];
    catalogs: string[];
    versions: string[];
    defaults: Record<string, unknown>;
    limits: Record<string, unknown>;
}

export interface SpectralLineJob {
    job_id: string;
    operation: "catalog_search" | "alma_coverage" | "confusion";
    status: SpectralLineJobStatus;
    phase: string;
    progress: number;
    payload: Record<string, unknown>;
    created_at: string;
    updated_at: string;
    warnings: string[];
    summary: Record<string, unknown>;
    error?: string | null;
    cancel_requested?: boolean;
    rows?: Array<SpectralLineRecord | SpectralCoverageProject>;
    pagination?: {
        dataset: string;
        page: number;
        page_size: number;
        total_rows: number;
        total_pages: number;
    };
    result_context?: Record<string, unknown>;
}

function spectralAuthHeaders(token?: string | null, json = false): Record<string, string> {
    const headers: Record<string, string> = bearerOnlyHeaders(token);
    if (json) headers["Content-Type"] = "application/json";
    return headers;
}

export async function getSpectralLineMetadata(token?: string | null): Promise<SpectralLineMetadata> {
    const res = await fetch(`${API_BASE}/api/spectral-lines/metadata`, { credentials: "include",
        headers: spectralAuthHeaders(token),
    });
    return parseJsonResponse<SpectralLineMetadata>(res);
}

export async function searchSpectralSpecies(
    query: string,
    limit = 25,
    token?: string | null,
): Promise<{ species: SpectralSpecies[]; query: string }> {
    const params = new URLSearchParams({ query, limit: String(limit) });
    const res = await fetch(`${API_BASE}/api/spectral-lines/species?${params}`, { credentials: "include",
        headers: spectralAuthHeaders(token),
    });
    return parseJsonResponse<{ species: SpectralSpecies[]; query: string }>(res);
}

export async function resolveSpectralTarget(
    input: {
        target_name: string;
        redshift?: number;
        ra_deg?: number;
        dec_deg?: number;
        /** Probe NOIRLab SPARCL for optical spectra at the resolved position. */
        include_sparcl?: boolean;
    },
    token?: string | null,
): Promise<Record<string, unknown>> {
    const res = await fetch(`${API_BASE}/api/spectral-lines/resolve-target`, { credentials: "include",
        method: "POST",
        headers: spectralAuthHeaders(token, true),
        body: JSON.stringify(input),
    });
    return parseJsonResponse<Record<string, unknown>>(res);
}

export async function startSpectralLineJob(
    operation: SpectralLineJob["operation"],
    payload: Record<string, unknown>,
    token?: string | null,
): Promise<SpectralLineJob> {
    const res = await fetch(`${API_BASE}/api/spectral-lines/jobs`, { credentials: "include",
        method: "POST",
        headers: spectralAuthHeaders(token, true),
        body: JSON.stringify({ operation, payload }),
    });
    return parseJsonResponse<SpectralLineJob>(res);
}

export async function getSpectralLineJob(
    jobId: string,
    page = 1,
    pageSize = 100,
    token?: string | null,
    dataset?: "lines" | "raw_lines" | "coverage" | "candidates",
): Promise<SpectralLineJob> {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (dataset) params.set("dataset", dataset);
    const res = await fetch(`${API_BASE}/api/spectral-lines/jobs/${jobId}?${params}`, { credentials: "include",
        headers: spectralAuthHeaders(token),
    });
    return parseJsonResponse<SpectralLineJob>(res);
}

export async function cancelSpectralLineJob(
    jobId: string,
    token?: string | null,
): Promise<SpectralLineJob> {
    const res = await fetch(`${API_BASE}/api/spectral-lines/jobs/${jobId}`, { credentials: "include",
        method: "DELETE",
        headers: spectralAuthHeaders(token),
    });
    return parseJsonResponse<SpectralLineJob>(res);
}

export async function downloadSpectralLineExport(
    jobId: string,
    dataset: "lines" | "raw_lines" | "coverage" | "candidates",
    format: "csv" | "tsv" | "json" | "casa",
    token?: string | null,
): Promise<{ blob: Blob; filename: string }> {
    const params = new URLSearchParams({ dataset, format });
    const res = await fetch(`${API_BASE}/api/spectral-lines/jobs/${jobId}/export?${params}`, { credentials: "include",
        headers: spectralAuthHeaders(token),
    });
    if (!res.ok) throw await createApiError(res);
    const disposition = res.headers.get("content-disposition") || "";
    const filename = disposition.match(/filename="([^"]+)"/)?.[1] || `spectral-lines.${format}`;
    return { blob: await res.blob(), filename };
}

export interface WorkbenchSession {
    session_id: string;
    source_url: string;
    filename: string;
    archive: string;
    project_code?: string;
    mous_uid?: string;
    status: string;
    metadata: Record<string, unknown>;
    evidence: Record<string, unknown>;
    state?: Record<string, unknown>;
    cache?: Record<string, unknown>;
    jobs?: WorkbenchJob[];
}

export interface WorkbenchMetadata {
    session_id: string;
    filename: string;
    source_url: string;
    archive: string;
    project_code?: string;
    mous_uid?: string;
    status: string;
    metadata: Record<string, unknown>;
    state: Record<string, unknown>;
    cache: Record<string, unknown>;
    jobs: WorkbenchJob[];
    evidence: Record<string, unknown>;
}

export type WorkbenchJobStatus = "queued" | "running" | "succeeded" | "failed" | "canceled" | "orphaned";

export interface WorkbenchJob {
    job_id: string;
    operation: string;
    status: WorkbenchJobStatus | string;
    phase: string;
    progress: number;
    created_at: number;
    started_at?: number | null;
    finished_at?: number | null;
    cancel_requested?: boolean;
    request?: Record<string, unknown>;
    result?: Record<string, unknown> | null;
    error?: string;
    metrics?: Record<string, unknown>;
}

export interface WorkbenchJobResponse {
    session_id: string;
    job: WorkbenchJob;
    jobs: WorkbenchJob[];
}

export interface WorkbenchLineOverlays {
    session_id: string;
    query_observed_frequency_ghz: number;
    query_rest_frequency_ghz: number;
    redshift: number;
    tolerance_ghz: number;
    preset?: WorkbenchLinePreset | null;
    presets?: WorkbenchLinePreset[];
    n_matches: number;
    lines: Record<string, unknown>[];
    backend?: "astroquery" | "slap" | "unavailable";
    query_note?: string | null;
    query_error?: string | null;
    evidence: Record<string, unknown>;
}

export interface WorkbenchLinePreset {
    key: string;
    label: string;
    rest_frequency_ghz: number;
    family: string;
}

export interface WorkbenchLinePresetResponse {
    presets: WorkbenchLinePreset[];
    evidence: Record<string, unknown>;
}

export interface WorkbenchRenderPlan {
    session_id: string;
    status: string;
    operation: string;
    state: Record<string, unknown>;
    image?: {
        data_url: string | null;
        data_status: string;
        label?: string;
        axis_labels?: Record<string, string>;
        wcs_status?: string;
        shape?: number[];
        channel?: number | null;
        line_labels?: Array<Record<string, unknown>>;
        analysis_product?: string;
        preview?: Record<string, unknown> | null;
    };
    stats?: {
        rms?: number | null;
        rms_region?: Record<string, number> | null;
        rms_method?: string | null;
        min?: number | null;
        max?: number | null;
        mean?: number | null;
        unit?: string;
        shape?: number[];
        analysis_product?: string;
        preview?: Record<string, unknown> | null;
    };
    contours?: {
        sigma?: number[];
        levels?: Array<number | null>;
        unit?: string;
    };
    next_phase?: string;
    evidence: Record<string, unknown>;
}

export interface WorkbenchPrepareResult {
    session_id: string;
    status: string;
    cache: Record<string, unknown>;
    cache_path: string;
    evidence: Record<string, unknown>;
}

export interface WorkbenchSpectrumPlan {
    session_id: string;
    status: string;
    operation: string;
    extraction: Record<string, unknown>;
    spectral_axis: {
        label: string;
        unit: string;
        channel_count: number;
        downsample_step: number;
        values: number[];
        indices: number[];
        rest_frequency_ghz?: number | null;
    };
    series: {
        x: number[];
        y: number[];
        x_label: string;
        y_label: string;
        data_status: string;
        analysis_product?: string;
        preview?: Record<string, unknown> | null;
    };
    next_phase?: string;
    evidence: Record<string, unknown>;
}

export interface WorkbenchPvSlicePlan {
    session_id: string;
    status: string;
    operation: string;
    path: { x: number; y: number }[];
    width_pixels: number;
    spatial_axis: Record<string, unknown>;
    spectral_axis: Record<string, unknown>;
    image: {
        data_url: string | null;
        data_status: string;
        analysis_product?: string;
        preview?: Record<string, unknown> | null;
    };
    next_phase?: string;
    evidence: Record<string, unknown>;
}

export interface WorkbenchExports {
    session_id: string;
    formats: string[];
    exports: Record<string, { filename: string; content: string; mime_type?: string; encoding?: string }>;
    evidence: Record<string, unknown>;
}

async function parseJsonResponse<T>(res: Response): Promise<T> {
    if (!res.ok) throw await createApiError(res);
    return res.json() as Promise<T>;
}

export async function createWorkbenchSession(
    input: {
        source_url: string;
        filename?: string;
        project_code?: string;
        mous_uid?: string;
    },
    token?: string | null,
): Promise<WorkbenchSession> {
    const headers: Record<string, string> = authHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/session`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchSession>(res);
}

export async function getWorkbenchMetadata(sessionId: string, token?: string | null): Promise<WorkbenchMetadata> {
    const headers: Record<string, string> = bearerOnlyHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/metadata`, { credentials: "include", headers });
    return parseJsonResponse<WorkbenchMetadata>(res);
}

export async function startWorkbenchJob(
    sessionId: string,
    input: {
        operation: string;
        payload?: Record<string, unknown>;
    },
    token?: string | null,
): Promise<WorkbenchJobResponse> {
    const headers: Record<string, string> = authHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/jobs`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchJobResponse>(res);
}

export async function getWorkbenchJob(
    sessionId: string,
    jobId: string,
    token?: string | null,
): Promise<WorkbenchJobResponse> {
    const headers: Record<string, string> = bearerOnlyHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/jobs/${jobId}`, { credentials: "include", headers });
    return parseJsonResponse<WorkbenchJobResponse>(res);
}

export async function cancelWorkbenchJob(
    sessionId: string,
    jobId: string,
    token?: string | null,
): Promise<WorkbenchJobResponse> {
    const headers: Record<string, string> = bearerOnlyHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/jobs/${jobId}`, { credentials: "include",
        method: "DELETE",
        headers,
    });
    return parseJsonResponse<WorkbenchJobResponse>(res);
}

export async function planWorkbenchRender(
    sessionId: string,
    input: {
        mode?: string;
        channel?: number | null;
        moment?: number | null;
        colormap?: string;
        stretch?: string;
        contour_sigma?: number[];
        rms_region?: { x1: number; y1: number; x2: number; y2: number } | null;
    },
    token?: string | null,
): Promise<WorkbenchRenderPlan> {
    const headers: Record<string, string> = authHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/render`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchRenderPlan>(res);
}

export async function prepareWorkbenchProduct(
    sessionId: string,
    input: {
        max_bytes?: number;
        user_cache_bytes?: number;
        cache_ttl_seconds?: number;
        force?: boolean;
    } = {},
    token?: string | null,
): Promise<WorkbenchPrepareResult> {
    const headers: Record<string, string> = authHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/prepare`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchPrepareResult>(res);
}

export async function planWorkbenchSpectrum(
    sessionId: string,
    input: {
        x_pixel?: number | null;
        y_pixel?: number | null;
        aperture_radius_pixels?: number;
        aperture_radius_arcsec?: number | null;
        max_points?: number;
    },
    token?: string | null,
): Promise<WorkbenchSpectrumPlan> {
    const headers: Record<string, string> = authHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/spectrum`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchSpectrumPlan>(res);
}

export async function planWorkbenchPvSlice(
    sessionId: string,
    input: {
        path?: { x: number; y: number }[];
        width_pixels?: number;
        max_points?: number;
    },
    token?: string | null,
): Promise<WorkbenchPvSlicePlan> {
    const headers: Record<string, string> = authHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/pv-slice`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchPvSlicePlan>(res);
}

export async function getWorkbenchLineOverlays(
    sessionId: string,
    input: {
        observed_frequency_ghz?: number;
        line_preset_key?: string;
        redshift?: number;
        tolerance_ghz?: number;
        top_n?: number;
    },
    token?: string | null,
): Promise<WorkbenchLineOverlays> {
    const headers: Record<string, string> = authHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/line-overlays`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchLineOverlays>(res);
}

export async function getWorkbenchLinePresets(): Promise<WorkbenchLinePresetResponse> {
    const res = await fetch(`${API_BASE}/api/workbench/line-presets`, { credentials: "include" });
    return parseJsonResponse<WorkbenchLinePresetResponse>(res);
}

export async function getWorkbenchExports(
    sessionId: string,
    formats: string[] = ["casa", "carta", "ds9", "python"],
    token?: string | null,
): Promise<WorkbenchExports> {
    const headers: Record<string, string> = authHeaders(token);
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/export`, { credentials: "include",
        method: "POST",
        headers,
        body: JSON.stringify({ formats }),
    });
    return parseJsonResponse<WorkbenchExports>(res);
}

export async function reviewProposal(file: File, callbacks: StreamCallbacks, signal?: AbortSignal, token?: string | null): Promise<void> {
    try {
        const form = new FormData();
        form.append("file", file);

        // S5: /api/proposals/review now requires authentication. Send the
        // user's Bearer token (do NOT set Content-Type — the browser sets the
        // multipart/form-data boundary for FormData automatically).
        const headers: Record<string, string> = bearerOnlyHeaders(token);
        const response = await fetch(`${API_BASE}/api/proposals/review`, { credentials: "include",
            method: "POST",
            headers,
            body: form,
            signal,
        });

        if (!response.ok) { callbacks.onError(await getErrorMessage(response)); return; }
        const reader = response.body?.getReader();
        if (!reader) { callbacks.onError("No response body"); return; }
        const decoder = new TextDecoder();
        let fullText = "";
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const data = line.slice(6);
                    if (data === "[DONE]") { callbacks.onComplete(fullText); return; }
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.type === "critique") {
                            fullText += parsed.content;
                            callbacks.onToken(parsed.content);
                        } else if (parsed.type === "status" && callbacks.onStatus) {
                            callbacks.onStatus(parsed.step, parsed.state);
                        } else if (parsed.type === "error") {
                            callbacks.onError(parsed.content);
                            return;
                        }
                    } catch {
                        // ignore
                    }
                }
            }
        }
        callbacks.onComplete(fullText);
    } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return; // user cancelled
        callbacks.onError(error instanceof Error ? error.message : "Proposal review request failed.");
    }
}

// ── Conversation History API ────────────────────────────────────

function authHeaders(token?: string | null): Record<string, string> {
    // INTERIM Bearer revert: the Bearer header (explicit token, else the
    // persisted one) is the credential that works everywhere; the httpOnly
    // cookie still rides along for browsers that accept it cross-site.
    return { "Content-Type": "application/json", ...bearerOnlyHeaders(token) };
}

export interface ServerConversation {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
    model?: string;
}

export interface ServerMessage {
    role: string;
    content: string;
    type: string;
    metadata?: Record<string, unknown>;
}

export async function fetchConversations(token?: string): Promise<ServerConversation[]> {
    try {
        const res = await fetch(`${API_BASE}/api/conversations`, { credentials: "include", headers: authHeaders(token) });
        if (!res.ok) return [];
        const data = await res.json();
        return data.conversations || [];
    } catch {
        return [];
    }
}

export async function fetchConversationMessages(conversationId: string, token?: string): Promise<ServerMessage[]> {
    try {
        const res = await fetch(`${API_BASE}/api/conversations/${conversationId}/messages`, { credentials: "include", headers: authHeaders(token) });
        if (!res.ok) return [];
        const data = await res.json();
        return data.messages || [];
    } catch {
        return [];
    }
}

/** This user's block ratings for one conversation (Feature 4) — used to
 *  re-light the stars after a reload. */
export async function fetchBlockRatings(conversationId: string, token?: string): Promise<Record<string, number>> {
    try {
        const res = await fetch(
            `${API_BASE}/api/block-feedback?conversation_id=${encodeURIComponent(conversationId)}`,
            { credentials: "include", headers: authHeaders(token) },
        );
        if (!res.ok) return {};
        const data = await res.json();
        const out: Record<string, number> = {};
        for (const [blockId, row] of Object.entries(data.ratings || {})) {
            const rating = (row as { rating?: number })?.rating;
            if (typeof rating === "number") out[blockId] = rating;
        }
        return out;
    } catch {
        return {};
    }
}

/** Persist a 1-5 star rating on one block. Returns false on failure so the
 *  caller can roll the optimistic star back instead of showing a lie. */
export async function submitBlockRating(payload: {
    blockId: string;
    rating: number;
    blockKind?: string;
    runId?: string;
    conversationId?: string;
    model?: string;
    comment?: string;
}, token?: string): Promise<boolean> {
    try {
        const res = await fetch(`${API_BASE}/api/block-feedback`, {
            method: "POST",
            credentials: "include",
            headers: authHeaders(token),
            body: JSON.stringify({
                block_id: payload.blockId,
                rating: payload.rating,
                block_kind: payload.blockKind || "",
                run_id: payload.runId || "",
                conversation_id: payload.conversationId || "",
                model: payload.model || "",
                comment: payload.comment || "",
            }),
        });
        return res.ok;
    } catch {
        return false;
    }
}

export async function deleteConversationApi(conversationId: string, token?: string): Promise<boolean> {
    // Retry up to 3 times — Render cold starts can cause transient failures
    for (let attempt = 1; attempt <= 3; attempt++) {
        try {
            const res = await fetch(`${API_BASE}/api/conversations/${conversationId}`, { credentials: "include",
                method: "DELETE",
                headers: authHeaders(token),
            });
            if (res.ok) {
                console.log(`[Quasar] Deleted conversation ${conversationId}`);
                return true;
            }
            console.warn(`[Quasar] Delete attempt ${attempt} failed: HTTP ${res.status}`);
        } catch (e) {
            console.warn(`[Quasar] Delete attempt ${attempt} error:`, e);
        }
        // Wait before retry (500ms, 1s, 2s)
        if (attempt < 3) await new Promise(r => setTimeout(r, attempt * 500));
    }
    console.error(`[Quasar] Failed to delete conversation ${conversationId} after 3 attempts`);
    return false;
}

/* ────────────────────────────────────────────
   DATA LAB: Jobs panel + My-tables (MyDB-lite)
   ──────────────────────────────────────────── */
export interface DatalabJobRecord {
    job_id: string;
    kind: string;
    status: "queued" | "running" | "succeeded" | "failed" | "canceled" | "submitted" | string;
    result?: { result_id?: string; rowcount?: number; note?: string; candidates_found?: number; available?: boolean } | null;
    error?: string | null;
    external?: boolean;
    created_at: number;
    updated_at: number;
}

export interface MyTableEntry {
    name: string;
    rowcount: number;
    columns: string[];
    saved_at: number;
    saved_from?: string;
    description?: string | null;
    catalog?: string | null;
    table?: string | null;
}

export async function listDatalabJobs(): Promise<DatalabJobRecord[]> {
    const res = await fetch(`${API_BASE}/api/datalab/jobs`, { credentials: "include", headers: bearerOnlyHeaders() });
    const data = await parseJsonResponse<{ jobs: DatalabJobRecord[] }>(res);
    return data.jobs || [];
}

export async function cancelDatalabJob(jobId: string): Promise<DatalabJobRecord> {
    const res = await fetch(`${API_BASE}/api/datalab/jobs/${encodeURIComponent(jobId)}`, {
        credentials: "include",
        method: "DELETE",
        headers: bearerOnlyHeaders(),
    });
    return parseJsonResponse<DatalabJobRecord>(res);
}

export async function listMyTables(): Promise<MyTableEntry[]> {
    const res = await fetch(`${API_BASE}/api/datalab/mytables`, { credentials: "include", headers: bearerOnlyHeaders() });
    const data = await parseJsonResponse<{ my_tables: MyTableEntry[] }>(res);
    return data.my_tables || [];
}

export async function deleteMyTable(name: string): Promise<boolean> {
    const res = await fetch(`${API_BASE}/api/datalab/mytables/${encodeURIComponent(name)}`, {
        credentials: "include",
        method: "DELETE",
        headers: bearerOnlyHeaders(),
    });
    return res.ok;
}

/* ────────────────────────────────────────────
   FULL-RESULT EXPORT (Feature 2)
   ──────────────────────────────────────────── */

/** Stream the COMPLETE stored result as CSV (f2-CX-09).
 *
 *  Returns the raw Response rather than a parsed blob: the caller must read
 *  the status AND the X-Quasar-Truncated / X-Quasar-Rowcount /
 *  X-Quasar-Total-Rows headers (lib/export-decision.js::interpretExportResponse)
 *  so an expired result fails loudly and a ceiling-clipped export can never be
 *  delivered silently as full. */
export async function exportResultCsv(resultId: string, token?: string | null): Promise<Response> {
    return fetch(`${API_BASE}/api/results/${encodeURIComponent(resultId)}/export.csv`, {
        credentials: "include",
        headers: bearerOnlyHeaders(token),
    });
}
