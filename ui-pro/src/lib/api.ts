import { DEFAULT_AVAILABLE_MODELS, mergeAvailableModels } from "./models";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatRequest {
    message: string;
    conversation_id?: string;
    model?: string;
    attachments?: File[];
    token?: string;  // auth token for personal RAG
    grounded_summary?: boolean;
    web_search?: boolean;
}

export interface StreamCallbacks {
    onToken: (token: string) => void;
    onThought?: (thought: string) => void;
    onToolCall?: (toolName: string, input: string) => void;
    onData?: (data: Record<string, unknown>) => void;
    onPapers?: (papers: Record<string, unknown>[]) => void;
    onNotebook?: (notebook: Record<string, unknown>) => void;
    onImage?: (image: { url: string; caption: string }) => void;
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
    onDownloadProgress?: (data: { filename: string; downloaded_bytes: number; total_bytes: number | null; speed_kbps: number; percent: number | null }) => void;
    onComplete: (fullResponse: string) => void;
    onError: (error: string) => void;
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

export async function sendChatMessage(request: ChatRequest, callbacks: StreamCallbacks, signal?: AbortSignal): Promise<void> {
    try {
        let response: Response;

        if (request.attachments && request.attachments.length > 0) {
            // Multipart upload for attachments
            const form = new FormData();
            form.append("message", request.message);
            if (request.conversation_id) form.append("conversation_id", request.conversation_id);
            if (request.model) form.append("model", request.model);
            if (request.grounded_summary) form.append("grounded_summary", "true");
            form.append("web_search", request.web_search !== false ? "true" : "false");
            request.attachments.forEach(f => form.append("files", f));
            const headers: Record<string, string> = {};
            if (request.token) headers["Authorization"] = `Bearer ${request.token}`;
            response = await fetch(`${API_BASE}/api/chat/upload`, { method: "POST", headers, body: form, signal });
        } else {
            const headers: Record<string, string> = { "Content-Type": "application/json" };
            if (request.token) headers["Authorization"] = `Bearer ${request.token}`;
            response = await fetch(`${API_BASE}/api/chat`, {
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
            callbacks.onError(await getErrorMessage(response));
            return;
        }
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
                        if (parsed.type === "token") {
                            fullText += parsed.content;
                            callbacks.onToken(parsed.content);
                        } else if (parsed.type === "thought" && callbacks.onThought) {
                            callbacks.onThought(parsed.content);
                        } else if (parsed.type === "status" && callbacks.onStatus) {
                            callbacks.onStatus(parsed.step, parsed.state);
                        } else if (parsed.type === "tool_call") {
                            // Show as a thinking step, not a separate message bubble
                            if (callbacks.onStatus) {
                                callbacks.onStatus(parsed.displayName || parsed.name, parsed.status || "completed");
                            }
                        } else if (parsed.type === "data" && callbacks.onData) {
                            callbacks.onData(parsed);
                        } else if (parsed.type === "papers" && callbacks.onPapers) {
                            callbacks.onPapers(parsed.papers);
                        } else if (parsed.type === "notebook" && callbacks.onNotebook) {
                            callbacks.onNotebook(parsed);
                        } else if (parsed.type === "image" && callbacks.onImage) {
                            callbacks.onImage(parsed);
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
                        fullText += data;
                        callbacks.onToken(data);
                    }
                }
            }
        }
        callbacks.onComplete(fullText);
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
): Promise<{ status: string; action: string }> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE}/api/plan-feedback`, {
        method: "POST",
        headers,
        body: JSON.stringify({
            conversation_id: conversationId,
            approve,
            feedback,
        }),
    });

    if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Plan feedback failed: ${errText}`);
    }
    return res.json();
}

export async function getModels(): Promise<string[]> {
    try {
        const res = await fetch(`${API_BASE}/api/models`);
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
    const headers: Record<string, string> = {};
    if (json) headers["Content-Type"] = "application/json";
    if (token) headers.Authorization = `Bearer ${token}`;
    return headers;
}

export async function getSpectralLineMetadata(token?: string | null): Promise<SpectralLineMetadata> {
    const res = await fetch(`${API_BASE}/api/spectral-lines/metadata`, {
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
    const res = await fetch(`${API_BASE}/api/spectral-lines/species?${params}`, {
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
    },
    token?: string | null,
): Promise<Record<string, unknown>> {
    const res = await fetch(`${API_BASE}/api/spectral-lines/resolve-target`, {
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
    const res = await fetch(`${API_BASE}/api/spectral-lines/jobs`, {
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
    const res = await fetch(`${API_BASE}/api/spectral-lines/jobs/${jobId}?${params}`, {
        headers: spectralAuthHeaders(token),
    });
    return parseJsonResponse<SpectralLineJob>(res);
}

export async function cancelSpectralLineJob(
    jobId: string,
    token?: string | null,
): Promise<SpectralLineJob> {
    const res = await fetch(`${API_BASE}/api/spectral-lines/jobs/${jobId}`, {
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
    const res = await fetch(`${API_BASE}/api/spectral-lines/jobs/${jobId}/export?${params}`, {
        headers: spectralAuthHeaders(token),
    });
    if (!res.ok) throw new Error(await getErrorMessage(res));
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
    if (!res.ok) throw new Error(await getErrorMessage(res));
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
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/session`, {
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchSession>(res);
}

export async function getWorkbenchMetadata(sessionId: string, token?: string | null): Promise<WorkbenchMetadata> {
    const headers: Record<string, string> = {};
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/metadata`, { headers });
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
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/jobs`, {
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
    const headers: Record<string, string> = {};
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/jobs/${jobId}`, { headers });
    return parseJsonResponse<WorkbenchJobResponse>(res);
}

export async function cancelWorkbenchJob(
    sessionId: string,
    jobId: string,
    token?: string | null,
): Promise<WorkbenchJobResponse> {
    const headers: Record<string, string> = {};
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/jobs/${jobId}`, {
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
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/render`, {
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
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/prepare`, {
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
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/spectrum`, {
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
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/pv-slice`, {
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
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/line-overlays`, {
        method: "POST",
        headers,
        body: JSON.stringify(input),
    });
    return parseJsonResponse<WorkbenchLineOverlays>(res);
}

export async function getWorkbenchLinePresets(): Promise<WorkbenchLinePresetResponse> {
    const res = await fetch(`${API_BASE}/api/workbench/line-presets`);
    return parseJsonResponse<WorkbenchLinePresetResponse>(res);
}

export async function getWorkbenchExports(
    sessionId: string,
    formats: string[] = ["casa", "carta", "ds9", "python"],
    token?: string | null,
): Promise<WorkbenchExports> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}/api/workbench/${sessionId}/export`, {
        method: "POST",
        headers,
        body: JSON.stringify({ formats }),
    });
    return parseJsonResponse<WorkbenchExports>(res);
}

export async function reviewProposal(file: File, callbacks: StreamCallbacks, signal?: AbortSignal): Promise<void> {
    try {
        const form = new FormData();
        form.append("file", file);

        const response = await fetch(`${API_BASE}/api/proposals/review`, {
            method: "POST",
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

function authHeaders(token: string): Record<string, string> {
    return { "Content-Type": "application/json", "Authorization": `Bearer ${token}` };
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

export async function fetchConversations(token: string): Promise<ServerConversation[]> {
    try {
        const res = await fetch(`${API_BASE}/api/conversations`, { headers: authHeaders(token) });
        if (!res.ok) return [];
        const data = await res.json();
        return data.conversations || [];
    } catch {
        return [];
    }
}

export async function fetchConversationMessages(conversationId: string, token: string): Promise<ServerMessage[]> {
    try {
        const res = await fetch(`${API_BASE}/api/conversations/${conversationId}/messages`, { headers: authHeaders(token) });
        if (!res.ok) return [];
        const data = await res.json();
        return data.messages || [];
    } catch {
        return [];
    }
}

export async function deleteConversationApi(conversationId: string, token: string): Promise<boolean> {
    // Retry up to 3 times — Render cold starts can cause transient failures
    for (let attempt = 1; attempt <= 3; attempt++) {
        try {
            const res = await fetch(`${API_BASE}/api/conversations/${conversationId}`, {
                method: "DELETE",
                headers: { "Authorization": `Bearer ${token}` },
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
