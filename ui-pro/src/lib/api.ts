const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatRequest {
    message: string;
    conversation_id?: string;
    model?: string;
    attachments?: File[];
    token?: string;  // auth token for personal RAG
}

export interface StreamCallbacks {
    onToken: (token: string) => void;
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
    onWebSources?: (data: { sources: { title: string; url: string; snippet: string }[]; images: { url: string; description: string }[]; query: string }) => void;
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
                body: JSON.stringify({ message: request.message, conversation_id: request.conversation_id, model: request.model }),
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
    try { const res = await fetch(`${API_BASE}/api/models`); const data = await res.json(); return data.models; }
    catch { return ["gpt-5.4-mini", "gpt-4.1", "gpt-4o-mini", "deepseek-v4-pro", "deepseek-v4-flash"]; }
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
