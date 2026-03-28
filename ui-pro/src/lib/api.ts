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
    onStatus?: (step: string, state: string) => void;
    onTaskGroup?: (group: Record<string, unknown>) => void;
    onTaskUpdate?: (update: Record<string, unknown>) => void;
    onTaskList?: (list: Record<string, unknown>) => void;
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
                        } else if (parsed.type === "task_group" && callbacks.onTaskGroup) {
                            callbacks.onTaskGroup(parsed);
                        } else if (parsed.type === "task_update" && callbacks.onTaskUpdate) {
                            callbacks.onTaskUpdate(parsed);
                        } else if (parsed.type === "task_list" && callbacks.onTaskList) {
                            callbacks.onTaskList(parsed);
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

export async function getModels(): Promise<string[]> {
    try { const res = await fetch(`${API_BASE}/api/models`); const data = await res.json(); return data.models; }
    catch { return ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "claude-3-7-sonnet-20250219", "gemini-3.1-pro"]; }
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
