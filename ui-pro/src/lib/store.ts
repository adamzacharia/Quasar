"use client";

import { create } from "zustand";
import type { Conversation, Message, Paper, TaskGroup, TaskItem, TaskChecklist, WebImage, WebSource } from "./types";
import type { ThoughtStep } from "@/components/ThoughtProcessWidget";
import {
    fetchConversations as apiFetchConversations,
    fetchConversationMessages as apiFetchMessages,
    deleteConversationApi,
    type ServerConversation,
    type ServerMessage,
} from "./api";
import {
    attachThinkingStepsToLastAssistant,
    findLastAssistantTextIndex,
    sanitizeAssistantContent,
    updateLastAssistantContent,
    updateLastAssistantThinking as updateAssistantThinking,
} from "./chat-message-updaters";
import { mergeEvidenceQuality, rankWebSources } from "./evidence-quality";
import { DEFAULT_AVAILABLE_MODELS, mergeAvailableModels } from "./models";

interface ChatStore {
    conversations: Conversation[];
    activeConversationId: string | null;
    messages: Message[];
    isStreaming: boolean;
    streamingContent: string;
    selectedModel: string;
    availableModels: string[];
    sidebarOpen: boolean;
    thinkingSteps: ThoughtStep[];
    thinkingStatus: "idle" | "running" | "completed";
    // Task Execution state (Perplexity-style)
    taskGroups: TaskGroup[];
    taskItems: Map<string, TaskItem>;
    taskChecklist: TaskChecklist | null;
    taskExecutionActive: boolean;
    savedPapers: Paper[];
    // Conversation history loading state
    _loadedConversationIds: Set<string>;
    _conversationsLoaded: boolean;
    _pendingDeletes: Set<string>;

    setActiveConversation: (id: string | null) => void;
    addMessage: (message: Message) => void;
    mergeWebSourcesMessage: (payload: {
        sources?: WebSource[];
        images?: WebImage[];
        provider?: string;
        imageProvider?: string;
        searchType?: string;
        query?: string;
    }) => void;
    updateLastAssistantMessage: (content: string) => void;
    updateLastAssistantThinking: (thinking: string) => void;
    updateLastAssistantRunMeta: (meta: import("./api").ChatRunMeta) => void;
    setStreaming: (streaming: boolean) => void;
    setStreamingContent: (content: string) => void;
    appendStreamingContent: (chunk: string) => void;
    setSelectedModel: (model: string) => void;
    createNewConversation: () => void;
    toggleSidebar: () => void;
    clearMessages: () => void;
    toggleStar: (id: string) => void;
    addThinkingStep: (step: string, state: "running" | "completed") => void;
    clearThinking: () => void;
    fetchModels: () => Promise<void>;
    attachThinkingToLastMessage: () => void;
    // Task Execution actions
    handleTaskGroup: (group: Record<string, unknown>) => void;
    handleTaskUpdate: (update: Record<string, unknown>) => void;
    handleTaskList: (list: Record<string, unknown>) => void;
    clearTaskExecution: () => void;
    savePaper: (paper: Paper) => void;
    removePaper: (paperId: string) => void;
    // Server-sync actions
    loadConversations: (token: string) => Promise<void>;
    loadConversationMessages: (conversationId: string, token: string) => Promise<void>;
    deleteConversation: (conversationId: string, token: string) => Promise<void>;
    setActiveConversationId: (id: string | null) => void;
    clearAllConversations: () => void;
}

function normalizeWebUrl(value: unknown): string {
    const raw = String(value || "").trim().replace(/^<|>$/g, "").replace(/[.,;:)\]}"']+$/g, "");
    if (!raw) return "";
    if (raw.startsWith("http://") || raw.startsWith("https://")) return raw;
    if (raw.startsWith("www.")) return `https://${raw}`;
    if (/^[A-Za-z0-9.-]+\.[A-Za-z]{2,}\/\S+$/.test(raw)) return `https://${raw}`;
    return "";
}

function mergeWebSources(existing: WebSource[] = [], incoming: WebSource[] = []): WebSource[] {
    const merged: WebSource[] = [];
    const byUrl = new Map<string, number>();

    [...existing, ...incoming].forEach((source) => {
        const url = normalizeWebUrl(source.url);
        if (!url) return;
        const key = url.toLowerCase().replace(/\/$/, "");
        if (byUrl.has(key)) {
            const current = merged[byUrl.get(key)!];
            if (!current.title && source.title) current.title = source.title;
            if (!current.snippet && source.snippet) current.snippet = source.snippet;
            current.evidenceQuality = mergeEvidenceQuality(current.evidenceQuality, source.evidenceQuality);
            return;
        }
        byUrl.set(key, merged.length);
        merged.push({ ...source, url });
    });

    return rankWebSources(merged);
}

function mergeWebImages(existing: WebImage[] = [], incoming: WebImage[] = []): WebImage[] {
    const merged: WebImage[] = [];
    const byUrl = new Map<string, number>();

    [...existing, ...incoming].forEach((image) => {
        const url = normalizeWebUrl(image.url) || String(image.url || "").trim();
        if (!url) return;
        const sourceUrl = normalizeWebUrl(image.sourceUrl);
        const sourceTitle = String(image.sourceTitle || "").trim();
        const key = url.toLowerCase().replace(/\/$/, "");
        if (byUrl.has(key)) {
            const current = merged[byUrl.get(key)!];
            if (!current.description && image.description) current.description = image.description;
            if (!current.sourceUrl && sourceUrl) current.sourceUrl = sourceUrl;
            if (!current.sourceTitle && sourceTitle) current.sourceTitle = sourceTitle;
            return;
        }
        byUrl.set(key, merged.length);
        merged.push({
            ...image,
            url,
            ...(sourceUrl ? { sourceUrl } : {}),
            ...(sourceTitle ? { sourceTitle } : {}),
        });
    });

    return merged;
}

function mergeLabel(existing?: string, incoming?: string): string | undefined {
    const label = String(incoming || "").trim();
    if (!label) return existing || undefined;
    if (!existing) return label;
    const parts = existing.split(" + ").map((part) => part.trim()).filter(Boolean);
    if (parts.includes(label)) return existing;
    return `${existing} + ${label}`;
}

function syncActiveConversationMessages(state: ChatStore, messages: Message[]): Conversation[] {
    if (!state.activeConversationId) return state.conversations;
    return state.conversations.map((conversation) =>
        conversation.id === state.activeConversationId
            ? {
                ...conversation,
                messages,
                updatedAt: new Date(),
                model: state.selectedModel,
            }
            : conversation
    );
}

function serverMessageToLocal(msg: ServerMessage, index: number): Message[] {
    const base: Message = {
        id: `srv-${index}-${Date.now().toString(36)}`,
        role: msg.role as Message["role"],
        content: msg.role === "assistant" ? sanitizeAssistantContent(msg.content || "") : msg.content || "",
        type: (msg.type || "text") as Message["type"],
        timestamp: new Date(),
    };

    // Restore thinking steps from metadata (shown in "Thinking" widget)
    const meta = msg.metadata || {};
    if (meta.thinkingSteps && Array.isArray(meta.thinkingSteps)) {
        base.thinkingSteps = (meta.thinkingSteps as { step: string; state: string }[]).map(
            (s) => ({ text: s.step, status: s.state as "running" | "completed" })
        );
    }
    if (meta.thinking) {
        base.thinking = sanitizeAssistantContent(meta.thinking as string);
    }
    if (meta.thinkingDuration || meta.thinking_duration) {
        base.thinkingDuration = Number(meta.thinkingDuration || meta.thinking_duration);
    }
    if (meta.runMeta && typeof meta.runMeta === "object") {
        base.runMeta = meta.runMeta as Message["runMeta"];
    }

    const messages: Message[] = [base];

    // Restore DataTableCard as a separate "data" message (same as live SSE)
    if (meta.dataTable) {
        messages.push({
            id: `srv-${index}-dt-${Date.now().toString(36)}`,
            role: "assistant",
            content: "",
            type: "data",
            timestamp: new Date(),
            dataTable: meta.dataTable as Message["dataTable"],
        });
    }

    // Restore Paper cards as a separate "papers" message
    if (meta.papers && Array.isArray(meta.papers) && meta.papers.length > 0) {
        // Map backend field names to frontend Paper interface
        const mappedPapers: Paper[] = (meta.papers as Record<string, unknown>[]).map((p, i) => ({
            id: (p.bibcode as string) || (p.id as string) || `paper-${i}`,
            title: (p.title as string) || "Untitled",
            authors: (p.authors as string) || "Unknown",
            year: Number(p.year) || 0,
            journal: (p.journal as string) || (p.pub as string) || "",
            citationCount: Number(p.citations ?? p.citationCount ?? p.citation_count ?? 0),
            type: (p.type as Paper["type"]) || "radio",
            bibcode: (p.bibcode as string) || undefined,
            doi: (p.doi as string) || undefined,
            abstract: (p.abstract as string) || undefined,
            // OpenAlex enrichment
            fwci: p.fwci != null ? Number(p.fwci) : null,
            citationPercentile: p.citation_percentile != null ? Number(p.citation_percentile) : null,
            isTop1Percent: Boolean(p.is_top_1_percent || p.isTop1Percent),
            isTop10Percent: Boolean(p.is_top_10_percent || p.isTop10Percent),
            funders: Array.isArray(p.funders) ? (p.funders as { name: string; id: string }[]) : undefined,
            oaPdfUrl: (p.oa_pdf_url as string) || (p.oaPdfUrl as string) || undefined,
            observationLinks: Array.isArray(p.observation_links)
                ? (p.observation_links as Record<string, unknown>[]).map((link) => ({
                    identifier: String(link.identifier || ""),
                    identifierType: String(link.identifier_type || link.identifierType || ""),
                    relation: String(link.relation || ""),
                    confidence: String(link.confidence || ""),
                    adsQuery: String(link.ads_query || link.adsQuery || ""),
                })).filter((link) => link.identifier)
                : Array.isArray(p.observationLinks)
                    ? p.observationLinks as Paper["observationLinks"]
                    : undefined,
        }));
        messages.push({
            id: `srv-${index}-pp-${Date.now().toString(36)}`,
            role: "assistant",
            content: "Here are the relevant papers I found:",
            type: "papers",
            timestamp: new Date(),
            papers: mappedPapers,
        });
    }

    // Restore Notebook as a separate "notebook" message
    if (meta.notebook) {
        const nb = meta.notebook as Record<string, unknown>;
        messages.push({
            id: `srv-${index}-nb-${Date.now().toString(36)}`,
            role: "assistant",
            content: `I've generated a Jupyter Notebook for your analysis: **${nb.title || "Dynamic Notebook"}**`,
            type: "notebook",
            timestamp: new Date(),
            notebookData: nb as unknown as Message["notebookData"],
        });
    }
    // Restore rendered image as a separate "image" message
    if (meta.image) {
        const img = meta.image as { url: string; caption: string };
        const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const imageUrl = img.url.startsWith("http") ? img.url : `${apiBase}${img.url}`;
        messages.push({
            id: `srv-${index}-img-${Date.now().toString(36)}`,
            role: "assistant",
            content: img.caption || "",
            type: "image",
            timestamp: new Date(),
            imageUrl: imageUrl,
            imageCaption: img.caption || "",
        });
    }

    // Restore web source cards from metadata (same as live SSE)
    if (
        (meta.webSources && Array.isArray(meta.webSources) && meta.webSources.length > 0) ||
        (meta.webImages && Array.isArray(meta.webImages) && meta.webImages.length > 0)
    ) {
        messages.push({
            id: `srv-${index}-web-${Date.now().toString(36)}`,
            role: "assistant",
            content: "",
            type: "web_sources",
            timestamp: new Date(),
            webSources: (meta.webSources as Message["webSources"]) || [],
            webImages: (meta.webImages as Message["webImages"]) || [],
            webProvider: meta.webProvider as string | undefined,
            webImageProvider: meta.webImageProvider as string | undefined,
            webSearchType: meta.webSearchType as string | undefined,
            webQuery: meta.webQuery as string | undefined,
        });
    }

    return messages;
}

function serverConvToLocal(conv: ServerConversation): Conversation {
    return {
        id: conv.id,
        title: conv.title || "New Chat",
        createdAt: new Date(conv.created_at),
        updatedAt: new Date(conv.updated_at),
        messages: [],
        model: conv.model || "",
    };
}

export const useChatStore = create<ChatStore>((set, get) => ({
    conversations: [],
    activeConversationId: null,
    messages: [],
    isStreaming: false,
    streamingContent: "",
    selectedModel: "gpt-oss-120b",
    availableModels: DEFAULT_AVAILABLE_MODELS,
    sidebarOpen: true,
    thinkingSteps: [],
    thinkingStatus: "idle",
    // Task Execution initial state
    taskGroups: [],
    taskItems: new Map(),
    taskChecklist: null,
    taskExecutionActive: false,
    savedPapers: [],
    _loadedConversationIds: new Set(),
    _conversationsLoaded: false,
    _pendingDeletes: new Set(),

    setActiveConversation: (id) => set((state) => {
        if (state.activeConversationId && state.messages.length > 0) {
            const updatedConversations = state.conversations.map(c =>
                c.id === state.activeConversationId
                    ? { ...c, messages: state.messages, updatedAt: new Date(), model: state.selectedModel }
                    : c
            );
            const conv = updatedConversations.find((c) => c.id === id);
            return {
                conversations: updatedConversations,
                activeConversationId: id,
                messages: conv?.messages || [],
                selectedModel: conv?.model || state.selectedModel || "gpt-oss-120b",
            };
        }
        const conv = state.conversations.find((c) => c.id === id);
        return {
            activeConversationId: id,
            messages: conv?.messages || [],
            selectedModel: conv?.model || state.selectedModel || "gpt-oss-120b",
        };
    }),

    addMessage: (message) => set((state) => {
        const newMessages = [...state.messages, message];

        let updatedConversations = [...state.conversations];
        if (state.activeConversationId) {
            const existingConv = updatedConversations.find(c => c.id === state.activeConversationId);
            if (existingConv) {
                updatedConversations = updatedConversations.map(c =>
                    c.id === state.activeConversationId
                        ? {
                            ...c,
                            messages: newMessages,
                            updatedAt: new Date(),
                            title: message.role === "user" && c.messages.length === 0
                                ? message.content.slice(0, 60) + (message.content.length > 60 ? "…" : "")
                                : c.title,
                        }
                        : c
                );
            } else {
                const title = message.role === "user"
                    ? message.content.slice(0, 60) + (message.content.length > 60 ? "…" : "")
                    : "New Chat";
                updatedConversations = [
                    {
                        id: state.activeConversationId,
                        title,
                        createdAt: new Date(),
                        updatedAt: new Date(),
                        messages: newMessages,
                        model: state.selectedModel,
                    },
                    ...updatedConversations,
                ];
            }
        }

        return { messages: newMessages, conversations: updatedConversations };
    }),

    mergeWebSourcesMessage: (payload) => set((state) => {
        const incomingSources = payload.sources || [];
        const incomingImages = payload.images || [];
        if (incomingSources.length === 0 && incomingImages.length === 0) return {};

        const messages = [...state.messages];
        let lastUserIdx = -1;
        for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === "user") {
                lastUserIdx = i;
                break;
            }
        }

        let existingIdx = -1;
        for (let i = messages.length - 1; i > lastUserIdx; i--) {
            if (messages[i].role === "assistant" && messages[i].type === "web_sources") {
                existingIdx = i;
                break;
            }
        }

        if (existingIdx >= 0) {
            const existing = messages[existingIdx];
            messages[existingIdx] = {
                ...existing,
                webSources: mergeWebSources(existing.webSources || [], incomingSources),
                webImages: mergeWebImages(existing.webImages || [], incomingImages),
                webProvider: mergeLabel(existing.webProvider, payload.provider),
                webImageProvider: mergeLabel(existing.webImageProvider, payload.imageProvider),
                webSearchType: mergeLabel(existing.webSearchType, payload.searchType),
                webQuery: existing.webQuery || payload.query,
            };
        } else {
            messages.push({
                id: `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
                role: "assistant",
                content: "",
                type: "web_sources",
                timestamp: new Date(),
                webSources: mergeWebSources([], incomingSources),
                webImages: mergeWebImages([], incomingImages),
                webProvider: payload.provider,
                webImageProvider: payload.imageProvider,
                webSearchType: payload.searchType,
                webQuery: payload.query,
            });
        }

        return {
            messages,
            conversations: syncActiveConversationMessages(state, messages),
        };
    }),

    updateLastAssistantMessage: (content) => set((state) => {
        return { messages: updateLastAssistantContent(state.messages, content) };
    }),

    updateLastAssistantThinking: (thinking) => set((state) => {
        return {
            messages: updateAssistantThinking(state.messages, thinking),
            thinkingStatus: "running",
        };
    }),

    updateLastAssistantRunMeta: (meta) => set((state) => {
        const messages = [...state.messages];
        const index = findLastAssistantTextIndex(messages);
        if (index < 0) return {};
        messages[index] = { ...messages[index], runMeta: meta };
        return {
            messages,
            conversations: syncActiveConversationMessages(state, messages),
        };
    }),

    setStreaming: (streaming) => set({ isStreaming: streaming }),
    setStreamingContent: (content) => set({ streamingContent: content }),
    appendStreamingContent: (chunk) => set((state) => ({ streamingContent: state.streamingContent + chunk })),
    setSelectedModel: (model) => set({ selectedModel: model }),

    createNewConversation: () => set((state) => {
        let updatedConversations = [...state.conversations];
        if (state.activeConversationId && state.messages.length > 0) {
            updatedConversations = updatedConversations.map(c =>
                c.id === state.activeConversationId
                    ? { ...c, messages: state.messages, updatedAt: new Date(), model: state.selectedModel }
                    : c
            );
        }
        return {
            conversations: updatedConversations,
            activeConversationId: null,
            messages: [],
            streamingContent: "",
        };
    }),

    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
    clearMessages: () => set({ messages: [], activeConversationId: null }),

    toggleStar: (id) => set((state) => ({
        conversations: state.conversations.map(c =>
            c.id === id ? { ...c, isStarred: !c.isStarred } : c
        )
    })),

    addThinkingStep: (step, state) => set((s) => {
        const steps = [...s.thinkingSteps];
        const existingIdx = steps.findIndex(t => t.text === step);
        if (existingIdx >= 0) {
            steps[existingIdx] = { ...steps[existingIdx], status: state };
        } else {
            steps.push({ text: step, status: state });
        }
        return { thinkingSteps: steps, thinkingStatus: "running" };
    }),

    clearThinking: () => set({ thinkingSteps: [], thinkingStatus: "idle" }),

    fetchModels: async () => {
        try {
            const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            const res = await fetch(`${API_BASE}/api/models`);
            if (res.ok) {
                const data = await res.json();
                set({ availableModels: mergeAvailableModels(data.models) });
            }
        } catch { /* keep defaults */ }
    },

    attachThinkingToLastMessage: () => set((s) => {
        const targetAssistantIdx = findLastAssistantTextIndex(s.messages);
        const hasThinkingText = targetAssistantIdx >= 0 && !!s.messages[targetAssistantIdx].thinking;
        if (s.thinkingSteps.length === 0 && !hasThinkingText) return {};
        return {
            messages: attachThinkingStepsToLastAssistant(s.messages, s.thinkingSteps).messages,
            thinkingStatus: "completed",
        };
    }),

    // ── Task Execution Actions ──────────────────────────────────────────────

    handleTaskGroup: (group) => set((s) => {
        const newGroup: TaskGroup = {
            groupId: group.groupId as string,
            title: group.title as string,
            taskIds: (group.taskIds as string[]) || [],
            tasks: ((group.tasks as Array<Record<string, unknown>>) || []).map(t => ({
                id: t.id as string,
                description: t.description as string,
                status: "pending" as const,
                agentType: t.agentType as string,
                icon: t.icon as string,
                groupId: group.groupId as string,
            })),
        };
        const items = new Map(s.taskItems);
        for (const task of newGroup.tasks) {
            if (!items.has(task.id)) {
                items.set(task.id, task);
            }
        }
        return {
            taskGroups: [...s.taskGroups, newGroup],
            taskItems: items,
            taskExecutionActive: true,
        };
    }),

    handleTaskUpdate: (update) => set((s) => {
        const items = new Map(s.taskItems);
        const taskId = update.taskId as string;
        const existing = items.get(taskId);
        items.set(taskId, {
            id: taskId,
            description: (update.title as string) || existing?.description || taskId,
            status: (update.status as TaskItem["status"]) || "running",
            agentType: (update.agentType as string) || existing?.agentType || "general",
            icon: (update.icon as string) || existing?.icon,
            detail: (update.detail as string) || existing?.detail,
            groupId: (update.groupId as string) || existing?.groupId,
        });
        return { taskItems: items };
    }),

    handleTaskList: (list) => set(() => {
        const checklist: TaskChecklist = {
            title: (list.title as string) || "Execution Plan",
            tasks: ((list.tasks as Array<Record<string, unknown>>) || []).map(t => ({
                id: t.id as string,
                description: t.description as string,
                status: (t.status as TaskItem["status"]) || "pending",
                agentType: (t.agentType as string) || "general",
            })),
        };
        return { taskChecklist: checklist, taskExecutionActive: true };
    }),

    clearTaskExecution: () => set({
        taskGroups: [],
        taskItems: new Map(),
        taskChecklist: null,
        taskExecutionActive: false,
    }),

    savePaper: (paper) => set((s) => {
        if (s.savedPapers.some(p => p.id === paper.id)) return {};
        return { savedPapers: [...s.savedPapers, paper] };
    }),

    removePaper: (paperId) => set((s) => ({
        savedPapers: s.savedPapers.filter(p => p.id !== paperId),
    })),

    // ── Server-sync Actions ─────────────────────────────────────────────────

    loadConversations: async (token: string) => {
        const serverConvos = await apiFetchConversations(token);
        const localConvos = serverConvos.map(serverConvToLocal);

        // Preserve locally-cached messages when refreshing the list
        const existing = get().conversations;
        const pendingDeletes = get()._pendingDeletes;
        const msgCache = new Map<string, Message[]>();
        for (const c of existing) {
            if (c.messages.length > 0) msgCache.set(c.id, c.messages);
        }

        // Filter out conversations that are currently being deleted
        const merged = localConvos
            .filter(c => !pendingDeletes.has(c.id))
            .map(c => {
                const cached = msgCache.get(c.id);
                return cached ? { ...c, messages: cached } : c;
            });

        set({
            conversations: merged,
            _conversationsLoaded: true,
        });
    },

    loadConversationMessages: async (conversationId: string, token: string) => {
        // Always fetch from server — the local cache may have been
        // invalidated by loadConversations refreshing the list.
        const serverMsgs = await apiFetchMessages(conversationId, token);
        const localMsgs = serverMsgs.flatMap(serverMessageToLocal);

        set((state) => {
            // Update the conversation's messages cache
            const updatedConvos = state.conversations.map(c =>
                c.id === conversationId ? { ...c, messages: localMsgs } : c
            );

            // If this is still the active conversation, set messages
            const updates: Partial<ChatStore> = {
                conversations: updatedConvos,
            };
            if (state.activeConversationId === conversationId) {
                (updates as { messages: Message[] }).messages = localMsgs;
            }
            return updates as ChatStore;
        });
    },

    deleteConversation: async (conversationId: string, token: string) => {
        // Optimistic update: remove from UI immediately for instant feedback
        const prevState = get();
        const removedConv = prevState.conversations.find(c => c.id === conversationId);

        set((state) => {
            const newLoaded = new Set(state._loadedConversationIds);
            newLoaded.delete(conversationId);
            // Track this ID so loadConversations won't re-add it
            const newPending = new Set(state._pendingDeletes);
            newPending.add(conversationId);
            return {
                conversations: state.conversations.filter(c => c.id !== conversationId),
                activeConversationId: state.activeConversationId === conversationId ? null : state.activeConversationId,
                messages: state.activeConversationId === conversationId ? [] : state.messages,
                _loadedConversationIds: newLoaded,
                _pendingDeletes: newPending,
            };
        });

        // Call API in background
        const ok = await deleteConversationApi(conversationId, token);
        // Clear from pending deletes either way
        set((state) => {
            const newPending = new Set(state._pendingDeletes);
            newPending.delete(conversationId);
            return { _pendingDeletes: newPending };
        });

        if (!ok && removedConv) {
            // Restore the conversation if the API failed
            set((state) => ({
                conversations: [...state.conversations, removedConv].sort(
                    (a, b) => b.updatedAt.getTime() - a.updatedAt.getTime()
                ),
            }));
        }
    },

    setActiveConversationId: (id) => set({ activeConversationId: id }),

    clearAllConversations: () => set({
        conversations: [],
        activeConversationId: null,
        messages: [],
        _loadedConversationIds: new Set(),
        _conversationsLoaded: false,
    }),
}));
