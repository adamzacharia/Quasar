"use client";

import { create } from "zustand";
import type { Conversation, Message, Paper, TaskGroup, TaskItem, TaskChecklist } from "./types";
import type { ThoughtStep } from "@/components/ThoughtProcessWidget";
import {
    fetchConversations as apiFetchConversations,
    fetchConversationMessages as apiFetchMessages,
    deleteConversationApi,
    type ServerConversation,
    type ServerMessage,
} from "./api";

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
    updateLastAssistantMessage: (content: string) => void;
    updateLastAssistantThinking: (thinking: string) => void;
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

function generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function serverMessageToLocal(msg: ServerMessage, index: number): Message[] {
    const base: Message = {
        id: `srv-${index}-${Date.now().toString(36)}`,
        role: msg.role as Message["role"],
        content: msg.content || "",
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
        base.thinking = meta.thinking as string;
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

    return messages;
}

function serverConvToLocal(conv: ServerConversation): Conversation {
    return {
        id: conv.id,
        title: conv.title || "New Chat",
        createdAt: new Date(conv.created_at),
        updatedAt: new Date(conv.updated_at),
        messages: [],
        model: "",
    };
}

export const useChatStore = create<ChatStore>((set, get) => ({
    conversations: [],
    activeConversationId: null,
    messages: [],
    isStreaming: false,
    streamingContent: "",
    selectedModel: "gpt-5.4-mini",
    availableModels: [
        "gpt-5.4-mini", "gpt-4.1", "gpt-4o-mini", "deepseek-v4-pro", "deepseek-v4-flash",
    ],
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
                    ? { ...c, messages: state.messages, updatedAt: new Date() }
                    : c
            );
            const conv = updatedConversations.find((c) => c.id === id);
            return {
                conversations: updatedConversations,
                activeConversationId: id,
                messages: conv?.messages || [],
            };
        }
        const conv = state.conversations.find((c) => c.id === id);
        return { activeConversationId: id, messages: conv?.messages || [] };
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

    updateLastAssistantMessage: (content) => set((state) => {
        const msgs = [...state.messages];
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === "assistant" && msgs[i].type === "text") {
                msgs[i] = { ...msgs[i], content };
                break;
            }
        }
        return { messages: msgs };
    }),

    updateLastAssistantThinking: (thinking) => set((state) => {
        const msgs = [...state.messages];
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === "assistant" && msgs[i].type === "text") {
                msgs[i] = { ...msgs[i], thinking };
                break;
            }
        }
        return { messages: msgs };
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
                    ? { ...c, messages: state.messages, updatedAt: new Date() }
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
                if (data.models?.length) set({ availableModels: data.models });
            }
        } catch { /* keep defaults */ }
    },

    attachThinkingToLastMessage: () => set((s) => {
        if (s.thinkingSteps.length === 0) return {};
        // Mark any still-running steps as completed (streaming is done)
        const finalSteps = s.thinkingSteps.map(step =>
            step.status === "running" ? { ...step, status: "completed" as const } : step
        );
        const msgs = [...s.messages];
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === "assistant" && msgs[i].type === "text") {
                msgs[i] = { ...msgs[i], thinkingSteps: finalSteps };
                break;
            }
        }
        return { messages: msgs };
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
