"use client";

import { create } from "zustand";
import type { Conversation, Message } from "./types";
import type { ThoughtStep } from "@/components/ThoughtProcessWidget";

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
    setActiveConversation: (id: string | null) => void;
    addMessage: (message: Message) => void;
    updateLastAssistantMessage: (content: string) => void;
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
}

function generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

export const useChatStore = create<ChatStore>((set, get) => ({
    conversations: [],
    activeConversationId: null,
    messages: [],
    isStreaming: false,
    streamingContent: "",
    selectedModel: "gpt-4o-mini",
    availableModels: [
        // OpenAI
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo",
        // Gemini free tier
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-3-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    sidebarOpen: true,
    thinkingSteps: [],
    thinkingStatus: "idle",

    setActiveConversation: (id) => set((state) => {
        // Save current messages to current conversation
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

        // If this is the first user message, create or update conversation
        let updatedConversations = [...state.conversations];
        if (state.activeConversationId) {
            const existingConv = updatedConversations.find(c => c.id === state.activeConversationId);
            if (existingConv) {
                // Update existing conversation
                updatedConversations = updatedConversations.map(c =>
                    c.id === state.activeConversationId
                        ? {
                            ...c,
                            messages: newMessages,
                            updatedAt: new Date(),
                            // Update title from first user message
                            title: message.role === "user" && c.messages.length === 0
                                ? message.content.slice(0, 60) + (message.content.length > 60 ? "…" : "")
                                : c.title,
                        }
                        : c
                );
            } else {
                // Create new conversation
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
        // Find the last assistant message (could be anywhere if tool calls were inserted)
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === "assistant" && msgs[i].type === "text") {
                msgs[i] = { ...msgs[i], content };
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
        // Save current conversation messages first
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
        // If this step already exists (running -> completed), update it
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
        const msgs = [...s.messages];
        for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === "assistant" && msgs[i].type === "text") {
                msgs[i] = { ...msgs[i], thinkingSteps: [...s.thinkingSteps] };
                break;
            }
        }
        return { messages: msgs };
    }),
}));
