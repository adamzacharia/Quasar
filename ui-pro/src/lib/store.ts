"use client";

import { create } from "zustand";
import type { Conversation, Message, Paper, TaskGroup, TaskItem, TaskChecklist, WebImage, WebSource } from "./types";
import type { ThoughtStep } from "@/components/ThoughtProcessWidget";
import {
    fetchConversations as apiFetchConversations,
    fetchConversationMessages as apiFetchMessages,
    fetchBlockRatings as apiFetchBlockRatings,
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
import { registerSessionScrubber } from "./auth-store";
import { currentAuthGeneration } from "./auth-generation";
import { DEFAULT_AVAILABLE_MODELS, mergeAvailableModels } from "./models";
import { normalizeHipsImageMeta } from "./hips-imagery";

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
    // Eval mode (Feature 4) — per-user client state, admin-gated at the toggle.
    evalMode: boolean;
    /** blockId -> this user's CONFIRMED 1-5 rating (the server stored it).
     *  The gate reads only this — see pendingRatings. */
    blockRatings: Record<string, number>;
    /** blockId -> a rating whose POST is still in flight. Shown optimistically,
     *  but deliberately NOT counted by the gate. */
    pendingRatings: Record<string, number>;
    /** blockId -> mount count of its rating widget. The gate may only demand a
     *  star on a block that is actually on screen; see registerBlock. */
    visibleBlocks: Record<string, number>;
    // Conversation history loading state
    _loadedConversationIds: Set<string>;
    _conversationsLoaded: boolean;
    _pendingDeletes: Set<string>;

    setActiveConversation: (id: string | null) => void;
    /* UI-01: every stream-fed action takes an optional ownerConversationId —
       when the stream's owner is no longer the active conversation, the update
       is routed into the owner's stored messages instead of the live list
       (the same CX-18 owner routing updateLastAssistantToolTrace pioneered). */
    addMessage: (message: Message, ownerConversationId?: string | null) => void;
    mergeWebSourcesMessage: (payload: {
        sources?: WebSource[];
        images?: WebImage[];
        provider?: string;
        imageProvider?: string;
        searchType?: string;
        query?: string;
    }, ownerConversationId?: string | null) => void;
    updateLastAssistantMessage: (content: string, ownerConversationId?: string | null) => void;
    updateLastAssistantThinking: (thinking: string, ownerConversationId?: string | null) => void;
    updateLastAssistantRunMeta: (meta: import("./api").ChatRunMeta, ownerConversationId?: string | null) => void;
    markLastAssistantRunFailed: (errorCode?: string, ownerConversationId?: string | null) => void;
    updateLastAssistantToolTrace: (calls: import("./api").ToolTraceCall[], ownerConversationId?: string | null) => void;
    updateLastAssistantUsage: (totalTokens: number, durationMs?: number, ownerConversationId?: string | null) => void;
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
    // Eval mode actions
    setEvalMode: (on: boolean) => void;
    setBlockRating: (blockId: string, rating: number) => void;
    beginRating: (blockId: string, rating: number) => void;
    resolveRating: (blockId: string, rating: number | null) => void;
    loadBlockRatings: (conversationId: string) => Promise<void>;
    registerBlock: (blockId: string) => void;
    unregisterBlock: (blockId: string) => void;
    // Server-sync actions (auth rides the httpOnly cookie; token is legacy-optional)
    loadConversations: (token?: string) => Promise<void>;
    loadConversationMessages: (conversationId: string, token?: string) => Promise<void>;
    deleteConversation: (conversationId: string, token?: string) => Promise<void>;
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

/* UI-01: true when a stream update must be routed into its OWNER conversation's
   stored messages instead of the live list — i.e. the stream outlived a
   conversation switch. Mirrors the CX-18 guard that used to live only in
   updateLastAssistantToolTrace; every stream-fed store action shares it now, so
   a mid-stream switch can no longer cross-contaminate conversations. */
function isOwnerRouted(
    state: ChatStore,
    ownerConversationId?: string | null,
): ownerConversationId is string {
    return Boolean(ownerConversationId && ownerConversationId !== state.activeConversationId);
}

/* UI-01: apply `patch` to the owner conversation's STORED message list.
   setActiveConversation snapshots the live messages into the conversation
   before switching, so the partial turn is there to patch; the persisted
   server copy re-asserts the same content on reload. A `null` patch result
   means "nothing to change". */
function patchOwnerConversation(
    state: ChatStore,
    ownerConversationId: string,
    patch: (messages: Message[]) => Message[] | null,
): Partial<ChatStore> {
    const conversations = state.conversations.map((conversation) => {
        if (conversation.id !== ownerConversationId) return conversation;
        const patched = patch(conversation.messages || []);
        if (!patched) return conversation;
        return { ...conversation, messages: patched, updatedAt: new Date() };
    });
    return { conversations };
}

/* UI-01: the web-sources merge, extracted pure so the live path and the
   owner-routed path run identical logic. Returns null when there is nothing
   to merge. */
function mergeWebSourcesIntoMessages(
    messages: Message[],
    payload: {
        sources?: WebSource[];
        images?: WebImage[];
        provider?: string;
        imageProvider?: string;
        searchType?: string;
        query?: string;
    },
): Message[] | null {
    const incomingSources = payload.sources || [];
    const incomingImages = payload.images || [];
    if (incomingSources.length === 0 && incomingImages.length === 0) return null;

    const next = [...messages];
    let lastUserIdx = -1;
    for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "user") {
            lastUserIdx = i;
            break;
        }
    }

    let existingIdx = -1;
    for (let i = next.length - 1; i > lastUserIdx; i--) {
        if (next[i].role === "assistant" && next[i].type === "web_sources") {
            existingIdx = i;
            break;
        }
    }

    if (existingIdx >= 0) {
        const existing = next[existingIdx];
        next[existingIdx] = {
            ...existing,
            webSources: mergeWebSources(existing.webSources || [], incomingSources),
            webImages: mergeWebImages(existing.webImages || [], incomingImages),
            webProvider: mergeLabel(existing.webProvider, payload.provider),
            webImageProvider: mergeLabel(existing.webImageProvider, payload.imageProvider),
            webSearchType: mergeLabel(existing.webSearchType, payload.searchType),
            webQuery: existing.webQuery || payload.query,
        };
    } else {
        next.push({
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

    return next;
}

/* Eval mode is per-user client state (same idiom as theme-store). The flag
   gates the composer, so it must survive a reload — a half-labeled session
   silently un-gating itself would let unrated turns into the label set. */
const EVAL_MODE_KEY = "quasar_eval_mode";

function getInitialEvalMode(): boolean {
    if (typeof window === "undefined") return false;
    try {
        return localStorage.getItem(EVAL_MODE_KEY) === "1";
    } catch {
        return false;
    }
}

function persistEvalMode(on: boolean) {
    if (typeof window === "undefined") return;
    try {
        localStorage.setItem(EVAL_MODE_KEY, on ? "1" : "0");
    } catch {
        // Private mode / storage disabled — the flag just won't survive reload.
    }
}

/* Block ids come from the server, minted once at emission and persisted in
   rich_meta. We never recompute them here: sse.py's card ordering depends on
   eager-vs-done emission and dedup, so a client-side recomputation could drift
   and orphan a rating that is already keyed to the stored id.

   `fallback` is for turns that predate stable ids (their rich_meta has none).
   It must still be reload-stable, which the old `Date.now()` ids were not —
   hence index-derived, not clock-derived. Such blocks are unrateable; the star
   widget hides itself when `blockId` is absent. */
function blockIdOr(persisted: unknown, fallback: string): string {
    return typeof persisted === "string" && persisted ? persisted : fallback;
}

/* One papers grid -> one "papers" message. Extracted so a turn that searched
   twice can rebuild BOTH grids: rich_meta.papersGroups holds every grid, while
   papers/papersRequest/papersBlockId are only the back-compat primary. */
function buildPapersMessage(
    rawPapers: Record<string, unknown>[],
    opts: { id: string; blockId?: string; request?: Message["request"]; runMeta?: Message["runMeta"] },
): Message {
    const mappedPapers: Paper[] = rawPapers.map((p, i) => ({
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
    return {
        id: opts.id,
        role: "assistant",
        content: "Here are the relevant papers I found:",
        type: "papers",
        timestamp: new Date(),
        papers: mappedPapers,
        // The ADS query behind the grid (Feature 1) — persisted, so the
        // provenance block survives a reload.
        request: opts.request,
        runMeta: opts.runMeta,
        blockId: opts.blockId,
        blockKind: "papers",
    };
}

function serverMessageToLocal(msg: ServerMessage, index: number): Message[] {
    const meta0 = msg.metadata || {};
    const runMeta0 = (meta0.runMeta as { text_block_id?: string } | undefined) || undefined;
    const base: Message = {
        id: blockIdOr(runMeta0?.text_block_id, `srv-${index}`),
        role: msg.role as Message["role"],
        content: msg.role === "assistant" ? sanitizeAssistantContent(msg.content || "") : msg.content || "",
        type: (msg.type || "text") as Message["type"],
        timestamp: new Date(),
        blockId: typeof runMeta0?.text_block_id === "string" ? runMeta0.text_block_id : undefined,
        blockKind: msg.role === "assistant" ? "text" : undefined,
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
    // Raw request provenance (Feature 1) — without this the exact queries are
    // dropped on history replay, which is the whole point of persisting them.
    if (Array.isArray(meta.toolTrace) && meta.toolTrace.length > 0) {
        base.toolTrace = meta.toolTrace as Message["toolTrace"];
    }

    const messages: Message[] = [base];
    // Every card of this turn shares the turn's run — the rating POST needs
    // run_id to join to chat_runs (model/tokens/cost) in the eval export, and
    // rich_meta only stores runMeta once, on the message as a whole.
    const turnRunMeta = base.runMeta;

    // Restore DataTableCard(s) as separate "data" messages (same as live SSE).
    // Prefer `dataTables` — sse.py persists the full list for multi-target
    // turns and `dataTable` is only its back-compat first element, so reading
    // `dataTable` alone silently dropped every table after the first on reload
    // (and with it, any rating those tables had earned).
    const storedTables = Array.isArray(meta.dataTables) && meta.dataTables.length > 0
        ? (meta.dataTables as Record<string, unknown>[])
        : meta.dataTable
            ? [meta.dataTable as Record<string, unknown>]
            : [];
    storedTables.forEach((table, tableIndex) => {
        if (!table) return;
        messages.push({
            id: blockIdOr(table.blockId, `srv-${index}-dt-${tableIndex}`),
            role: "assistant",
            content: "",
            type: "data",
            timestamp: new Date(),
            dataTable: table as unknown as Message["dataTable"],
            runMeta: turnRunMeta,
            blockId: typeof table.blockId === "string" ? table.blockId : undefined,
            blockKind: "data",
        });
    });

    // Restore Paper cards as separate "papers" messages. papersGroups carries
    // every grid of a multi-search turn (papers/papersBlockId are only the
    // back-compat primary), so a rating on the first grid isn't orphaned.
    const paperGroups: { papers: unknown[]; request?: unknown; blockId?: string }[] =
        Array.isArray(meta.papersGroups) && meta.papersGroups.length > 0
            ? (meta.papersGroups as { papers: unknown[]; request?: unknown; blockId?: string }[])
            : (meta.papers && Array.isArray(meta.papers) && meta.papers.length > 0
                ? [{ papers: meta.papers as unknown[], request: meta.papersRequest, blockId: meta.papersBlockId as string | undefined }]
                : []);
    paperGroups.forEach((group, groupIndex) => {
        const rawPapers = Array.isArray(group?.papers) ? group.papers : [];
        if (rawPapers.length === 0) return;
        messages.push(buildPapersMessage(rawPapers as Record<string, unknown>[], {
            id: blockIdOr(group.blockId, `srv-${index}-pp-${groupIndex}`),
            blockId: typeof group.blockId === "string" ? group.blockId : undefined,
            request: group.request as Message["request"],
            runMeta: turnRunMeta,
        }));
    });

    // Restore Notebook(s). `notebooks` carries every notebook of a turn that
    // produced more than one (Conductor emits its own alongside the normal
    // path); `notebook` is only the back-compat last one.
    const storedNotebooks = Array.isArray(meta.notebooks) && meta.notebooks.length > 0
        ? (meta.notebooks as Record<string, unknown>[])
        : meta.notebook
            ? [meta.notebook as Record<string, unknown>]
            : [];
    storedNotebooks.forEach((nb, nbIndex) => {
        if (!nb) return;
        messages.push({
            id: blockIdOr(nb.blockId, `srv-${index}-nb-${nbIndex}`),
            role: "assistant",
            content: `I've generated a Jupyter Notebook for your analysis: **${nb.title || "Dynamic Notebook"}**`,
            type: "notebook",
            timestamp: new Date(),
            notebookData: nb as unknown as Message["notebookData"],
            runMeta: turnRunMeta,
            blockId: typeof nb.blockId === "string" ? nb.blockId : undefined,
            blockKind: "notebook",
        });
    });
    // Restore rendered images as separate "image" messages.
    const storedImages = Array.isArray(meta.images) && meta.images.length > 0
        ? meta.images
        : meta.image
            ? [meta.image]
            : [];
    if (storedImages.length > 0) {
        const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const seenImageUrls = new Set<string>();
        storedImages.forEach((rawImage, imageIndex) => {
            if (rawImage === null || rawImage === undefined) return;
            if (typeof rawImage === "object") {
                const img = rawImage as {
                    url?: string; caption?: string; meta?: unknown;
                    request?: Message["request"]; blockId?: string; blockKind?: string;
                };
                const rawUrl = String(img.url || "").trim();
                if (rawUrl === "" || seenImageUrls.has(rawUrl)) return;
                seenImageUrls.add(rawUrl);
                const imageUrl = rawUrl.startsWith("http") || rawUrl.startsWith("data:") ? rawUrl : apiBase + rawUrl;
                messages.push({
                    id: blockIdOr(img.blockId, "srv-" + index + "-img-" + imageIndex),
                    role: "assistant",
                    content: img.caption || "",
                    type: "image",
                    timestamp: new Date(),
                    imageUrl,
                    imageCaption: img.caption || "",
                    imageMeta: normalizeHipsImageMeta(img.meta),
                    request: img.request,   // Feature 1: survives reload
                    runMeta: turnRunMeta,
                    blockId: typeof img.blockId === "string" ? img.blockId : undefined,
                    // A plotly figure persists into `images` with its PNG; the
                    // stored kind is what decides which star row it re-lights.
                    blockKind: img.blockKind === "plotly" ? "plotly" : "image",
                });
            }
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
    sidebarOpen: false,
    thinkingSteps: [],
    thinkingStatus: "idle",
    // Task Execution initial state
    taskGroups: [],
    taskItems: new Map(),
    taskChecklist: null,
    taskExecutionActive: false,
    savedPapers: [],
    evalMode: getInitialEvalMode(),
    blockRatings: {},
    pendingRatings: {},
    visibleBlocks: {},
    _loadedConversationIds: new Set(),
    _conversationsLoaded: false,
    _pendingDeletes: new Set(),

    setActiveConversation: (id) => set((state) => {
        // Switching chats mid-stream used to leave the GLOBAL isStreaming flag
        // set forever (composer stuck on "QUASAR is thinking..." until a page
        // reload — live F-12b, reproduced twice on 2026-07-13). The old chat's
        // stream keeps running server-side and its answer persists; the UI for
        // the newly opened chat must not stay locked by it.
        const streamingReset = state.isStreaming
            ? { isStreaming: false, thinkingSteps: [], streamingContent: "" }
            : {};
        if (state.activeConversationId && state.messages.length > 0) {
            const updatedConversations = state.conversations.map(c =>
                c.id === state.activeConversationId
                    ? { ...c, messages: state.messages, updatedAt: new Date(), model: state.selectedModel }
                    : c
            );
            const conv = updatedConversations.find((c) => c.id === id);
            return {
                ...streamingReset,
                conversations: updatedConversations,
                activeConversationId: id,
                messages: conv?.messages || [],
                selectedModel: conv?.model || state.selectedModel || "gpt-oss-120b",
            };
        }
        const conv = state.conversations.find((c) => c.id === id);
        return {
            ...streamingReset,
            activeConversationId: id,
            messages: conv?.messages || [],
            selectedModel: conv?.model || state.selectedModel || "gpt-oss-120b",
        };
    }),

    addMessage: (message, ownerConversationId) => set((state) => {
        // UI-01: a card/tool bubble from a stream whose conversation was
        // switched away must append to the OWNER's stored messages — not the
        // newly active conversation's transcript.
        if (isOwnerRouted(state, ownerConversationId)) {
            return patchOwnerConversation(state, ownerConversationId, (msgs) => [...msgs, message]);
        }
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

    mergeWebSourcesMessage: (payload, ownerConversationId) => set((state) => {
        // UI-01: web sources retrieved for a switched-away stream belong to
        // the owner conversation's transcript.
        if (isOwnerRouted(state, ownerConversationId)) {
            return patchOwnerConversation(state, ownerConversationId, (msgs) =>
                mergeWebSourcesIntoMessages(msgs, payload));
        }
        const messages = mergeWebSourcesIntoMessages(state.messages, payload);
        if (!messages) return {};
        return {
            messages,
            conversations: syncActiveConversationMessages(state, messages),
        };
    }),

    updateLastAssistantMessage: (content, ownerConversationId) => set((state) => {
        // UI-01: after a mid-stream switch, tokens must patch the OWNER's last
        // assistant text message — writing into the live list overwrote the
        // newly opened conversation's previous answer with the old stream's.
        if (isOwnerRouted(state, ownerConversationId)) {
            return patchOwnerConversation(state, ownerConversationId, (msgs) =>
                updateLastAssistantContent(msgs, content));
        }
        return { messages: updateLastAssistantContent(state.messages, content) };
    }),

    updateLastAssistantThinking: (thinking, ownerConversationId) => set((state) => {
        // UI-01: owner-routed thinking never touches the global thinkingStatus,
        // which belongs to whatever conversation is on screen.
        if (isOwnerRouted(state, ownerConversationId)) {
            return patchOwnerConversation(state, ownerConversationId, (msgs) =>
                updateAssistantThinking(msgs, thinking));
        }
        return {
            messages: updateAssistantThinking(state.messages, thinking),
            thinkingStatus: "running",
        };
    }),

    updateLastAssistantRunMeta: (meta, ownerConversationId) => set((state) => {
        const stamp = (messages: Message[]): Message[] | null => {
            const index = findLastAssistantTextIndex(messages);
            if (index < 0) return null;
            const next = [...messages];
            // run_meta is where the text block learns its identity (Feature 4):
            // the assistant message is created optimistically in handleSend,
            // before any event arrives, so it has no id of its own until now.
            // The message id is left alone — React is already keyed on it and
            // remounting the live bubble mid-stream would drop the streamed text.
            next[index] = {
                ...next[index],
                runMeta: meta,
                blockId: meta.text_block_id || next[index].blockId,
                blockKind: "text",
            };
            return next;
        };
        // UI-01: run identity follows the stream's owner conversation.
        if (isOwnerRouted(state, ownerConversationId)) {
            return patchOwnerConversation(state, ownerConversationId, stamp);
        }
        const messages = stamp(state.messages);
        if (!messages) return {};
        return {
            messages,
            conversations: syncActiveConversationMessages(state, messages),
        };
    }),

    /* Mark this turn as an infrastructure failure (Feature 4).

       run_meta arrives at the START of a turn, when the status is necessarily
       unknown, and the persisted rich_meta only learns "failed" at the end — so
       WITHOUT this the live session has no failed marker at all, and the eval
       gate would demand a rating for an answer the backend never produced.
       Mirrors what sse.py writes into rich_meta.runMeta, so the live turn and
       the reloaded one agree. */
    markLastAssistantRunFailed: (errorCode, ownerConversationId) => set((state) => {
        const stamp = (messages: Message[]): Message[] | null => {
            const index = findLastAssistantTextIndex(messages);
            if (index < 0) return null;
            const next = [...messages];
            const existing = next[index].runMeta;
            next[index] = {
                ...next[index],
                runMeta: { ...(existing || {}), status: "failed", errorCode } as Message["runMeta"],
            };
            return next;
        };
        // UI-01: a failure on a switched-away stream marks the OWNER's turn.
        if (isOwnerRouted(state, ownerConversationId)) {
            return patchOwnerConversation(state, ownerConversationId, stamp);
        }
        const messages = stamp(state.messages);
        if (!messages) return {};
        return {
            messages,
            conversations: syncActiveConversationMessages(state, messages),
        };
    }),

    // Raw request provenance for the turn (Feature 1). Lands on the assistant
    // text message; card messages carry their own `request` on the card payload.
    updateLastAssistantToolTrace: (calls, ownerConversationId) => set((state) => {
        if (!Array.isArray(calls) || calls.length === 0) return {};
        const stamp = (messages: Message[]): Message[] | null => {
            const index = findLastAssistantTextIndex(messages);
            if (index < 0) return null;
            const next = [...messages];
            next[index] = { ...next[index], toolTrace: calls };
            return next;
        };
        // Route to the conversation that OWNS the stream (CX-18 / UI-01): a
        // terminal trace can arrive after the user switched conversations
        // mid-stream, and attaching it to the newly ACTIVE conversation would
        // pin one run's provenance onto another conversation's last answer.
        // When the owner is no longer active, patch the owner's stored copy
        // (the persisted rich_meta.toolTrace re-asserts it on reload too).
        if (isOwnerRouted(state, ownerConversationId)) {
            return patchOwnerConversation(state, ownerConversationId, stamp);
        }
        const messages = stamp(state.messages);
        if (!messages) return {};
        return {
            messages,
            conversations: syncActiveConversationMessages(state, messages),
        };
    }),

    updateLastAssistantUsage: (totalTokens, durationMs, ownerConversationId) => set((state) => {
        const stamp = (messages: Message[]): Message[] | null => {
            const index = findLastAssistantTextIndex(messages);
            if (index < 0) return null;
            const next = [...messages];
            next[index] = {
                ...next[index],
                usageTokens: totalTokens,
                // Backend compute time wins over the frontend stream-lifetime clock
                // (drip throttling inflates the latter — live P15's "Thought 1149s").
                ...(durationMs && durationMs > 0
                    ? { thinkingDuration: Math.max(1, Math.round(durationMs / 1000)) }
                    : {}),
            };
            return next;
        };
        // UI-01: a switched-away stream's usage stamps the OWNER's turn.
        if (isOwnerRouted(state, ownerConversationId)) {
            return patchOwnerConversation(state, ownerConversationId, stamp);
        }
        const messages = stamp(state.messages);
        if (!messages) return {};
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
            // Same stuck-composer gap as setActiveConversation: a New Chat
            // opened mid-stream must not inherit the old chat's lock.
            isStreaming: false,
            thinkingSteps: [],
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
            const res = await fetch(`${API_BASE}/api/models`, { credentials: "include" });
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

    // ── Eval mode (Feature 4) ───────────────────────────────────────────────

    setEvalMode: (on) => {
        persistEvalMode(on);
        set({ evalMode: on });
    },

    setBlockRating: (blockId, rating) => set((s) => ({
        blockRatings: { ...s.blockRatings, [blockId]: rating },
    })),

    /* A rating whose POST is in flight: lights the star at once, but does NOT
       count toward the gate.

       Optimism and the gate are different questions. If an in-flight rating
       satisfied the gate, the user could star a block and send the next prompt
       before the POST landed — and if it then failed, the turn is gone and the
       label set has a hole exactly where the gate was supposed to guarantee one
       could not exist. On a slow network that window is seconds wide, not
       milliseconds. So the star is optimistic and the gate is not. */
    beginRating: (blockId, rating) => set((s) => ({
        pendingRatings: { ...s.pendingRatings, [blockId]: rating },
    })),

    /* Settle an in-flight rating: `rating` on success, null when the POST failed.
       A failure simply drops the pending value — the confirmed map was never
       touched, so there is no phantom star to roll back. */
    resolveRating: (blockId, rating) => set((s) => {
        const pending = { ...s.pendingRatings };
        delete pending[blockId];
        if (rating === null) return { pendingRatings: pending };
        return { pendingRatings: pending, blockRatings: { ...s.blockRatings, [blockId]: rating } };
    }),

    /* The gate may only demand a rating for a block the user can actually see,
       and the ONLY thing that knows that is the rating widget itself: a block
       can carry a stable id and still render nothing (a 0-row data card is
       suppressed; a papers grid is swallowed by the ObservationPaperGraph).
       Mirroring those conditions in the gate would work until someone adds the
       next suppression rule to ChatMessage and silently wedges the composer —
       an unrateable-but-required block blocks sending with no way out. So the
       mounted widgets are the source of truth, and the gate follows automatically.

       Refcounted rather than a Set: React StrictMode double-mounts in dev, and
       a plain delete on the first cleanup would unregister a live widget. */
    registerBlock: (blockId) => set((s) => {
        if (!blockId) return {};
        return { visibleBlocks: { ...s.visibleBlocks, [blockId]: (s.visibleBlocks[blockId] || 0) + 1 } };
    }),

    unregisterBlock: (blockId) => set((s) => {
        if (!blockId || !s.visibleBlocks[blockId]) return {};
        const next = { ...s.visibleBlocks };
        const count = next[blockId] - 1;
        if (count > 0) next[blockId] = count;
        else delete next[blockId];
        return { visibleBlocks: next };
    }),

    loadBlockRatings: async (conversationId) => {
        if (!conversationId) return;
        try {
            const ratings = await apiFetchBlockRatings(conversationId);
            // Merge rather than replace: a rating cast while this was in flight
            // must not be clobbered by a response that predates it.
            set((s) => ({ blockRatings: { ...ratings, ...s.blockRatings } }));
        } catch {
            // Non-critical — an unreachable ratings endpoint must not break chat.
        }
    },

    // ── Server-sync Actions ─────────────────────────────────────────────────

    loadConversations: async (token?: string) => {
        // CX-07: a slow response from account A must not commit over account B.
        const generation = currentAuthGeneration();
        const serverConvos = await apiFetchConversations(token);
        if (generation !== currentAuthGeneration()) return; // account changed mid-flight
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

    loadConversationMessages: async (conversationId: string, token?: string) => {
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

        // Re-light the stars this user already earned on these blocks. Fired
        // after the messages land so the ids exist to key against; awaited so a
        // reload can't briefly show an unrated turn and trip the gate.
        await get().loadBlockRatings(conversationId);
    },

    deleteConversation: async (conversationId: string, token?: string) => {
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

// Scrub per-user chat state on sign-out / account change so the next account on
// this browser can never see the previous user's conversations or messages
// (CX-06). Registered imperatively — and synchronously at sign-out — rather than
// via a React effect, which would race component unmount.
registerSessionScrubber(() => {
    useChatStore.getState().clearAllConversations();
    useChatStore.setState({ _pendingDeletes: new Set() });
});
