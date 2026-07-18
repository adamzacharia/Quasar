"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { PanelLeft, Star, ArrowDown } from "lucide-react";
import { useChatStore } from "../lib/store";
import { sendChatMessage, reviewProposal, submitPlanFeedback } from "../lib/api";
import { EmptyState } from "./EmptyState";
import { ChatInput } from "./ChatInput";
import { ChatMessage } from "./ChatMessage";
import { MobileHomeBar } from "./MobileHomeBar";
import { useIsMobile } from "../lib/use-is-mobile";
import type { ResearchGraph } from "./ObservationPaperGraph";
import { DownloadProgress } from "./DownloadProgress";
import { PlanReviewWidget } from "./PlanReviewWidget";
import type { PlanReviewData } from "./PlanReviewWidget";
import type { Message, DataTableResult, Paper, ToolCall, NotebookData, WebImage, WebSource } from "../lib/types";
import { normalizeEvidenceQuality } from "../lib/evidence-quality";
import { normalizeHipsImageMeta } from "../lib/hips-imagery";
import { PREFILL_PROMPT_EVENT } from "../lib/prompt-dispatch";
import { isSafeWebImage, isSafeWebSource } from "../lib/content-safety";
import { buildObservationPaperGraph } from "../lib/research-graph";
import { useAuthStore, verifyAuth } from "../lib/auth-store";
import { shouldBlockSend, unratedBlocks, nudgeText } from "../lib/eval-mode";
import { useEvalModeActive } from "../lib/use-eval-mode";

interface AttachedFile { file: File; preview?: string; type: "image" | "document"; }

function generateId(): string { return Date.now().toString(36) + Math.random().toString(36).slice(2); }

export function ChatArea() {
    const {
        messages, addMessage, mergeWebSourcesMessage, updateLastAssistantMessage, updateLastAssistantThinking,
        updateLastAssistantRunMeta, updateLastAssistantUsage, updateLastAssistantToolTrace,
        isStreaming, setStreaming,
        toggleSidebar,
        activeConversationId, setActiveConversation,
        selectedModel, conversations,
        toggleStar,
        thinkingSteps, thinkingStatus, addThinkingStep, clearThinking,
        attachThinkingToLastMessage,
        taskGroups, taskItems, taskChecklist, taskExecutionActive,
        handleTaskGroup, handleTaskUpdate, handleTaskList, clearTaskExecution,
        setActiveConversationId, loadConversations,
        blockRatings, visibleBlocks, markLastAssistantRunFailed,
    } = useChatStore();

    // Admin + build-flag aware; the raw store flag alone is per-browser and
    // would gate a non-admin who inherited an admin's localStorage.
    const evalMode = useEvalModeActive();

    const { isAuthenticated, clearAuth } = useAuthStore();
    const isMobile = useIsMobile();

    const [inputValue, setInputValue] = useState("");
    // Inline nudge shown when the eval-mode gate refuses a send (Feature 4).
    const [evalNudge, setEvalNudge] = useState("");
    // Retract the nudge the moment the last star lands, rather than making the
    // user press send again to discover they're unblocked.
    const pendingEvalBlocks = useMemo(
        () => (evalMode ? unratedBlocks(messages, blockRatings, visibleBlocks) : []),
        [evalMode, messages, blockRatings, visibleBlocks],
    );
    useEffect(() => {
        if (pendingEvalBlocks.length === 0) setEvalNudge("");
    }, [pendingEvalBlocks.length]);
    // Composer options + visit counter live here (not in ChatInput) so they
    // persist across the hero→docked switch and the hit endpoint fires once.
    const [grounded, setGrounded] = useState(false);
    const [webSearch, setWebSearch] = useState(true);
    const [hitCount, setHitCount] = useState<number | null>(null);
    useEffect(() => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 4000);
        const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        fetch(`${apiBase}/api/analytics/hit`, { credentials: "include", signal: controller.signal })
            .then((res) => res.json())
            .then((data) => { if (typeof data.hits === "number") setHitCount(data.hits); })
            .catch(() => {/* silent: timeout or network error */})
            .finally(() => clearTimeout(timeout));
        return () => { controller.abort(); clearTimeout(timeout); };
    }, []);
    // Prompt handoff from other pages (/gallery recipes, SLE chips): /?prompt=…
    // prefills the composer WITHOUT sending, then strips the param so a reload
    // doesn't re-prefill. window.location (not useSearchParams) keeps ChatArea
    // free of a Suspense-boundary requirement.
    useEffect(() => {
        try {
            const params = new URLSearchParams(window.location.search);
            const prompt = params.get("prompt");
            if (prompt && prompt.trim()) {
                setInputValue(prompt);
                params.delete("prompt");
                const rest = params.toString();
                window.history.replaceState(null, "", window.location.pathname + (rest ? `?${rest}` : ""));
            }
        } catch { /* URL parsing is best-effort */ }
    }, []);
    // Same-page prompt handoff (T7.3): in-page cards (e.g. the sky-map
    // "Cross-match here" button) dispatch a window event to prefill the composer
    // without a navigation — the /?prompt= effect above only fires on mount.
    // ChatInput re-syncs + focuses on the changed value; no auto-send.
    useEffect(() => {
        const onPrefill = (event: Event) => {
            const detail = (event as CustomEvent<{ prompt?: string }>).detail;
            const prompt = detail?.prompt?.trim();
            if (prompt) setInputValue(prompt);
        };
        window.addEventListener(PREFILL_PROMPT_EVENT, onPrefill);
        return () => window.removeEventListener(PREFILL_PROMPT_EVENT, onPrefill);
    }, []);
    const abortControllerRef = useRef<AbortController | null>(null);
    const [downloadProgress, setDownloadProgress] = useState<{
        filename: string; downloaded_bytes: number;
        total_bytes: number | null; speed_kbps: number; percent: number | null;
        eta_seconds?: number | null; phase?: string;
    } | null>(null);
    // ── Plan Review (Human-in-the-Loop) ─────────────────────────
    const [pendingPlan, setPendingPlan] = useState<PlanReviewData | null>(null);
    const [planSubmitting, setPlanSubmitting] = useState(false);
    // Track conversation_id from SSE meta for plan feedback
    const conversationIdRef = useRef<string | null>(null);
    useEffect(() => {
        conversationIdRef.current = activeConversationId ?? null;
    }, [activeConversationId]);

    const activeConversation = conversations.find(c => c.id === activeConversationId);
    const isStarred = activeConversation?.isStarred || false;
    const scrollRef = useRef<HTMLDivElement>(null);

    const turnGraphs = useMemo(() => {
        const graphs: Record<string, ResearchGraph> = {};
        let currentTurnMessages: Message[] = [];
        let currentTurnDataMessageIds: string[] = [];
        let currentTurnMessageIds: string[] = [];

        // UI-06: data messages RENDER the graph inline (ChatMessage's data
        // branch), so only the LAST data card of the turn carries it — a
        // multi-table turn used to stack the identical graph once per table.
        // Non-data ids still carry it: the papers grid keys its
        // self-suppression off having a graph for its own message id.
        const assignTurnGraph = (
            turnMessages: Message[],
            turnMessageIds: string[],
            dataMessageIds: string[],
        ) => {
            const graph = buildObservationPaperGraph(turnMessages) as ResearchGraph | null;
            if (!graph) return;
            const anchorDataId = dataMessageIds[dataMessageIds.length - 1];
            for (const id of turnMessageIds) {
                if (dataMessageIds.includes(id) && id !== anchorDataId) continue;
                graphs[id] = graph;
            }
        };

        for (const msg of messages) {
            if (msg.role === "user") {
                if (currentTurnMessages.length > 0 && currentTurnDataMessageIds.length > 0) {
                    assignTurnGraph(currentTurnMessages, currentTurnMessageIds, currentTurnDataMessageIds);
                }
                currentTurnMessages = [];
                currentTurnDataMessageIds = [];
                currentTurnMessageIds = [];
            }
            currentTurnMessages.push(msg);
            currentTurnMessageIds.push(msg.id);
            if (msg.type === "data") {
                currentTurnDataMessageIds.push(msg.id);
            }
        }

        if (currentTurnMessages.length > 0 && currentTurnDataMessageIds.length > 0) {
            assignTurnGraph(currentTurnMessages, currentTurnMessageIds, currentTurnDataMessageIds);
        }

        return graphs;
    }, [messages]);

    // ── Smart auto-scroll ──────────────────────────────────────
    // Only scroll to bottom if the user hasn't manually scrolled up.
    // This lets users read earlier messages while the agent is streaming.
    const userScrolledUpRef = useRef(false);
    const userScrollIntentRef = useRef(false);
    const scrollIntentTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const autoScrollRafRef = useRef<number | null>(null);
    const [showScrollBtn, setShowScrollBtn] = useState(false);

    const markUserScrollIntent = useCallback(() => {
        userScrollIntentRef.current = true;
        if (scrollIntentTimeoutRef.current) {
            clearTimeout(scrollIntentTimeoutRef.current);
        }
        scrollIntentTimeoutRef.current = setTimeout(() => {
            userScrollIntentRef.current = false;
        }, 250);
    }, []);

    const handleScroll = useCallback(() => {
        if (!scrollRef.current) return;
        const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
        const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
        const nearBottom = distanceFromBottom <= 80;

        if (nearBottom) {
            userScrolledUpRef.current = false;
            setShowScrollBtn(false);
            return;
        }

        if (userScrollIntentRef.current) {
            userScrolledUpRef.current = true;
            setShowScrollBtn(true);
        }
    }, []);

    useEffect(() => {
        if (!scrollRef.current || userScrolledUpRef.current) return;

        if (autoScrollRafRef.current !== null) {
            cancelAnimationFrame(autoScrollRafRef.current);
        }

        autoScrollRafRef.current = requestAnimationFrame(() => {
            if (scrollRef.current && !userScrolledUpRef.current) {
                scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }
            autoScrollRafRef.current = null;
        });
    }, [messages, thinkingSteps, taskGroups, taskItems, taskChecklist]);

    useEffect(() => {
        return () => {
            if (scrollIntentTimeoutRef.current) clearTimeout(scrollIntentTimeoutRef.current);
            if (autoScrollRafRef.current !== null) cancelAnimationFrame(autoScrollRafRef.current);
        };
    }, []);

    // Reset scroll lock when the user sends a new message
    useEffect(() => {
        const lastMsg = messages[messages.length - 1];
        if (lastMsg?.role === "user") {
            userScrolledUpRef.current = false;
            setShowScrollBtn(false);
        }
    }, [messages]);

    const scrollToBottom = useCallback(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
            userScrolledUpRef.current = false;
            setShowScrollBtn(false);
        }
    }, []);

    const normalizeWebSourcesPayload = useCallback((data: {
        sources?: unknown;
        images?: unknown;
        query?: unknown;
        provider?: unknown;
        image_provider?: unknown;
        search_type?: unknown;
    }) => {
        const isWebSource = (source: WebSource | null): source is WebSource => source !== null;
        const isWebImage = (image: WebImage | null): image is WebImage => image !== null;

        const normalizeUrl = (value: unknown): string => {
            const raw = String(value || "").trim().replace(/^<|>$/g, "").replace(/[.,;:)\]}"']+$/g, "");
            if (!raw) return "";
            if (raw.startsWith("http://") || raw.startsWith("https://")) return raw;
            if (raw.startsWith("www.")) return `https://${raw}`;
            if (/^[A-Za-z0-9.-]+\.[A-Za-z]{2,}\/\S+$/.test(raw)) return `https://${raw}`;
            return "";
        };

        const titleFromUrl = (url: string): string => {
            try {
                return new URL(url).hostname.replace("www.", "");
            } catch {
                return url;
            }
        };

        const normalizeWebSource = (source: unknown): WebSource | null => {
            if (typeof source === "string") {
                const url = normalizeUrl(source);
                return url ? { title: titleFromUrl(url), url, snippet: "" } : null;
            }
            if (!source || typeof source !== "object") return null;
            const item = source as Record<string, unknown>;
            const url = normalizeUrl(item.url || item.link || item.href || item.source_url || "");
            if (!url) return null;
            const normalized = {
                title: String(item.title || item.name || titleFromUrl(url)).trim(),
                url,
                snippet: String(item.snippet || item.content || item.text || item.description || "").trim(),
                evidenceQuality: normalizeEvidenceQuality(item.evidenceQuality || item.evidence_quality),
            };
            return isSafeWebSource(normalized) ? normalized : null;
        };

        const normalizeWebImage = (image: unknown): WebImage | null => {
            if (typeof image === "string") {
                const url = image.trim();
                return url ? { url, description: "" } : null;
            }
            if (!image || typeof image !== "object") return null;
            const item = image as Record<string, unknown>;
            const url = String(item.url || item.src || item.image_url || "").trim();
            if (!url) return null;
            const sourceUrl = normalizeUrl(
                item.sourceUrl
                || item.source_url
                || item.sourcePageUrl
                || item.source_page_url
                || item.pageUrl
                || item.page_url
                || item.source
                || ""
            );
            const sourceTitle = String(
                item.sourceTitle
                || item.source_title
                || item.sourcePageTitle
                || item.source_page_title
                || item.pageTitle
                || item.page_title
                || ""
            ).trim();
            const normalized: WebImage = {
                url,
                description: String(item.description || item.alt || item.title || "").trim(),
            };
            if (sourceUrl) normalized.sourceUrl = sourceUrl;
            if (sourceTitle) normalized.sourceTitle = sourceTitle;
            return isSafeWebImage(normalized) ? normalized : null;
        };

        const rawSources = Array.isArray(data.sources)
            ? data.sources
            : data.sources && typeof data.sources === "object" && Array.isArray((data.sources as Record<string, unknown>).results)
                ? ((data.sources as Record<string, unknown>).results as unknown[])
                : [];
        const rawImages = Array.isArray(data.images) ? data.images : [];

        return {
            sources: rawSources.map(normalizeWebSource).filter(isWebSource),
            images: rawImages.map(normalizeWebImage).filter(isWebImage),
            provider: String(data.provider || "").trim(),
            imageProvider: String(data.image_provider || "").trim(),
            searchType: String(data.search_type || "").trim(),
            query: String(data.query || "").trim(),
        };
    }, []);

    const handleStop = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        attachThinkingToLastMessage();
        setStreaming(false);
    }, [attachThinkingToLastMessage, setStreaming]);

    const handleSend = useCallback(async (text: string, attachments?: AttachedFile[], options?: { groundedSummary?: boolean; webSearch?: boolean }) => {
        const hasContent = text.trim() || (attachments && attachments.length > 0);
        if (!hasContent || isStreaming) return;

        // ── Eval-mode gate (Feature 4) ──────────────────────────
        // Every rateable block of the last turn must be scored before the next
        // prompt, so the label set has no holes. Turns that died on an
        // infrastructure error are exempt (see eval-mode.js::isFailedTurn) —
        // there is nothing to judge, and gating one would wedge the composer.
        const pending = unratedBlocks(messages, blockRatings, visibleBlocks);
        if (shouldBlockSend(evalMode, messages, blockRatings, visibleBlocks)) {
            setEvalNudge(nudgeText(pending));
            // Hand the prompt back. ChatInput clears its own box the moment it
            // calls onSend (ChatInput.tsx handleSubmit), so a refusal would
            // otherwise silently eat what the user typed; its initialValue
            // effect re-fills and refocuses the composer.
            setInputValue(text);
            return;
        }
        setEvalNudge("");

        // Create a new AbortController for this request
        const controller = new AbortController();
        abortControllerRef.current = controller;

        if (!activeConversationId) {
            const newId = "conv-" + generateId();
            setActiveConversation(newId);
        }

        // Build display content for the user message
        const displayContent = text.trim() || (attachments && attachments.length > 0
            ? `Attached files: ${attachments.map(a => a.file.name).join(", ")}`
            : "");
        const imageAttachments = attachments?.filter(a => a.type === "image") || [];

        const userMsg: Message = {
            id: generateId(), role: "user", content: displayContent, type: "text", timestamp: new Date(),
            attachmentPreviews: imageAttachments.map(a => a.preview!).filter(Boolean),
            attachmentNames: attachments?.filter(a => a.type === "document").map(a => a.file.name) || [],
        };
        addMessage(userMsg);
        setInputValue("");
        setStreaming(true);
        clearThinking();
        clearTaskExecution();
        setPendingPlan(null);
        setPlanSubmitting(false);
        // Thinking steps will arrive dynamically from backend SSE events


        const assistantMsgId = generateId();
        const assistantMsg: Message = { id: assistantMsgId, role: "assistant", content: "", type: "text", timestamp: new Date() };
        addMessage(assistantMsg);

        let accumulated = "";
        let accumulatedThought = "";
        // run_meta lands before any card event, so every card of this turn can
        // carry its run — the rating POST needs run_id to join to chat_runs
        // (model/tokens/cost) in the eval export.
        let turnRunMeta: import("../lib/api").ChatRunMeta | undefined;
        // The conversation THIS stream belongs to (CX-18 / UI-02). NOT the
        // live conversationIdRef: that ref is re-pointed to whatever
        // conversation becomes active, so reading it at trace time after a
        // mid-stream switch identified the WRONG conversation. Seeded from
        // the send-time active id; upgraded to the server UUID at meta time.
        let streamOwnerConversationId: string | null =
            useChatStore.getState().activeConversationId ?? null;
        // UI-01: every stream callback routes through these. While the owner
        // is still the active conversation the callbacks patch the live
        // message list as before; after a mid-stream switch they hand the
        // owner id to the store, which patches the owner's STORED messages
        // instead (mirroring updateLastAssistantToolTrace's CX-18 routing) —
        // tokens/cards from chat A must never land in chat B's transcript.
        const ownerIsActive = () =>
            useChatStore.getState().activeConversationId === streamOwnerConversationId;
        const ownerFor = () => (ownerIsActive() ? undefined : streamOwnerConversationId);

        // Build message with file context
        let messageWithContext = text.trim();
        if (attachments && attachments.length > 0) {
            const fileNames = attachments.map(a => a.file.name).join(", ");
            messageWithContext = messageWithContext
                ? `${messageWithContext}\n\n[Attached files: ${fileNames}]`
                : `[Attached files: ${fileNames}]`;
        }

        try {
            // Check for proposal review special case
            const isReviewRequest = text.toLowerCase().includes("review proposal");
            const firstPdf = attachments?.find(a => a.file.type === "application/pdf" || a.file.name.endsWith(".pdf"))?.file;

            if (isReviewRequest && firstPdf) {
                // RED TEAM TAC Workflow
                await reviewProposal(firstPdf, {
                    // UI-01: same owner routing as the standard workflow below.
                    onToken: (token: string) => {
                        accumulated += token;
                        updateLastAssistantMessage(accumulated, ownerFor());
                    },
                    onStatus: (step: string, state: string) => {
                        if (ownerIsActive()) addThinkingStep(step, state as "running" | "completed");
                    },
                    onComplete: () => {
                        if (ownerIsActive()) {
                            attachThinkingToLastMessage();
                            setStreaming(false);
                        }
                    },
                    onError: (error: string) => {
                        if (ownerIsActive()) attachThinkingToLastMessage();
                        updateLastAssistantMessage(`Error: ${error}`, ownerFor());
                        // Feature 4: an infrastructure failure is exempt from the
                        // eval gate — there is no answer to judge.
                        markLastAssistantRunFailed("stream_error", ownerFor());
                        if (ownerIsActive()) setStreaming(false);
                    }
                }, controller.signal);  // S5 auth rides the httpOnly cookie (credentials: "include")
            } else {
                // Standard workflow
                let accumulatedWebSources: {
                    sources: WebSource[];
                    images: WebImage[];
                    provider?: string;
                    imageProvider?: string;
                    searchType?: string;
                    query?: string;
                } | null = null;

                await sendChatMessage(
                    {
                        message: messageWithContext,
                        // Only send conversation_id if it's a server-created UUID (not a client-generated "conv-" ID)
                        conversation_id: activeConversationId && !activeConversationId.startsWith("conv-") ? activeConversationId : undefined,
                        model: selectedModel,
                        attachments: attachments?.map(a => a.file),
                        grounded_summary: Boolean(options?.groundedSummary),
                        web_search: options?.webSearch !== false,
                    },
                    {
                        onToken: (token: string) => {
                            accumulated += token;
                            updateLastAssistantMessage(accumulated, ownerFor());
                        },
                        onThought: (thought: string) => {
                            accumulatedThought += thought;
                            updateLastAssistantThinking(accumulatedThought, ownerFor());
                        },
                        onToolCall: (toolName: string, input: string) => {
                            const displayName = toolName.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
                            // UI-01: the global thinking rail belongs to the ACTIVE
                            // conversation; a switched-away stream must not write to it.
                            if (ownerIsActive()) addThinkingStep(`Calling tool: ${displayName}`, "completed");
                            const toolCall: ToolCall = {
                                id: generateId(),
                                name: toolName,
                                displayName,
                                status: "completed",
                                input: (() => { try { return JSON.parse(input); } catch { return { raw: input }; } })(),
                                output: "Completed",
                            };
                            addMessage({
                                id: generateId(),
                                role: "assistant",
                                content: "",
                                type: "tool_call",
                                timestamp: new Date(),
                                toolCall,
                            }, ownerFor());
                        },
                        onStatus: (step: string, state: string) => {
                            // UI-01: same global-rail guard as onToolCall.
                            if (ownerIsActive()) addThinkingStep(step, state as "running" | "completed");
                        },
                        onData: (data: Record<string, unknown>) => {
                            const tableData = data as unknown as DataTableResult;
                            // The server-minted block id doubles as the message id
                            // (Feature 4), so the card the user just rated keeps
                            // that identity through a reload. generateId() is the
                            // fallback for a backend that didn't send one.
                            const blockId = typeof data.blockId === "string" ? data.blockId : undefined;
                            addMessage({
                                id: blockId || generateId(),
                                role: "assistant",
                                content: "",
                                type: "data",
                                timestamp: new Date(),
                                dataTable: tableData,
                                runMeta: turnRunMeta,
                                blockId,
                                blockKind: "data",
                            }, ownerFor());  // UI-01
                        },
                        onPapers: (rawPapers: Record<string, unknown>[], papersRequest?: import("../lib/api").ToolRequest, papersBlockId?: string) => {
                            // Map backend field names to frontend Paper interface
                            const papers: Paper[] = rawPapers.map((p, i) => ({
                                id: (p.bibcode as string) || `paper-${i}`,
                                title: (p.title as string) || "Untitled",
                                authors: (p.authors as string) || "Unknown",
                                year: Number(p.year) || 0,
                                journal: (p.journal as string) || (p.pub as string) || "",
                                citationCount: Number(p.citations ?? p.citationCount ?? p.citation_count ?? 0),
                                type: "radio",
                                bibcode: (p.bibcode as string) || undefined,
                                doi: (p.doi as string) || undefined,
                                abstract: (p.abstract as string) || undefined,
                                // OpenAlex enrichment
                                fwci: p.fwci != null ? Number(p.fwci) : null,
                                citationPercentile: p.citation_percentile != null ? Number(p.citation_percentile) : null,
                                isTop1Percent: Boolean(p.is_top_1_percent),
                                isTop10Percent: Boolean(p.is_top_10_percent),
                                funders: Array.isArray(p.funders) ? (p.funders as { name: string; id: string }[]) : undefined,
                                oaPdfUrl: (p.oa_pdf_url as string) || undefined,
                                observationLinks: Array.isArray(p.observation_links)
                                    ? (p.observation_links as Record<string, unknown>[]).map((link) => ({
                                        identifier: String(link.identifier || ""),
                                        identifierType: String(link.identifier_type || link.identifierType || ""),
                                        relation: String(link.relation || ""),
                                        confidence: String(link.confidence || ""),
                                        adsQuery: String(link.ads_query || link.adsQuery || ""),
                                    })).filter((link) => link.identifier)
                                    : undefined,
                            }));
                            addMessage({
                                id: papersBlockId || generateId(),
                                role: "assistant",
                                content: "Here are the relevant papers I found:",
                                type: "papers",
                                timestamp: new Date(),
                                papers,
                                request: papersRequest,
                                runMeta: turnRunMeta,
                                blockId: papersBlockId,
                                blockKind: "papers",
                            }, ownerFor());  // UI-01
                        },
                        onNotebook: (notebook: Record<string, unknown>) => {
                            const blockId = typeof notebook.blockId === "string" ? notebook.blockId : undefined;
                            addMessage({
                                id: blockId || generateId(),
                                role: "assistant",
                                content: `I've generated a Jupyter Notebook for your analysis: **${notebook.title || 'Dynamic Notebook'}**`,
                                type: "notebook",
                                timestamp: new Date(),
                                notebookData: notebook as unknown as NotebookData,
                                runMeta: turnRunMeta,
                                blockId,
                                blockKind: "notebook",
                            }, ownerFor());  // UI-01
                        },
                        onImage: (img) => {
                            // Resolve relative URL to absolute backend URL
                            const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                            const toAbsolute = (u: string) =>
                                u.startsWith("http") || u.startsWith("data:") ? u : `${apiBase}${u}`;
                            const imageUrl = toAbsolute(img.url);
                            // Blink frames are also server-relative /plots URLs — resolve them
                            // to the backend origin too, or a split UI/API deploy 404s them.
                            const meta = normalizeHipsImageMeta(img.meta);
                            if (meta?.frames?.length) {
                                meta.frames = meta.frames.map((f) => ({ ...f, url: toAbsolute(f.url) }));
                            }
                            addMessage({
                                id: img.blockId || generateId(),
                                role: "assistant",
                                content: img.caption || "",
                                type: "image",
                                timestamp: new Date(),
                                imageUrl: imageUrl,
                                imageCaption: img.caption || "",
                                imageMeta: meta,
                                request: img.request,
                                runMeta: turnRunMeta,
                                blockId: img.blockId,
                                blockKind: "image",
                            }, ownerFor());  // UI-01
                        },
                        onPlotly: (plot) => {
                            // Interactive Plotly figure — same flow as images (ephemeral,
                            // not persisted server-side). Resolve the PNG fallback URL to
                            // an absolute backend URL like image cards do.
                            const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                            const rawFallback = plot.png_fallback || "";
                            const pngFallback = rawFallback
                                ? (rawFallback.startsWith("http") || rawFallback.startsWith("data:") ? rawFallback : `${apiBase}${rawFallback}`)
                                : undefined;
                            const spec = plot.spec && typeof plot.spec === "object" && Array.isArray(plot.spec.data)
                                ? { data: plot.spec.data, layout: plot.spec.layout }
                                : undefined;
                            if (!spec && !pngFallback) return; // nothing renderable
                            addMessage({
                                id: plot.blockId || generateId(),
                                role: "assistant",
                                content: plot.title || "",
                                type: "plotly",
                                timestamp: new Date(),
                                plotlySpec: spec,
                                plotlyTitle: plot.title || "",
                                plotlyPngFallback: pngFallback,
                                request: plot.request,
                                plotlyMeta: plot.meta && typeof plot.meta === "object" ? plot.meta : undefined,
                                runMeta: turnRunMeta,
                                blockId: plot.blockId,
                                blockKind: "plotly",
                            }, ownerFor());  // UI-01
                        },
                        onTaskGroup: (group) => handleTaskGroup(group),
                        onTaskUpdate: (update) => handleTaskUpdate(update),
                        onTaskList: (list) => handleTaskList(list),
                        onPlanReview: (plan) => {
                            setPendingPlan({
                                // The stream's OWNER, not the live ref — a review from a
                                // switched-away stream must still address its own
                                // conversation (UI-02 sibling of the tool-trace fix).
                                conversationId: streamOwnerConversationId || plan.conversationId || "",
                                // UIAPI-08: this turn's run id addresses the run-scoped
                                // plan-feedback queue server-side.
                                runId: turnRunMeta?.run_id,
                                title: plan.title,
                                subtasks: plan.subtasks,
                                reasoning: plan.reasoning,
                                iteration: plan.iteration,
                                maxIterations: plan.maxIterations,
                                query: plan.query,
                            });
                        },
                        onWebSources: (data) => {
                            const normalized = normalizeWebSourcesPayload(data);
                            if (normalized.sources.length === 0 && normalized.images.length === 0) return;

                            if (!accumulatedWebSources) {
                                accumulatedWebSources = {
                                    sources: normalized.sources,
                                    images: normalized.images,
                                    provider: normalized.provider,
                                    imageProvider: normalized.imageProvider,
                                    searchType: normalized.searchType,
                                    query: normalized.query,
                                };
                            } else {
                                accumulatedWebSources.sources = [...accumulatedWebSources.sources, ...normalized.sources];
                                accumulatedWebSources.images = [...accumulatedWebSources.images, ...normalized.images];
                                if (normalized.provider) accumulatedWebSources.provider = normalized.provider;
                                if (normalized.imageProvider) accumulatedWebSources.imageProvider = normalized.imageProvider;
                                if (normalized.searchType) accumulatedWebSources.searchType = normalized.searchType;
                                if (normalized.query) accumulatedWebSources.query = normalized.query;
                            }
                        },
                        onDownloadProgress: (data) => {
                            setDownloadProgress(data);
                            // Clear when complete
                            if (data.percent && data.percent >= 100) {
                                setTimeout(() => setDownloadProgress(null), 4000);
                            }
                        },
                        onConversationMeta: (meta) => {
                            // Server assigned a conversation ID — adopt it and migrate the local entry
                            if (meta.conversation_id) {
                                const wasActive = ownerIsActive();
                                const prevOwner = streamOwnerConversationId;
                                // Migrate the local conversation entry from client ID to
                                // server UUID. UI-01: keyed on the stream's OWNER, not the
                                // active id — after a mid-stream switch the active
                                // conversation is someone else's and renaming IT to this
                                // stream's server UUID would corrupt both threads.
                                if (prevOwner && prevOwner !== meta.conversation_id) {
                                    useChatStore.setState((state) => ({
                                        conversations: state.conversations.map(c =>
                                            c.id === prevOwner ? { ...c, id: meta.conversation_id } : c
                                        ),
                                    }));
                                }
                                // The server UUID is this stream's true owner (CX-18).
                                streamOwnerConversationId = meta.conversation_id;
                                // UI-01: only re-point the ACTIVE conversation while this
                                // stream still owns it — the meta of a switched-away
                                // stream must not yank the user back.
                                if (wasActive) {
                                    setActiveConversationId(meta.conversation_id);
                                    // Update the ref immediately so plan feedback uses correct ID
                                    conversationIdRef.current = meta.conversation_id;
                                }
                            }
                        },
                        onRunMeta: (meta) => {
                            turnRunMeta = meta;
                            updateLastAssistantRunMeta(meta, ownerFor());  // UI-01
                        },
                        onToolTrace: (calls) => {
                            // Pass the stream's OWNING conversation (captured at
                            // send time, upgraded at meta time) — the live ref
                            // tracks the ACTIVE conversation and misroutes after
                            // a mid-stream switch (CX-18 / UI-02).
                            updateLastAssistantToolTrace(calls, streamOwnerConversationId);
                        },
                        onUsage: (usage) => {
                            if (usage?.totalTokens > 0) updateLastAssistantUsage(usage.totalTokens, usage.durationMs, ownerFor());  // UI-01
                        },
                        onComplete: () => {
                            // UI-01: the thinking rail + global streaming flag belong to
                            // the ACTIVE conversation. After a switch the rail was reset
                            // and isStreaming (if set) describes the new conversation's
                            // stream — a finished background stream must touch neither.
                            if (ownerIsActive()) attachThinkingToLastMessage();

                            // Merge web sources now that text generation is complete
                            if (accumulatedWebSources) {
                                mergeWebSourcesMessage(accumulatedWebSources, ownerFor());
                            }

                            if (ownerIsActive()) setStreaming(false);
                            // Reload conversation list from server so new/updated chats appear in sidebar
                            if (isAuthenticated) {
                                loadConversations();
                            }
                        },
                        // UI-03: the stream drained without [DONE] or an error event —
                        // keep whatever streamed, but mark the turn degraded instead of
                        // presenting the truncated text as a clean completion.
                        onIncomplete: (partialText: string) => {
                            if (ownerIsActive()) attachThinkingToLastMessage();
                            if (accumulatedWebSources) {
                                mergeWebSourcesMessage(accumulatedWebSources, ownerFor());
                            }
                            updateLastAssistantMessage(
                                (partialText.trim() ? partialText + "\n\n" : "") +
                                "⚠️ *The response ended unexpectedly — this answer may be incomplete.*",
                                ownerFor(),
                            );
                            // Reuses the Feature-4 failed/degraded machinery: the eval
                            // gate exempts the turn, matching how the backend marks
                            // timed-out turns in persisted runMeta.
                            markLastAssistantRunFailed("stream_truncated", ownerFor());
                            if (ownerIsActive()) setStreaming(false);
                            if (isAuthenticated) {
                                loadConversations();
                            }
                        },
                        onError: (error: string, status?: number) => {
                            if (ownerIsActive()) attachThinkingToLastMessage();  // UI-01
                            // Feature 4: mark the turn failed so the eval gate
                            // exempts it. run_meta arrives before the outcome is
                            // known and rich_meta only learns "failed" at persist
                            // time, so without this the LIVE turn would demand a
                            // rating for an answer the backend never produced.
                            markLastAssistantRunFailed(status ? `http_${status}` : "stream_error", ownerFor());
                            if (status === 401) {
                                // CX-05: a runtime 401 means the cookie expired.
                                // But transient/cold-start 401s (backend just
                                // restarted) used to log the user out instantly
                                // and scrub the session (live F-12b: ≥3
                                // spontaneous logouts in one campaign night).
                                // Probe /me ONCE; only clear auth if it also
                                // rejects.
                                void (async () => {
                                    try {
                                        await verifyAuth();
                                    } catch { /* network blip — treat as unverified */ }
                                    if (!useAuthStore.getState().isAuthenticated) {
                                        clearAuth();
                                        updateLastAssistantMessage("Your session expired — please sign in again.", ownerFor());
                                    } else {
                                        updateLastAssistantMessage(
                                            "That request hit a transient authorization error — your session is still active, please retry.",
                                            ownerFor(),
                                        );
                                    }
                                })();
                            } else {
                                updateLastAssistantMessage(`Error: ${error}`, ownerFor());
                            }

                            // Merge web sources on error too if they were retrieved
                            if (accumulatedWebSources) {
                                mergeWebSourcesMessage(accumulatedWebSources, ownerFor());
                            }

                            if (ownerIsActive()) setStreaming(false);  // UI-01
                        },
                    },
                    controller.signal,
                );
            } // end if-else
        } catch (err) {
            const message = err instanceof Error ? err.message : "Request failed.";
            updateLastAssistantMessage(`Error: ${message}`, ownerFor());  // UI-01
            if (ownerIsActive()) setStreaming(false);
        }
    }, [
        addMessage,
        mergeWebSourcesMessage,
        updateLastAssistantMessage,
        updateLastAssistantThinking,
        updateLastAssistantRunMeta,
        updateLastAssistantUsage,
        updateLastAssistantToolTrace,
        setStreaming,
        isStreaming,
        activeConversationId,
        setActiveConversation,
        selectedModel,
        // Eval gate reads live message/rating state — stale closures here would
        // let an unrated turn through, or block an already-rated one.
        evalMode,
        blockRatings,
        visibleBlocks,
        markLastAssistantRunFailed,
        messages,
        addThinkingStep,
        clearThinking,
        attachThinkingToLastMessage,
        clearTaskExecution,
        handleTaskGroup,
        handleTaskUpdate,
        handleTaskList,
        setActiveConversationId,
        loadConversations,
        isAuthenticated,
        clearAuth,
        normalizeWebSourcesPayload,
    ]);

    // ── Plan review handlers ────────────────────────────────────
    const handlePlanApprove = useCallback(async () => {
        // The plan carries its own conversation (the stream owner's) — prefer
        // it over the active-conversation ref, which a mid-stream switch
        // re-points (UI-02).
        const cid = pendingPlan?.conversationId || conversationIdRef.current;
        if (!cid || !pendingPlan) return;
        setPlanSubmitting(true);
        try {
            await submitPlanFeedback(cid, true, "", undefined, pendingPlan.runId);  // UIAPI-08
            setPendingPlan(null);
        } catch (err) {
            console.error("Plan approval failed:", err);
            // Surface error to the user via a status message
            updateLastAssistantMessage(
                `⚠️ Plan approval failed: ${err instanceof Error ? err.message : "Network error"}. Please try again.`
            );
        } finally {
            setPlanSubmitting(false);
        }
    }, [pendingPlan, updateLastAssistantMessage]);

    const handlePlanFeedback = useCallback(async (feedback: string) => {
        const cid = pendingPlan?.conversationId || conversationIdRef.current;
        if (!cid || !pendingPlan) return;
        setPlanSubmitting(true);
        try {
            await submitPlanFeedback(cid, false, feedback, undefined, pendingPlan.runId);  // UIAPI-08
            // Don't clear pendingPlan — the Conductor will emit a new plan_review event
        } catch (err) {
            console.error("Plan feedback failed:", err);
            updateLastAssistantMessage(
                `⚠️ Plan feedback failed: ${err instanceof Error ? err.message : "Network error"}. Please try again.`
            );
        } finally {
            setPlanSubmitting(false);
        }
    }, [pendingPlan, updateLastAssistantMessage]);

    const handleSuggestionClick = (prompt: string) => { setInputValue(prompt); handleSend(prompt); };
    const hasMessages = messages.length > 0;
    // The phone home screen gets its own chrome: a drawer trigger, the wordmark,
    // and a theme toggle. Once a chat exists it reverts to the standard header.
    const isMobileHome = isMobile && !hasMessages;
    const conversationTitle = activeConversationId
        ? conversations.find(c => c.id === activeConversationId)?.title || "New Research Session"
        : "New Research Session";

    return (
        <main className={`flex-1 flex flex-col h-full overflow-hidden relative z-10 ${hasMessages ? "chat-session-active" : ""}`}>
            {isMobileHome ? (
                <MobileHomeBar />
            ) : (
                <header className="shrink-0 flex items-center justify-between px-3 md:px-6 py-1.5 md:py-2 border-b border-slate-800/80 glass-panel">
                    <div className="flex items-center gap-3">
                        <button onClick={toggleSidebar} className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/10 transition-all"><PanelLeft className="w-5 h-5" /></button>
                        <h2 className="text-base font-semibold text-white tracking-tight">
                            {hasMessages ? messages[0].content.slice(0, 50) + (messages[0].content.length > 50 ? "…" : "") : conversationTitle}
                        </h2>
                    </div>
                    <div className="flex items-center gap-1 transition-opacity">
                        <button
                            onClick={() => activeConversationId && toggleStar(activeConversationId)}
                            title="Star this chat"
                            className={`p-2 rounded-lg transition-all ${isStarred
                                ? "text-yellow-500 hover:bg-white/10 hover:text-yellow-400"
                                : "text-slate-400 hover:text-yellow-500 hover:bg-white/10"
                                }`}
                        >
                            <Star className="w-[18px] h-[18px]" fill={isStarred ? "currentColor" : "none"} />
                        </button>
                    </div>
                </header>
            )}

            {hasMessages ? (
              <>
                <div
                    ref={scrollRef}
                    onScroll={handleScroll}
                    onWheel={markUserScrollIntent}
                    onTouchStart={markUserScrollIntent}
                    onTouchMove={markUserScrollIntent}
                    className="flex-1 overflow-y-auto chat-scrim px-4 md:px-8 py-6 space-y-6"
                >
                    <div className="w-full max-w-[var(--q-chat-content-width)] mx-auto space-y-6">
                        {(() => {
                            // Find the last text-type assistant message for attaching thinking/task state
                            const lastTextAssistantIdx = (() => {
                                for (let j = messages.length - 1; j >= 0; j--) {
                                    if (messages[j].role === "assistant" && messages[j].type === "text") return j;
                                }
                                return -1;
                            })();

                            return messages.map((msg, i) => {
                                const isLastAssistant = isStreaming && i === lastTextAssistantIdx;
                                // Task execution state persists AFTER streaming ends
                                // so the widget auto-collapses instead of vanishing.
                                const isLastAssistantMsg = i === lastTextAssistantIdx;
                                const hasTaskData = taskGroups.length > 0 || taskItems.size > 0 || taskChecklist !== null;
                                const execState = isLastAssistantMsg && hasTaskData ? {
                                    groups: taskGroups,
                                    tasks: taskItems,
                                    checklist: taskChecklist,
                                    isActive: taskExecutionActive && isStreaming,
                                } : null;
                                let reportPrompt = "";
                                for (let j = i - 1; j >= 0; j--) {
                                    if (messages[j].role === "user") {
                                        reportPrompt = messages[j].content;
                                        break;
                                    }
                                }
                                return (
                                    <ChatMessage
                                        key={msg.id}
                                        message={msg}
                                        isStreaming={isLastAssistant}
                                        thinkingSteps={isLastAssistant ? thinkingSteps : msg.thinkingSteps}
                                        thinkingStatus={isLastAssistant ? thinkingStatus : (msg.thinkingSteps ? "completed" : undefined)}
                                        taskExecutionState={execState}
                                        observationGraph={turnGraphs[msg.id]}
                                        reportPrompt={reportPrompt}
                                    />
                                );
                            });
                        })()}
                        {pendingPlan && (
                            <PlanReviewWidget
                                plan={pendingPlan}
                                onApprove={handlePlanApprove}
                                onFeedback={handlePlanFeedback}
                                isSubmitting={planSubmitting}
                            />
                        )}
                    </div>
                </div>

                {/* Scroll-to-bottom button */}
                {showScrollBtn && (
                    <div className="absolute bottom-24 left-1/2 -translate-x-1/2 z-20">
                        <button
                            onClick={scrollToBottom}
                            className="glass-control w-9 h-9 flex items-center justify-center rounded-full transition-all duration-200 hover:scale-110 animate-in fade-in slide-in-from-bottom-2 duration-200"
                            style={{
                                color: 'var(--q-text-secondary)',
                                boxShadow: '0 4px 16px -2px rgba(0,0,0,0.25)',
                            }}
                            title="Scroll to bottom"
                        >
                            <ArrowDown className="w-4 h-4" />
                        </button>
                    </div>
                )}

                {/* Download progress bar — inline before input */}
                {downloadProgress && (
                    <div className="px-4 md:px-8 max-w-[var(--q-chat-content-width)] mx-auto w-full">
                        <DownloadProgress data={downloadProgress} />
                    </div>
                )}

                {/* Eval-mode gate nudge (Feature 4). Only reachable in a
                    conversation — the hero composer has no prior turn to rate. */}
                {evalNudge && (
                    <div className="px-4 md:px-8 max-w-[var(--q-chat-content-width)] mx-auto w-full">
                        <div className="mb-2 flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
                            <Star className="h-3.5 w-3.5 shrink-0" />
                            <span>{evalNudge}</span>
                        </div>
                    </div>
                )}

                <ChatInput
                    onSend={handleSend} onStop={handleStop} isStreaming={isStreaming} initialValue={inputValue}
                    grounded={grounded} onGroundedChange={setGrounded}
                    webSearch={webSearch} onWebSearchChange={setWebSearch} hitCount={hitCount}
                />
              </>
            ) : (
              <>
                {/* On phones the composer docks at the bottom of the screen instead
                    of sitting inline under the hero, so it stays thumb-reachable. */}
                <EmptyState
                    onSuggestionClick={handleSuggestionClick}
                    hitCount={hitCount}
                    showDisclaimer={!isMobileHome}
                    composer={isMobileHome ? undefined : (
                        <ChatInput
                            variant="hero" onSend={handleSend} onStop={handleStop} isStreaming={isStreaming} initialValue={inputValue}
                            grounded={grounded} onGroundedChange={setGrounded}
                            webSearch={webSearch} onWebSearchChange={setWebSearch} hitCount={hitCount}
                        />
                    )}
                />
                {isMobileHome && (
                    <ChatInput
                        onSend={handleSend} onStop={handleStop} isStreaming={isStreaming} initialValue={inputValue}
                        grounded={grounded} onGroundedChange={setGrounded}
                        webSearch={webSearch} onWebSearchChange={setWebSearch} hitCount={hitCount}
                    />
                )}
              </>
            )}
        </main>
    );
}
