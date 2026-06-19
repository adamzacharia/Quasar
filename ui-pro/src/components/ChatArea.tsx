"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { PanelLeft, Star, ArrowDown } from "lucide-react";
import { useChatStore } from "../lib/store";
import { sendChatMessage, reviewProposal, submitPlanFeedback } from "../lib/api";
import { EmptyState } from "./EmptyState";
import { ChatInput } from "./ChatInput";
import { ChatMessage } from "./ChatMessage";
import type { ResearchGraph } from "./ObservationPaperGraph";
import { DownloadProgress } from "./DownloadProgress";
import { PlanReviewWidget } from "./PlanReviewWidget";
import type { PlanReviewData } from "./PlanReviewWidget";
import type { Message, DataTableResult, Paper, ToolCall, NotebookData, WebImage, WebSource } from "../lib/types";
import { normalizeEvidenceQuality } from "../lib/evidence-quality";
import { buildObservationPaperGraph } from "../lib/research-graph";
import { useAuthStore } from "../lib/auth-store";

interface AttachedFile { file: File; preview?: string; type: "image" | "document"; }

function generateId(): string { return Date.now().toString(36) + Math.random().toString(36).slice(2); }

export function ChatArea() {
    const {
        messages, addMessage, mergeWebSourcesMessage, updateLastAssistantMessage, updateLastAssistantThinking,
        updateLastAssistantRunMeta,
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
    } = useChatStore();

    const { token, isAuthenticated } = useAuthStore();

    // Use a ref so handleSend always reads the CURRENT token (avoids stale closure)
    const tokenRef = useRef<string | null>(null);
    useEffect(() => {
        tokenRef.current = token ?? null;
    }, [token]);

    const [inputValue, setInputValue] = useState("");
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

        for (const msg of messages) {
            if (msg.role === "user") {
                if (currentTurnMessages.length > 0 && currentTurnDataMessageIds.length > 0) {
                    const graph = buildObservationPaperGraph(currentTurnMessages) as ResearchGraph | null;
                    if (graph) {
                        for (const id of currentTurnMessageIds) {
                            graphs[id] = graph;
                        }
                    }
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
            const graph = buildObservationPaperGraph(currentTurnMessages) as ResearchGraph | null;
            if (graph) {
                for (const id of currentTurnMessageIds) {
                    graphs[id] = graph;
                }
            }
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
            return {
                title: String(item.title || item.name || titleFromUrl(url)).trim(),
                url,
                snippet: String(item.snippet || item.content || item.text || item.description || "").trim(),
                evidenceQuality: normalizeEvidenceQuality(item.evidenceQuality || item.evidence_quality),
            };
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
            return normalized;
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
                    onToken: (token: string) => {
                        accumulated += token;
                        updateLastAssistantMessage(accumulated);
                    },
                    onStatus: (step: string, state: string) => {
                        addThinkingStep(step, state as "running" | "completed");
                    },
                    onComplete: () => {
                        attachThinkingToLastMessage();
                        setStreaming(false);
                    },
                    onError: (error: string) => {
                        attachThinkingToLastMessage();
                        updateLastAssistantMessage(`Error: ${error}`);
                        setStreaming(false);
                    }
                }, controller.signal);
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
                        token: tokenRef.current || undefined,  // always reads current auth state
                        grounded_summary: Boolean(options?.groundedSummary),
                        web_search: options?.webSearch !== false,
                    },
                    {
                        onToken: (token: string) => {
                            accumulated += token;
                            updateLastAssistantMessage(accumulated);
                        },
                        onThought: (thought: string) => {
                            accumulatedThought += thought;
                            updateLastAssistantThinking(accumulatedThought);
                        },
                        onToolCall: (toolName: string, input: string) => {
                            const displayName = toolName.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
                            addThinkingStep(`Calling tool: ${displayName}`, "completed");
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
                            });
                        },
                        onStatus: (step: string, state: string) => {
                            addThinkingStep(step, state as "running" | "completed");
                        },
                        onData: (data: Record<string, unknown>) => {
                            const tableData = data as unknown as DataTableResult;
                            addMessage({
                                id: generateId(),
                                role: "assistant",
                                content: "",
                                type: "data",
                                timestamp: new Date(),
                                dataTable: tableData,
                            });
                        },
                        onPapers: (rawPapers: Record<string, unknown>[]) => {
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
                                id: generateId(),
                                role: "assistant",
                                content: "Here are the relevant papers I found:",
                                type: "papers",
                                timestamp: new Date(),
                                papers,
                            });
                        },
                        onNotebook: (notebook: Record<string, unknown>) => {
                            addMessage({
                                id: generateId(),
                                role: "assistant",
                                content: `I've generated a Jupyter Notebook for your analysis: **${notebook.title || 'Dynamic Notebook'}**`,
                                type: "notebook",
                                timestamp: new Date(),
                                notebookData: notebook as unknown as NotebookData,
                            });
                        },
                        onImage: (img) => {
                            // Resolve relative URL to absolute backend URL
                            const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                            const imageUrl = img.url.startsWith("http") ? img.url : `${apiBase}${img.url}`;
                            addMessage({
                                id: generateId(),
                                role: "assistant",
                                content: img.caption || "",
                                type: "image",
                                timestamp: new Date(),
                                imageUrl: imageUrl,
                                imageCaption: img.caption || "",
                            });
                        },
                        onTaskGroup: (group) => handleTaskGroup(group),
                        onTaskUpdate: (update) => handleTaskUpdate(update),
                        onTaskList: (list) => handleTaskList(list),
                        onPlanReview: (plan) => {
                            setPendingPlan({
                                conversationId: conversationIdRef.current || plan.conversationId || "",
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
                                const oldId = useChatStore.getState().activeConversationId;
                                // Migrate the local conversation entry from client ID to server UUID
                                if (oldId && oldId !== meta.conversation_id) {
                                    useChatStore.setState((state) => ({
                                        conversations: state.conversations.map(c =>
                                            c.id === oldId ? { ...c, id: meta.conversation_id } : c
                                        ),
                                    }));
                                }
                                setActiveConversationId(meta.conversation_id);
                                // Update the ref immediately so plan feedback uses correct ID
                                conversationIdRef.current = meta.conversation_id;
                            }
                        },
                        onRunMeta: (meta) => {
                            updateLastAssistantRunMeta(meta);
                        },
                        onComplete: () => {
                            attachThinkingToLastMessage();
                            
                            // Merge web sources now that text generation is complete
                            if (accumulatedWebSources) {
                                mergeWebSourcesMessage(accumulatedWebSources);
                            }
                            
                            setStreaming(false);
                            // Reload conversation list from server so new/updated chats appear in sidebar
                            if (isAuthenticated && tokenRef.current) {
                                loadConversations(tokenRef.current);
                            }
                        },
                        onError: (error: string) => {
                            attachThinkingToLastMessage();
                            updateLastAssistantMessage(`Error: ${error}`);
                            
                            // Merge web sources on error too if they were retrieved
                            if (accumulatedWebSources) {
                                mergeWebSourcesMessage(accumulatedWebSources);
                            }
                            
                            setStreaming(false);
                        },
                    },
                    controller.signal,
                );
            } // end if-else
        } catch (err) {
            const message = err instanceof Error ? err.message : "Request failed.";
            updateLastAssistantMessage(`Error: ${message}`);
            setStreaming(false);
        }
    }, [
        addMessage,
        mergeWebSourcesMessage,
        updateLastAssistantMessage,
        updateLastAssistantThinking,
        updateLastAssistantRunMeta,
        setStreaming,
        isStreaming,
        activeConversationId,
        setActiveConversation,
        selectedModel,
        addThinkingStep,
        clearThinking,
        attachThinkingToLastMessage,
        clearTaskExecution,
        handleTaskGroup,
        handleTaskUpdate,
        handleTaskList,
        tokenRef,
        setActiveConversationId,
        loadConversations,
        isAuthenticated,
        normalizeWebSourcesPayload,
    ]);

    // ── Plan review handlers ────────────────────────────────────
    const handlePlanApprove = useCallback(async () => {
        const cid = conversationIdRef.current || pendingPlan?.conversationId;
        if (!cid || !pendingPlan) return;
        setPlanSubmitting(true);
        try {
            await submitPlanFeedback(cid, true, "", tokenRef.current || undefined);
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
        const cid = conversationIdRef.current || pendingPlan?.conversationId;
        if (!cid || !pendingPlan) return;
        setPlanSubmitting(true);
        try {
            await submitPlanFeedback(cid, false, feedback, tokenRef.current || undefined);
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
    const conversationTitle = activeConversationId
        ? conversations.find(c => c.id === activeConversationId)?.title || "New Research Session"
        : "New Research Session";

    return (
        <main className="flex-1 flex flex-col h-full overflow-hidden relative z-10">
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

            {hasMessages ? (
                <div
                    ref={scrollRef}
                    onScroll={handleScroll}
                    onWheel={markUserScrollIntent}
                    onTouchStart={markUserScrollIntent}
                    onTouchMove={markUserScrollIntent}
                    className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6"
                >
                    <div className="max-w-[var(--q-chat-content-width)] mx-auto space-y-6">
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
            ) : (
                <EmptyState onSuggestionClick={handleSuggestionClick} />
            )}

            {/* Scroll-to-bottom button */}
            {hasMessages && showScrollBtn && (
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

            <ChatInput onSend={handleSend} onStop={handleStop} isStreaming={isStreaming} initialValue={inputValue} />
        </main>
    );
}
