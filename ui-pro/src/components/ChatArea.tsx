"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { PanelLeft, Star } from "lucide-react";
import { useChatStore } from "../lib/store";
import { sendChatMessage, reviewProposal } from "../lib/api";
import { EmptyState } from "./EmptyState";
import { ChatInput } from "./ChatInput";
import { ChatMessage } from "./ChatMessage";
import type { Message, DataTableResult, Paper, ToolCall, NotebookData } from "../lib/types";
import { TaskExecutionWidget } from "./TaskExecutionWidget";
import { useAuthStore } from "../lib/auth-store";

interface AttachedFile { file: File; preview?: string; type: "image" | "document"; }

function generateId(): string { return Date.now().toString(36) + Math.random().toString(36).slice(2); }

export function ChatArea() {
    const {
        messages, addMessage, updateLastAssistantMessage,
        isStreaming, setStreaming, streamingContent,
        sidebarOpen, toggleSidebar,
        activeConversationId, setActiveConversation,
        selectedModel, conversations,
        toggleStar,
        thinkingSteps, thinkingStatus, addThinkingStep, clearThinking,
        attachThinkingToLastMessage,
        taskGroups, taskItems, taskChecklist, taskExecutionActive,
        handleTaskGroup, handleTaskUpdate, handleTaskList, clearTaskExecution,
    } = useChatStore();

    const { token } = useAuthStore();

    // Use a ref so handleSend always reads the CURRENT token (avoids stale closure)
    const tokenRef = useRef<string | null>(null);
    useEffect(() => {
        tokenRef.current = token ?? null;
    }, [token]);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const [inputValue, setInputValue] = useState("");
    const abortControllerRef = useRef<AbortController | null>(null);

    const activeConversation = conversations.find(c => c.id === activeConversationId);
    const isStarred = activeConversation?.isStarred || false;
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [messages, thinkingSteps]);

    const handleStop = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        attachThinkingToLastMessage();
        setStreaming(false);
    }, [attachThinkingToLastMessage, setStreaming]);

    const handleSend = useCallback(async (text: string, attachments?: AttachedFile[]) => {
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
        // ↓ Instantly show the Processing Pipeline widget — before the first network event
        addThinkingStep("Connecting to QUASAR engine", "running");


        const assistantMsgId = generateId();
        const assistantMsg: Message = { id: assistantMsgId, role: "assistant", content: "", type: "text", timestamp: new Date() };
        addMessage(assistantMsg);

        let accumulated = "";

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
                await sendChatMessage(
                    {
                        message: messageWithContext,
                        conversation_id: activeConversationId || undefined,
                        model: selectedModel,
                        attachments: attachments?.map(a => a.file),
                        token: tokenRef.current || undefined,  // always reads current auth state
                    },
                    {
                        onToken: (token: string) => {
                            accumulated += token;
                            updateLastAssistantMessage(accumulated);
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
                        onStatus: (() => {
                            let connected = false;
                            return (step: string, state: string) => {
                                if (!connected) {
                                    connected = true;
                                    addThinkingStep("Connecting to QUASAR engine", "completed");
                                }
                                addThinkingStep(step, state as "running" | "completed");
                            };
                        })(),
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
                        onPapers: (papers: Record<string, unknown>[]) => {
                            addMessage({
                                id: generateId(),
                                role: "assistant",
                                content: "Here are the relevant papers I found:",
                                type: "papers",
                                timestamp: new Date(),
                                papers: papers as unknown as Paper[],
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
                        onTaskGroup: (group) => handleTaskGroup(group),
                        onTaskUpdate: (update) => handleTaskUpdate(update),
                        onTaskList: (list) => handleTaskList(list),
                        onComplete: () => {
                            attachThinkingToLastMessage();
                            setStreaming(false);
                        },
                        onError: (error: string) => {
                            attachThinkingToLastMessage();
                            updateLastAssistantMessage(`Error: ${error}`);
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
        updateLastAssistantMessage,
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
    ]);

    const handleSuggestionClick = (prompt: string) => { setInputValue(prompt); handleSend(prompt); };
    const hasMessages = messages.length > 0;
    const conversationTitle = activeConversationId
        ? conversations.find(c => c.id === activeConversationId)?.title || "New Research Session"
        : "New Research Session";

    return (
        <main className="flex-1 flex flex-col h-full overflow-hidden relative z-10">
            <header className="shrink-0 flex items-center justify-between px-3 md:px-6 py-2 md:py-4 border-b border-slate-800/80 glass-panel">
                <div className="flex items-center gap-3">
                    <button onClick={toggleSidebar} className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800/50 transition-all"><PanelLeft className="w-5 h-5" /></button>
                    <h2 className="text-base font-semibold text-white tracking-tight">
                        {hasMessages ? messages[0].content.slice(0, 50) + (messages[0].content.length > 50 ? "…" : "") : conversationTitle}
                    </h2>
                </div>
                <div className="flex items-center gap-1 transition-opacity">
                    <button
                        onClick={() => activeConversationId && toggleStar(activeConversationId)}
                        title="Star this chat"
                        className={`p-2 rounded-lg transition-all ${isStarred
                            ? "text-yellow-500 hover:bg-slate-800/50 hover:text-yellow-400"
                            : "text-slate-400 hover:text-yellow-500 hover:bg-slate-800/50"
                            }`}
                    >
                        <Star className="w-[18px] h-[18px]" fill={isStarred ? "currentColor" : "none"} />
                    </button>
                </div>
            </header>

            {hasMessages ? (
                <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6">
                    <div className="max-w-4xl mx-auto space-y-6">
                        {messages.map((msg, i) => {
                            const isLastAssistant = isStreaming && msg.role === "assistant" && msg.type === "text" && i === messages.length - 1;
                            return (
                                <>
                                {/* Show TaskExecutionWidget before the last streaming assistant message */}
                                {isLastAssistant && taskExecutionActive && (
                                    <TaskExecutionWidget state={{
                                        groups: taskGroups,
                                        tasks: taskItems,
                                        checklist: taskChecklist,
                                        isActive: taskExecutionActive && isStreaming,
                                    }} />
                                )}
                                <ChatMessage
                                    key={msg.id}
                                    message={msg}
                                    isStreaming={isLastAssistant}
                                    thinkingSteps={isLastAssistant ? thinkingSteps : msg.thinkingSteps}
                                    thinkingStatus={isLastAssistant ? thinkingStatus : (msg.thinkingSteps ? "completed" : undefined)}
                                />
                                </>
                            );
                        })}
                    </div>
                </div>
            ) : (
                <EmptyState onSuggestionClick={handleSuggestionClick} />
            )}

            <ChatInput onSend={handleSend} onStop={handleStop} isStreaming={isStreaming} initialValue={inputValue} />
        </main>
    );
}
