/* eslint-disable @next/next/no-img-element */
"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Copy, Check, Loader2, User as UserIcon, ThumbsUp, ThumbsDown, Send, X } from "lucide-react";
import { IconOpenBook, IconWebGlobe } from "./icons/QuasarIcons";
import { useState, useRef, useEffect, type ReactNode } from "react";
import type { Message } from "../lib/types";
import { useAuthStore } from "../lib/auth-store";
import { DataTableCard } from "./DataTableCard";
import { PaperCard } from "./PaperCard";
import { ThoughtProcessWidget, ThoughtStep } from "./ThoughtProcessWidget";
import { TaskExecutionWidget, type TaskExecutionState } from "./TaskExecutionWidget";
import { WebSourcesCard } from "./WebSourcesCard";
import { HipsImageCard } from "./HipsImageCard";
import { useChatStore } from "../lib/store";
import { useThemeStore } from "../lib/theme-store";
import { ObservationPaperGraph, type ResearchGraph } from "./ObservationPaperGraph";
import { canSubmitIssueReport, shouldOpenIssueReport } from "../lib/feedback-report";
import { safeAssistantWebText } from "../lib/content-safety";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";


function CodeBlock({ language, children }: { language: string; children: string }) {
    const [copied, setCopied] = useState(false);
    // Match the syntax theme to the active app theme so token colors keep proper
    // contrast in light mode (oneDark's dark-tuned pastels look washed out on a
    // light background — see the light-mode pre/code rules in globals.css).
    const theme = useThemeStore((s) => s.theme);
    const copy = () => { navigator.clipboard.writeText(children); setCopied(true); setTimeout(() => setCopied(false), 2000); };
    return (
        <div className="relative rounded-xl overflow-hidden my-3 group">
            <div className="absolute top-2 right-2 z-10">
                <button onClick={copy} className="p-1.5 bg-slate-700/70 text-slate-300 hover:text-white rounded-md transition-all opacity-0 group-hover:opacity-100">
                    {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </button>
            </div>
            <SyntaxHighlighter language={language} style={theme === "light" ? oneLight : oneDark} customStyle={{ margin: 0, borderRadius: "0.75rem", fontSize: "0.8rem", padding: "1.25rem" }}>
                {children}
            </SyntaxHighlighter>
        </div>
    );
}

function formatTokenCount(n: number): string {
    if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
    return String(n);
}

/* Provider-reported tokens when the backend sent a usage event; otherwise a
   ~chars/4 estimate so older / still-streaming messages get a figure too. */
function tokenLabelFor(content: string, usageTokens?: number): string {
    if (usageTokens && usageTokens > 0) return `${formatTokenCount(usageTokens)} tokens`;
    const estimate = Math.ceil((content || "").length / 4);
    return estimate > 0 ? `~${formatTokenCount(estimate)} tokens` : "";
}

function MessageActions({ message, reportPrompt = "" }: { message: Message; reportPrompt?: string }) {
    const [copied, setCopied] = useState(false);
    const [feedback, setFeedback] = useState<"like" | "dislike" | null>(null);
    const [showReport, setShowReport] = useState(false);
    const [category, setCategory] = useState("stuck_slow");
    const [description, setDescription] = useState("");
    const [includeContext, setIncludeContext] = useState(false);
    const [reportState, setReportState] = useState<"idle" | "sending" | "sent">("idle");
    const [reportError, setReportError] = useState("");
    const { activeConversationId, selectedModel } = useChatStore();

    const copyText = () => {
        navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const sendFeedback = async (type: "like" | "dislike") => {
        const newFeedback = feedback === type ? null : type;
        setFeedback(newFeedback);
        if (!newFeedback) return; // toggled off — no API call
        try {
            const token = useAuthStore.getState().token;
            await fetch(`${API_BASE}/api/feedback`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                body: JSON.stringify({
                    message_id: message.id,
                    run_id: message.runMeta?.run_id || "",
                    feedback: newFeedback,
                    conversation_id: activeConversationId || "",
                    model: message.runMeta?.model || selectedModel || "",
                    prompt_preview: reportPrompt.slice(0, 200),
                    response_preview: message.content?.slice(0, 500) || "",
                }),
            });
        } catch {
            // Silent fail — feedback is non-critical
        }
    };

    const submitIssueReport = async () => {
        if (!description.trim() || !message.runMeta?.run_id) return;
        setReportState("sending");
        setReportError("");
        try {
            const token = useAuthStore.getState().token;
            const response = await fetch(`${API_BASE}/api/issue-reports`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                body: JSON.stringify({
                    run_id: message.runMeta.run_id,
                    message_id: message.id,
                    category,
                    description: description.trim(),
                    include_context: includeContext,
                    prompt_excerpt: includeContext ? reportPrompt : "",
                    response_excerpt: includeContext ? message.content : "",
                    technical_context: {
                        user_agent: navigator.userAgent,
                        page: window.location.pathname,
                        viewport: `${window.innerWidth}x${window.innerHeight}`,
                    },
                }),
            });
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || `HTTP ${response.status}`);
            }
            setReportState("sent");
        } catch (error) {
            setReportState("idle");
            setReportError(error instanceof Error ? error.message : "Could not submit report.");
        }
    };

    const tokenLabel = tokenLabelFor(message.content, message.usageTokens);

    return (
        <div className="mt-2">
            <div className="flex items-center gap-1">
                <div className={`flex items-center gap-1 ${showReport ? "opacity-100" : "opacity-0 group-hover/msg:opacity-100"} transition-opacity`}>
                    <button onClick={copyText} title="Copy response"
                        className="p-1.5 text-slate-500 hover:text-slate-200 hover:bg-slate-700/50 rounded-lg transition-all">
                        {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                    <button onClick={() => sendFeedback("like")} title="Good response"
                        className={`p-1.5 rounded-lg transition-all ${feedback === "like" ? "text-emerald-400 bg-emerald-500/10" : "text-slate-500 hover:text-slate-200 hover:bg-slate-700/50"}`}>
                        <ThumbsUp className="w-4 h-4" fill={feedback === "like" ? "currentColor" : "none"} />
                    </button>
                    <button onClick={() => { sendFeedback("dislike"); setShowReport(shouldOpenIssueReport("dislike")); }} title="Report a problem"
                        className={`p-1.5 rounded-lg transition-all ${feedback === "dislike" ? "text-red-400 bg-red-500/10" : "text-slate-500 hover:text-slate-200 hover:bg-slate-700/50"}`}>
                        <ThumbsDown className="w-4 h-4" fill={feedback === "dislike" ? "currentColor" : "none"} />
                    </button>
                </div>
                {tokenLabel && (
                    <span className="ml-1 text-[11px] text-slate-500 tabular-nums select-none" title={message.usageTokens ? "Tokens used for this response (provider-reported)" : "Estimated from response length"}>
                        {tokenLabel}
                    </span>
                )}
            </div>
            {showReport && (
                <div className="mt-2 max-w-xl rounded-xl border border-red-500/20 bg-slate-950/90 p-3 shadow-xl">
                    <div className="flex items-center justify-between">
                        <div className="text-xs font-semibold text-slate-200">Report a problem</div>
                        <button type="button" onClick={() => setShowReport(false)} className="text-slate-500 hover:text-slate-200">
                            <X className="h-3.5 w-3.5" />
                        </button>
                    </div>
                    {reportState === "sent" ? (
                        <div className="mt-3 text-xs text-emerald-300">Report saved. Thank you.</div>
                    ) : (
                        <>
                            <select value={category} onChange={(event) => setCategory(event.target.value)}
                                className="mt-3 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-200">
                                <option value="stuck_slow">Stuck or slow</option>
                                <option value="wrong_answer">Wrong answer</option>
                                <option value="incorrect_data">Incorrect data</option>
                                <option value="ui_problem">Interface problem</option>
                                <option value="other">Other</option>
                            </select>
                            <textarea value={description} onChange={(event) => setDescription(event.target.value)}
                                placeholder="What happened, and what did you expect?" rows={3}
                                className="mt-2 w-full resize-y rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-200 placeholder:text-slate-600" />
                            <label className="mt-2 flex items-start gap-2 text-[11px] text-slate-400">
                                <input type="checkbox" checked={includeContext}
                                    onChange={(event) => setIncludeContext(event.target.checked)} className="mt-0.5" />
                                Include this question and response. Technical diagnostics are included automatically.
                            </label>
                            {!message.runMeta?.run_id && (
                                <div className="mt-2 text-[11px] text-amber-300">Diagnostics are unavailable for this older message.</div>
                            )}
                            {reportError && <div className="mt-2 text-[11px] text-red-300">{reportError}</div>}
                            <button type="button" onClick={submitIssueReport}
                                disabled={!canSubmitIssueReport(description, message.runMeta?.run_id) || reportState === "sending"}
                                className="mt-3 flex items-center gap-2 rounded-lg bg-red-500/15 px-3 py-2 text-xs font-semibold text-red-200 disabled:opacity-40">
                                {reportState === "sending" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
                                Send report
                            </button>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}

/**
 * Replace emoji characters in text children with custom SVG icons.
 * Handles: 📚 → IconOpenBook, 🌐 → IconWebGlobe
 */
function replaceEmojisWithIcons(children: ReactNode): ReactNode {
    if (!children) return children;

    // Process arrays of children
    if (Array.isArray(children)) {
        return <>{children.map((child, i) => {
            if (typeof child === "string") {
                return <span key={`ec-${i}`}>{replaceEmojisInString(child, i)}</span>;
            }
            return child;
        })}</>;
    }

    // Process single string child
    if (typeof children === "string") {
        return replaceEmojisInString(children, 0);
    }

    return children;
}

function replaceEmojisInString(text: string, keyBase: number): ReactNode {
    // Check if text contains any of our target emojis
    if (!text.includes("\ud83d\udcda") && !text.includes("\ud83c\udf10")) {
        return text;
    }

    // Split by emoji and interleave with icon components
    const parts: ReactNode[] = [];
    let remaining = text;
    let partKey = 0;

    while (remaining.length > 0) {
        const bookIdx = remaining.indexOf("\ud83d\udcda");
        const globeIdx = remaining.indexOf("\ud83c\udf10");

        // Find the earliest emoji
        let nextIdx = -1;
        let emojiType: "book" | "globe" | null = null;

        if (bookIdx >= 0 && (globeIdx < 0 || bookIdx < globeIdx)) {
            nextIdx = bookIdx;
            emojiType = "book";
        } else if (globeIdx >= 0) {
            nextIdx = globeIdx;
            emojiType = "globe";
        }

        if (nextIdx < 0) {
            // No more emojis — push remaining text
            parts.push(<span key={`t-${keyBase}-${partKey++}`}>{remaining}</span>);
            break;
        }

        // Push text before emoji
        if (nextIdx > 0) {
            parts.push(<span key={`t-${keyBase}-${partKey++}`}>{remaining.substring(0, nextIdx)}</span>);
        }

        // Push the icon component
        if (emojiType === "book") {
            parts.push(
                <IconOpenBook key={`icon-${keyBase}-${partKey++}`} className="w-4 h-4 text-primary inline-block align-text-bottom mr-0.5" />
            );
            remaining = remaining.substring(nextIdx + 2); // 📚 is 2 chars (surrogate pair)
        } else {
            parts.push(
                <IconWebGlobe key={`icon-${keyBase}-${partKey++}`} className="w-4 h-4 text-cyan-400 inline-block align-text-bottom mr-0.5" />
            );
            remaining = remaining.substring(nextIdx + 2); // 🌐 is 2 chars
        }
    }

    return parts.length === 1 ? parts[0] : <>{parts}</>;
}

const CITATION_PATTERN = /(\u3010(?=[^\u3011]*\bSource\s*:)[^\u3011]*\u3011)/gi;

function CitationChip({ citation }: { citation: string }) {
    return (
        <span className="citation-chip" title={citation} aria-label={citation}>
            {getCitationLabel(citation)}
        </span>
    );
}

function replaceCitations(children: ReactNode): ReactNode {
    if (!children) return children;

    if (Array.isArray(children)) {
        return children.flatMap((child, i) => {
            if (typeof child !== "string") return [child];

            const replaced = replaceCitationsInString(child, i);
            return Array.isArray(replaced) ? replaced : [replaced];
        });
    }

    if (typeof children === "string") {
        return replaceCitationsInString(children, 0);
    }

    return children;
}

function renderAnswerText(children: ReactNode): ReactNode {
    return replaceEmojisWithIcons(replaceCitations(children));
}

function replaceCitationsInString(text: string, keyBase: number): ReactNode {
    if (!text.includes("\u3010") || !/\bSource\s*:/i.test(text)) {
        return text;
    }

    const parts: ReactNode[] = [];
    let lastIndex = 0;

    for (const match of text.matchAll(CITATION_PATTERN)) {
        const citation = match[0];
        const index = match.index ?? 0;

        if (index > lastIndex) {
            parts.push(text.slice(lastIndex, index));
        }

        parts.push(<CitationChip key={`cite-${keyBase}-${index}`} citation={citation} />);
        lastIndex = index + citation.length;
    }

    if (parts.length === 0) return text;

    if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex));
    }

    return parts;
}

function getCitationLabel(citation: string): string {
    const inner = citation.replace(/^\u3010|\u3011$/g, "");
    const source = extractCitationSource(inner);

    if (!source) return "Source";

    const page = extractCitationPage(inner);
    let label: string;

    if (/^(?:https?:)?\/\//i.test(source)) {
        // Explicit URL: keep the hostname (preserve the TLD), drop protocol/path.
        // Detect URLs only by scheme so filenames like "guide.pdf" aren't mistaken
        // for domains (their extension must not be stripped as if it were a TLD).
        try {
            const normalized = /^https?:\/\//i.test(source) ? source : `https:${source}`;
            label = new URL(normalized).hostname.replace(/^www\./, "");
        } catch {
            label = source;
        }
    } else {
        // File path / name: take the basename and strip a trailing extension.
        const filename = source.split(/[\\/]/).pop()?.trim() || source;
        label = filename.replace(/\.[^.\s\\/]+$/, "") || filename;
    }

    return page ? `${label} \u00b7 p.${page}` : label;
}

function extractCitationSource(text: string): string | null {
    const match = text.match(/\bSource\s*:\s*([\s\S]*?)(?=,\s*(?:Page|Date|Relevance)\b\s*:?\s*|$)/i);
    const source = match?.[1]?.trim();
    return source || null;
}

function extractCitationPage(text: string): string | null {
    const match = text.match(/\bPage\s*:?\s*([0-9]+(?:\s*[-\u2013]\s*[0-9]+)?)/i);
    return match?.[1]?.replace(/\s+/g, "") || null;
}

function StreamingThinking({ text }: { text: string }) {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [text]);

    return (
        <div
            ref={containerRef}
            className="text-xs leading-relaxed max-w-none pt-2 border-t border-slate-800/40 select-none overflow-y-hidden transition-all duration-300 max-h-[5.5rem] relative"
            style={{
                WebkitMaskImage: 'linear-gradient(to bottom, transparent 0%, rgba(0,0,0,0.3) 15%, black 60%)',
                maskImage: 'linear-gradient(to bottom, transparent 0%, rgba(0,0,0,0.3) 15%, black 60%)',
            }}
        >
            <p className="font-sans text-slate-300/90 whitespace-pre-wrap animate-in fade-in duration-300">
                {text}
            </p>
        </div>
    );
}

function AnswerBuffer() {
    return (
        <div className="flex max-w-2xl items-center gap-2 px-1 text-xs text-slate-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
            <span>Generating answer</span>
        </div>
    );
}

function InitialWaitingIndicator({ startTime }: { startTime: Date }) {
    const [elapsed, setElapsed] = useState(() => Math.max(0, Math.round((Date.now() - startTime.getTime()) / 1000)));

    useEffect(() => {
        const interval = setInterval(() => {
            setElapsed(Math.max(0, Math.round((Date.now() - startTime.getTime()) / 1000)));
        }, 1000);
        return () => clearInterval(interval);
    }, [startTime]);

    return (
        <div className="flex items-center gap-3 py-1">
            <div className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-primary/70 animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-primary/70 animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-primary/70 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
            <span className="text-xs text-slate-500">
                Waiting for response
                <span className="font-mono ml-1.5 text-slate-600">{elapsed}s</span>
            </span>
        </div>
    );
}

interface ChatMessageProps {
    message: Message;
    isStreaming?: boolean;
    thinkingSteps?: ThoughtStep[];
    thinkingStatus?: "idle" | "running" | "completed";
    taskExecutionState?: TaskExecutionState | null;
    observationGraph?: ResearchGraph;
    reportPrompt?: string;
}

export function ChatMessage({ message, isStreaming, thinkingSteps, thinkingStatus, taskExecutionState, observationGraph, reportPrompt }: ChatMessageProps) {
    const isUser = message.role === "user";
    const displayContent = isUser ? message.content : safeAssistantWebText(message.content);
    const hasContent = !!displayContent;
    const hasThinkingSteps = !!thinkingSteps?.length;
    const hasThinking = hasThinkingSteps || !!message.thinking;
    const thinkingIsRunning = thinkingStatus === "running" && !hasContent;
    const showAnswerBuffer = Boolean(isStreaming && hasContent);
    // Show immediate waiting indicator when streaming but nothing has arrived yet
    const showInitialWaiting = Boolean(isStreaming && !hasContent && !hasThinking && !taskExecutionState);

    const { user } = useAuthStore();
    const userInitials = user?.display_name
        ? user.display_name.split(' ').filter(Boolean).map((n: string) => n[0]).join('').substring(0, 2).toUpperCase()
        : user?.username
            ? user.username.substring(0, 2).toUpperCase()
            : null;

    if (isUser) {
        return (
            <div className="flex justify-end">
                <div className="flex items-end gap-3 max-w-[80%]">
                    <div className="space-y-1">
                        <div className="flex justify-end"><span className="text-[10px] text-slate-400 uppercase font-medium tracking-wider mr-1">You</span></div>
                        <div className="glass-surface border-primary/20 rounded-2xl rounded-tr-sm px-5 py-3 text-slate-100 space-y-3">
                            {message.content && (
                                <div className="prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-p:my-1">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
                                        // Keep user messages lightweight — no code blocks, just inline code
                                        code({ children, ...props }) {
                                            return <code className="bg-slate-800 px-1.5 py-0.5 rounded text-primary text-sm font-mono" {...props}>{children}</code>;
                                        },
                                        a({ href, children }) {
                                            return <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">{children}</a>;
                                        },
                                    }}>{message.content}</ReactMarkdown>
                                </div>
                            )}

                            {/* Image attachment previews */}
                            {message.attachmentPreviews && message.attachmentPreviews.length > 0 && (
                                <div className="flex flex-wrap gap-2">
                                    {message.attachmentPreviews.map((src, i) => (
                                        <img key={i} src={src} alt={`attachment-${i}`}
                                            className="max-h-48 max-w-xs rounded-xl object-cover border border-slate-700 cursor-pointer hover:opacity-90 transition-opacity"
                                            onClick={() => window.open(src, "_blank")}
                                        />
                                    ))}
                                </div>
                            )}

                            {/* Document chips */}
                            {message.attachmentNames && message.attachmentNames.length > 0 && (
                                <div className="flex flex-wrap gap-2">
                                    {message.attachmentNames.map((name, i) => (
                                        <span key={i} className="glass-control flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs text-slate-300">
                                            <svg className="w-3.5 h-3.5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                                            {name}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                    {user?.picture_url ? (
                        <img src={user.picture_url} alt={user.display_name || "You"} className="size-8 rounded-full object-cover shrink-0 mb-1 shadow-md" referrerPolicy="no-referrer" />
                    ) : (
                        <div className="size-8 rounded-full bg-gradient-to-tr from-[#818cf8] to-[#c77dff] shrink-0 mb-1 flex items-center justify-center text-on-accent text-xs font-bold">
                            {userInitials || <UserIcon className="w-4 h-4" />}
                        </div>
                    )}
                </div>
            </div>
        );
    }


    // Tool call only message
    if (message.toolCall && !hasThinking) {
        const steps: ThoughtStep[] = [];
        if (message.toolCall.input) {
            steps.push({ text: `INPUT: ${JSON.stringify(message.toolCall.input)}`, status: "completed", isCode: true });
        }
        steps.push({
            text: message.toolCall.status === "running" ? "Executing Tool..." : message.toolCall.status === "completed" ? "Completed" : "Error executing tool",
            status: message.toolCall.status
        });
        if (message.toolCall.output) {
            steps.push({ text: `OUTPUT: ${message.toolCall.output}`, status: "completed", isCode: true });
        }

        return (
            <div className="flex justify-start">
                <div className="flex items-start gap-3 w-full lg:max-w-[95%]">
                    <div className="size-8 rounded-xl flex items-center justify-center shrink-0 mt-1 overflow-hidden" style={{ background: 'var(--q-bg)' }}>
                        <img src="/quasar_logo.png" alt="Quasar" className="size-7 object-contain" />
                    </div>
                    <ThoughtProcessWidget
                        title={message.toolCall.displayName || message.toolCall.name}
                        status={message.toolCall.status}
                        steps={steps}
                        startTime={message.timestamp}
                    />
                </div>
            </div>
        );
    }

    // ── Data-only / Papers-only / Notebook-only messages ──
    // These are continuation payloads injected into the stream and should
    // NOT render their own avatar + "QUASAR AI" header.
    if (message.type === "data" && message.dataTable) {
        // Suppress empty data tables (0 rows)
        const rows = message.dataTable.rows;
        if (!rows || rows.length === 0) return null;
        return (
            <>
                <div className="pl-11">
                    <DataTableCard data={message.dataTable} />
                </div>
                {observationGraph && (
                    <div className="mt-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                        <ObservationPaperGraph graph={observationGraph} />
                    </div>
                )}
            </>
        );
    }

    if (message.type === "papers" && message.papers && message.papers.length > 0) {
        if (observationGraph) {
            return null;
        }
        const shouldScroll = message.papers.length > 6;
        return (
            <div className="pl-11">
                <div className={`grid grid-cols-1 sm:grid-cols-2 gap-3 ${shouldScroll ? "max-h-[680px] overflow-y-auto pr-1 custom-scrollbar" : ""}`}>
                    {message.papers.map((paper) => (
                        <PaperCard key={paper.id} paper={paper} />
                    ))}
                </div>
                {shouldScroll && (
                    <p className="text-xs text-slate-500 mt-2 text-right">
                        All {message.papers.length} papers loaded
                    </p>
                )}
            </div>
        );
    }

    if (message.type === "notebook" && message.notebookData) {
        const nbData = message.notebookData.data as Record<string, unknown> | undefined;
        const cells = (nbData?.cells ?? []) as { cell_type: string; source: string[] }[];
        const cellCount = cells.length;
        const title = message.notebookData.title || "Analysis Notebook";
        return (
            <div className="pl-11">
                <button
                    onClick={() => {
                        const blob = new Blob([JSON.stringify(nbData, null, 2)], { type: "application/json" });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${title.replace(/\s+/g, '_').toLowerCase()}.ipynb`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                    }}
                    className="inline-flex items-center gap-2 mt-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:brightness-110 active:scale-[0.97]"
                    style={{
                        background: "rgba(99, 102, 241, 0.15)",
                        border: "1px solid rgba(99, 102, 241, 0.3)",
                        color: "rgb(165, 180, 252)",
                    }}
                >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                    {title} · {cellCount} cells
                </button>
            </div>
        );
    }

    // ── Rendered FITS Image ──────────────────────────────────────
    if (message.type === "image" && message.imageUrl) {
        return (
            <HipsImageCard
                key={message.imageUrl}
                imageUrl={message.imageUrl}
                caption={message.imageCaption || ""}
                imageMeta={message.imageMeta}
            />
        );
    }

    // -- Web Sources (source cards + image grid from web search) --
    if (message.type === "web_sources" && (message.webSources?.length || message.webImages?.length)) {
        return (
            <div className="pl-11">
                <WebSourcesCard
                    sources={message.webSources}
                    images={message.webImages}
                />
            </div>
        );
    }

    // Skip empty assistant text messages (placeholder bubbles with no content and no thinking)
    if (!hasContent && !hasThinking && !isStreaming) return null;

    // Assistant message — with integrated thinking process
    return (
        <div className="flex justify-start group/msg">
            <div className="flex items-start gap-3 w-full lg:max-w-[95%]">
                <div className="size-8 rounded-xl flex items-center justify-center shrink-0 mt-1 overflow-hidden" style={{ background: 'var(--q-bg)' }}>
                    <img src="/quasar_logo.png" alt="Quasar" className="size-7 object-contain" />
                </div>
                <div className="space-y-3 w-full min-w-0">
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] text-slate-400 uppercase font-medium tracking-wider">Quasar AI</span>
                        <span className="text-[10px] text-slate-500">{message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                    </div>

                    {/* Immediate waiting indicator — shown before any SSE events arrive */}
                    {showInitialWaiting && <InitialWaitingIndicator startTime={message.timestamp} />}

                    {/* Thinking Process Widget — rendered ABOVE content */}
                    {hasThinking && (
                        <ThoughtProcessWidget
                            title="Thinking"
                            status={thinkingIsRunning ? "running" : "completed"}
                            steps={thinkingSteps || []}
                            forceCollapsed={hasContent}
                            startTime={message.timestamp}
                            duration={message.thinkingDuration}
                        >
                            {message.thinking && (() => {
                                if (thinkingStatus === "running") {
                                    return <StreamingThinking text={message.thinking} />;
                                }

                                // When thinking is completed, display full markdown at normal 100% opacity
                                return (
                                    <div className="text-slate-300 text-xs leading-relaxed max-w-none pt-2 border-t border-slate-800/40 prose prose-invert prose-sm">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.thinking}</ReactMarkdown>
                                    </div>
                                );
                            })()}
                        </ThoughtProcessWidget>
                    )}

                    {/* Multi-Agent Workforce — always visible (full when active, collapsed when done) */}
                    {taskExecutionState && (
                        <TaskExecutionWidget state={taskExecutionState} />
                    )}

                    {/* Content — rendered BELOW thinking */}
                    {hasContent && (
                        <div className={`prose-answer text-slate-200 prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-pre:p-0 prose-pre:bg-transparent ${message.type === 'critique' ? 'border-l-4 border-red-500 pl-4 py-1 bg-red-950/10 rounded-r-xl' : ''}`}>
                            {message.type === 'critique' && (
                                <div className="text-red-400 font-bold mb-2 flex items-center gap-2">
                                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                                    Red Team TAC Critique
                                </div>
                            )}
                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
                                code({ className, children, ...props }) {
                                    const match = /language-(\w+)/.exec(className || "");
                                    const code = String(children).replace(/\n$/, "");
                                    if (match) return <CodeBlock language={match[1]}>{code}</CodeBlock>;
                                    return <code className="bg-slate-800 px-1.5 py-0.5 rounded text-primary text-sm font-mono" {...props}>{children}</code>;
                                },
                                // Replace source citation blobs and book/globe emojis in text-bearing nodes.
                                p({ children }) {
                                    return <p>{renderAnswerText(children)}</p>;
                                },
                                em({ children }) {
                                    return <em>{renderAnswerText(children)}</em>;
                                },
                                strong({ children }) {
                                    return <strong>{renderAnswerText(children)}</strong>;
                                },
                                li({ children, className }) {
                                    return <li className={className}>{renderAnswerText(children)}</li>;
                                },
                                h1({ children }) {
                                    return <h1>{renderAnswerText(children)}</h1>;
                                },
                                h2({ children }) {
                                    return <h2>{renderAnswerText(children)}</h2>;
                                },
                                h3({ children }) {
                                    return <h3>{renderAnswerText(children)}</h3>;
                                },
                                h4({ children }) {
                                    return <h4>{renderAnswerText(children)}</h4>;
                                },
                                h5({ children }) {
                                    return <h5>{renderAnswerText(children)}</h5>;
                                },
                                h6({ children }) {
                                    return <h6>{renderAnswerText(children)}</h6>;
                                },
                                a({ children, ...props }) {
                                    // Spread remaining props so GFM footnote anchors keep their
                                    // id / data-footnote-* / aria-* attributes (and href/title).
                                    return <a {...props}>{renderAnswerText(children)}</a>;
                                },
                                table({ children }) {
                                    return (
                                        <div className="overflow-x-auto my-4 rounded-xl border border-slate-700/30">
                                            <table className="w-full border-collapse text-sm">{children}</table>
                                        </div>
                                    );
                                },
                                thead({ children }) {
                                    return <thead className="bg-gradient-to-r from-primary/10 to-accent-purple/8 border-b border-primary/20">{children}</thead>;
                                },
                                tbody({ children }) {
                                    return <tbody>{children}</tbody>;
                                },
                                tr({ children }) {
                                    return <tr className="border-b border-slate-700/20 transition-colors hover:bg-white/[0.03]">{children}</tr>;
                                },
                                th({ children }) {
                                    return <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-200 whitespace-nowrap">{renderAnswerText(children)}</th>;
                                },
                                td({ children }) {
                                    return <td className="px-4 py-2 text-slate-300 text-xs leading-relaxed">{renderAnswerText(children)}</td>;
                                },
                                // Block LLM-hallucinated image URLs -- only system-provided images
                                // (FITS, web search grid) should render via their own components.
                                img() {
                                    return null;
                                },
                            }}>{displayContent.replace(/!\[([^\]]*)\]\([^)]+\)/g, '')}</ReactMarkdown>
                        </div>
                    )}

                    {showAnswerBuffer && <AnswerBuffer />}

                    {/* Live token estimate while the answer is still streaming */}
                    {hasContent && isStreaming && (
                        <div className="mt-2 text-[11px] text-slate-500 tabular-nums select-none">
                            {tokenLabelFor(displayContent)}
                        </div>
                    )}

                    {/* Action bar: copy, like, dislike — shown at bottom on hover */}
                    {hasContent && !isStreaming && (
                        <MessageActions
                            message={{ ...message, content: displayContent }}
                            reportPrompt={reportPrompt}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
