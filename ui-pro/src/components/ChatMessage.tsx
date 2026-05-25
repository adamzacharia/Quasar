"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Copy, Check, Database, User as UserIcon, ThumbsUp, ThumbsDown } from "lucide-react";
import { IconOpenBook, IconWebGlobe } from "./icons/QuasarIcons";
import { useState, useRef, useEffect, type ReactNode } from "react";
import type { Message } from "../lib/types";
import { useAuthStore } from "../lib/auth-store";
import { DataTableCard } from "./DataTableCard";
import { PaperCard } from "./PaperCard";
import { ThoughtProcessWidget, ThoughtStep } from "./ThoughtProcessWidget";
import { TaskExecutionWidget, type TaskExecutionState } from "./TaskExecutionWidget";
import { WebSourcesCard } from "./WebSourcesCard";
import { useChatStore } from "../lib/store";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";


function CodeBlock({ language, children }: { language: string; children: string }) {
    const [copied, setCopied] = useState(false);
    const copy = () => { navigator.clipboard.writeText(children); setCopied(true); setTimeout(() => setCopied(false), 2000); };
    return (
        <div className="relative rounded-xl overflow-hidden my-3 group">
            <div className="absolute top-2 right-2 z-10">
                <button onClick={copy} className="p-1.5 bg-slate-700/70 text-slate-300 hover:text-white rounded-md transition-all opacity-0 group-hover:opacity-100">
                    {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </button>
            </div>
            <SyntaxHighlighter language={language} style={oneDark} customStyle={{ margin: 0, borderRadius: "0.75rem", fontSize: "0.8rem", padding: "1.25rem" }}>
                {children}
            </SyntaxHighlighter>
        </div>
    );
}

function MessageActions({ message }: { message: Message }) {
    const [copied, setCopied] = useState(false);
    const [feedback, setFeedback] = useState<"like" | "dislike" | null>(null);
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
                    feedback: newFeedback,
                    conversation_id: activeConversationId || "",
                    model: selectedModel || "",
                    response_preview: message.content?.slice(0, 500) || "",
                }),
            });
        } catch {
            // Silent fail — feedback is non-critical
        }
    };

    return (
        <div className="flex items-center gap-1 mt-2 opacity-0 group-hover/msg:opacity-100 transition-opacity">
            <button onClick={copyText} title="Copy response"
                className="p-1.5 text-slate-500 hover:text-slate-200 hover:bg-slate-700/50 rounded-lg transition-all">
                {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            </button>
            <button onClick={() => sendFeedback("like")} title="Good response"
                className={`p-1.5 rounded-lg transition-all ${feedback === "like" ? "text-emerald-400 bg-emerald-500/10" : "text-slate-500 hover:text-slate-200 hover:bg-slate-700/50"}`}>
                <ThumbsUp className="w-4 h-4" fill={feedback === "like" ? "currentColor" : "none"} />
            </button>
            <button onClick={() => sendFeedback("dislike")} title="Bad response"
                className={`p-1.5 rounded-lg transition-all ${feedback === "dislike" ? "text-red-400 bg-red-500/10" : "text-slate-500 hover:text-slate-200 hover:bg-slate-700/50"}`}>
                <ThumbsDown className="w-4 h-4" fill={feedback === "dislike" ? "currentColor" : "none"} />
            </button>
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

interface ChatMessageProps {
    message: Message;
    isStreaming?: boolean;
    thinkingSteps?: ThoughtStep[];
    thinkingStatus?: "idle" | "running" | "completed";
    taskExecutionState?: TaskExecutionState | null;
}

export function ChatMessage({ message, isStreaming, thinkingSteps, thinkingStatus, taskExecutionState }: ChatMessageProps) {
    const isUser = message.role === "user";
    const hasContent = !!message.content;
    const hasThinking = (thinkingSteps && thinkingSteps.length > 0) || !!message.thinking;

    const { user, isAuthenticated } = useAuthStore();
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
                        <div className="bg-card-dark border border-primary/20 rounded-2xl rounded-tr-sm px-5 py-3 text-slate-100 shadow-sm space-y-3">
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
                                        <span key={i} className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800/80 border border-slate-700 rounded-xl text-xs text-slate-300">
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
                        <div className="size-8 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-400 shrink-0 mb-1 flex items-center justify-center text-white text-xs font-bold">
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
                    <ThoughtProcessWidget title={message.toolCall.displayName || message.toolCall.name} status={message.toolCall.status} steps={steps} />
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
            <div className="pl-11">
                <DataTableCard data={message.dataTable} />
            </div>
        );
    }

    if (message.type === "papers" && message.papers && message.papers.length > 0) {
        const VISIBLE_MAX = 10;
        const shown = message.papers.slice(0, VISIBLE_MAX);
        const hasMore = message.papers.length > VISIBLE_MAX;
        return (
            <div className="pl-11">
                <div className={`grid grid-cols-1 sm:grid-cols-2 gap-3 ${hasMore ? "max-h-[680px] overflow-y-auto pr-1 custom-scrollbar" : ""}`}>
                    {shown.map((paper) => (
                        <PaperCard key={paper.id} paper={paper} />
                    ))}
                </div>
                {hasMore && (
                    <p className="text-xs text-slate-500 mt-2 text-right">
                        Showing {VISIBLE_MAX} of {message.papers.length} papers
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
            <div className="pl-11">
                <div className="mt-4 rounded-xl border border-cyan-500/30 bg-slate-900/60 overflow-hidden shadow-xl shadow-cyan-500/5">
                    <img
                        src={message.imageUrl}
                        alt={message.imageCaption || "Rendered FITS image"}
                        className="w-full max-h-[600px] object-contain bg-black"
                        loading="lazy"
                    />
                    {message.imageCaption && (
                        <div className="px-4 py-2.5 border-t border-slate-700/50 flex items-center justify-between">
                            <span className="text-xs text-slate-400">{message.imageCaption}</span>
                            <a
                                href={message.imageUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1"
                            >
                                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                                Full size
                            </a>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    // -- Web Sources (source cards + image grid from web search) --
    if (message.type === "web_sources" && (message.webSources?.length || message.webImages?.length)) {
        return (
            <div className="pl-11">
                <WebSourcesCard sources={message.webSources} images={message.webImages} />
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

                    {/* Thinking Process Widget — rendered ABOVE content */}
                    {hasThinking && (
                        <ThoughtProcessWidget
                            title="Thinking"
                            status={thinkingStatus === "running" && !hasContent ? "running" : "completed"}
                            steps={thinkingSteps || []}
                            forceCollapsed={hasContent}
                        >
                            {message.thinking && (() => {
                                if (thinkingStatus === "running") {
                                    return <StreamingThinking text={message.thinking} />;
                                }

                                // When thinking is completed, display full markdown at normal 100% opacity
                                return (
                                    <div className="text-slate-300 text-xs leading-relaxed max-w-none pt-2 border-t border-slate-800/40 prose prose-invert prose-xs">
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
                        <div className={`text-slate-200 prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-pre:p-0 prose-pre:bg-transparent ${message.type === 'critique' ? 'border-l-4 border-red-500 pl-4 py-1 bg-red-950/10 rounded-r-xl' : ''}`}>
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
                                // Replace 📚 and 🌐 emojis with custom SVG icons
                                p({ children }) {
                                    return <p>{replaceEmojisWithIcons(children)}</p>;
                                },
                                em({ children }) {
                                    return <em>{replaceEmojisWithIcons(children)}</em>;
                                },
                                strong({ children }) {
                                    return <strong>{replaceEmojisWithIcons(children)}</strong>;
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
                                    return <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-200 whitespace-nowrap">{children}</th>;
                                },
                                td({ children }) {
                                    return <td className="px-4 py-2 text-slate-300 text-xs leading-relaxed">{children}</td>;
                                },
                                // Block LLM-hallucinated image URLs -- only system-provided images
                                // (FITS, web search grid) should render via their own components.
                                img() {
                                    return null;
                                },
                            }}>{message.content.replace(/!\[([^\]]*)\]\([^)]+\)/g, '')}</ReactMarkdown>
                        </div>
                    )}

                    {/* Streaming cursor */}
                    {isStreaming && <div className="w-2 h-5 bg-primary/80 animate-pulse rounded-sm" />}

                    {/* Action bar: copy, like, dislike — shown at bottom on hover */}
                    {hasContent && !isStreaming && <MessageActions message={message} />}
                </div>
            </div>
        </div>
    );
}
