"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Copy, Check, Database } from "lucide-react";
import { useState } from "react";
import type { Message } from "@/lib/types";
import { DataTableCard } from "./DataTableCard";
import { PaperCard } from "./PaperCard";
import { ThoughtProcessWidget, ThoughtStep } from "./ThoughtProcessWidget";

function renderWithTags(content: string): React.ReactNode {
    const parts = content.split(/(@archive|@paper|@search)/gi);
    return parts.map((part, i) => {
        const lower = part.toLowerCase();
        if (lower === "@archive") return <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary text-white text-xs font-bold mr-1 align-middle select-none"><Database className="w-3.5 h-3.5" />@archive</span>;
        if (lower === "@paper") return <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-600 text-white text-xs font-bold mr-1 align-middle select-none">@paper</span>;
        if (lower === "@search") return <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-indigo-500 text-white text-xs font-bold mr-1 align-middle select-none">@search</span>;
        return <span key={i}>{part}</span>;
    });
}

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

interface ChatMessageProps {
    message: Message;
    isStreaming?: boolean;
    thinkingSteps?: ThoughtStep[];
    thinkingStatus?: "idle" | "running" | "completed";
}

export function ChatMessage({ message, isStreaming, thinkingSteps, thinkingStatus }: ChatMessageProps) {
    const isUser = message.role === "user";
    const hasContent = !!message.content;
    const hasThinking = thinkingSteps && thinkingSteps.length > 0;

    if (isUser) {
        return (
            <div className="flex justify-end">
                <div className="flex items-end gap-3 max-w-[80%]">
                    <div className="space-y-1">
                        <div className="flex justify-end"><span className="text-[10px] text-slate-400 uppercase font-medium tracking-wider mr-1">You</span></div>
                        <div className="bg-card-dark border border-primary/20 rounded-2xl rounded-tr-sm px-5 py-3 text-slate-100 shadow-sm space-y-3">
                            {message.content && <p className="leading-relaxed">{renderWithTags(message.content)}</p>}

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
                    <div className="size-8 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-400 shrink-0 mb-1 flex items-center justify-center text-white text-xs font-bold">AZ</div>
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
                    <div className="size-8 rounded-xl bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center shadow-lg shadow-primary/20 shrink-0 mt-1">
                        <span className="text-white font-bold text-sm">Q</span>
                    </div>
                    <ThoughtProcessWidget title={message.toolCall.displayName || message.toolCall.name} status={message.toolCall.status} steps={steps} />
                </div>
            </div>
        );
    }

    // Assistant message — with integrated thinking process
    return (
        <div className="flex justify-start">
            <div className="flex items-start gap-3 w-full lg:max-w-[95%]">
                <div className="size-8 rounded-xl bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center shadow-lg shadow-primary/20 shrink-0 mt-1">
                    <span className="text-white font-bold text-sm">Q</span>
                </div>
                <div className="space-y-3 w-full min-w-0">
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] text-slate-400 uppercase font-medium tracking-wider">Quasar AI</span>
                        <span className="text-[10px] text-slate-500">{message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                    </div>

                    {/* Thinking Process Widget — rendered ABOVE content */}
                    {hasThinking && (
                        <ThoughtProcessWidget
                            title="Processing Pipeline"
                            status={thinkingStatus === "running" && !hasContent ? "running" : "completed"}
                            steps={thinkingSteps}
                            forceCollapsed={hasContent}
                        />
                    )}

                    {/* Content — rendered BELOW thinking */}
                    {hasContent && (
                        <div className="text-slate-200 prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-pre:p-0 prose-pre:bg-transparent">
                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
                                code({ className, children, ...props }) {
                                    const match = /language-(\w+)/.exec(className || "");
                                    const code = String(children).replace(/\n$/, "");
                                    if (match) return <CodeBlock language={match[1]}>{code}</CodeBlock>;
                                    return <code className="bg-slate-800 px-1.5 py-0.5 rounded text-primary text-sm font-mono" {...props}>{children}</code>;
                                },
                            }}>{message.content}</ReactMarkdown>
                        </div>
                    )}

                    {/* Streaming cursor */}
                    {isStreaming && <div className="w-2 h-5 bg-primary/80 animate-pulse rounded-sm" />}

                    {/* Data table */}
                    {message.dataTable && <DataTableCard data={message.dataTable} />}

                    {/* Papers */}
                    {message.papers && message.papers.length > 0 && (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {message.papers.map((paper) => (<PaperCard key={paper.id} paper={paper} />))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
