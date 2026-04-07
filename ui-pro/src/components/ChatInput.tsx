"use client";

import { useState, useRef, useEffect } from "react";
import { Send, PlusCircle, X, FileText, Image as ImageIcon, Square } from "lucide-react";

interface AttachedFile {
    file: File;
    preview?: string; // base64 data URL for images
    type: "image" | "document";
}

interface ChatInputProps {
    onSend: (message: string, attachments?: AttachedFile[]) => void;
    onStop?: () => void;
    isStreaming: boolean;
    initialValue?: string;
}

export function ChatInput({ onSend, onStop, isStreaming, initialValue = "" }: ChatInputProps) {
    const [value, setValue] = useState(initialValue);
    const [attachments, setAttachments] = useState<AttachedFile[]>([]);
    const inputRef = useRef<HTMLInputElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { if (initialValue) { setValue(initialValue); inputRef.current?.focus(); } }, [initialValue]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        files.forEach(file => {
            const type = file.type.startsWith("image/") ? "image" : "document";
            if (type === "image") {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    setAttachments(prev => [...prev, { file, preview: ev.target?.result as string, type }]);
                };
                reader.readAsDataURL(file);
            } else {
                setAttachments(prev => [...prev, { file, type }]);
            }
        });
        // Reset so same file can be re-selected
        e.target.value = "";
    };

    const removeAttachment = (index: number) => {
        setAttachments(prev => prev.filter((_, i) => i !== index));
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const hasContent = value.trim() || attachments.length > 0;
        if (hasContent && !isStreaming) {
            onSend(value.trim(), attachments.length > 0 ? attachments : undefined);
            setValue("");
            setAttachments([]);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as unknown as React.FormEvent);
        }
    };

    return (
        <div className="w-full px-4 pb-6 pt-4 z-20">
            <div className="max-w-3xl mx-auto relative">
                <form onSubmit={handleSubmit} className="relative group">
                    <div className="absolute inset-0 bg-primary/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    <div className="relative w-full bg-card-dark/90 backdrop-blur-xl border border-slate-600 rounded-2xl shadow-2xl ring-1 ring-white/10 focus-within:border-primary/50 focus-within:ring-primary/50 transition-all">

                        {/* Attachment previews */}
                        {attachments.length > 0 && (
                            <div className="flex flex-wrap gap-2 px-3 pt-3">
                                {attachments.map((att, i) => (
                                    <div key={i} className="relative group/att flex items-center gap-2 bg-slate-800/80 border border-slate-700 rounded-xl px-3 py-2 max-w-[200px]">
                                        {att.type === "image" && att.preview ? (
                                            <img src={att.preview} alt={att.file.name} className="w-8 h-8 rounded-lg object-cover shrink-0" />
                                        ) : (
                                            <FileText className="w-5 h-5 text-primary shrink-0" />
                                        )}
                                        <span className="text-xs text-slate-300 truncate max-w-[120px]">{att.file.name}</span>
                                        <button
                                            type="button"
                                            onClick={() => removeAttachment(i)}
                                            className="ml-1 p-0.5 rounded-full bg-slate-700 hover:bg-red-500/80 text-slate-400 hover:text-white transition-all opacity-0 group-hover/att:opacity-100"
                                        >
                                            <X className="w-3 h-3" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Input row */}
                        <div className="flex items-center gap-2 p-2">
                            {/* Hidden file input */}
                            <input
                                ref={fileInputRef}
                                type="file"
                                multiple
                                accept="image/*,.pdf,.txt,.csv,.md,.json,.fits"
                                className="hidden"
                                onChange={handleFileChange}
                            />
                            <button
                                type="button"
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isStreaming}
                                className="p-2.5 text-slate-400 hover:text-primary hover:bg-slate-700/50 rounded-full transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                title="Attach image or document"
                            >
                                <PlusCircle className="w-5 h-5" />
                            </button>
                            <input
                                ref={inputRef}
                                type="text"
                                value={value}
                                onChange={(e) => setValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={isStreaming ? "QUASAR is thinking..." : "Ask QUASAR about observations, data, or literature..."}
                                className="flex-1 bg-transparent border-none outline-none text-white placeholder-slate-500 focus:ring-0 text-sm"
                                disabled={isStreaming}
                            />
                            {isStreaming ? (
                                <button
                                    type="button"
                                    onClick={onStop}
                                    className="p-2.5 bg-red-500/80 hover:bg-red-500 text-white rounded-full transition-all shadow-lg shadow-red-500/20 flex items-center justify-center animate-pulse"
                                    title="Stop generating"
                                >
                                    <Square className="w-4 h-4 fill-current" />
                                </button>
                            ) : (
                                <button
                                    type="submit"
                                    disabled={!value.trim() && attachments.length === 0}
                                    className="p-2.5 bg-primary hover:bg-primary/90 text-white rounded-full transition-all shadow-lg shadow-primary/20 flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                    <Send className="w-5 h-5" />
                                </button>
                            )}
                        </div>
                    </div>
                </form>
                <div className="text-center mt-3">
                    <p className="text-[10px] text-slate-600">
                        QUASAR may produce inaccurate information.
                        <span className="text-slate-700 ml-2">· Accepts images, PDFs, FITS, CSV</span>
                    </p>
                </div>
            </div>
        </div>
    );
}
