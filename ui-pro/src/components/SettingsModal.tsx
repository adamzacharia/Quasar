"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { X, Upload, FileText, Trash2, Lock, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { useAuthStore } from "@/lib/auth-store";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PersonalDoc {
    id: string;
    filename: string;
    size_bytes: number;
    uploaded_at: string;
    chunk_count: number;
}

function formatBytes(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string) {
    return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/* ── Personalization Panel ───────────────────────────────── */
function PersonalizationPanel() {
    const { isAuthenticated, token } = useAuthStore();
    const [docs, setDocs] = useState<PersonalDoc[]>([]);
    const [loading, setLoading] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadMsg, setUploadMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
    const [dragging, setDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const authHeaders = { Authorization: `Bearer ${token}` };

    const fetchDocs = useCallback(async () => {
        if (!token) return;
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/personalization/documents`, { headers: authHeaders });
            if (res.ok) setDocs(await res.json());
        } catch { /* noop */ }
        setLoading(false);
    }, [token]);

    useEffect(() => { if (isAuthenticated) fetchDocs(); }, [isAuthenticated, fetchDocs]);

    const uploadFiles = async (files: FileList | File[]) => {
        if (!token) return;
        setUploading(true);
        setUploadMsg(null);
        const form = new FormData();
        Array.from(files).forEach(f => form.append("files", f));
        try {
            const res = await fetch(`${API_BASE}/api/personalization/upload`, {
                method: "POST", headers: authHeaders, body: form,
            });
            const data = await res.json();
            if (res.ok) {
                setUploadMsg({ type: "success", text: data.message || "Documents uploaded and indexed!" });
                fetchDocs();
            } else {
                setUploadMsg({ type: "error", text: data.detail || "Upload failed." });
            }
        } catch (e) {
            setUploadMsg({ type: "error", text: "Upload failed — is the backend running?" });
        }
        setUploading(false);
        setTimeout(() => setUploadMsg(null), 4000);
    };

    const deleteDoc = async (docId: string, filename: string) => {
        if (!token) return;
        if (!confirm(`Remove "${filename}" from your personal knowledge base?`)) return;
        try {
            const res = await fetch(`${API_BASE}/api/personalization/document/${encodeURIComponent(docId)}`, {
                method: "DELETE", headers: authHeaders,
            });
            if (res.ok) setDocs(prev => prev.filter(d => d.id !== docId));
        } catch { /* noop */ }
    };

    const onDrop = (e: React.DragEvent) => {
        e.preventDefault(); setDragging(false);
        if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
    };

    if (!isAuthenticated) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8 py-16">
                <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center">
                    <Lock className="w-7 h-7 text-slate-500" />
                </div>
                <div>
                    <p className="text-sm font-semibold text-slate-300">Sign in to use Personalization</p>
                    <p className="text-xs text-slate-500 mt-1.5 max-w-xs">
                        Your personal knowledge base is tied to your account. Sign in to upload documents, books, and notes that Quasar will remember.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6 overflow-y-auto h-full">
            <div>
                <h3 className="text-sm font-semibold text-white mb-1">Personal Knowledge Base</h3>
                <p className="text-xs text-slate-400">
                    Upload documents, PDFs, books, or notes. Quasar will index them and use them as part of your long-term memory during conversations. Only you can access your documents.
                </p>
            </div>

            {/* Drop zone */}
            <div
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-2xl p-8 cursor-pointer transition-all
                    ${dragging ? "border-primary bg-primary/10" : "border-slate-700 hover:border-slate-500 hover:bg-slate-800/40"}`}
            >
                <input ref={fileInputRef} type="file" multiple hidden
                    accept=".pdf,.txt,.md,.csv,.json,.docx"
                    onChange={e => e.target.files && uploadFiles(e.target.files)}
                />
                {uploading ? (
                    <><Loader2 className="w-8 h-8 text-primary animate-spin" /><p className="text-sm text-slate-400">Uploading and indexing...</p></>
                ) : (
                    <>
                        <Upload className="w-8 h-8 text-slate-500" />
                        <div className="text-center">
                            <p className="text-sm font-medium text-slate-300">Drop files here or click to browse</p>
                            <p className="text-xs text-slate-500 mt-1">Supports PDF, TXT, MD, CSV, JSON, DOCX</p>
                        </div>
                    </>
                )}
            </div>

            {/* Upload status message */}
            {uploadMsg && (
                <div className={`flex items-center gap-2.5 px-4 py-3 rounded-xl text-sm ${uploadMsg.type === "success" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-red-500/10 text-red-400 border border-red-500/20"}`}>
                    {uploadMsg.type === "success" ? <CheckCircle className="w-4 h-4 shrink-0" /> : <AlertCircle className="w-4 h-4 shrink-0" />}
                    {uploadMsg.text}
                </div>
            )}

            {/* Document list */}
            <div>
                <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Indexed Documents ({docs.length})
                    </span>
                    {loading && <Loader2 className="w-3.5 h-3.5 text-slate-500 animate-spin" />}
                </div>

                {docs.length === 0 && !loading ? (
                    <div className="text-center py-8 text-xs text-slate-500">
                        No documents yet. Upload something to get started.
                    </div>
                ) : (
                    <div className="space-y-2">
                        {docs.map(doc => (
                            <div key={doc.id} className="flex items-center gap-3 px-4 py-3 bg-slate-800/60 border border-slate-700/50 rounded-xl group">
                                <FileText className="w-4 h-4 text-primary shrink-0" />
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm text-slate-200 font-medium truncate">{doc.filename}</p>
                                    <p className="text-[10px] text-slate-500">
                                        {formatBytes(doc.size_bytes)} · {doc.chunk_count} chunks · {formatDate(doc.uploaded_at)}
                                    </p>
                                </div>
                                <button
                                    onClick={() => deleteDoc(doc.id, doc.filename)}
                                    className="p-1.5 text-slate-600 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-all opacity-0 group-hover:opacity-100"
                                    title="Remove document"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

/* ── Main Settings Modal ─────────────────────────────────── */
interface SettingsModalProps {
    open: boolean;
    onClose: () => void;
}

export function SettingsModal({ open, onClose }: SettingsModalProps) {
    const backdropRef = useRef<HTMLDivElement>(null);

    // Close on Escape
    useEffect(() => {
        const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
        if (open) document.addEventListener("keydown", handler);
        return () => document.removeEventListener("keydown", handler);
    }, [open, onClose]);

    if (!open) return null;

    return (
        <div
            ref={backdropRef}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm"
            onClick={(e) => { if (e.target === backdropRef.current) onClose(); }}
        >
            <div className="relative w-[720px] max-w-[95vw] h-[560px] max-h-[90vh] bg-[#0f1117] border border-slate-700/60 rounded-2xl shadow-2xl shadow-black/60 flex overflow-hidden animate-in fade-in zoom-in-95 duration-200">

                {/* Close button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 z-10 p-1.5 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-all"
                >
                    <X className="w-4 h-4" />
                </button>

                {/* Left nav */}
                <div className="w-48 shrink-0 border-r border-slate-700/50 p-4">
                    <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold px-2 mb-3">Settings</p>
                    <button className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium bg-primary/10 text-primary border border-primary/20">
                        <FileText className="w-4 h-4" />
                        Personalization
                    </button>
                </div>

                {/* Right content */}
                <div className="flex-1 overflow-hidden flex flex-col">
                    <div className="px-6 py-5 border-b border-slate-700/50">
                        <h2 className="text-base font-semibold text-white">Personalization</h2>
                        <p className="text-xs text-slate-400 mt-0.5">Your private knowledge base for smarter conversations</p>
                    </div>
                    <div className="flex-1 overflow-hidden">
                        <PersonalizationPanel />
                    </div>
                </div>
            </div>
        </div>
    );
}
