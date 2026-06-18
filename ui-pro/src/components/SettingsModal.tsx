"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { X, Upload, FileText, Trash2, Lock, Loader2, CheckCircle, AlertCircle, Wrench, Plus, Copy, BarChart3, Download, Sparkles, Sun, Moon, ScrollText, ExternalLink, Github } from "lucide-react";
import { useAuthStore } from "../lib/auth-store";
import { resetOnboarding } from "./OnboardingOverlay";
import { useThemeStore } from "../lib/theme-store";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────

interface PersonalDoc {
    id: string;
    filename: string;
    size_bytes: number;
    uploaded_at: string;
    chunk_count: number;
}

interface UserTool {
    name: string;
    description: string;
    code: string;
    api_key_name: string;
}

interface MCPServer {
    name: string;
    transport?: "stdio" | "http";
    command?: string;
    args?: string[];
    url?: string;
    env?: Record<string, string>;
}

interface ProviderKeyMeta {
    provider: string;
    key_last4: string;
    token_limit?: number | null;
    status: string;
    created_at: string;
    updated_at: string;
    last_tested_at?: string | null;
}

interface QuotaBucket {
    used_tokens: number;
    limit_tokens?: number | null;
    unlimited: boolean;
    exhausted: boolean;
}

interface UsageQuota {
    platform: Record<string, QuotaBucket>;
    byok: Record<string, QuotaBucket>;
    is_admin: boolean;
    is_quota_exempt?: boolean;
    platform_quota_window_days?: number;
}

// ── Utilities ─────────────────────────────────────────────────────────────

function formatBytes(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string) {
    return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function formatTokens(value?: number | null) {
    const n = Number(value || 0);
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(n % 1_000 === 0 ? 0 : 1)}K`;
    return n.toLocaleString();
}

// ── Personalization Panel ─────────────────────────────────────────────────

function PersonalizationPanel() {
    const { isAuthenticated, token } = useAuthStore();
    const [docs, setDocs] = useState<PersonalDoc[]>([]);
    const [loading, setLoading] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadMsg, setUploadMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
    const [dragging, setDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const fetchDocs = useCallback(async () => {
        if (!token) return;
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/personalization/documents`, { headers: { Authorization: `Bearer ${token}` } });
            if (res.ok) setDocs(await res.json());
        } catch { /* noop */ }
        setLoading(false);
    }, [token]);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { if (isAuthenticated) fetchDocs(); }, [isAuthenticated, fetchDocs]);

    const uploadFiles = async (files: FileList | File[]) => {
        if (!token) return;
        setUploading(true);
        setUploadMsg(null);
        const form = new FormData();
        Array.from(files).forEach(f => form.append("files", f));
        try {
            const res = await fetch(`${API_BASE}/api/personalization/upload`, {
                method: "POST", headers: { Authorization: `Bearer ${token}` }, body: form,
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
                method: "DELETE", headers: { Authorization: `Bearer ${token}` },
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
                <div className="w-14 h-14 rounded-2xl glass-control flex items-center justify-center">
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
                            <div key={doc.id} className="glass-control flex items-center gap-3 px-4 py-3 rounded-xl group">
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

// ── Custom Tools Panel ────────────────────────────────────────────────────

const _SIMPLE_TEMPLATE = `def greet_astronomer(name: str) -> dict:
    """Greets a radio astronomer by name. Pass the astronomer's last name."""
    return {"message": \`Hello, Dr. \${name}! Welcome to Quasar.\`}
`;

const _API_TEMPLATE = `import os, requests

def search_my_service(query: str, max_results: int = 5) -> dict:
    """Search MyService for astronomy data. Requires MY_SERVICE_KEY."""
    api_key = os.environ.get("MY_SERVICE_KEY", "")
    if not api_key:
        return {"error": "MY_SERVICE_KEY not configured. Add it in the API Key fields."}
    resp = requests.get(
        "https://api.myservice.com/search",
        params={"q": query, "limit": max_results, "key": api_key},
        timeout=15,
    )
    return resp.json()
`;

function CustomToolsPanel() {
    const { isAuthenticated, token } = useAuthStore();
    const [activeTab, setActiveTab] = useState<"templates" | "add" | "installed">("installed");
    const [tools, setTools] = useState<UserTool[]>([]);
    const [loading, setLoading] = useState(false);
    
    // Form state
    const [tName, setTName] = useState("");
    const [tDesc, setTDesc] = useState("");
    const [tCode, setTCode] = useState("");
    const [tKeyName, setTKeyName] = useState("");
    const [tKeyVal, setTKeyVal] = useState("");
    const [formMsg, setFormMsg] = useState<{type: "success" | "error", text: string} | null>(null);
    const [saving, setSaving] = useState(false);

    const fetchTools = useCallback(async () => {
        if (!token) return;
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/user-tools`, { headers: { Authorization: `Bearer ${token}` } });
            if (res.ok) {
                setTools(await res.json());
            }
        } catch { /* noop */ }
        setLoading(false);
    }, [token]);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { if (isAuthenticated) fetchTools(); }, [isAuthenticated, fetchTools]);

    const handleSave = async () => {
        if (!token) return;
        setFormMsg(null);
        if (!tName.trim() || !tCode.trim()) {
            setFormMsg({ type: "error", text: "Tool name and code are required." });
            return;
        }
        
        setSaving(true);
        try {
            const res = await fetch(`${API_BASE}/api/user-tools`, {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}` 
                },
                body: JSON.stringify({
                    name: tName,
                    description: tDesc || tName,
                    code: tCode,
                    api_key_name: tKeyName || undefined,
                    api_key_value: tKeyVal || undefined
                })
            });
            const data = await res.json();
            
            if (res.ok) {
                setFormMsg({ type: "success", text: `Tool ${tName} added successfully!` });
                setTName(""); setTDesc(""); setTCode(""); setTKeyName(""); setTKeyVal("");
                fetchTools();
                setTimeout(() => { setActiveTab("installed"); setFormMsg(null); }, 1500);
            } else {
                setFormMsg({ type: "error", text: data.detail || "Failed to save tool." });
            }
        } catch (e) {
            setFormMsg({ type: "error", text: "Network error saving tool." });
        }
        setSaving(false);
    };

    const handleDelete = async (name: string) => {
        if (!token) return;
        if (!confirm(`Delete custom tool "${name}"?`)) return;
        
        try {
            const res = await fetch(`${API_BASE}/api/user-tools/${encodeURIComponent(name)}`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` }
            });
            if (res.ok) fetchTools();
        } catch { /* noop */ }
    };

    const applyTemplate = (name: string, desc: string, code: string) => {
        setTName(name);
        setTDesc(desc);
        setTCode(code);
        setActiveTab("add");
    };

    if (!isAuthenticated) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8 py-16">
                <div className="w-14 h-14 rounded-2xl glass-control flex items-center justify-center">
                    <Lock className="w-7 h-7 text-slate-500" />
                </div>
                <div>
                    <p className="text-sm font-semibold text-slate-300">Sign in to use Custom Tools</p>
                    <p className="text-xs text-slate-500 mt-1.5 max-w-xs">
                        Your custom Python tools are tied to your account. Sign in to add tools that Quasar can use to solve tasks.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full bg-[#0a0a0f]">
            {/* Top Sub-tabs */}
            <div className="flex border-b border-slate-700/50 px-4 pt-2">
                <button 
                    onClick={() => setActiveTab("installed")}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors duration-200 \${activeTab === "installed" ? "border-primary text-white" : "border-transparent text-slate-400 hover:text-slate-200"}`}
                >
                    Installed ({tools.length})
                </button>
                <button 
                    onClick={() => setActiveTab("add")}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors duration-200 flex flex-row items-center gap-1.5 \${activeTab === "add" ? "border-primary text-white" : "border-transparent text-slate-400 hover:text-slate-200"}`}
                >
                    <Plus className="w-4 h-4" /> Add Tool
                </button>
                <button 
                    onClick={() => setActiveTab("templates")}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors duration-200 \${activeTab === "templates" ? "border-primary text-white" : "border-transparent text-slate-400 hover:text-slate-200"}`}
                >
                    Templates
                </button>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                
                {activeTab === "installed" && (
                    <div>
                        {loading && <div className="text-slate-400 text-sm flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin"/> Loading...</div>}
                        {!loading && tools.length === 0 && (
                            <div className="text-center py-12">
                                <Wrench className="w-12 h-12 text-slate-700 mx-auto mb-3" />
                                <p className="text-sm text-slate-400">No custom tools installed yet.</p>
                                <button onClick={() => setActiveTab("add")} className="mt-4 text-xs font-semibold text-primary hover:text-primary-400">
                                    + Create your first tool
                                </button>
                            </div>
                        )}
                        {!loading && tools.length > 0 && (
                            <div className="space-y-3">
                                {tools.map(tool => (
                                    <div key={tool.name} className="glass-control rounded-xl p-4 group flex items-start justify-between">
                                        <div className="min-w-0 flex-1 pr-4">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="text-sm font-semibold text-slate-200 truncate">{tool.name}</h4>
                                                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 bg-slate-700/50 px-2 py-0.5 rounded">Custom</span>
                                            </div>
                                            <p className="text-xs text-slate-400 line-clamp-2">{tool.description || <span className="italic">No description</span>}</p>
                                            {tool.api_key_name && (
                                                <div className="mt-2 text-[10px] font-mono text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded inline-block">
                                                    KEY: {tool.api_key_name}
                                                </div>
                                            )}
                                        </div>
                                        <button
                                            onClick={() => handleDelete(tool.name)}
                                            className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-all opacity-0 group-hover:opacity-100 shrink-0"
                                            title="Delete tool"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {activeTab === "add" && (
                    <div className="space-y-4 max-w-2xl">
                        {formMsg && (
                            <div className={`flex items-center gap-2.5 px-4 py-3 rounded-xl text-sm \${formMsg.type === "success" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-red-500/10 text-red-400 border border-red-500/20"}`}>
                                {formMsg.type === "success" ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                                {formMsg.text}
                            </div>
                        )}

                        <div>
                            <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Tool Name</label>
                            <input 
                                type="text" 
                                value={tName} onChange={e => setTName(e.target.value)}
                                placeholder="my_custom_tool"
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                            />
                            <p className="text-[10px] text-slate-500 mt-1">Lowercase, digits, underscores only. Must exactly match the Python function name.</p>
                        </div>

                        <div>
                            <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Description</label>
                            <textarea 
                                value={tDesc} onChange={e => setTDesc(e.target.value)}
                                placeholder="What does this tool do? Be descriptive, the AI reads this to know when to use it."
                                rows={2}
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                            />
                        </div>

                        <div>
                            <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Python Code</label>
                            <textarea 
                                value={tCode} onChange={e => setTCode(e.target.value)}
                                placeholder="def my_custom_tool(param: str) -> dict:\n    return {}"
                                rows={8}
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-white placeholder-slate-600 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary leading-relaxed"
                            />
                        </div>

                        <div className="glass-control p-4 rounded-xl space-y-3">
                            <h4 className="text-xs font-semibold text-slate-300">API Key Storage (Optional)</h4>
                            <p className="text-[10px] text-slate-500 leading-snug">If your tool needs an API key, name it here. We will securely inject it into <code className="text-pink-400">os.environ</code> when the agent starts.</p>
                            <div className="flex gap-3">
                                <div className="flex-1">
                                    <input 
                                        type="text" value={tKeyName} onChange={e => setTKeyName(e.target.value)}
                                        placeholder="MY_SERVICE_KEY"
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-primary"
                                    />
                                </div>
                                <div className="flex-1">
                                    <input 
                                        type="password" value={tKeyVal} onChange={e => setTKeyVal(e.target.value)}
                                        placeholder="sk-..."
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-primary"
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="pt-2 flex justify-end">
                            <button 
                                onClick={handleSave}
                                disabled={saving}
                                className="bg-primary hover:bg-primary-600 disabled:opacity-50 text-white text-sm font-semibold py-2 px-6 rounded-lg transition-colors flex items-center gap-2"
                            >
                                {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
                                Save & Register Tool
                            </button>
                        </div>
                    </div>
                )}

                {activeTab === "templates" && (
                    <div className="space-y-6 max-w-2xl pb-8">
                        <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-xl p-4 text-sm text-indigo-200">
                            <strong>How tools work:</strong> Write a standard Python function. The AI sees your function&apos;s name, docstring, and parameter types to auto-generate the JSON schema. Must return a <code>dict</code> or <code>str</code>.
                        </div>

                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="text-sm font-bold text-white">Simple Tool</h4>
                                <button onClick={() => applyTemplate("greet_astronomer", "Greets a radio astronomer by name.", _SIMPLE_TEMPLATE)} className="text-xs font-semibold text-primary hover:text-white flex items-center gap-1 bg-primary/10 hover:bg-primary px-2 py-1 rounded transition-colors"><Copy className="w-3 h-3"/> Use Template</button>
                            </div>
                            <pre className="text-xs font-mono text-slate-300 bg-[#0f1117] border border-slate-700 p-4 rounded-xl overflow-x-auto">{_SIMPLE_TEMPLATE}</pre>
                        </div>

                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="text-sm font-bold text-white">API Key Tool</h4>
                                <button onClick={() => applyTemplate("search_my_service", "Search MyService for astronomy data.", _API_TEMPLATE)} className="text-xs font-semibold text-primary hover:text-white flex items-center gap-1 bg-primary/10 hover:bg-primary px-2 py-1 rounded transition-colors"><Copy className="w-3 h-3"/> Use Template</button>
                            </div>
                            <pre className="text-xs font-mono text-slate-300 bg-[#0f1117] border border-slate-700 p-4 rounded-xl overflow-x-auto">{_API_TEMPLATE}</pre>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// ── MCP Servers Panel ───────────────────────────────────────────────────────

function MCPServersPanel() {
    const { isAuthenticated, token } = useAuthStore();
    const [activeTab, setActiveTab] = useState<"installed" | "add">("installed");
    const [servers, setServers] = useState<MCPServer[]>([]);
    const [loading, setLoading] = useState(false);
    
    // Form state
    const [sName, setSName] = useState("");
    const [sTransport, setSTransport] = useState<"stdio" | "http">("stdio");
    const [sCmd, setSCmd] = useState("npx");
    const [sArgs, setSargs] = useState("-y @modelcontextprotocol/server-everything");
    const [sUrl, setSUrl] = useState("https://huggingface.co/mcp");
    const [sEnvStr, setSEnvStr] = useState("");
    const [formMsg, setFormMsg] = useState<{type: "success" | "error", text: string} | null>(null);
    const [saving, setSaving] = useState(false);

    const fetchServers = useCallback(async () => {
        if (!token) return;
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/mcp-servers`, { headers: { Authorization: `Bearer ${token}` } });
            if (res.ok) setServers(await res.json());
        } catch { /* noop */ }
        setLoading(false);
    }, [token]);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { if (isAuthenticated) fetchServers(); }, [isAuthenticated, fetchServers]);

    const handleSave = async () => {
        if (!token) return;
        setFormMsg(null);
        if (!sName.trim()) {
            setFormMsg({ type: "error", text: "Server name is required." });
            return;
        }
        if (sTransport === "stdio" && !sCmd.trim()) {
            setFormMsg({ type: "error", text: "Command is required for stdio transport." });
            return;
        }
        if (sTransport === "http" && !sUrl.trim()) {
            setFormMsg({ type: "error", text: "URL is required for http transport." });
            return;
        }
        
        // Parse args
        const cmdArgs = sArgs.split(/\s+/).filter(a => a.trim().length > 0);
        
        // Parse Env
        const envDict: Record<string, string> = {};
        if (sEnvStr.trim()) {
            const lines = sEnvStr.split("\n");
            for (const line of lines) {
                const parts = line.split("=");
                if (parts.length >= 2) {
                    const k = parts[0].trim();
                    const v = parts.slice(1).join("=").trim();
                    if (k) envDict[k] = v;
                }
            }
        }
        
        setSaving(true);
        try {
            const bodyPayload = {
                name: sName.trim(),
                transport: sTransport,
                command: sTransport === "stdio" ? sCmd.trim() : null,
                args: sTransport === "stdio" ? cmdArgs : [],
                url: sTransport === "http" ? sUrl.trim() : null,
                env: envDict
            };
            
            const res = await fetch(`${API_BASE}/api/mcp-servers`, {
                method: "POST",
                headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
                body: JSON.stringify(bodyPayload)
            });
            const data = await res.json();
            if (res.ok) {
                setFormMsg({ type: "success", text: `Server ${sName} added successfully!` });
                setSName(""); setSCmd("npx"); setSargs(""); setSEnvStr(""); setSUrl("");
                fetchServers();
                setTimeout(() => { setActiveTab("installed"); setFormMsg(null); }, 1500);
            } else {
                setFormMsg({ type: "error", text: data.detail || "Failed to save server." });
            }
        } catch (e) {
            setFormMsg({ type: "error", text: "Network error saving server." });
        }
        setSaving(false);
    };

    const handleDelete = async (name: string) => {
        if (!token) return;
        if (!confirm(`Delete MCP server "${name}"?`)) return;
        try {
            const res = await fetch(`${API_BASE}/api/mcp-servers/${encodeURIComponent(name)}`, {
                method: "DELETE", headers: { Authorization: `Bearer ${token}` }
            });
            if (res.ok) fetchServers();
        } catch { /* noop */ }
    };

    if (!isAuthenticated) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8 py-16">
                <div className="w-14 h-14 rounded-2xl glass-control flex items-center justify-center">
                    <Lock className="w-7 h-7 text-slate-500" />
                </div>
                <div>
                    <p className="text-sm font-semibold text-slate-300">Sign in to use MCP Servers</p>
                    <p className="text-xs text-slate-500 mt-1.5 max-w-xs">Connecting external tools via the Model Context Protocol requires an account.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full bg-[#0a0a0f]">
            <div className="flex border-b border-slate-700/50 px-4 pt-2 shrink-0">
                <button 
                    onClick={() => setActiveTab("installed")}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors duration-200 \${activeTab === "installed" ? "border-primary text-white" : "border-transparent text-slate-400 hover:text-slate-200"}`}
                >
                    Connected ({servers.length})
                </button>
                <button 
                    onClick={() => setActiveTab("add")}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors duration-200 flex flex-row items-center gap-1.5 \${activeTab === "add" ? "border-primary text-white" : "border-transparent text-slate-400 hover:text-slate-200"}`}
                >
                    <Plus className="w-4 h-4" /> Add Server
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {activeTab === "installed" && (
                    <div>
                        {loading && <div className="text-slate-400 text-sm flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin"/> Loading...</div>}
                        {!loading && servers.length === 0 && (
                            <div className="text-center py-12">
                                <Wrench className="w-12 h-12 text-slate-700 mx-auto mb-3" />
                                <p className="text-sm text-slate-400 mb-2">No external MCP servers configured.</p>
                                <p className="text-[10px] text-slate-500 max-w-xs mx-auto">Connect official integrations like GitHub, Postgres, or Google Drive via stdio.</p>
                                <button onClick={() => setActiveTab("add")} className="mt-4 text-xs font-semibold text-primary hover:text-primary-400">
                                    + Connect a server
                                </button>
                            </div>
                        )}
                        {!loading && servers.length > 0 && (
                            <div className="space-y-3">
                                {servers.map(srv => (
                                    <div key={srv.name} className="glass-control rounded-xl p-4 group flex items-start justify-between">
                                        <div className="min-w-0 flex-1 pr-4">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="text-sm font-semibold text-slate-200 truncate">{srv.name}</h4>
                                                <span className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded ${srv.transport === 'http' ? 'text-purple-400 bg-purple-400/10' : 'text-blue-400 bg-blue-400/10'}`}>MCP {srv.transport || 'stdio'}</span>
                                            </div>
                                            {srv.transport === "http" ? (
                                                <p className="text-xs text-slate-400 font-mono mt-2 truncate bg-black/30 p-1.5 rounded border border-slate-700/50">
                                                    {srv.url}
                                                </p>
                                            ) : (
                                                <p className="text-xs text-slate-400 font-mono mt-2 truncate bg-black/30 p-1.5 rounded border border-slate-700/50">
                                                    $ {srv.command} {(srv.args || []).join(" ")}
                                                </p>
                                            )}
                                            {srv.env && Object.keys(srv.env).length > 0 && (
                                                <div className="mt-2 text-[10px] font-mono text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded inline-block">
                                                    Environment: {Object.keys(srv.env).join(", ")}
                                                </div>
                                            )}
                                        </div>
                                        <button
                                            onClick={() => handleDelete(srv.name)}
                                            className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-all opacity-0 group-hover:opacity-100 shrink-0"
                                            title="Delete server"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {activeTab === "add" && (
                    <div className="space-y-4 max-w-2xl">
                        {formMsg && (
                            <div className={`flex items-center gap-2.5 px-4 py-3 rounded-xl text-sm \${formMsg.type === "success" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-red-500/10 text-red-400 border border-red-500/20"}`}>
                                {formMsg.type === "success" ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                                {formMsg.text}
                            </div>
                        )}

                        <div>
                            <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Server Name</label>
                            <input 
                                type="text" value={sName} onChange={e => setSName(e.target.value)}
                                placeholder="e.g. github_integration"
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-primary"
                            />
                        </div>

                        <div>
                            <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Transport</label>
                            <div className="flex gap-4">
                                <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                                    <input type="radio" name="mcp_transport" value="stdio" checked={sTransport === "stdio"} onChange={() => setSTransport("stdio")} className="accent-primary" />
                                    Command Line (stdio)
                                </label>
                                <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                                    <input type="radio" name="mcp_transport" value="http" checked={sTransport === "http"} onChange={() => setSTransport("http")} className="accent-primary" />
                                    HTTP/SSE (url)
                                </label>
                            </div>
                        </div>

                        {sTransport === "stdio" ? (
                            <div className="flex gap-3">
                                <div className="w-1/3">
                                    <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Command</label>
                                    <input 
                                        type="text" value={sCmd} onChange={e => setSCmd(e.target.value)}
                                        placeholder="npx, uvx, docker, etc."
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-white placeholder-slate-600 focus:outline-none focus:border-primary"
                                    />
                                </div>
                                <div className="flex-1">
                                    <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Arguments (Space separated)</label>
                                    <input 
                                        type="text" value={sArgs} onChange={e => setSargs(e.target.value)}
                                        placeholder="-y @modelcontextprotocol/server-github"
                                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-white placeholder-slate-600 focus:outline-none focus:border-primary"
                                    />
                                </div>
                            </div>
                        ) : (
                            <div>
                                <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">SSE URL</label>
                                <input 
                                    type="text" value={sUrl} onChange={e => setSUrl(e.target.value)}
                                    placeholder="https://huggingface.co/mcp"
                                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-white placeholder-slate-600 focus:outline-none focus:border-primary"
                                />
                            </div>
                        )}

                        <div>
                            <label className="block text-xs font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Environment Variables</label>
                            <p className="text-[10px] text-slate-500 mb-2">One per line, format: <code>KEY=value</code>. These will be securely passed to the process.</p>
                            <textarea 
                                value={sEnvStr} onChange={e => setSEnvStr(e.target.value)}
                                placeholder="GITHUB_PERSONAL_ACCESS_TOKEN=ghp_..."
                                rows={3}
                                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-white placeholder-slate-600 focus:outline-none focus:border-primary leading-relaxed"
                            />
                        </div>

                        <div className="pt-2 flex justify-end">
                            <button 
                                onClick={handleSave} disabled={saving}
                                className="bg-primary hover:bg-primary-600 disabled:opacity-50 text-white text-sm font-semibold py-2 px-6 rounded-lg transition-colors flex items-center gap-2"
                            >
                                {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
                                Save & Connect
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// ── Admin Analytics Panel ─────────────────────────────────────────────────

// Provider Keys Panel

const PROVIDER_OPTIONS = [
    { id: "deepseek", label: "DeepSeek", quotaLabel: "500K/week included tokens" },
    { id: "openai", label: "OpenAI", quotaLabel: "100K/week included tokens" },
    { id: "anthropic", label: "Anthropic", quotaLabel: "BYOK only" },
    { id: "google", label: "Google Gemini", quotaLabel: "BYOK only" },
];

const PLATFORM_PROVIDER_LABELS: Record<string, string> = {
    deepseek: "DeepSeek",
    openai: "OpenAI",
    tacc: "TACC Tejas",
};

function ProviderKeysPanel() {
    const { isAuthenticated, token } = useAuthStore();
    const [keys, setKeys] = useState<ProviderKeyMeta[]>([]);
    const [quota, setQuota] = useState<UsageQuota | null>(null);
    const [apiKeys, setApiKeys] = useState<Record<string, string>>({});
    const [limits, setLimits] = useState<Record<string, string>>({});
    const [busy, setBusy] = useState<string | null>(null);
    const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

    const getKey = (provider: string) => keys.find(k => k.provider === provider);

    const fetchState = useCallback(async () => {
        if (!token) return;
        try {
            const [keysRes, quotaRes] = await Promise.all([
                fetch(`${API_BASE}/api/provider-keys`, { headers: { Authorization: `Bearer ${token}` } }),
                fetch(`${API_BASE}/api/usage-quota`, { headers: { Authorization: `Bearer ${token}` } }),
            ]);
            if (keysRes.ok) {
                const data = await keysRes.json();
                setKeys(data.keys || []);
                const nextLimits: Record<string, string> = {};
                (data.keys || []).forEach((k: ProviderKeyMeta) => {
                    nextLimits[k.provider] = k.token_limit ? String(k.token_limit) : "";
                });
                setLimits(nextLimits);
            }
            if (quotaRes.ok) setQuota(await quotaRes.json());
        } catch {
            setMessage({ type: "error", text: "Could not load provider key settings." });
        }
    }, [token]);

    useEffect(() => { if (isAuthenticated) fetchState(); }, [isAuthenticated, fetchState]);

    const saveKey = async (provider: string) => {
        if (!token) return;
        const rawKey = (apiKeys[provider] || "").trim();
        if (!rawKey) {
            setMessage({ type: "error", text: "Enter an API key before saving." });
            return;
        }
        const limitText = (limits[provider] || "").trim();
        setBusy(`${provider}:save`);
        setMessage(null);
        try {
            const res = await fetch(`${API_BASE}/api/provider-keys`, {
                method: "POST",
                headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
                body: JSON.stringify({
                    provider,
                    api_key: rawKey,
                    token_limit: limitText ? Number(limitText) : null,
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Save failed.");
            setApiKeys(prev => ({ ...prev, [provider]: "" }));
            setMessage({ type: "success", text: `${provider} key saved.` });
            fetchState();
        } catch (e) {
            setMessage({ type: "error", text: e instanceof Error ? e.message : "Save failed." });
        } finally {
            setBusy(null);
        }
    };

    const updateLimit = async (provider: string) => {
        if (!token) return;
        const limitText = (limits[provider] || "").trim();
        setBusy(`${provider}:limit`);
        setMessage(null);
        try {
            const res = await fetch(`${API_BASE}/api/provider-keys/${provider}/limit`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
                body: JSON.stringify({ token_limit: limitText ? Number(limitText) : null }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Limit update failed.");
            setMessage({ type: "success", text: limitText ? "Token limit updated." : "Token limit cleared." });
            fetchState();
        } catch (e) {
            setMessage({ type: "error", text: e instanceof Error ? e.message : "Limit update failed." });
        } finally {
            setBusy(null);
        }
    };

    const testKey = async (provider: string) => {
        if (!token) return;
        setBusy(`${provider}:test`);
        setMessage(null);
        try {
            const res = await fetch(`${API_BASE}/api/provider-keys/${provider}/test`, {
                method: "POST",
                headers: { Authorization: `Bearer ${token}` },
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Test failed.");
            setMessage({ type: "success", text: `${provider} key is valid.` });
            fetchState();
        } catch (e) {
            setMessage({ type: "error", text: e instanceof Error ? e.message : "Test failed." });
        } finally {
            setBusy(null);
        }
    };

    const deleteKey = async (provider: string) => {
        if (!token) return;
        if (!confirm(`Delete your ${provider} API key?`)) return;
        setBusy(`${provider}:delete`);
        setMessage(null);
        try {
            const res = await fetch(`${API_BASE}/api/provider-keys/${provider}`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` },
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || "Delete failed.");
            setMessage({ type: "success", text: `${provider} key deleted.` });
            fetchState();
        } catch (e) {
            setMessage({ type: "error", text: e instanceof Error ? e.message : "Delete failed." });
        } finally {
            setBusy(null);
        }
    };

    if (!isAuthenticated) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8 py-16">
                <div className="w-14 h-14 rounded-2xl glass-control flex items-center justify-center">
                    <Lock className="w-7 h-7 text-slate-500" />
                </div>
                <div>
                    <p className="text-sm font-semibold text-slate-300">Sign in to manage provider keys</p>
                    <p className="text-xs text-slate-500 mt-1.5 max-w-xs">Your keys are encrypted server-side and used before Quasar platform keys.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-5 overflow-y-auto h-full">
            <div>
                <h3 className="text-sm font-semibold text-white mb-1">Provider Keys</h3>
                <p className="text-xs text-slate-400">BYOK traffic uses your key first. No limit is applied unless you set one.</p>
            </div>

            {message && (
                <div className={`flex items-start gap-2 text-xs rounded-lg px-3 py-2 border ${message.type === "success" ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300" : "bg-red-500/10 border-red-500/30 text-red-300"}`}>
                    {message.type === "success" ? <CheckCircle className="w-4 h-4 shrink-0" /> : <AlertCircle className="w-4 h-4 shrink-0" />}
                    <span>{message.text}</span>
                </div>
            )}

            <div className="grid grid-cols-2 gap-3">
                {["deepseek", "openai", "tacc"].map(provider => {
                    const bucket = quota?.platform?.[provider];
                    const pct = bucket?.limit_tokens ? Math.min(100, Math.round((bucket.used_tokens / bucket.limit_tokens) * 100)) : 0;
                    return (
                        <div key={provider} className="glass-control rounded-xl px-4 py-3">
                            <div className="flex items-center justify-between text-xs mb-2">
                                <span className="font-semibold text-slate-300">{PLATFORM_PROVIDER_LABELS[provider] || provider} weekly quota</span>
                                <span className={bucket?.exhausted ? "text-red-300" : "text-slate-400"}>
                                    {bucket?.unlimited ? "Unlimited" : `${formatTokens(bucket?.used_tokens)} / ${formatTokens(bucket?.limit_tokens)}`}
                                </span>
                            </div>
                            {!bucket?.unlimited && (
                                <div className="h-1.5 rounded-full bg-slate-700 overflow-hidden">
                                    <div className={`h-full ${bucket?.exhausted ? "bg-red-400" : "bg-primary"}`} style={{ width: `${pct}%` }} />
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            <div className="space-y-3">
                {PROVIDER_OPTIONS.map(providerInfo => {
                    const meta = getKey(providerInfo.id);
                    const byokBucket = quota?.byok?.[providerInfo.id];
                    const isBusy = busy?.startsWith(`${providerInfo.id}:`);
                    return (
                        <div key={providerInfo.id} className="glass-control rounded-xl p-4 space-y-3">
                            <div className="flex items-start justify-between gap-3">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <h4 className="text-sm font-semibold text-white">{providerInfo.label}</h4>
                                        {meta ? (
                                            <span className="text-[10px] text-emerald-300 bg-emerald-500/10 border border-emerald-500/20 rounded px-1.5 py-0.5">Connected</span>
                                        ) : (
                                            <span className="text-[10px] text-slate-400 bg-slate-700/50 border border-slate-600/50 rounded px-1.5 py-0.5">Not connected</span>
                                        )}
                                    </div>
                                    <p className="text-xs text-slate-500 mt-1">
                                        {meta ? `Key ending ${meta.key_last4} - ${meta.status}${meta.last_tested_at ? ` - tested ${formatDate(meta.last_tested_at)}` : ""}` : providerInfo.quotaLabel}
                                    </p>
                                </div>
                                {meta && (
                                    <div className="flex items-center gap-2">
                                        <button onClick={() => testKey(providerInfo.id)} disabled={isBusy} className="text-xs px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 disabled:opacity-50">
                                            {busy === `${providerInfo.id}:test` ? "Testing..." : "Test"}
                                        </button>
                                        <button onClick={() => deleteKey(providerInfo.id)} disabled={isBusy} className="p-1.5 rounded-lg text-red-300 hover:bg-red-500/10 disabled:opacity-50" title="Delete key">
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                )}
                            </div>

                            <div className="grid grid-cols-[1fr_150px_auto] gap-2">
                                <input
                                    type="password"
                                    value={apiKeys[providerInfo.id] || ""}
                                    onChange={e => setApiKeys(prev => ({ ...prev, [providerInfo.id]: e.target.value }))}
                                    placeholder={meta ? "Rotate key" : "Paste API key"}
                                    className="bg-slate-900/70 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary"
                                />
                                <input
                                    type="number"
                                    min="1"
                                    value={limits[providerInfo.id] || ""}
                                    onChange={e => setLimits(prev => ({ ...prev, [providerInfo.id]: e.target.value }))}
                                    placeholder="No limit"
                                    className="bg-slate-900/70 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 outline-none focus:border-primary"
                                />
                                <button onClick={() => saveKey(providerInfo.id)} disabled={isBusy} className="px-3 py-2 rounded-lg bg-primary hover:bg-primary-600 text-white text-sm font-semibold disabled:opacity-50">
                                    {busy === `${providerInfo.id}:save` ? "Saving..." : meta ? "Rotate" : "Save"}
                                </button>
                            </div>

                            {meta && (
                                <div className="flex items-center justify-between text-xs text-slate-500">
                                    <span>BYOK used: {formatTokens(byokBucket?.used_tokens)}{byokBucket?.limit_tokens ? ` / ${formatTokens(byokBucket.limit_tokens)}` : " - no limit"}</span>
                                    <button onClick={() => updateLimit(providerInfo.id)} disabled={isBusy} className="text-primary hover:text-white disabled:opacity-50">
                                        {busy === `${providerInfo.id}:limit` ? "Updating..." : "Update limit"}
                                    </button>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

interface AnalyticsSummary {
    total_chats: number;
    unique_users: number;
    anonymous_chats: number;
    top_models: { model: string; count: number }[];
    top_tools: { tool: string; count: number }[];
    total_page_views: number;
    unique_visitors: number;
    feedback?: { likes: number; dislikes: number; total: number };
}

interface AdminIssueReport {
    id: string;
    created_at: string;
    status: "new" | "investigating" | "resolved" | "dismissed";
    category: string;
    description: string;
    provider: string;
    model: string;
    trace_id?: string;
    run_id: string;
    admin_notes?: string;
    technical_context?: {
        run_status?: string;
        duration_ms?: number;
        tools_called?: string[];
        first_token_ms?: number | null;
        provider_chunk_count?: number;
    };
}

function IssueReportsPanel({ token }: { token: string }) {
    const [reports, setReports] = useState<AdminIssueReport[]>([]);
    const [loading, setLoading] = useState(true);
    const [filters, setFilters] = useState({
        status: "",
        category: "",
        provider: "",
        model: "",
        date_from: "",
        date_to: "",
    });
    const [notes, setNotes] = useState<Record<string, string>>({});

    const loadReports = useCallback(async () => {
        setLoading(true);
        const params = new URLSearchParams(
            Object.entries(filters).filter(([, value]) => value) as [string, string][],
        );
        try {
            const response = await fetch(`${API_BASE}/api/admin/issue-reports?${params}`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            const rows = (payload.reports || []) as AdminIssueReport[];
            setReports(rows);
            setNotes(Object.fromEntries(rows.map((report) => [report.id, report.admin_notes || ""])));
        } finally {
            setLoading(false);
        }
    }, [filters, token]);

    useEffect(() => {
        loadReports().catch(() => setReports([]));
    }, [loadReports]);

    const updateReport = async (report: AdminIssueReport, status = report.status) => {
        const response = await fetch(`${API_BASE}/api/admin/issue-reports/${report.id}`, {
            method: "PATCH",
            headers: {
                Authorization: `Bearer ${token}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ status, admin_notes: notes[report.id] || "" }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        await loadReports();
    };

    const exportReports = async (format: "csv" | "json") => {
        const params = new URLSearchParams({
            ...Object.fromEntries(Object.entries(filters).filter(([, value]) => value)),
            format,
        });
        const response = await fetch(`${API_BASE}/api/admin/issue-reports/export?${params}`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        const anchor = document.createElement("a");
        anchor.href = URL.createObjectURL(blob);
        anchor.download = `quasar_issue_reports_${new Date().toISOString().slice(0, 10)}.${format}`;
        anchor.click();
        URL.revokeObjectURL(anchor.href);
    };

    const setFilter = (key: keyof typeof filters, value: string) => {
        setFilters((current) => ({ ...current, [key]: value }));
    };

    return (
        <div className="space-y-3">
            <div className="flex items-center justify-between">
                <div>
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Private Issue Reports</h4>
                    <p className="mt-1 text-[10px] text-slate-500">Structured reports linked to model runs and traces.</p>
                </div>
                <div className="flex gap-2">
                    <button onClick={() => exportReports("csv")} className="glass-control rounded-lg px-2.5 py-1.5 text-[10px] text-slate-300">CSV</button>
                    <button onClick={() => exportReports("json")} className="glass-control rounded-lg px-2.5 py-1.5 text-[10px] text-slate-300">JSON</button>
                </div>
            </div>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
                <select value={filters.status} onChange={(event) => setFilter("status", event.target.value)} className="glass-control rounded-lg px-2 py-2 text-xs">
                    <option value="">All statuses</option>
                    <option value="new">New</option>
                    <option value="investigating">Investigating</option>
                    <option value="resolved">Resolved</option>
                    <option value="dismissed">Dismissed</option>
                </select>
                <select value={filters.category} onChange={(event) => setFilter("category", event.target.value)} className="glass-control rounded-lg px-2 py-2 text-xs">
                    <option value="">All categories</option>
                    <option value="stuck_slow">Stuck or slow</option>
                    <option value="wrong_answer">Wrong answer</option>
                    <option value="incorrect_data">Incorrect data</option>
                    <option value="ui_problem">Interface problem</option>
                    <option value="other">Other</option>
                </select>
                <input value={filters.provider} onChange={(event) => setFilter("provider", event.target.value)} placeholder="Provider" className="glass-control rounded-lg px-2 py-2 text-xs" />
                <input value={filters.model} onChange={(event) => setFilter("model", event.target.value)} placeholder="Exact model" className="glass-control rounded-lg px-2 py-2 text-xs" />
                <input type="date" value={filters.date_from} onChange={(event) => setFilter("date_from", event.target.value)} className="glass-control rounded-lg px-2 py-2 text-xs" />
                <input type="date" value={filters.date_to} onChange={(event) => setFilter("date_to", event.target.value)} className="glass-control rounded-lg px-2 py-2 text-xs" />
            </div>
            {loading ? (
                <div className="flex items-center gap-2 py-4 text-xs text-slate-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading reports...</div>
            ) : reports.length === 0 ? (
                <div className="rounded-xl border border-slate-800 p-4 text-xs text-slate-500">No reports match these filters.</div>
            ) : (
                <div className="space-y-2">
                    {reports.map((report) => (
                        <div key={report.id} className="rounded-xl border border-slate-800 bg-slate-950/50 p-3">
                            <div className="flex flex-wrap items-center gap-2 text-[10px] text-slate-500">
                                <span className="rounded bg-red-500/10 px-2 py-0.5 text-red-300">{report.category.replace("_", " ")}</span>
                                <span>{report.provider} / {report.model}</span>
                                <span>{new Date(report.created_at).toLocaleString()}</span>
                                <span>run {report.run_id.slice(0, 8)}</span>
                            </div>
                            <p className="mt-2 text-xs leading-5 text-slate-200">{report.description}</p>
                            <div className="mt-2 text-[10px] text-slate-500">
                                Run: {report.technical_context?.run_status || "unknown"}
                                {" · "}First token: {report.technical_context?.first_token_ms ?? "—"} ms
                                {" · "}Chunks: {report.technical_context?.provider_chunk_count ?? "—"}
                                {" · "}Tools: {(report.technical_context?.tools_called || []).join(", ") || "none"}
                            </div>
                            <div className="mt-3 grid gap-2 md:grid-cols-[160px_1fr_auto]">
                                <select value={report.status} onChange={(event) => updateReport(report, event.target.value as AdminIssueReport["status"])}
                                    className="glass-control rounded-lg px-2 py-2 text-xs">
                                    <option value="new">New</option>
                                    <option value="investigating">Investigating</option>
                                    <option value="resolved">Resolved</option>
                                    <option value="dismissed">Dismissed</option>
                                </select>
                                <input value={notes[report.id] || ""} onChange={(event) => setNotes((current) => ({ ...current, [report.id]: event.target.value }))}
                                    placeholder="Admin notes" className="glass-control rounded-lg px-2 py-2 text-xs" />
                                <button onClick={() => updateReport(report)} className="glass-control rounded-lg px-3 py-2 text-xs text-slate-200">Save</button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function AnalyticsPanel() {
    const { token } = useAuthStore();
    const [data, setData] = useState<AnalyticsSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [downloading, setDownloading] = useState<string | null>(null);

    const downloadFile = async (url: string, filename: string, key: string) => {
        if (!token) return;
        setDownloading(key);
        try {
            const res = await fetch(`${API_BASE}${url}`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const blob = await res.blob();
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(a.href);
        } catch (e: unknown) {
            alert(`Download failed: ${e instanceof Error ? e.message : String(e)}`);
        }
        setDownloading(null);
    };

    useEffect(() => {
        if (!token) return;
        setLoading(true);
        fetch(`${API_BASE}/api/admin/analytics/summary`, {
            headers: { Authorization: `Bearer ${token}` },
        })
            .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
            .then(d => { setData(d); setError(null); })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false));
    }, [token]);

    if (loading) return <div className="flex items-center justify-center h-full gap-2 text-slate-400 text-sm"><Loader2 className="w-4 h-4 animate-spin" /> Loading analytics...</div>;
    if (error) return <div className="flex items-center justify-center h-full text-red-400 text-sm">Error: {error}</div>;
    if (!data) return null;

    const statCards = [
        { label: "Total Chats", value: data.total_chats, color: "text-blue-400" },
        { label: "Unique Users", value: data.unique_users, color: "text-emerald-400" },
        { label: "Anonymous", value: data.anonymous_chats, color: "text-slate-400" },
        { label: "Page Views", value: data.total_page_views, color: "text-purple-400" },
        { label: "Unique Visitors", value: data.unique_visitors, color: "text-cyan-400" },
    ];

    return (
        <div className="p-6 space-y-6 overflow-y-auto h-full">
            <div>
                <h3 className="text-sm font-semibold text-white mb-1">Platform Analytics</h3>
                <p className="text-xs text-slate-400">Usage metrics stored in Turso cloud DB (persistent across deploys).</p>
            </div>

            {/* Stat cards */}
            <div className="grid grid-cols-3 gap-3">
                {statCards.map(s => (
                    <div key={s.label} className="glass-control rounded-xl px-4 py-3">
                        <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">{s.label}</p>
                        <p className={`text-2xl font-bold ${s.color} mt-1`}>{s.value.toLocaleString()}</p>
                    </div>
                ))}
            </div>

            {/* Feedback */}
            {data.feedback && data.feedback.total > 0 && (
                <div>
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Response Feedback</h4>
                    <div className="flex gap-3">
                        <div className="flex-1 bg-emerald-500/10 border border-emerald-500/20 rounded-xl px-4 py-3 text-center">
                            <p className="text-2xl font-bold text-emerald-400">👍 {data.feedback.likes}</p>
                            <p className="text-[10px] text-emerald-400/60 mt-1">Likes</p>
                        </div>
                        <div className="flex-1 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 text-center">
                            <p className="text-2xl font-bold text-red-400">👎 {data.feedback.dislikes}</p>
                            <p className="text-[10px] text-red-400/60 mt-1">Dislikes</p>
                        </div>
                    </div>
                </div>
            )}

            {/* Top models */}
            {data.top_models.length > 0 && (
                <div>
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Top Models</h4>
                    <div className="space-y-1.5">
                        {data.top_models.map((m, i) => (
                            <div key={m.model} className="glass-control flex items-center gap-3 rounded-lg px-3 py-2">
                                <span className="text-xs font-bold text-slate-500 w-5">{i + 1}</span>
                                <span className="text-sm text-slate-200 flex-1 font-mono">{m.model}</span>
                                <span className="text-xs text-slate-400 font-semibold">{m.count} calls</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Top tools */}
            {data.top_tools.length > 0 && (
                <div>
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Top Tools</h4>
                    <div className="space-y-1.5">
                        {data.top_tools.map((t, i) => (
                            <div key={t.tool} className="glass-control flex items-center gap-3 rounded-lg px-3 py-2">
                                <span className="text-xs font-bold text-slate-500 w-5">{i + 1}</span>
                                <span className="text-sm text-slate-200 flex-1 font-mono">{t.tool}</span>
                                <span className="text-xs text-slate-400 font-semibold">{t.count}×</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {token && <IssueReportsPanel token={token} />}

            {/* Export / Download */}
            <div>
                <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Export Data</h4>
                <div className="grid grid-cols-2 gap-2">
                    <button
                        onClick={() => downloadFile("/api/admin/analytics/export?format=csv", `quasar_analytics_${new Date().toISOString().slice(0,10)}.csv`, "csv")}
                        disabled={downloading === "csv"}
                        className="glass-control flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm text-slate-200 font-medium disabled:opacity-50"
                    >
                        {downloading === "csv" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                        Chat Analytics (CSV)
                    </button>
                    <button
                        onClick={() => downloadFile("/api/admin/analytics/export?format=json", `quasar_analytics_${new Date().toISOString().slice(0,10)}.json`, "json")}
                        disabled={downloading === "json"}
                        className="glass-control flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm text-slate-200 font-medium disabled:opacity-50"
                    >
                        {downloading === "json" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                        Chat Analytics (JSON)
                    </button>
                    <button
                        onClick={() => downloadFile("/api/admin/feedback/export", `quasar_feedback_${new Date().toISOString().slice(0,10)}.json`, "feedback")}
                        disabled={downloading === "feedback"}
                        className="glass-control flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm text-slate-200 font-medium disabled:opacity-50"
                    >
                        {downloading === "feedback" ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                        Feedback Data (JSON)
                    </button>
                </div>
                <p className="text-[10px] text-slate-500 mt-2">Includes: user ID, email, prompt, response, model used, tools called, timestamps, response time.</p>
            </div>
        </div>
    );
}

// ── Theme Toggle Switch ───────────────────────────────────────────────────

function ThemeSwitch() {
    const { theme, toggle } = useThemeStore();
    const isDark = theme === "dark";
    return (
        <button
            onClick={toggle}
            className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent transition-colors"
            title={`Switch to ${isDark ? "light" : "dark"} mode`}
        >
            <div className="flex items-center gap-2.5">
                {isDark ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
                <span>{isDark ? "Dark Mode" : "Light Mode"}</span>
            </div>
            {/* Toggle pill */}
            <div className={`relative w-9 h-5 rounded-full transition-colors ${isDark ? "bg-slate-600" : "bg-primary/40"}`}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full shadow-sm transition-all duration-200 ${isDark ? "left-0.5 bg-slate-300" : "left-[18px] bg-primary"}`} />
            </div>
        </button>
    );
}

// ── Main Settings Modal ───────────────────────────────────────────────────

interface SettingsModalProps {
    open: boolean;
    onClose: () => void;
}

type TabType = 'personalization' | 'providerKeys' | 'tools' | 'mcp' | 'analytics';

export function SettingsModal({ open, onClose }: SettingsModalProps) {
    const backdropRef = useRef<HTMLDivElement>(null);
    const [currentTab, setCurrentTab] = useState<TabType>('personalization');
    const { user } = useAuthStore();
    const isAdmin = Boolean(user?.is_admin);

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
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-lg"
            onClick={(e) => { if (e.target === backdropRef.current) onClose(); }}
        >
            <div className="relative w-[850px] max-w-[95vw] h-[650px] max-h-[90vh] glass-surface rounded-2xl flex overflow-hidden animate-in fade-in zoom-in-95 duration-200">

                {/* Close button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 z-10 p-1.5 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-all"
                >
                    <X className="w-4 h-4" />
                </button>

                {/* Left nav */}
                <div className="w-48 shrink-0 border-r border-slate-700/50 p-4 glass-sidebar">
                    <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold px-2 mb-3">Settings</p>
                    
                    <div className="space-y-1">
                        <button 
                            onClick={() => setCurrentTab('personalization')}
                            className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors \${currentTab === 'personalization' ? "bg-primary/10 text-primary border border-primary/20" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent"}`}
                        >
                            <FileText className="w-4 h-4" />
                            Personalization
                        </button>

                        <button
                            onClick={() => setCurrentTab('providerKeys')}
                            className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${currentTab === 'providerKeys' ? "bg-primary/10 text-primary border border-primary/20" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent"}`}
                        >
                            <Lock className="w-4 h-4" />
                            Provider Keys
                        </button>
                        
                        <button 
                            onClick={() => setCurrentTab('tools')}
                            className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors \${currentTab === 'tools' ? "bg-primary/10 text-primary border border-primary/20" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent"}`}
                        >
                            <Wrench className="w-4 h-4" />
                            Custom Tools
                        </button>

                        <button 
                            onClick={() => setCurrentTab('mcp')}
                            className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors \${currentTab === 'mcp' ? "bg-primary/10 text-primary border border-primary/20" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent"}`}
                        >
                            <span className="text-lg leading-none">🔌</span>
                            MCP Servers
                        </button>

                        {isAdmin && (
                            <button 
                                onClick={() => setCurrentTab('analytics')}
                                className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${currentTab === 'analytics' ? "bg-amber-500/10 text-amber-400 border border-amber-500/20" : "text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent"}`}
                            >
                                <BarChart3 className="w-4 h-4" />
                                Analytics
                            </button>
                        )}
                    </div>

                    {/* Bottom utilities */}
                    <div className="mt-auto pt-4 border-t border-slate-700/50 mt-6 space-y-2">
                        {/* Theme toggle */}
                        <ThemeSwitch />

                        {/* Restart Tutorial */}
                        <button
                            onClick={() => { resetOnboarding(); onClose(); window.location.reload(); }}
                            className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent transition-colors"
                        >
                            <Sparkles className="w-4 h-4" />
                            Restart Tutorial
                        </button>

                        {/* Terms & Conditions */}
                        <a
                            href="/terms"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent transition-colors"
                        >
                            <div className="flex items-center gap-2.5">
                                <ScrollText className="w-4 h-4" />
                                <span>Terms & Privacy</span>
                            </div>
                            <ExternalLink className="w-3 h-3 text-slate-600" />
                        </a>

                        <a
                            href="https://github.com/adamzacharia/Quasar"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent transition-colors"
                        >
                            <div className="flex items-center gap-2.5 min-w-0">
                                <Github className="w-4 h-4 shrink-0" />
                                <span className="truncate">GitHub Repository</span>
                            </div>
                            <ExternalLink className="w-3 h-3 text-slate-600 shrink-0" />
                        </a>
                    </div>
                </div>

                {/* Right content */}
                <div className="flex-1 overflow-hidden flex flex-col">
                    <div className="px-6 py-5 border-b border-slate-700/50 shrink-0">
                        <h2 className="text-base font-semibold text-white">
                            {currentTab === 'personalization' ? "Personalization" : currentTab === 'providerKeys' ? "Provider Keys" : currentTab === 'tools' ? "Custom Tools" : currentTab === 'analytics' ? "Analytics" : "MCP Servers"}
                        </h2>
                        <p className="text-xs text-slate-400 mt-0.5">
                            {currentTab === 'personalization' 
                                ? "Your private knowledge base for smarter conversations" 
                                : currentTab === 'providerKeys'
                                ? "Bring your own provider API keys and view included quota"
                                : currentTab === 'tools'
                                ? "Extend Quasar with your own Python agent tools"
                                : currentTab === 'analytics'
                                ? "Platform usage metrics and feedback data"
                                : "Connect standard Model Context Protocol servers"}
                        </p>
                    </div>
                    
                    <div className="flex-1 overflow-hidden">
                        {currentTab === 'personalization' && <PersonalizationPanel />}
                        {currentTab === 'providerKeys' && <ProviderKeysPanel />}
                        {currentTab === 'tools' && <CustomToolsPanel />}
                        {currentTab === 'mcp' && <MCPServersPanel />}
                        {currentTab === 'analytics' && <AnalyticsPanel />}
                    </div>
                </div>
            </div>
        </div>
    );
}
