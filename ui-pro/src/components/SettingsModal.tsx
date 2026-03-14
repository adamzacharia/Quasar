"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { X, Upload, FileText, Trash2, Lock, Loader2, CheckCircle, AlertCircle, Wrench, Plus, Copy } from "lucide-react";
import { useAuthStore } from "../lib/auth-store";

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

// ── Utilities ─────────────────────────────────────────────────────────────

function formatBytes(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string) {
    return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
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
                <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center">
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
                    🔧 Installed ({tools.length})
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
                    📋 Templates
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
                                    <div key={tool.name} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4 group flex items-start justify-between">
                                        <div className="min-w-0 flex-1 pr-4">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="text-sm font-semibold text-slate-200 truncate">{tool.name}</h4>
                                                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 bg-slate-700/50 px-2 py-0.5 rounded">Custom</span>
                                            </div>
                                            <p className="text-xs text-slate-400 line-clamp-2">{tool.description || <span className="italic">No description</span>}</p>
                                            {tool.api_key_name && (
                                                <div className="mt-2 text-[10px] font-mono text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded inline-block">
                                                    🔑 {tool.api_key_name}
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

                        <div className="p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl space-y-3">
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
                            <strong>How tools work:</strong> Write a standard Python function. The AI sees your function's name, docstring, and parameter types to auto-generate the JSON schema. Must return a <code>dict</code> or <code>str</code>.
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
    const [servers, setServers] = useState<any[]>([]);
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
                <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center">
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
                    🔌 Connected ({servers.length})
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
                                    <div key={srv.name} className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4 group flex items-start justify-between">
                                        <div className="min-w-0 flex-1 pr-4">
                                            <div className="flex items-center gap-2 mb-1">
                                                <h4 className="text-sm font-semibold text-slate-200 truncate">{srv.name}</h4>
                                                <span className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded ${srv.transport === 'http' ? 'text-purple-400 bg-purple-400/10' : 'text-blue-400 bg-blue-400/10'}`}>MCP {srv.transport || 'stdio'}</span>
                                            </div>
                                            {srv.transport === "http" ? (
                                                <p className="text-xs text-slate-400 font-mono mt-2 truncate bg-black/30 p-1.5 rounded border border-slate-700/50">
                                                    🔗 {srv.url}
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

// ── Main Settings Modal ───────────────────────────────────────────────────

interface SettingsModalProps {
    open: boolean;
    onClose: () => void;
}

type TabType = 'personalization' | 'tools' | 'mcp';

export function SettingsModal({ open, onClose }: SettingsModalProps) {
    const backdropRef = useRef<HTMLDivElement>(null);
    const [currentTab, setCurrentTab] = useState<TabType>('personalization');

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
            <div className="relative w-[850px] max-w-[95vw] h-[650px] max-h-[90vh] bg-[#0f1117] border border-slate-700/60 rounded-2xl shadow-2xl shadow-black/60 flex overflow-hidden animate-in fade-in zoom-in-95 duration-200">

                {/* Close button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 z-10 p-1.5 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-all"
                >
                    <X className="w-4 h-4" />
                </button>

                {/* Left nav */}
                <div className="w-48 shrink-0 border-r border-slate-700/50 p-4 bg-[#0c0d12]">
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
                    </div>
                </div>

                {/* Right content */}
                <div className="flex-1 overflow-hidden flex flex-col bg-[#0f1117]">
                    <div className="px-6 py-5 border-b border-slate-700/50 shrink-0">
                        <h2 className="text-base font-semibold text-white">
                            {currentTab === 'personalization' ? "Personalization" : currentTab === 'tools' ? "Custom Tools" : "MCP Servers"}
                        </h2>
                        <p className="text-xs text-slate-400 mt-0.5">
                            {currentTab === 'personalization' 
                                ? "Your private knowledge base for smarter conversations" 
                                : currentTab === 'tools'
                                ? "Extend Quasar with your own Python agent tools"
                                : "Connect standard Model Context Protocol servers"}
                        </p>
                    </div>
                    
                    <div className="flex-1 overflow-hidden">
                        {currentTab === 'personalization' && <PersonalizationPanel />}
                        {currentTab === 'tools' && <CustomToolsPanel />}
                        {currentTab === 'mcp' && <MCPServersPanel />}
                    </div>
                </div>
            </div>
        </div>
    );
}
