"use client";

// Settings > Memory: everything Quasar remembers about this user across chats
// (services/user_memory_service.py via /api/memory). View, edit, forget each
// item, pause memory, export it, or delete all of it.

import { useCallback, useEffect, useMemo, useState } from "react";
import { Brain, Download, Loader2, Lock, Pause, Play, Pencil, Trash2, Check, X, Undo2 } from "lucide-react";
import { useAuthStore, authBearerHeaders } from "../lib/auth-store";
import type { MemoryUpdateEvent } from "../lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface SlotSchema {
    slot: string;
    label: string;
    kind: "enum" | "list" | "text";
    group: string;
    choices: { value: string; label: string }[] | null;
    max_items?: number | null;
}

interface MemoryItem {
    id: string;
    slot: string;
    label: string;
    value: string | string[];
    display: string;
    source: "manual" | "chat_explicit" | "chat_inferred";
    valid_from: string;
}

interface MemoryEvent {
    id: string;
    slot: string;
    op: string;
    source: string;
    created_at: string;
    undone: boolean;
}

interface MemoryState {
    paused: boolean;
    items: MemoryItem[];
    schema: SlotSchema[];
    recent: MemoryEvent[];
}

const SOURCE_LABEL: Record<string, string> = {
    manual: "set by you",
    chat_explicit: "from chat (you asked)",
    chat_inferred: "learned from chat",
};

async function api(path: string, init?: RequestInit) {
    const res = await fetch(`${API_BASE}${path}`, {
        credentials: "include",
        ...init,
        headers: { ...(init?.body ? { "Content-Type": "application/json" } : {}), ...authBearerHeaders(), ...(init?.headers || {}) },
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
    return data;
}

function SlotEditor({ schema, item, onSave, onCancel }: {
    schema: SlotSchema;
    item?: MemoryItem;
    onSave: (value: string | string[]) => void;
    onCancel: () => void;
}) {
    const initial = item ? (Array.isArray(item.value) ? item.value.join(", ") : String(item.value)) : "";
    const [draft, setDraft] = useState(initial);
    const submit = () => {
        if (schema.kind === "list") onSave(draft.split(",").map(s => s.trim()).filter(Boolean));
        else onSave(draft.trim());
    };
    return (
        <div className="flex items-center gap-2 mt-2">
            {schema.kind === "enum" && schema.choices ? (
                <select
                    value={draft}
                    onChange={e => setDraft(e.target.value)}
                    className="flex-1 rounded-lg px-2 py-1.5 text-sm bg-slate-900 border border-slate-700 text-slate-200"
                    aria-label={schema.label}
                >
                    <option value="" disabled>Choose…</option>
                    {schema.choices.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
                </select>
            ) : (
                <input
                    value={draft}
                    onChange={e => setDraft(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") submit(); if (e.key === "Escape") onCancel(); }}
                    placeholder={schema.kind === "list" ? "Comma-separated" : ""}
                    className="flex-1 rounded-lg px-2 py-1.5 text-sm bg-slate-900 border border-slate-700 text-slate-200"
                    aria-label={schema.label}
                    autoFocus
                />
            )}
            <button onClick={submit} disabled={!draft.trim()} title="Save" aria-label={`Save ${schema.label}`}
                className="p-1.5 rounded-lg text-emerald-400 hover:bg-emerald-400/10 disabled:opacity-40">
                <Check className="w-4 h-4" />
            </button>
            <button onClick={onCancel} title="Cancel" aria-label="Cancel" className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-700/40">
                <X className="w-4 h-4" />
            </button>
        </div>
    );
}

export function MemoryPanel() {
    const { isAuthenticated } = useAuthStore();
    const [state, setState] = useState<MemoryState | null>(null);
    const [loading, setLoading] = useState(false);
    const [editing, setEditing] = useState<string | null>(null);
    const [msg, setMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
    const [showAll, setShowAll] = useState(false);

    const load = useCallback(async () => {
        if (!isAuthenticated) return;
        setLoading(true);
        try {
            setState(await api("/api/memory"));
        } catch (e) {
            setMsg({ type: "error", text: (e as Error).message });
        }
        setLoading(false);
    }, [isAuthenticated]);

    // eslint-disable-next-line react-hooks/set-state-in-effect
    useEffect(() => { load(); }, [load]);

    const run = async (fn: () => Promise<unknown>, ok?: string) => {
        try {
            await fn();
            if (ok) setMsg({ type: "success", text: ok });
            await load();
        } catch (e) {
            setMsg({ type: "error", text: (e as Error).message });
        }
    };

    const itemsBySlot = useMemo(() => {
        const m: Record<string, MemoryItem> = {};
        (state?.items || []).forEach(it => { m[it.slot] = it; });
        return m;
    }, [state]);

    const groups = useMemo(() => {
        const out: Record<string, SlotSchema[]> = {};
        (state?.schema || []).forEach(s => {
            if (!showAll && !itemsBySlot[s.slot] && editing !== s.slot) return;
            (out[s.group] = out[s.group] || []).push(s);
        });
        return out;
    }, [state, showAll, itemsBySlot, editing]);

    if (!isAuthenticated) {
        return (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8 py-16">
                <Lock className="w-7 h-7 text-slate-500" />
                <p className="text-sm text-slate-400">Sign in to use Memory.</p>
            </div>
        );
    }

    const exportJson = async () => {
        try {
            const data = await api("/api/memory/export");
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `quasar_memory_${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            setMsg({ type: "error", text: (e as Error).message });
        }
    };

    const saved = state?.items.length || 0;

    return (
        <div className="p-6 space-y-5 overflow-y-auto h-full" data-testid="memory-panel">
            <div>
                <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2"><Brain className="w-4 h-4 text-primary" />What Quasar remembers about you</h3>
                <p className="text-xs text-slate-400">
                    Quasar saves your preferences and research context from chat (units, conventions, archives, citation style,
                    expertise, current targets) and uses them in every new chat. You will see a &ldquo;Memory updated&rdquo; note
                    under an answer when something is saved. Uploaded documents never change your memory.
                </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
                <button
                    onClick={() => run(() => api("/api/memory/pause", { method: "POST", body: JSON.stringify({ paused: !state?.paused }) }),
                        state?.paused ? "Memory is on again." : "Memory paused: nothing is read or saved until you resume.")}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-slate-700 text-slate-300 hover:bg-slate-800"
                    data-testid="memory-pause-toggle"
                >
                    {state?.paused ? <><Play className="w-3.5 h-3.5" />Resume memory</> : <><Pause className="w-3.5 h-3.5" />Pause memory</>}
                </button>
                <button onClick={exportJson}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-slate-700 text-slate-300 hover:bg-slate-800">
                    <Download className="w-3.5 h-3.5" />Export
                </button>
                <button
                    onClick={() => { if (confirm("Delete everything Quasar remembers about you? This cannot be undone.")) run(() => api("/api/memory", { method: "DELETE" }), "All memory deleted."); }}
                    disabled={!saved}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-red-500/30 text-red-400 hover:bg-red-500/10 disabled:opacity-40"
                >
                    <Trash2 className="w-3.5 h-3.5" />Delete all
                </button>
                <label className="ml-auto inline-flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
                    <input type="checkbox" checked={showAll} onChange={e => setShowAll(e.target.checked)} />
                    Show all fields
                </label>
            </div>

            {state?.paused && (
                <div className="px-4 py-3 rounded-xl text-sm bg-amber-500/10 text-amber-300 border border-amber-500/20">
                    Memory is paused. Quasar is not using or saving anything until you resume.
                </div>
            )}

            {msg && (
                <div className={`px-4 py-3 rounded-xl text-sm ${msg.type === "success" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-red-500/10 text-red-400 border border-red-500/20"}`}>
                    {msg.text}
                </div>
            )}

            {loading && !state ? (
                <div className="flex justify-center py-8"><Loader2 className="w-5 h-5 animate-spin text-slate-500" /></div>
            ) : saved === 0 && !showAll ? (
                <div className="text-center py-8 text-xs text-slate-500">
                    Nothing saved yet. Tell Quasar something like &ldquo;remember I report line sensitivities in K&rdquo;, or tick
                    &ldquo;Show all fields&rdquo; to fill them in yourself.
                </div>
            ) : (
                Object.entries(groups).map(([group, slots]) => (
                    <div key={group}>
                        <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">{group}</p>
                        <div className="space-y-2">
                            {slots.map(s => {
                                const it = itemsBySlot[s.slot];
                                return (
                                    <div key={s.slot} className="glass-control px-4 py-3 rounded-xl group" data-testid={`memory-slot-${s.slot}`}>
                                        <div className="flex items-start gap-3">
                                            <div className="flex-1 min-w-0">
                                                <p className="text-[11px] text-slate-500">{s.label}</p>
                                                {it ? (
                                                    <>
                                                        <p className="text-sm text-slate-200 break-words">{it.display}</p>
                                                        <p className="text-[10px] text-slate-500">{SOURCE_LABEL[it.source] || it.source} · {String(it.valid_from || "").slice(0, 10)}</p>
                                                    </>
                                                ) : (
                                                    <p className="text-sm text-slate-500 italic">Not set</p>
                                                )}
                                            </div>
                                            <button onClick={() => setEditing(s.slot)} title="Edit" aria-label={`Edit ${s.label}`}
                                                className="p-1.5 rounded-lg text-slate-500 hover:text-slate-200 hover:bg-slate-700/40">
                                                <Pencil className="w-4 h-4" />
                                            </button>
                                            {it && (
                                                <button
                                                    onClick={() => run(() => api(`/api/memory/slot/${encodeURIComponent(s.slot)}`, { method: "DELETE" }), `Forgot ${s.label.toLowerCase()}.`)}
                                                    title="Forget" aria-label={`Forget ${s.label}`}
                                                    className="p-1.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-400/10">
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            )}
                                        </div>
                                        {editing === s.slot && (
                                            <SlotEditor
                                                schema={s}
                                                item={it}
                                                onCancel={() => setEditing(null)}
                                                onSave={value => {
                                                    setEditing(null);
                                                    run(() => api(`/api/memory/slot/${encodeURIComponent(s.slot)}`, { method: "PUT", body: JSON.stringify({ value }) }), `Saved ${s.label.toLowerCase()}.`);
                                                }}
                                            />
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ))
            )}
        </div>
    );
}

/** The "Memory updated" note under an answer (one per turn that saved something). */
export function MemoryUpdateChip({ events }: { events: MemoryUpdateEvent[] }) {
    const [undone, setUndone] = useState<Record<string, boolean>>({});
    const [error, setError] = useState<string | null>(null);
    if (!events || events.length === 0) return null;
    const undo = async (ev: MemoryUpdateEvent) => {
        try {
            await api(`/api/memory/events/${encodeURIComponent(ev.id)}/undo`, { method: "POST" });
            setUndone(u => ({ ...u, [ev.id]: true }));
        } catch (e) {
            setError((e as Error).message);
        }
    };
    return (
        <div data-testid="memory-update-chip" className="mt-2 inline-flex flex-col gap-1 rounded-xl px-3 py-1.5 text-[11px]"
            style={{ color: "var(--q-text-muted)", border: "1px solid var(--q-glass-border)" }}>
            {events.map(ev => (
                <span key={ev.id} className="inline-flex items-center gap-1.5">
                    <Brain className="h-3 w-3" />
                    {undone[ev.id] ? (
                        <span>Undone: {ev.label.toLowerCase()} {ev.previous ? `back to ${ev.previous}` : "not saved"}</span>
                    ) : (
                        <>
                            <span>
                                Memory updated: {ev.label.toLowerCase()} {ev.op === "clear" ? "forgotten" : <>&rarr; <strong>{ev.display}</strong></>}
                            </span>
                            <button onClick={() => undo(ev)} className="inline-flex items-center gap-0.5 underline hover:no-underline" aria-label={`Undo memory change to ${ev.label}`}>
                                <Undo2 className="h-3 w-3" />Undo
                            </button>
                        </>
                    )}
                </span>
            ))}
            {error && <span className="text-red-400">{error}</span>}
        </div>
    );
}
