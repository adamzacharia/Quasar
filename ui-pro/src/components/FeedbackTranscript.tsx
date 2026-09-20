"use client";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { authBearerHeaders } from "../lib/auth-store";
import { FeedbackTranscriptView, type FeedbackSnapshot } from "./FeedbackTranscriptView";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function FeedbackTranscript({ snapshotId, error }: { snapshotId?: string; error?: string }) {
    const [open, setOpen] = useState(false);
    const [snapshot, setSnapshot] = useState<FeedbackSnapshot | null>(null);
    const [loadError, setLoadError] = useState("");
    const dialog = useRef<HTMLDialogElement>(null);
    useEffect(() => {
        if (!open || !snapshotId) return;
        dialog.current?.showModal();
        const controller = new AbortController();
        setSnapshot(null);
        setLoadError("");
        fetch(`${API_BASE}/api/admin/feedback/snapshots/${encodeURIComponent(snapshotId)}`, {
            credentials: "include", headers: authBearerHeaders(), signal: controller.signal,
        }).then(async response => {
            if (!response.ok) throw new Error(`Could not load transcript (HTTP ${response.status})`);
            const data = await response.json();
            if (!controller.signal.aborted) setSnapshot(data);
        }).catch(e => { if (!controller.signal.aborted) setLoadError(String(e.message || e)); });
        return () => controller.abort();
    }, [open, snapshotId]);

    async function exportJson() {
        try {
            const response = await fetch(`${API_BASE}/api/admin/feedback/snapshots/${encodeURIComponent(snapshotId || "")}?export=true`, {
                credentials: "include", headers: authBearerHeaders(),
            });
            if (!response.ok) throw new Error(`Export failed (HTTP ${response.status})`);
            const url = URL.createObjectURL(await response.blob());
            const link = document.createElement("a");
            link.href = url;
            link.download = `quasar-feedback-${snapshotId}.json`;
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch (e) { setLoadError(e instanceof Error ? e.message : "Export failed"); }
    }

    return <div className="mt-3">
        {snapshotId ? <button type="button" onClick={() => setOpen(true)} className="rounded-lg border border-sky-600/50 px-3 py-2 text-xs font-semibold text-sky-300">View full conversation</button>
            : <p className="text-xs text-slate-400">{error || "No full snapshot: context was not shared, or this feedback predates snapshot capture."}</p>}
        {open && createPortal(<dialog ref={dialog} onCancel={() => setOpen(false)} onClose={() => setOpen(false)} aria-labelledby="feedback-transcript-title"
            className="fixed inset-0 m-auto max-h-[90vh] w-[min(960px,94vw)] overflow-y-auto rounded-2xl border border-slate-700 bg-slate-950 p-6 shadow-2xl backdrop:bg-black/70">
            <div className="sticky top-0 z-10 mb-4 flex justify-end gap-3 bg-slate-950 py-2">
                {snapshot && <button type="button" onClick={exportJson} className="rounded-lg bg-sky-700 px-3 py-2 text-sm text-white">Export JSON</button>}
                <button type="button" autoFocus onClick={() => setOpen(false)} className="rounded-lg border border-slate-600 px-3 py-2 text-sm text-slate-200">Close transcript</button>
            </div>
            {loadError && <p role="alert" className="mb-3 text-sm text-red-300">{loadError}</p>}
            {snapshot ? <FeedbackTranscriptView snapshot={snapshot} /> : !loadError && <p role="status" className="text-slate-300">Loading full conversation…</p>}
        </dialog>, document.body)}
    </div>;
}
