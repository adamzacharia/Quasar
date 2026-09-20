import type { ReactNode } from "react";

export interface FeedbackSnapshot {
    id: string;
    captured_at: string;
    conversation_id: string;
    message_id: string;
    run_id: string;
    model: string;
    provider: string;
    vote: string;
    feedback_text: string;
    tool_evidence?: { note?: string };
    limits: { max_messages: number; max_bytes: number };
    truncation: { truncated: boolean; marker?: string; captured_messages: number; total_messages: number; reasons: string[] };
    messages: Array<{ id: string; role: string; content: string; created_at?: string; is_reported_answer?: boolean;
        metadata?: { runMeta?: Record<string, unknown>; toolTrace?: Array<Record<string, unknown>>;
            feedbackToolTrace?: { calls: Array<Record<string, unknown>>; truncated: boolean; total_calls: number; partial?: boolean; run_status?: string };
            [key: string]: unknown } }>;
}

/** Pure transcript surface shared by the admin dialog and rendering tests. */
export function FeedbackTranscriptView({ snapshot, actions }: { snapshot: FeedbackSnapshot; actions?: ReactNode }) {
    return <section className="space-y-5 text-slate-200">
        <header className="space-y-2">
            <h2 id="feedback-transcript-title" className="text-xl font-semibold">Full conversation</h2>
            <p className="text-xs text-slate-400">Captured {snapshot.captured_at} · {snapshot.provider || "Provider unavailable"} / {snapshot.model || "Model unavailable"} · {snapshot.vote}</p>
            <p className="break-all font-mono text-xs text-slate-400">Conversation {snapshot.conversation_id} · Run {snapshot.run_id || "unavailable"} · Message {snapshot.message_id}</p>
            <p className="text-xs text-slate-400">{snapshot.truncation.captured_messages} of {snapshot.truncation.total_messages} messages · Limit {snapshot.limits.max_messages} messages / {snapshot.limits.max_bytes / (1024 * 1024)} MiB</p>
            {actions}
        </header>
        {snapshot.truncation.truncated && <p role="status" className="rounded-lg border border-amber-500/40 bg-amber-950/30 p-3 text-sm text-amber-200">{snapshot.truncation.marker} ({snapshot.truncation.reasons.join(", ")}). Later messages are not in this snapshot.</p>}
        {snapshot.feedback_text && <div className="rounded-lg bg-red-950/30 p-3"><h3 className="text-xs font-semibold">Reporter feedback</h3><p className="mt-1 whitespace-pre-wrap break-words text-sm">{snapshot.feedback_text}</p></div>}
        <p className="text-xs text-slate-400">{snapshot.tool_evidence?.note}</p>
        {snapshot.messages.map((turn, index) => {
            const meta = turn.metadata || {};
            const fullTrace = meta.feedbackToolTrace;
            const calls = fullTrace?.calls || meta.toolTrace || [];
            return <article key={`${turn.id}-${index}`} className={`rounded-xl border p-4 ${turn.is_reported_answer ? "border-red-500/50 bg-red-950/10" : "border-slate-700"}`}>
                <h3 className="text-sm font-semibold">{index + 1}. {turn.role}{turn.is_reported_answer ? " · Reported answer" : ""}</h3>
                <p className="mt-1 break-all text-xs text-slate-400">Message {turn.id} · {turn.created_at || "Timestamp unavailable"}{meta.runMeta ? ` · ${String(meta.runMeta.provider || "")} / ${String(meta.runMeta.model || "")} · run ${String(meta.runMeta.run_id || "")}` : ""}</p>
                <div className="mt-3 whitespace-pre-wrap break-words text-sm leading-relaxed">{turn.content}</div>
                {calls.map((call, callIndex) => <details key={callIndex} className="mt-3 rounded-lg bg-slate-900 p-3">
                    <summary className="cursor-pointer text-xs font-medium">Tool {callIndex + 1}: {String(call.name || call.tool || "unnamed")}{fullTrace ? " · call and result" : " · persisted request (result unavailable)"}</summary>
                    <pre className="mt-3 whitespace-pre-wrap break-words text-xs">{JSON.stringify(call, null, 2)}</pre>
                </details>)}
                {fullTrace?.truncated && <p className="mt-2 text-xs text-amber-200">Tool trace truncated at {calls.length} of {fullTrace.total_calls} calls (2 MiB per turn).</p>}
                {fullTrace?.partial && <p className="mt-2 text-xs text-amber-200">Partial tool evidence: run {fullTrace.run_status}. Includes calls completed before the run ended; calls still running then have no captured result.</p>}
                <details className="mt-3 text-xs text-slate-400"><summary className="cursor-pointer">Persisted turn metadata</summary><pre className="mt-2 whitespace-pre-wrap break-words">{JSON.stringify(meta, null, 2)}</pre></details>
            </article>;
        })}
    </section>;
}
