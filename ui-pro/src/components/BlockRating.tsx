"use client";

/**
 * Per-block 1-5 star rating (Feature 4).
 *
 * Renders only in eval mode, which is admin-gated: rating is a labeling
 * workflow for the benchmark, not a product surface. Normal users keep the
 * like/dislike bar in MessageActions and the issue-report flow untouched.
 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Star, MessageSquarePlus, X, Check } from "lucide-react";
import type { BlockKind } from "../lib/types";
import { useChatStore } from "../lib/store";
import { useEvalModeActive } from "../lib/use-eval-mode";
import { submitBlockRating } from "../lib/api";

/* useLayoutEffect warns during SSR (it cannot run there) and Next pre-renders
   client components — so fall back to useEffect on the server, where there is no
   paint to race anyway. */
const useIsomorphicLayoutEffect = typeof window !== "undefined" ? useLayoutEffect : useEffect;

export function BlockRating({
    blockId,
    blockKind,
    runId,
    model,
}: {
    blockId?: string;
    blockKind?: BlockKind;
    runId?: string;
    model?: string;
}) {
    const evalMode = useEvalModeActive();
    // Confirmed vs in-flight: the star shows either, the GATE counts only the
    // confirmed one (store.ts::beginRating).
    const confirmed = useChatStore((s) => (blockId ? s.blockRatings[blockId] : undefined));
    const pending = useChatStore((s) => (blockId ? s.pendingRatings[blockId] : undefined));
    const rating = pending ?? confirmed;
    const beginRating = useChatStore((s) => s.beginRating);
    const resolveRating = useChatStore((s) => s.resolveRating);
    const activeConversationId = useChatStore((s) => s.activeConversationId);

    const registerBlock = useChatStore((s) => s.registerBlock);
    const unregisterBlock = useChatStore((s) => s.unregisterBlock);

    const [hovered, setHovered] = useState(0);
    const [showComment, setShowComment] = useState(false);
    const [comment, setComment] = useState("");
    const [commentSaved, setCommentSaved] = useState(false);
    const [error, setError] = useState("");

    /* Announce that this block is on screen and therefore rateable. The gate
       demands stars only for blocks that got this far, so a card ChatMessage
       suppressed (0-row table, papers absorbed by the graph) can never wedge
       the composer. Must sit above the early return — hooks run unconditionally.

       A LAYOUT effect, not a passive one: layout effects flush synchronously in
       the same commit, before the browser paints. That makes "registered" true by
       the time this card is first visible, so there is no frame in which a user
       could see an unrated block and submit around a still-pending registration —
       a passive effect only makes that unlikely, not impossible. */
    const visible = Boolean(evalMode && blockId);
    useIsomorphicLayoutEffect(() => {
        if (!visible || !blockId) return;
        registerBlock(blockId);
        return () => unregisterBlock(blockId);
    }, [visible, blockId, registerBlock, unregisterBlock]);

    // No id means the block predates stable ids — a rating could not be keyed
    // to it, so offering the control would be a lie. Spelled out rather than
    // `if (!visible)` so TypeScript narrows blockId to string below.
    if (!evalMode || !blockId) return null;

    /* At most ONE rating request per block may be in flight.

       Two overlapping POSTs for the same block can complete out of order (the
       endpoint's DB write finishes before its per-request Langfuse work, which
       varies), and both the server's delete+insert and the client's resolve
       apply whichever lands last. A 4★ then 2★ double-click could therefore
       leave the UI showing 4★ while the database holds 2★ — the widget lying
       about the stored label, which is the one thing a label-collection tool
       must never do. Serializing removes the reorder at the source instead of
       trying to detect it afterwards; a ref (not state) because two clicks in
       one frame would both see a stale `pending`. */
    const inFlight = useRef(false);

    const send = async (value: number, withComment: string) => {
        if (inFlight.current) return false;   // buttons are disabled meanwhile
        inFlight.current = true;
        beginRating(blockId, value);   // lights the star; does NOT open the gate
        setError("");
        try {
            const ok = await submitBlockRating({
                blockId,
                rating: value,
                blockKind,
                runId,
                conversationId: activeConversationId || "",
                model,
                comment: withComment,
            });
            // Confirm or drop. A failure needs no rollback: the confirmed map was
            // never written, so the star reverts to whatever the server actually
            // holds, and the gate never opened on a rating that failed.
            resolveRating(blockId, ok ? value : null);
            if (!ok) setError("Rating failed to save");
            return ok;
        } finally {
            inFlight.current = false;
        }
    };

    const saveComment = async () => {
        if (!rating) return;
        const ok = await send(rating, comment);
        if (ok) {
            setCommentSaved(true);
            setTimeout(() => { setCommentSaved(false); setShowComment(false); }, 1200);
        }
    };

    const active = hovered || rating || 0;

    return (
        <div className="mt-1.5 flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wide text-slate-500 select-none">
                {blockKind || "block"}
            </span>
            <div className="flex items-center gap-0.5" onMouseLeave={() => setHovered(0)}>
                {[1, 2, 3, 4, 5].map((value) => (
                    <button
                        key={value}
                        type="button"
                        aria-label={`Rate ${value} of 5`}
                        title={`Rate ${value}/5`}
                        onMouseEnter={() => setHovered(value)}
                        onClick={() => send(value, comment)}
                        disabled={pending !== undefined}
                        className="p-0.5 transition-transform hover:scale-110 active:scale-95 disabled:cursor-progress"
                    >
                        <Star
                            className={`w-3.5 h-3.5 ${value <= active ? "text-amber-400" : "text-slate-600"}`}
                            fill={value <= active ? "currentColor" : "none"}
                        />
                    </button>
                ))}
            </div>
            {rating ? (
                <button
                    type="button"
                    onClick={() => setShowComment((v) => !v)}
                    title="Add a note about this block"
                    className="p-1 text-slate-500 hover:text-slate-300 rounded transition-colors"
                >
                    <MessageSquarePlus className="w-3.5 h-3.5" />
                </button>
            ) : null}
            {error ? <span className="text-[10px] text-red-400">{error}</span> : null}

            {showComment && (
                <div className="flex items-center gap-1">
                    <input
                        value={comment}
                        onChange={(e) => setComment(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") saveComment(); }}
                        placeholder="Why this rating?"
                        autoFocus
                        className="w-56 px-2 py-1 text-xs rounded-md bg-slate-900/80 border border-slate-700 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-slate-500"
                    />
                    <button type="button" onClick={saveComment} title="Save note"
                        className="p-1 text-slate-400 hover:text-emerald-400">
                        {commentSaved ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Check className="w-3.5 h-3.5" />}
                    </button>
                    <button type="button" onClick={() => setShowComment(false)} title="Close"
                        className="p-1 text-slate-500 hover:text-slate-300">
                        <X className="w-3.5 h-3.5" />
                    </button>
                </div>
            )}
        </div>
    );
}
