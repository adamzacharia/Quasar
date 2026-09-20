"use client";

/**
 * Chat model selector, backed by the live per-user provider catalog
 * (GET /api/models/available) instead of a hardcoded id list.
 *
 * The visual language is unchanged from the pre-catalog dropdown: same trigger,
 * same upward panel, same section headers, same `In: $X/M · Out: $Y/M` price
 * line. What is new is the search box, the grouped fuzzy results, the collapsed
 * not-connected providers, and listbox keyboard/ARIA semantics.
 *
 * The sidebar is deliberately dark in both themes (it has no `dark:` variants
 * anywhere), so this panel keeps the same slate palette rather than introducing
 * a light variant that would clash with its container.
 */
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { Check, ChevronDown, Plus, Search } from "lucide-react";

import { ModelIcon } from "./ModelIcon";
import { isTaccModel } from "../lib/models";
import { useAvailableModels } from "../lib/useAvailableModels";
import {
    buildGroups,
    connectedLabels,
    contextChip,
    flattenRows,
    loadRecentModels,
    modelLabel,
    nextRecentModels,
    persistRecentModels,
    priceLine,
    suggestedLockedProvider,
} from "../lib/model-catalog";
import type { MatchRange, ModelInfo } from "../lib/model-catalog";

/** Render text with the matched character runs highlighted. */
function Highlight({ text, ranges }: { text: string; ranges: MatchRange[] }) {
    if (!ranges || ranges.length === 0) return <>{text}</>;
    const parts: React.ReactNode[] = [];
    let cursor = 0;
    ranges.forEach(([start, end], index) => {
        if (start > cursor) parts.push(text.slice(cursor, start));
        parts.push(
            <mark key={`${start}-${index}`} className="bg-transparent text-primary font-semibold">
                {text.slice(start, end)}
            </mark>,
        );
        cursor = end;
    });
    if (cursor < text.length) parts.push(text.slice(cursor));
    return <>{parts}</>;
}

function tagFor(model: ModelInfo): string | null {
    return model.provider === "tacc" ? "US hosted" : null;
}

export function ModelDropdown({
    selectedModel,
    onSelect,
    onAddProviderKey,
}: {
    selectedModel: string;
    onSelect: (model: string) => void;
    onAddProviderKey?: () => void;
}) {
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState("");
    const [activeIndex, setActiveIndex] = useState(0);
    // Read once on first render. The panel is closed on mount, so the stored
    // list never reaches the server-rendered markup and cannot mismatch.
    const [recent, setRecent] = useState<string[]>(loadRecentModels);
    const ref = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const listRef = useRef<HTMLDivElement>(null);
    const listId = useId();

    const { providers } = useAvailableModels();

    useEffect(() => {
        function handleClick(e: MouseEvent) {
            if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
        }
        document.addEventListener("mousedown", handleClick);
        return () => document.removeEventListener("mousedown", handleClick);
    }, []);

    const { groups, locked } = useMemo(
        () => buildGroups(query, providers, { recent }),
        [query, providers, recent],
    );
    const flat = useMemo(() => flattenRows(groups), [groups]);
    /** Index of each group's first row within `flat`, for keyboard addressing. */
    const groupOffsets = useMemo(() => {
        const offsets: number[] = [];
        let running = 0;
        for (const group of groups) {
            offsets.push(running);
            running += group.rows.length;
        }
        return offsets;
    }, [groups]);

    // Derived rather than clamped in an effect: the query can shrink the result
    // list under the cursor between renders.
    const active = flat.length === 0 ? 0 : Math.min(activeIndex, flat.length - 1);

    const commit = useCallback(
        (modelId: string) => {
            const updated = nextRecentModels(recent, modelId);
            setRecent(updated);
            persistRecentModels(updated);
            onSelect(modelId);
            setOpen(false);
        },
        [onSelect, recent],
    );

    const openKeys = useCallback(() => {
        setOpen(false);
        onAddProviderKey?.();
    }, [onAddProviderKey]);

    const handleKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLDivElement>) => {
            if (event.key === "ArrowDown") {
                event.preventDefault();
                if (flat.length > 0) setActiveIndex((active + 1) % flat.length);
                return;
            }
            if (event.key === "ArrowUp") {
                event.preventDefault();
                if (flat.length > 0) setActiveIndex((active - 1 + flat.length) % flat.length);
                return;
            }
            if (event.key === "Enter") {
                event.preventDefault();
                const row = flat[active];
                if (row) commit(row.model.id);
                return;
            }
            if (event.key === "Escape") {
                event.preventDefault();
                // First press clears the query, second closes the panel.
                if (query) setQuery("");
                else setOpen(false);
                return;
            }
            // Typing anywhere in the panel lands in the search box.
            if (
                event.key.length === 1 &&
                !event.ctrlKey &&
                !event.metaKey &&
                !event.altKey &&
                document.activeElement !== inputRef.current
            ) {
                inputRef.current?.focus();
            }
        },
        [active, commit, flat, query],
    );

    // Keep the active option visible during keyboard navigation.
    useEffect(() => {
        if (!open || !listRef.current) return;
        const activeEl = listRef.current.querySelector<HTMLElement>('[data-active="true"]');
        activeEl?.scrollIntoView({ block: "nearest" });
    }, [active, open]);

    const selectedInfo = useMemo(() => {
        for (const catalog of providers) {
            for (const model of catalog.models) {
                if (model.id === selectedModel) return model;
            }
        }
        return null;
    }, [providers, selectedModel]);

    const currentIn = selectedInfo?.inputPricePerM;
    const currentOut = selectedInfo?.outputPricePerM;
    const showCurrentCost =
        typeof currentIn === "number" && typeof currentOut === "number" && !isTaccModel(selectedModel);
    const currentTag = isTaccModel(selectedModel) ? "US hosted" : null;

    const connected = connectedLabels(providers);
    const lockedHint = suggestedLockedProvider(query, providers);

    return (
        <div ref={ref} className="relative" onKeyDown={open ? handleKeyDown : undefined}>
            <button
                onClick={() => {
                    // Every open starts from a fresh, unfiltered list.
                    if (!open) {
                        setQuery("");
                        setActiveIndex(0);
                    }
                    setOpen(!open);
                }}
                aria-haspopup="listbox"
                aria-expanded={open}
                className="flex flex-col items-start w-full px-3 py-2 text-xs font-medium text-slate-300 bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors border border-slate-700/50"
            >
                <div className="flex items-center justify-between w-full">
                    <div className="flex items-center gap-2">
                        <ModelIcon model={selectedModel} className="w-4 h-4 shrink-0" />
                        <span className="truncate">
                            Model: {selectedModel.startsWith("local/") ? selectedModel.replace("local/", "[Local] ") : selectedModel}
                        </span>
                    </div>
                    <ChevronDown className={`w-4 h-4 shrink-0 transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
                </div>
                {showCurrentCost && (
                    <div className="flex items-center gap-2 mt-1 ml-6 text-[10px] text-slate-400/80 font-mono">
                        <span>In: ${currentIn}/M</span>
                        <span className="text-slate-600">|</span>
                        <span>Out: ${currentOut}/M</span>
                    </div>
                )}
                {currentTag && (
                    <div className="mt-1 ml-6">
                        <span className="inline-flex items-center rounded border border-cyan-400/30 bg-cyan-400/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-cyan-200">
                            {currentTag}
                        </span>
                    </div>
                )}
            </button>

            {open && (
                <div className="absolute bottom-full left-0 right-0 mb-1.5 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl shadow-black/50 overflow-hidden z-50 animate-in fade-in slide-in-from-bottom-2 duration-150 flex flex-col max-h-96">
                    <div className="shrink-0 p-2 border-b border-slate-700/50 bg-slate-900">
                        <div className="relative">
                            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
                            <input
                                ref={inputRef}
                                type="text"
                                // The panel only mounts on open, so this is the
                                // "focus the search box when it opens" behaviour.
                                autoFocus
                                role="combobox"
                                aria-expanded={open}
                                aria-controls={listId}
                                aria-autocomplete="list"
                                aria-label="Search models"
                                aria-activedescendant={flat[active] ? `${listId}-opt-${active}` : undefined}
                                value={query}
                                onChange={(e) => {
                                    setQuery(e.target.value);
                                    setActiveIndex(0);
                                }}
                                placeholder="Search models…"
                                className="w-full bg-slate-800 border border-slate-700/60 rounded-lg pl-8 pr-2 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-primary/60"
                            />
                        </div>
                    </div>

                    <div
                        ref={listRef}
                        id={listId}
                        role="listbox"
                        aria-label="Available models"
                        className="flex-1 overflow-y-auto custom-scrollbar"
                    >
                        {groups.map((group, groupIndex) => (
                            <div key={group.key} role="group" aria-label={group.label}>
                                {groupIndex > 0 && <div className="border-t border-slate-700/50 mx-2" />}
                                <div className="px-3 py-1.5 flex items-center gap-1.5">
                                    <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                                        {group.label}
                                    </span>
                                    {group.byok && (
                                        <span className="inline-flex items-center rounded border border-emerald-400/30 bg-emerald-400/10 px-1 py-px text-[8px] font-semibold uppercase tracking-wide text-emerald-200">
                                            BYOK
                                        </span>
                                    )}
                                    {group.stale && (
                                        <span
                                            title="Showing a cached list — the provider could not be reached."
                                            className="inline-flex items-center rounded border border-amber-400/30 bg-amber-400/10 px-1 py-px text-[8px] font-semibold uppercase tracking-wide text-amber-200"
                                        >
                                            stale
                                        </span>
                                    )}
                                </div>
                                {group.rows.map((row, offset) => {
                                    const index = groupOffsets[groupIndex] + offset;
                                    const model = row.model;
                                    const label = modelLabel(model);
                                    const price = priceLine(model);
                                    const ctx = contextChip(model);
                                    const tag = tagFor(model);
                                    const isSelected = model.id === selectedModel;
                                    const isActive = index === active;
                                    const showId = model.id !== label;
                                    // Ranges are measured against displayName; only reuse
                                    // them when the rendered label is that same string
                                    // (a local/ prefix strip would shift every offset).
                                    const labelRanges = label === model.displayName ? row.nameRanges : [];
                                    return (
                                        <div
                                            key={`${group.key}:${model.id}`}
                                            id={`${listId}-opt-${index}`}
                                            role="option"
                                            aria-selected={isSelected}
                                            data-active={isActive}
                                            tabIndex={-1}
                                            onClick={() => commit(model.id)}
                                            onMouseEnter={() => setActiveIndex(index)}
                                            className={`w-full flex items-center justify-between px-3 py-2 text-sm transition-colors cursor-pointer ${
                                                isSelected
                                                    ? "bg-primary/15 text-primary border-l-2 border-primary"
                                                    : isActive
                                                      ? "bg-slate-700/70 text-white"
                                                      : "text-slate-300 hover:bg-slate-700/70 hover:text-white"
                                            }`}
                                        >
                                            <div className="flex flex-col items-start truncate overflow-hidden pr-2">
                                                <span className="font-medium truncate w-full text-left">
                                                    <Highlight text={label} ranges={labelRanges} />
                                                </span>
                                                {showId && (
                                                    <span className="text-[9px] text-slate-500 font-mono truncate w-full text-left">
                                                        <Highlight text={model.id} ranges={row.idRanges} />
                                                    </span>
                                                )}
                                                <div className="flex items-center gap-1 flex-wrap">
                                                    {tag && (
                                                        <span className="mt-1 inline-flex items-center rounded border border-cyan-400/30 bg-cyan-400/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-cyan-200">
                                                            {tag}
                                                        </span>
                                                    )}
                                                    {ctx && (
                                                        <span className="mt-1 inline-flex items-center rounded border border-slate-600/60 bg-slate-700/40 px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide text-slate-300">
                                                            {ctx}
                                                        </span>
                                                    )}
                                                </div>
                                                {price && (
                                                    <span className="text-[9px] text-slate-500 font-mono mt-0.5">{price}</span>
                                                )}
                                            </div>
                                            {isSelected && <Check className="w-4 h-4 text-primary shrink-0" />}
                                        </div>
                                    );
                                })}
                            </div>
                        ))}

                        {flat.length === 0 && (
                            <div className="px-3 py-4 text-xs text-slate-400 space-y-1.5">
                                <p>
                                    No models match &ldquo;{query}&rdquo;.
                                    {connected.length > 0 && (
                                        <> Connected providers: {connected.join(", ")}.</>
                                    )}
                                </p>
                                {lockedHint && (
                                    <button
                                        onClick={openKeys}
                                        className="inline-flex items-center gap-1 text-primary hover:underline"
                                    >
                                        <Plus className="w-3 h-3" />
                                        Add a{/^[aeiou]/i.test(lockedHint.label) ? "n" : ""} {lockedHint.label} API key
                                    </button>
                                )}
                            </div>
                        )}
                    </div>

                    {locked.length > 0 && (
                        <div className="shrink-0 border-t border-slate-700/50 bg-slate-900">
                            {locked.map((entry) => (
                                <button
                                    key={entry.provider}
                                    onClick={openKeys}
                                    className="w-full flex items-center justify-between px-3 py-2 text-[11px] text-slate-500 hover:text-slate-300 hover:bg-slate-800/60 transition-colors"
                                >
                                    <span className="truncate text-left">
                                        {entry.label} — Add API key to unlock {entry.unlockCount} models
                                    </span>
                                    <Plus className="w-3 h-3 shrink-0" />
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
