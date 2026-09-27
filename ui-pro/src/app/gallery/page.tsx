"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, BookOpen, Play, Search } from "lucide-react";
import { AuthModal } from "@/components/AuthModal";
import { Sidebar } from "@/components/Sidebar";
import { useAuthStore } from "@/lib/auth-store";
import { useChatStore } from "@/lib/store";
import { RECIPES, RECIPE_TOPICS, type Recipe, type RecipeDifficulty } from "@/lib/recipes";

const DIFFICULTIES: Array<{ key: RecipeDifficulty | "all"; label: string }> = [
    { key: "all", label: "All levels" },
    { key: "starter", label: "Starter" },
    { key: "intermediate", label: "Intermediate" },
    { key: "advanced", label: "Advanced" },
];

// Soft semantic tints (green = easy, amber = medium, red = hard), no violet.
const DIFFICULTY_CHIP: Record<RecipeDifficulty, string> = {
    starter: "bg-emerald-500/10 text-emerald-600",
    intermediate: "bg-amber-500/10 text-amber-600",
    advanced: "bg-rose-500/10 text-rose-600",
};

const TOPIC_COUNTS: Record<string, number> = RECIPES.reduce<Record<string, number>>((acc, recipe) => {
    acc[recipe.topic] = (acc[recipe.topic] ?? 0) + 1;
    return acc;
}, {});

function matches(recipe: Recipe, topic: string, difficulty: string, query: string): boolean {
    if (topic !== "all" && recipe.topic !== topic) return false;
    if (difficulty !== "all" && recipe.difficulty !== difficulty) return false;
    const text = query.trim().toLowerCase();
    if (!text) return true;
    const haystack = [recipe.title, recipe.prompt, recipe.topic, ...recipe.catalogs].join(" ").toLowerCase();
    return text.split(/\s+/).every((token) => haystack.includes(token));
}

/** A filter pill: pressed = ink fill, like Orbita's segmented choices. */
function FilterPill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
    return (
        <button
            type="button"
            onClick={onClick}
            aria-pressed={active}
            className={active ? "q-pill-ink h-8 px-3 text-xs" : "q-pill h-8 px-3 text-xs"}
        >
            {children}
        </button>
    );
}

function RecipeCard({ recipe, index, onRun }: { recipe: Recipe; index: number; onRun: (recipe: Recipe) => void }) {
    return (
        <button
            type="button"
            onClick={() => onRun(recipe)}
            className="glass-card q-rise group flex h-full flex-col rounded-2xl p-4 text-left"
            style={{ animationDelay: `${Math.min(index, 12) * 30}ms` }}
        >
            <div className="flex items-start justify-between gap-2">
                <h3 className="text-[13px] font-medium leading-snug text-[var(--q-text)]">{recipe.title}</h3>
                <span className={`shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-medium capitalize ${DIFFICULTY_CHIP[recipe.difficulty]}`}>
                    {recipe.difficulty}
                </span>
            </div>
            <p className="mt-2 line-clamp-4 flex-1 text-xs leading-relaxed text-[var(--q-text-muted)]">{recipe.prompt}</p>
            <div className="mt-3 flex items-center justify-between gap-2 border-t border-[var(--q-border)] pt-2.5">
                <span className="truncate text-[10px] text-[var(--q-text-faint)]">
                    {recipe.catalogs.length ? recipe.catalogs.join(" · ") : recipe.topic}
                </span>
                <span className="inline-flex shrink-0 items-center gap-1 text-[11px] font-medium text-[var(--q-text)] opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
                    <Play className="h-3 w-3" strokeWidth={1.75} />
                    Try it
                </span>
            </div>
        </button>
    );
}

export default function GalleryPage() {
    const router = useRouter();
    const sidebarOpen = useChatStore((state) => state.sidebarOpen);
    const toggleSidebar = useChatStore((state) => state.toggleSidebar);
    const { isAuthenticated, openAuthModal } = useAuthStore();

    const [topic, setTopic] = useState<string>("all");
    const [difficulty, setDifficulty] = useState<string>("all");
    const [query, setQuery] = useState("");

    const visible = useMemo(
        () => RECIPES.filter((recipe) => matches(recipe, topic, difficulty, query)),
        [topic, difficulty, query],
    );

    const runRecipe = (recipe: Recipe) => {
        // Prefill the chat composer (no auto-send) — ChatArea consumes ?prompt=.
        router.push(`/?prompt=${encodeURIComponent(recipe.prompt)}`);
    };

    if (!isAuthenticated) {
        return (
            <div className="flex h-full w-full flex-col items-center justify-center gap-4 bg-[var(--q-bg)] text-[var(--q-text-muted)]">
                <p className="text-sm">Sign in to browse the recipe gallery.</p>
                <button type="button" onClick={openAuthModal} className="q-pill-ink h-9 px-5 text-[13px]">
                    Sign in
                </button>
                <AuthModal />
            </div>
        );
    }

    return (
        <div className="flex h-full w-full overflow-hidden">
            <div className={`${sidebarOpen ? "w-[calc(var(--q-sidebar-rail-width)+var(--q-sidebar-width))]" : "w-[var(--q-sidebar-rail-width)]"} shrink-0 overflow-hidden transition-all duration-300`}>
                <Sidebar collapsed={!sidebarOpen} onToggle={toggleSidebar} />
            </div>
            <main className="flex min-w-0 flex-1 flex-col overflow-hidden bg-[var(--q-bg)]">
                <header className="flex items-center justify-between gap-3 px-5 py-3">
                    <div className="min-w-0">
                        <div className="flex items-center gap-2">
                            <BookOpen className="h-[18px] w-[18px] text-[var(--q-text-muted)]" strokeWidth={1.75} />
                            <h1 className="text-[15px] font-medium text-[var(--q-text)]">Recipe Gallery</h1>
                            <span className="q-tag">{RECIPES.length}</span>
                        </div>
                        <p className="mt-0.5 truncate text-xs text-[var(--q-text-faint)]">
                            Research questions Quasar handles well. Pick one and it opens in the chat, ready to send.
                        </p>
                    </div>
                    <button type="button" onClick={() => router.push("/")} className="q-pill h-9 shrink-0 px-3.5 text-[13px]">
                        <ArrowLeft className="h-4 w-4" strokeWidth={1.75} />
                        Back to chat
                    </button>
                </header>

                <div className="flex min-h-0 flex-1 flex-col overflow-hidden q-canvas md:mx-3 md:mb-3 md:rounded-2xl">
                    {/* Filters */}
                    <div className="flex flex-wrap items-center gap-1.5 px-5 pb-2 pt-4">
                        <div className="relative mr-1">
                            <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[var(--q-text-faint)]" />
                            <input
                                value={query}
                                onChange={(event) => setQuery(event.target.value)}
                                placeholder="Search questions, archives, objects…"
                                className="h-8 w-64 rounded-full border border-[var(--q-border)] bg-[var(--q-card)] pl-8 pr-3 text-xs text-[var(--q-text)] placeholder:text-[var(--q-text-faint)] focus:border-[var(--q-border-strong)] focus:outline-none"
                            />
                        </div>
                        {DIFFICULTIES.map(({ key, label }) => (
                            <FilterPill key={key} active={difficulty === key} onClick={() => setDifficulty(key)}>{label}</FilterPill>
                        ))}
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5 px-5 pb-4">
                        <FilterPill active={topic === "all"} onClick={() => setTopic("all")}>
                            All categories <span className="opacity-60">{RECIPES.length}</span>
                        </FilterPill>
                        {RECIPE_TOPICS.map((item) => (
                            <FilterPill key={item} active={topic === item} onClick={() => setTopic(item)}>
                                {item} <span className="opacity-60">{TOPIC_COUNTS[item] ?? 0}</span>
                            </FilterPill>
                        ))}
                    </div>

                    {/* Cards */}
                    <div className="flex-1 overflow-y-auto px-5 pb-5">
                        {visible.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-20 text-center">
                                <p className="text-sm font-medium text-[var(--q-text-muted)]">Nothing matches those filters</p>
                                <p className="mt-1.5 text-xs text-[var(--q-text-faint)]">Try clearing the search or picking another category.</p>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                                {visible.map((recipe, i) => (
                                    <RecipeCard key={recipe.id} recipe={recipe} index={i} onRun={runRecipe} />
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </main>
            <AuthModal />
        </div>
    );
}
