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

const DIFFICULTY_CHIP: Record<RecipeDifficulty, string> = {
    starter: "bg-emerald-500/15 text-emerald-400",
    intermediate: "bg-amber-500/15 text-amber-400",
    advanced: "bg-fuchsia-500/15 text-fuchsia-400",
};

function matches(recipe: Recipe, topic: string, difficulty: string, query: string): boolean {
    if (topic !== "all" && recipe.topic !== topic) return false;
    if (difficulty !== "all" && recipe.difficulty !== difficulty) return false;
    const text = query.trim().toLowerCase();
    if (!text) return true;
    const haystack = [
        recipe.title, recipe.description, recipe.prompt, recipe.topic,
        ...recipe.uat, ...recipe.catalogs,
    ].join(" ").toLowerCase();
    return text.split(/\s+/).every((token) => haystack.includes(token));
}

function RecipeCard({ recipe, onRun }: { recipe: Recipe; onRun: (recipe: Recipe) => void }) {
    return (
        <button
            type="button"
            onClick={() => onRun(recipe)}
            className="glass-card group flex h-full flex-col rounded-2xl p-4 text-left transition-all hover:border-cyan-500/40 hover:shadow-lg hover:shadow-cyan-500/5"
        >
            <div className="flex items-start justify-between gap-2">
                <h3 className="text-sm font-semibold text-white leading-snug">{recipe.title}</h3>
                <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium capitalize ${DIFFICULTY_CHIP[recipe.difficulty]}`}>
                    {recipe.difficulty}
                </span>
            </div>
            <p className="mt-2 flex-1 text-xs leading-relaxed text-slate-400">{recipe.description}</p>
            <div className="mt-3 flex flex-wrap gap-1">
                {recipe.uat.slice(0, 3).map((keyword) => (
                    <span key={keyword} className="rounded bg-white/5 px-1.5 py-0.5 text-[10px] text-slate-500">
                        {keyword}
                    </span>
                ))}
            </div>
            <div className="mt-3 flex items-center justify-between border-t border-slate-700/40 pt-2.5">
                <span className="truncate text-[10px] text-slate-500">{recipe.catalogs.join(" · ")}</span>
                <span className="inline-flex shrink-0 items-center gap-1 text-[11px] font-semibold text-cyan-300 opacity-0 transition-opacity group-hover:opacity-100">
                    <Play className="h-3 w-3" />
                    Use recipe
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
            <div className="flex h-full w-full flex-col items-center justify-center gap-4 bg-slate-950 text-slate-400">
                <p>Sign in to browse the recipe gallery.</p>
                <button
                    type="button"
                    onClick={openAuthModal}
                    className="rounded-md bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400"
                >
                    Sign in
                </button>
                <AuthModal />
            </div>
        );
    }

    return (
        <div className="flex h-full w-full overflow-hidden">
            <div className={`${sidebarOpen ? "w-[var(--q-sidebar-width)]" : "w-[var(--q-sidebar-rail-width)]"} shrink-0 overflow-hidden transition-all duration-300`}>
                <Sidebar collapsed={!sidebarOpen} onToggle={toggleSidebar} />
            </div>
            <main className="flex min-w-0 flex-1 flex-col overflow-hidden bg-[#0b0d12]">
                <header className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                    <div>
                        <div className="flex items-center gap-2">
                            <BookOpen className="h-5 w-5 text-cyan-300" />
                            <h1 className="text-lg font-semibold text-slate-100">Recipe Gallery</h1>
                        </div>
                        <p className="mt-1 text-xs text-slate-500">
                            One-click catalog-science workflows — each card prefills a ready-to-run prompt in the chat
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={() => router.push("/")}
                        className="inline-flex items-center gap-2 rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-cyan-500/50 hover:bg-slate-800 hover:text-cyan-200"
                    >
                        <ArrowLeft className="h-4 w-4" />
                        Back to chat
                    </button>
                </header>

                {/* Filters */}
                <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 px-5 py-3">
                    <div className="relative">
                        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-500" />
                        <input
                            value={query}
                            onChange={(event) => setQuery(event.target.value)}
                            placeholder="Search topic, catalog, UAT keyword…"
                            className="w-64 rounded-lg border border-slate-700 bg-slate-900 py-1.5 pl-8 pr-3 text-xs text-slate-200 placeholder:text-slate-600 focus:border-cyan-500/50 focus:outline-none"
                        />
                    </div>
                    <div className="flex flex-wrap gap-1">
                        <button
                            type="button"
                            onClick={() => setTopic("all")}
                            className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${topic === "all" ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}
                        >
                            All topics
                        </button>
                        {RECIPE_TOPICS.map((item) => (
                            <button
                                key={item}
                                type="button"
                                onClick={() => setTopic(item)}
                                className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${topic === item ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}
                            >
                                {item}
                            </button>
                        ))}
                    </div>
                    <div className="ml-auto flex gap-1">
                        {DIFFICULTIES.map(({ key, label }) => (
                            <button
                                key={key}
                                type="button"
                                onClick={() => setDifficulty(key)}
                                className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${difficulty === key ? "bg-primary/10 text-primary" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}
                            >
                                {label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Cards */}
                <div className="flex-1 overflow-y-auto p-5">
                    {visible.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-20 text-center">
                            <p className="text-sm font-medium text-slate-400">No recipes match those filters</p>
                            <p className="mt-1.5 text-xs text-slate-600">Try clearing the search or picking another topic.</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                            {visible.map((recipe) => (
                                <RecipeCard key={recipe.id} recipe={recipe} onRun={runRecipe} />
                            ))}
                        </div>
                    )}
                </div>
            </main>
            <AuthModal />
        </div>
    );
}
