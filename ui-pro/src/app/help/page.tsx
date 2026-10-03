"use client";

import { Sidebar } from "@/components/Sidebar";
import { useChatStore } from "../../lib/store";
import { AuthModal } from "@/components/AuthModal";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { LucideIcon } from "lucide-react";
import {
    ArrowLeft, ArrowUpRight, BookOpen, Bookmark, Brain, Database, ExternalLink, FileUp, Github, Globe, KeyRound,
    LayoutGrid, ListTree, MessageSquareText, Play, Plug, Shield, ShieldCheck, Sparkles, Star, Telescope, Waves,
} from "lucide-react";
import { IconOpenBook } from "@/components/icons/QuasarIcons";
import { PartnerLogos } from "@/components/PartnerLogos";
import { TERMS_UPDATED, TermsContent } from "@/components/TermsContent";
import { RECIPES } from "@/lib/recipes";

/** Example questions, taken from the Recipe Gallery (benchmark questions
 * Quasar answered well), grouped by the kind of request. */
const EXAMPLE_GROUPS: Array<{ label: string; ids: string[] }> = [
    { label: "ALMA and NRAO archives", ids: ["nos-02", "ab-d-48", "ab-d-39"] },
    { label: "Catalogs and other archives", ids: ["ab-d-12", "ab-d-17", "dlb-02"] },
    { label: "Plots and sky images", ids: ["dlb-03", "ab-d-25"] },
    { label: "Literature", ids: ["ab-d-59", "ab-d-60"] },
    { label: "Proposals and policy", ids: ["pol-01", "pol-03"] },
    { label: "Transients and exoplanets", ids: ["ab-d-47", "ab-d-42"] },
];

const PROMPTS_BY_ID = new Map(RECIPES.map((r) => [r.id, r.prompt]));

const STEPS = [
    { title: "Ask in plain language", body: "Name the target, band, line, instrument or date range you care about. Coordinates help when a name is ambiguous." },
    { title: "Watch it work", body: "The Research timeline shows each tool Quasar runs, what it asked the archive, and what came back." },
    { title: "Use the results", body: "Answers come with tables, plots, sky images, code and download links. Ask follow-ups to filter or plot further." },
    { title: "Check the sources", body: "Open the archive records and papers Quasar cites. AI answers can be wrong, so verify before you publish." },
];

const FEATURES: Array<{ icon: LucideIcon; title: string; body: string }> = [
    { icon: ListTree, title: "Research timeline", body: "A live, step-by-step view of the tools each answer used, with sources you can open." },
    { icon: Globe, title: "Web search modes", body: "Off, Auto or Always, from the Web button in the composer. Auto searches the web only when archives and papers are not enough." },
    { icon: ShieldCheck, title: "Grounded mode", body: "Found in the composer's sources menu. Limits the answer to what the tools actually returned, with no outside background knowledge." },
    { icon: FileUp, title: "Attach files", body: "Images, PDFs, FITS files, CSV and JSON tables, text and Word documents. FITS headers are read and previewed." },
    { icon: LayoutGrid, title: "Recipe Gallery", body: "Ready-to-run research questions by topic and difficulty. Pick one and it opens in the chat." },
    { icon: Bookmark, title: "Saved papers", body: "Bookmark papers from any answer. They are kept with your account in the Saved Papers panel." },
    { icon: Database, title: "Data Lab jobs and tables", body: "Long NOIRLab Astro Data Lab queries run in the background, and you can keep result tables in My tables." },
    { icon: Waves, title: "Spectral Line Explorer", body: "Find spectral lines in an ALMA band or a custom frequency window from Splatalogue catalogs, and export them." },
    { icon: Telescope, title: "Cube workbench", body: "Open a FITS cube from a data card to inspect its shape, beam, channels and rest frequency." },
    { icon: Brain, title: "Memory", body: "Quasar remembers your research focus, facilities and targets. View, edit, pause or clear it in Settings." },
    { icon: BookOpen, title: "Your knowledge base", body: "Add your own PDFs, text or Markdown notes in Settings so Quasar can search them in your chats." },
    { icon: KeyRound, title: "Your own API keys", body: "Add a DeepSeek, OpenAI, Anthropic or Google key in Settings to unlock more models. Keys are stored encrypted." },
    { icon: Plug, title: "MCP servers", body: "Connect extra tools through the Model Context Protocol, including verified astronomy servers from the registry." },
    { icon: Star, title: "Feedback", body: "Rate answers, or use the thumbs-down \"Report a problem\" button to tell us what went wrong." },
];

const DATA_SOURCES: Array<{ group: string; items: string[] }> = [
    { group: "Radio and millimeter", items: ["ALMA Science Archive", "NRAO (VLA, VLBA, GBT, VLASS)", "Splatalogue"] },
    { group: "Optical and infrared", items: ["NOIRLab Astro Data Lab", "Gaia", "MAST (HST, JWST, TESS)", "IRSA (WISE, 2MASS, Spitzer, ZTF)", "ESO", "CADC", "DESI / SPARCL"] },
    { group: "High energy and time domain", items: ["HEASARC (Chandra, XMM-Newton)", "Fermi", "ALeRCE", "TNS", "GCN", "GWOSC"] },
    { group: "Reference and images", items: ["SIMBAD", "NED", "VizieR", "NASA Exoplanet Archive", "CDS hips2fits"] },
    { group: "Literature and web", items: ["NASA ADS", "arXiv", "OpenAlex", "Web search"] },
];

const RESOURCES = [
    { href: "https://github.com/adamzacharia/Quasar", title: "GitHub repository", sub: "Source code and technical docs", icon: Github },
    { href: "https://almascience.nrao.edu/aq/", title: "ALMA Science Archive", sub: "Official archive query", icon: Telescope },
    { href: "https://data.nrao.edu/", title: "NRAO Archive", sub: "VLA, VLBA and GBT data", icon: Waves },
    { href: "https://ui.adsabs.harvard.edu/", title: "NASA ADS", sub: "Astrophysics Data System", icon: BookOpen },
    { href: "https://datalab.noirlab.edu/", title: "Astro Data Lab", sub: "NOIRLab catalogs and images", icon: Database },
];

const TIPS = [
    "Be specific: \"CO(3-2) in NGC 253\" beats \"molecular gas in a starburst\".",
    "Give coordinates (RA, Dec in degrees) when a target name could mean more than one object.",
    "Ask for Python if you want to reproduce a query yourself, for example with astroquery or TAP.",
    "Follow up on a result: \"only Band 6\", \"plot these on the sky\", \"which of these are public?\".",
    "Say \"don't search the web\" when you only want answers from archives and papers.",
];

function SectionTitle({ icon: Icon, children }: { icon: LucideIcon; children: React.ReactNode }) {
    return (
        <h2 className="mb-4 flex items-center gap-2.5 text-[15px] font-medium text-[var(--q-text)]">
            <span className="flex size-7 items-center justify-center rounded-lg bg-[var(--q-canvas)] text-[var(--q-text-muted)]">
                <Icon className="h-4 w-4" strokeWidth={1.75} />
            </span>
            {children}
        </h2>
    );
}

export default function HelpPage() {
    const router = useRouter();
    const sidebarOpen = useChatStore((s) => s.sidebarOpen);
    const toggleSidebar = useChatStore((s) => s.toggleSidebar);
    const [mounted, setMounted] = useState(false);
    const [activeTab, setActiveTab] = useState<"docs" | "terms">("docs");

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setMounted(true);
    }, []);

    if (!mounted) return null; // Prevent hydration mismatch flash

    const tryPrompt = (prompt: string) => {
        // Prefill the chat composer (no auto-send); ChatArea consumes ?prompt=.
        router.push(`/?prompt=${encodeURIComponent(prompt)}`);
    };

    const tabs = [
        { id: "docs" as const, label: "Documentation", icon: IconOpenBook },
        { id: "terms" as const, label: "Terms & Privacy", icon: Shield },
    ];

    return (
        <div className="flex h-full w-full overflow-hidden">
            <div className={`${sidebarOpen ? "w-[calc(var(--q-sidebar-rail-width)+var(--q-sidebar-width))]" : "w-[var(--q-sidebar-rail-width)]"} shrink-0 overflow-hidden transition-all duration-300`}>
                <Sidebar collapsed={!sidebarOpen} onToggle={toggleSidebar} />
            </div>

            <main className="flex min-w-0 flex-1 flex-col overflow-hidden bg-[var(--q-bg)]">
                <header className="flex items-center justify-between gap-3 px-5 py-3">
                    <div className="flex items-center gap-2">
                        <IconOpenBook className="h-[18px] w-[18px] text-[var(--q-text-muted)]" />
                        <h1 className="text-[15px] font-medium text-[var(--q-text)]">Help &amp; Docs</h1>
                    </div>
                    <button type="button" onClick={() => router.push("/")} className="q-pill h-9 shrink-0 px-3.5 text-[13px]">
                        <ArrowLeft className="h-4 w-4" strokeWidth={1.75} />
                        Back to chat
                    </button>
                </header>

                <div className="min-h-0 flex-1 overflow-y-auto q-canvas md:mx-3 md:mb-3 md:rounded-2xl">
                    <div className="mx-auto max-w-5xl space-y-10 px-5 py-8 md:px-8 md:py-10">
                        {/* Intro */}
                        <div className="space-y-3">
                            <h2 className="text-2xl font-medium tracking-tight text-[var(--q-text)] md:text-3xl">
                                Your research assistant for astronomy
                            </h2>
                            <p className="max-w-3xl text-[14px] leading-relaxed text-[var(--q-text-muted)]">
                                Ask Quasar a question in plain language. It picks from about 200 tools to search
                                ALMA, NRAO and dozens of other archives, catalogs and the literature, runs the
                                queries, and shows you every step so you can check its work.
                            </p>
                        </div>

                        {/* Tabs */}
                        <div className="flex gap-1.5" role="tablist">
                            {tabs.map((tab) => (
                                <button
                                    key={tab.id}
                                    type="button"
                                    role="tab"
                                    aria-selected={activeTab === tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={activeTab === tab.id ? "q-pill-ink h-9 px-4 text-[13px]" : "q-pill h-9 px-4 text-[13px]"}
                                >
                                    <tab.icon className="h-4 w-4" />
                                    {tab.label}
                                </button>
                            ))}
                        </div>

                        {activeTab === "docs" && (
                            <div className="space-y-12 animate-in fade-in duration-300">
                                {/* Getting started */}
                                <section>
                                    <SectionTitle icon={Sparkles}>Getting started</SectionTitle>
                                    <ol className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                                        {STEPS.map((step, i) => (
                                            <li key={step.title} className="glass-card rounded-2xl p-4">
                                                <span className="mb-2 flex size-6 items-center justify-center rounded-full bg-[var(--q-ink)] text-[11px] font-medium text-[var(--q-on-ink)]">
                                                    {i + 1}
                                                </span>
                                                <h3 className="mb-1 text-[13px] font-medium text-[var(--q-text)]">{step.title}</h3>
                                                <p className="text-xs leading-relaxed text-[var(--q-text-muted)]">{step.body}</p>
                                            </li>
                                        ))}
                                    </ol>
                                </section>

                                {/* Example questions */}
                                <section>
                                    <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                                        <SectionTitle icon={MessageSquareText}>Example questions</SectionTitle>
                                        <button type="button" onClick={() => router.push("/gallery")} className="q-pill mb-4 h-8 px-3 text-xs">
                                            All {RECIPES.length} in the Recipe Gallery
                                            <ArrowUpRight className="h-3.5 w-3.5" strokeWidth={1.75} />
                                        </button>
                                    </div>
                                    <p className="-mt-2 mb-4 text-xs text-[var(--q-text-faint)]">
                                        These come from Quasar&apos;s benchmark suites. Click one to open it in the chat, ready to send.
                                    </p>
                                    <div className="grid grid-cols-1 gap-x-6 gap-y-5 md:grid-cols-2">
                                        {EXAMPLE_GROUPS.map((group) => (
                                            <div key={group.label}>
                                                <h3 className="mb-2 px-1 text-[11px] font-medium uppercase tracking-wider text-[var(--q-text-faint)]">{group.label}</h3>
                                                <div className="space-y-2">
                                                    {group.ids.map((id) => {
                                                        const prompt = PROMPTS_BY_ID.get(id);
                                                        if (!prompt) return null;
                                                        return (
                                                            <button
                                                                key={id}
                                                                type="button"
                                                                onClick={() => tryPrompt(prompt)}
                                                                className="glass-card group flex w-full items-start gap-3 rounded-xl px-3.5 py-3 text-left"
                                                            >
                                                                <span className="flex-1 text-[12.5px] leading-relaxed text-[var(--q-text-secondary)]">{prompt}</span>
                                                                <Play className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--q-text-faint)] transition-colors group-hover:text-[var(--q-text)]" strokeWidth={1.75} />
                                                            </button>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </section>

                                {/* Features */}
                                <section>
                                    <SectionTitle icon={LayoutGrid}>What Quasar can do</SectionTitle>
                                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                                        {FEATURES.map(({ icon: Icon, title, body }) => (
                                            <div key={title} className="glass-card flex gap-3 rounded-2xl p-4">
                                                <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-[var(--q-canvas)] text-[var(--q-text-muted)]">
                                                    <Icon className="h-4 w-4" strokeWidth={1.75} />
                                                </span>
                                                <div className="min-w-0">
                                                    <h3 className="mb-0.5 text-[13px] font-medium text-[var(--q-text)]">{title}</h3>
                                                    <p className="text-xs leading-relaxed text-[var(--q-text-muted)]">{body}</p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </section>

                                <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
                                    {/* Data sources */}
                                    <section>
                                        <SectionTitle icon={Database}>Where the data comes from</SectionTitle>
                                        <div className="glass-card space-y-4 rounded-2xl p-5">
                                            {DATA_SOURCES.map((src) => (
                                                <div key={src.group}>
                                                    <h3 className="mb-2 text-[11px] font-medium uppercase tracking-wider text-[var(--q-text-faint)]">{src.group}</h3>
                                                    <div className="flex flex-wrap gap-1.5">
                                                        {src.items.map((item) => (
                                                            <span key={item} className="q-tag">{item}</span>
                                                        ))}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </section>

                                    {/* Models + tips */}
                                    <div className="space-y-8">
                                        <section>
                                            <SectionTitle icon={Sparkles}>Models</SectionTitle>
                                            <div className="glass-card space-y-2.5 rounded-2xl p-5 text-[13px] leading-relaxed text-[var(--q-text-secondary)]">
                                                <p>
                                                    The default model is gpt-oss-120b, run at the Texas Advanced Computing Center (TACC).
                                                    Switch models with the model menu at the bottom of the sidebar panel.
                                                </p>
                                                <p>
                                                    Models Quasar provides come with a weekly token allowance per account. Add your own
                                                    DeepSeek, OpenAI, Anthropic or Google key in Settings &gt; Provider Keys to use those
                                                    models on your own account.
                                                </p>
                                            </div>
                                        </section>
                                        <section>
                                            <SectionTitle icon={Star}>Tips for better answers</SectionTitle>
                                            <ul className="glass-card space-y-2 rounded-2xl p-5">
                                                {TIPS.map((tip) => (
                                                    <li key={tip} className="flex gap-2.5 text-[13px] leading-relaxed text-[var(--q-text-secondary)]">
                                                        <span className="mt-[9px] size-1 shrink-0 rounded-full bg-[var(--q-text-faint)]" />
                                                        {tip}
                                                    </li>
                                                ))}
                                            </ul>
                                        </section>
                                    </div>
                                </div>

                                {/* Resources */}
                                <section>
                                    <SectionTitle icon={ExternalLink}>Official resources</SectionTitle>
                                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                                        {RESOURCES.map(({ href, title, sub, icon: Icon }) => (
                                            <a
                                                key={href}
                                                href={href}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="glass-card group flex items-center gap-3 rounded-2xl p-4"
                                            >
                                                <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-[var(--q-canvas)] text-[var(--q-text-muted)]">
                                                    <Icon className="h-4 w-4" strokeWidth={1.75} />
                                                </span>
                                                <div className="min-w-0 flex-1">
                                                    <span className="block text-[13px] font-medium text-[var(--q-text)]">{title}</span>
                                                    <span className="block truncate text-xs text-[var(--q-text-muted)]">{sub}</span>
                                                </div>
                                                <ArrowUpRight className="h-4 w-4 shrink-0 text-[var(--q-text-faint)] transition-colors group-hover:text-[var(--q-text)]" strokeWidth={1.75} />
                                            </a>
                                        ))}
                                    </div>
                                </section>
                            </div>
                        )}

                        {activeTab === "terms" && (
                            <div className="max-w-3xl space-y-6 animate-in fade-in duration-300">
                                <p className="text-[13px] text-[var(--q-text-faint)]">
                                    Last updated: {TERMS_UPDATED}.{" "}
                                    <a href="/terms" target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-[var(--q-text)]">
                                        Open as a full page
                                    </a>
                                </p>
                                <TermsContent />
                            </div>
                        )}

                        {/* Footer */}
                        <footer className="border-t border-[var(--q-border)] pb-4 pt-10 text-center">
                            <p className="text-[12px] font-medium uppercase tracking-wider text-[var(--q-text-muted)]">Quasar Research Assistant v2.0</p>
                            <p className="mt-1.5 text-xs text-[var(--q-text-faint)]">
                                Built for astronomers. Supported by the NSF-Simons AI Institute for Cosmic Origins (CosmicAI).
                            </p>
                            <PartnerLogos className="mt-8" />
                        </footer>
                    </div>
                </div>
            </main>

            <AuthModal />
        </div>
    );
}
