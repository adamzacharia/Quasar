"use client";

import { Sidebar } from "@/components/Sidebar";
import { useChatStore } from "@/lib/store";
import { AuthModal } from "@/components/AuthModal";
import { useEffect, useState } from "react";
import { Telescope, FileText, Search, Zap, Github, BookOpen, ExternalLink, Bot, X } from "lucide-react";
import Link from "next/link";

export default function HelpPage() {
    const sidebarOpen = useChatStore((s) => s.sidebarOpen);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setMounted(true);
    }, []);

    if (!mounted) return null; // Prevent hydration mismatch Flash

    const commands = [
        { tag: "@archive", desc: "Search ALMA observation archives for RAW data cubes and visibilities.", icon: Telescope, color: "text-primary", bg: "bg-primary/10" },
        { tag: "@paper", desc: "Search NASA ADS for research papers, authors, and abstracts.", icon: FileText, color: "text-accent-purple", bg: "bg-accent-purple/10" },
        { tag: "@search", desc: "General search across all astrophysical databases and sources.", icon: Search, color: "text-emerald-accent", bg: "bg-emerald-accent/10" },
    ];

    const tips = [
        `Ask natural questions like "Find ALMA observations of HL Tau"`,
        `Use filter commands: "Show results with resolution < 0.05 arcsec"`,
        `Request plots: "Plot the sky distribution of results"`,
        `Search papers: "Find papers about protoplanetary disks"`,
        `Resolve targets: "Where is RXJ1347-1145?"`,
    ];

    const tools = [
        "search_by_target", "search_by_position", "search_by_frequency",
        "search_papers", "filter_results", "plot_alma_results",
        "resolve_target", "check_co_lines", "download_alma_data",
    ];

    return (
        <div className="flex h-screen w-full bg-[#0a0f1c] text-slate-300 overflow-hidden font-sans selection:bg-primary/30">
            {/* Sidebar */}
            <div className={`${sidebarOpen ? "w-[280px]" : "w-0"} transition-all duration-300 shrink-0 overflow-hidden z-20 bg-sidebar-dark`}>
                <Sidebar />
            </div>

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto relative p-8 md:p-12 lg:p-16 z-10">
                {/* Close Button to return to Chat */}
                <Link href="/" className="absolute top-6 right-6 md:top-8 md:right-8 p-3 bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-white rounded-full transition-all shadow-lg border border-slate-700/50 z-50 group">
                    <X className="w-6 h-6 transition-transform group-hover:scale-110" />
                </Link>

                {/* Background Decorators */}
                <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl pointer-events-none" />
                <div className="absolute bottom-[-10%] left-[-5%] w-[600px] h-[600px] bg-accent-purple/5 rounded-full blur-3xl pointer-events-none" />

                <div className="max-w-5xl mx-auto space-y-16 relative">

                    {/* Header */}
                    <div className="space-y-4">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider">
                            <BookOpen className="w-4 h-4" /> Documentation
                        </div>
                        <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
                            Help & Docs
                        </h1>
                        <p className="text-lg text-slate-400 max-w-2xl leading-relaxed">
                            QUASAR is your AI-powered research assistant for radio astronomy.
                            It connects directly to the ALMA Science Archive, NASA ADS, and SIMBAD to help you
                            find, analyze, and explore astronomical data through natural conversation.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Left Column */}
                        <div className="space-y-8">
                            {/* Command Tags Section */}
                            <section className="bg-slate-800/30 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
                                <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                                    <div className="p-1.5 bg-slate-700/50 rounded-md"><Zap className="w-5 h-5 text-yellow-400" /></div>
                                    Command Tags
                                </h2>
                                <div className="space-y-4">
                                    {commands.map((cmd) => (
                                        <div key={cmd.tag} className="flex gap-4 p-4 rounded-xl bg-slate-800/50 hover:bg-slate-800 transition-colors border border-slate-700/50 hover:border-slate-600/50">
                                            <div className={`p-3 rounded-lg ${cmd.bg} shrink-0 h-fit`}>
                                                <cmd.icon className={`w-6 h-6 ${cmd.color}`} />
                                            </div>
                                            <div>
                                                <span className="text-sm font-mono font-bold text-white">{cmd.tag}</span>
                                                <p className="text-sm text-slate-400 mt-1 leading-relaxed">{cmd.desc}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </section>

                            {/* Pro Tips Section */}
                            <section className="bg-slate-800/30 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
                                <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                                    <div className="p-1.5 bg-slate-700/50 rounded-md"><span className="text-xl">💡</span></div>
                                    Example Queries
                                </h2>
                                <ul className="space-y-3">
                                    {tips.map((tip, i) => (
                                        <li key={i} className="flex items-start gap-3 bg-slate-800/50 p-3 rounded-lg border border-slate-700/30">
                                            <span className="text-primary font-bold mt-0.5">•</span>
                                            <span className="text-sm text-slate-300 font-medium">&quot;{tip}&quot;</span>
                                        </li>
                                    ))}
                                </ul>
                            </section>
                        </div>

                        {/* Right Column */}
                        <div className="space-y-8">
                            {/* Link Resources Section */}
                            <section className="bg-slate-800/30 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
                                <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                                    <div className="p-1.5 bg-slate-700/50 rounded-md"><ExternalLink className="w-5 h-5 text-blue-400" /></div>
                                    Official Resources
                                </h2>
                                <div className="space-y-3">
                                    <a href="https://github.com/adamzacharia/Quasar2" target="_blank" rel="noopener noreferrer"
                                        className="flex items-center gap-4 bg-slate-800/50 hover:bg-slate-700 border border-slate-700/50 rounded-xl p-4 transition-all hover:scale-[1.02] group">
                                        <div className="p-2 bg-slate-700 rounded-lg group-hover:bg-slate-600 transition-colors"><Github className="w-5 h-5 text-white" /></div>
                                        <div className="flex-1">
                                            <span className="text-sm font-semibold text-white">GitHub Repository</span>
                                            <p className="text-xs text-slate-400">Source code & technical docs</p>
                                        </div>
                                        <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-primary transition-colors" />
                                    </a>
                                    <a href="https://almascience.nrao.edu/aq/" target="_blank" rel="noopener noreferrer"
                                        className="flex items-center gap-4 bg-slate-800/50 hover:bg-slate-700 border border-slate-700/50 rounded-xl p-4 transition-all hover:scale-[1.02] group">
                                        <div className="p-2 bg-blue-900/40 rounded-lg group-hover:bg-blue-800/50 transition-colors"><Telescope className="w-5 h-5 text-blue-400" /></div>
                                        <div className="flex-1">
                                            <span className="text-sm font-semibold text-white">ALMA Archive</span>
                                            <p className="text-xs text-slate-400">Official Science Archive Search</p>
                                        </div>
                                        <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-primary transition-colors" />
                                    </a>
                                    <a href="https://ui.adsabs.harvard.edu/" target="_blank" rel="noopener noreferrer"
                                        className="flex items-center gap-4 bg-slate-800/50 hover:bg-slate-700 border border-slate-700/50 rounded-xl p-4 transition-all hover:scale-[1.02] group">
                                        <div className="p-2 bg-accent-purple/20 rounded-lg group-hover:bg-accent-purple/30 transition-colors"><Search className="w-5 h-5 text-accent-purple" /></div>
                                        <div className="flex-1">
                                            <span className="text-sm font-semibold text-white">NASA ADS</span>
                                            <p className="text-xs text-slate-400">Astrophysics Data System</p>
                                        </div>
                                        <ExternalLink className="w-4 h-4 text-slate-500 group-hover:text-primary transition-colors" />
                                    </a>
                                </div>
                            </section>

                            {/* Available Tools Reference */}
                            <section className="bg-slate-800/30 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
                                <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                                    <div className="p-1.5 bg-slate-700/50 rounded-md"><Bot className="w-5 h-5 text-emerald-accent" /></div>
                                    Agent Tools Arsenal
                                </h2>
                                <p className="text-sm text-slate-400 mb-4">The QUASAR agent autonomously routes requests to these Python tools:</p>
                                <div className="flex flex-wrap gap-2">
                                    {tools.map((t) => (
                                        <div key={t} className="px-3 py-1.5 bg-[#111] border border-slate-700/50 rounded-md flex items-center gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-accent/60 animate-pulse" />
                                            <span className="text-xs font-mono text-slate-300">{t}</span>
                                        </div>
                                    ))}
                                </div>
                            </section>
                        </div>
                    </div>

                    <div className="pt-8 text-center border-t border-slate-800 pb-12">
                        <p className="text-sm text-slate-500 font-medium tracking-wide">QUASAR RESEARCH ASSISTANT v2.0</p>
                        <p className="text-xs text-slate-600 mt-2">Built for Radio Astronomers.</p>
                    </div>

                </div>
            </main>

            {/* Auth Modal Overlay */}
            <AuthModal />
        </div>
    );
}
