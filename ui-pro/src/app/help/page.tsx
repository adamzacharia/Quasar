"use client";

import { Sidebar } from "@/components/Sidebar";
import { useChatStore } from "../../lib/store";
import { AuthModal } from "@/components/AuthModal";
import { useEffect, useState } from "react";
import { Telescope, FileText, Search, Zap, Github, BookOpen, ExternalLink, Bot, X, Shield, Users, Heart } from "lucide-react";
import Link from "next/link";

export default function HelpPage() {
    const sidebarOpen = useChatStore((s) => s.sidebarOpen);
    const [mounted, setMounted] = useState(false);
    const [activeTab, setActiveTab] = useState<"docs" | "terms" | "credits">("docs");

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

    const tabs = [
        { id: "docs" as const, label: "Documentation", icon: BookOpen },
        { id: "terms" as const, label: "Terms & Conditions", icon: Shield },
        { id: "credits" as const, label: "Credits", icon: Users },
    ];

    const teamMembers = [
        { name: "Adam Zacharia Anil", role: "Lead Developer & Creator" },
        { name: "Adele Plunkett", role: "Scientific Advisor" },
        { name: "Brian Mason", role: "Scientific Advisor" },
        { name: "Stella Offner", role: "Scientific Advisor" },
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

                <div className="max-w-5xl mx-auto space-y-10 relative">

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

                    {/* Tab Navigation */}
                    <div className="flex gap-2 p-1 bg-slate-800/50 rounded-xl border border-slate-700/50 w-fit">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                                    activeTab === tab.id
                                        ? "bg-primary/20 text-primary border border-primary/30 shadow-lg shadow-primary/10"
                                        : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                                }`}
                            >
                                <tab.icon className="w-4 h-4" />
                                {tab.label}
                            </button>
                        ))}
                    </div>

                    {/* ═══════════════ DOCS TAB ═══════════════ */}
                    {activeTab === "docs" && (
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 animate-in fade-in duration-300">
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
                    )}

                    {/* ═══════════════ TERMS & CONDITIONS TAB ═══════════════ */}
                    {activeTab === "terms" && (
                        <div className="max-w-3xl animate-in fade-in duration-300 space-y-8">
                            <section className="bg-slate-800/30 border border-slate-700/50 rounded-2xl p-8 backdrop-blur-sm">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="p-2 bg-primary/10 rounded-lg"><Shield className="w-6 h-6 text-primary" /></div>
                                    <div>
                                        <h2 className="text-2xl font-bold text-white">Terms & Conditions</h2>
                                        <p className="text-xs text-slate-500 mt-1">Last updated: April 6, 2026</p>
                                    </div>
                                </div>

                                <div className="space-y-6 text-sm text-slate-300 leading-relaxed">
                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">1. Acceptance of Terms</h3>
                                        <p>By accessing or using QUASAR (&quot;the Service&quot;), you agree to be bound by these Terms and Conditions. If you do not agree to these terms, please do not use the Service. QUASAR is an AI-powered research assistant developed for astronomical research purposes.</p>
                                    </div>

                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">2. Description of Service</h3>
                                        <p>QUASAR provides AI-assisted access to astronomical databases including the ALMA Science Archive, NASA Astrophysics Data System (ADS), and SIMBAD. The Service is designed to facilitate radio astronomy research by enabling natural language queries, data retrieval, analysis, and visualization.</p>
                                    </div>

                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">3. Use of Service</h3>
                                        <p>The Service is provided for academic and scientific research purposes. You agree to use the Service only for lawful purposes and in accordance with community standards for scientific research. You shall not use the Service to:</p>
                                        <ul className="list-disc list-inside mt-2 space-y-1 text-slate-400">
                                            <li>Circumvent access controls on underlying data archives</li>
                                            <li>Misrepresent AI-generated analysis as peer-reviewed findings</li>
                                            <li>Submit excessive or automated bulk queries that degrade service quality</li>
                                            <li>Use the Service in violation of data use policies of upstream archives (ALMA, NASA ADS, SIMBAD)</li>
                                        </ul>
                                    </div>

                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">4. Data & Privacy</h3>
                                        <p>QUASAR processes queries using third-party AI models and retrieves data from public astronomical archives. While we do not intentionally collect personal data, queries and conversation history may be temporarily stored to improve service quality. We do not sell or share your data with third parties for commercial purposes.</p>
                                    </div>

                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">5. AI-Generated Content Disclaimer</h3>
                                        <p>QUASAR uses large language models to generate responses. While the system is designed to provide accurate scientific information, AI-generated outputs may contain errors, omissions, or inaccuracies. All results should be independently verified before use in publications or scientific conclusions. The Service does not replace professional scientific judgment.</p>
                                    </div>

                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">6. Intellectual Property</h3>
                                        <p>Astronomical data retrieved through QUASAR is subject to the data use and citation policies of the originating archives (ALMA, NASA ADS, SIMBAD). Users are responsible for proper attribution when using data obtained through the Service in publications or presentations.</p>
                                    </div>

                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">7. Limitation of Liability</h3>
                                        <p>The Service is provided &quot;as is&quot; without warranties of any kind. We shall not be liable for any direct, indirect, incidental, special, or consequential damages arising from your use of the Service, including but not limited to loss of data, incorrect analysis results, or service interruptions.</p>
                                    </div>

                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">8. Modifications</h3>
                                        <p>We reserve the right to modify these Terms at any time. Continued use of the Service after changes constitutes acceptance of the updated Terms. Users will be notified of material changes through the Service interface.</p>
                                    </div>

                                    <div>
                                        <h3 className="text-base font-semibold text-white mb-2">9. Contact</h3>
                                        <p>For questions about these Terms, please reach out via the <a href="https://github.com/adamzacharia/Quasar2" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">GitHub repository</a>.</p>
                                    </div>
                                </div>
                            </section>
                        </div>
                    )}

                    {/* ═══════════════ CREDITS TAB ═══════════════ */}
                    {activeTab === "credits" && (
                        <div className="max-w-3xl animate-in fade-in duration-300 space-y-8">
                            {/* Funding Section */}
                            <section className="bg-gradient-to-br from-primary/5 via-slate-800/30 to-accent-purple/5 border border-primary/20 rounded-2xl p-8 backdrop-blur-sm">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="p-2 bg-primary/10 rounded-lg"><Heart className="w-6 h-6 text-primary" /></div>
                                    <h2 className="text-2xl font-bold text-white">Funding & Support</h2>
                                </div>
                                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
                                    <p className="text-slate-300 leading-relaxed">
                                        QUASAR is supported by a <span className="text-white font-semibold">Seed Fund</span> from the{" "}
                                        <span className="text-primary font-semibold">CosmicAI</span>{" "}
                                        <span className="text-white font-semibold">NSF–Simons Foundation</span>.
                                    </p>
                                    <div className="flex items-center gap-4 mt-4 pt-4 border-t border-slate-700/50">
                                        <div className="flex gap-1">
                                            {[...Array(5)].map((_, i) => (
                                                <div key={i} className="w-1.5 h-1.5 rounded-full bg-primary/60 animate-pulse" style={{ animationDelay: `${i * 200}ms` }} />
                                            ))}
                                        </div>
                                        <span className="text-xs text-slate-500 uppercase tracking-wider font-medium">National Science Foundation &bull; Simons Foundation</span>
                                    </div>
                                </div>
                            </section>

                            {/* Team Section */}
                            <section className="bg-slate-800/30 border border-slate-700/50 rounded-2xl p-8 backdrop-blur-sm">
                                <div className="flex items-center gap-3 mb-6">
                                    <div className="p-2 bg-accent-purple/10 rounded-lg"><Users className="w-6 h-6 text-accent-purple" /></div>
                                    <h2 className="text-2xl font-bold text-white">Team</h2>
                                </div>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                    {teamMembers.map((member) => (
                                        <div key={member.name} className="flex items-center gap-4 bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 hover:border-slate-600/50 transition-colors">
                                            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary/30 to-accent-purple/30 border border-slate-600/50 flex items-center justify-center shrink-0">
                                                <span className="text-lg font-bold text-white">{member.name.split(" ").map(n => n[0]).join("").slice(0, 2)}</span>
                                            </div>
                                            <div>
                                                <p className="text-sm font-semibold text-white">{member.name}</p>
                                                <p className="text-xs text-slate-400 mt-0.5">{member.role}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </section>

                            {/* Acknowledgements */}
                            <section className="bg-slate-800/30 border border-slate-700/50 rounded-2xl p-8 backdrop-blur-sm">
                                <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                                    <div className="p-1.5 bg-slate-700/50 rounded-md"><BookOpen className="w-5 h-5 text-blue-400" /></div>
                                    Acknowledgements
                                </h2>
                                <div className="space-y-3 text-sm text-slate-400 leading-relaxed">
                                    <p>This project makes use of data from the <span className="text-slate-300">Atacama Large Millimeter/submillimeter Array (ALMA)</span>, <span className="text-slate-300">NASA Astrophysics Data System (ADS)</span>, and <span className="text-slate-300">SIMBAD</span> astronomical database operated at CDS, Strasbourg, France.</p>
                                    <p>ALMA is a partnership of ESO, NSF (USA), and NINS (Japan), together with NRC (Canada), NSTC and ASIAA (Taiwan), and KASI (Republic of Korea), in cooperation with the Republic of Chile.</p>
                                    <p>Special thanks to <span className="text-slate-300 font-medium">Dr Nikhil Mukund</span> for inspiring me to build tools and teaching me so much about LLMs, and to <span className="text-slate-300 font-medium">Dr Lisa Barsotti</span> and the <span className="text-slate-300 font-medium">MIT LIGO Lab</span> for giving invaluable opportunity and support.</p>
                                </div>
                            </section>
                        </div>
                    )}

                    <div className="pt-8 text-center border-t border-slate-800 pb-12">
                        <p className="text-sm text-slate-500 font-medium tracking-wide">QUASAR RESEARCH ASSISTANT v2.0</p>
                        <p className="text-xs text-slate-600 mt-2">Built for Radio Astronomers. Supported by CosmicAI NSF–Simons Foundation.</p>
                    </div>

                </div>
            </main>

            {/* Auth Modal Overlay */}
            <AuthModal />
        </div>
    );
}
