"use client";

import { ArrowLeft, Shield, Eye, Database, Activity, FileText, Trash2, HelpCircle } from "lucide-react";
import Link from "next/link";

export default function TermsPage() {
  const sections = [
    { id: "intro", title: "1. Scope & Access", icon: Shield },
    { id: "collection", title: "2. What We Collect", icon: Eye },
    { id: "processing", title: "3. Tech Stack & Tracing", icon: Activity },
    { id: "fits-policy", title: "4. Your FITS & Science Data", icon: FileText },
    { id: "storage", title: "5. Storage & Continuity", icon: Database },
    { id: "control", title: "6. You Own Your Data", icon: Trash2 },
    { id: "support", title: "7. Questions & Support", icon: HelpCircle },
  ];

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen w-full overflow-y-auto bg-[#0d0d0e] text-slate-300 font-sans relative flex flex-col">
      {/* Cosmic Background Gradients */}
      <div 
        className="absolute top-[-10%] right-[-10%] w-[600px] h-[600px] pointer-events-none opacity-20 select-none" 
        style={{ background: 'radial-gradient(circle, rgba(168,85,247,0.1) 0%, rgba(168,85,247,0) 70%)' }} 
      />
      <div 
        className="absolute bottom-[-10%] left-[-10%] w-[600px] h-[600px] pointer-events-none opacity-20 select-none" 
        style={{ background: 'radial-gradient(circle, rgba(244,113,181,0.08) 0%, rgba(244,113,181,0) 70%)' }} 
      />
      
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-30 w-full bg-[#0d0d0e]/80 backdrop-blur-md border-b border-slate-900/60 px-6 py-4 flex items-center justify-between">
        <Link 
          href="/" 
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors group"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
          <span>Back to Workspace</span>
        </Link>
        <div className="flex items-center gap-2">
          <span className="font-serif text-white font-medium tracking-wider text-sm">QUASAR SYSTEM PROTOCOL</span>
        </div>
      </header>

      {/* Main Container */}
      <div className="flex-1 max-w-[1200px] w-full mx-auto px-6 py-12 flex flex-col md:flex-row gap-10">
        
        {/* Sticky Sidebar Navigation (Desktop) */}
        <aside className="w-full md:w-64 shrink-0 md:sticky md:top-24 h-fit space-y-2 hidden md:block">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest px-3 mb-4">Navigation</p>
          {sections.map((sec) => {
            const Icon = sec.icon;
            return (
              <button
                key={sec.id}
                onClick={() => scrollToSection(sec.id)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-950/40 border border-transparent hover:border-slate-900/60 transition-all group"
              >
                <Icon className="w-4 h-4 text-slate-500 group-hover:text-purple-400 transition-colors" />
                <span>{sec.title}</span>
              </button>
            );
          })}
        </aside>

        {/* Content Area */}
        <main className="flex-1 space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-300">
          
          {/* Header Title */}
          <div className="space-y-4 border-b border-slate-900 pb-8">
            <h1 className="text-4xl md:text-5xl text-white font-[400] font-serif tracking-tight">
              Terms of Service & Privacy Policy
            </h1>
            <p className="text-sm text-slate-500">
              Last Updated: May 24, 2026 • Version 2.0 (The human-readable edition)
            </p>
            <p className="text-slate-400 leading-relaxed text-sm md:text-base">
              Welcome to the QUASAR Astronomical Assistant! We built Quasar to make radio astronomy research faster, simpler, and actually enjoyable. Below, we maintain absolute transparency regarding data collection, operational tracing, and user security in plain English—no corporate legal jargon required. By signing up or using Quasar, you agree to these simple guidelines.
            </p>
          </div>

          {/* Section: Scope */}
          <section id="intro" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/10">
                <Shield className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">1. Scope & Access</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                You are completely free to use our natural language query interface, agent pipelines, and spectral FITS processing tools to explore the cosmos.
              </p>
              <p>
                <strong>No Guest Access:</strong> To protect our APIs, avoid spam, and keep our services fast, guest access is disabled. You must create a verified account (via Email or Google login) to access any Quasar APIs, coordinate query pipelines, or workspace buffers.
              </p>
            </div>
          </section>

          {/* Section: Data Collection */}
          <section id="collection" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-pink-500/10 text-pink-400 border border-pink-500/10">
                <Eye className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">2. What We Collect</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-4 text-sm leading-relaxed text-slate-400">
              <p>
                We only collect what's necessary to make Quasar work and keep your sessions alive. This includes:
              </p>
              <ul className="list-disc pl-5 space-y-2">
                <li>
                  <strong className="text-slate-200">Your Chats:</strong> Prompt histories, research outlines, and agent responses, so you don't lose your work.
                </li>
                <li>
                  <strong className="text-slate-200">Celestial Metadata:</strong> Coordinate lookups (RA/Dec), spectral frequencies, and target names you search.
                </li>
                <li>
                  <strong className="text-slate-200">FITS Metadata:</strong> Pixel counts, headers, and coordinate grids extracted when you analyze files.
                </li>
                <li>
                  <strong className="text-slate-200">Session Info:</strong> IP address and browser headers, strictly to keep your real-time WebSocket connection stable.
                </li>
              </ul>
            </div>
          </section>

          {/* Section: Langfuse & Turso */}
          <section id="processing" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/10">
                <Activity className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">3. Tech Stack & Tracing</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-4 text-sm leading-relaxed text-slate-400">
              <p>
                We process your queries using a clean, modern observability pipeline:
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60">
                  <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-indigo-400" />
                    Langfuse Observability
                  </h4>
                  <p className="text-[12px] leading-relaxed text-slate-400">
                    When you run a complex query (like searching ALMA or VizieR), we trace the steps in Langfuse. This logs LLM token usage and tool times so we can debug things when they break and keep everything running at peak performance.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60">
                  <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-400" />
                    Turso Thread Storage
                  </h4>
                  <p className="text-[12px] leading-relaxed text-slate-400">
                    Your profile, chat histories, and settings are stored in a distributed Turso database. This ensures your sessions load instantly no matter where in the world you are logging in from.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Section: FITS Data */}
          <section id="fits-policy" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-pink-500/10 text-pink-400 border border-pink-500/10">
                <FileText className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">4. Your FITS & Science Data</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                Flexible Image Transport System (FITS) files are the lifeblood of astronomy. We treat your science with absolute respect:
              </p>
              <p>
                <strong>Scientific Sovereignty:</strong> Quasar acts strictly as an automated processing assistant. We claim zero ownership, intellectual property rights, or publication claims over the observations, FITS images, or scientific findings you generate here. Your science is yours, period.
              </p>
              <p>
                <strong>Isolated Processing:</strong> Uploaded FITS files are processed inside isolated, temporary containers and visual plots are cached securely, meaning other users cannot see or access your data.
              </p>
            </div>
          </section>

          {/* Section: Storage */}
          <section id="storage" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/10">
                <Database className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">5. Storage & Continuity</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                We keep your chat threads safe in our database so your notes don't vanish. If you log out, your browser token is safely wiped.
              </p>
            </div>
          </section>

          {/* Section: User Control */}
          <section id="control" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-red-500/10 text-red-400 border border-red-500/10">
                <Trash2 className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">6. You Own Your Data</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                You retain complete control over your scientific search history and profile:
              </p>
              <p>
                <strong>Instant Wiping:</strong> You can delete individual chat threads from the sidebar at any time, or trigger a complete account wipe in your Settings. Once you click delete, your records are permanently purged from our Turso database. No hidden back-ups, no keeping your data forever.
              </p>
            </div>
          </section>

          {/* Section: Support */}
          <section id="support" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/10">
                <HelpCircle className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">7. Questions & Support</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                Got questions about how we handle coordinates, FITS files, or telemetry? We are astronomers and engineers—just check our repository logs or reach out to the project maintainers, and we will happily walk you through the codebase.
              </p>
              <p className="text-xs text-slate-500 pt-2 border-t border-slate-900">
                QUASAR Observatory — Astronomy Orchestration and Data Security.
              </p>
            </div>
          </section>
        </main>
      </div>

      {/* Footer */}
      <footer className="w-full bg-[#0d0d0e] border-t border-slate-900/60 py-6 text-center text-xs text-slate-500 mt-auto">
        &copy; {new Date().getFullYear()} Quasar Observatory. All rights reserved. Traced securely via Langfuse v2.
      </footer>
    </div>
  );
}
