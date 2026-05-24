"use client";

import { ArrowLeft, Shield, Eye, Database, Activity, FileText, Trash2, HelpCircle } from "lucide-react";
import Link from "next/link";

export default function TermsPage() {
  const sections = [
    { id: "intro", title: "1. Scope & System Access", icon: Shield },
    { id: "collection", title: "2. Data We Collect", icon: Eye },
    { id: "processing", title: "3. Langfuse & Turso Pipelines", icon: Activity },
    { id: "fits-policy", title: "4. Astronomical FITS Data", icon: FileText },
    { id: "storage", title: "5. Thread Continuity & Storage", icon: Database },
    { id: "control", title: "6. User Control & Deletion", icon: Trash2 },
    { id: "support", title: "7. Compliance & Support", icon: HelpCircle },
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
              Terms & Conditions & Privacy Policy
            </h1>
            <p className="text-sm text-slate-500">
              Last Updated: May 24, 2026 • Security Protocol Version 1.4.2
            </p>
            <p className="text-slate-400 leading-relaxed text-sm md:text-base">
              Welcome to the QUASAR Astronomical Assistant platform. As a public service utilized globally by professional investigators, academic institutions, and independent researchers, we maintain absolute transparency regarding data collection, operational tracing, and user security. By provisioning an account or accessing the Quasar workspaces, you explicitly consent to these security covenants.
            </p>
          </div>

          {/* Section: Scope */}
          <section id="intro" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/10">
                <Shield className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">1. Scope & System Access</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                Quasar grants authorized investigators a non-transferable, revocable license to utilize the natural language celestial query interface, observational conductor pipeline, and spectral FITS processing environments.
              </p>
              <p>
                <strong>Mandatory Account Control:</strong> Guest access is strictly prohibited. You must create and authenticate through verified credentials (email/password or SSO) to access any Quasar APIs, coordinate queries, or workspace buffers.
              </p>
            </div>
          </section>

          {/* Section: Data Collection */}
          <section id="collection" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-pink-500/10 text-pink-400 border border-pink-500/10">
                <Eye className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">2. Data We Collect</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-4 text-sm leading-relaxed text-slate-400">
              <p>
                To provide deterministic radio astronomy agent orchestration and trace complex astronomical sub-queries, Quasar collects the following parameters during system interactions:
              </p>
              <ul className="list-disc pl-5 space-y-2">
                <li>
                  <strong className="text-slate-200">Conversation Buffers:</strong> Input queries, structural research outlines, and agent output records.
                </li>
                <li>
                  <strong className="text-slate-200">Scientific Metadata:</strong> Right Ascension (RA) coordinates, declination (Dec) inputs, frequency parameters, and target designations.
                </li>
                <li>
                  <strong className="text-slate-200">FITS File Metadata:</strong> Dimensions, headers, spectral coordinates, and pixel metrics extracted from uploaded Flexible Image Transport System files.
                </li>
                <li>
                  <strong className="text-slate-200">Telemetry & Client Metrics:</strong> IP addresses, browser footprints, and active session tokens required to authenticate websocket streaming channels.
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
              <h2 className="text-xl font-serif text-white font-[400]">3. Langfuse Tracing & Turso Pipeline</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-4 text-sm leading-relaxed text-slate-400">
              <p>
                Quasar processes user inputs using advanced analytical pipelines. This telemetry is handled through two major integrations:
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60">
                  <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-indigo-400" />
                    Langfuse Observability
                  </h4>
                  <p className="text-[12px] leading-relaxed text-slate-400">
                    To maintain strict debugging pipelines and audit LLM scores, every scientific sub-query, web-search routing (Tavily/SIMBAD), and internal tool run is traced dynamically under secure spans in Langfuse. This ensures perfect observability and performance validation.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/60">
                  <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-400" />
                    Turso Thread Continuity
                  </h4>
                  <p className="text-[12px] leading-relaxed text-slate-400">
                    Your conversation indexes, workflow nodes, and user profile registries are securely stored in a globally-distributed Turso SQLite database (hosted securely with enterprise LibSQL engines), ensuring fast latency and seamless device transition.
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
              <h2 className="text-xl font-serif text-white font-[400]">4. Astronomical FITS Data Policy</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                Flexible Image Transport System (FITS) files contain sensitive astronomical and spatial research data.
              </p>
              <p>
                <strong>Scientific Confidentiality:</strong> Quasar acts purely as an automated analytical processor. We do not claim any proprietary rights, Intellectual Property, or publication claims over the observations or findings generated by processing your FITS files.
              </p>
              <p>
                <strong>Processing Environment:</strong> Uploaded FITS files are processed within dedicated production containers. Temporary visual plots and header extractions are cached securely and are completely isolated from other platform users.
              </p>
            </div>
          </section>

          {/* Section: Storage */}
          <section id="storage" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/10">
                <Database className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">5. Thread Continuity & Storage</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                To provide cohesive conversation flow, your research threads remain active inside our database indefinitely unless explicit deletion commands are executed. Platform caches and session headers are automatically scrubbed upon logout or JWT expiration.
              </p>
            </div>
          </section>

          {/* Section: User Control */}
          <section id="control" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-red-500/10 text-red-400 border border-red-500/10">
                <Trash2 className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">6. User Control & Deletion</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                We believe in absolute investigator sovereignty. You retain full control over your scientific search history:
              </p>
              <p>
                <strong>Session Clearing & Account Wipe:</strong> At any point, you can manually delete individual research threads from the Sidebar interface, or request a complete database wipe of your verified account metrics through the user dashboard settings. Once confirmed, all referenced keys, files, and conversation indexes are permanently deleted from our active Turso instances.
              </p>
            </div>
          </section>

          {/* Section: Support */}
          <section id="support" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/10">
                <HelpCircle className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">7. Compliance & Support</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                For further clarification regarding astronomical data privacy, legal tracing architectures, or integration configurations, please consult the system logs or reach out to the project maintainers.
              </p>
              <p className="text-xs text-slate-500 pt-2 border-t border-slate-900">
                QUASAR System Authority — Advanced Astronomy Orchestration and Security Registry.
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
