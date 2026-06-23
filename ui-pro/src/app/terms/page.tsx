"use client";

import { ArrowLeft, Shield, Eye, Database, Activity, FileText, Trash2, HelpCircle } from "lucide-react";
import Link from "next/link";

export default function TermsPage() {
  const sections = [
    { id: "intro", title: "1. Using Quasar", icon: Shield },
    { id: "collection", title: "2. What We Collect", icon: Eye },
    { id: "processing", title: "3. How We Process Data", icon: Activity },
    { id: "fits-policy", title: "4. Your Data & Discoveries", icon: FileText },
    { id: "storage", title: "5. Keeping Your History", icon: Database },
    { id: "control", title: "6. You Control Your History", icon: Trash2 },
    { id: "support", title: "7. Support", icon: HelpCircle },
  ];

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen w-full overflow-y-auto bg-[var(--q-bg)] text-slate-300 font-sans relative flex flex-col">
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
      <header className="sticky top-0 z-30 w-full bg-[var(--q-bg)]/80 backdrop-blur-md border-b border-slate-900/60 px-6 py-4 flex items-center justify-between">
        <Link 
          href="/" 
          className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors group"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
          <span>Back to Workspace</span>
        </Link>
        <div className="flex items-center gap-2">
          <span className="font-serif text-white font-medium tracking-wider text-sm">QUASAR PROTOCOL</span>
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
              Terms & Conditions
            </h1>
            <p className="text-sm text-slate-500">
              Last Updated: May 24, 2026
            </p>
            <p className="text-slate-400 leading-relaxed text-sm md:text-base">
              Welcome to QUASAR! We built this platform to make astronomical research faster, simpler, and more accessible. Here is a plain and simple summary of how we handle your data and account access. By using Quasar, you agree to these simple terms.
            </p>
          </div>

          {/* Section: Scope */}
          <section id="intro" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/10">
                <Shield className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">1. Using Quasar</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                You are completely free to use Quasar and all of its associated features to explore the cosmos.
              </p>
              <p>
                <strong>No Guest Access:</strong> To keep our services running fast and prevent abuse, guest access is disabled. You will need to log in or create an account to access Quasar.
              </p>
              <p>
                <strong>Age Requirements &amp; Protection of Minors:</strong> You must be at least 13 years of age to use the Service. We do not knowingly collect, store, or process personal data from children under the age of 13. If we become aware that an account belongs to an individual under 13, we will immediately terminate the account and permanently purge all associated data from our systems.
              </p>
            </div>
          </section>

          {/* Section: Data Collection */}
          <section id="collection" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-fuchsia-500/10 text-fuchsia-300 border border-fuchsia-500/10">
                <Eye className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">2. What We Collect</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-4 text-sm leading-relaxed text-slate-400">
              <p>
                We only collect what is needed to make Quasar work and save your chats. This includes:
              </p>
              <ul className="list-disc pl-5 space-y-2">
                <li>
                  <strong className="text-slate-200">Your Chats:</strong> Prompt histories, research outlines, and responses so you don&apos;t lose your work.
                </li>
                <li>
                  <strong className="text-slate-200">Your Searches:</strong> Coordinate lookups and target names you search.
                </li>
                <li>
                  <strong className="text-slate-200">Your Uploaded Files:</strong> Metadata from files you upload to analyze.
                </li>
                <li>
                  <strong className="text-slate-200">Session Info:</strong> Basic connection details to keep your session stable.
                </li>
              </ul>
            </div>
          </section>

          {/* Section: How We Process Data */}
          <section id="processing" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/10">
                <Activity className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">3. How We Process Data</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                To keep Quasar fast and help us fix bugs when things break, your queries are traced for performance metrics, and your chat history is stored securely in our database.
              </p>
              <p>
                <strong>Training & Improvement:</strong> Any data you share with Quasar—including queries, conversations, uploaded documents, and search parameters—may be used to train, fine-tune, or improve Quasar and its associated AI models. This helps us build better astronomical reasoning and improve accuracy for everyone.
              </p>
            </div>
          </section>

          {/* Section: Your Data & Discoveries */}
          <section id="fits-policy" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-fuchsia-500/10 text-fuchsia-300 border border-fuchsia-500/10">
                <FileText className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">4. Your Data & Discoveries</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                We treat your research with complete respect:
              </p>
              <p>
                <strong>Your Discoveries are Yours:</strong> Quasar is just an assistant. We claim zero ownership or rights over your searches, files, or findings. Your science is yours, period.
              </p>
              <p>
                <strong>Privacy:</strong> Your uploaded files and results are processed securely, and other users cannot see them.
              </p>
            </div>
          </section>

          {/* Section: Storage */}
          <section id="storage" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-teal-500/10 text-teal-400 border border-teal-500/10">
                <Database className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">5. Keeping Your History</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                Your chat history stays saved so you don&apos;t lose your work. If you log out, your browser session is safely cleared.
              </p>
            </div>
          </section>

          {/* Section: User Control */}
          <section id="control" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-red-500/10 text-red-400 border border-red-500/10">
                <Trash2 className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">6. You Control Your History</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                You have full control over your scientific search history and profile:
              </p>
              <p>
                <strong>Wiping History:</strong> You can delete individual chat threads from the sidebar at any time, or request a complete wipe of your account and history in your Settings. Once deleted, it&apos;s permanently gone.
              </p>
            </div>
          </section>

          {/* Section: Support */}
          <section id="support" className="space-y-4 scroll-mt-24">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/10">
                <HelpCircle className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-serif text-white font-[400]">7. Support</h2>
            </div>
            <div className="bg-slate-950/30 border border-slate-900 rounded-2xl p-6 space-y-3 text-sm leading-relaxed text-slate-400">
              <p>
                Got questions? Feel free to check our logs or reach out to the project team.
              </p>
              <p className="text-xs text-slate-500 pt-2 border-t border-slate-900">
                QUASAR Observatory — Astronomy Orchestration and Data Security.
              </p>
            </div>
          </section>
        </main>
      </div>

      {/* Footer */}
      <footer className="w-full bg-[var(--q-bg)] border-t border-slate-900/60 py-6 text-center text-xs text-slate-500 mt-auto">
        &copy; {new Date().getFullYear()} Quasar Observatory. All rights reserved.
      </footer>
    </div>
  );
}
