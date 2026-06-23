"use client";

import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { useChatStore } from "../lib/store";
import { AuthModal } from "@/components/AuthModal";
import { OnboardingOverlay, useShowOnboarding } from "@/components/OnboardingOverlay";
import { useAuthStore } from "../lib/auth-store";
import { useEffect, useState } from "react";
import { Lock, Compass, FileText, Activity, Sparkles, LogIn } from "lucide-react";

export default function Home() {
  const sidebarOpen = useChatStore((s) => s.sidebarOpen);
  const toggleSidebar = useChatStore((s) => s.toggleSidebar);
  const { isAuthenticated, openAuthModal } = useAuthStore();
  const [mounted, setMounted] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [showOnboarding, dismissOnboarding] = useShowOnboarding();

  useEffect(() => {
    // Client hydration state is intentionally established after the first render.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);

    const mq = window.matchMedia("(max-width: 767px)");
    const handleChange = (e: MediaQueryListEvent | MediaQueryList) => {
      setIsMobile(e.matches);
      if (e.matches && useChatStore.getState().sidebarOpen) {
        useChatStore.getState().toggleSidebar();
      }
    };
    handleChange(mq);
    mq.addEventListener("change", handleChange);
    return () => mq.removeEventListener("change", handleChange);
  }, []);

  // Proactively open Auth Modal on initial mount if guest tries to view landing page
  useEffect(() => {
    if (mounted && !isAuthenticated) {
      openAuthModal();
    }
  }, [mounted, isAuthenticated, openAuthModal]);

  if (!mounted) return null; // Prevent hydration mismatch Flash

  // ── Authentication Lock Screen / Landing Gate ──
  if (!isAuthenticated) {
    return (
      <div className="relative min-h-screen w-full flex flex-col items-center justify-center p-6 overflow-hidden bg-[#070510]">
        {/* Cosmic Background Gradients */}
        <div 
          className="absolute top-[-20%] right-[-10%] w-[800px] h-[800px] pointer-events-none opacity-30 select-none animate-pulse duration-[10s]" 
          style={{ background: 'radial-gradient(circle, rgba(168,85,247,0.15) 0%, rgba(168,85,247,0) 70%)' }} 
        />
        <div 
          className="absolute bottom-[-20%] left-[-10%] w-[800px] h-[800px] pointer-events-none opacity-30 select-none animate-pulse duration-[8s]" 
          style={{ background: 'radial-gradient(circle, rgba(244,113,181,0.1) 0%, rgba(244,113,181,0) 70%)' }} 
        />

        {/* Floating star sparks mock */}
        <div className="absolute inset-0 bg-[url('/stars_pattern.png')] bg-repeat opacity-20 pointer-events-none select-none" />

        {/* Central Card */}
        <div className="relative z-10 w-full max-w-[850px] flex flex-col items-center text-center space-y-8 animate-in fade-in zoom-in-95 duration-500">
          
          {/* System Badge */}
          <div className="glass-control flex items-center gap-2 px-4 py-1.5 rounded-full text-slate-300 text-xs font-semibold uppercase tracking-widest">
            <Lock className="w-3.5 h-3.5 text-purple-400" />
            Strict Access Control
          </div>

          {/* Heading */}
          <div className="space-y-4">
            <h1 className="text-5xl md:text-6xl text-white font-[400] font-serif tracking-tight leading-none">
              QUASAR
            </h1>
            <p className="text-base md:text-lg text-slate-400 max-w-[600px] mx-auto font-sans font-light">
              An AI-Powered Research Assistant for Advanced Astronomy Workflows & Multi-Archive Radio Observations.
            </p>
          </div>

          {/* Central Call-to-action */}
          <div className="flex flex-col items-center space-y-4">
            <button
              onClick={openAuthModal}
              className="flex items-center gap-3 bg-white hover:bg-slate-200 text-black font-semibold text-base py-4 px-10 rounded-2xl shadow-xl shadow-purple-950/20 transition-all duration-300 transform hover:scale-[1.02] group"
            >
              <LogIn className="w-5 h-5 group-hover:translate-x-0.5 transition-transform" />
              Sign In to Workspace
            </button>
            <p className="text-xs text-slate-500">
              Only authorized investigators can access conversation and tool pipelines.
            </p>
          </div>

          {/* Feature Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full pt-10 border-t border-slate-900/60">
            {/* Card 1 */}
            <div className="glass-card flex flex-col items-center md:items-start text-center md:text-left p-5 rounded-2xl">
              <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400 mb-4 border border-purple-500/10">
                <Compass className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Conductor DAG Engine</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Decomposes complex research prompts into parallel processing steps to query CADC & ALMA science archives automatically.
              </p>
            </div>

            {/* Card 2 */}
            <div className="glass-card flex flex-col items-center md:items-start text-center md:text-left p-5 rounded-2xl">
              <div className="p-3 rounded-xl bg-fuchsia-500/10 text-fuchsia-300 mb-4 border border-fuchsia-500/10">
                <FileText className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">FITS Processing</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Upload images or raw tables to analyze spectral line coverage, CO isotope maps, and resolve Simbad coordinates instantly.
              </p>
            </div>

            {/* Card 3 */}
            <div className="glass-card flex flex-col items-center md:items-start text-center md:text-left p-5 rounded-2xl">
              <div className="p-3 rounded-xl bg-indigo-500/10 text-indigo-400 mb-4 border border-indigo-500/10">
                <Activity className="w-5 h-5" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">Observability & Privacy</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Full end-to-end tracing via Langfuse with secure encryption. You control your query histories and database persistence.
              </p>
            </div>
          </div>

          {/* Legal link footer */}
          <div className="text-[11px] text-slate-600 pt-6">
            By signing in, you agree to our{" "}
            <a href="/terms" className="underline hover:text-slate-400 transition-colors">Terms & Conditions & Privacy Policy</a>.
          </div>
        </div>

        {/* Global Auth Modal Overlay */}
        <AuthModal />
      </div>
    );
  }

  // ── Authorized Workspace Layout ──
  return (
    <>
      {/* ── MOBILE: sidebar as a full-screen overlay ── */}
      {isMobile && sidebarOpen && (
        <div className="fixed inset-0 z-40 flex">
          {/* Dark backdrop — click to close */}
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-lg z-40"
            onClick={toggleSidebar}
          />
          {/* Sidebar panel */}
          <div className="relative z-50 w-[var(--q-sidebar-width)] h-full animate-in slide-in-from-left duration-200">
            <Sidebar onToggle={toggleSidebar} />
          </div>
        </div>
      )}

      {/* ── DESKTOP: sidebar as a push panel ── */}
      {!isMobile && (
        <div className={`${sidebarOpen ? "w-[var(--q-sidebar-width)]" : "w-[var(--q-sidebar-rail-width)]"} transition-all duration-300 shrink-0 overflow-hidden`}>
          <Sidebar collapsed={!sidebarOpen} onToggle={toggleSidebar} />
        </div>
      )}

      <ChatArea />

      {/* Auth Modal Overlay */}
      <AuthModal />

      {/* Onboarding Tutorial — first visit only */}
      {showOnboarding && <OnboardingOverlay onComplete={dismissOnboarding} />}

      {/* Radial ambient glow gradients */}
      <div 
        className="fixed top-[-10%] right-[-5%] w-[500px] h-[500px] pointer-events-none" 
        style={{ background: 'radial-gradient(circle, rgba(244,113,181,0.05) 0%, rgba(244,113,181,0) 70%)' }} 
      />
      <div 
        className="fixed bottom-[-10%] left-[-5%] w-[600px] h-[600px] pointer-events-none" 
        style={{ background: 'radial-gradient(circle, rgba(168,85,247,0.05) 0%, rgba(168,85,247,0) 70%)' }} 
      />
    </>
  );
}
