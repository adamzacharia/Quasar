"use client";

import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { useChatStore } from "../lib/store";
import { AuthModal } from "@/components/AuthModal";
import { OnboardingOverlay, useShowOnboarding } from "@/components/OnboardingOverlay";
import { useAuthStore } from "../lib/auth-store";
import { useIsMobile } from "../lib/use-is-mobile";
import { useEffect, useState } from "react";
import { Lock, Compass, FileText, Activity, LogIn } from "lucide-react";

export default function Home() {
  const sidebarOpen = useChatStore((s) => s.sidebarOpen);
  const toggleSidebar = useChatStore((s) => s.toggleSidebar);
  const { isAuthenticated, isInitialized, openAuthModal } = useAuthStore();
  const [mounted, setMounted] = useState(false);
  const isMobile = useIsMobile();
  const [showOnboarding, dismissOnboarding] = useShowOnboarding();

  useEffect(() => {
    // Client hydration state is intentionally established after the first render.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  // A sidebar left open on desktop must not reappear as a drawer when the
  // viewport narrows past the breakpoint.
  useEffect(() => {
    if (isMobile && useChatStore.getState().sidebarOpen) {
      useChatStore.getState().toggleSidebar();
    }
  }, [isMobile]);

  // Escape closes the drawer, matching ImageLightbox/SettingsModal.
  useEffect(() => {
    if (!isMobile || !sidebarOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") toggleSidebar();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isMobile, sidebarOpen, toggleSidebar]);

  // Proactively open Auth Modal for guests — only AFTER the cookie session has
  // been verified (/api/auth/me), so a valid-cookie reload never flashes it.
  useEffect(() => {
    if (mounted && isInitialized && !isAuthenticated) {
      openAuthModal();
    }
  }, [mounted, isInitialized, isAuthenticated, openAuthModal]);

  // Neutral gate: prevents both the hydration-mismatch flash and a premature
  // signed-out screen while the httpOnly-cookie session is being verified.
  if (!mounted || !isInitialized) return null;

  // ── Authentication Lock Screen / Landing Gate ──
  if (!isAuthenticated) {
    return (
      <div className="relative min-h-screen w-full flex flex-col items-center justify-center p-6 overflow-hidden bg-[var(--q-bg)]">
        {/* Orbita landing gate: a calm centered card on the plain shell. */}
        <div className="relative z-10 w-full max-w-[820px] flex flex-col items-center text-center space-y-8 q-rise">

          {/* System Badge */}
          <div className="q-pill h-8 px-3.5 text-xs text-[var(--q-text-muted)]">
            <Lock className="w-3.5 h-3.5" strokeWidth={1.75} />
            Strict access control
          </div>

          {/* Heading */}
          <div className="space-y-3">
            <img src="/quasar_logo.png" alt="" className="mx-auto size-12 rounded-full object-contain" />
            <h1 className="text-4xl md:text-5xl font-medium tracking-tight leading-none text-[var(--q-text)]">
              Quasar
            </h1>
            <p className="text-[15px] md:text-base text-[var(--q-text-muted)] max-w-[560px] mx-auto">
              An AI research assistant for astronomy workflows and multi-archive radio observations.
            </p>
          </div>

          {/* Central Call-to-action */}
          <div className="flex flex-col items-center space-y-3">
            <button onClick={openAuthModal} className="q-pill-ink h-11 px-7 text-[14px] group">
              <LogIn className="w-4 h-4 transition-transform group-hover:translate-x-0.5" strokeWidth={1.75} />
              Sign in to workspace
            </button>
            <p className="text-xs text-[var(--q-text-faint)]">
              Only authorized investigators can access conversation and tool pipelines.
            </p>
          </div>

          {/* Feature Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full pt-8 border-t border-[var(--q-border)]">
            {[
              { icon: Compass, title: "Conductor DAG engine", body: "Decomposes complex research prompts into parallel steps that query the CADC and ALMA science archives automatically." },
              { icon: FileText, title: "FITS processing", body: "Upload images or tables to analyze spectral line coverage and CO isotope maps, and resolve SIMBAD coordinates." },
              { icon: Activity, title: "Observability and privacy", body: "End-to-end tracing with secure encryption. You control your query history and persistence." },
            ].map(({ icon: Icon, title, body }, i) => (
              <div key={title} className="glass-card q-rise flex flex-col items-center md:items-start text-center md:text-left p-5 rounded-2xl" style={{ animationDelay: `${80 + i * 60}ms` }}>
                <div className="mb-4 flex size-9 items-center justify-center rounded-xl bg-[var(--q-canvas)] text-[var(--q-text-muted)]">
                  <Icon className="w-[18px] h-[18px]" strokeWidth={1.75} />
                </div>
                <h3 className="text-[13px] font-medium text-[var(--q-text)] mb-1.5">{title}</h3>
                <p className="text-xs text-[var(--q-text-muted)] leading-relaxed">{body}</p>
              </div>
            ))}
          </div>

          {/* Legal link footer */}
          <div className="text-[11px] text-[var(--q-text-faint)] pt-4">
            By signing in, you agree to our{" "}
            <a href="/terms" className="underline hover:text-[var(--q-text)] transition-colors">Terms &amp; Conditions &amp; Privacy Policy</a>.
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
      {/* ── MOBILE: sidebar as a slide-in drawer ── */}
      {isMobile && sidebarOpen && (
        <div className="fixed inset-0 z-40">
          {/* Scrim — tap to close */}
          <div
            className="drawer-scrim animate-scrim-in absolute inset-0"
            onClick={toggleSidebar}
            aria-hidden="true"
          />
          {/* Drawer panel */}
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Navigation"
            className="animate-drawer-in absolute inset-y-0 left-0 z-50 w-[var(--q-drawer-width)]"
          >
            <Sidebar variant="drawer" onClose={toggleSidebar} />
          </div>
        </div>
      )}

      {/* ── DESKTOP: sidebar as a push panel ── */}
      {!isMobile && (
        // Orbita: the icon rail is always visible; opening adds the list panel beside it.
        <div className={`${sidebarOpen ? "w-[calc(var(--q-sidebar-rail-width)+var(--q-sidebar-width))]" : "w-[var(--q-sidebar-rail-width)]"} transition-all duration-300 shrink-0 overflow-hidden`}>
          <Sidebar collapsed={!sidebarOpen} onToggle={toggleSidebar} />
        </div>
      )}

      <ChatArea />

      {/* Auth Modal Overlay */}
      <AuthModal />

      {/* Onboarding Tutorial — first visit only */}
      {showOnboarding && <OnboardingOverlay onComplete={dismissOnboarding} />}

    </>
  );
}
