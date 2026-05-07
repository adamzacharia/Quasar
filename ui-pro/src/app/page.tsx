"use client";

import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { useChatStore } from "../lib/store";
import { AuthModal } from "@/components/AuthModal";
import { OnboardingOverlay, useShowOnboarding } from "@/components/OnboardingOverlay";
import { useEffect, useState } from "react";

export default function Home() {
  const sidebarOpen = useChatStore((s) => s.sidebarOpen);
  const toggleSidebar = useChatStore((s) => s.toggleSidebar);
  const [mounted, setMounted] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [showOnboarding, dismissOnboarding] = useShowOnboarding();

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);

    const mq = window.matchMedia("(max-width: 767px)");
    const handleChange = (e: MediaQueryListEvent | MediaQueryList) => {
      setIsMobile(e.matches);
      // Auto-close sidebar when switching to mobile
      if (e.matches && useChatStore.getState().sidebarOpen) {
        useChatStore.getState().toggleSidebar();
      }
    };
    handleChange(mq); // initial check
    mq.addEventListener("change", handleChange);
    return () => mq.removeEventListener("change", handleChange);
  }, []);

  if (!mounted) return null; // Prevent hydration mismatch Flash

  return (
    <>
      {/* ── MOBILE: sidebar as a full-screen overlay ── */}
      {isMobile && sidebarOpen && (
        <div className="fixed inset-0 z-40 flex">
          {/* Dark backdrop — click to close */}
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
            onClick={toggleSidebar}
          />
          {/* Sidebar panel */}
          <div className="relative z-50 w-[280px] h-full animate-in slide-in-from-left duration-200">
            <Sidebar />
          </div>
        </div>
      )}

      {/* ── DESKTOP: sidebar as a push panel (original behaviour) ── */}
      {!isMobile && (
        <div className={`${sidebarOpen ? "w-[280px]" : "w-0"} transition-all duration-300 shrink-0 overflow-hidden`}>
          <Sidebar />
        </div>
      )}

      <ChatArea />

      {/* Auth Modal Overlay */}
      <AuthModal />

      {/* Onboarding Tutorial — first visit only */}
      {showOnboarding && <OnboardingOverlay onComplete={dismissOnboarding} />}

      {/* Replaced heavy CSS blurs with GPU-safe radial gradients to prevent monitor-handoff crashes */}
      <div 
        className="fixed top-[-10%] right-[-5%] w-[500px] h-[500px] pointer-events-none" 
        style={{ background: 'radial-gradient(circle, rgba(244,113,181,0.05) 0%, rgba(244,113,181,0) 70%)' }} 
      />
      <div 
        className="fixed bottom-[-10%] left-[-5%] w-[600px] h-[600px] pointer-events-none" 
        style={{ background: 'radial-gradient(circle, rgba(168,85,247,0.05) 0%, rgba(168,85,247,0) 70%)' }} 
      />
      
      {/* Temporarily disabled SVG noise overlay as feTurbulence can also crash GPU on 4k resizes */}
      {/* <div className="fixed inset-0 noise-overlay opacity-[0.03] pointer-events-none" /> */}
    </>
  );
}
