"use client";

import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { useChatStore } from "@/lib/store";
import { AuthModal } from "@/components/AuthModal";
import { useEffect, useState } from "react";

export default function Home() {
  const sidebarOpen = useChatStore((s) => s.sidebarOpen);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  if (!mounted) return null; // Prevent hydration mismatch Flash

  return (
    <>
      <div className={`${sidebarOpen ? "w-[280px]" : "w-0"} transition-all duration-300 shrink-0 overflow-hidden`}>
        <Sidebar />
      </div>
      <ChatArea />

      {/* Auth Modal Overlay */}
      <AuthModal />

      <div className="fixed top-[-10%] right-[-5%] w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed bottom-[-10%] left-[-5%] w-[600px] h-[600px] bg-accent-purple/5 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed inset-0 noise-overlay opacity-[0.03] pointer-events-none" />
    </>
  );
}
