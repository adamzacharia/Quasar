"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { PartnerLogos } from "@/components/PartnerLogos";
import { TERMS_SECTIONS, TERMS_UPDATED, TermsContent } from "@/components/TermsContent";

export default function TermsPage() {
  const scrollToSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="relative flex h-full min-h-screen w-full flex-col overflow-y-auto bg-[var(--q-bg)] text-[var(--q-text-secondary)]">
      {/* Top bar */}
      <header className="sticky top-0 z-30 flex w-full items-center justify-between border-b border-[var(--q-border)] bg-[var(--q-bg)]/85 px-6 py-3 backdrop-blur-md">
        <Link href="/" className="q-pill h-9 px-3.5 text-[13px]">
          <ArrowLeft className="h-4 w-4" strokeWidth={1.75} />
          Back to chat
        </Link>
        <div className="flex items-center gap-2">
          <img src="/quasar_logo.png" alt="" className="size-5 object-contain" />
          <span className="text-[13px] font-medium text-[var(--q-text)]">Quasar</span>
        </div>
      </header>

      <div className="mx-auto flex w-full max-w-[1100px] flex-1 flex-col gap-10 px-6 py-12 md:flex-row">
        {/* Section nav (desktop) */}
        <aside className="hidden h-fit w-60 shrink-0 space-y-1 md:sticky md:top-20 md:block">
          <p className="mb-3 px-3 text-[11px] font-medium uppercase tracking-wider text-[var(--q-text-faint)]">On this page</p>
          {TERMS_SECTIONS.map((sec) => (
            <button
              key={sec.id}
              onClick={() => scrollToSection(sec.id)}
              className="w-full rounded-lg px-3 py-1.5 text-left text-[12.5px] text-[var(--q-text-muted)] transition-colors hover:bg-[var(--q-canvas)] hover:text-[var(--q-text)]"
            >
              {sec.title}
            </button>
          ))}
        </aside>

        <main className="min-w-0 flex-1 space-y-10">
          <div className="space-y-3 border-b border-[var(--q-border)] pb-8">
            <h1 className="text-3xl font-medium tracking-tight text-[var(--q-text)] md:text-4xl">Terms &amp; Privacy</h1>
            <p className="text-[13px] text-[var(--q-text-faint)]">Last updated: {TERMS_UPDATED}</p>
            <p className="max-w-2xl text-[14px] leading-relaxed text-[var(--q-text-muted)]">
              These terms explain how you can use Quasar and what happens to your data when you do. We have kept them
              short and plain. By signing in to Quasar, you agree to them.
            </p>
          </div>

          <TermsContent />
        </main>
      </div>

      <footer className="mt-auto w-full border-t border-[var(--q-border)] px-6 py-10 text-center">
        <p className="mb-6 text-[11px] font-medium uppercase tracking-wider text-[var(--q-text-faint)]">With support from</p>
        <PartnerLogos />
        <p className="mt-8 text-xs text-[var(--q-text-faint)]">&copy; {new Date().getFullYear()} Quasar</p>
      </footer>
    </div>
  );
}
