import type { Metadata, Viewport } from "next";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { ThemeInitializer } from "@/components/ThemeInitializer";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export const metadata: Metadata = {
  title: "QUASAR — AI Radio Astronomy Research Assistant",
  description: "Professional AI-powered interface for radio astronomy research, data analysis, and literature review.",
  icons: {
    icon: "/favicon.ico?v=2",
    apple: "/favicon.png?v=2",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <head>
        {/* Inline script to set theme BEFORE React hydration to prevent flash */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("quasar_theme");document.documentElement.setAttribute("data-theme",t==="dark"||t==="light"?t:"light")}catch(e){document.documentElement.setAttribute("data-theme","light")}})()`,
          }}
        />
      </head>
      <body className="bg-[var(--q-bg)] font-display text-[var(--q-text)] antialiased transition-colors duration-300" suppressHydrationWarning>
        <ThemeInitializer />
        <ErrorBoundary>
          <div className="fixed inset-0 flex overflow-hidden quasar-backdrop">
            {children}
          </div>
        </ErrorBoundary>
        <SpeedInsights />
      </body>
    </html>
  );
}
